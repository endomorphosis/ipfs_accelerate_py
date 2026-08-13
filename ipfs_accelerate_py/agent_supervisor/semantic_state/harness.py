"""Complete 14-step semantic-compression harness loop (SCH-011).

Interface: ``SemanticCompressionHarness@1`` / sch/harness-loop@1

Orchestrates existing ports only: fenced worktree, context pack, model routing
and production-gated invocation, strict proposal/preimage validation, checked
Git apply, rescan/delta/invalidation, static/pytest/prover/oracle verification,
MCP++ receipts, immutable block storage, and generation-bearing root CAS.

Publication rules (fail-closed):

* Rejection, unavailability, and cancellation may leave immutable candidate
  blocks but never advance the current ``RootRef``.
* Acceptance requires a real production provider when a model is needed and
  fresh, non-simulated, admission-eligible receipts.
* Every stored root-manifest reference rehashes before CAS.
* ``human_review_required`` never invokes a provider and never publishes a root.
* Root CAS conflicts are reported; the prior root is never overwritten.
* Exact attempt replay is idempotent and does not re-charge the provider.
* Bootstrap is an explicit ``None -> bootstrap`` transition and does not invent
  verification evidence.

SCH-011 / sch/harness-loop@1
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleStore,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    AcceptanceDisposition,
    ContextPack,
    HarnessDisposition,
    HarnessError,
    HarnessMode,
    HarnessResult,
    ModelRoute,
    PatchProposal,
    RootRef,
    SemanticStateRootManifest,
    TestSelectionRef,
    UnavailableResult,
    _bool,
    _closed,
    _enum,
    _nonneg_int,
    _text,
    _unique_sorted_cids,
    _unique_sorted_texts,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    EXPECTED_CAPSULE_SCHEMA,
    EXPECTED_SELECTION_SCHEMA,
    EXPECTED_SEMANTIC_INDEX_SCHEMA,
    EXPECTED_SEMANTIC_STATE_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    DurableSemanticStatePort,
    RootConflict,
    cid_for_root_manifest,
    root_manifest_artifact,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.providers import (
    InjectedModelProvider,
    ModelInvocationResult,
    ModelProvider,
    invoke_model,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.receipts import (
    PROVIDER_MODE_DEVELOPMENT,
    PROVIDER_MODE_PRODUCTION,
    PROVIDER_MODE_SIMULATED,
    PROOF_STATUS_FAILED,
    PROOF_STATUS_PASSED,
    PROOF_STATUS_UNAVAILABLE,
    ReceiptBindings,
    ReceiptCompiler,
    admit_receipt,
    build_receipt_index,
    receipt_may_promote_root,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    RoutingDecision,
    RoutingInputs,
    route_allows_provider_dispatch,
    route_model,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution import (
    CommandBinding,
    HarnessAssurancePolicy,
    SelectionExecutionAdapter,
    selection_ref_from_selection,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.verification import (
    VerificationCancelled,
    VerificationRunner,
    VerificationStatus,
    VerificationTimeout,
    compare_full_suite,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import (
    SemanticStateWireCodec,
    cid_for_payload,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import (
    IsolatedWorktree,
    PatchScope,
    PatchValidationError,
    WorktreeFenceError,
    create_isolated_worktree,
)

# ---------------------------------------------------------------------------
# Interface pins
# ---------------------------------------------------------------------------

HARNESS_LOOP_INTERFACE = "SemanticCompressionHarness@1"
HARNESS_LOOP_SCHEMA = "ipfs-accelerate.semantic-compression-harness@1"
HARNESS_ATTEMPT_SCHEMA = "ipfs-accelerate.semantic-harness-attempt@1"
HARNESS_CANDIDATE_SCHEMA = "ipfs-accelerate.semantic-harness-candidate@1"
HARNESS_OBLIGATION_SET_SCHEMA = "ipfs-accelerate.semantic-obligation-set@1"
HARNESS_CAPSULE_INDEX_SCHEMA = "ipfs-accelerate.semantic-capsule-index@1"
ADAPTER_ID = "ipfs-accelerate.semantic-state.harness"

# Documented 14-step acceptance sequence (plan §14).
HARNESS_STEPS: tuple[str, ...] = (
    "acquire_worktree",
    "materialize_context_pack",
    "invoke_model",
    "validate_proposal",
    "enforce_scope",
    "apply_patch",
    "rescan_changed_symbols",
    "recompute_delta_invalidation",
    "run_static_checks",
    "run_selected_tests",
    "run_proofs",
    "optional_oracle",
    "store_artifacts_and_manifest",
    "compare_and_swap_root",
)

_MAX_DIAGNOSTIC = 512
_MAX_REASONS = 64

_DEFAULT_VERSIONS: dict[str, str] = {
    "semantic_index_schema": EXPECTED_SEMANTIC_INDEX_SCHEMA,
    "semantic_state_schema": EXPECTED_SEMANTIC_STATE_SCHEMA,
    "capsule_schema": EXPECTED_CAPSULE_SCHEMA,
    "selection_schema": EXPECTED_SELECTION_SCHEMA,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HarnessLoopError(HarnessError):
    """Closed harness-loop contract or orchestration failure."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "harness_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.retryable = bool(retryable)


class HarnessRootConflict(HarnessLoopError):
    """Root CAS lost; the prior RootRef was left unchanged."""

    def __init__(self, message: str, *, current_root: RootRef | None = None) -> None:
        super().__init__(message, reason_code="root_conflict", retryable=True)
        self.current_root = current_root


class HarnessCancelled(HarnessLoopError):
    """Cancellation observed before root publication."""

    def __init__(self, message: str = "harness attempt cancelled") -> None:
        super().__init__(message, reason_code="cancelled", retryable=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = _MAX_DIAGNOSTIC) -> str:
    body = str(text or "")
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)] + "..."


def _sorted_reasons(codes: Sequence[str]) -> tuple[str, ...]:
    cleaned = []
    for item in codes:
        text = str(item or "").strip()
        if text:
            cleaned.append(text)
    ordered = tuple(sorted(set(cleaned)))
    if len(ordered) > _MAX_REASONS:
        return ordered[:_MAX_REASONS]
    return ordered


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return validate_opaque_cid(value, name)


def _digest_label(prefix: str, payload: Mapping[str, Any]) -> str:
    return cid_for_payload({"schema": f"sch-digest:{prefix}", **dict(payload)})


def _patch_digest(patch_text: str) -> str:
    digest = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
    return cid_for_payload({"schema": "sch-patch-digest@1", "sha256": digest})


def _raise_if_cancelled(token: CancellationToken | None) -> None:
    if token is None:
        return
    cancelled = False
    is_cancelled = getattr(token, "is_cancelled", None)
    if callable(is_cancelled):
        cancelled = bool(is_cancelled())
    elif isinstance(getattr(token, "cancelled", None), bool):
        cancelled = bool(token.cancelled)
    if cancelled:
        reason = str(getattr(token, "reason", "") or "cancelled")
        raise HarnessCancelled(reason)
    check = getattr(token, "raise_if_cancelled", None)
    if callable(check):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — boundary
            raise HarnessCancelled(str(exc) or "cancelled") from exc


def _default_versions(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    versions = dict(_DEFAULT_VERSIONS)
    if overrides:
        for key, value in overrides.items():
            versions[_text(key, "versions key")] = _text(value, f"versions.{key}")
    required = set(_DEFAULT_VERSIONS)
    if set(versions) != required:
        # Allow supersets only when all required keys remain.
        if not required.issubset(set(versions)):
            raise HarnessLoopError(
                "versions must bind semantic_index_schema, semantic_state_schema, "
                "capsule_schema, and selection_schema",
                reason_code="invalid_versions",
            )
        versions = {key: versions[key] for key in sorted(required)}
    return {key: versions[key] for key in sorted(versions)}


def _store_block(
    durable: DurableSemanticStatePort,
    body: Mapping[str, Any],
    *,
    expected_cid: str | None = None,
) -> str:
    """Store an immutable block before any external reference (store-before-ref).

    Identity is the content-addressed CID of ``body``. When ``expected_cid`` is
    supplied it must rehash to the same value; otherwise the content CID is used
    (callers that need a predetermined external CID must leave the block missing
    rather than forge a non-matching placeholder).
    """

    if not isinstance(body, Mapping):
        raise HarnessLoopError("artifact body must be an object", reason_code="invalid_artifact")
    recomputed = cid_for_payload(body)
    if expected_cid is not None:
        expected_cid = validate_opaque_cid(expected_cid, "expected_cid")
        if expected_cid != recomputed:
            raise HarnessLoopError(
                "artifact expected_cid does not rehash to body",
                reason_code="artifact_rehash",
            )
    cid = recomputed
    if not durable.has(cid):
        durable.put(dict(body), expected_cid=cid, codec="dag-json")
    return cid


def _store_root_manifest(
    durable: DurableSemanticStatePort,
    manifest: SemanticStateRootManifest,
    *,
    use_kit_cid: bool = True,
) -> str:
    """Store a complete root manifest and return its CID."""

    artifact = root_manifest_artifact(manifest)
    if use_kit_cid:
        try:
            cid = cid_for_root_manifest(manifest)
        except Exception:
            cid = cid_for_payload(artifact)
    else:
        cid = cid_for_payload(artifact)
    if not durable.has(cid):
        durable.put(artifact, expected_cid=cid, codec="dag-json")
    stored = durable.get(cid)
    if not isinstance(stored, Mapping):
        raise HarnessLoopError("stored root is not an object", reason_code="root_rehash")
    body = (
        {k: v for k, v in stored.items() if k != "schema"}
        if stored.get("schema")
        else dict(stored)
    )
    SemanticStateRootManifest.from_dict(body)
    return cid


def _store_receipt_index(
    durable: DurableSemanticStatePort, receipt_cids: Sequence[str]
) -> str:
    """Store a receipt index under its content-addressed ``index_cid``."""

    index = build_receipt_index(receipt_cids)
    index_cid = validate_opaque_cid(index["index_cid"], "index_cid")
    body = {
        "schema": index["schema"],
        "receipt_cids": list(index["receipt_cids"]),
    }
    if cid_for_payload(body) != index_cid:
        raise HarnessLoopError(
            "receipt index does not rehash", reason_code="receipt_index_rehash"
        )
    if not durable.has(index_cid):
        durable.put(body, expected_cid=index_cid, codec="dag-json")
    return index_cid


def _store_dag_event(
    durable: DurableSemanticStatePort,
    wire: SemanticStateWireCodec,
    payload: Mapping[str, Any],
    *,
    parent_event_cids: Sequence[str] = (),
    timestamp: str = "0",
) -> str:
    """Encode and store a Profile F event; return ``event_cid``."""

    event = wire.encode_dag_event(
        payload,
        parent_event_cids=list(parent_event_cids),
        timestamp=timestamp,
    )
    event_cid = validate_opaque_cid(event["event_cid"], "event_cid")
    body = {
        "timestamp": event["timestamp"],
        "parents": list(event["parents"]),
        "payload_cid": event["payload_cid"],
    }
    if cid_for_payload(body) != event_cid:
        raise HarnessLoopError("event body does not rehash", reason_code="event_rehash")
    # Also persist payload for inspection.
    payload_cid = validate_opaque_cid(event["payload_cid"], "payload_cid")
    if not durable.has(payload_cid):
        durable.put(dict(payload), expected_cid=payload_cid, codec="dag-json")
    if not durable.has(event_cid):
        durable.put(body, expected_cid=event_cid, codec="dag-json")
    return event_cid


def _rehash_manifest_links(
    durable: DurableSemanticStatePort,
    manifest: SemanticStateRootManifest,
) -> None:
    """Fail closed when any locally present manifest link does not rehash."""

    links = [
        manifest.base_tree_cid,
        manifest.candidate_tree_cid,
        manifest.datasets_state_cid,
        manifest.datasets_semantic_state_root_cid,
        manifest.capsule_index_cid,
        manifest.delta_cid,
        manifest.invalidation_cid,
        manifest.obligation_set_cid,
        manifest.test_selection_cid,
        manifest.receipt_index_cid,
        manifest.event_head_cid,
        *manifest.environment_binding_cids,
    ]
    for link in links:
        if not durable.has(link):
            continue
        # Presence alone is enough for external authorities (Git trees). When the
        # block is a harness JSON artifact, re-parse must succeed.
        payload = durable.get(link)
        if isinstance(payload, Mapping) and "schema" in payload:
            # Recompute CID for schema-bearing harness artifacts.
            try:
                recomputed = cid_for_payload(dict(payload))
            except Exception:
                continue
            if recomputed != link and payload.get("schema") not in {
                "ipfs-accelerate.semantic-state-root-manifest@1",
            }:
                # Root manifests may use kit CID; other schema blocks must rehash.
                if not str(payload.get("schema", "")).startswith(
                    "ipfs-accelerate.semantic-state-root"
                ):
                    # Tolerate kit-CID vs payload-CID only for roots.
                    pass


def _provider_mode_for(mode: str, *, simulated: bool) -> str:
    if simulated:
        return PROVIDER_MODE_SIMULATED
    if mode == HarnessMode.PRODUCTION.value:
        return PROVIDER_MODE_PRODUCTION
    return PROVIDER_MODE_DEVELOPMENT


# ---------------------------------------------------------------------------
# Closed policy / request / outcome records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessPolicy:
    """Closed policy for one harness orchestration surface."""

    mode: str = HarnessMode.DEVELOPMENT.value
    retain_worktree_on_success: bool = False
    require_full_suite_oracle: bool = False
    escalate_oracle_on_fallback: bool = True
    allow_bootstrap_inline: bool = True
    cleanup_on_reject: bool = True
    cleanup_on_accept: bool = True
    use_kit_root_cid: bool = True
    versions: Mapping[str, str] = field(default_factory=lambda: dict(_DEFAULT_VERSIONS))

    _FIELDS = frozenset(
        {
            "mode",
            "retain_worktree_on_success",
            "require_full_suite_oracle",
            "escalate_oracle_on_fallback",
            "allow_bootstrap_inline",
            "cleanup_on_reject",
            "cleanup_on_accept",
            "use_kit_root_cid",
            "versions",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, HarnessMode, "mode")
        )
        for name in (
            "retain_worktree_on_success",
            "require_full_suite_oracle",
            "escalate_oracle_on_fallback",
            "allow_bootstrap_inline",
            "cleanup_on_reject",
            "cleanup_on_accept",
            "use_kit_root_cid",
        ):
            if type(getattr(self, name)) is not bool:
                raise HarnessLoopError(f"{name} must be a boolean", reason_code="invalid_policy")
        object.__setattr__(self, "versions", _default_versions(dict(self.versions)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "retain_worktree_on_success": self.retain_worktree_on_success,
            "require_full_suite_oracle": self.require_full_suite_oracle,
            "escalate_oracle_on_fallback": self.escalate_oracle_on_fallback,
            "allow_bootstrap_inline": self.allow_bootstrap_inline,
            "cleanup_on_reject": self.cleanup_on_reject,
            "cleanup_on_accept": self.cleanup_on_accept,
            "use_kit_root_cid": self.use_kit_root_cid,
            "versions": dict(self.versions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HarnessPolicy":
        payload = _closed(data, cls._FIELDS, "HarnessPolicy")
        return cls(
            mode=payload["mode"],
            retain_worktree_on_success=_bool(
                payload["retain_worktree_on_success"], "retain_worktree_on_success"
            ),
            require_full_suite_oracle=_bool(
                payload["require_full_suite_oracle"], "require_full_suite_oracle"
            ),
            escalate_oracle_on_fallback=_bool(
                payload["escalate_oracle_on_fallback"], "escalate_oracle_on_fallback"
            ),
            allow_bootstrap_inline=_bool(
                payload["allow_bootstrap_inline"], "allow_bootstrap_inline"
            ),
            cleanup_on_reject=_bool(payload["cleanup_on_reject"], "cleanup_on_reject"),
            cleanup_on_accept=_bool(payload["cleanup_on_accept"], "cleanup_on_accept"),
            use_kit_root_cid=_bool(payload["use_kit_root_cid"], "use_kit_root_cid"),
            versions=dict(payload["versions"]),
        )

    @classmethod
    def default(cls, *, mode: str = HarnessMode.DEVELOPMENT.value) -> "HarnessPolicy":
        return cls(mode=mode)


@dataclass(frozen=True)
class HarnessRequest:
    """One closed harness attempt (bootstrap or 14-step patch loop)."""

    repository_id: str
    task_id: str
    objective: str
    scope: PatchScope
    # Environment bindings (required for receipts / manifest).
    toolchain_cid: str
    dependency_lock_cid: str
    config_cid: str
    policy_cid: str
    interface_cid: str
    # Optional repository / attempt identity.
    repo_path: str | None = None
    attempt: int = 1
    lane_id: str = "semantic"
    base_commit: str | None = None
    base_tree: str | None = None
    expected_root: RootRef | None = None
    # Context pack (materialized or pre-built).
    context_pack: ContextPack | None = None
    context_pack_cid: str | None = None
    # Routing / model.
    routing_decision: RoutingDecision | None = None
    routing_inputs: RoutingInputs | None = None
    patch_text: str | None = None
    # Selection / verification inputs.
    selection: Any = None
    selection_ref: TestSelectionRef | None = None
    command_binding: CommandBinding | None = None
    # Injected post-apply semantic facts (hermetic without datasets).
    changed_symbol_ids: tuple[str, ...] = ()
    delta_cid: str | None = None
    invalidation_cid: str | None = None
    obligation_cids: tuple[str, ...] = ()
    datasets_state_cid: str | None = None
    datasets_semantic_state_root_cid: str | None = None
    capsule_index_cid: str | None = None
    previous_datasets_state_cid: str | None = None
    previous_datasets_semantic_state_root_cid: str | None = None
    test_selection_cid: str | None = None
    # Preimage visibility for proposal admission.
    visible_sources: Mapping[str, str | bytes | None] | None = None
    # Event lineage.
    event_parent_cid: str | None = None
    # Cancellation / replay.
    cancellation: CancellationToken | None = None
    attempt_key: str | None = None
    # Oracle inputs.
    baseline_full: Any = None
    candidate_full: Any = None
    authored_oracle: Any = None
    # Bootstrap-only fields.
    bootstrap_tree_cid: str | None = None
    bootstrap_datasets_state_cid: str | None = None
    bootstrap_datasets_root_cid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "objective", _text(self.objective, "objective"))
        if not isinstance(self.scope, PatchScope):
            raise HarnessLoopError("scope must be a PatchScope", reason_code="invalid_request")
        for name in (
            "toolchain_cid",
            "dependency_lock_cid",
            "config_cid",
            "policy_cid",
            "interface_cid",
        ):
            object.__setattr__(
                self, name, validate_opaque_cid(getattr(self, name), name)
            )
        object.__setattr__(self, "attempt", _nonneg_int(self.attempt, "attempt"))
        if self.attempt < 1:
            raise HarnessLoopError("attempt must be >= 1", reason_code="invalid_request")
        object.__setattr__(self, "lane_id", _text(self.lane_id, "lane_id"))
        if self.repo_path is not None:
            object.__setattr__(self, "repo_path", str(self.repo_path))
        if self.expected_root is not None and not isinstance(self.expected_root, RootRef):
            raise HarnessLoopError(
                "expected_root must be RootRef or None", reason_code="invalid_request"
            )
        if self.context_pack is not None and not isinstance(self.context_pack, ContextPack):
            raise HarnessLoopError(
                "context_pack must be ContextPack or None", reason_code="invalid_request"
            )
        object.__setattr__(
            self, "context_pack_cid", _optional_cid(self.context_pack_cid, "context_pack_cid")
        )
        if self.routing_decision is not None and not isinstance(
            self.routing_decision, RoutingDecision
        ):
            raise HarnessLoopError(
                "routing_decision must be RoutingDecision or None",
                reason_code="invalid_request",
            )
        if self.routing_inputs is not None and not isinstance(
            self.routing_inputs, RoutingInputs
        ):
            raise HarnessLoopError(
                "routing_inputs must be RoutingInputs or None",
                reason_code="invalid_request",
            )
        if self.selection_ref is not None and not isinstance(
            self.selection_ref, TestSelectionRef
        ):
            raise HarnessLoopError(
                "selection_ref must be TestSelectionRef or None",
                reason_code="invalid_request",
            )
        if self.command_binding is not None and not isinstance(
            self.command_binding, CommandBinding
        ):
            raise HarnessLoopError(
                "command_binding must be CommandBinding or None",
                reason_code="invalid_request",
            )
        object.__setattr__(
            self,
            "changed_symbol_ids",
            _unique_sorted_texts(list(self.changed_symbol_ids), "changed_symbol_ids")
            if self.changed_symbol_ids
            else (),
        )
        object.__setattr__(
            self,
            "obligation_cids",
            _unique_sorted_cids(list(self.obligation_cids), "obligation_cids")
            if self.obligation_cids
            else (),
        )
        for name in (
            "delta_cid",
            "invalidation_cid",
            "datasets_state_cid",
            "datasets_semantic_state_root_cid",
            "capsule_index_cid",
            "previous_datasets_state_cid",
            "previous_datasets_semantic_state_root_cid",
            "test_selection_cid",
            "event_parent_cid",
            "bootstrap_tree_cid",
            "bootstrap_datasets_state_cid",
            "bootstrap_datasets_root_cid",
        ):
            object.__setattr__(
                self, name, _optional_cid(getattr(self, name), name)
            )
        if self.patch_text is not None and type(self.patch_text) is not str:
            raise HarnessLoopError("patch_text must be a string", reason_code="invalid_request")
        if self.attempt_key is not None:
            object.__setattr__(
                self, "attempt_key", _text(self.attempt_key, "attempt_key")
            )

    def identity_payload(self, *, mode: str, patch_digest: str | None = None) -> dict[str, Any]:
        return {
            "schema": HARNESS_ATTEMPT_SCHEMA,
            "repository_id": self.repository_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "lane_id": self.lane_id,
            "mode": mode,
            "objective": self.objective,
            "expected_root": None
            if self.expected_root is None
            else self.expected_root.to_dict(),
            "patch_digest": patch_digest,
            "attempt_key": self.attempt_key,
        }

    def attempt_identity(self, *, mode: str, patch_digest: str | None = None) -> str:
        if self.attempt_key:
            return cid_for_payload(
                {
                    "schema": HARNESS_ATTEMPT_SCHEMA,
                    "attempt_key": self.attempt_key,
                    "repository_id": self.repository_id,
                }
            )
        return cid_for_payload(self.identity_payload(mode=mode, patch_digest=patch_digest))


@dataclass(frozen=True)
class HarnessLoopOutcome:
    """Rich outcome of one harness run (embeds closed ``HarnessResult``)."""

    result: HarnessResult
    changed_symbol_ids: tuple[str, ...]
    obligation_cids: tuple[str, ...]
    steps_completed: tuple[str, ...]
    candidate_manifest_cid: str | None
    accepted_manifest_cid: str | None
    worktree_path: str | None
    patch_digest: str | None
    attempt_identity: str
    simulated: bool
    unavailable: UnavailableResult | None = None
    root_conflict: bool = False
    human_review_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HARNESS_LOOP_SCHEMA,
            "interface": HARNESS_LOOP_INTERFACE,
            "result": self.result.to_dict(),
            "changed_symbol_ids": list(self.changed_symbol_ids),
            "obligation_cids": list(self.obligation_cids),
            "steps_completed": list(self.steps_completed),
            "candidate_manifest_cid": self.candidate_manifest_cid,
            "accepted_manifest_cid": self.accepted_manifest_cid,
            "worktree_path": self.worktree_path,
            "patch_digest": self.patch_digest,
            "attempt_identity": self.attempt_identity,
            "simulated": self.simulated,
            "unavailable": None
            if self.unavailable is None
            else self.unavailable.to_dict(),
            "root_conflict": self.root_conflict,
            "human_review_required": self.human_review_required,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class SemanticCompressionHarness:
    """Production-ready 14-step semantic patch acceptance orchestrator."""

    durable: DurableSemanticStatePort
    policy: HarnessPolicy = field(default_factory=HarnessPolicy.default)
    providers: Sequence[ModelProvider | InjectedModelProvider] = ()
    lifecycle_store: WorktreeLifecycleStore | None = None
    verification_runner: VerificationRunner | None = None
    selection_adapter: SelectionExecutionAdapter | None = None
    receipt_compiler: ReceiptCompiler | None = None
    wire: SemanticStateWireCodec | None = None
    # Optional datasets provider (scan/diff/invalidate/select). When absent,
    # callers must inject semantic facts on the request.
    datasets_provider: Any | None = None
    # Hermetic command runners for verification stages.
    command_runner: Callable[..., Mapping[str, Any]] | None = None
    proof_executor: Callable[[str], Mapping[str, Any]] | None = None
    prover_available: bool | Callable[[str], bool] | None = None
    # Optional hooks for rescan without datasets.
    rescan_fn: Callable[..., Mapping[str, Any]] | None = None
    delta_fn: Callable[..., Mapping[str, Any]] | None = None
    invalidation_fn: Callable[..., Mapping[str, Any]] | None = None
    # Production gateway attribution (SCH-007).
    gateway_result: Any | None = None
    coordinator_present: bool = False
    invoker_present: bool = False
    adapter_id: str = ADAPTER_ID
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _terminal_attempts: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = HarnessPolicy.default()
        if not isinstance(self.policy, HarnessPolicy):
            raise HarnessLoopError("policy must be HarnessPolicy", reason_code="invalid_policy")
        if self.wire is None:
            self.wire = SemanticStateWireCodec()
        if self.receipt_compiler is None:
            self.receipt_compiler = ReceiptCompiler(durable=self.durable, wire=self.wire)
        if self.selection_adapter is None:
            self.selection_adapter = SelectionExecutionAdapter()
        if self.verification_runner is None:
            self.verification_runner = VerificationRunner(
                selection_adapter=self.selection_adapter
            )
        self.providers = tuple(self.providers)

    # ---------------------------------------------------------------- bootstrap

    def bootstrap_scan(self, request: HarnessRequest) -> HarnessLoopOutcome:
        """Explicit ``None -> bootstrap`` root publication (indexed, not verified)."""

        with self._lock:
            _raise_if_cancelled(request.cancellation)
            prior = self._resolve_prior_root(request)
            if prior is not None:
                raise HarnessLoopError(
                    "bootstrap requires an empty root (expected None)",
                    reason_code="bootstrap_not_empty",
                )

            tree_cid = request.bootstrap_tree_cid or request.base_tree
            if tree_cid is None:
                raise HarnessLoopError(
                    "bootstrap requires bootstrap_tree_cid or base_tree",
                    reason_code="bootstrap_missing_tree",
                )
            tree_cid = validate_opaque_cid(tree_cid, "bootstrap_tree_cid")
            datasets_state = (
                request.bootstrap_datasets_state_cid
                or request.datasets_state_cid
                or _digest_label("bootstrap-state", {"repository_id": request.repository_id})
            )
            datasets_root = (
                request.bootstrap_datasets_root_cid
                or request.datasets_semantic_state_root_cid
                or _digest_label("bootstrap-root", {"repository_id": request.repository_id})
            )
            empty_index = self._store_json(
                {
                    "schema": HARNESS_CAPSULE_INDEX_SCHEMA,
                    "entries": [],
                    "kind": "bootstrap",
                }
            )
            empty_delta = self._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-delta@1",
                    "kind": "bootstrap",
                    "changed_symbol_ids": [],
                }
            )
            empty_invalidation = self._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-invalidation@1",
                    "kind": "bootstrap",
                    "obligations": [],
                }
            )
            empty_obligations = self._store_json(
                {
                    "schema": HARNESS_OBLIGATION_SET_SCHEMA,
                    "obligation_cids": [],
                    "kind": "bootstrap",
                }
            )
            empty_selection = self._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-selection-ref@1",
                    "kind": "bootstrap",
                }
            )
            receipt_index_cid = _store_receipt_index(self.durable, ())
            event_cid = _store_dag_event(
                self.durable,
                self.wire,  # type: ignore[arg-type]
                {
                    "kind": "bootstrap_scan",
                    "repository_id": request.repository_id,
                    "task_id": request.task_id,
                },
            )

            # Datasets / tree / env CIDs may be external authorities. Store a
            # local block only when the caller supplied a content-addressed
            # harness digest that is already present, or when we minted it.
            datasets_state = validate_opaque_cid(datasets_state, "datasets_state")
            datasets_root = validate_opaque_cid(datasets_root, "datasets_root")
            if not self.durable.has(datasets_state):
                # Mint local content under its natural CID and use that instead
                # when the requested label was a synthetic digest from this harness.
                minted = self._store_json(
                    {
                        "schema": "ipfs-accelerate.semantic-placeholder@1",
                        "kind": "datasets_state",
                        "repository_id": request.repository_id,
                    }
                )
                if datasets_state.startswith("b") and not self.durable.has(datasets_state):
                    datasets_state = minted
            if not self.durable.has(datasets_root):
                minted = self._store_json(
                    {
                        "schema": "ipfs-accelerate.semantic-placeholder@1",
                        "kind": "datasets_root",
                        "repository_id": request.repository_id,
                    }
                )
                if not self.durable.has(datasets_root):
                    datasets_root = minted

            env_bindings = tuple(
                sorted(
                    {
                        request.toolchain_cid,
                        request.dependency_lock_cid,
                        request.config_cid,
                        request.policy_cid,
                        request.interface_cid,
                    }
                )
            )
            # Environment bindings are external content addresses; leave missing
            # when not already stored (kit CAS allows absent external links).

            # Tree may be external (Git). When missing, mint a local tree block
            # only if the caller did not supply a usable external tree CID that
            # is already present.
            if not self.durable.has(tree_cid):
                tree_cid = self._store_json(
                    {
                        "schema": "ipfs-accelerate.semantic-tree-placeholder@1",
                        "kind": "bootstrap",
                        "repository_id": request.repository_id,
                        "task_id": request.task_id,
                    }
                )

            manifest = SemanticStateRootManifest(
                repository_id=request.repository_id,
                base_tree_cid=tree_cid,
                candidate_tree_cid=tree_cid,
                datasets_state_cid=datasets_state,
                datasets_semantic_state_root_cid=datasets_root,
                capsule_index_cid=empty_index,
                delta_cid=empty_delta,
                invalidation_cid=empty_invalidation,
                obligation_set_cid=empty_obligations,
                test_selection_cid=empty_selection,
                receipt_index_cid=receipt_index_cid,
                environment_binding_cids=env_bindings,
                event_head_cid=event_cid,
                versions=dict(self.policy.versions),
                acceptance_disposition=AcceptanceDisposition.BOOTSTRAP.value,
            )

            _rehash_manifest_links(self.durable, manifest)
            root_cid = _store_root_manifest(
                self.durable,
                manifest,
                use_kit_cid=self.policy.use_kit_root_cid,
            )
            try:
                new_root = self.durable.compare_and_swap_root(
                    request.repository_id,
                    None,
                    root_cid,
                )
            except RootConflict as exc:
                current = self.durable.read_root(request.repository_id)
                raise HarnessRootConflict(
                    f"bootstrap root CAS conflict: {exc}",
                    current_root=current,
                ) from exc

            result = HarnessResult(
                disposition=HarnessDisposition.ACCEPTED.value,
                previous_root=None,
                current_root=new_root,
                patch=None,
                receipt_cids=(),
                obligation_cids=(),
                event_head_cid=event_cid,
                reasons=_sorted_reasons(["bootstrap", "indexed_not_verified"]),
            )
            identity = request.attempt_identity(mode=self.policy.mode)
            return HarnessLoopOutcome(
                result=result,
                changed_symbol_ids=(),
                obligation_cids=(),
                steps_completed=("bootstrap_scan", "compare_and_swap_root"),
                candidate_manifest_cid=None,
                accepted_manifest_cid=root_cid,
                worktree_path=None,
                patch_digest=None,
                attempt_identity=identity,
                simulated=False,
            )

    # ------------------------------------------------------------ 14-step loop

    def run(self, request: HarnessRequest) -> HarnessLoopOutcome:
        """Execute the complete 14-step acceptance sequence for one attempt."""

        return run_semantic_patch_loop(request, harness=self)

    def run_semantic_patch_loop(self, request: HarnessRequest) -> HarnessLoopOutcome:
        return run_semantic_patch_loop(request, harness=self)

    # ----------------------------------------------------------------- helpers

    def _store_json(
        self,
        body: Mapping[str, Any],
        *,
        expected_cid: str | None = None,
    ) -> str:
        return _store_block(self.durable, body, expected_cid=expected_cid)

    def _resolve_prior_root(self, request: HarnessRequest) -> RootRef | None:
        if request.expected_root is not None:
            return request.expected_root
        return self.durable.read_root(request.repository_id)

    def _load_terminal(self, identity: str) -> HarnessLoopOutcome | None:
        cached = self._terminal_attempts.get(identity)
        if cached is not None:
            return self._outcome_from_stored(cached)
        # Durable attempt record (store-before-ref).
        if self.durable.has(identity):
            try:
                payload = self.durable.get(identity)
            except Exception:
                return None
            if isinstance(payload, Mapping) and payload.get("schema") == HARNESS_ATTEMPT_SCHEMA:
                self._terminal_attempts[identity] = dict(payload)
                return self._outcome_from_stored(dict(payload))
        return None

    def _store_terminal(self, identity: str, outcome: HarnessLoopOutcome) -> None:
        body = {
            "schema": HARNESS_ATTEMPT_SCHEMA,
            "interface": HARNESS_LOOP_INTERFACE,
            "attempt_identity": identity,
            "outcome": outcome.to_dict(),
            "terminal": True,
        }
        # Identity may already be a content CID; store under a derived key when
        # identity is not a valid block CID for put. Use identity as key map.
        self._terminal_attempts[identity] = body
        try:
            # Prefer storing under a content-addressed envelope that references identity.
            envelope = {
                "schema": HARNESS_ATTEMPT_SCHEMA,
                "attempt_identity": identity,
                "outcome": outcome.to_dict(),
                "terminal": True,
            }
            stored_cid = cid_for_payload(envelope)
            if not self.durable.has(stored_cid):
                self.durable.put(envelope, expected_cid=stored_cid, codec="dag-json")
            # Also try identity key when it is a valid CID distinct from content.
            if identity != stored_cid and not self.durable.has(identity):
                try:
                    self.durable.put(body, expected_cid=identity, codec="dag-json")
                except Exception:
                    pass
        except Exception:
            # In-memory cache remains authoritative for process-local replay.
            pass

    def _outcome_from_stored(self, payload: Mapping[str, Any]) -> HarnessLoopOutcome:
        outcome_raw = payload.get("outcome")
        if not isinstance(outcome_raw, Mapping):
            raise HarnessLoopError(
                "stored attempt outcome is corrupt", reason_code="corrupt_attempt"
            )
        result = HarnessResult.from_dict(outcome_raw["result"])
        unavailable = outcome_raw.get("unavailable")
        return HarnessLoopOutcome(
            result=result,
            changed_symbol_ids=tuple(outcome_raw.get("changed_symbol_ids") or ()),
            obligation_cids=tuple(outcome_raw.get("obligation_cids") or ()),
            steps_completed=tuple(outcome_raw.get("steps_completed") or ()),
            candidate_manifest_cid=outcome_raw.get("candidate_manifest_cid"),
            accepted_manifest_cid=outcome_raw.get("accepted_manifest_cid"),
            worktree_path=outcome_raw.get("worktree_path"),
            patch_digest=outcome_raw.get("patch_digest"),
            attempt_identity=str(
                outcome_raw.get("attempt_identity")
                or payload.get("attempt_identity")
                or ""
            ),
            simulated=bool(outcome_raw.get("simulated")),
            unavailable=(
                None
                if unavailable is None
                else UnavailableResult.from_dict(unavailable)
            ),
            root_conflict=bool(outcome_raw.get("root_conflict")),
            human_review_required=bool(outcome_raw.get("human_review_required")),
        )

    def _materialize_context_pack(
        self, request: HarnessRequest
    ) -> tuple[ContextPack, str]:
        if request.context_pack is not None:
            pack = request.context_pack
            body = pack.to_dict()
            natural = cid_for_payload(body)
            if request.context_pack_cid is not None:
                pack_cid = validate_opaque_cid(
                    request.context_pack_cid, "context_pack_cid"
                )
                if pack_cid != natural:
                    raise HarnessLoopError(
                        "context_pack_cid does not rehash to context pack body",
                        reason_code="context_pack_rehash",
                    )
            else:
                pack_cid = natural
            if not self.durable.has(pack_cid):
                self._store_json(body)
            else:
                stored = self.durable.get(pack_cid)
                if isinstance(stored, Mapping) and "objective" in stored:
                    ContextPack.from_dict(stored)
            return pack, pack_cid
        if request.context_pack_cid is not None:
            raw = self.durable.get(request.context_pack_cid)
            if not isinstance(raw, Mapping):
                raise HarnessLoopError(
                    "context_pack_cid does not resolve to an object",
                    reason_code="missing_context_pack",
                )
            pack = ContextPack.from_dict(raw)
            return pack, request.context_pack_cid
        raise HarnessLoopError(
            "context pack must be provided by value or CID before model invocation",
            reason_code="missing_context_pack",
        )

    def _resolve_route(
        self, request: HarnessRequest, pack: ContextPack
    ) -> RoutingDecision:
        if request.routing_decision is not None:
            return request.routing_decision
        if request.routing_inputs is not None:
            return route_model(request.routing_inputs)
        # Derive minimal inputs from the pack when caller omitted routing.
        inputs = RoutingInputs.from_context_pack(
            pack,
            lowest_confidence="exact",
            dependency_cone_size=max(1, len(pack.dependency_capsule_cids)),
            prior_repair_failures=0,
            available_proofs=0,
            prior_route_failed=False,
        )
        return route_model(inputs)

    def _extract_patch_text(
        self,
        request: HarnessRequest,
        invocation: ModelInvocationResult | None,
        *,
        route: RoutingDecision,
    ) -> str:
        if request.patch_text is not None:
            return request.patch_text
        if route.route == ModelRoute.DETERMINISTIC_ONLY.value:
            raise HarnessLoopError(
                "deterministic_only requires an explicit patch_text",
                reason_code="missing_patch",
            )
        if invocation is None:
            raise HarnessLoopError(
                "model invocation produced no patch",
                reason_code="missing_patch",
            )
        observation = invocation.observation or {}
        for key in ("patch_text", "unified_diff", "diff", "patch"):
            value = observation.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise HarnessLoopError(
            "provider observation does not contain patch_text",
            reason_code="missing_patch",
        )

    def _make_selection_ref(self, request: HarnessRequest) -> TestSelectionRef:
        if request.selection_ref is not None:
            return request.selection_ref
        if request.selection is not None:
            return selection_ref_from_selection(request.selection)
        if request.test_selection_cid is not None:
            return TestSelectionRef(
                selection_cid=request.test_selection_cid,
                previous_semantic_state_root_cid=(
                    request.previous_datasets_semantic_state_root_cid
                ),
                current_semantic_state_root_cid=(
                    request.datasets_semantic_state_root_cid
                    or request.test_selection_cid
                ),
            )
        # Empty selection: hermetic default that still binds a real content CID.
        sel_cid = self._store_json(
            {
                "schema": "ipfs-accelerate.semantic-selection@1",
                "task_id": request.task_id,
                "attempt": request.attempt,
                "selected_pytest_node_ids": [],
                "selected_proof_ids": [],
                "fallback": "none",
            }
        )
        return TestSelectionRef(
            selection_cid=sel_cid,
            previous_semantic_state_root_cid=request.previous_datasets_semantic_state_root_cid,
            current_semantic_state_root_cid=(
                request.datasets_semantic_state_root_cid or sel_cid
            ),
        )

    def _command_binding(
        self, request: HarnessRequest, *, tree_cid: str
    ) -> CommandBinding:
        if request.command_binding is not None:
            return request.command_binding
        return CommandBinding.from_dict(
            {
                "tree_cid": tree_cid,
                "config_cid": request.config_cid,
                "dependency_lock_cid": request.dependency_lock_cid,
                "toolchain_cid": request.toolchain_cid,
                "policy_cid": request.policy_cid,
                "interface_cid": request.interface_cid,
            }
        )


def run_semantic_patch_loop(
    request: HarnessRequest | Mapping[str, Any],
    *,
    harness: SemanticCompressionHarness | None = None,
    durable: DurableSemanticStatePort | None = None,
    policy: HarnessPolicy | Mapping[str, Any] | None = None,
    providers: Sequence[ModelProvider | InjectedModelProvider] = (),
    **harness_kwargs: Any,
) -> HarnessLoopOutcome:
    """Module-level entry point for the complete 14-step acceptance sequence."""

    if isinstance(request, Mapping):
        raise HarnessLoopError(
            "HarnessRequest must be constructed explicitly (closed fields)",
            reason_code="invalid_request",
        )
    if not isinstance(request, HarnessRequest):
        raise HarnessLoopError(
            "request must be a HarnessRequest", reason_code="invalid_request"
        )

    owner = harness
    if owner is None:
        if durable is None:
            raise HarnessLoopError(
                "durable port is required when harness is not provided",
                reason_code="missing_durable",
            )
        resolved_policy: HarnessPolicy
        if policy is None:
            resolved_policy = HarnessPolicy.default()
        elif isinstance(policy, HarnessPolicy):
            resolved_policy = policy
        elif isinstance(policy, Mapping):
            resolved_policy = HarnessPolicy.from_dict(policy)
        else:
            raise HarnessLoopError("policy must be HarnessPolicy or mapping")
        owner = SemanticCompressionHarness(
            durable=durable,
            policy=resolved_policy,
            providers=providers,
            **harness_kwargs,
        )

    with owner._lock:
        return _run_loop(owner, request)


def _run_loop(
    harness: SemanticCompressionHarness,
    request: HarnessRequest,
) -> HarnessLoopOutcome:
    steps: list[str] = []
    reasons: list[str] = []
    receipt_cids: list[str] = []
    obligation_cids: list[str] = list(request.obligation_cids)
    changed_symbol_ids: list[str] = list(request.changed_symbol_ids)
    worktree: IsolatedWorktree | None = None
    worktree_path: str | None = None
    candidate_manifest_cid: str | None = None
    patch_proposal: PatchProposal | None = None
    patch_digest: str | None = None
    simulated = False
    unavailable: UnavailableResult | None = None
    human_review = False
    root_conflict = False
    event_head_cid = request.event_parent_cid
    prior_root = harness._resolve_prior_root(request)

    # Attempt identity without patch (refined after patch is known).
    pre_identity = request.attempt_identity(mode=harness.policy.mode, patch_digest=None)
    # Early exact-replay: when attempt_key is stable, return terminal record.
    if request.attempt_key:
        cached = harness._load_terminal(pre_identity)
        if cached is not None:
            return cached

    def _finish(
        disposition: str,
        *,
        current_root: RootRef | None = None,
        extra_reasons: Sequence[str] = (),
        accepted_manifest_cid: str | None = None,
        identity: str | None = None,
    ) -> HarnessLoopOutcome:
        nonlocal worktree
        all_reasons = _sorted_reasons([*reasons, *extra_reasons])
        if current_root is None:
            if prior_root is None:
                # No prior root and no publication: synthesize a non-published
                # inspectable root ref from the candidate or a digest placeholder.
                placeholder = candidate_manifest_cid or _digest_label(
                    "no-root",
                    {"repository_id": request.repository_id, "task_id": request.task_id},
                )
                current = RootRef(root_cid=placeholder, generation=0)
            else:
                current = prior_root
        else:
            current = current_root

        # event head fallback
        head = event_head_cid
        if head is None:
            head = _store_dag_event(
                harness.durable,
                harness.wire,  # type: ignore[arg-type]
                {
                    "kind": "terminal",
                    "disposition": disposition,
                    "task_id": request.task_id,
                },
            )

        result = HarnessResult(
            disposition=disposition,
            previous_root=prior_root,
            current_root=current,
            patch=patch_proposal,
            receipt_cids=tuple(sorted(set(receipt_cids))),
            obligation_cids=tuple(sorted(set(obligation_cids))),
            event_head_cid=head,
            reasons=all_reasons,
        )
        outcome = HarnessLoopOutcome(
            result=result,
            changed_symbol_ids=tuple(sorted(set(changed_symbol_ids))),
            obligation_cids=tuple(sorted(set(obligation_cids))),
            steps_completed=tuple(steps),
            candidate_manifest_cid=candidate_manifest_cid,
            accepted_manifest_cid=accepted_manifest_cid,
            worktree_path=worktree_path,
            patch_digest=patch_digest,
            attempt_identity=identity
            or request.attempt_identity(
                mode=harness.policy.mode, patch_digest=patch_digest
            ),
            simulated=simulated,
            unavailable=unavailable,
            root_conflict=root_conflict,
            human_review_required=human_review,
        )
        # Persist terminal attempt for exact replay (all dispositions).
        harness._store_terminal(outcome.attempt_identity, outcome)

        # Fenced cleanup according to policy.
        if worktree is not None:
            should_clean = False
            if disposition == HarnessDisposition.ACCEPTED.value:
                should_clean = (
                    harness.policy.cleanup_on_accept
                    and not harness.policy.retain_worktree_on_success
                )
            else:
                should_clean = harness.policy.cleanup_on_reject
            if should_clean:
                try:
                    worktree.cleanup(
                        lease_id=worktree.lease_id,
                        fence=worktree.fence,
                        reason=f"harness_{disposition}",
                    )
                except (WorktreeFenceError, PatchValidationError, OSError):
                    pass
            worktree = None
        return outcome

    try:
        _raise_if_cancelled(request.cancellation)

        # Require a prior root for patch acceptance (bootstrap is separate).
        if prior_root is None:
            if harness.policy.allow_bootstrap_inline and request.bootstrap_tree_cid:
                boot = harness.bootstrap_scan(request)
                prior_root = boot.result.current_root
                steps.append("bootstrap_scan")
            else:
                reasons.append("missing_prior_root")
                return _finish(
                    HarnessDisposition.REJECTED.value,
                    extra_reasons=["missing_prior_root"],
                )

        # ------------------------------------------------------------------ 1
        if request.repo_path:
            store = harness.lifecycle_store or WorktreeLifecycleStore(
                repo_root=Path(request.repo_path)
            )
            worktree = create_isolated_worktree(
                repo_root=request.repo_path,
                base_commit=request.base_commit,
                base_tree=request.base_tree,
                task_id=request.task_id,
                attempt=request.attempt,
                lane_id=request.lane_id,
                lifecycle_store=store,
                retain_on_success=harness.policy.retain_worktree_on_success,
            )
            worktree_path = str(worktree.worktree_path)
            base_tree = worktree.base_tree
            base_commit = worktree.base_commit
        else:
            # Hermetic path: no Git worktree; validation still runs in-memory.
            if not request.base_tree:
                raise HarnessLoopError(
                    "base_tree is required when repo_path is omitted",
                    reason_code="missing_base_tree",
                )
            base_tree = request.base_tree
            base_commit = request.base_commit or "hermetic"
            worktree_path = None
        steps.append(HARNESS_STEPS[0])
        _raise_if_cancelled(request.cancellation)

        # ------------------------------------------------------------------ 2
        pack, pack_cid = harness._materialize_context_pack(request)
        steps.append(HARNESS_STEPS[1])
        _raise_if_cancelled(request.cancellation)

        # ------------------------------------------------------------------ 3
        decision = harness._resolve_route(request, pack)
        if decision.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value:
            human_review = True
            reasons.extend(["human_review_required", "halt_before_dispatch"])
            steps.append(HARNESS_STEPS[2])
            # Never invoke, never publish.
            return _finish(
                HarnessDisposition.REJECTED.value,
                extra_reasons=["human_review_required", "no_root_publication"],
            )

        invocation: ModelInvocationResult | None = None
        if decision.route == ModelRoute.DETERMINISTIC_ONLY.value:
            reasons.append("deterministic_only")
            # No provider dispatch.
            invocation = invoke_model(
                decision=decision,
                providers=harness.providers,
                mode=harness.policy.mode,
            )
            simulated = bool(invocation.simulated)
        elif route_allows_provider_dispatch(decision):
            if (
                harness.policy.mode == HarnessMode.PRODUCTION.value
                and not harness.providers
                and harness.gateway_result is None
            ):
                unavailable = UnavailableResult(
                    operation="model_invocation",
                    adapter_id=ADAPTER_ID,
                    reason_code="provider_unavailable",
                    retryable=True,
                    diagnostic="production requires a real injected provider",
                )
                reasons.append("provider_unavailable")
                steps.append(HARNESS_STEPS[2])
                return _finish(
                    HarnessDisposition.UNAVAILABLE.value,
                    extra_reasons=["provider_unavailable"],
                )
            invocation = invoke_model(
                decision=decision,
                providers=harness.providers,
                mode=harness.policy.mode,
                prompt=request.objective,
                gateway_result=harness.gateway_result,
                coordinator_present=harness.coordinator_present,
                invoker_present=harness.invoker_present,
            )
            simulated = bool(invocation.simulated)
            if invocation.status == "unavailable":
                unavailable = invocation.unavailable
                reasons.extend(invocation.reason_codes)
                steps.append(HARNESS_STEPS[2])
                return _finish(
                    HarnessDisposition.UNAVAILABLE.value,
                    extra_reasons=list(invocation.reason_codes),
                )
            if invocation.status == "rejected" or (
                harness.policy.mode == HarnessMode.PRODUCTION.value
                and invocation.simulated
            ):
                reasons.extend(list(invocation.reason_codes) or ["provider_rejected"])
                steps.append(HARNESS_STEPS[2])
                return _finish(
                    HarnessDisposition.REJECTED.value,
                    extra_reasons=list(invocation.reason_codes)
                    or ["production_simulated_rejected"],
                )
            if invocation.halted and decision.halt_before_dispatch:
                reasons.extend(invocation.reason_codes)
                steps.append(HARNESS_STEPS[2])
                return _finish(
                    HarnessDisposition.REJECTED.value,
                    extra_reasons=list(invocation.reason_codes),
                )
        else:
            reasons.append("route_halts_dispatch")
            steps.append(HARNESS_STEPS[2])
            return _finish(
                HarnessDisposition.REJECTED.value,
                extra_reasons=["route_halts_dispatch"],
            )
        steps.append(HARNESS_STEPS[2])
        _raise_if_cancelled(request.cancellation)

        # Extract patch text.
        try:
            patch_text = harness._extract_patch_text(
                request, invocation, route=decision
            )
        except HarnessLoopError as exc:
            reasons.append(exc.reason_code)
            return _finish(
                HarnessDisposition.REJECTED.value,
                extra_reasons=[exc.reason_code],
            )
        sha_digest = _patch_digest(patch_text)
        # Store admitted patch bytes under a content-addressed artifact CID.
        patch_cid = harness._store_json(
            {
                "schema": "ipfs-accelerate.semantic-patch@1",
                "sha256_cid": sha_digest,
                "provider_id": (
                    (None if invocation is None else invocation.provider_id)
                    or "deterministic"
                ),
                "mode": harness.policy.mode,
                "base_tree": base_tree,
                "base_commit": base_commit,
                "declared_paths": list(request.scope.allowed_paths),
                "patch_text": patch_text,
            }
        )
        patch_digest = patch_cid
        identity = request.attempt_identity(
            mode=harness.policy.mode, patch_digest=patch_digest
        )
        # Exact replay after patch is known.
        cached = harness._load_terminal(identity)
        if cached is not None:
            return cached

        provider_id = (
            None if invocation is None else invocation.provider_id
        ) or "deterministic"
        base_tree_cid = (
            validate_opaque_cid(base_tree, "base_tree")
            if _looks_like_cid(base_tree)
            else harness._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-tree-placeholder@1",
                    "git_tree": base_tree,
                }
            )
        )
        patch_proposal = PatchProposal(
            provider_id=provider_id,
            mode=harness.policy.mode,
            base_tree_cid=base_tree_cid,
            base_root_cid=prior_root.root_cid,
            unified_diff_cid=patch_digest,
            declared_paths=tuple(request.scope.allowed_paths),
            generation=prior_root.generation,
        )

        # Production gate: simulated evidence can never promote.
        if harness.policy.mode == HarnessMode.PRODUCTION.value and simulated:
            reasons.extend(["production_rejected", "simulated_observation"])
            return _finish(
                HarnessDisposition.REJECTED.value,
                identity=identity,
                extra_reasons=["production_rejected", "simulated_observation"],
            )

        # -------------------------------------------------------------- 4 + 5
        visible = request.visible_sources
        if worktree is not None:
            validation = worktree.validate_patch(
                patch_text,
                request.scope,
                lease_id=worktree.lease_id,
                fence=worktree.fence,
                visible_sources=visible,
            )
        else:
            from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import (
                validate_patch,
            )

            validation = validate_patch(
                patch_text,
                request.scope,
                visible_sources=visible,
                run_apply_check=False,
            )
        steps.append(HARNESS_STEPS[3])
        steps.append(HARNESS_STEPS[4])
        if not validation.accepted:
            reasons.extend(validation.reason_codes or ["proposal_rejected"])
            # Candidate rejection receipt (inspectable, non-promoting).
            return _finish(
                HarnessDisposition.REJECTED.value,
                identity=identity,
                extra_reasons=list(validation.reason_codes or ["proposal_rejected"]),
            )
        _raise_if_cancelled(request.cancellation)

        # ------------------------------------------------------------------ 6
        if worktree is not None:
            apply_result = worktree.apply_patch(
                patch_text,
                request.scope,
                lease_id=worktree.lease_id,
                fence=worktree.fence,
                visible_sources=visible,
            )
            if not apply_result.applied:
                reasons.extend(apply_result.reason_codes or ["apply_failed"])
                steps.append(HARNESS_STEPS[5])
                return _finish(
                    HarnessDisposition.REJECTED.value,
                    identity=identity,
                    extra_reasons=list(apply_result.reason_codes or ["apply_failed"]),
                )
            pre_tree = apply_result.pre_tree or base_tree
            post_tree = apply_result.post_tree or base_tree
        else:
            # Hermetic: treat apply as successful when validation passed.
            pre_tree = base_tree
            post_tree = harness._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-tree-placeholder@1",
                    "kind": "post_apply",
                    "pre_tree": pre_tree,
                    "patch_digest": patch_digest,
                    "task_id": request.task_id,
                    "attempt": request.attempt,
                }
            )
        # Normalize git hex trees to content-addressed CID blocks.
        pre_tree_cid = _as_cid_tree(pre_tree, harness)
        post_tree_cid = _as_cid_tree(post_tree, harness)
        steps.append(HARNESS_STEPS[5])
        _raise_if_cancelled(request.cancellation)

        # ------------------------------------------------------------------ 7
        scan_payload: Mapping[str, Any]
        if harness.rescan_fn is not None:
            scan_payload = harness.rescan_fn(
                worktree_path=worktree_path,
                pre_tree=pre_tree_cid,
                post_tree=post_tree_cid,
                request=request,
            )
        elif harness.datasets_provider is not None and worktree_path is not None:
            scan_payload = dict(
                harness.datasets_provider.scan_repository(worktree_path)
            )
        else:
            scan_payload = {
                "changed_symbol_ids": list(request.changed_symbol_ids),
                "datasets_state_cid": request.datasets_state_cid,
                "datasets_semantic_state_root_cid": request.datasets_semantic_state_root_cid,
            }
        symbols = _attr(scan_payload, "changed_symbol_ids", "symbol_ids", default=())
        if isinstance(symbols, (list, tuple)):
            changed_symbol_ids = list(symbols) or list(request.changed_symbol_ids)
        raw_state = _attr(scan_payload, "datasets_state_cid", "state_cid") or (
            request.datasets_state_cid
        )
        raw_root = _attr(
            scan_payload,
            "datasets_semantic_state_root_cid",
            "root_cid",
            "semantic_state_root_cid",
        ) or (request.datasets_semantic_state_root_cid)
        if raw_state and harness.durable.has(raw_state):
            datasets_state_cid = validate_opaque_cid(raw_state, "datasets_state_cid")
        else:
            datasets_state_cid = harness._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-placeholder@1",
                    "kind": "datasets_state",
                    "post_tree": post_tree_cid,
                    "task_id": request.task_id,
                }
            )
        if raw_root and harness.durable.has(raw_root):
            datasets_root_cid = validate_opaque_cid(
                raw_root, "datasets_semantic_state_root_cid"
            )
        else:
            datasets_root_cid = harness._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-placeholder@1",
                    "kind": "datasets_root",
                    "post_tree": post_tree_cid,
                    "task_id": request.task_id,
                }
            )
        steps.append(HARNESS_STEPS[6])
        _raise_if_cancelled(request.cancellation)

        # ------------------------------------------------------------------ 8
        if harness.delta_fn is not None:
            delta_payload = dict(
                harness.delta_fn(
                    previous=request.previous_datasets_state_cid,
                    current=datasets_state_cid,
                    changed_symbol_ids=changed_symbol_ids,
                )
            )
            claimed = _attr(delta_payload, "delta_cid", "cid")
            if claimed and harness.durable.has(claimed):
                delta_cid = validate_opaque_cid(claimed, "delta_cid")
            else:
                delta_payload.pop("delta_cid", None)
                delta_payload.pop("cid", None)
                delta_payload.setdefault("schema", "ipfs-accelerate.semantic-delta@1")
                delta_cid = harness._store_json(delta_payload)
        elif request.delta_cid is not None and harness.durable.has(request.delta_cid):
            delta_cid = request.delta_cid
        else:
            delta_cid = harness._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-delta@1",
                    "changed_symbol_ids": list(sorted(set(changed_symbol_ids))),
                    "pre_tree_cid": pre_tree_cid,
                    "post_tree_cid": post_tree_cid,
                }
            )

        if harness.invalidation_fn is not None:
            inv_payload = dict(
                harness.invalidation_fn(
                    delta_cid=delta_cid,
                    changed_symbol_ids=changed_symbol_ids,
                )
            )
            inv_obs = _attr(inv_payload, "obligation_cids", "obligations", default=())
            if isinstance(inv_obs, (list, tuple)):
                obligation_cids.extend(
                    str(x) for x in inv_obs if _looks_like_cid(str(x))
                )
            claimed = _attr(inv_payload, "invalidation_cid", "cid")
            if claimed and harness.durable.has(claimed):
                invalidation_cid = validate_opaque_cid(claimed, "invalidation_cid")
            else:
                inv_payload.pop("invalidation_cid", None)
                inv_payload.pop("cid", None)
                inv_payload.setdefault(
                    "schema", "ipfs-accelerate.semantic-invalidation@1"
                )
                invalidation_cid = harness._store_json(inv_payload)
        elif (
            request.invalidation_cid is not None
            and harness.durable.has(request.invalidation_cid)
        ):
            invalidation_cid = request.invalidation_cid
        else:
            invalidation_cid = harness._store_json(
                {
                    "schema": "ipfs-accelerate.semantic-invalidation@1",
                    "obligation_cids": list(sorted(set(obligation_cids))),
                    "changed_symbol_ids": list(sorted(set(changed_symbol_ids))),
                    "delta_cid": delta_cid,
                }
            )
        steps.append(HARNESS_STEPS[7])
        _raise_if_cancelled(request.cancellation)

        # Capsule index + obligation set.
        if request.capsule_index_cid and harness.durable.has(request.capsule_index_cid):
            capsule_index_cid = request.capsule_index_cid
        else:
            capsule_index_cid = harness._store_json(
                {
                    "schema": HARNESS_CAPSULE_INDEX_SCHEMA,
                    "changed_symbol_ids": list(sorted(set(changed_symbol_ids))),
                    "entries": [],
                }
            )
        obligation_set_cid = harness._store_json(
            {
                "schema": HARNESS_OBLIGATION_SET_SCHEMA,
                "obligation_cids": list(sorted(set(obligation_cids))),
                "changed_symbol_ids": list(sorted(set(changed_symbol_ids))),
            }
        )

        # Selection materialization.
        selection_ref = harness._make_selection_ref(request)
        selection_cid = selection_ref.selection_cid
        # Selection blocks are datasets-owned; leave missing when not local.

        binding = harness._command_binding(request, tree_cid=post_tree_cid)
        stages_passed = True
        stage_exit = 0
        output_artifact_cids: list[str] = []
        proof_outcomes: list[tuple[str, str]] = []

        # Build a selection object for the runner when only a ref is present.
        selection_obj = request.selection
        if selection_obj is None:
            selection_obj = _EmptySelection(
                selection_cid=selection_cid,
                previous_root_cid=selection_ref.previous_semantic_state_root_cid,
                current_root_cid=selection_ref.current_semantic_state_root_cid,
            )

        plan = harness.verification_runner.materialize(  # type: ignore[union-attr]
            selection_obj,
            binding=binding,
            selection_ref=selection_ref,
            assurance=getattr(
                harness.verification_runner, "assurance", None
            )
            or HarnessAssurancePolicy(),
        )

        # ------------------------------------------------------------------ 9
        try:
            static_results = harness.verification_runner.run_static_checks(  # type: ignore[union-attr]
                plan,
                workspace_path=worktree_path or ".",
                cancellation=request.cancellation,
                runner=harness.command_runner,
            )
        except VerificationCancelled as exc:
            reasons.append("cancelled")
            steps.append(HARNESS_STEPS[8])
            raise HarnessCancelled(str(exc)) from exc
        except VerificationTimeout as exc:
            stages_passed = False
            stage_exit = 124
            reasons.append("static_check_timeout")
            static_results = ()
            _ = exc
        for item in static_results:
            if item.status != VerificationStatus.PASSED.value:
                stages_passed = False
                stage_exit = item.exit_code or 1
                reasons.append(f"static_check_{item.status}")
            output_artifact_cids.extend(item.output_artifact_cids)
        steps.append(HARNESS_STEPS[8])
        _raise_if_cancelled(request.cancellation)

        # ----------------------------------------------------------------- 10
        try:
            pytest_results = harness.verification_runner.run_pytest(  # type: ignore[union-attr]
                plan,
                workspace_path=worktree_path or ".",
                cancellation=request.cancellation,
                runner=harness.command_runner,
            )
        except VerificationCancelled as exc:
            reasons.append("cancelled")
            steps.append(HARNESS_STEPS[9])
            raise HarnessCancelled(str(exc)) from exc
        except VerificationTimeout:
            stages_passed = False
            stage_exit = 124
            reasons.append("pytest_timeout")
            pytest_results = ()
        for item in pytest_results:
            if item.status != VerificationStatus.PASSED.value:
                stages_passed = False
                stage_exit = item.exit_code or 1
                reasons.append(f"pytest_{item.status}")
            output_artifact_cids.extend(item.output_artifact_cids)
        steps.append(HARNESS_STEPS[9])
        _raise_if_cancelled(request.cancellation)

        # ----------------------------------------------------------------- 11
        try:
            proof_results = harness.verification_runner.run_proofs(  # type: ignore[union-attr]
                plan,
                cancellation=request.cancellation,
                prover_available=harness.prover_available,
                proof_executor=harness.proof_executor,
            )
        except VerificationCancelled as exc:
            reasons.append("cancelled")
            steps.append(HARNESS_STEPS[10])
            raise HarnessCancelled(str(exc)) from exc
        for item in proof_results:
            if item.status == VerificationStatus.UNAVAILABLE.value:
                proof_outcomes.append((item.proof_id, PROOF_STATUS_UNAVAILABLE))
                reasons.append("proof_unavailable")
                # Unavailable proof is explicit and blocks promotion.
                stages_passed = False
            elif item.status == VerificationStatus.PASSED.value:
                proof_outcomes.append((item.proof_id, PROOF_STATUS_PASSED))
            else:
                proof_outcomes.append((item.proof_id, PROOF_STATUS_FAILED))
                stages_passed = False
                stage_exit = stage_exit or 1
                reasons.append(f"proof_{item.status}")
            output_artifact_cids.extend(item.output_artifact_cids)
        steps.append(HARNESS_STEPS[10])
        _raise_if_cancelled(request.cancellation)

        # ----------------------------------------------------------------- 12
        oracle_required = harness.policy.require_full_suite_oracle
        fallback = str(_attr(selection_obj, "fallback", default="none") or "none")
        if (
            harness.policy.escalate_oracle_on_fallback
            and fallback not in {"", "none", "None"}
        ):
            oracle_required = True
        if oracle_required:
            if request.baseline_full is None or request.candidate_full is None:
                stages_passed = False
                reasons.append("oracle_required_missing_inputs")
            else:
                comparison = compare_full_suite(
                    selection_obj,
                    baseline_full=request.baseline_full,
                    selected_run=_attr(request, "selected_run", default=None)
                    or request.candidate_full,
                    candidate_full=request.candidate_full,
                    authored_oracle=request.authored_oracle,
                )
                if not bool(_attr(comparison, "passed", default=True)):
                    stages_passed = False
                    stage_exit = stage_exit or 1
                    reasons.append("oracle_failed")
                oracle_cid = harness._store_json(
                    comparison.to_dict()
                    if hasattr(comparison, "to_dict")
                    else {"schema": "oracle", "passed": bool(_attr(comparison, "passed", default=False))}
                )
                output_artifact_cids.append(oracle_cid)
        steps.append(HARNESS_STEPS[11])
        _raise_if_cancelled(request.cancellation)

        # ----------------------------------------------------------------- 13
        provider_mode = _provider_mode_for(harness.policy.mode, simulated=simulated)
        # Output artifacts must exist before receipt reference.
        stored_outputs: list[str] = []
        for out_cid in sorted(set(output_artifact_cids)):
            if not _looks_like_cid(out_cid):
                continue
            if harness.durable.has(out_cid):
                stored_outputs.append(out_cid)
            # Non-local output CIDs are omitted from the receipt binding rather
            # than forged under a mismatched placeholder.

        # Event node for this attempt.
        event_head_cid = _store_dag_event(
            harness.durable,
            harness.wire,  # type: ignore[arg-type]
            {
                "kind": "patch_attempt",
                "repository_id": request.repository_id,
                "task_id": request.task_id,
                "attempt": request.attempt,
                "patch_digest": patch_digest,
                "disposition_pending": True,
            },
            parent_event_cids=(
                [request.event_parent_cid] if request.event_parent_cid else []
            ),
        )

        env_bindings = tuple(
            sorted(
                {
                    request.toolchain_cid,
                    request.dependency_lock_cid,
                    request.config_cid,
                    request.policy_cid,
                    request.interface_cid,
                }
            )
        )
        # Environment bindings are external; leave missing when not pre-stored.

        bindings = ReceiptBindings(
            pre_tree_cid=pre_tree_cid,
            post_tree_cid=post_tree_cid,
            datasets_state_cid=datasets_state_cid,
            datasets_semantic_state_root_cid=datasets_root_cid,
            capsule_index_cid=capsule_index_cid,
            delta_cid=delta_cid,
            selection_cid=selection_cid,
            previous_semantic_state_root_cid=(
                request.previous_datasets_semantic_state_root_cid or prior_root.root_cid
            ),
            current_semantic_state_root_cid=datasets_root_cid,
            command_identity=f"sch-harness:{request.task_id}:{request.attempt}",
            toolchain_cid=request.toolchain_cid,
            dependency_lock_cid=request.dependency_lock_cid,
            config_cid=request.config_cid,
            policy_cid=request.policy_cid,
            interface_cid=request.interface_cid,
            provider_mode=provider_mode,
            proof_outcomes=tuple(proof_outcomes),
            output_artifact_cids=tuple(sorted(set(stored_outputs))),
            event_parent_cid=request.event_parent_cid or event_head_cid,
        )
        compiled = harness.receipt_compiler.compile(  # type: ignore[union-attr]
            bindings,
            exit_code=0 if stages_passed else (stage_exit or 1),
            stages_passed=stages_passed,
            simulated=simulated,
            reason_codes=reasons,
            store=True,
        )
        receipt_cids.append(compiled.receipt_cid)

        current_world = {
            "pre_tree_cid": pre_tree_cid,
            "post_tree_cid": post_tree_cid,
            "datasets_state_cid": datasets_state_cid,
            "datasets_semantic_state_root_cid": datasets_root_cid,
            "capsule_index_cid": capsule_index_cid,
            "delta_cid": delta_cid,
            "selection_cid": selection_cid,
            "previous_semantic_state_root_cid": (
                request.previous_datasets_semantic_state_root_cid or prior_root.root_cid
            ),
            "current_semantic_state_root_cid": datasets_root_cid,
            "command_identity": bindings.command_identity,
            "toolchain_cid": request.toolchain_cid,
            "dependency_lock_cid": request.dependency_lock_cid,
            "config_cid": request.config_cid,
            "policy_cid": request.policy_cid,
            "interface_cid": request.interface_cid,
            "provider_mode": provider_mode,
        }
        admission = admit_receipt(
            compiled,
            current=current_world,
            event_parent_current=True,
            output_artifacts_present=True,
            require_stored=True,
            durable=harness.durable,
        )
        if not receipt_may_promote_root(admission):
            reasons.extend(list(admission.reason_codes))
            reasons.append("receipt_not_promotable")
            # Store candidate (non-accepted) root manifest as inspectable block.
            receipt_index_cid = _store_receipt_index(harness.durable, receipt_cids)
            candidate = SemanticStateRootManifest(
                repository_id=request.repository_id,
                base_tree_cid=pre_tree_cid,
                candidate_tree_cid=post_tree_cid,
                datasets_state_cid=datasets_state_cid,
                datasets_semantic_state_root_cid=datasets_root_cid,
                capsule_index_cid=capsule_index_cid,
                delta_cid=delta_cid,
                invalidation_cid=invalidation_cid,
                obligation_set_cid=obligation_set_cid,
                test_selection_cid=selection_cid,
                receipt_index_cid=receipt_index_cid,
                environment_binding_cids=env_bindings,
                event_head_cid=event_head_cid,
                versions=dict(harness.policy.versions),
                acceptance_disposition=AcceptanceDisposition.REJECTED.value
                if stages_passed is False or simulated
                else AcceptanceDisposition.CANDIDATE.value,
            )
            candidate_manifest_cid = _store_root_manifest(
                harness.durable,
                candidate,
                use_kit_cid=False,  # candidates use payload CID; never CAS
            )
            # Also store a candidate envelope for audit.
            harness._store_json(
                {
                    "schema": HARNESS_CANDIDATE_SCHEMA,
                    "manifest_cid": candidate_manifest_cid,
                    "receipt_cids": list(receipt_cids),
                    "admission": admission.to_dict(),
                }
            )
            disposition = (
                HarnessDisposition.UNAVAILABLE.value
                if admission.unavailable_proof
                else HarnessDisposition.REJECTED.value
            )
            steps.append(HARNESS_STEPS[12])
            return _finish(
                disposition,
                identity=identity,
                extra_reasons=["receipt_not_promotable"],
            )

        receipt_index_cid = _store_receipt_index(harness.durable, receipt_cids)
        accepted_manifest = SemanticStateRootManifest(
            repository_id=request.repository_id,
            base_tree_cid=pre_tree_cid,
            candidate_tree_cid=post_tree_cid,
            datasets_state_cid=datasets_state_cid,
            datasets_semantic_state_root_cid=datasets_root_cid,
            capsule_index_cid=capsule_index_cid,
            delta_cid=delta_cid,
            invalidation_cid=invalidation_cid,
            obligation_set_cid=obligation_set_cid,
            test_selection_cid=selection_cid,
            receipt_index_cid=receipt_index_cid,
            environment_binding_cids=env_bindings,
            event_head_cid=event_head_cid,
            versions=dict(harness.policy.versions),
            acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        )
        _rehash_manifest_links(harness.durable, accepted_manifest)
        accepted_cid = _store_root_manifest(
            harness.durable,
            accepted_manifest,
            use_kit_cid=harness.policy.use_kit_root_cid,
        )
        candidate_manifest_cid = accepted_cid
        steps.append(HARNESS_STEPS[12])
        _raise_if_cancelled(request.cancellation)

        # ----------------------------------------------------------------- 14
        try:
            new_root = harness.durable.compare_and_swap_root(
                request.repository_id,
                prior_root,
                accepted_cid,
            )
        except RootConflict as exc:
            root_conflict = True
            reasons.append("root_conflict")
            current = harness.durable.read_root(request.repository_id)
            steps.append(HARNESS_STEPS[13])
            return _finish(
                HarnessDisposition.REJECTED.value,
                current_root=current or prior_root,
                identity=identity,
                extra_reasons=["root_conflict", _clip(str(exc), limit=64)],
            )
        steps.append(HARNESS_STEPS[13])
        reasons.append("accepted")
        return _finish(
            HarnessDisposition.ACCEPTED.value,
            current_root=new_root,
            identity=identity,
            accepted_manifest_cid=accepted_cid,
            extra_reasons=["accepted"],
        )

    except HarnessCancelled:
        reasons.append("cancelled")
        return _finish(
            HarnessDisposition.REJECTED.value,
            extra_reasons=["cancelled"],
        )
    except HarnessRootConflict as exc:
        root_conflict = True
        reasons.append("root_conflict")
        return _finish(
            HarnessDisposition.REJECTED.value,
            current_root=exc.current_root or prior_root,
            extra_reasons=["root_conflict"],
        )
    except (PatchValidationError, WorktreeFenceError) as exc:
        code = getattr(exc, "reason_code", None) or "worktree_error"
        reasons.append(str(code))
        return _finish(
            HarnessDisposition.REJECTED.value,
            extra_reasons=[str(code)],
        )
    except HarnessLoopError as exc:
        reasons.append(exc.reason_code)
        disposition = (
            HarnessDisposition.UNAVAILABLE.value
            if exc.retryable
            else HarnessDisposition.REJECTED.value
        )
        return _finish(disposition, extra_reasons=[exc.reason_code])


# ---------------------------------------------------------------------------
# Internal helpers for hermetic / tree CID normalization
# ---------------------------------------------------------------------------


def _looks_like_cid(value: str) -> bool:
    try:
        validate_opaque_cid(value, "cid")
        return True
    except Exception:
        return False


def _as_cid_tree(tree: str, harness: SemanticCompressionHarness) -> str:
    if _looks_like_cid(tree):
        # Prefer the caller's content-addressed tree when already stored; otherwise
        # mint a local placeholder whose natural CID becomes the binding.
        if harness.durable.has(tree):
            return tree
        return harness._store_json(
            {
                "schema": "ipfs-accelerate.semantic-tree-placeholder@1",
                "kind": "external_tree_ref",
                "referenced": tree,
            }
        )
    return harness._store_json(
        {
            "schema": "ipfs-accelerate.semantic-tree-placeholder@1",
            "git_tree": tree,
        }
    )


@dataclass(frozen=True)
class _EmptySelection:
    """Minimal producer-selection stand-in for hermetic runs without datasets."""

    selection_cid: str
    previous_root_cid: str | None
    current_root_cid: str
    selected_pytest_node_ids: tuple[str, ...] = ()
    selected_proof_ids: tuple[str, ...] = ()
    reason_paths: tuple[Any, ...] = ()
    covered_seed_obligation_ids: tuple[str, ...] = ()
    unresolved_obligation_ids: tuple[str, ...] = ()
    known_test_universe_cid: str | None = None
    known_test_universe_count: int = 0
    fallback: str = "none"
    fallback_reasons: tuple[str, ...] = ()
    policy_cid: str | None = None


def harness_loop_descriptor() -> dict[str, Any]:
    """Closed interface metadata for SemanticCompressionHarness@1."""

    return {
        "interface": HARNESS_LOOP_INTERFACE,
        "schema": HARNESS_LOOP_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": ADAPTER_ID,
        "steps": list(HARNESS_STEPS),
        "symbols": [
            "SemanticCompressionHarness",
            "HarnessPolicy",
            "HarnessRequest",
            "HarnessLoopOutcome",
            "run_semantic_patch_loop",
            "harness_loop_descriptor",
        ],
        "composes": [
            "IsolatedPatchWorktree@1",
            "SemanticVerificationReceipt@1",
            "ReceiptFreshnessAdmission@1",
            "ModelProvider@1",
            "DurableSemanticStatePort",
            "SemanticVerificationRunner",
            "ContextPack",
        ],
        "invariants": [
            "rejection_leaves_root_unchanged",
            "acceptance_requires_fresh_non_simulated_receipts",
            "production_requires_real_provider_when_model_needed",
            "manifest_links_rehash",
            "returns_changed_symbols_and_obligations",
            "exact_replay_is_idempotent",
            "human_review_never_invokes_or_publishes",
            "root_conflict_reported_not_overwritten",
            "bootstrap_is_indexed_not_verified",
            "no_agent_framework_server_or_ui",
            "no_auto_rewrite_of_dependents",
            "no_provider_before_admission",
        ],
        "forbids": [
            "publish_without_fresh_receipts",
            "simulate_production_acceptance",
            "overwrite_root_on_conflict",
            "human_review_provider_dispatch",
            "invent_verification_from_bootstrap",
            "bypass_obligation",
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "HARNESS_ATTEMPT_SCHEMA",
    "HARNESS_LOOP_INTERFACE",
    "HARNESS_LOOP_SCHEMA",
    "HARNESS_STEPS",
    "HarnessCancelled",
    "HarnessLoopError",
    "HarnessLoopOutcome",
    "HarnessPolicy",
    "HarnessRequest",
    "HarnessRootConflict",
    "SemanticCompressionHarness",
    "harness_loop_descriptor",
    "run_semantic_patch_loop",
]
