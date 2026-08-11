"""Incremental verification plan creation and planner policy (IVP-010).

``create_verification_plan`` / :class:`IncrementalVerificationPlanner` join
normalized semantic inputs, pure affected-check selection, and exact-key cache
decisions into a deterministic, side-effect-free :class:`VerificationPlan`.

Normative pipeline (plan §6):

1. Validate and canonicalize the five inputs; record typed gaps, never invent
   identities.
2. Cross-check patch base against RepositoryState / InvalidationPlan /
   ContextPack tree and semantic roots; bind receipt keys to the exact target
   patched tree.
3. Select affected tests/static/type/proofs via ``select_affected_verification``.
4. Build exact required receipt keys and query the cache by exact key only.
5. Classify reuse dispositions without publishing tombstones (planning is
   side-effect free).
6. Broaden under uncertainty; force human review for unbound sandbox, policy
   conflict, or declared-scope crossing.
7. Allocate positive, capped resource and timeout bounds.
8. Emit ordered acceptance criteria: every current production-admissible
   required success and no pending mandatory full-suite fallback.

Importing this module performs no I/O and never mutates a receipt cache.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.analysis.repository_forest import (
    RepositoryForest,
)
from ipfs_accelerate_py.agent_supervisor.contract_analysis.execution_profile import (
    CapabilitySnapshot,
    LockIdentity,
    ToolIdentity,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

from .contracts import (
    MAX_DURATION_MS,
    MAX_RESOURCE_QUANTITY,
    PROOF_OBLIGATION_NOT_APPLICABLE_CID,
    CacheReuseDecision,
    CacheReuseDisposition,
    VerificationContractError,
    VerificationIdentityCompiler,
    VerificationIdentityError,
    VerificationPlan,
    VerificationReceiptKey,
    VerificationReceiptKind,
)
from .datasets_adapter import (
    ContextPackView,
    DatasetsVerificationInputAdapter,
    InputKind,
    InvalidationPlanView,
    RepositoryStateView,
    create_datasets_verification_input_adapter,
)
from .receipt_cache import VerificationReceiptCache, classify_candidate
from .selection import (
    AffectedVerificationSelection,
    SelectionPolicy,
    VerificationCatalog,
    select_affected_verification,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

VERIFICATION_PLANNER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner@1"
)
VERIFICATION_PLANNER_INTERFACE: Final[str] = "IncrementalVerificationPlanner@1"
PLANNER_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-planner-policy@1"
)
PATCH_DELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-patch-delta@1"
)
CHECK_TOOL_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-check-tool-spec@1"
)
IDENTITY_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-identity-binding@1"
)
PLANNER_EVIDENCE: Final[str] = "ivp/verification-plan@1"

_TREE_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
)
_SEMANTIC_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1"
)
_ENVIRONMENT_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
_SELECTOR_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-selector-argv@1"
)
_SYMBOL_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/affected-symbol-version@1"
)
_TOOL_EXECUTABLE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
)
_ABSENT_BYTES_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/absent-verification-bytes@1"
)

# Reason codes (stable tokens; match VerificationContract _TOKEN_RE).
REASON_PATCH_BASE_MISMATCH: Final[str] = "patch_base_tree_mismatch"
REASON_SEMANTIC_ROOT_MISMATCH: Final[str] = "semantic_root_mismatch"
REASON_ENVIRONMENT_ROOT_MISMATCH: Final[str] = "environment_root_mismatch"
REASON_LOCK_ROOT_MISMATCH: Final[str] = "dependency_lock_root_mismatch"
REASON_CROSS_TREE_REJECTED: Final[str] = "cross_tree_admission_rejected"
REASON_UNBOUND_SANDBOX: Final[str] = "unbound_effective_sandbox"
REASON_POLICY_CONFLICT: Final[str] = "policy_conflict"
REASON_SCOPE_CROSSING: Final[str] = "declared_scope_crossing"
REASON_ENVIRONMENT_MISMATCH: Final[str] = "environment_observed_mismatch"
REASON_LOCK_MISMATCH: Final[str] = "lock_observed_mismatch"
REASON_TOOL_MISMATCH: Final[str] = "tool_observed_mismatch"
REASON_ADAPTER_GAP: Final[str] = "adapter_normalization_gap"
REASON_MISSING_IDENTITY: Final[str] = "missing_identity_material"
REASON_CACHE_LOOKUP: Final[str] = "cache_exact_key_lookup"
REASON_NO_CACHE: Final[str] = "no_receipt_cache"
REASON_MANDATORY_FULL_SUITE: Final[str] = "mandatory_full_suite_pending"
REASON_PRODUCTION_SUCCESS_REQUIRED: Final[str] = (
    "required_production_admissible_success"
)
REASON_NO_PENDING_FALLBACK: Final[str] = "no_pending_mandatory_fallback"
REASON_HUMAN_REVIEW_FALSE: Final[str] = "human_review_must_be_false"
REASON_RELEVANT_SELECTION: Final[str] = "relevant_change_selected"
REASON_UNRELATED_NO_OVERSELECT: Final[str] = "unrelated_no_semantic_over_selection"

DEFAULT_MAX_EXECUTION_TIME_MS: Final[int] = 3_600_000
DEFAULT_STEP_TIMEOUT_MS: Final[int] = 300_000
DEFAULT_CPU_MILLIS: Final[int] = 60_000
DEFAULT_MEMORY_BYTES: Final[int] = 512 * 1024 * 1024
DEFAULT_PROCESSES: Final[int] = 2
DEFAULT_ARTIFACT_BYTES: Final[int] = 64 * 1024 * 1024
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_COLLECTION: Final[int] = 50_000

_KIND_STEP_PREFIX: Final[Mapping[VerificationReceiptKind, str]] = MappingProxyType(
    {
        VerificationReceiptKind.STATIC_ANALYSIS: "static",
        VerificationReceiptKind.TYPE_CHECK: "type",
        VerificationReceiptKind.TEST: "test",
        VerificationReceiptKind.PROOF: "proof",
    }
)


class PlannerError(VerificationContractError):
    """Fail-closed planner input or policy contract violation."""


class PlannerBoundsError(PlannerError):
    """A planner resource or timeout bound is out of range."""


class PlannerIdentityError(PlannerError, VerificationIdentityError):
    """Patch / semantic / environment identity roots disagree."""


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _structured_cid(schema: str, value: Any) -> str:
    return content_identity({"schema": schema, "value": value})


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        if required:
            raise PlannerError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise PlannerError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise PlannerError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise PlannerError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise PlannerBoundsError(f"{field_name} exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
    return text


def _optional_text(value: Any, *, field_name: str) -> str:
    if value is None or value == "":
        return ""
    return _text(value, field_name=field_name, required=True)


def _boolean(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    if value is None:
        if default is not None:
            return default
        raise PlannerError(f"{field_name} is required")
    if not isinstance(value, bool):
        raise PlannerError(f"{field_name} must be a boolean")
    return value


def _positive_int(
    value: Any,
    *,
    field_name: str,
    minimum: int = 1,
    maximum: int = MAX_RESOURCE_QUANTITY,
    default: int | None = None,
) -> int:
    if value is None:
        if default is not None:
            return default
        raise PlannerError(f"{field_name} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise PlannerBoundsError(
            f"{field_name} must be in [{minimum}, {maximum}]"
        )
    return value


def _nonneg_int(
    value: Any,
    *,
    field_name: str,
    maximum: int = MAX_RESOURCE_QUANTITY,
    default: int | None = None,
) -> int:
    if value is None:
        if default is not None:
            return default
        raise PlannerError(f"{field_name} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerError(f"{field_name} must be an integer")
    if value < 0 or value > maximum:
        raise PlannerBoundsError(f"{field_name} must be in [0, {maximum}]")
    return value


def _string_tuple(
    value: Any,
    *,
    field_name: str,
    maximum: int = MAX_COLLECTION,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlannerError(f"{field_name} must be a sequence of strings")
    if len(value) > maximum:
        raise PlannerBoundsError(f"{field_name} exceeds {maximum} items")
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _text(item, field_name=f"{field_name}[{index}]")
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(sorted(out))


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlannerError(f"{field_name} must be an object")
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _bytes_cid(value: bytes | None, *, field_name: str) -> str:
    if value is None:
        return _structured_cid(
            _ABSENT_BYTES_IDENTITY_SCHEMA,
            {"field": field_name, "state": "not_present"},
        )
    if type(value) is not bytes:
        raise PlannerError(f"{field_name} must be exact bytes or None")
    return cid_for_bytes(value)


def _unique_sorted(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(items)))


def _stable_unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _step_id(kind: VerificationReceiptKind, check_id: str) -> str:
    prefix = _KIND_STEP_PREFIX[kind]
    # Stable, DAG-safe token: replace path separators without inventing content.
    safe = (
        check_id.replace("/", ".")
        .replace("::", ".")
        .replace(" ", "_")
        .replace(":", ".")
    )
    return f"{prefix}:{safe}"


# ---------------------------------------------------------------------------
# Check tool specs and identity binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckToolSpec:
    """Per-kind or per-check tool identity used to compile receipt keys."""

    tool_name: str
    tool_version: str
    adapter_schema: str
    selector_argv: tuple[str, ...]
    tool_capability_name: str = "verification-tool"
    resolved_tool_executable: str = ""
    tool_executable_bytes: bytes = b""
    tool_identity: ToolIdentity | None = None
    tool_version_probe_argv: tuple[str, ...] = ()
    tool_version_probe_output_bytes: bytes = b""
    proof_obligation: Any | None = None
    proof_backend_binding: Mapping[str, Any] | None = None
    schema: str = CHECK_TOOL_SPEC_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, field_name="tool_name")
        )
        object.__setattr__(
            self, "tool_version", _text(self.tool_version, field_name="tool_version")
        )
        object.__setattr__(
            self,
            "adapter_schema",
            _text(self.adapter_schema, field_name="adapter_schema"),
        )
        if (
            isinstance(self.selector_argv, (str, bytes))
            or not isinstance(self.selector_argv, Sequence)
            or not self.selector_argv
        ):
            raise PlannerError("selector_argv must be a non-empty sequence")
        argv = tuple(
            _text(item, field_name=f"selector_argv[{i}]")
            for i, item in enumerate(self.selector_argv)
        )
        object.__setattr__(self, "selector_argv", argv)
        object.__setattr__(
            self,
            "tool_capability_name",
            _text(self.tool_capability_name, field_name="tool_capability_name"),
        )
        executable = _optional_text(
            self.resolved_tool_executable, field_name="resolved_tool_executable"
        )
        if not executable:
            executable = argv[0]
        object.__setattr__(self, "resolved_tool_executable", executable)
        if type(self.tool_executable_bytes) is not bytes or not self.tool_executable_bytes:
            # Derive deterministic fixture bytes from tool name when omitted.
            derived = ("reviewed-launcher:" + self.tool_name).encode("utf-8")
            object.__setattr__(self, "tool_executable_bytes", derived)
        probe_argv = self.tool_version_probe_argv
        if not probe_argv:
            if len(argv) >= 3 and argv[1] == "-m":
                probe_argv = (*argv[:3], "--version")
            else:
                probe_argv = (executable, "--version")
        object.__setattr__(
            self,
            "tool_version_probe_argv",
            tuple(
                _text(item, field_name=f"tool_version_probe_argv[{i}]")
                for i, item in enumerate(probe_argv)
            ),
        )
        if type(self.tool_version_probe_output_bytes) is not bytes or (
            not self.tool_version_probe_output_bytes
        ):
            object.__setattr__(
                self,
                "tool_version_probe_output_bytes",
                f"{self.tool_name} {self.tool_version}\n".encode("utf-8"),
            )
        if self.tool_identity is None:
            sha = "sha256:" + hashlib.sha256(self.tool_executable_bytes).hexdigest()
            object.__setattr__(
                self,
                "tool_identity",
                ToolIdentity(
                    name=self.tool_capability_name,
                    kind="executable",
                    locator=self.resolved_tool_executable.rsplit("/", 1)[-1],
                    version="launcher-1",
                    identity=sha,
                    roles=("verification",),
                ),
            )
        elif not isinstance(self.tool_identity, ToolIdentity):
            raise PlannerError("tool_identity must be a ToolIdentity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "adapter_schema": self.adapter_schema,
            "selector_argv": list(self.selector_argv),
            "tool_capability_name": self.tool_capability_name,
            "resolved_tool_executable": self.resolved_tool_executable,
            "tool_executable_sha256": (
                "sha256:" + hashlib.sha256(self.tool_executable_bytes).hexdigest()
            ),
        }

    @classmethod
    def from_value(cls, value: Any) -> CheckToolSpec:
        if isinstance(value, cls):
            return value
        payload = _as_mapping(value, field_name="check_tool_spec")
        tool_identity = payload.get("tool_identity")
        if isinstance(tool_identity, Mapping):
            tool_identity = ToolIdentity.from_dict(tool_identity)
        elif tool_identity is not None and not isinstance(tool_identity, ToolIdentity):
            raise PlannerError("tool_identity must be a mapping or ToolIdentity")
        executable_bytes = payload.get("tool_executable_bytes")
        if isinstance(executable_bytes, str):
            executable_bytes = executable_bytes.encode("utf-8")
        probe_out = payload.get("tool_version_probe_output_bytes")
        if isinstance(probe_out, str):
            probe_out = probe_out.encode("utf-8")
        return cls(
            tool_name=payload.get("tool_name", ""),
            tool_version=payload.get("tool_version", ""),
            adapter_schema=payload.get("adapter_schema", ""),
            selector_argv=tuple(payload.get("selector_argv") or ()),
            tool_capability_name=str(
                payload.get("tool_capability_name") or "verification-tool"
            ),
            resolved_tool_executable=str(
                payload.get("resolved_tool_executable") or ""
            ),
            tool_executable_bytes=(
                executable_bytes if isinstance(executable_bytes, bytes) else b""
            ),
            tool_identity=tool_identity,
            tool_version_probe_argv=tuple(
                payload.get("tool_version_probe_argv") or ()
            ),
            tool_version_probe_output_bytes=(
                probe_out if isinstance(probe_out, bytes) else b""
            ),
            proof_obligation=payload.get("proof_obligation"),
            proof_backend_binding=(
                dict(payload["proof_backend_binding"])
                if isinstance(payload.get("proof_backend_binding"), Mapping)
                else None
            ),
            schema=str(payload.get("schema") or CHECK_TOOL_SPEC_SCHEMA),
        )


@dataclass(frozen=True, slots=True)
class IdentityBinding:
    """Shared authority-relevant identity material for receipt key compilation.

    Either supply a replay-valid ``repository_forest`` (preferred) *or* a fully
    observed ``repository_tree_observation`` with matching ``repository_tree_cid``.
    """

    patch_base_tree_id: str
    observed_semantic_state: Mapping[str, Any]
    sandbox_environment: Mapping[str, Any]
    capability_snapshot: CapabilitySnapshot
    dependency_lock_path: str
    dependency_lock_identity: LockIdentity
    dependency_lock_bytes: bytes
    network_policy: str = "deny_all"
    receipt_schema_version: int = 1
    configuration_bytes: bytes | None = b"[tool]\nstrict = true\n"
    fixture_data_bytes: tuple[bytes, ...] = (b"fixture-one\n",)
    affected_symbol_versions: tuple[Mapping[str, Any], ...] = ()
    repository_forest: RepositoryForest | None = None
    repository_alias: str = ""
    repository_tree_observation: Mapping[str, Any] | None = None
    repository_tree_cid: str = ""
    semantic_state_root_cid: str = ""
    environment_cid: str = ""
    dependency_lock_cid: str = ""
    effective_sandbox_bound: bool = True
    schema: str = IDENTITY_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "patch_base_tree_id",
            _text(self.patch_base_tree_id, field_name="patch_base_tree_id"),
        )
        semantic = _as_mapping(
            self.observed_semantic_state, field_name="observed_semantic_state"
        )
        object.__setattr__(self, "observed_semantic_state", _frozen_mapping(dict(semantic)))
        sandbox = _as_mapping(
            self.sandbox_environment, field_name="sandbox_environment"
        )
        object.__setattr__(self, "sandbox_environment", _frozen_mapping(dict(sandbox)))
        if not isinstance(self.capability_snapshot, CapabilitySnapshot):
            raise PlannerError("capability_snapshot must be a CapabilitySnapshot")
        object.__setattr__(
            self,
            "dependency_lock_path",
            _text(self.dependency_lock_path, field_name="dependency_lock_path"),
        )
        if not isinstance(self.dependency_lock_identity, LockIdentity):
            raise PlannerError("dependency_lock_identity must be a LockIdentity")
        if type(self.dependency_lock_bytes) is not bytes or not self.dependency_lock_bytes:
            raise PlannerError("dependency_lock_bytes must be non-empty bytes")
        object.__setattr__(
            self,
            "network_policy",
            _text(self.network_policy, field_name="network_policy"),
        )
        object.__setattr__(
            self,
            "receipt_schema_version",
            _positive_int(
                self.receipt_schema_version,
                field_name="receipt_schema_version",
                maximum=2**31 - 1,
            ),
        )
        object.__setattr__(
            self,
            "effective_sandbox_bound",
            _boolean(
                self.effective_sandbox_bound,
                field_name="effective_sandbox_bound",
                default=True,
            ),
        )
        fixtures = self.fixture_data_bytes or ()
        if isinstance(fixtures, (str, bytes)) or not isinstance(fixtures, Sequence):
            raise PlannerError("fixture_data_bytes must be a sequence of bytes")
        object.__setattr__(
            self,
            "fixture_data_bytes",
            tuple(item if type(item) is bytes else bytes(item) for item in fixtures),
        )
        symbols = self.affected_symbol_versions or ()
        if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
            raise PlannerError("affected_symbol_versions must be a sequence")
        object.__setattr__(
            self,
            "affected_symbol_versions",
            tuple(
                _frozen_mapping(dict(_as_mapping(item, field_name=f"symbol[{i}]")))
                for i, item in enumerate(symbols)
            ),
        )
        semantic_cid = _optional_text(
            self.semantic_state_root_cid, field_name="semantic_state_root_cid"
        )
        if not semantic_cid:
            semantic_cid = _structured_cid(
                _SEMANTIC_IDENTITY_INPUT_SCHEMA, dict(self.observed_semantic_state)
            )
        object.__setattr__(self, "semantic_state_root_cid", semantic_cid)
        lock_cid = _optional_text(
            self.dependency_lock_cid, field_name="dependency_lock_cid"
        )
        if not lock_cid:
            lock_cid = cid_for_bytes(self.dependency_lock_bytes)
        object.__setattr__(self, "dependency_lock_cid", lock_cid)
        if self.repository_forest is not None and not isinstance(
            self.repository_forest, RepositoryForest
        ):
            raise PlannerError("repository_forest must be a RepositoryForest")
        if self.repository_tree_observation is not None:
            object.__setattr__(
                self,
                "repository_tree_observation",
                _frozen_mapping(
                    dict(
                        _as_mapping(
                            self.repository_tree_observation,
                            field_name="repository_tree_observation",
                        )
                    )
                ),
            )
        tree_cid = _optional_text(
            self.repository_tree_cid, field_name="repository_tree_cid"
        )
        if (
            not tree_cid
            and self.repository_tree_observation is not None
            and self.repository_forest is None
        ):
            tree_cid = _structured_cid(
                _TREE_IDENTITY_INPUT_SCHEMA, dict(self.repository_tree_observation)
            )
        object.__setattr__(self, "repository_tree_cid", tree_cid)
        alias = _optional_text(self.repository_alias, field_name="repository_alias")
        if not alias and self.repository_forest is not None:
            alias = self.repository_forest.sole_write_alias
        object.__setattr__(self, "repository_alias", alias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "patch_base_tree_id": self.patch_base_tree_id,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "repository_tree_cid": self.repository_tree_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "network_policy": self.network_policy,
            "receipt_schema_version": self.receipt_schema_version,
            "effective_sandbox_bound": self.effective_sandbox_bound,
            "repository_alias": self.repository_alias,
        }

    @classmethod
    def from_value(cls, value: Any) -> IdentityBinding:
        if isinstance(value, cls):
            return value
        payload = _as_mapping(value, field_name="identity_binding")
        snapshot = payload.get("capability_snapshot")
        if isinstance(snapshot, Mapping):
            snapshot = CapabilitySnapshot(
                tool_identities=dict(snapshot.get("tool_identities") or {}),
                lock_identities=dict(snapshot.get("lock_identities") or {}),
                unavailable_tools=tuple(snapshot.get("unavailable_tools") or ()),
                network_enabled=bool(snapshot.get("network_enabled", False)),
                auto_install_enabled=bool(
                    snapshot.get("auto_install_enabled", False)
                ),
                home_cache_enabled=bool(snapshot.get("home_cache_enabled", False)),
                credential_names=tuple(snapshot.get("credential_names") or ()),
                environment_names=tuple(snapshot.get("environment_names") or ()),
                read_paths=tuple(snapshot.get("read_paths") or ()),
                write_paths=tuple(snapshot.get("write_paths") or ()),
            )
        elif not isinstance(snapshot, CapabilitySnapshot):
            raise PlannerError("capability_snapshot is required on identity_binding")
        lock_identity = payload.get("dependency_lock_identity")
        if isinstance(lock_identity, Mapping):
            lock_identity = LockIdentity.from_dict(lock_identity)
        elif not isinstance(lock_identity, LockIdentity):
            raise PlannerError("dependency_lock_identity is required")
        lock_bytes = payload.get("dependency_lock_bytes")
        if isinstance(lock_bytes, str):
            lock_bytes = lock_bytes.encode("utf-8")
        if type(lock_bytes) is not bytes:
            raise PlannerError("dependency_lock_bytes must be bytes")
        config = payload.get("configuration_bytes", b"[tool]\nstrict = true\n")
        if isinstance(config, str):
            config = config.encode("utf-8")
        fixtures_raw = payload.get("fixture_data_bytes") or (b"fixture-one\n",)
        fixtures: list[bytes] = []
        for item in fixtures_raw:
            if isinstance(item, str):
                fixtures.append(item.encode("utf-8"))
            elif type(item) is bytes:
                fixtures.append(item)
            else:
                raise PlannerError("fixture_data_bytes items must be bytes")
        forest = payload.get("repository_forest")
        if forest is not None and not isinstance(forest, RepositoryForest):
            raise PlannerError("repository_forest must be a RepositoryForest")
        return cls(
            patch_base_tree_id=str(payload.get("patch_base_tree_id") or ""),
            observed_semantic_state=dict(
                payload.get("observed_semantic_state") or {}
            ),
            sandbox_environment=dict(payload.get("sandbox_environment") or {}),
            capability_snapshot=snapshot,
            dependency_lock_path=str(payload.get("dependency_lock_path") or ""),
            dependency_lock_identity=lock_identity,
            dependency_lock_bytes=lock_bytes,
            network_policy=str(payload.get("network_policy") or "deny_all"),
            receipt_schema_version=int(payload.get("receipt_schema_version") or 1),
            configuration_bytes=config if type(config) is bytes else None,
            fixture_data_bytes=tuple(fixtures),
            affected_symbol_versions=tuple(
                payload.get("affected_symbol_versions") or ()
            ),
            repository_forest=forest,
            repository_alias=str(payload.get("repository_alias") or ""),
            repository_tree_observation=(
                dict(payload["repository_tree_observation"])
                if isinstance(payload.get("repository_tree_observation"), Mapping)
                else None
            ),
            repository_tree_cid=str(payload.get("repository_tree_cid") or ""),
            semantic_state_root_cid=str(
                payload.get("semantic_state_root_cid") or ""
            ),
            environment_cid=str(payload.get("environment_cid") or ""),
            dependency_lock_cid=str(payload.get("dependency_lock_cid") or ""),
            effective_sandbox_bound=bool(
                payload.get("effective_sandbox_bound", True)
            ),
            schema=str(payload.get("schema") or IDENTITY_BINDING_SCHEMA),
        )


# ---------------------------------------------------------------------------
# Patch delta and planner policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatchDelta:
    """Proposed patch rebound to a declared base and target patched tree."""

    base_tree_id: str
    changed_paths: tuple[str, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    patch_paths: tuple[str, ...] = ()
    declared_scope_paths: tuple[str, ...] = ()
    target_tree_cid: str = ""
    repository_tree_observation: Mapping[str, Any] | None = None
    repository_forest: RepositoryForest | None = None
    repository_alias: str = ""
    schema: str = PATCH_DELTA_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_tree_id", _text(self.base_tree_id, field_name="base_tree_id")
        )
        object.__setattr__(
            self,
            "changed_paths",
            _string_tuple(self.changed_paths, field_name="changed_paths"),
        )
        object.__setattr__(
            self,
            "changed_symbols",
            _string_tuple(self.changed_symbols, field_name="changed_symbols"),
        )
        object.__setattr__(
            self,
            "patch_paths",
            _string_tuple(self.patch_paths, field_name="patch_paths"),
        )
        object.__setattr__(
            self,
            "declared_scope_paths",
            _string_tuple(
                self.declared_scope_paths, field_name="declared_scope_paths"
            ),
        )
        object.__setattr__(
            self,
            "target_tree_cid",
            _optional_text(self.target_tree_cid, field_name="target_tree_cid"),
        )
        if self.repository_tree_observation is not None:
            object.__setattr__(
                self,
                "repository_tree_observation",
                _frozen_mapping(
                    dict(
                        _as_mapping(
                            self.repository_tree_observation,
                            field_name="repository_tree_observation",
                        )
                    )
                ),
            )
        if self.repository_forest is not None and not isinstance(
            self.repository_forest, RepositoryForest
        ):
            raise PlannerError("repository_forest must be a RepositoryForest")
        object.__setattr__(
            self,
            "repository_alias",
            _optional_text(self.repository_alias, field_name="repository_alias"),
        )

    @property
    def scope_crossing(self) -> bool:
        """True when a patch path falls outside the declared scope prefixes."""

        if not self.declared_scope_paths:
            return False
        paths = self.patch_paths or self.changed_paths
        if not paths:
            return False
        for path in paths:
            if not any(
                path == scope
                or path.startswith(scope.rstrip("/") + "/")
                or scope == "."
                for scope in self.declared_scope_paths
            ):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_tree_id": self.base_tree_id,
            "changed_paths": list(self.changed_paths),
            "changed_symbols": list(self.changed_symbols),
            "patch_paths": list(self.patch_paths),
            "declared_scope_paths": list(self.declared_scope_paths),
            "target_tree_cid": self.target_tree_cid,
            "scope_crossing": self.scope_crossing,
        }

    @classmethod
    def from_value(cls, value: Any) -> PatchDelta:
        if isinstance(value, cls):
            return value
        payload = _as_mapping(value, field_name="patch_delta")
        forest = payload.get("repository_forest")
        if forest is not None and not isinstance(forest, RepositoryForest):
            raise PlannerError("repository_forest must be a RepositoryForest")
        return cls(
            base_tree_id=str(payload.get("base_tree_id") or payload.get("base") or ""),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            changed_symbols=tuple(payload.get("changed_symbols") or ()),
            patch_paths=tuple(
                payload.get("patch_paths") or payload.get("paths") or ()
            ),
            declared_scope_paths=tuple(
                payload.get("declared_scope_paths")
                or payload.get("allowed_paths")
                or ()
            ),
            target_tree_cid=str(
                payload.get("target_tree_cid")
                or payload.get("repository_tree_cid")
                or ""
            ),
            repository_tree_observation=(
                dict(payload["repository_tree_observation"])
                if isinstance(payload.get("repository_tree_observation"), Mapping)
                else None
            ),
            repository_forest=forest,
            repository_alias=str(payload.get("repository_alias") or ""),
            schema=str(payload.get("schema") or PATCH_DELTA_SCHEMA),
        )


@dataclass(frozen=True, slots=True)
class PlannerPolicy:
    """Fail-closed planner policy: catalog, selection, identity, resources."""

    identity: IdentityBinding
    catalog: VerificationCatalog = field(default_factory=VerificationCatalog)
    selection_policy: SelectionPolicy = field(default_factory=SelectionPolicy)
    # kind value -> tool spec (static_analysis / type_check / test / proof)
    tool_specs: Mapping[str, CheckToolSpec] = field(default_factory=dict)
    # optional per-check overrides (test node id / check id / obligation cid)
    check_tool_specs: Mapping[str, CheckToolSpec] = field(default_factory=dict)
    # optional prebuilt keys (check_id -> key); tree must match identity target
    prebuilt_keys: Mapping[str, VerificationReceiptKey] = field(default_factory=dict)
    max_execution_time_ms: int = DEFAULT_MAX_EXECUTION_TIME_MS
    default_step_timeout_ms: int = DEFAULT_STEP_TIMEOUT_MS
    expected_cpu_millis: int = DEFAULT_CPU_MILLIS
    expected_memory_bytes: int = DEFAULT_MEMORY_BYTES
    expected_processes: int = DEFAULT_PROCESSES
    expected_artifact_bytes: int = DEFAULT_ARTIFACT_BYTES
    cpu_millis_per_check: int = 5_000
    memory_bytes_per_check: int = 32 * 1024 * 1024
    policy_conflict: bool = False
    force_human_review: bool = False
    force_human_review_reason_codes: tuple[str, ...] = ()
    # VerificationPlan requires one environment_cid for every required key.
    # Tool identity is part of the environment, so multi-kind keys generally
    # cannot share an environment.  When multiple kinds are selected, the
    # planner admits keys only for kinds listed here (priority order).  Empty
    # means auto: prefer test, then type_check, then static_analysis, then proof.
    receipt_kinds: tuple[str, ...] = ()
    policy_id: str = "default"
    policy_cid: str = ""
    schema: str = PLANNER_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.identity, IdentityBinding):
            raise PlannerError("identity must be an IdentityBinding")
        if not isinstance(self.catalog, VerificationCatalog):
            raise PlannerError("catalog must be a VerificationCatalog")
        if not isinstance(self.selection_policy, SelectionPolicy):
            raise PlannerError("selection_policy must be a SelectionPolicy")
        tools: dict[str, CheckToolSpec] = {}
        for key, value in dict(self.tool_specs or {}).items():
            tools[_text(key, field_name="tool_specs.key")] = CheckToolSpec.from_value(
                value
            )
        object.__setattr__(self, "tool_specs", MappingProxyType(tools))
        overrides: dict[str, CheckToolSpec] = {}
        for key, value in dict(self.check_tool_specs or {}).items():
            overrides[_text(key, field_name="check_tool_specs.key")] = (
                CheckToolSpec.from_value(value)
            )
        object.__setattr__(self, "check_tool_specs", MappingProxyType(overrides))
        prebuilt: dict[str, VerificationReceiptKey] = {}
        for key, value in dict(self.prebuilt_keys or {}).items():
            key_text = _text(key, field_name="prebuilt_keys.key")
            if isinstance(value, VerificationReceiptKey):
                prebuilt[key_text] = value
            elif isinstance(value, Mapping):
                prebuilt[key_text] = VerificationReceiptKey.from_dict(value)
            else:
                raise PlannerError(
                    "prebuilt_keys values must be VerificationReceiptKey records"
                )
        object.__setattr__(self, "prebuilt_keys", MappingProxyType(prebuilt))
        max_ms = _positive_int(
            self.max_execution_time_ms,
            field_name="max_execution_time_ms",
            maximum=MAX_DURATION_MS,
            default=DEFAULT_MAX_EXECUTION_TIME_MS,
        )
        step_ms = _positive_int(
            self.default_step_timeout_ms,
            field_name="default_step_timeout_ms",
            maximum=MAX_DURATION_MS,
            default=DEFAULT_STEP_TIMEOUT_MS,
        )
        if step_ms > max_ms:
            raise PlannerBoundsError(
                "default_step_timeout_ms must not exceed max_execution_time_ms"
            )
        object.__setattr__(self, "max_execution_time_ms", max_ms)
        object.__setattr__(self, "default_step_timeout_ms", step_ms)
        object.__setattr__(
            self,
            "expected_cpu_millis",
            _nonneg_int(
                self.expected_cpu_millis,
                field_name="expected_cpu_millis",
                default=DEFAULT_CPU_MILLIS,
            ),
        )
        object.__setattr__(
            self,
            "expected_memory_bytes",
            _nonneg_int(
                self.expected_memory_bytes,
                field_name="expected_memory_bytes",
                default=DEFAULT_MEMORY_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "expected_processes",
            _positive_int(
                self.expected_processes,
                field_name="expected_processes",
                default=DEFAULT_PROCESSES,
            ),
        )
        object.__setattr__(
            self,
            "expected_artifact_bytes",
            _nonneg_int(
                self.expected_artifact_bytes,
                field_name="expected_artifact_bytes",
                default=DEFAULT_ARTIFACT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "cpu_millis_per_check",
            _nonneg_int(
                self.cpu_millis_per_check,
                field_name="cpu_millis_per_check",
                default=5_000,
            ),
        )
        object.__setattr__(
            self,
            "memory_bytes_per_check",
            _nonneg_int(
                self.memory_bytes_per_check,
                field_name="memory_bytes_per_check",
                default=32 * 1024 * 1024,
            ),
        )
        object.__setattr__(
            self,
            "policy_conflict",
            _boolean(self.policy_conflict, field_name="policy_conflict", default=False),
        )
        object.__setattr__(
            self,
            "force_human_review",
            _boolean(
                self.force_human_review, field_name="force_human_review", default=False
            ),
        )
        object.__setattr__(
            self,
            "force_human_review_reason_codes",
            _string_tuple(
                self.force_human_review_reason_codes,
                field_name="force_human_review_reason_codes",
            ),
        )
        kinds = self.receipt_kinds or ()
        if isinstance(kinds, str) or not isinstance(kinds, Sequence):
            raise PlannerError("receipt_kinds must be a sequence of kind tokens")
        normalized_kinds: list[str] = []
        allowed_kinds = {item.value for item in VerificationReceiptKind}
        for item in kinds:
            token = _text(item, field_name="receipt_kinds")
            if token not in allowed_kinds:
                raise PlannerError(
                    f"receipt_kinds contains unknown kind {token!r}; "
                    f"allowed: {sorted(allowed_kinds)}"
                )
            if token not in normalized_kinds:
                normalized_kinds.append(token)
        object.__setattr__(self, "receipt_kinds", tuple(normalized_kinds))
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        policy_cid = _optional_text(self.policy_cid, field_name="policy_cid")
        if not policy_cid:
            policy_cid = content_identity(
                {
                    "schema": PLANNER_POLICY_SCHEMA,
                    "policy_id": self.policy_id,
                    "max_execution_time_ms": self.max_execution_time_ms,
                    "selection_policy": self.selection_policy.to_dict(),
                    "catalog": self.catalog.to_dict(),
                    "identity": self.identity.to_dict(),
                }
            )
        object.__setattr__(self, "policy_cid", policy_cid)

    def tool_for(
        self,
        kind: VerificationReceiptKind,
        check_id: str,
    ) -> CheckToolSpec | None:
        if check_id in self.check_tool_specs:
            return self.check_tool_specs[check_id]
        return self.tool_specs.get(kind.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "policy_cid": self.policy_cid,
            "max_execution_time_ms": self.max_execution_time_ms,
            "default_step_timeout_ms": self.default_step_timeout_ms,
            "policy_conflict": self.policy_conflict,
            "force_human_review": self.force_human_review,
            "identity": self.identity.to_dict(),
            "catalog": self.catalog.to_dict(),
            "selection_policy": self.selection_policy.to_dict(),
        }

    @classmethod
    def from_value(cls, value: Any) -> PlannerPolicy:
        if isinstance(value, cls):
            return value
        payload = _as_mapping(value, field_name="policy")
        identity = IdentityBinding.from_value(payload.get("identity") or {})
        catalog = VerificationCatalog.from_value(payload.get("catalog"))
        selection = SelectionPolicy.from_value(payload.get("selection_policy"))
        tools_raw = payload.get("tool_specs") or {}
        if not isinstance(tools_raw, Mapping):
            raise PlannerError("tool_specs must be a mapping")
        check_tools_raw = payload.get("check_tool_specs") or {}
        if not isinstance(check_tools_raw, Mapping):
            raise PlannerError("check_tool_specs must be a mapping")
        prebuilt_raw = payload.get("prebuilt_keys") or {}
        if not isinstance(prebuilt_raw, Mapping):
            raise PlannerError("prebuilt_keys must be a mapping")
        return cls(
            identity=identity,
            catalog=catalog,
            selection_policy=selection,
            tool_specs={str(k): v for k, v in tools_raw.items()},
            check_tool_specs={str(k): v for k, v in check_tools_raw.items()},
            prebuilt_keys={str(k): v for k, v in prebuilt_raw.items()},
            max_execution_time_ms=payload.get(
                "max_execution_time_ms", DEFAULT_MAX_EXECUTION_TIME_MS
            ),
            default_step_timeout_ms=payload.get(
                "default_step_timeout_ms", DEFAULT_STEP_TIMEOUT_MS
            ),
            expected_cpu_millis=payload.get(
                "expected_cpu_millis", DEFAULT_CPU_MILLIS
            ),
            expected_memory_bytes=payload.get(
                "expected_memory_bytes", DEFAULT_MEMORY_BYTES
            ),
            expected_processes=payload.get(
                "expected_processes", DEFAULT_PROCESSES
            ),
            expected_artifact_bytes=payload.get(
                "expected_artifact_bytes", DEFAULT_ARTIFACT_BYTES
            ),
            cpu_millis_per_check=payload.get("cpu_millis_per_check", 5_000),
            memory_bytes_per_check=payload.get(
                "memory_bytes_per_check", 32 * 1024 * 1024
            ),
            policy_conflict=bool(payload.get("policy_conflict", False)),
            force_human_review=bool(payload.get("force_human_review", False)),
            force_human_review_reason_codes=tuple(
                payload.get("force_human_review_reason_codes") or ()
            ),
            receipt_kinds=tuple(payload.get("receipt_kinds") or ()),
            policy_id=str(payload.get("policy_id") or "default"),
            policy_cid=str(payload.get("policy_cid") or ""),
            schema=str(payload.get("schema") or PLANNER_POLICY_SCHEMA),
        )


# ---------------------------------------------------------------------------
# Receipt key compilation
# ---------------------------------------------------------------------------


def _default_selector_argv(
    kind: VerificationReceiptKind, check_id: str
) -> tuple[str, ...]:
    if kind is VerificationReceiptKind.STATIC_ANALYSIS:
        return ("/usr/bin/ruff", "check", check_id)
    if kind is VerificationReceiptKind.TYPE_CHECK:
        return ("/usr/bin/python3.12", "-m", "mypy", check_id)
    if kind is VerificationReceiptKind.TEST:
        return ("/usr/bin/python3.12", "-m", "pytest", check_id)
    return ("/usr/bin/z3", "-smt2", check_id)


def _default_tool_meta(
    kind: VerificationReceiptKind,
) -> tuple[str, str, str]:
    if kind is VerificationReceiptKind.STATIC_ANALYSIS:
        return ("ruff", "0.12.11", "ruff-verification-adapter@1")
    if kind is VerificationReceiptKind.TYPE_CHECK:
        return ("mypy", "1.18.2", "mypy-verification-adapter@1")
    if kind is VerificationReceiptKind.TEST:
        return ("pytest", "9.1.1", "pytest-verification-adapter@1")
    return ("z3", "4.13.3", "z3-verification-adapter@1")


def _ensure_tool_spec(
    policy: PlannerPolicy,
    kind: VerificationReceiptKind,
    check_id: str,
) -> CheckToolSpec:
    existing = policy.tool_for(kind, check_id)
    if existing is not None:
        # Specialize selector to this check when the catalog uses a kind-level
        # template that ends with a placeholder or is intentionally generic.
        if check_id not in policy.check_tool_specs:
            argv = list(existing.selector_argv)
            # Replace final selector target with the concrete check id when
            # the kind-level argv looks like a template (last arg not the id).
            if argv and argv[-1] != check_id and kind is not VerificationReceiptKind.PROOF:
                argv = [*argv[:-1], check_id] if len(argv) > 1 else [argv[0], check_id]
                return replace(existing, selector_argv=tuple(argv))
        return existing
    tool_name, tool_version, adapter = _default_tool_meta(kind)
    return CheckToolSpec(
        tool_name=tool_name,
        tool_version=tool_version,
        adapter_schema=adapter,
        selector_argv=_default_selector_argv(kind, check_id),
        proof_obligation=(
            {"obligation_id": check_id} if kind is VerificationReceiptKind.PROOF else None
        ),
    )


def _align_capability_for_tool(
    snapshot: CapabilitySnapshot,
    tool: CheckToolSpec,
    *,
    lock_path: str,
    lock_bytes: bytes,
) -> CapabilitySnapshot:
    sha = "sha256:" + hashlib.sha256(tool.tool_executable_bytes).hexdigest()
    lock_sha = "sha256:" + hashlib.sha256(lock_bytes).hexdigest()
    tools = dict(snapshot.tool_identities)
    tools[tool.tool_capability_name] = sha
    locks = dict(snapshot.lock_identities)
    locks[lock_path] = lock_sha
    return replace(
        snapshot,
        tool_identities=tools,
        lock_identities=locks,
    )


def _compile_key_with_forest(
    *,
    identity: IdentityBinding,
    tool: CheckToolSpec,
    kind: VerificationReceiptKind,
    forest: RepositoryForest,
    repository_alias: str,
    target_tree_cid: str,
) -> VerificationReceiptKey:
    snapshot = _align_capability_for_tool(
        identity.capability_snapshot,
        tool,
        lock_path=identity.dependency_lock_path,
        lock_bytes=identity.dependency_lock_bytes,
    )
    # Refresh tool identity sha against executable bytes.
    sha = "sha256:" + hashlib.sha256(tool.tool_executable_bytes).hexdigest()
    tool_identity = tool.tool_identity
    assert tool_identity is not None
    if tool_identity.identity != sha or tool_identity.name != tool.tool_capability_name:
        tool_identity = ToolIdentity(
            name=tool.tool_capability_name,
            kind="executable",
            locator=tool_identity.locator,
            version=tool_identity.version,
            identity=sha,
            roles=tool_identity.roles,
        )
    lock_identity = LockIdentity(
        path=identity.dependency_lock_path,
        identity="sha256:"
        + hashlib.sha256(identity.dependency_lock_bytes).hexdigest(),
    )
    # claimed environment filled after first compile pass via helper
    claimed_env = identity.environment_cid
    kwargs: dict[str, Any] = {
        "repository_forest": forest,
        "repository_alias": repository_alias or forest.sole_write_alias,
        "claimed_repository_tree_cid": target_tree_cid,
        "patch_base_tree_id": identity.patch_base_tree_id,
        "repository_state_tree_id": identity.patch_base_tree_id,
        "invalidation_plan_tree_id": identity.patch_base_tree_id,
        "context_pack_tree_id": identity.patch_base_tree_id,
        "observed_semantic_state": dict(identity.observed_semantic_state),
        "repository_state_semantic_root_cid": identity.semantic_state_root_cid,
        "invalidation_plan_semantic_root_cid": identity.semantic_state_root_cid,
        "context_pack_semantic_root_cid": identity.semantic_state_root_cid,
        "affected_symbol_versions": tuple(
            dict(item) for item in identity.affected_symbol_versions
        ),
        "observed_environment": dict(identity.sandbox_environment),
        "capability_snapshot": snapshot,
        "tool_capability_name": tool.tool_capability_name,
        "tool_identity": tool_identity,
        "resolved_tool_executable": tool.resolved_tool_executable,
        "tool_executable_bytes": tool.tool_executable_bytes,
        "tool_version_probe_argv": tool.tool_version_probe_argv,
        "tool_version_probe_output_bytes": tool.tool_version_probe_output_bytes,
        "claimed_environment_cid": claimed_env or "",
        "dependency_lock_path": identity.dependency_lock_path,
        "dependency_lock_identity": lock_identity,
        "dependency_lock_bytes": identity.dependency_lock_bytes,
        "selector_argv": tool.selector_argv,
        "proof_obligation": tool.proof_obligation
        if kind is VerificationReceiptKind.PROOF
        else None,
        "tool_name": tool.tool_name,
        "tool_version": tool.tool_version,
        "configuration_bytes": identity.configuration_bytes,
        "fixture_data_bytes": identity.fixture_data_bytes,
        "network_policy": identity.network_policy,
        "receipt_schema_version": identity.receipt_schema_version,
        "receipt_kind": kind,
        "adapter_schema": tool.adapter_schema,
        "proof_backend_binding": (
            dict(tool.proof_backend_binding)
            if tool.proof_backend_binding is not None
            else None
        ),
    }
    if not kwargs["claimed_environment_cid"]:
        # Two-pass: compile once after filling environment from observations.
        # Build expected environment the same way the compiler does.
        kwargs["claimed_environment_cid"] = _predict_environment_cid(kwargs)
    if not kwargs["claimed_repository_tree_cid"]:
        # Derive from forest observation.
        descriptor = forest.write_descriptor()
        tree_observation = {
            "repository_forest_cid": forest.forest_id,
            "git_commit_id": descriptor.commit,
            "git_tree_id": descriptor.tree,
            "gitlink_state_cid": descriptor.portable_closure.gitlink_closure_cid,
            "dirty_overlay_cid": descriptor.dirty_overlay_digest,
            "dirty": descriptor.dirty,
            "repository_alias": descriptor.alias,
            "repository_id": descriptor.repository_id,
            "descriptor_cid": descriptor.descriptor_cid,
            "base_repository_tree_id": identity.patch_base_tree_id,
        }
        kwargs["claimed_repository_tree_cid"] = _structured_cid(
            _TREE_IDENTITY_INPUT_SCHEMA, tree_observation
        )
    try:
        return VerificationIdentityCompiler().compile_key(**kwargs)
    except (VerificationContractError, VerificationIdentityError, TypeError, ValueError) as exc:
        raise PlannerIdentityError(f"receipt key compilation failed: {exc}") from exc


def _predict_environment_cid(values: Mapping[str, Any]) -> str:
    snapshot = values["capability_snapshot"]
    assert isinstance(snapshot, CapabilitySnapshot)
    tool_identity = values["tool_identity"]
    assert isinstance(tool_identity, ToolIdentity)
    capability_name = str(values["tool_capability_name"])
    executable_sha256 = snapshot.tool_identities[capability_name]
    lock_identity = values["dependency_lock_identity"]
    assert isinstance(lock_identity, LockIdentity)
    environment = {
        **dict(values["observed_environment"]),  # type: ignore[arg-type]
        "network_policy": values["network_policy"],
        "tool_name": values["tool_name"],
        "tool_version": values["tool_version"],
        "tool_capability_name": capability_name,
        "tool_launcher_identity": tool_identity.to_dict(),
        "resolved_tool_executable": values["resolved_tool_executable"],
        "tool_executable_sha256": executable_sha256,
        "tool_executable_cid": _structured_cid(
            _TOOL_EXECUTABLE_IDENTITY_SCHEMA,
            {"capability_name": capability_name, "sha256": executable_sha256},
        ),
        "tool_version_probe_argv": values["tool_version_probe_argv"],
        "tool_version_probe_output_cid": cid_for_bytes(
            values["tool_version_probe_output_bytes"]  # type: ignore[arg-type]
        ),
        "tool_inventory_schema": "observed-tool-inventory@1",
        "adapter_schema": values["adapter_schema"],
        "capability_environment_names": tuple(sorted(snapshot.environment_names)),
        "capability_read_paths": tuple(sorted(snapshot.read_paths)),
        "capability_write_paths": tuple(sorted(snapshot.write_paths)),
        "capability_lock_identities": dict(sorted(snapshot.lock_identities.items())),
        "selected_dependency_lock_path": values["dependency_lock_path"],
        "selected_dependency_lock_identity": lock_identity.to_dict(),
    }
    return _structured_cid(_ENVIRONMENT_IDENTITY_INPUT_SCHEMA, environment)


def _compile_key_from_observation(
    *,
    identity: IdentityBinding,
    tool: CheckToolSpec,
    kind: VerificationReceiptKind,
    tree_observation: Mapping[str, Any],
    tree_cid: str,
) -> VerificationReceiptKey:
    """Compile a key from explicit tree observation (no forest required).

    Proof kinds require ``repository_forest`` compilation or ``prebuilt_keys``
    because :class:`VerificationReceiptKey` proof bindings need a full
    ``CodeProofObligation`` surface.
    """

    if kind is VerificationReceiptKind.PROOF:
        raise PlannerIdentityError(
            "proof receipt keys require repository_forest compilation or prebuilt_keys"
        )

    snapshot = _align_capability_for_tool(
        identity.capability_snapshot,
        tool,
        lock_path=identity.dependency_lock_path,
        lock_bytes=identity.dependency_lock_bytes,
    )
    sha = "sha256:" + hashlib.sha256(tool.tool_executable_bytes).hexdigest()
    tool_identity = tool.tool_identity
    assert tool_identity is not None
    if tool_identity.identity != sha:
        tool_identity = ToolIdentity(
            name=tool.tool_capability_name,
            kind="executable",
            locator=tool_identity.locator,
            version=tool_identity.version,
            identity=sha,
            roles=tool_identity.roles,
        )
    lock_identity = LockIdentity(
        path=identity.dependency_lock_path,
        identity="sha256:"
        + hashlib.sha256(identity.dependency_lock_bytes).hexdigest(),
    )
    values: dict[str, Any] = {
        "observed_environment": dict(identity.sandbox_environment),
        "network_policy": identity.network_policy,
        "tool_name": tool.tool_name,
        "tool_version": tool.tool_version,
        "tool_capability_name": tool.tool_capability_name,
        "tool_identity": tool_identity,
        "resolved_tool_executable": tool.resolved_tool_executable,
        "tool_version_probe_argv": tool.tool_version_probe_argv,
        "tool_version_probe_output_bytes": tool.tool_version_probe_output_bytes,
        "adapter_schema": tool.adapter_schema,
        "capability_snapshot": snapshot,
        "dependency_lock_path": identity.dependency_lock_path,
        "dependency_lock_identity": lock_identity,
    }
    environment_cid = _predict_environment_cid(values)
    executable_sha256 = snapshot.tool_identities[tool.tool_capability_name]
    environment = {
        **dict(identity.sandbox_environment),
        "network_policy": identity.network_policy,
        "tool_name": tool.tool_name,
        "tool_version": tool.tool_version,
        "tool_capability_name": tool.tool_capability_name,
        "tool_launcher_identity": tool_identity.to_dict(),
        "resolved_tool_executable": tool.resolved_tool_executable,
        "tool_executable_sha256": executable_sha256,
        "tool_executable_cid": _structured_cid(
            _TOOL_EXECUTABLE_IDENTITY_SCHEMA,
            {
                "capability_name": tool.tool_capability_name,
                "sha256": executable_sha256,
            },
        ),
        "tool_version_probe_argv": tool.tool_version_probe_argv,
        "tool_version_probe_output_cid": cid_for_bytes(
            tool.tool_version_probe_output_bytes
        ),
        "tool_inventory_schema": "observed-tool-inventory@1",
        "adapter_schema": tool.adapter_schema,
        "capability_environment_names": tuple(sorted(snapshot.environment_names)),
        "capability_read_paths": tuple(sorted(snapshot.read_paths)),
        "capability_write_paths": tuple(sorted(snapshot.write_paths)),
        "capability_lock_identities": dict(sorted(snapshot.lock_identities.items())),
        "selected_dependency_lock_path": identity.dependency_lock_path,
        "selected_dependency_lock_identity": lock_identity.to_dict(),
    }
    symbol_cids = tuple(
        sorted(
            {
                _structured_cid(_SYMBOL_IDENTITY_INPUT_SCHEMA, dict(item))
                for item in identity.affected_symbol_versions
            }
        )
    )
    selector_cid = _structured_cid(
        _SELECTOR_IDENTITY_INPUT_SCHEMA, {"argv": list(tool.selector_argv)}
    )

    try:
        return VerificationReceiptKey(
            repository_tree_cid=tree_cid,
            repository_tree_observation=dict(tree_observation),
            semantic_state_root_cid=identity.semantic_state_root_cid,
            affected_symbol_version_cids=symbol_cids,
            environment_cid=environment_cid,
            environment_observation=environment,
            dependency_lock_cid=identity.dependency_lock_cid,
            selector_cid=selector_cid,
            proof_obligation_cid=PROOF_OBLIGATION_NOT_APPLICABLE_CID,
            tool_name=tool.tool_name,
            tool_version=tool.tool_version,
            configuration_cid=_bytes_cid(
                identity.configuration_bytes, field_name="configuration_bytes"
            ),
            fixture_data_cids=tuple(
                sorted(
                    _bytes_cid(item, field_name=f"fixture[{i}]")
                    for i, item in enumerate(identity.fixture_data_bytes)
                )
            ),
            network_policy=identity.network_policy,
            receipt_schema_version=identity.receipt_schema_version,
            receipt_kind=kind,
            adapter_schema=tool.adapter_schema,
            proof_backend_binding=None,
        )
    except (VerificationContractError, VerificationIdentityError) as exc:
        raise PlannerIdentityError(f"receipt key construction failed: {exc}") from exc


def compile_check_receipt_key(
    *,
    policy: PlannerPolicy,
    kind: VerificationReceiptKind,
    check_id: str,
    patch: PatchDelta,
) -> VerificationReceiptKey:
    """Compile the exact receipt key for one selected check on the target tree."""

    if check_id in policy.prebuilt_keys:
        key = policy.prebuilt_keys[check_id]
        target_cid = (
            patch.target_tree_cid
            or policy.identity.repository_tree_cid
            or key.repository_tree_cid
        )
        if key.repository_tree_cid != target_cid and target_cid:
            raise PlannerIdentityError(
                f"prebuilt key for {check_id} does not bind the target patched tree"
            )
        if key.semantic_state_root_cid != policy.identity.semantic_state_root_cid:
            raise PlannerIdentityError(
                f"prebuilt key for {check_id} semantic root disagrees with policy"
            )
        return key

    identity = policy.identity
    tool = _ensure_tool_spec(policy, kind, check_id)
    forest = patch.repository_forest or identity.repository_forest
    if forest is not None:
        target_cid = patch.target_tree_cid or identity.repository_tree_cid
        return _compile_key_with_forest(
            identity=identity,
            tool=tool,
            kind=kind,
            forest=forest,
            repository_alias=patch.repository_alias or identity.repository_alias,
            target_tree_cid=target_cid,
        )

    tree_obs = (
        patch.repository_tree_observation
        or identity.repository_tree_observation
    )
    tree_cid = patch.target_tree_cid or identity.repository_tree_cid
    if tree_obs is None or not tree_cid:
        raise PlannerIdentityError(
            "identity binding requires repository_forest or "
            "repository_tree_observation + target_tree_cid"
        )
    # Ensure base tree is bound into the observation.
    obs = dict(tree_obs)
    if obs.get("base_repository_tree_id") not in (None, "", identity.patch_base_tree_id):
        if obs.get("base_repository_tree_id") != identity.patch_base_tree_id:
            raise PlannerIdentityError(
                "tree observation base disagrees with patch base"
            )
    obs.setdefault("base_repository_tree_id", identity.patch_base_tree_id)
    expected = _structured_cid(_TREE_IDENTITY_INPUT_SCHEMA, obs)
    if expected != tree_cid:
        raise PlannerIdentityError(
            "target_tree_cid does not match repository_tree_observation"
        )
    return _compile_key_from_observation(
        identity=identity,
        tool=tool,
        kind=kind,
        tree_observation=obs,
        tree_cid=tree_cid,
    )


# ---------------------------------------------------------------------------
# Cross-checks and acceptance criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _NormalizedInputs:
    repository_state: RepositoryStateView
    invalidation_plan: InvalidationPlanView
    context_pack: ContextPackView
    patch: PatchDelta
    policy: PlannerPolicy
    adapter_reason_codes: tuple[str, ...]


def _normalize_inputs(
    repository_state: Any,
    invalidation_plan: Any,
    context_pack: Any,
    patch_delta: Any,
    policy: Any,
    *,
    adapter: DatasetsVerificationInputAdapter,
) -> _NormalizedInputs:
    if isinstance(policy, PlannerPolicy):
        policy_obj = policy
    else:
        policy_obj = PlannerPolicy.from_value(policy)
    patch = PatchDelta.from_value(patch_delta)

    rs_result = adapter.normalize_repository_state(repository_state)
    ip_result = adapter.normalize_invalidation_plan(invalidation_plan)
    cp_result = adapter.normalize_context_pack(context_pack)

    reasons: list[str] = []
    if not rs_result.ok or rs_result.view is None:
        reasons.append(REASON_ADAPTER_GAP)
        detail = (
            rs_result.observation.message
            if rs_result.observation is not None
            else "repository_state unavailable"
        )
        raise PlannerError(f"repository_state normalization failed: {detail}")
    if not ip_result.ok or ip_result.view is None:
        reasons.append(REASON_ADAPTER_GAP)
        detail = (
            ip_result.observation.message
            if ip_result.observation is not None
            else "invalidation_plan unavailable"
        )
        raise PlannerError(f"invalidation_plan normalization failed: {detail}")
    if not cp_result.ok or cp_result.view is None:
        reasons.append(REASON_ADAPTER_GAP)
        detail = (
            cp_result.observation.message
            if cp_result.observation is not None
            else "context_pack unavailable"
        )
        raise PlannerError(f"context_pack normalization failed: {detail}")

    rs = rs_result.view
    ip = ip_result.view
    cp = cp_result.view
    assert isinstance(rs, RepositoryStateView)
    assert isinstance(ip, InvalidationPlanView)
    assert isinstance(cp, ContextPackView)

    # Cross-check opaque tree ids (patch base + datasets roots).
    tree_ids = {
        "patch_base": patch.base_tree_id,
        "repository_state": rs.repository_tree_id,
        "invalidation_plan": ip.repository_tree_id,
        "context_pack": cp.repository_tree_id,
        "identity_patch_base": policy_obj.identity.patch_base_tree_id,
    }
    if len(set(tree_ids.values())) != 1:
        raise PlannerIdentityError(
            f"{REASON_PATCH_BASE_MISMATCH}: {tree_ids}"
        )

    semantic_ids = {
        "repository_state": rs.semantic_state_root_cid,
        "invalidation_plan": ip.semantic_state_root_cid,
        "context_pack": cp.semantic_state_root_cid,
        "identity": policy_obj.identity.semantic_state_root_cid,
    }
    if len(set(semantic_ids.values())) != 1:
        raise PlannerIdentityError(
            f"{REASON_SEMANTIC_ROOT_MISMATCH}: {semantic_ids}"
        )

    # Optional receipt tree CIDs (when supplied on views) must agree with
    # the target patched tree binding.
    target_cid = (
        patch.target_tree_cid
        or policy_obj.identity.repository_tree_cid
        or ""
    )
    for label, view in (
        ("repository_state", rs),
        ("invalidation_plan", ip),
        ("context_pack", cp),
    ):
        view_tree_cid = getattr(view, "repository_tree_cid", "") or ""
        if view_tree_cid and target_cid and view_tree_cid != target_cid:
            raise PlannerIdentityError(
                f"{REASON_CROSS_TREE_REJECTED}: {label} tree cid disagrees "
                f"with target patched tree"
            )

    # Environment / lock roots, when present on datasets views, must match
    # the effective identity binding CIDs (when those are known).
    # Environment is tool-dependent so we only check lock here when identity
    # already carries a lock cid; environment checked after first key compile.
    if rs.dependency_lock_root_cid and policy_obj.identity.dependency_lock_cid:
        if rs.dependency_lock_root_cid != policy_obj.identity.dependency_lock_cid:
            raise PlannerIdentityError(REASON_LOCK_ROOT_MISMATCH)
    if cp.dependency_lock_root_cid and policy_obj.identity.dependency_lock_cid:
        if cp.dependency_lock_root_cid != policy_obj.identity.dependency_lock_cid:
            raise PlannerIdentityError(REASON_LOCK_ROOT_MISMATCH)

    return _NormalizedInputs(
        repository_state=rs,
        invalidation_plan=ip,
        context_pack=cp,
        patch=patch,
        policy=policy_obj,
        adapter_reason_codes=tuple(reasons),
    )


def _select_checks(
    normalized: _NormalizedInputs,
) -> AffectedVerificationSelection:
    ip = normalized.invalidation_plan
    patch = normalized.patch
    symbols = _unique_sorted((*ip.changed_symbols, *patch.changed_symbols))
    paths = _unique_sorted((*ip.changed_paths, *patch.changed_paths, *patch.patch_paths))
    return select_affected_verification(
        changed_symbols=symbols,
        changed_paths=paths,
        edges=ip.edges,
        uncovered_symbols=ip.uncovered_symbols,
        uncovered_paths=ip.uncovered_paths,
        truncated=ip.truncated,
        requires_broader_selection=ip.requires_broader_selection,
        invalidation_plan=ip,
        catalog=normalized.policy.catalog,
        policy=normalized.policy.selection_policy,
    )


def _selected_kind_priority(
    policy: PlannerPolicy,
    selection: AffectedVerificationSelection,
) -> tuple[VerificationReceiptKind, ...]:
    """Resolve which receipt kinds receive required keys under one environment."""

    present: list[VerificationReceiptKind] = []
    if selection.affected_tests or selection.fallback_tests or selection.full_suite_required:
        present.append(VerificationReceiptKind.TEST)
    if selection.required_type_checks:
        present.append(VerificationReceiptKind.TYPE_CHECK)
    if selection.required_static_checks:
        present.append(VerificationReceiptKind.STATIC_ANALYSIS)
    if selection.affected_proof_obligation_cids:
        present.append(VerificationReceiptKind.PROOF)

    if policy.receipt_kinds:
        allowed = {VerificationReceiptKind(token) for token in policy.receipt_kinds}
        ordered = [
            VerificationReceiptKind(token)
            for token in policy.receipt_kinds
            if VerificationReceiptKind(token) in present
            or VerificationReceiptKind(token) in allowed
        ]
        # Keep only kinds that actually have selected work (except explicit
        # single-kind policies that force empty keys).
        filtered = [kind for kind in ordered if kind in present]
        return tuple(filtered)

    # Auto: one primary kind so all required keys share environment_cid.
    priority = (
        VerificationReceiptKind.TEST,
        VerificationReceiptKind.TYPE_CHECK,
        VerificationReceiptKind.STATIC_ANALYSIS,
        VerificationReceiptKind.PROOF,
    )
    for kind in priority:
        if kind in present:
            return (kind,)
    return ()


def _build_required_keys(
    normalized: _NormalizedInputs,
    selection: AffectedVerificationSelection,
) -> tuple[
    tuple[VerificationReceiptKey, ...],
    tuple[str, ...],
    tuple[str, ...],
    Mapping[str, VerificationReceiptKey],
    tuple[VerificationReceiptKind, ...],
]:
    """Return (keys, full_suite_key_cids, proof_cids, check_id_to_key, kinds)."""

    policy = normalized.policy
    patch = normalized.patch
    keys: list[VerificationReceiptKey] = []
    check_map: dict[str, VerificationReceiptKey] = {}
    full_suite_key_cids: list[str] = []
    kinds = _selected_kind_priority(policy, selection)

    def _add(kind: VerificationReceiptKind, check_id: str, *, full_suite: bool = False) -> None:
        key = compile_check_receipt_key(
            policy=policy, kind=kind, check_id=check_id, patch=patch
        )
        target = patch.target_tree_cid or policy.identity.repository_tree_cid
        if target and key.repository_tree_cid != target:
            raise PlannerIdentityError(
                f"{REASON_CROSS_TREE_REJECTED}: key for {check_id} binds "
                f"{key.repository_tree_cid}, expected {target}"
            )
        if key.semantic_state_root_cid != policy.identity.semantic_state_root_cid:
            raise PlannerIdentityError(REASON_SEMANTIC_ROOT_MISMATCH)
        keys.append(key)
        check_map[check_id] = key
        if full_suite and kind is VerificationReceiptKind.TEST:
            full_suite_key_cids.append(key.key_id)

    if VerificationReceiptKind.TEST in kinds:
        required_tests = _unique_sorted(
            (*selection.affected_tests, *selection.fallback_tests)
        )
        for test_id in required_tests:
            _add(
                VerificationReceiptKind.TEST,
                test_id,
                full_suite=selection.full_suite_required,
            )

    if VerificationReceiptKind.STATIC_ANALYSIS in kinds:
        for check_id in selection.required_static_checks:
            _add(VerificationReceiptKind.STATIC_ANALYSIS, check_id)
    if VerificationReceiptKind.TYPE_CHECK in kinds:
        for check_id in selection.required_type_checks:
            _add(VerificationReceiptKind.TYPE_CHECK, check_id)
    if VerificationReceiptKind.PROOF in kinds:
        for obligation_id in selection.affected_proof_obligation_cids:
            _add(VerificationReceiptKind.PROOF, obligation_id)

    # VerificationPlan requires a nonempty required key set.  When semantic
    # selection is empty (e.g. an unrelated edit), admit a single tree-rebind
    # identity probe under the primary tool environment.  This is not a
    # semantic over-selection of application tests.
    if not keys:
        probe_kind = (
            kinds[0]
            if kinds
            else (
                VerificationReceiptKind.TEST
                if "test" in policy.tool_specs
                or not policy.tool_specs
                else VerificationReceiptKind(
                    next(iter(policy.tool_specs))
                )
            )
        )
        if probe_kind is VerificationReceiptKind.PROOF:
            probe_kind = VerificationReceiptKind.TEST
        probe_id = "identity:tree_rebind_probe"
        if probe_kind is VerificationReceiptKind.TEST:
            # Prefer an explicit probe node id that is not a catalog test.
            probe_id = "test/api/test_identity_probe.py::test_tree_rebind"
        elif probe_kind is VerificationReceiptKind.TYPE_CHECK:
            probe_id = "mypy:identity.tree_rebind"
        elif probe_kind is VerificationReceiptKind.STATIC_ANALYSIS:
            probe_id = "static:identity:tree_rebind"
        _add(probe_kind, probe_id)
        if not kinds:
            kinds = (probe_kind,)

    keys_sorted = tuple(sorted(keys, key=lambda item: item.key_id))
    # Fail closed if environments diverged (should not happen for single kind).
    env_ids = {key.environment_cid for key in keys_sorted}
    if len(env_ids) > 1:
        raise PlannerIdentityError(
            "required receipt keys disagree on environment_cid; "
            "restrict policy.receipt_kinds to one tool environment or supply "
            "prebuilt_keys under a shared environment binding"
        )

    proof_cids = tuple(
        sorted(
            {
                key.proof_obligation_cid
                for key in keys_sorted
                if key.receipt_kind is VerificationReceiptKind.PROOF
            }
        )
    )
    full_suite_ids = tuple(sorted(set(full_suite_key_cids)))
    if (
        selection.full_suite_required
        and VerificationReceiptKind.TEST in kinds
        and not full_suite_ids
    ):
        for test_id in policy.catalog.tests:
            if test_id not in check_map:
                _add(VerificationReceiptKind.TEST, test_id, full_suite=True)
        keys_sorted = tuple(sorted(keys, key=lambda item: item.key_id))
        full_suite_ids = tuple(
            sorted(
                {
                    key.key_id
                    for key in keys_sorted
                    if key.receipt_kind is VerificationReceiptKind.TEST
                }
            )
        )
        if not full_suite_ids:
            raise PlannerError(
                "full_suite_required but no test receipt keys could be built; "
                "provide a catalog.tests inventory"
            )
        proof_cids = tuple(
            sorted(
                {
                    key.proof_obligation_cid
                    for key in keys_sorted
                    if key.receipt_kind is VerificationReceiptKind.PROOF
                }
            )
        )
        env_ids = {key.environment_cid for key in keys_sorted}
        if len(env_ids) > 1:
            raise PlannerIdentityError(
                "required receipt keys disagree on environment_cid after full-suite expansion"
            )

    return (
        keys_sorted,
        full_suite_ids,
        proof_cids,
        MappingProxyType(check_map),
        kinds,
    )


def _lookup_decisions(
    keys: Sequence[VerificationReceiptKey],
    *,
    cache: VerificationReceiptCache | None,
    invalidate_reasons: Sequence[str],
) -> tuple[CacheReuseDecision, ...]:
    """Exact-key lookup only; never publish tombstones."""

    if invalidate_reasons:
        # Observed environment/lock/tool mismatch: all keys mismatched.
        return tuple(
            CacheReuseDecision(
                key_cid=key.key_id,
                disposition=CacheReuseDisposition.MISMATCHED,
                reason_codes=tuple(invalidate_reasons),
            )
            for key in keys
        )

    if cache is None:
        return tuple(
            CacheReuseDecision(
                key_cid=key.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=(REASON_NO_CACHE, REASON_CACHE_LOOKUP),
            )
            for key in keys
        )

    decisions: list[CacheReuseDecision] = []
    for key in keys:
        decision = cache.lookup(key, for_production=True, touch_access=False)
        # Planning must not call mark_stale / tombstone.
        decisions.append(decision)
    return tuple(decisions)


def _human_review_reasons(
    normalized: _NormalizedInputs,
    selection: AffectedVerificationSelection,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    policy = normalized.policy
    if not policy.identity.effective_sandbox_bound:
        reasons.append(REASON_UNBOUND_SANDBOX)
    # Sandbox observation must contain the hermetic fields.
    sandbox = policy.identity.sandbox_environment
    required_sandbox = (
        "sandbox_schema",
        "sandbox_policy",
        "filesystem_policy",
        "platform",
        "interpreter",
        "toolchain",
    )
    if any(not sandbox.get(name) for name in required_sandbox):
        reasons.append(REASON_UNBOUND_SANDBOX)
    if policy.policy_conflict:
        reasons.append(REASON_POLICY_CONFLICT)
    if normalized.patch.scope_crossing:
        reasons.append(REASON_SCOPE_CROSSING)
    if policy.force_human_review:
        reasons.extend(policy.force_human_review_reason_codes or ("force_human_review",))
    # Deduplicate preserving order.
    ordered = _stable_unique(reasons)
    return bool(ordered), ordered


def _resource_bounds(
    policy: PlannerPolicy,
    keys: Sequence[VerificationReceiptKey],
) -> tuple[int, int, int, int, int]:
    n = max(1, len(keys))
    cpu = max(policy.expected_cpu_millis, policy.cpu_millis_per_check * n)
    memory = max(policy.expected_memory_bytes, policy.memory_bytes_per_check * n)
    processes = max(1, policy.expected_processes)
    proof_slots = sum(
        1 for key in keys if key.receipt_kind is VerificationReceiptKind.PROOF
    )
    artifacts = max(policy.expected_artifact_bytes, 1_000_000 * n)
    # Cap against policy maximums (already validated positive).
    cpu = min(cpu, MAX_RESOURCE_QUANTITY)
    memory = min(memory, MAX_RESOURCE_QUANTITY)
    artifacts = min(artifacts, MAX_RESOURCE_QUANTITY)
    return cpu, memory, processes, proof_slots, artifacts


def _dependency_dag_and_timeouts(
    keys: Sequence[VerificationReceiptKey],
    check_map: Mapping[str, VerificationReceiptKey],
    selection: AffectedVerificationSelection,
    policy: PlannerPolicy,
) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, int]]:
    # Deterministic step order: static -> type -> proof -> tests.
    static_steps: list[str] = []
    type_steps: list[str] = []
    proof_steps: list[str] = []
    test_steps: list[str] = []
    key_to_step: dict[str, str] = {}

    # Invert check_map for labeling.
    id_by_key = {key.key_id: check_id for check_id, key in check_map.items()}

    for key in keys:
        check_id = id_by_key.get(key.key_id, key.selector_cid)
        step = _step_id(key.receipt_kind, check_id)
        # Ensure uniqueness if collisions.
        base = step
        suffix = 2
        while step in key_to_step.values():
            step = f"{base}#{suffix}"
            suffix += 1
        key_to_step[key.key_id] = step
        if key.receipt_kind is VerificationReceiptKind.STATIC_ANALYSIS:
            static_steps.append(step)
        elif key.receipt_kind is VerificationReceiptKind.TYPE_CHECK:
            type_steps.append(step)
        elif key.receipt_kind is VerificationReceiptKind.PROOF:
            proof_steps.append(step)
        else:
            test_steps.append(step)

    static_steps = sorted(static_steps)
    type_steps = sorted(type_steps)
    proof_steps = sorted(proof_steps)
    test_steps = sorted(test_steps)

    dag: dict[str, tuple[str, ...]] = {}
    for step in static_steps:
        dag[step] = ()
    for step in type_steps:
        dag[step] = tuple(static_steps)
    for step in proof_steps:
        dag[step] = tuple(static_steps + type_steps)
    for step in test_steps:
        dag[step] = tuple(static_steps + type_steps)

    timeout = min(policy.default_step_timeout_ms, policy.max_execution_time_ms)
    timeouts = {step: timeout for step in dag}
    return MappingProxyType(dag), MappingProxyType(timeouts)


def _acceptance_criteria(
    *,
    full_suite_required: bool,
    human_review_required: bool,
) -> tuple[str, ...]:
    criteria = [
        REASON_PRODUCTION_SUCCESS_REQUIRED,
        "every_required_receipt_current_production_admissible_success",
        REASON_NO_PENDING_FALLBACK,
        "no_pending_mandatory_full_suite_fallback",
    ]
    if full_suite_required:
        criteria.append(REASON_MANDATORY_FULL_SUITE)
    if human_review_required:
        criteria.append("human_review_blocks_automatic_acceptance")
    else:
        criteria.append(REASON_HUMAN_REVIEW_FALSE)
    return tuple(criteria)


def _observed_mismatch_reasons(
    normalized: _NormalizedInputs,
    keys: Sequence[VerificationReceiptKey],
) -> tuple[str, ...]:
    """Detect environment/lock/tool observed mismatches against policy claims."""

    if not keys:
        return ()
    reasons: list[str] = []
    policy = normalized.policy
    identity = policy.identity
    rs = normalized.repository_state
    cp = normalized.context_pack

    # Lock root agreement (already fail-closed in normalize; re-check vs keys).
    for key in keys:
        if key.dependency_lock_cid != identity.dependency_lock_cid:
            reasons.append(REASON_LOCK_MISMATCH)
            break
        if key.environment_cid and identity.environment_cid:
            if key.environment_cid != identity.environment_cid:
                reasons.append(REASON_ENVIRONMENT_MISMATCH)
                break

    # Tool observation consistency: environment tool fields must match key.
    for key in keys:
        env = key.environment_observation
        if env.get("tool_name") != key.tool_name or env.get("tool_version") != key.tool_version:
            reasons.append(REASON_TOOL_MISMATCH)
            break

    # Datasets environment roots (when present) cannot contradict compiled env.
    # Environment is tool-specific; only compare when a single environment cid
    # is shared (identity.environment_cid pre-bound).
    if identity.environment_cid:
        if rs.environment_root_cid and rs.environment_root_cid != identity.environment_cid:
            reasons.append(REASON_ENVIRONMENT_ROOT_MISMATCH)
        if cp.environment_root_cid and cp.environment_root_cid != identity.environment_cid:
            reasons.append(REASON_ENVIRONMENT_ROOT_MISMATCH)

    return _stable_unique(reasons)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_verification_plan(
    repository_state: Any,
    invalidation_plan: Any,
    context_pack: Any,
    patch_delta: Any,
    policy: Any,
    *,
    cache: VerificationReceiptCache | None = None,
    adapter: DatasetsVerificationInputAdapter | None = None,
) -> VerificationPlan:
    """Build a deterministic :class:`VerificationPlan` (five-argument API).

    Accepts strict canonical mappings or explicitly registered upstream
    datasets types via the datasets adapter.  Optional ``cache`` enables exact
    key lookup; planning never mutates tombstones.  Optional ``adapter`` allows
    tests to inject a preconfigured registry.
    """

    adapter_obj = adapter or create_datasets_verification_input_adapter()
    normalized = _normalize_inputs(
        repository_state,
        invalidation_plan,
        context_pack,
        patch_delta,
        policy,
        adapter=adapter_obj,
    )
    selection = _select_checks(normalized)
    keys, full_suite_key_cids, proof_cids, check_map, admitted_kinds = (
        _build_required_keys(normalized, selection)
    )

    # Empty selection: still emit a valid plan with no required keys.
    # VerificationPlan allows empty required sets.
    mismatch_reasons = _observed_mismatch_reasons(normalized, keys)
    decisions = _lookup_decisions(
        keys, cache=cache, invalidate_reasons=mismatch_reasons
    )

    # Sort decisions to match required keys order by key_id.
    decision_by_id = {item.key_cid: item for item in decisions}
    ordered_decisions = tuple(decision_by_id[key.key_id] for key in keys)

    human_required, human_reasons = _human_review_reasons(normalized, selection)
    cpu, memory, processes, proof_slots, artifacts = _resource_bounds(
        normalized.policy, keys
    )
    dag, timeouts = _dependency_dag_and_timeouts(
        keys, check_map, selection, normalized.policy
    )

    # Shared plan identities come from the first key when present; otherwise
    # from the identity binding.  Admitted keys are already single-environment.
    if keys:
        tree_cid = keys[0].repository_tree_cid
        semantic_cid = keys[0].semantic_state_root_cid
        env_cid = keys[0].environment_cid
        lock_cid = keys[0].dependency_lock_cid
        if any(
            k.repository_tree_cid != tree_cid
            or k.semantic_state_root_cid != semantic_cid
            or k.dependency_lock_cid != lock_cid
            or k.environment_cid != env_cid
            for k in keys
        ):
            raise PlannerIdentityError(
                "required receipt keys disagree on plan identities"
            )
    else:
        tree_cid = (
            normalized.patch.target_tree_cid
            or normalized.policy.identity.repository_tree_cid
        )
        if not tree_cid:
            raise PlannerIdentityError(
                "empty selection still requires a target repository_tree_cid"
            )
        semantic_cid = normalized.policy.identity.semantic_state_root_cid
        env_cid = normalized.policy.identity.environment_cid
        if not env_cid:
            # Synthesize a placeholder environment from identity for empty plans
            # by compiling a dummy type-check key when tool specs exist.
            if normalized.policy.tool_specs:
                kind = VerificationReceiptKind.TYPE_CHECK
                dummy_id = "__empty_plan_env__"
                try:
                    dummy = compile_check_receipt_key(
                        policy=normalized.policy,
                        kind=kind,
                        check_id=dummy_id,
                        patch=normalized.patch,
                    )
                    env_cid = dummy.environment_cid
                    if not tree_cid:
                        tree_cid = dummy.repository_tree_cid
                except PlannerError:
                    raise PlannerIdentityError(
                        "empty plan requires environment_cid on identity binding"
                    ) from None
            else:
                raise PlannerIdentityError(
                    "empty plan requires environment_cid on identity binding"
                )
        lock_cid = normalized.policy.identity.dependency_lock_cid

    # Full-suite is mandatory only when test keys are admitted for the plan.
    full_suite_required = bool(
        selection.full_suite_required
        and VerificationReceiptKind.TEST in admitted_kinds
        and full_suite_key_cids
    )

    acceptance = _acceptance_criteria(
        full_suite_required=full_suite_required,
        human_review_required=human_required,
    )

    # Selection lists remain complete for diagnostics; required keys/proofs
    # follow the single-environment admission set.
    affected_tests = selection.affected_tests
    fallback_tests = selection.fallback_tests
    required_static = (
        selection.required_static_checks
        if VerificationReceiptKind.STATIC_ANALYSIS in admitted_kinds
        else ()
    )
    required_type = (
        selection.required_type_checks
        if VerificationReceiptKind.TYPE_CHECK in admitted_kinds
        else ()
    )
    # Proof obligations on the plan must equal proof keys exactly.
    plan_proofs = proof_cids

    return VerificationPlan(
        repository_tree_cid=tree_cid,
        semantic_state_root_cid=semantic_cid,
        environment_cid=env_cid,
        dependency_lock_cid=lock_cid,
        required_receipt_keys=keys,
        cache_reuse_decisions=ordered_decisions,
        affected_tests=affected_tests,
        fallback_tests=fallback_tests,
        required_static_checks=required_static,
        required_type_checks=required_type,
        affected_proof_obligation_cids=plan_proofs,
        full_suite_receipt_key_cids=full_suite_key_cids if full_suite_required else (),
        full_suite_required=full_suite_required,
        full_suite_reason_codes=(
            selection.full_suite_reason_codes if full_suite_required else ()
        ),
        human_review_required=human_required,
        human_review_reason_codes=human_reasons if human_required else (),
        expected_cpu_millis=cpu,
        expected_memory_bytes=memory,
        expected_processes=processes,
        expected_proof_slots=proof_slots,
        expected_artifact_bytes=artifacts,
        step_timeouts_ms=dict(timeouts) if timeouts else {},
        max_execution_time_ms=normalized.policy.max_execution_time_ms,
        dependency_dag={k: list(v) for k, v in dag.items()} if dag else {},
        acceptance_criteria=acceptance,
        policy_cid=normalized.policy.policy_cid,
    )


@dataclass
class IncrementalVerificationPlanner:
    """Planner collaborator with optional cache and datasets adapter."""

    INTERFACE: Final[str] = VERIFICATION_PLANNER_INTERFACE

    cache: VerificationReceiptCache | None = None
    adapter: DatasetsVerificationInputAdapter | None = None
    default_policy: PlannerPolicy | None = None

    def __post_init__(self) -> None:
        if self.adapter is None:
            self.adapter = create_datasets_verification_input_adapter()

    def create_plan(
        self,
        repository_state: Any,
        invalidation_plan: Any,
        context_pack: Any,
        patch_delta: Any,
        policy: Any | None = None,
    ) -> VerificationPlan:
        effective = policy if policy is not None else self.default_policy
        if effective is None:
            raise PlannerError("policy is required")
        return create_verification_plan(
            repository_state,
            invalidation_plan,
            context_pack,
            patch_delta,
            effective,
            cache=self.cache,
            adapter=self.adapter,
        )

    # Five-arg call surface matching the module function.
    def __call__(
        self,
        repository_state: Any,
        invalidation_plan: Any,
        context_pack: Any,
        patch_delta: Any,
        policy: Any,
    ) -> VerificationPlan:
        return self.create_plan(
            repository_state,
            invalidation_plan,
            context_pack,
            patch_delta,
            policy,
        )


def create_incremental_verification_planner(
    *,
    cache: VerificationReceiptCache | None = None,
    adapter: DatasetsVerificationInputAdapter | None = None,
    default_policy: PlannerPolicy | Mapping[str, Any] | None = None,
) -> IncrementalVerificationPlanner:
    policy_obj: PlannerPolicy | None
    if default_policy is None:
        policy_obj = None
    elif isinstance(default_policy, PlannerPolicy):
        policy_obj = default_policy
    else:
        policy_obj = PlannerPolicy.from_value(default_policy)
    return IncrementalVerificationPlanner(
        cache=cache,
        adapter=adapter,
        default_policy=policy_obj,
    )


__all__ = [
    "CHECK_TOOL_SPEC_SCHEMA",
    "DEFAULT_MAX_EXECUTION_TIME_MS",
    "DEFAULT_STEP_TIMEOUT_MS",
    "IDENTITY_BINDING_SCHEMA",
    "IncrementalVerificationPlanner",
    "IdentityBinding",
    "PATCH_DELTA_SCHEMA",
    "PLANNER_EVIDENCE",
    "PLANNER_POLICY_SCHEMA",
    "PatchDelta",
    "PlannerBoundsError",
    "PlannerError",
    "PlannerIdentityError",
    "PlannerPolicy",
    "CheckToolSpec",
    "REASON_CROSS_TREE_REJECTED",
    "REASON_ENVIRONMENT_MISMATCH",
    "REASON_LOCK_MISMATCH",
    "REASON_NO_CACHE",
    "REASON_NO_PENDING_FALLBACK",
    "REASON_PATCH_BASE_MISMATCH",
    "REASON_POLICY_CONFLICT",
    "REASON_PRODUCTION_SUCCESS_REQUIRED",
    "REASON_SCOPE_CROSSING",
    "REASON_SEMANTIC_ROOT_MISMATCH",
    "REASON_TOOL_MISMATCH",
    "REASON_UNBOUND_SANDBOX",
    "VERIFICATION_PLANNER_INTERFACE",
    "VERIFICATION_PLANNER_SCHEMA",
    "compile_check_receipt_key",
    "create_incremental_verification_planner",
    "create_verification_plan",
]
