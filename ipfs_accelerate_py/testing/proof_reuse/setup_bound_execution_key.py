"""Setup-bound TestExecutionKeyV2 assembly (PCTDD-027).

Accelerate-owned pytest integration that assembles the exact datasets-owned
``TestExecutionKeyV2`` after current setup and before call.  Datasets owns
the V2 schema; this module only binds that schema in the post-setup window.

Rules:

* Assembly is authoritative only after current setup.  Collection, pre-setup,
  call, and teardown cannot mint an authoritative V2 key.
* The datasets ``TestExecutionKeyV2`` contract never self-assembles
  (``assembled_after_setup`` remains False on the key).  The accelerate
  assembly record is the only place that records post-setup assembly.
* Opaque, incomplete, unknown, unreviewed, or unbound identities force
  normal full execution.  Assembly never authorizes pytest skip and never
  admits production.
* Guarded post-setup reuse, composite phase receipts, signed runner
  attestations, production ZK, and direct-execution profiles remain typed
  unavailable.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

# This module is accelerate integration, not a pytest test module.
__test__ = False

SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE: Final = (
    "SetupBoundExecutionKeyAssembly@1"
)
SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_RESULT_INTERFACE: Final = (
    "SetupBoundExecutionKeyAssemblyResult@1"
)
SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/setup-bound-execution-key-assembly@1"
)
SETUP_BOUND_EXECUTION_KEY_POLICY_INTERFACE: Final = (
    "SetupBoundExecutionKeyAssemblyPolicy@1"
)
CLAIM_CLASS: Final = "IntegrityCommitment"
AUTHORITATIVE_AFTER: Final = "current_setup"
ASSEMBLY_WINDOW: Final = "after_setup_before_call"
AUTHORITATIVE_LIFECYCLE_PHASE: Final = "post_setup"
SCHEMA_AUTHORITY: Final = (
    "ipfs_datasets_py.logic.zkp.pctdd.test_execution_key_v2"
)
ASSEMBLY_AUTHORITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.setup_bound_execution_key"
)
ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_setup_bound_assembly"
)
ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_setup_bound_execution_key_v2"
)
ITEM_SETUP_BOUND_PHASE_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_setup_bound_phase"
)
# Well-known predecessor item pins.  Duplicated by name so this module stays
# import-cold and does not load collection-time identity assemblers.
ITEM_LOCATOR_PIN_ATTRIBUTE: Final = "_ipfs_proof_reuse_locator"
ITEM_PREDECESSOR_EXECUTION_KEY_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_execution_key"
)
MAX_TEXT_CHARS: Final = 4_096
MAX_NAME_CHARS: Final = 256
MAX_SEQUENCE_ITEMS: Final = 64
MAX_EXTRA_KEYS: Final = 32
_DIGEST_PREFIX: Final = "sha256:"

LIFECYCLE_PHASES: Final[tuple[str, ...]] = (
    "collection",
    "pre_setup",
    "setup",
    "post_setup",
    "call",
    "teardown",
    "setup_failed",
)
FORBIDDEN_ASSEMBLY_PHASES: Final[frozenset[str]] = frozenset(
    {
        "collection",
        "pre_setup",
        "setup",
        "call",
        "teardown",
        "setup_failed",
    }
)

ASSEMBLY_ESTABLISHES: Final = (
    "exact V2 key is assembled after setup and before call with normal "
    "execution fallback"
)
ASSEMBLY_DOES_NOT: Final = (
    "execution or semantics; skip; collection-time fixture-instance "
    "authority; composite phase receipts; current-root publication; "
    "production ZK; guarded post-setup reuse; signed admission"
)

FORBIDDEN_COMMIT_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "raw_value",
        "refresh_token",
        "secret",
        "secret_bytes",
        "session_token",
        "value_bytes",
        "witness",
    }
)
_PRIVATE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "private",
    "secret",
    "password",
    "api_key",
    "witness",
    "credential",
    "token",
)
_TYPED_UNAVAILABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "collection_time_fixture_instance_key",
        "fixture_instance_authoritative_only_after_current_setup",
        "exact fixture-instance identity becomes safely authoritative only "
        "after current setup; collection and pre-setup cannot backdate an "
        "instance key",
    ),
    (
        "guarded_post_setup_reuse",
        "guarded_post_setup_reuse_not_implemented",
        "setup-bound assembly attaches the exact V2 key only; guarded "
        "post-setup pre-call reuse remains a later accelerate task",
    ),
    (
        "composite_phase_receipt",
        "composite_phase_receipt_not_assembled_by_execution_key",
        "composite phase receipts are outside setup-bound assembly and remain "
        "typed unavailable without changing TestPassStatementV1",
    ),
    (
        "signed_runner_attestation",
        "signed_runner_attestation_not_verified",
        "setup-bound assembly binds V2 trust pins only; signature "
        "verification remains a later accelerate attestation task",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; setup-bound "
        "assembly cannot admit simulated, structural, or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by setup-bound "
        "execution-key assembly",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade setup-bound V2 integrity commitments",
    ),
)


class SetupBoundExecutionKeyAssemblyError(ValueError):
    """Raised when setup-bound V2 assembly is unsafe."""

    __test__ = False


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in FORBIDDEN_COMMIT_FIELD_MARKERS:
        return True
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise SetupBoundExecutionKeyAssemblyError(
            "floating-point values are not JSON-safe for setup-bound assembly"
        )
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key)
            if _is_private_key(name):
                raise SetupBoundExecutionKeyAssemblyError(
                    f"assembly rejects private material key {name!r}"
                )
            ready[name] = _json_ready(item)
        return ready
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise SetupBoundExecutionKeyAssemblyError(
            "assembly rejects secret or raw bytes"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise SetupBoundExecutionKeyAssemblyError(
        f"value of type {type(value).__name__} is not JSON-serializable"
    )


def canonical_public_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for a privacy-safe public payload."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def public_digest(value: Any) -> str:
    """Return ``sha256:<hex>`` of canonical public bytes."""

    return _DIGEST_PREFIX + hashlib.sha256(canonical_public_bytes(value)).hexdigest()


ASSEMBLY_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": SETUP_BOUND_EXECUTION_KEY_POLICY_INTERFACE,
        "assembly_interface": SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE,
        "authoritative_after": AUTHORITATIVE_AFTER,
        "assembly_window": ASSEMBLY_WINDOW,
        "authoritative_lifecycle_phase": AUTHORITATIVE_LIFECYCLE_PHASE,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "normal_execution_fallback": True,
        "schema_authority": SCHEMA_AUTHORITY,
    }
)
DEFAULT_POLICY_CID: Final = public_digest(dict(ASSEMBLY_POLICY))


def _load_datasets_v2() -> Any | None:
    try:
        from ipfs_datasets_py.logic.zkp.pctdd import test_execution_key_v2 as contracts
    except Exception:
        return None
    return contracts


def datasets_contracts_available() -> bool:
    """Return whether datasets-owned TestExecutionKeyV2 codecs import."""

    return _load_datasets_v2() is not None


def record_typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    """Record a typed unavailable case without changing claim meaning."""

    record = {
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if record["production_admitted"] or record["self_approved"] or not record["claim_unchanged"]:
        raise SetupBoundExecutionKeyAssemblyError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-027 typed unavailable capabilities."""

    records = [
        record_typed_unavailable(
            capability=capability,
            reason_code=reason_code,
            message=message,
        )
        for capability, reason_code, message in _TYPED_UNAVAILABLE
    ]
    if not datasets_contracts_available():
        records.append(
            record_typed_unavailable(
                capability="datasets_test_execution_key_v2",
                reason_code="datasets_test_execution_key_v2_unresolved",
                message=(
                    "datasets TestExecutionKeyV2 codecs did not import in this "
                    "sealed environment; assembly forces full execution and "
                    "does not mint a V2 key"
                ),
            )
        )
    return tuple(records)


def authority_descriptor() -> dict[str, Any]:
    """Return the assembly authority without widening others."""

    return {
        "canonical_semantic_and_statement_authority": "ipfs_datasets_py",
        "execution_scheduling_admission_authority": "ipfs_accelerate_py",
        "verified_storage_wal_cas_authority": "ipfs_kit_py",
        "assembly": ASSEMBLY_AUTHORITY,
        "schema_authority": SCHEMA_AUTHORITY,
        "does_not": ASSEMBLY_DOES_NOT,
        "establishes": ASSEMBLY_ESTABLISHES,
        "claim_class": CLAIM_CLASS,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "worker_authored_test_is_sufficient_alone": False,
        "authoritative_after": AUTHORITATIVE_AFTER,
        "assembly_window": ASSEMBLY_WINDOW,
        "authoritative_lifecycle_phase": AUTHORITATIVE_LIFECYCLE_PHASE,
        "normal_execution_fallback": True,
        "assembly_interface": SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE,
        "test_execution_key_v2": "TestExecutionKeyV2",
    }


def classify_lifecycle_phase(value: Any) -> str:
    """Map a caller-supplied phase onto the closed lifecycle vocabulary."""

    if value is None:
        return "collection"
    if not isinstance(value, str):
        raise SetupBoundExecutionKeyAssemblyError(
            "lifecycle_phase must be a string"
        )
    text = value.strip()
    if not text:
        return "collection"
    if text not in LIFECYCLE_PHASES:
        raise SetupBoundExecutionKeyAssemblyError(
            "lifecycle_phase must be one of " + ", ".join(LIFECYCLE_PHASES)
        )
    return text


def is_authoritative_assembly_phase(phase: Any) -> bool:
    """Return whether *phase* is the post-setup, pre-call assembly window."""

    return classify_lifecycle_phase(phase) == AUTHORITATIVE_LIFECYCLE_PHASE


def mark_setup_bound_phase(item: Any, phase: str) -> str:
    """Record the current assembly lifecycle phase on *item*."""

    classified = classify_lifecycle_phase(phase)
    try:
        setattr(item, ITEM_SETUP_BOUND_PHASE_ATTRIBUTE, classified)
    except Exception:
        pass
    return classified


def get_setup_bound_phase(item: Any) -> str:
    """Return the recorded lifecycle phase, defaulting to collection."""

    recorded = getattr(item, ITEM_SETUP_BOUND_PHASE_ATTRIBUTE, None)
    if isinstance(recorded, str) and recorded in LIFECYCLE_PHASES:
        return recorded
    return "collection"


def get_attached_setup_bound_assembly(item: Any) -> Any | None:
    """Return the attached assembly result, if any."""

    existing = getattr(item, ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE, None)
    if isinstance(existing, SetupBoundExecutionKeyAssemblyResult):
        return existing
    return None


def get_attached_setup_bound_execution_key(item: Any) -> Any | None:
    """Return the attached V2 key, if any."""

    existing = getattr(item, ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE, None)
    if existing is not None:
        return existing
    assembly = get_attached_setup_bound_assembly(item)
    if assembly is not None:
        return assembly.execution_key
    return None


def setup_failed_from_hook_outcome(outcome: Any) -> bool:
    """Return whether a pytest hookwrapper outcome recorded a setup exception.

    Never raises.  Unknown outcome shapes are treated as not failed so
    assembly can still run in-window; later completeness checks force RUN.
    """

    if outcome is None:
        return False
    excinfo = getattr(outcome, "excinfo", None)
    if callable(excinfo):
        try:
            excinfo = excinfo()
        except Exception:
            excinfo = None
    if excinfo is not None:
        return True
    exception = getattr(outcome, "exception", None)
    if callable(exception):
        try:
            exception = exception()
        except Exception:
            exception = None
    if exception is not None:
        return True
    get_result = getattr(outcome, "get_result", None)
    if not callable(get_result):
        return False
    try:
        get_result()
    except Exception:
        return True
    return False


def _public_cid(value: Any, names: Sequence[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.strip():
        return value.strip()[:MAX_TEXT_CHARS]
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:MAX_TEXT_CHARS]
    to_dict = getattr(value, "to_dict", None)
    payload: Any = value
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = value
    if isinstance(payload, Mapping):
        for name in names:
            candidate = payload.get(name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:MAX_TEXT_CHARS]
    return ""


def _closure_cid(value: Any) -> str:
    cid = _public_cid(
        value,
        ("closure_cid", "commitment_cid", "execution_key_cid"),
    )
    if cid:
        return cid
    cid = getattr(value, "closure_cid", "")
    if isinstance(cid, str):
        return cid.strip()[:MAX_TEXT_CHARS]
    return ""


def item_assembly_pins(item: Any) -> dict[str, str]:
    """Return public locator/predecessor pins already attached to *item*."""

    if item is None:
        return {"locator_cid": "", "predecessor_execution_key_cid": ""}
    return {
        "locator_cid": _public_cid(
            getattr(item, ITEM_LOCATOR_PIN_ATTRIBUTE, None),
            ("locator_cid", "locator_id", "content_id"),
        ),
        "predecessor_execution_key_cid": _public_cid(
            getattr(item, ITEM_PREDECESSOR_EXECUTION_KEY_ATTRIBUTE, None),
            ("execution_key_cid", "content_id"),
        ),
    }


def _commit_item_funcargs(
    item: Any,
    *,
    fixture_adapter_map: Mapping[str, str] | None = None,
) -> tuple[Any, ...]:
    funcargs = getattr(item, "funcargs", None)
    if not isinstance(funcargs, Mapping) or not funcargs:
        return ()
    try:
        from .dependency_commitment_adapters import (
            commit_fixture_instance,
            commit_opaque_dependency,
        )
    except Exception:
        return ()
    adapters = dict(fixture_adapter_map or {})
    results: list[Any] = []
    for raw_name, value in funcargs.items():
        name = str(raw_name or "")[:MAX_NAME_CHARS]
        if not name:
            continue
        adapter_id = str(adapters.get(name, "") or "")
        try:
            results.append(
                commit_fixture_instance(
                    fixture_name=name,
                    value=value,
                    adapter_id=adapter_id,
                    reviewed=bool(adapter_id),
                )
            )
        except Exception:
            try:
                results.append(
                    commit_opaque_dependency(
                        kind="pytest_fixture",
                        name=name,
                        reason_code="fixture_instance_commit_failed",
                    )
                )
            except Exception:
                continue
    return tuple(results)


def _derive_fixture_identity(
    contracts: Any,
    *,
    fixture: Any,
    fixture_definition_closure: Any,
    fixture_instance_results: Sequence[Any],
    locator_cid: str,
) -> Any:
    if fixture is not None:
        return fixture
    definition_cid = _closure_cid(fixture_definition_closure)
    instance_cid = ""
    reuse_class = "opaque"
    completeness = "unknown"
    reviewed = False
    if fixture_instance_results:
        try:
            from .dependency_commitment_adapters import (
                build_committed_closure,
                requires_full_execution,
            )

            if not requires_full_execution(fixture_instance_results):
                closure = build_committed_closure(
                    fixture_instance_results,
                    locator_cid=locator_cid,
                )
                instance_cid = _closure_cid(closure)
                reuse_class = "pure"
                completeness = "exact" if definition_cid and instance_cid else "incomplete"
                reviewed = True
        except Exception:
            instance_cid = ""
            reuse_class = "opaque"
            completeness = "unknown"
            reviewed = False
    return contracts.build_fixture_identity(
        fixture_definition_closure_cid=definition_cid,
        fixture_instance_closure_cid=instance_cid,
        reuse_class=reuse_class,
        completeness=completeness,
        reviewed=reviewed,
    )


def _build_execution_key(
    contracts: Any,
    *,
    fixture: Any,
    environment: Any,
    toolchain: Any,
    trust: Any,
    completeness_identity: Any,
    locator_cid: str,
    repository_forest_cid: str,
    git_commit_id: str,
    git_tree_id: str,
    gitlink_state_cid: str,
    dirty_overlay_cid: str,
    test_module_cid: str,
    test_class_cid: str,
    test_function_cid: str,
    test_ast_cid: str,
    parameter_source_cid: str,
    predecessor_execution_key_cid: str,
    extra: Mapping[str, Any] | None,
) -> Any:
    return contracts.build_test_execution_key_v2(
        fixture=fixture,
        environment=environment,
        toolchain=toolchain,
        trust=trust,
        completeness_identity=completeness_identity,
        locator_cid=locator_cid,
        repository_forest_cid=repository_forest_cid,
        git_commit_id=git_commit_id,
        git_tree_id=git_tree_id,
        gitlink_state_cid=gitlink_state_cid,
        dirty_overlay_cid=dirty_overlay_cid,
        test_module_cid=test_module_cid,
        test_class_cid=test_class_cid,
        test_function_cid=test_function_cid,
        test_ast_cid=test_ast_cid,
        parameter_source_cid=parameter_source_cid,
        predecessor_execution_key_cid=predecessor_execution_key_cid,
        extra=extra or {},
    )


@dataclass(frozen=True, slots=True)
class SetupBoundExecutionKeyAssemblyResult:
    """Outcome of one post-setup, pre-call V2 assembly attempt.

    ``assembled_after_setup`` is an accelerate record.  The datasets V2 key
    itself never self-assembles.  Not a pytest test class.
    """

    __test__ = False

    execution_key: Any
    lifecycle_phase: str
    assembled_after_setup: bool
    assembled_before_call: bool
    datasets_contracts_available: bool = True
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    claim_unchanged: bool = True
    full_execution_reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        phase = classify_lifecycle_phase(self.lifecycle_phase)
        object.__setattr__(self, "lifecycle_phase", phase)
        if self.may_authorize_skip:
            raise SetupBoundExecutionKeyAssemblyError(
                "setup-bound assembly must not authorize skip"
            )
        if self.production_admitted or self.self_approved or not self.claim_unchanged:
            raise SetupBoundExecutionKeyAssemblyError(
                "setup-bound assembly cannot admit production, self-approve, "
                "or change claims"
            )
        if self.assembled_after_setup and phase != AUTHORITATIVE_LIFECYCLE_PHASE:
            raise SetupBoundExecutionKeyAssemblyError(
                "exact V2 key may be assembled only after setup and before call"
            )
        if self.assembled_before_call and not self.assembled_after_setup:
            raise SetupBoundExecutionKeyAssemblyError(
                "pre-call assembly requires a post-setup V2 key"
            )
        if self.assembled_after_setup and self.execution_key is not None:
            key_assembled = bool(
                getattr(self.execution_key, "assembled_after_setup", False)
            )
            if key_assembled:
                raise SetupBoundExecutionKeyAssemblyError(
                    "datasets TestExecutionKeyV2 must not self-assemble after setup"
                )
        object.__setattr__(
            self, "full_execution_reasons", tuple(self.full_execution_reasons)
        )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def interface(self) -> str:
        return SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_RESULT_INTERFACE

    @property
    def schema(self) -> str:
        return SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_SCHEMA

    @property
    def assembly_interface(self) -> str:
        return SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE

    @property
    def authoritative_after(self) -> str:
        return AUTHORITATIVE_AFTER

    @property
    def assembly_window(self) -> str:
        return ASSEMBLY_WINDOW

    @property
    def normal_execution_fallback(self) -> bool:
        return True

    @property
    def action(self) -> str:
        return "RUN"

    @property
    def requires_full_execution(self) -> bool:
        if not self.assembled_after_setup or self.execution_key is None:
            return True
        if self.full_execution_reasons:
            return True
        return bool(getattr(self.execution_key, "requires_full_execution", True))

    @property
    def execution_disposition(self) -> str:
        if self.requires_full_execution:
            return "full_execution"
        return "committed"

    @property
    def claim_class(self) -> str:
        return CLAIM_CLASS

    def to_dict(self) -> dict[str, Any]:
        key_payload = None
        if self.execution_key is not None and hasattr(self.execution_key, "to_dict"):
            key_payload = self.execution_key.to_dict()
        return {
            "schema": self.schema,
            "interface": self.interface,
            "assembly_interface": self.assembly_interface,
            "lifecycle_phase": self.lifecycle_phase,
            "assembled_after_setup": self.assembled_after_setup,
            "assembled_before_call": self.assembled_before_call,
            "authoritative_after": self.authoritative_after,
            "assembly_window": self.assembly_window,
            "datasets_contracts_available": self.datasets_contracts_available,
            "may_authorize_skip": False,
            "production_admitted": False,
            "self_approved": False,
            "claim_unchanged": True,
            "claim_class": self.claim_class,
            "normal_execution_fallback": True,
            "action": self.action,
            "requires_full_execution": self.requires_full_execution,
            "execution_disposition": self.execution_disposition,
            "full_execution_reasons": list(self.full_execution_reasons),
            "test_execution_key_v2": key_payload,
            "diagnostics": dict(self.diagnostics),
        }


def _fallback_result(
    *,
    phase: str,
    reasons: Sequence[str],
    datasets_available: bool,
    diagnostics: Mapping[str, Any] | None = None,
    execution_key: Any = None,
) -> SetupBoundExecutionKeyAssemblyResult:
    unique_reasons = tuple(dict.fromkeys(str(item) for item in reasons if item))
    return SetupBoundExecutionKeyAssemblyResult(
        execution_key=execution_key,
        lifecycle_phase=phase,
        assembled_after_setup=False,
        assembled_before_call=False,
        datasets_contracts_available=datasets_available,
        full_execution_reasons=unique_reasons or ("normal_execution_fallback",),
        diagnostics=dict(diagnostics or {}),
    )


def assemble_setup_bound_execution_key(
    *,
    lifecycle_phase: str,
    fixture: Any = None,
    environment: Any = None,
    toolchain: Any = None,
    trust: Any = None,
    completeness_identity: Any = None,
    locator_cid: str = "",
    repository_forest_cid: str = "",
    git_commit_id: str = "",
    git_tree_id: str = "",
    gitlink_state_cid: str = "",
    dirty_overlay_cid: str = "",
    test_module_cid: str = "",
    test_class_cid: str = "",
    test_function_cid: str = "",
    test_ast_cid: str = "",
    parameter_source_cid: str = "",
    predecessor_execution_key_cid: str = "",
    fixture_definition_closure: Any = None,
    fixture_instance_results: Sequence[Any] = (),
    extra: Mapping[str, Any] | None = None,
    item: Any = None,
    fixture_adapter_map: Mapping[str, str] | None = None,
) -> SetupBoundExecutionKeyAssemblyResult:
    """Assemble the exact V2 key after setup and before call.

    Wrong lifecycle phases, missing datasets codecs, and opaque or incomplete
    identities force normal execution.  Never authorizes skip.
    """

    phase = classify_lifecycle_phase(lifecycle_phase)
    contracts = _load_datasets_v2()
    if contracts is None:
        return _fallback_result(
            phase=phase,
            reasons=("datasets_test_execution_key_v2_unresolved",),
            datasets_available=False,
            diagnostics={"reason": "datasets_contracts_unavailable"},
        )
    if phase != AUTHORITATIVE_LIFECYCLE_PHASE:
        reason = f"lifecycle_phase_{phase}_not_post_setup"
        if phase in FORBIDDEN_ASSEMBLY_PHASES:
            reason = f"lifecycle_phase_{phase}_outside_assembly_window"
        return _fallback_result(
            phase=phase,
            reasons=(reason, "normal_execution_fallback"),
            datasets_available=True,
            diagnostics={
                "assembly_window": ASSEMBLY_WINDOW,
                "authoritative_lifecycle_phase": AUTHORITATIVE_LIFECYCLE_PHASE,
            },
        )

    derived_results = tuple(fixture_instance_results)
    if not derived_results and fixture is None and item is not None:
        derived_results = _commit_item_funcargs(
            item,
            fixture_adapter_map=fixture_adapter_map,
        )
    definition_closure = fixture_definition_closure
    if definition_closure is None and item is not None:
        from .fixture_definition_extraction import (
            ITEM_FIXTURE_DEFINITION_CLOSURE_ATTRIBUTE,
        )

        definition_closure = getattr(
            item,
            ITEM_FIXTURE_DEFINITION_CLOSURE_ATTRIBUTE,
            None,
        )
    nodeid = ""
    pins = item_assembly_pins(item)
    if item is not None:
        nodeid = str(getattr(item, "nodeid", "") or "")[:MAX_TEXT_CHARS]
    if not locator_cid:
        locator_cid = pins["locator_cid"]
    if not predecessor_execution_key_cid:
        predecessor_execution_key_cid = pins["predecessor_execution_key_cid"]
    locator = locator_cid or (
        f"cid:locator:{public_digest({'nodeid': nodeid})[7:23]}"
        if nodeid
        else ""
    )

    try:
        fixture_identity = _derive_fixture_identity(
            contracts,
            fixture=fixture,
            fixture_definition_closure=definition_closure,
            fixture_instance_results=derived_results,
            locator_cid=locator,
        )
        key = _build_execution_key(
            contracts,
            fixture=fixture_identity,
            environment=environment,
            toolchain=toolchain,
            trust=trust,
            completeness_identity=completeness_identity,
            locator_cid=locator,
            repository_forest_cid=repository_forest_cid,
            git_commit_id=git_commit_id,
            git_tree_id=git_tree_id,
            gitlink_state_cid=gitlink_state_cid,
            dirty_overlay_cid=dirty_overlay_cid,
            test_module_cid=test_module_cid,
            test_class_cid=test_class_cid,
            test_function_cid=test_function_cid,
            test_ast_cid=test_ast_cid,
            parameter_source_cid=parameter_source_cid,
            predecessor_execution_key_cid=predecessor_execution_key_cid,
            extra=extra,
        )
    except Exception as exc:
        return _fallback_result(
            phase=phase,
            reasons=("execution_key_v2_build_failed", "normal_execution_fallback"),
            datasets_available=True,
            diagnostics={"exception_type": type(exc).__name__},
        )

    reasons: list[str] = []
    if derived_results:
        if any(not bool(getattr(item_result, "committed", False)) for item_result in derived_results):
            reasons.append("opaque_or_incomplete_fixture_instance")
    key_reasons = tuple(getattr(key, "full_execution_reasons", ()) or ())
    reasons.extend(key_reasons)
    if getattr(key, "assembled_after_setup", False):
        return _fallback_result(
            phase=phase,
            reasons=("self_assembled_after_setup", "normal_execution_fallback"),
            datasets_available=True,
            diagnostics={"reason": "datasets_key_must_not_self_assemble"},
        )
    unique_reasons = tuple(dict.fromkeys(reasons))
    return SetupBoundExecutionKeyAssemblyResult(
        execution_key=key,
        lifecycle_phase=phase,
        assembled_after_setup=True,
        assembled_before_call=True,
        datasets_contracts_available=True,
        full_execution_reasons=unique_reasons,
        diagnostics={"nodeid": nodeid, "assembly_window": ASSEMBLY_WINDOW},
    )


def attach_setup_bound_execution_key(
    item: Any,
    result: SetupBoundExecutionKeyAssemblyResult,
) -> SetupBoundExecutionKeyAssemblyResult:
    """Attach an assembly result without authorizing skip."""

    if not isinstance(result, SetupBoundExecutionKeyAssemblyResult):
        raise SetupBoundExecutionKeyAssemblyError(
            "attach requires a SetupBoundExecutionKeyAssemblyResult"
        )
    try:
        setattr(item, ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE, result)
        if result.execution_key is not None and result.assembled_after_setup:
            setattr(
                item,
                ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE,
                result.execution_key,
            )
        mark_setup_bound_phase(item, result.lifecycle_phase)
    except SetupBoundExecutionKeyAssemblyError:
        raise
    except Exception as exc:
        return _fallback_result(
            phase=result.lifecycle_phase,
            reasons=("attachment_failed", "normal_execution_fallback"),
            datasets_available=result.datasets_contracts_available,
            diagnostics={"exception_type": type(exc).__name__},
        )
    return result


def assemble_and_attach_from_item(
    item: Any,
    *,
    lifecycle_phase: str,
    attach: bool = True,
    fixture_adapter_map: Mapping[str, str] | None = None,
    **key_fields: Any,
) -> SetupBoundExecutionKeyAssemblyResult:
    """Assemble from a pytest item and optionally attach the result."""

    result = assemble_setup_bound_execution_key(
        lifecycle_phase=lifecycle_phase,
        item=item,
        fixture_adapter_map=fixture_adapter_map,
        **key_fields,
    )
    if attach:
        return attach_setup_bound_execution_key(item, result)
    return result


def ensure_setup_bound_execution_key_before_call(
    item: Any,
) -> SetupBoundExecutionKeyAssemblyResult:
    """Return the post-setup key or fall back to normal execution before call.

    Does not mint an authoritative key during call.  Never skips.
    """

    existing = get_attached_setup_bound_assembly(item)
    if (
        isinstance(existing, SetupBoundExecutionKeyAssemblyResult)
        and existing.assembled_after_setup
        and existing.assembled_before_call
    ):
        return existing
    phase = get_setup_bound_phase(item)
    if phase == AUTHORITATIVE_LIFECYCLE_PHASE and existing is None:
        # Setup completed but assembly was missed; attempt once in-window.
        return assemble_and_attach_from_item(item, lifecycle_phase=phase)
    fallback = _fallback_result(
        phase="call" if phase == AUTHORITATIVE_LIFECYCLE_PHASE else phase,
        reasons=(
            "setup_bound_execution_key_missing_before_call",
            "normal_execution_fallback",
        ),
        datasets_available=datasets_contracts_available(),
        diagnostics={"reason": "assembly_required_after_setup_before_call"},
        execution_key=existing.execution_key if existing is not None else None,
    )
    return attach_setup_bound_execution_key(item, fallback)


def before_runtest_setup(item: Any) -> str:
    """Mark the pre-setup frontier.  Must not assemble a V2 key."""

    return mark_setup_bound_phase(item, "pre_setup")


def after_runtest_setup(
    item: Any,
    *,
    setup_failed: bool = False,
    **key_fields: Any,
) -> SetupBoundExecutionKeyAssemblyResult:
    """Assemble the exact V2 key after current setup succeeds."""

    if setup_failed:
        mark_setup_bound_phase(item, "setup_failed")
        return attach_setup_bound_execution_key(
            item,
            _fallback_result(
                phase="setup_failed",
                reasons=("setup_failed", "normal_execution_fallback"),
                datasets_available=datasets_contracts_available(),
                diagnostics={"reason": "setup_failed"},
            ),
        )
    mark_setup_bound_phase(item, AUTHORITATIVE_LIFECYCLE_PHASE)
    return assemble_and_attach_from_item(
        item,
        lifecycle_phase=AUTHORITATIVE_LIFECYCLE_PHASE,
        **key_fields,
    )


def before_runtest_call(item: Any) -> SetupBoundExecutionKeyAssemblyResult:
    """Ensure a post-setup V2 key exists, then enter call with RUN fallback."""

    result = ensure_setup_bound_execution_key_before_call(item)
    mark_setup_bound_phase(item, "call")
    return result


@dataclass(frozen=True, slots=True)
class SetupBoundLifecycleRecord:
    """Ordered setup -> assemble -> call evidence.  Not a pytest test class."""

    __test__ = False

    events: tuple[str, ...]
    assembly: SetupBoundExecutionKeyAssemblyResult
    before_call: SetupBoundExecutionKeyAssemblyResult

    @property
    def assembled_after_setup_before_call(self) -> bool:
        return self.events == ("setup", "assembled", "call")


def run_setup_bound_assembly_lifecycle(
    item: Any,
    **key_fields: Any,
) -> SetupBoundLifecycleRecord:
    """Prove the exact V2 key is assembled after setup and before call."""

    events: list[str] = []
    before_runtest_setup(item)
    events.append("setup")
    assembly = after_runtest_setup(item, setup_failed=False, **key_fields)
    events.append("assembled")
    before_call = before_runtest_call(item)
    events.append("call")
    return SetupBoundLifecycleRecord(
        events=tuple(events),
        assembly=assembly,
        before_call=before_call,
    )


__all__ = [
    "ASSEMBLY_DOES_NOT",
    "ASSEMBLY_ESTABLISHES",
    "ASSEMBLY_POLICY",
    "ASSEMBLY_WINDOW",
    "AUTHORITATIVE_AFTER",
    "AUTHORITATIVE_LIFECYCLE_PHASE",
    "CLAIM_CLASS",
    "DEFAULT_POLICY_CID",
    "ITEM_LOCATOR_PIN_ATTRIBUTE",
    "ITEM_PREDECESSOR_EXECUTION_KEY_ATTRIBUTE",
    "ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE",
    "ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE",
    "ITEM_SETUP_BOUND_PHASE_ATTRIBUTE",
    "LIFECYCLE_PHASES",
    "SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE",
    "SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_RESULT_INTERFACE",
    "SetupBoundExecutionKeyAssemblyError",
    "SetupBoundExecutionKeyAssemblyResult",
    "SetupBoundLifecycleRecord",
    "after_runtest_setup",
    "assemble_and_attach_from_item",
    "assemble_setup_bound_execution_key",
    "attach_setup_bound_execution_key",
    "authority_descriptor",
    "before_runtest_call",
    "before_runtest_setup",
    "classify_lifecycle_phase",
    "datasets_contracts_available",
    "ensure_setup_bound_execution_key_before_call",
    "get_attached_setup_bound_assembly",
    "get_attached_setup_bound_execution_key",
    "get_setup_bound_phase",
    "is_authoritative_assembly_phase",
    "item_assembly_pins",
    "mark_setup_bound_phase",
    "public_digest",
    "record_typed_unavailable",
    "run_setup_bound_assembly_lifecycle",
    "setup_failed_from_hook_outcome",
    "typed_unavailable_records",
]
