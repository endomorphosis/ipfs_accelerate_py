"""Fail-closed policy boundary for the deterministic doctor (LPR-029).

:class:`DeterministicDoctorPolicy` is the operational gate between doctor
contracts and any later diagnostic, plan, repair, or rollback stage.  It:

* defaults to report-only with inspect/explain/plan treated as read-only;
* keeps LLM, remote model-provider, remote embedding, network, and target-code
  import flags hard-off in deterministic mode;
* refuses semantic authority on KG/vector/embedding/Tactician/Hammer candidates
  and proof-cache metadata;
* rejects repair without an existing admitted plan, writer lease, checkpoint,
  and rollback strategy;
* protects doctor/proof/identity/transaction trusted-base paths; and
* classifies public/native/stateful/cross-root/new-dependency work as
  approval-required.

This module never grants mutation authority itself.  It only admits or
rejects a requested operation against closed policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..analysis.deterministic_doctor_contracts import (
    ALLOWED_DOCTOR_MODES,
    ALL_APPROVAL_CLASSES,
    DEFAULT_DOCTOR_MODE,
    DETERMINISTIC_DOCTOR_POLICY_SCHEMA,
    DETERMINISTIC_DOCTOR_VERSION,
    DOCTOR_TCB_PATH_MARKERS,
    FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS,
    READ_ONLY_OPERATIONS,
    DeterministicDoctorAuthorityError,
    DeterministicDoctorError,
    DeterministicDoctorPlan,
    DeterministicDoctorRunReceipt,
    DeterministicDoctorSafetyError,
    DoctorApprovalClass,
    DoctorMode,
    DoctorOperation,
    DoctorPlanDisposition,
    DoctorRejectionReason,
    DoctorRepairDisposition,
    DoctorResourceBounds,
    ForgedDeterministicDoctorIdentityError,
    is_doctor_tcb_path,
    operation_is_read_only,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)


DETERMINISTIC_DOCTOR_POLICY_INTERFACE: Final[str] = "DeterministicDoctorPolicy@1"
POLICY_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/policy-decision@1"
)
MAX_POLICY_RECORD_BYTES: Final[int] = 262_144

DEFAULT_APPROVAL_REQUIRED_CLASSES: Final[tuple[str, ...]] = tuple(
    item.value for item in ALL_APPROVAL_CLASSES
)

DEFAULT_LIMITS: Final[dict[str, int]] = {
    "max_findings": 256,
    "max_candidates_per_finding": 64,
    "max_graph_nodes_per_query": 2048,
    "max_proof_routes_per_goal": 32,
    "max_operators_per_finding": 32,
    "max_plan_steps": 256,
    "max_fixed_point_iterations": 8,
    "max_changed_files": 128,
    "max_changed_bytes": 1_048_576,
    "max_processes": 8,
    "max_wall_time_seconds": 3600,
    "max_cpu_time_seconds": 1800,
    "max_memory_bytes": 4_294_967_296,
}

# Safety flags that must remain false under every deterministic-doctor policy.
# ``enabled`` and ``narrow_autonomous_mutation_enabled`` default false and may
# be elevated explicitly; they are not listed here.
HARD_FALSE_FLAGS: Final[tuple[str, ...]] = (
    "llm_router_enabled",
    "llm_invocations_allowed",
    "remote_model_provider_calls_allowed",
    "remote_embeddings_allowed",
    "network_access_allowed",
    "target_code_import_allowed",
    *FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS,
)

# Operator elevation flags (default false; may be set true only explicitly).
ELEVATION_FLAGS: Final[tuple[str, ...]] = (
    "enabled",
    "narrow_autonomous_mutation_enabled",
)

# Gates that must remain true under deterministic mode.
HARD_TRUE_GATES: Final[tuple[str, ...]] = (
    "explicit_repair_operation_required",
    "exact_evidence_snapshot_required",
    "clean_rebuild_identity_equivalence_required",
    "canonical_cid_preimage_validation_required",
    "proof_cache_binding_revalidation_required",
    "native_kernel_reconstruction_required",
    "independent_countermodel_validation_required",
    "complete_impact_closure_required",
    "one_disposition_per_resolved_consumer",
    "unique_target_value_placement_operator_required",
    "closed_operator_registry_required",
    "isolated_candidate_worktree_required",
    "enforced_sandbox_required_for_target_execution",
    "writer_lease_and_checkpoint_required",
    "atomic_scc_transaction_required",
    "post_edit_reindex_and_cache_invalidation_required",
    "logic_and_program_fixed_point_required",
    "compensating_rollback_required",
)

# Path markers for public API / schema surfaces (approval-required).
PUBLIC_API_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/api/",
    "/schema/",
    "/schemas/",
    "/openapi",
    "/proto/",
    "/public/",
    "_schema.py",
    "_api.py",
    "py.typed",
)

NATIVE_PATH_MARKERS: Final[tuple[str, ...]] = (
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".so",
    ".dylib",
    ".dll",
    "/ffi/",
    "/native/",
    "_ffi.py",
    "ctypes",
    "cffi",
)

STATEFUL_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/state/",
    "/migration",
    "/migrations/",
    "alembic",
    "lifecycle",
    "transaction",
    "checkpoint",
    "lease",
)


class DeterministicDoctorPolicyError(ContractValidationError):
    """Raised when a deterministic-doctor policy or decision is malformed."""


class PolicyVerdict(str, Enum):
    """Closed outcomes of a policy evaluation."""

    ALLOW = "allow"
    ABSTAIN = "abstain"
    REJECT = "reject"
    APPROVAL_REQUIRED = "approval_required"


def _text(value: Any, field_name: str, *, required: bool = False) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise DeterministicDoctorPolicyError(f"{field_name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise DeterministicDoctorPolicyError(f"{field_name} is required")
    return text


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DeterministicDoctorPolicyError(f"{field_name} must be a boolean")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeterministicDoctorPolicyError(
            f"{field_name} must be a positive integer"
        )
    return value


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise DeterministicDoctorPolicyError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _strings(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise DeterministicDoctorPolicyError(
            f"{field_name} must be a sequence of strings"
        )
    else:
        raw = values
    if preserve_order:
        out_list: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = _text(item, field_name, required=True)
            if text not in seen:
                seen.add(text)
                out_list.append(text)
        out = tuple(out_list)
    else:
        out = tuple(sorted({_text(item, field_name, required=True) for item in raw}))
    if required and not out:
        raise DeterministicDoctorPolicyError(f"{field_name} must not be empty")
    return out


def classify_path_approval_classes(path: str) -> frozenset[DoctorApprovalClass]:
    """Classify a repository path into approval-required classes."""

    normalized = PurePosixPath(str(path or "").replace("\\", "/")).as_posix().lower()
    classes: set[DoctorApprovalClass] = set()
    if not normalized:
        return frozenset()
    if is_doctor_tcb_path(path):
        classes.add(DoctorApprovalClass.DOCTOR_TRUSTED_COMPUTING_BASE)
    if any(marker in normalized for marker in PUBLIC_API_PATH_MARKERS):
        classes.add(DoctorApprovalClass.PUBLIC_API_OR_SCHEMA)
    if any(normalized.endswith(marker) or marker in normalized for marker in NATIVE_PATH_MARKERS):
        classes.add(DoctorApprovalClass.NATIVE_OR_FFI)
    if any(marker in normalized for marker in STATEFUL_PATH_MARKERS):
        classes.add(DoctorApprovalClass.STATEFUL_BEHAVIOR)
    if "generated" in normalized or "/vendor/" in normalized or normalized.endswith(".gen.py"):
        classes.add(DoctorApprovalClass.DYNAMIC_OR_GENERATED_CODE)
    return frozenset(classes)


def classify_change_approval_classes(
    *,
    paths: Sequence[str] = (),
    cross_repository: bool = False,
    new_external_dependency: bool = False,
    unsupported_memory_or_lifetime_claim: bool = False,
    stateful_behavior: bool = False,
    public_api_or_schema: bool = False,
    dynamic_or_generated: bool = False,
    native_or_ffi: bool = False,
) -> frozenset[DoctorApprovalClass]:
    """Union path-derived and explicit change-class approval requirements."""

    classes: set[DoctorApprovalClass] = set()
    for path in paths:
        classes.update(classify_path_approval_classes(path))
    if cross_repository:
        classes.add(DoctorApprovalClass.CROSS_REPOSITORY_EDIT)
    if new_external_dependency:
        classes.add(DoctorApprovalClass.NEW_EXTERNAL_DEPENDENCY)
    if unsupported_memory_or_lifetime_claim:
        classes.add(DoctorApprovalClass.UNSUPPORTED_MEMORY_OR_LIFETIME_CLAIM)
    if stateful_behavior:
        classes.add(DoctorApprovalClass.STATEFUL_BEHAVIOR)
    if public_api_or_schema:
        classes.add(DoctorApprovalClass.PUBLIC_API_OR_SCHEMA)
    if dynamic_or_generated:
        classes.add(DoctorApprovalClass.DYNAMIC_OR_GENERATED_CODE)
    if native_or_ffi:
        classes.add(DoctorApprovalClass.NATIVE_OR_FFI)
    return frozenset(classes)


@dataclass(frozen=True)
class DeterministicDoctorPolicy(CanonicalContract):
    """Closed deterministic-doctor policy (scheduler schema @1).

    Default mode is report-only.  All model / network / import / semantic-
    authority flags stay false.  Repair requires explicit operation elevation
    plus admitted plan/lease/checkpoint/rollback evidence.
    """

    SCHEMA: ClassVar[str] = DETERMINISTIC_DOCTOR_POLICY_SCHEMA

    policy_id: str = "policy:deterministic-doctor:default"
    enabled: bool = False
    default_mode: DoctorMode = DoctorMode.REPORT_ONLY
    allowed_modes: tuple[DoctorMode, ...] = ALLOWED_DOCTOR_MODES
    narrow_autonomous_mutation_enabled: bool = False
    explicit_repair_operation_required: bool = True
    llm_router_enabled: bool = False
    llm_invocations_allowed: bool = False
    remote_model_provider_calls_allowed: bool = False
    remote_embeddings_allowed: bool = False
    network_access_allowed: bool = False
    target_code_import_allowed: bool = False
    exact_evidence_snapshot_required: bool = True
    clean_rebuild_identity_equivalence_required: bool = True
    canonical_cid_preimage_validation_required: bool = True
    proof_cache_binding_revalidation_required: bool = True
    native_kernel_reconstruction_required: bool = True
    independent_countermodel_validation_required: bool = True
    complete_impact_closure_required: bool = True
    one_disposition_per_resolved_consumer: bool = True
    unique_target_value_placement_operator_required: bool = True
    closed_operator_registry_required: bool = True
    isolated_candidate_worktree_required: bool = True
    enforced_sandbox_required_for_target_execution: bool = True
    writer_lease_and_checkpoint_required: bool = True
    atomic_scc_transaction_required: bool = True
    post_edit_reindex_and_cache_invalidation_required: bool = True
    logic_and_program_fixed_point_required: bool = True
    compensating_rollback_required: bool = True
    knowledge_graph_semantic_authority: bool = False
    vector_semantic_authority: bool = False
    embedding_semantic_authority: bool = False
    tactician_semantic_authority: bool = False
    hammer_candidate_semantic_authority: bool = False
    proof_cache_metadata_semantic_authority: bool = False
    unknown_or_unsupported_disposition: str = "abstain"
    ambiguous_disposition: str = "abstain"
    approval_required_classes: tuple[str, ...] = DEFAULT_APPROVAL_REQUIRED_CLASSES
    limits: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    protected_tcb_path_markers: tuple[str, ...] = DOCTOR_TCB_PATH_MARKERS

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=True)
        )
        object.__setattr__(
            self, "default_mode", _enum(self.default_mode, DoctorMode, "default_mode")
        )
        if self.default_mode is not DoctorMode.REPORT_ONLY:
            # Construction may elevate only when enabled and allowed; still
            # default construction path is report-only.
            pass
        modes_raw = self.allowed_modes
        if not modes_raw:
            modes_raw = ALLOWED_DOCTOR_MODES
        modes: list[DoctorMode] = []
        for item in modes_raw:
            mode = _enum(item, DoctorMode, "allowed_modes")
            assert isinstance(mode, DoctorMode)
            if mode not in modes:
                modes.append(mode)
        expected = list(ALLOWED_DOCTOR_MODES)
        if [m.value for m in modes] != [m.value for m in expected]:
            # Require exact set/order matching the scheduler contract.
            raise DeterministicDoctorPolicyError(
                "allowed_modes must be report_only, plan, sandbox_auto, narrow_auto"
            )
        object.__setattr__(self, "allowed_modes", tuple(modes))

        for name in ELEVATION_FLAGS:
            object.__setattr__(self, name, _bool(getattr(self, name), name))

        for name in HARD_FALSE_FLAGS:
            value = _bool(getattr(self, name), name)
            if value is not False:
                raise DeterministicDoctorSafetyError(
                    f"deterministic doctor safety flag must be false: {name}"
                )
            object.__setattr__(self, name, False)

        for name in HARD_TRUE_GATES:
            value = _bool(getattr(self, name), name)
            if value is not True:
                raise DeterministicDoctorSafetyError(
                    f"deterministic doctor gate must remain enabled: {name}"
                )
            object.__setattr__(self, name, True)

        object.__setattr__(
            self,
            "unknown_or_unsupported_disposition",
            _text(
                self.unknown_or_unsupported_disposition,
                "unknown_or_unsupported_disposition",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "ambiguous_disposition",
            _text(self.ambiguous_disposition, "ambiguous_disposition", required=True),
        )
        if self.unknown_or_unsupported_disposition != "abstain":
            raise DeterministicDoctorPolicyError(
                "unknown_or_unsupported_disposition must be abstain"
            )
        if self.ambiguous_disposition != "abstain":
            raise DeterministicDoctorPolicyError(
                "ambiguous_disposition must be abstain"
            )

        approval = _strings(
            self.approval_required_classes,
            "approval_required_classes",
            required=True,
            preserve_order=True,
        )
        if set(approval) != set(DEFAULT_APPROVAL_REQUIRED_CLASSES):
            raise DeterministicDoctorPolicyError(
                "approval_required_classes must match the closed scheduler set"
            )
        object.__setattr__(self, "approval_required_classes", approval)

        limits_raw = self.limits if self.limits is not None else DEFAULT_LIMITS
        if not isinstance(limits_raw, Mapping):
            raise DeterministicDoctorPolicyError("limits must be a mapping")
        if set(limits_raw) != set(DEFAULT_LIMITS):
            raise DeterministicDoctorPolicyError(
                "limits keys must match the closed scheduler set"
            )
        limits = {
            key: _positive_int(limits_raw[key], key) for key in sorted(DEFAULT_LIMITS)
        }
        object.__setattr__(self, "limits", limits)

        markers = _strings(
            self.protected_tcb_path_markers,
            "protected_tcb_path_markers",
            required=True,
            preserve_order=True,
        )
        object.__setattr__(self, "protected_tcb_path_markers", markers)

        if self.default_mode not in self.allowed_modes:
            raise DeterministicDoctorPolicyError(
                "default_mode must be a member of allowed_modes"
            )
        if (
            self.narrow_autonomous_mutation_enabled
            and DoctorMode.NARROW_AUTO not in self.allowed_modes
        ):
            raise DeterministicDoctorPolicyError(
                "narrow_auto mode must be allowed when mutation is enabled"
            )

        payload_bytes = canonical_json_bytes(self.to_dict())
        if len(payload_bytes) > MAX_POLICY_RECORD_BYTES:
            raise DeterministicDoctorPolicyError(
                "policy exceeds its serialized byte bound"
            )

    # -- accessors ---------------------------------------------------------

    @property
    def resource_bounds(self) -> DoctorResourceBounds:
        return DoctorResourceBounds(**dict(self.limits))

    def mode_allows_operation(
        self, mode: DoctorMode | str, operation: DoctorOperation | str
    ) -> bool:
        """Return whether ``operation`` is permitted under ``mode`` without write elevation."""

        mode_e = _enum(mode, DoctorMode, "mode")
        op_e = _enum(operation, DoctorOperation, "operation")
        assert isinstance(mode_e, DoctorMode)
        assert isinstance(op_e, DoctorOperation)
        if mode_e not in self.allowed_modes:
            return False
        if op_e in (DoctorOperation.INSPECT, DoctorOperation.EXPLAIN):
            return True
        if op_e is DoctorOperation.PLAN:
            return mode_e.rank >= DoctorMode.PLAN.rank or mode_e is DoctorMode.REPORT_ONLY
            # report_only may still *produce* a plan receipt without writes;
            # materialization stays non-writing under plan disposition.
        if op_e is DoctorOperation.REPLAY:
            return True
        if op_e is DoctorOperation.ROLLBACK:
            return mode_e.rank >= DoctorMode.SANDBOX_AUTO.rank or mode_e is DoctorMode.PLAN
        if op_e is DoctorOperation.REPAIR:
            if not self.explicit_repair_operation_required:
                return False
            if mode_e is DoctorMode.SANDBOX_AUTO:
                return True
            if mode_e is DoctorMode.NARROW_AUTO:
                return self.narrow_autonomous_mutation_enabled
            return False
        return False

    def is_path_protected(self, path: str) -> bool:
        normalized = PurePosixPath(str(path or "").replace("\\", "/")).as_posix()
        if is_doctor_tcb_path(normalized):
            return True
        lower = normalized.lower()
        for marker in self.protected_tcb_path_markers:
            marker_norm = marker.rstrip("/")
            if normalized == marker_norm or normalized.startswith(marker_norm + "/"):
                return True
            if marker.endswith("_") and normalized.startswith(marker):
                return True
            if marker.lower() in lower:
                # Conservative: only exact marker prefix matches above; substring
                # matches are not used for non-prefix markers to avoid over-block.
                pass
        return False

    def reject_semantic_authority_claims(
        self, flags: Mapping[str, Any] | None
    ) -> None:
        """Raise if any forbidden semantic-authority flag is true."""

        if not flags:
            return
        for key in FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS:
            if flags.get(key) is True:
                raise DeterministicDoctorSafetyError(
                    f"semantic authority forbidden in deterministic mode: {key}"
                )
        # Also reject alternate key spellings commonly used by nominators.
        aliases = {
            "kg_semantic_authority": DoctorRejectionReason.SEMANTIC_AUTHORITY_KG,
            "knowledge_graph_authority": DoctorRejectionReason.SEMANTIC_AUTHORITY_KG,
            "vector_authority": DoctorRejectionReason.SEMANTIC_AUTHORITY_VECTOR,
            "embedding_authority": DoctorRejectionReason.SEMANTIC_AUTHORITY_EMBEDDING,
            "tactician_authority": DoctorRejectionReason.SEMANTIC_AUTHORITY_TACTICIAN,
            "hammer_authority": DoctorRejectionReason.SEMANTIC_AUTHORITY_HAMMER,
            "cache_metadata_authority": DoctorRejectionReason.SEMANTIC_AUTHORITY_CACHE_METADATA,
        }
        for key, reason in aliases.items():
            if flags.get(key) is True:
                raise DeterministicDoctorSafetyError(
                    f"semantic authority forbidden ({reason.value}): {key}"
                )

    def reject_model_invocation(
        self,
        *,
        llm_router_invoked: bool = False,
        remote_model_provider_invoked: bool = False,
        model_invocation_count: int = 0,
        provider_invocation_count: int = 0,
        network_access: bool = False,
        target_code_imported: bool = False,
    ) -> None:
        """Hard-fail any LLM / remote provider / network / import attempt."""

        if self.llm_invocations_allowed or self.llm_router_enabled:
            raise DeterministicDoctorSafetyError(
                "policy misconfigured: LLM flags must remain false"
            )
        if self.remote_model_provider_calls_allowed:
            raise DeterministicDoctorSafetyError(
                "policy misconfigured: remote model provider must remain false"
            )
        if llm_router_invoked or self.llm_router_enabled:
            raise DeterministicDoctorSafetyError(
                DoctorRejectionReason.LLM_INVOCATION.value
            )
        if remote_model_provider_invoked:
            raise DeterministicDoctorSafetyError(
                DoctorRejectionReason.REMOTE_MODEL_PROVIDER.value
            )
        if model_invocation_count != 0 or provider_invocation_count != 0:
            raise DeterministicDoctorSafetyError(
                DoctorRejectionReason.NONZERO_MODEL_INVOCATION.value
            )
        if network_access or self.network_access_allowed:
            raise DeterministicDoctorSafetyError(
                DoctorRejectionReason.NETWORK_ACCESS.value
            )
        if target_code_imported or self.target_code_import_allowed:
            raise DeterministicDoctorSafetyError(
                DoctorRejectionReason.TARGET_CODE_IMPORT.value
            )

    def require_repair_prerequisites(
        self,
        *,
        plan: DeterministicDoctorPlan | Mapping[str, Any] | None,
        lease_id: str = "",
        checkpoint_ref: str = "",
        rollback_ref: str = "",
    ) -> DeterministicDoctorPlan:
        """Validate repair prerequisites and return the admitted plan."""

        if plan is None:
            raise DeterministicDoctorAuthorityError(
                DoctorRejectionReason.REPAIR_WITHOUT_ADMITTED_PLAN.value
            )
        if isinstance(plan, Mapping):
            plan = DeterministicDoctorPlan.from_dict(plan)
        if not isinstance(plan, DeterministicDoctorPlan):
            raise DeterministicDoctorPolicyError("plan must be DeterministicDoctorPlan")
        if plan.disposition is not DoctorPlanDisposition.ADMITTED:
            raise DeterministicDoctorAuthorityError(
                DoctorRejectionReason.REPAIR_WITHOUT_ADMITTED_PLAN.value
            )
        if plan.open_required_frontiers:
            raise DeterministicDoctorAuthorityError(
                DoctorRejectionReason.OPEN_REQUIRED_FRONTIER.value
            )
        effective_lease = lease_id or plan.lease_id or plan.roots.lease_id
        if self.writer_lease_and_checkpoint_required and not effective_lease:
            raise DeterministicDoctorAuthorityError(
                DoctorRejectionReason.REPAIR_WITHOUT_LEASE.value
            )
        effective_checkpoint = checkpoint_ref or plan.checkpoint_ref
        if self.writer_lease_and_checkpoint_required and not effective_checkpoint:
            raise DeterministicDoctorAuthorityError(
                DoctorRejectionReason.REPAIR_WITHOUT_CHECKPOINT.value
            )
        effective_rollback = rollback_ref or plan.rollback_ref
        if self.compensating_rollback_required and not effective_rollback:
            raise DeterministicDoctorAuthorityError(
                DoctorRejectionReason.REPAIR_WITHOUT_ROLLBACK.value
            )
        for path in plan.permitted_write_paths:
            if self.is_path_protected(path):
                raise DeterministicDoctorAuthorityError(
                    DoctorRejectionReason.TCB_PATH.value
                )
        self.reject_semantic_authority_claims(plan.semantic_authority_flags)
        self.reject_model_invocation(
            model_invocation_count=plan.model_invocation_count,
            llm_router_invoked=plan.llm_router_enabled,
        )
        return plan

    def evaluate(
        self,
        *,
        operation: DoctorOperation | str,
        mode: DoctorMode | str | None = None,
        plan: DeterministicDoctorPlan | Mapping[str, Any] | None = None,
        write_paths: Sequence[str] = (),
        approval_classes: Sequence[str] = (),
        open_required_frontiers: Sequence[str] = (),
        lease_id: str = "",
        checkpoint_ref: str = "",
        rollback_ref: str = "",
        llm_router_invoked: bool = False,
        remote_model_provider_invoked: bool = False,
        model_invocation_count: int = 0,
        provider_invocation_count: int = 0,
        network_access: bool = False,
        target_code_imported: bool = False,
        semantic_authority_flags: Mapping[str, Any] | None = None,
        cross_repository: bool = False,
        new_external_dependency: bool = False,
        unsupported_memory_or_lifetime_claim: bool = False,
        forged_cid: bool = False,
        has_body_or_secret: bool = False,
        has_cycle: bool = False,
        unbounded: bool = False,
        partial_plan: bool = False,
    ) -> "DoctorPolicyDecision":
        """Evaluate a doctor operation request and return a closed decision."""

        op = _enum(operation, DoctorOperation, "operation")
        assert isinstance(op, DoctorOperation)
        effective_mode = _enum(
            mode if mode is not None else self.default_mode, DoctorMode, "mode"
        )
        assert isinstance(effective_mode, DoctorMode)

        reasons: list[str] = []
        classes = set(approval_classes)
        classes.update(
            item.value
            for item in classify_change_approval_classes(
                paths=write_paths,
                cross_repository=cross_repository,
                new_external_dependency=new_external_dependency,
                unsupported_memory_or_lifetime_claim=unsupported_memory_or_lifetime_claim,
            )
        )

        try:
            self.reject_model_invocation(
                llm_router_invoked=llm_router_invoked,
                remote_model_provider_invoked=remote_model_provider_invoked,
                model_invocation_count=model_invocation_count,
                provider_invocation_count=provider_invocation_count,
                network_access=network_access,
                target_code_imported=target_code_imported,
            )
            self.reject_semantic_authority_claims(semantic_authority_flags)
        except DeterministicDoctorSafetyError as exc:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(str(exc),),
                policy_id=self.policy_id,
            )

        if forged_cid:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.FORGED_CID.value,),
                policy_id=self.policy_id,
            )
        if has_body_or_secret:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.BODY_OR_SECRET.value,),
                policy_id=self.policy_id,
            )
        if has_cycle:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.CYCLE.value,),
                policy_id=self.policy_id,
            )
        if unbounded:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.UNBOUNDED_DATA.value,),
                policy_id=self.policy_id,
            )
        if partial_plan:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.PARTIAL_PLAN.value,),
                policy_id=self.policy_id,
            )

        if effective_mode not in self.allowed_modes:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.MODE_FORBIDS_OPERATION.value,),
                policy_id=self.policy_id,
            )

        if not self.mode_allows_operation(effective_mode, op):
            # Read-only ops under report_only: always allowed for inspect/explain/plan-as-report.
            if op in READ_ONLY_OPERATIONS and effective_mode is DoctorMode.REPORT_ONLY:
                if op is DoctorOperation.PLAN:
                    # Report-only may inspect and explain; plan production is
                    # allowed as a non-writing analytical artifact.
                    pass
                elif op in (DoctorOperation.INSPECT, DoctorOperation.EXPLAIN, DoctorOperation.REPLAY):
                    pass
                else:
                    return DoctorPolicyDecision(
                        verdict=PolicyVerdict.REJECT,
                        operation=op,
                        mode=effective_mode,
                        reason_codes=(DoctorRejectionReason.MODE_FORBIDS_OPERATION.value,),
                        policy_id=self.policy_id,
                    )
            elif op is DoctorOperation.PLAN and effective_mode is DoctorMode.REPORT_ONLY:
                pass
            else:
                return DoctorPolicyDecision(
                    verdict=PolicyVerdict.REJECT,
                    operation=op,
                    mode=effective_mode,
                    reason_codes=(DoctorRejectionReason.MODE_FORBIDS_OPERATION.value,),
                    policy_id=self.policy_id,
                )

        for path in write_paths:
            if self.is_path_protected(path):
                classes.add(DoctorApprovalClass.DOCTOR_TRUSTED_COMPUTING_BASE.value)
                return DoctorPolicyDecision(
                    verdict=PolicyVerdict.REJECT,
                    operation=op,
                    mode=effective_mode,
                    reason_codes=(DoctorRejectionReason.TCB_PATH.value,),
                    approval_classes=tuple(sorted(classes)),
                    policy_id=self.policy_id,
                )

        if open_required_frontiers and op is DoctorOperation.REPAIR:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.ABSTAIN,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.OPEN_REQUIRED_FRONTIER.value,),
                policy_id=self.policy_id,
            )

        if classes & set(self.approval_required_classes):
            if op is DoctorOperation.REPAIR or write_paths:
                return DoctorPolicyDecision(
                    verdict=PolicyVerdict.APPROVAL_REQUIRED,
                    operation=op,
                    mode=effective_mode,
                    reason_codes=(DoctorRejectionReason.APPROVAL_REQUIRED.value,),
                    approval_classes=tuple(sorted(classes)),
                    policy_id=self.policy_id,
                )

        if op is DoctorOperation.REPAIR:
            if not self.enabled and effective_mode is DoctorMode.NARROW_AUTO:
                return DoctorPolicyDecision(
                    verdict=PolicyVerdict.REJECT,
                    operation=op,
                    mode=effective_mode,
                    reason_codes=(DoctorRejectionReason.MODE_FORBIDS_OPERATION.value,),
                    policy_id=self.policy_id,
                )
            try:
                admitted = self.require_repair_prerequisites(
                    plan=plan,
                    lease_id=lease_id,
                    checkpoint_ref=checkpoint_ref,
                    rollback_ref=rollback_ref,
                )
            except (
                DeterministicDoctorAuthorityError,
                DeterministicDoctorSafetyError,
                DeterministicDoctorError,
                ForgedDeterministicDoctorIdentityError,
            ) as exc:
                reason = str(exc) or DoctorRejectionReason.REPAIR_WITHOUT_ADMITTED_PLAN.value
                verdict = (
                    PolicyVerdict.ABSTAIN
                    if "frontier" in reason
                    else PolicyVerdict.REJECT
                )
                return DoctorPolicyDecision(
                    verdict=verdict,
                    operation=op,
                    mode=effective_mode,
                    reason_codes=(reason,),
                    policy_id=self.policy_id,
                )
            reasons.append("admitted_plan")
            _ = admitted

        # Feature may be disabled while still allowing pure report/inspect.
        if not self.enabled and op is DoctorOperation.REPAIR:
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.REJECT,
                operation=op,
                mode=effective_mode,
                reason_codes=(DoctorRejectionReason.MODE_FORBIDS_OPERATION.value,),
                policy_id=self.policy_id,
            )

        return DoctorPolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            operation=op,
            mode=effective_mode,
            reason_codes=tuple(reasons) or ("policy_allow",),
            approval_classes=tuple(sorted(classes)),
            policy_id=self.policy_id,
            read_only=operation_is_read_only(op),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "policy_id": self.policy_id,
            "enabled": bool(self.enabled),
            "default_mode": self.default_mode.value
            if isinstance(self.default_mode, DoctorMode)
            else self.default_mode,
            "allowed_modes": [
                m.value if isinstance(m, DoctorMode) else m for m in self.allowed_modes
            ],
            "narrow_autonomous_mutation_enabled": self.narrow_autonomous_mutation_enabled,
            "explicit_repair_operation_required": True,
            "llm_router_enabled": False,
            "llm_invocations_allowed": False,
            "remote_model_provider_calls_allowed": False,
            "remote_embeddings_allowed": False,
            "network_access_allowed": False,
            "target_code_import_allowed": False,
            "exact_evidence_snapshot_required": True,
            "clean_rebuild_identity_equivalence_required": True,
            "canonical_cid_preimage_validation_required": True,
            "proof_cache_binding_revalidation_required": True,
            "native_kernel_reconstruction_required": True,
            "independent_countermodel_validation_required": True,
            "complete_impact_closure_required": True,
            "one_disposition_per_resolved_consumer": True,
            "unique_target_value_placement_operator_required": True,
            "closed_operator_registry_required": True,
            "isolated_candidate_worktree_required": True,
            "enforced_sandbox_required_for_target_execution": True,
            "writer_lease_and_checkpoint_required": True,
            "atomic_scc_transaction_required": True,
            "post_edit_reindex_and_cache_invalidation_required": True,
            "logic_and_program_fixed_point_required": True,
            "compensating_rollback_required": True,
            "knowledge_graph_semantic_authority": False,
            "vector_semantic_authority": False,
            "embedding_semantic_authority": False,
            "tactician_semantic_authority": False,
            "hammer_candidate_semantic_authority": False,
            "proof_cache_metadata_semantic_authority": False,
            "unknown_or_unsupported_disposition": "abstain",
            "ambiguous_disposition": "abstain",
            "approval_required_classes": list(self.approval_required_classes),
            "limits": dict(self.limits),
            "protected_tcb_path_markers": list(self.protected_tcb_path_markers),
        }

    def to_scheduler_dict(self) -> dict[str, Any]:
        """Return the scheduler-compatible policy object (schema field included)."""

        payload = self.to_dict()
        # Strip contract_version for scheduler shape if present as extra.
        return {
            "schema": self.SCHEMA,
            **{k: v for k, v in payload.items() if k != "contract_version"},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeterministicDoctorPolicy":
        if not isinstance(payload, Mapping):
            raise DeterministicDoctorPolicyError("policy payload must be a mapping")
        schema = payload.get("schema")
        if schema not in (None, "", cls.SCHEMA):
            raise DeterministicDoctorPolicyError(
                f"unsupported policy schema; use {cls.SCHEMA}"
            )
        data = dict(payload)
        data.pop("schema", None)
        data.pop("content_id", None)
        data.pop("cid", None)
        data.pop("contract_version", None)
        allowed = set(cls.__dataclass_fields__) - {"SCHEMA"}
        unknown = set(data) - allowed
        if unknown:
            raise DeterministicDoctorPolicyError(
                "policy contains unsupported fields; rebuild its canonical payload"
            )
        value = cls(**{k: data[k] for k in data if k in allowed})
        supplied = payload.get("content_id", payload.get("cid", ""))
        if supplied not in (None, ""):
            if not isinstance(supplied, str) or supplied != value.content_id:
                raise ForgedDeterministicDoctorIdentityError(
                    "stored content identity does not match the canonical policy"
                )
        return value

    @classmethod
    def default(cls) -> "DeterministicDoctorPolicy":
        """Return the fail-closed report-only default policy."""

        return cls()


@dataclass(frozen=True)
class DoctorPolicyDecision(CanonicalContract):
    """Content-addressed outcome of one policy evaluation."""

    SCHEMA: ClassVar[str] = POLICY_DECISION_SCHEMA

    verdict: PolicyVerdict
    operation: DoctorOperation
    mode: DoctorMode
    reason_codes: tuple[str, ...] = ()
    approval_classes: tuple[str, ...] = ()
    policy_id: str = "policy:deterministic-doctor:default"
    read_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum(self.verdict, PolicyVerdict, "verdict")
        )
        object.__setattr__(
            self, "operation", _enum(self.operation, DoctorOperation, "operation")
        )
        object.__setattr__(self, "mode", _enum(self.mode, DoctorMode, "mode"))
        object.__setattr__(
            self,
            "reason_codes",
            _strings(self.reason_codes, "reason_codes", preserve_order=True),
        )
        object.__setattr__(
            self,
            "approval_classes",
            _strings(self.approval_classes, "approval_classes"),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=True)
        )
        if not isinstance(self.read_only, bool):
            raise DeterministicDoctorPolicyError("read_only must be a boolean")
        op = self.operation
        assert isinstance(op, DoctorOperation)
        object.__setattr__(self, "read_only", op.is_read_only)

    @property
    def allowed(self) -> bool:
        return self.verdict is PolicyVerdict.ALLOW

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "verdict": self.verdict.value
            if isinstance(self.verdict, PolicyVerdict)
            else self.verdict,
            "operation": self.operation.value
            if isinstance(self.operation, DoctorOperation)
            else self.operation,
            "mode": self.mode.value if isinstance(self.mode, DoctorMode) else self.mode,
            "reason_codes": list(self.reason_codes),
            "approval_classes": list(self.approval_classes),
            "policy_id": self.policy_id,
            "read_only": self.read_only,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorPolicyDecision":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DeterministicDoctorPolicyError(
                "policy decision has an unsupported schema"
            )
        data = {
            key: payload[key]
            for key in (
                "verdict",
                "operation",
                "mode",
                "reason_codes",
                "approval_classes",
                "policy_id",
                "read_only",
            )
            if key in payload
        }
        value = cls(**data)
        supplied = payload.get("content_id", payload.get("cid", ""))
        if supplied not in (None, ""):
            if not isinstance(supplied, str) or supplied != value.content_id:
                raise ForgedDeterministicDoctorIdentityError(
                    "stored content identity does not match the canonical decision"
                )
        return value


def default_deterministic_doctor_policy() -> DeterministicDoctorPolicy:
    """Return the production default (disabled, report-only, no model)."""

    return DeterministicDoctorPolicy.default()


def load_deterministic_doctor_policy(
    payload: Mapping[str, Any] | DeterministicDoctorPolicy | None,
) -> DeterministicDoctorPolicy:
    """Load a policy from a scheduler mapping or return the default."""

    if payload is None:
        return DeterministicDoctorPolicy.default()
    if isinstance(payload, DeterministicDoctorPolicy):
        return payload
    return DeterministicDoctorPolicy.from_dict(payload)


def assert_run_receipt_policy(
    receipt: DeterministicDoctorRunReceipt | Mapping[str, Any],
    policy: DeterministicDoctorPolicy | None = None,
) -> DeterministicDoctorRunReceipt:
    """Validate a run receipt against policy (zero model invocations, etc.)."""

    policy = policy or DeterministicDoctorPolicy.default()
    if isinstance(receipt, Mapping):
        receipt = DeterministicDoctorRunReceipt.from_dict(receipt)
    if not isinstance(receipt, DeterministicDoctorRunReceipt):
        raise DeterministicDoctorPolicyError(
            "receipt must be DeterministicDoctorRunReceipt"
        )
    policy.reject_model_invocation(
        llm_router_invoked=receipt.llm_router_invoked,
        remote_model_provider_invoked=receipt.remote_model_provider_invoked,
        model_invocation_count=receipt.model_invocation_count,
        provider_invocation_count=receipt.provider_invocation_count,
        network_access=not receipt.network_denied,
        target_code_imported=receipt.target_code_imported,
    )
    if receipt.operation is DoctorOperation.REPAIR:
        if receipt.disposition is DoctorRepairDisposition.SUPPORTED:
            if not receipt.plan_id:
                raise DeterministicDoctorAuthorityError(
                    DoctorRejectionReason.REPAIR_WITHOUT_ADMITTED_PLAN.value
                )
    return receipt


def evaluate_doctor_operation(
    operation: DoctorOperation | str,
    *,
    policy: DeterministicDoctorPolicy | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> DoctorPolicyDecision:
    """Convenience entry point for policy evaluation."""

    resolved = load_deterministic_doctor_policy(policy)
    return resolved.evaluate(operation=operation, **kwargs)


__all__ = [
    "DEFAULT_APPROVAL_REQUIRED_CLASSES",
    "DEFAULT_LIMITS",
    "DETERMINISTIC_DOCTOR_POLICY_INTERFACE",
    "DETERMINISTIC_DOCTOR_POLICY_SCHEMA",
    "HARD_FALSE_FLAGS",
    "HARD_TRUE_GATES",
    "POLICY_DECISION_SCHEMA",
    "DeterministicDoctorPolicy",
    "DeterministicDoctorPolicyError",
    "DoctorPolicyDecision",
    "PolicyVerdict",
    "assert_run_receipt_policy",
    "classify_change_approval_classes",
    "classify_path_approval_classes",
    "default_deterministic_doctor_policy",
    "evaluate_doctor_operation",
    "load_deterministic_doctor_policy",
]
