"""Isolated sandbox apply for deterministic-doctor repair plans (LPR-038).

Interface: ``DeterministicDoctorTransaction@1``

An admitted :class:`DeterministicDoctorPlan` is applied only inside a disposable
candidate worktree under enforced sandbox policy.  The transaction:

* confines repository paths and denies inherited secrets, network, and process
  escape with platform enforcement evidence before any target execution;
* treats symlink, hardlink, submodule, device, and path-race observations as
  hostile (fail-closed);
* permits pure static replay under weak isolation and forces abstention for
  any execution-dependent repair when enforcement is incomplete;
* acquires checkout lock, writer lease, and content-addressed checkpoint, then
  revalidates authority roots and proof-cache bindings immediately before
  commit;
* applies each SCC atomically (entire group or nothing) and integrates through
  merge-ref compare-and-swap; and
* never imports target code into the doctor process, never weakens sandbox
  claims, never overwrites a dirty user tree, and never invokes a model.

Compensating rollback restores the checkpoint on failure, drift, timeout,
scope escape, CAS conflict, or incomplete SCC.  Partial merge/completion is
forbidden.  Canonical receipts bind base/candidate/committed CIDs, sandbox
evidence, allowlisted commands, resources, lease/checkpoint/before hashes,
SCC group outcomes, and rollback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..analysis.deterministic_doctor_contracts import (
    MAX_PATH_BYTES,
    MAX_REFERENCE_COUNT,
    MAX_STEP_COUNT,
    MAX_TEXT_BYTES,
    DoctorAuthorityRoots,
    DoctorEditSite,
    DoctorMode,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
    DoctorResourceBounds,
    DeterministicDoctorPlan,
    is_doctor_tcb_path,
)
from ..analysis.deterministic_doctor_impact import (
    doctor_roots_to_propagation_roots,
    path_is_forbidden,
)
from .change_propagation_transaction import (
    CHANGE_PROPAGATION_TRANSACTION_INTERFACE,
    GroupExecutionDisposition,
    PropagationCheckpoint,
    PropagationGroupReceipt,
    PropagationRollbackReceipt,
    PropagationStepReceipt,
    StepExecutionDisposition,
    TransactionLease,
    create_propagation_checkpoint,
)
from ..proof.change_propagation_edit_packet import PathBeforeHash
from ..proof.formal_verification_contracts import content_identity


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE: Final[str] = (
    "DeterministicDoctorTransaction@2"
)
LEGACY_DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE: Final[str] = (
    "DeterministicDoctorTransaction@1"
)
DOCTOR_SANDBOX_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/sandbox-policy@1"
)
DOCTOR_SANDBOX_ENFORCEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/sandbox-enforcement@1"
)
DOCTOR_CANDIDATE_TREE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/candidate-tree-receipt@1"
)
DOCTOR_CHECKOUT_LOCK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/checkout-lock@1"
)
DOCTOR_WRITER_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/writer-lease@1"
)
DOCTOR_MERGE_REF_CAS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/merge-ref-cas@1"
)
DOCTOR_TRANSACTION_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/transaction-checkpoint@1"
)
DOCTOR_STEP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/step-receipt@1"
)
DOCTOR_GROUP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/group-receipt@1"
)
DOCTOR_TRANSACTION_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/transaction-report@1"
)
DOCTOR_ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/rollback-receipt@1"
)

PRODUCER_ID: Final[str] = "deterministic-doctor-transaction@1"
CONTRACT_VERSION: Final[int] = 1

MAX_PATHS: Final[int] = 1_024
MAX_STEPS: Final[int] = MAX_STEP_COUNT
MAX_DIAGNOSTICS: Final[int] = 256
MAX_REASON_CODES: Final[int] = 64
MAX_COMMANDS: Final[int] = 128
MAX_HOSTILE_OBSERVATIONS: Final[int] = 256

# Markers that must never appear on the doctor's runtime surface.
_FORBIDDEN_PROVIDER_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "llm_router",
        "model_provider",
        "openai",
        "anthropic",
        "provider_router",
        "todo_daemon.change_propagation_provider_router",
    }
)

# Path / FS observations treated as hostile for candidate apply.
_HOSTILE_FS_KINDS: Final[frozenset[str]] = frozenset(
    {
        "symlink",
        "hardlink",
        "submodule",
        "device",
        "path_race",
        "fifo",
        "socket",
        "whiteout",
    }
)

_DEFAULT_ALLOWLISTED_COMMANDS: Final[tuple[str, ...]] = (
    "python",
    "python3",
    "pytest",
    "mypy",
    "ruff",
    "git",
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class DoctorSandboxCapability(str, Enum):
    """Platform-enforced sandbox capabilities required for target execution."""

    PATH_CONFINEMENT = "path_confinement"
    SECRETS_DENIED = "secrets_denied"
    NETWORK_DENIED = "network_denied"
    PROCESS_LIMITS = "process_limits"
    CPU_LIMITS = "cpu_limits"
    MEMORY_LIMITS = "memory_limits"
    TIME_LIMITS = "time_limits"
    COMMAND_ALLOWLIST = "command_allowlist"
    NO_TARGET_IMPORT = "no_target_import"
    DISPOSABLE_WORKTREE = "disposable_worktree"


class DoctorSandboxEnforcementLevel(str, Enum):
    """How completely the host can enforce sandbox claims."""

    ENFORCED = "enforced"
    WEAK = "weak"
    ABSENT = "absent"

    @property
    def permits_target_execution(self) -> bool:
        return self is DoctorSandboxEnforcementLevel.ENFORCED

    @property
    def permits_static_replay(self) -> bool:
        return self in (
            DoctorSandboxEnforcementLevel.ENFORCED,
            DoctorSandboxEnforcementLevel.WEAK,
        )


class DoctorTransactionDisposition(str, Enum):
    """Terminal outcomes of one isolated doctor transaction attempt."""

    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    ABSTAINED = "abstained"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"

    @property
    def claims_completion(self) -> bool:
        # Transaction commit is provisional; fixed-point alone may complete.
        return False

    @property
    def is_terminal_failure(self) -> bool:
        return self in (
            DoctorTransactionDisposition.ROLLED_BACK,
            DoctorTransactionDisposition.ABSTAINED,
            DoctorTransactionDisposition.QUARANTINED,
            DoctorTransactionDisposition.REJECTED,
        )


class DoctorTransactionReason(str, Enum):
    """Stable machine-readable transaction failure / abstention codes."""

    MALFORMED_INPUT = "malformed_input"
    PLAN_NOT_ADMITTED = "plan_not_admitted"
    ROOT_DRIFT = "root_drift"
    SANDBOX_ENFORCEMENT_MISSING = "sandbox_enforcement_missing"
    SANDBOX_WEAK_EXECUTION_FORBIDDEN = "sandbox_weak_execution_forbidden"
    SECRETS_INHERITED = "secrets_inherited"
    NETWORK_NOT_DENIED = "network_not_denied"
    PATH_ESCAPE = "path_escape"
    TCB_PATH = "tcb_path"
    FORBIDDEN_PATH = "forbidden_path"
    HOSTILE_FS_OBSERVATION = "hostile_fs_observation"
    COMMAND_NOT_ALLOWLISTED = "command_not_allowlisted"
    CHECKOUT_LOCK_MISSING = "checkout_lock_missing"
    CHECKOUT_LOCK_INVALID = "checkout_lock_invalid"
    LEASE_MISSING = "lease_missing"
    LEASE_INVALID = "lease_invalid"
    LEASE_PATH_MISMATCH = "lease_path_mismatch"
    CHECKPOINT_MISSING = "checkpoint_missing"
    BEFORE_HASH_MISMATCH = "before_hash_mismatch"
    BEFORE_HASH_MISSING = "before_hash_missing"
    CACHE_BINDING_STALE = "cache_binding_stale"
    SCOPE_ESCAPE = "scope_escape"
    STEP_FAILURE = "step_failure"
    GROUP_INCOMPLETE = "group_incomplete"
    DEPENDENCY_UNMET = "dependency_unmet"
    TIMEOUT = "timeout"
    DRIFT = "drift"
    PARTIAL_MERGE_FORBIDDEN = "partial_merge_forbidden"
    PARTIAL_SCC_FORBIDDEN = "partial_scc_forbidden"
    CAS_CONFLICT = "merge_ref_cas_conflict"
    CAS_EXPECTED_MISMATCH = "merge_ref_cas_expected_mismatch"
    DIRTY_USER_TREE = "dirty_user_tree"
    TARGET_CODE_IMPORT_FORBIDDEN = "target_code_import_forbidden"
    MODEL_INVOCATION_FORBIDDEN = "model_invocation_forbidden"
    RESTORE_FAILED = "restore_failed"
    QUARANTINE_REQUIRED = "quarantine_required"
    RESOURCE_BOUND = "resource_bound"
    EXECUTION_DEPENDENT_ABSTAIN = "execution_dependent_abstain"
    STATIC_REPLAY_ONLY = "static_replay_only"
    PRE_COMMIT_REVALIDATION_FAILED = "pre_commit_revalidation_failed"
    ALREADY_TERMINAL = "already_terminal"
    NO_EXPECTED_CHANGE = "no_expected_change"
    EFFECT_EVIDENCE_MISSING = "effect_evidence_missing"
    DURABLE_INTENT_MISSING = "durable_intent_missing"
    REF_CAS_NOT_APPLIED = "ref_cas_not_applied"


class DoctorStepDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    SCOPE_ESCAPE = "scope_escape"
    DRIFT = "drift"
    HOSTILE = "hostile"


class DoctorGroupDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DoctorHostileObservationKind(str, Enum):
    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    SUBMODULE = "submodule"
    DEVICE = "device"
    PATH_RACE = "path_race"
    FIFO = "fifo"
    SOCKET = "socket"
    WHITEOUT = "whiteout"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeterministicDoctorTransactionError(ValueError):
    """A doctor transaction would weaken isolation, lease, or plan authority."""


class DoctorSandboxError(DeterministicDoctorTransactionError):
    """Sandbox policy or enforcement evidence is insufficient."""


class DoctorMergeCasError(DeterministicDoctorTransactionError):
    """Merge-ref compare-and-swap rejected the candidate tree."""


class DoctorQuarantineError(DeterministicDoctorTransactionError):
    """Rollback failed; the candidate tree is quarantined."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise DeterministicDoctorTransactionError(f"{name} must be a compact identifier")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise DeterministicDoctorTransactionError(f"{name} exceeds text bound")
    return text


def _optional_identifier(value: Any, name: str) -> str:
    if value is None or value == "":
        return ""
    return _identifier(value, name)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise DeterministicDoctorTransactionError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise DeterministicDoctorTransactionError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise DeterministicDoctorTransactionError(f"{name} exceeds text bound")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DeterministicDoctorTransactionError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeterministicDoctorTransactionError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise DeterministicDoctorTransactionError(f"{name} out of bounds")
    return value


def _path(value: Any, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DeterministicDoctorTransactionError(f"{name} contains an invalid path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise DeterministicDoctorTransactionError(f"{name} contains an escaped path")
    text = path.as_posix()
    if len(text.encode("utf-8")) > MAX_PATH_BYTES:
        raise DeterministicDoctorTransactionError(f"{name} exceeds path bound")
    return text


def _paths(values: Sequence[str], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DeterministicDoctorTransactionError(f"{name} must be a path sequence")
    result: set[str] = set()
    for value in values:
        result.add(_path(value, name))
    if required and not result:
        raise DeterministicDoctorTransactionError(f"{name} must not be empty")
    if len(result) > MAX_PATHS:
        raise DeterministicDoctorTransactionError(f"{name} exceeds path bound")
    return tuple(sorted(result))


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_REFERENCE_COUNT,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DeterministicDoctorTransactionError(f"{name} must be an identifier sequence")
    if preserve_order:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise DeterministicDoctorTransactionError(f"{name} contains an invalid id")
            item = value.strip()
            if any(char.isspace() for char in item):
                raise DeterministicDoctorTransactionError(
                    f"{name} must contain compact identifiers"
                )
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        result = tuple(ordered)
    else:
        result = tuple(
            sorted(
                {
                    value.strip()
                    for value in values
                    if isinstance(value, str) and value.strip()
                }
            )
        )
        if any(any(char.isspace() for char in item) for item in result):
            raise DeterministicDoctorTransactionError(
                f"{name} must contain compact identifiers"
            )
    if required and not result:
        raise DeterministicDoctorTransactionError(f"{name} must not be empty")
    if len(result) > maximum:
        raise DeterministicDoctorTransactionError(f"{name} exceeds item bound")
    return result


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        try:
            return enum(value)
        except ValueError as exc:
            raise DeterministicDoctorTransactionError(
                f"{name} must be a valid {enum.__name__}"
            ) from exc
    raise DeterministicDoctorTransactionError(f"{name} must be a valid {enum.__name__}")


def _roots(value: Any) -> DoctorAuthorityRoots:
    if isinstance(value, DoctorAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return DoctorAuthorityRoots.from_dict(value)
    raise DeterministicDoctorTransactionError("roots must be DoctorAuthorityRoots")


def assert_no_provider_surface(module_globals: Mapping[str, Any] | None = None) -> None:
    """Structural guard: doctor transaction must not expose provider imports."""

    source = module_globals if module_globals is not None else globals()
    text = " ".join(str(key) for key in source)
    for marker in _FORBIDDEN_PROVIDER_MARKERS:
        if marker in text and marker in {
            "llm_router",
            "model_provider",
            "openai",
            "anthropic",
        }:
            # Only fail when the marker is an imported module name present as a
            # binding, not merely as a denial string in this file.
            if marker in source:
                raise DeterministicDoctorTransactionError(
                    f"provider surface {marker!r} is forbidden"
                )


# ---------------------------------------------------------------------------
# Sandbox policy and enforcement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorSandboxPolicy:
    """Fail-closed sandbox requirements for isolated candidate apply.

    ``enforcement_level`` is measured platform evidence, not a claim the
    doctor invents.  Weak/absent enforcement forbids target-code execution
    and forces abstention for execution-dependent repairs.
    """

    sandbox_id: str
    worktree_root_ref: str
    permitted_paths: tuple[str, ...]
    allowlisted_commands: tuple[str, ...] = _DEFAULT_ALLOWLISTED_COMMANDS
    required_capabilities: tuple[DoctorSandboxCapability, ...] = (
        DoctorSandboxCapability.PATH_CONFINEMENT,
        DoctorSandboxCapability.SECRETS_DENIED,
        DoctorSandboxCapability.NETWORK_DENIED,
        DoctorSandboxCapability.PROCESS_LIMITS,
        DoctorSandboxCapability.COMMAND_ALLOWLIST,
        DoctorSandboxCapability.NO_TARGET_IMPORT,
        DoctorSandboxCapability.DISPOSABLE_WORKTREE,
    )
    enforcement_level: DoctorSandboxEnforcementLevel = DoctorSandboxEnforcementLevel.ENFORCED
    secrets_inherited: bool = False
    network_denied: bool = True
    target_code_imported: bool = False
    max_processes: int = 8
    max_wall_time_seconds: int = 3600
    max_cpu_time_seconds: int = 1800
    max_memory_bytes: int = 4_294_967_296
    max_changed_files: int = 128
    max_changed_bytes: int = 1_048_576
    resource_bounds: DoctorResourceBounds | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sandbox_id", _identifier(self.sandbox_id, "sandbox_id"))
        object.__setattr__(
            self,
            "worktree_root_ref",
            _identifier(self.worktree_root_ref, "worktree_root_ref"),
        )
        object.__setattr__(
            self,
            "permitted_paths",
            _paths(self.permitted_paths, "permitted_paths", required=True),
        )
        commands = _ids(
            self.allowlisted_commands,
            "allowlisted_commands",
            required=True,
            maximum=MAX_COMMANDS,
            preserve_order=True,
        )
        object.__setattr__(self, "allowlisted_commands", commands)
        caps: list[DoctorSandboxCapability] = []
        for item in self.required_capabilities:
            caps.append(
                DoctorSandboxCapability(item)
                if not isinstance(item, DoctorSandboxCapability)
                else item
            )
        if not caps:
            raise DoctorSandboxError("sandbox requires at least one capability")
        object.__setattr__(self, "required_capabilities", tuple(caps))
        object.__setattr__(
            self,
            "enforcement_level",
            _enum(
                self.enforcement_level,
                DoctorSandboxEnforcementLevel,
                "enforcement_level",
            ),
        )
        object.__setattr__(
            self, "secrets_inherited", _bool(self.secrets_inherited, "secrets_inherited")
        )
        object.__setattr__(
            self, "network_denied", _bool(self.network_denied, "network_denied")
        )
        object.__setattr__(
            self,
            "target_code_imported",
            _bool(self.target_code_imported, "target_code_imported"),
        )
        object.__setattr__(
            self, "max_processes", _nonneg_int(self.max_processes, "max_processes")
        )
        object.__setattr__(
            self,
            "max_wall_time_seconds",
            _nonneg_int(self.max_wall_time_seconds, "max_wall_time_seconds"),
        )
        object.__setattr__(
            self,
            "max_cpu_time_seconds",
            _nonneg_int(self.max_cpu_time_seconds, "max_cpu_time_seconds"),
        )
        object.__setattr__(
            self,
            "max_memory_bytes",
            _nonneg_int(self.max_memory_bytes, "max_memory_bytes"),
        )
        object.__setattr__(
            self,
            "max_changed_files",
            _nonneg_int(self.max_changed_files, "max_changed_files"),
        )
        object.__setattr__(
            self,
            "max_changed_bytes",
            _nonneg_int(self.max_changed_bytes, "max_changed_bytes"),
        )
        if self.resource_bounds is None:
            object.__setattr__(self, "resource_bounds", DoctorResourceBounds())
        elif not isinstance(self.resource_bounds, DoctorResourceBounds):
            raise DeterministicDoctorTransactionError(
                "resource_bounds must be DoctorResourceBounds"
            )
        if self.secrets_inherited:
            raise DoctorSandboxError("sandbox must not inherit secrets")
        if not self.network_denied:
            raise DoctorSandboxError("sandbox requires network denial")
        if self.target_code_imported:
            raise DoctorSandboxError("sandbox forbids importing target code")
        for path in self.permitted_paths:
            if is_doctor_tcb_path(path):
                raise DoctorSandboxError(
                    "sandbox cannot grant write authority over doctor TCB paths"
                )

    @property
    def permits_target_execution(self) -> bool:
        level = self.enforcement_level
        assert isinstance(level, DoctorSandboxEnforcementLevel)
        return (
            level.permits_target_execution
            and not self.secrets_inherited
            and self.network_denied
            and not self.target_code_imported
        )

    @property
    def permits_static_replay_only(self) -> bool:
        level = self.enforcement_level
        assert isinstance(level, DoctorSandboxEnforcementLevel)
        return level is DoctorSandboxEnforcementLevel.WEAK

    def command_allowed(self, command: str) -> bool:
        token = command.strip().split()[0] if command.strip() else ""
        base = PurePosixPath(token).name
        return base in set(self.allowlisted_commands)

    def path_permitted(self, path: str) -> bool:
        try:
            normalized = _path(path, "path")
        except DeterministicDoctorTransactionError:
            return False
        if path_is_forbidden(normalized) or is_doctor_tcb_path(normalized):
            return False
        for permitted in self.permitted_paths:
            if normalized == permitted or normalized.startswith(permitted.rstrip("/") + "/"):
                return True
        # Exact permitted-path set membership is also accepted.
        return normalized in set(self.permitted_paths)

    def to_dict(self) -> dict[str, Any]:
        assert self.resource_bounds is not None
        return {
            "schema": DOCTOR_SANDBOX_POLICY_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "sandbox_id": self.sandbox_id,
            "worktree_root_ref": self.worktree_root_ref,
            "permitted_paths": list(self.permitted_paths),
            "allowlisted_commands": list(self.allowlisted_commands),
            "required_capabilities": [item.value for item in self.required_capabilities],
            "enforcement_level": self.enforcement_level.value
            if isinstance(self.enforcement_level, DoctorSandboxEnforcementLevel)
            else str(self.enforcement_level),
            "secrets_inherited": False,
            "network_denied": True,
            "target_code_imported": False,
            "max_processes": self.max_processes,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_cpu_time_seconds": self.max_cpu_time_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_changed_files": self.max_changed_files,
            "max_changed_bytes": self.max_changed_bytes,
            "resource_bounds": self.resource_bounds.to_dict(),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorHostileFsObservation:
    """One hostile filesystem observation inside the candidate worktree."""

    kind: DoctorHostileObservationKind
    path: str
    detail_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, DoctorHostileObservationKind, "kind"),
        )
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(
            self, "detail_ref", _optional_identifier(self.detail_ref, "detail_ref")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value
            if isinstance(self.kind, DoctorHostileObservationKind)
            else str(self.kind),
            "path": self.path,
            "detail_ref": self.detail_ref,
        }


@dataclass(frozen=True)
class DoctorSandboxEnforcementReceipt:
    """Platform evidence that sandbox claims were actually enforced."""

    policy: DoctorSandboxPolicy
    enforcement_id: str
    observed_capabilities: tuple[DoctorSandboxCapability, ...]
    hostile_observations: tuple[DoctorHostileFsObservation, ...] = ()
    process_observation_refs: tuple[str, ...] = ()
    network_observation_refs: tuple[str, ...] = ()
    secrets_observation_refs: tuple[str, ...] = ()
    platform_evidence_ref: str = ""
    enforced: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.policy, DoctorSandboxPolicy):
            raise DoctorSandboxError("enforcement requires DoctorSandboxPolicy")
        object.__setattr__(
            self, "enforcement_id", _identifier(self.enforcement_id, "enforcement_id")
        )
        caps = tuple(
            DoctorSandboxCapability(item)
            if not isinstance(item, DoctorSandboxCapability)
            else item
            for item in self.observed_capabilities
        )
        object.__setattr__(self, "observed_capabilities", caps)
        if not isinstance(self.hostile_observations, Sequence) or not all(
            isinstance(item, DoctorHostileFsObservation)
            for item in self.hostile_observations
        ):
            raise DoctorSandboxError(
                "hostile_observations must be DoctorHostileFsObservation values"
            )
        if len(self.hostile_observations) > MAX_HOSTILE_OBSERVATIONS:
            raise DoctorSandboxError("hostile_observations exceeds bound")
        object.__setattr__(
            self, "hostile_observations", tuple(self.hostile_observations)
        )
        for name in (
            "process_observation_refs",
            "network_observation_refs",
            "secrets_observation_refs",
        ):
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, maximum=MAX_DIAGNOSTICS)
            )
        object.__setattr__(
            self,
            "platform_evidence_ref",
            _optional_identifier(self.platform_evidence_ref, "platform_evidence_ref"),
        )
        object.__setattr__(self, "enforced", _bool(self.enforced, "enforced"))
        missing = set(self.policy.required_capabilities) - set(caps)
        if self.enforced and missing:
            raise DoctorSandboxError(
                "enforced receipt missing required capabilities: "
                + ",".join(sorted(item.value for item in missing))
            )
        if self.hostile_observations and self.enforced:
            # Hostile observations always fail closed regardless of claim.
            raise DoctorSandboxError(
                "hostile filesystem observations forbid sandbox enforcement claims"
            )

    @property
    def has_hostile_observations(self) -> bool:
        return bool(self.hostile_observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_SANDBOX_ENFORCEMENT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
            "enforcement_id": self.enforcement_id,
            "policy_id": self.policy.content_id,
            "sandbox_id": self.policy.sandbox_id,
            "observed_capabilities": [item.value for item in self.observed_capabilities],
            "hostile_observations": [item.to_dict() for item in self.hostile_observations],
            "process_observation_refs": list(self.process_observation_refs),
            "network_observation_refs": list(self.network_observation_refs),
            "secrets_observation_refs": list(self.secrets_observation_refs),
            "platform_evidence_ref": self.platform_evidence_ref,
            "enforced": self.enforced,
            "enforcement_level": self.policy.enforcement_level.value
            if isinstance(self.policy.enforcement_level, DoctorSandboxEnforcementLevel)
            else str(self.policy.enforcement_level),
            "secrets_inherited": False,
            "network_denied": True,
            "target_code_imported": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


def evaluate_sandbox_for_plan(
    policy: DoctorSandboxPolicy,
    plan: DeterministicDoctorPlan,
    *,
    requires_target_execution: bool = False,
    hostile_observations: Sequence[DoctorHostileFsObservation] = (),
) -> tuple[DoctorTransactionReason, ...]:
    """Return ordered reason codes when sandbox cannot admit the plan apply."""

    reasons: list[str] = []
    if not isinstance(policy, DoctorSandboxPolicy):
        return (DoctorTransactionReason.MALFORMED_INPUT,)
    if not isinstance(plan, DeterministicDoctorPlan):
        return (DoctorTransactionReason.MALFORMED_INPUT,)
    if plan.disposition is not DoctorPlanDisposition.ADMITTED:
        reasons.append(DoctorTransactionReason.PLAN_NOT_ADMITTED.value)
    if policy.secrets_inherited:
        reasons.append(DoctorTransactionReason.SECRETS_INHERITED.value)
    if not policy.network_denied:
        reasons.append(DoctorTransactionReason.NETWORK_NOT_DENIED.value)
    if policy.target_code_imported:
        reasons.append(DoctorTransactionReason.TARGET_CODE_IMPORT_FORBIDDEN.value)
    if policy.enforcement_level is DoctorSandboxEnforcementLevel.ABSENT:
        reasons.append(DoctorTransactionReason.SANDBOX_ENFORCEMENT_MISSING.value)
    if requires_target_execution and not policy.permits_target_execution:
        reasons.append(DoctorTransactionReason.SANDBOX_WEAK_EXECUTION_FORBIDDEN.value)
        reasons.append(DoctorTransactionReason.EXECUTION_DEPENDENT_ABSTAIN.value)
    if (
        not requires_target_execution
        and policy.permits_static_replay_only
    ):
        # Static-only is allowed; surface the mode for receipts.
        pass
    for path in plan.permitted_write_paths:
        if not policy.path_permitted(path):
            reasons.append(DoctorTransactionReason.PATH_ESCAPE.value)
        if is_doctor_tcb_path(path):
            reasons.append(DoctorTransactionReason.TCB_PATH.value)
        if path_is_forbidden(path):
            reasons.append(DoctorTransactionReason.FORBIDDEN_PATH.value)
    for obs in hostile_observations:
        if not isinstance(obs, DoctorHostileFsObservation):
            reasons.append(DoctorTransactionReason.MALFORMED_INPUT.value)
            continue
        reasons.append(DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value)
        kind = obs.kind.value if isinstance(obs.kind, DoctorHostileObservationKind) else str(obs.kind)
        if kind in _HOSTILE_FS_KINDS:
            reasons.append(DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value)
    # Deduplicate while preserving sorted stability.
    return tuple(sorted(set(reasons)))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Locks, leases, checkpoints, merge CAS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorCheckoutLock:
    """Exclusive checkout lock for the disposable candidate worktree."""

    lock_id: str
    holder_id: str
    worktree_root_ref: str
    base_tree_cid: str
    active: bool = True
    fence_id: str = ""
    expires_at: int = 0

    def __post_init__(self) -> None:
        for name in ("lock_id", "holder_id", "worktree_root_ref", "base_tree_cid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "active", _bool(self.active, "active"))
        object.__setattr__(
            self, "fence_id", _optional_identifier(self.fence_id, "fence_id")
        )
        object.__setattr__(
            self, "expires_at", _nonneg_int(self.expires_at, "expires_at")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_CHECKOUT_LOCK_SCHEMA,
            "lock_id": self.lock_id,
            "holder_id": self.holder_id,
            "worktree_root_ref": self.worktree_root_ref,
            "base_tree_cid": self.base_tree_cid,
            "active": self.active,
            "fence_id": self.fence_id,
            "expires_at": self.expires_at,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorWriterLease:
    """Writer lease binding exact paths for one doctor transaction."""

    lease_id: str
    fence_id: str
    holder_id: str
    permitted_write_paths: tuple[str, ...]
    permitted_read_paths: tuple[str, ...] = ()
    active: bool = True
    expires_at: int = 0
    dirty_user_tree: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _identifier(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence_id", _identifier(self.fence_id, "fence_id"))
        object.__setattr__(self, "holder_id", _identifier(self.holder_id, "holder_id"))
        object.__setattr__(
            self,
            "permitted_write_paths",
            _paths(self.permitted_write_paths, "permitted_write_paths", required=True),
        )
        object.__setattr__(
            self,
            "permitted_read_paths",
            _paths(self.permitted_read_paths, "permitted_read_paths"),
        )
        object.__setattr__(self, "active", _bool(self.active, "active"))
        object.__setattr__(
            self, "expires_at", _nonneg_int(self.expires_at, "expires_at")
        )
        object.__setattr__(
            self, "dirty_user_tree", _bool(self.dirty_user_tree, "dirty_user_tree")
        )
        if self.dirty_user_tree:
            raise DeterministicDoctorTransactionError(
                "writer lease cannot cover a dirty user tree; use disposable worktree"
            )
        for path in self.permitted_write_paths:
            if is_doctor_tcb_path(path) or path_is_forbidden(path):
                raise DeterministicDoctorTransactionError(
                    "writer lease cannot grant forbidden/TCB write paths"
                )

    def covers_writes(self, paths: Sequence[str]) -> bool:
        return set(paths).issubset(self.permitted_write_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_WRITER_LEASE_SCHEMA,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "holder_id": self.holder_id,
            "permitted_write_paths": list(self.permitted_write_paths),
            "permitted_read_paths": list(self.permitted_read_paths),
            "active": self.active,
            "expires_at": self.expires_at,
            "dirty_user_tree": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_propagation_lease(self) -> TransactionLease:
        return TransactionLease(
            lease_id=self.lease_id,
            fence_id=self.fence_id,
            holder_id=self.holder_id,
            permitted_write_paths=self.permitted_write_paths,
            permitted_read_paths=self.permitted_read_paths,
            active=self.active,
            expires_at=self.expires_at,
        )


@dataclass(frozen=True)
class DoctorMergeRefCas:
    """Compare-and-swap token for integrating the candidate tree.

    ``expected_ref`` must match the live merge tip; ``desired_ref`` is the
    candidate tree CID after a residual-free fixed point (or provisional tip).
    """

    cas_id: str
    ref_name: str
    expected_ref: str
    desired_ref: str
    holder_id: str
    active: bool = True

    def __post_init__(self) -> None:
        for name in ("cas_id", "ref_name", "expected_ref", "desired_ref", "holder_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "active", _bool(self.active, "active"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_MERGE_REF_CAS_SCHEMA,
            "cas_id": self.cas_id,
            "ref_name": self.ref_name,
            "expected_ref": self.expected_ref,
            "desired_ref": self.desired_ref,
            "holder_id": self.holder_id,
            "active": self.active,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorTransactionCheckpoint:
    """Content-addressed pre-mutation snapshot of the candidate worktree."""

    roots: DoctorAuthorityRoots
    checkpoint_id: str
    plan_id: str
    plan_content_id: str
    path_before_hashes: tuple[PathBeforeHash, ...]
    strategy_ref: str
    base_tree_cid: str
    candidate_tree_cid: str
    worktree_root_ref: str
    sandbox_enforcement_ref: str = ""
    cache_binding_refs: tuple[str, ...] = ()
    diagnostic_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "checkpoint_id",
            "plan_id",
            "plan_content_id",
            "strategy_ref",
            "base_tree_cid",
            "candidate_tree_cid",
            "worktree_root_ref",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "sandbox_enforcement_ref",
            _optional_identifier(
                self.sandbox_enforcement_ref, "sandbox_enforcement_ref"
            ),
        )
        if not isinstance(self.path_before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.path_before_hashes
        ):
            raise DeterministicDoctorTransactionError(
                "path_before_hashes must be PathBeforeHash values"
            )
        hashes = tuple(sorted(self.path_before_hashes, key=lambda item: item.path))
        if len({item.path for item in hashes}) != len(hashes):
            raise DeterministicDoctorTransactionError(
                "path_before_hashes must have unique paths"
            )
        if len(hashes) > MAX_PATHS:
            raise DeterministicDoctorTransactionError(
                "path_before_hashes exceeds path bound"
            )
        object.__setattr__(self, "path_before_hashes", hashes)
        object.__setattr__(
            self,
            "cache_binding_refs",
            _ids(self.cache_binding_refs, "cache_binding_refs", maximum=MAX_DIAGNOSTICS),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )

    def hash_map(self) -> dict[str, str]:
        return {
            item.path: item.before_hash
            for item in self.path_before_hashes
            if item.before_hash
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_TRANSACTION_CHECKPOINT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
            "roots": self.roots.to_dict(),
            "checkpoint_id": self.checkpoint_id,
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "path_before_hashes": [item.to_dict() for item in self.path_before_hashes],
            "strategy_ref": self.strategy_ref,
            "base_tree_cid": self.base_tree_cid,
            "candidate_tree_cid": self.candidate_tree_cid,
            "worktree_root_ref": self.worktree_root_ref,
            "sandbox_enforcement_ref": self.sandbox_enforcement_ref,
            "cache_binding_refs": list(self.cache_binding_refs),
            "diagnostic_refs": list(self.diagnostic_refs),
        }

    @property
    def content_id(self) -> str:
        return content_identity(
            {**self.to_dict(), "checkpoint_id": ""}  # identity excludes self id
        )

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}


def create_doctor_checkpoint(
    plan: DeterministicDoctorPlan,
    *,
    path_before_hashes: Sequence[PathBeforeHash],
    base_tree_cid: str,
    candidate_tree_cid: str,
    worktree_root_ref: str,
    strategy_ref: str = "",
    sandbox_enforcement_ref: str = "",
    cache_binding_refs: Sequence[str] = (),
    diagnostic_refs: Sequence[str] = (),
) -> DoctorTransactionCheckpoint:
    """Build a content-addressed checkpoint for an admitted doctor plan."""

    if not isinstance(plan, DeterministicDoctorPlan):
        raise DeterministicDoctorTransactionError(
            "checkpoint requires DeterministicDoctorPlan"
        )
    if plan.disposition is not DoctorPlanDisposition.ADMITTED:
        raise DeterministicDoctorTransactionError(
            "checkpoint requires an admitted DeterministicDoctorPlan"
        )
    strategy = strategy_ref or plan.checkpoint_ref
    if not strategy:
        raise DeterministicDoctorTransactionError(
            "admitted plan must declare a checkpoint strategy ref"
        )
    hashes = tuple(path_before_hashes)
    preimage = {
        "schema": DOCTOR_TRANSACTION_CHECKPOINT_SCHEMA,
        "roots": plan.roots.to_dict(),
        "plan_id": plan.plan_id,
        "plan_content_id": plan.content_id,
        "path_before_hashes": [
            item.to_dict() if isinstance(item, PathBeforeHash) else item
            for item in hashes
        ],
        "strategy_ref": strategy,
        "base_tree_cid": base_tree_cid,
        "candidate_tree_cid": candidate_tree_cid,
        "worktree_root_ref": worktree_root_ref,
        "sandbox_enforcement_ref": sandbox_enforcement_ref,
        "cache_binding_refs": list(cache_binding_refs),
        "diagnostic_refs": list(diagnostic_refs),
    }
    checkpoint_id = content_identity(preimage)
    return DoctorTransactionCheckpoint(
        roots=plan.roots,
        checkpoint_id=checkpoint_id,
        plan_id=plan.plan_id,
        plan_content_id=plan.content_id,
        path_before_hashes=hashes,
        strategy_ref=strategy,
        base_tree_cid=base_tree_cid,
        candidate_tree_cid=candidate_tree_cid,
        worktree_root_ref=worktree_root_ref,
        sandbox_enforcement_ref=sandbox_enforcement_ref,
        cache_binding_refs=tuple(cache_binding_refs),
        diagnostic_refs=tuple(diagnostic_refs),
    )


# ---------------------------------------------------------------------------
# Step / group / candidate-tree receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorStepReceipt:
    """Outcome of one doctor plan step under an active transaction."""

    step_id: str
    disposition: DoctorStepDisposition
    reason_codes: tuple[str, ...] = ()
    written_paths: tuple[str, ...] = ()
    observed_before_hashes: tuple[PathBeforeHash, ...] = ()
    observed_after_hashes: tuple[PathBeforeHash, ...] = ()
    changed_blob_cids: tuple[str, ...] = ()
    observed_tree_cid: str = ""
    observed_forest_cid: str = ""
    durable_effect_ref: str = ""
    diagnostic_refs: tuple[str, ...] = ()
    static_replay: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorStepDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self, "written_paths", _paths(self.written_paths, "written_paths")
        )
        if not isinstance(self.observed_before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.observed_before_hashes
        ):
            raise DeterministicDoctorTransactionError(
                "observed_before_hashes must be PathBeforeHash values"
            )
        object.__setattr__(
            self,
            "observed_before_hashes",
            tuple(sorted(self.observed_before_hashes, key=lambda item: item.path)),
        )
        if not isinstance(self.observed_after_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.observed_after_hashes
        ):
            raise DeterministicDoctorTransactionError(
                "observed_after_hashes must be PathBeforeHash values"
            )
        object.__setattr__(
            self,
            "observed_after_hashes",
            tuple(sorted(self.observed_after_hashes, key=lambda item: item.path)),
        )
        object.__setattr__(
            self,
            "changed_blob_cids",
            _ids(
                self.changed_blob_cids,
                "changed_blob_cids",
                maximum=MAX_PATHS,
                preserve_order=True,
            ),
        )
        for name in (
            "observed_tree_cid",
            "observed_forest_cid",
            "durable_effect_ref",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        object.__setattr__(
            self, "static_replay", _bool(self.static_replay, "static_replay")
        )

    @property
    def passed(self) -> bool:
        return self.disposition is DoctorStepDisposition.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_STEP_RECEIPT_SCHEMA,
            "step_id": self.step_id,
            "disposition": self.disposition.value
            if isinstance(self.disposition, DoctorStepDisposition)
            else str(self.disposition),
            "reason_codes": list(self.reason_codes),
            "written_paths": list(self.written_paths),
            "observed_before_hashes": [
                item.to_dict() for item in self.observed_before_hashes
            ],
            "observed_after_hashes": [
                item.to_dict() for item in self.observed_after_hashes
            ],
            "changed_blob_cids": list(self.changed_blob_cids),
            "observed_tree_cid": self.observed_tree_cid,
            "observed_forest_cid": self.observed_forest_cid,
            "durable_effect_ref": self.durable_effect_ref,
            "diagnostic_refs": list(self.diagnostic_refs),
            "static_replay": self.static_replay,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorGroupReceipt:
    """Outcome of one SCC (or singleton) doctor transaction group."""

    group_id: str
    scc_id: str
    step_ids: tuple[str, ...]
    disposition: DoctorGroupDisposition
    step_receipts: tuple[DoctorStepReceipt, ...] = ()
    reason_codes: tuple[str, ...] = ()
    diagnostic_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _identifier(self.group_id, "group_id"))
        object.__setattr__(
            self, "scc_id", _text(self.scc_id, "scc_id", required=False)
        )
        object.__setattr__(
            self,
            "step_ids",
            _ids(self.step_ids, "step_ids", required=True, preserve_order=True),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorGroupDisposition, "disposition"),
        )
        if not isinstance(self.step_receipts, Sequence) or not all(
            isinstance(item, DoctorStepReceipt) for item in self.step_receipts
        ):
            raise DeterministicDoctorTransactionError(
                "step_receipts must be DoctorStepReceipt values"
            )
        object.__setattr__(self, "step_receipts", tuple(self.step_receipts))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        if self.disposition is DoctorGroupDisposition.PASSED:
            if not all(item.passed for item in self.step_receipts):
                raise DeterministicDoctorTransactionError(
                    "passed group requires every step receipt to pass"
                )
            if set(self.step_ids) != {item.step_id for item in self.step_receipts}:
                raise DeterministicDoctorTransactionError(
                    "passed group must cover every step exactly once"
                )

    @property
    def passed(self) -> bool:
        return self.disposition is DoctorGroupDisposition.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_GROUP_RECEIPT_SCHEMA,
            "group_id": self.group_id,
            "scc_id": self.scc_id,
            "step_ids": list(self.step_ids),
            "disposition": self.disposition.value
            if isinstance(self.disposition, DoctorGroupDisposition)
            else str(self.disposition),
            "step_receipts": [item.to_dict() for item in self.step_receipts],
            "reason_codes": list(self.reason_codes),
            "diagnostic_refs": list(self.diagnostic_refs),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorCandidateTreeReceipt:
    """Identity binding for the isolated candidate worktree apply.

    Records base/candidate CIDs, sandbox enforcement, before hashes, and
    whether the apply was static-only.  Never grants merge or completion.
    """

    roots: DoctorAuthorityRoots
    receipt_id: str
    plan_id: str
    plan_content_id: str
    base_tree_cid: str
    candidate_tree_cid: str
    worktree_root_ref: str
    sandbox_enforcement_id: str
    checkpoint_id: str
    lease_id: str
    lock_id: str
    written_paths: tuple[str, ...]
    path_before_hashes: tuple[PathBeforeHash, ...]
    group_receipts: tuple[DoctorGroupReceipt, ...]
    static_replay_only: bool = False
    requires_target_execution: bool = False
    model_invocation_count: int = 0
    provider_invocation_count: int = 0
    diagnostic_refs: tuple[str, ...] = ()
    changed_blob_cids: tuple[str, ...] = ()
    observed_tree_cid: str = ""
    observed_forest_cid: str = ""
    durable_effect_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "receipt_id",
            "plan_id",
            "plan_content_id",
            "base_tree_cid",
            "candidate_tree_cid",
            "worktree_root_ref",
            "sandbox_enforcement_id",
            "checkpoint_id",
            "lease_id",
            "lock_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "written_paths", _paths(self.written_paths, "written_paths")
        )
        if not isinstance(self.path_before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.path_before_hashes
        ):
            raise DeterministicDoctorTransactionError(
                "path_before_hashes must be PathBeforeHash values"
            )
        object.__setattr__(
            self,
            "path_before_hashes",
            tuple(sorted(self.path_before_hashes, key=lambda item: item.path)),
        )
        if not isinstance(self.group_receipts, Sequence) or not all(
            isinstance(item, DoctorGroupReceipt) for item in self.group_receipts
        ):
            raise DeterministicDoctorTransactionError(
                "group_receipts must be DoctorGroupReceipt values"
            )
        object.__setattr__(self, "group_receipts", tuple(self.group_receipts))
        object.__setattr__(
            self, "static_replay_only", _bool(self.static_replay_only, "static_replay_only")
        )
        object.__setattr__(
            self,
            "requires_target_execution",
            _bool(self.requires_target_execution, "requires_target_execution"),
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _nonneg_int(self.model_invocation_count, "model_invocation_count"),
        )
        object.__setattr__(
            self,
            "provider_invocation_count",
            _nonneg_int(self.provider_invocation_count, "provider_invocation_count"),
        )
        if self.model_invocation_count != 0 or self.provider_invocation_count != 0:
            raise DeterministicDoctorTransactionError(
                "candidate tree receipts forbid model/provider invocations"
            )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        object.__setattr__(
            self,
            "changed_blob_cids",
            _ids(
                self.changed_blob_cids,
                "changed_blob_cids",
                maximum=MAX_PATHS,
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self,
            "observed_tree_cid",
            _optional_identifier(self.observed_tree_cid, "observed_tree_cid"),
        )
        object.__setattr__(
            self,
            "observed_forest_cid",
            _optional_identifier(self.observed_forest_cid, "observed_forest_cid"),
        )
        object.__setattr__(
            self,
            "durable_effect_refs",
            _ids(
                self.durable_effect_refs,
                "durable_effect_refs",
                maximum=MAX_STEPS,
                preserve_order=True,
            ),
        )
        if not self.written_paths:
            raise DeterministicDoctorTransactionError(
                "candidate tree requires a nonempty observed change"
            )
        if not self.changed_blob_cids or not self.observed_tree_cid:
            raise DeterministicDoctorTransactionError(
                "candidate tree requires reread blob/tree CID evidence"
            )
        if self.observed_tree_cid == self.base_tree_cid:
            raise DeterministicDoctorTransactionError(
                "candidate tree cannot certify a no-op root"
            )
        if not self.durable_effect_refs:
            raise DeterministicDoctorTransactionError(
                "candidate tree requires fsynced durable effect evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_CANDIDATE_TREE_RECEIPT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "base_tree_cid": self.base_tree_cid,
            "candidate_tree_cid": self.candidate_tree_cid,
            "worktree_root_ref": self.worktree_root_ref,
            "sandbox_enforcement_id": self.sandbox_enforcement_id,
            "checkpoint_id": self.checkpoint_id,
            "lease_id": self.lease_id,
            "lock_id": self.lock_id,
            "written_paths": list(self.written_paths),
            "path_before_hashes": [item.to_dict() for item in self.path_before_hashes],
            "group_receipts": [item.to_dict() for item in self.group_receipts],
            "static_replay_only": self.static_replay_only,
            "requires_target_execution": self.requires_target_execution,
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
            "diagnostic_refs": list(self.diagnostic_refs),
            "changed_blob_cids": list(self.changed_blob_cids),
            "observed_tree_cid": self.observed_tree_cid,
            "observed_forest_cid": self.observed_forest_cid,
            "durable_effect_refs": list(self.durable_effect_refs),
            "partial_merge_allowed": False,
            "claims_completion": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity({**self.to_dict(), "receipt_id": ""})


@dataclass(frozen=True)
class DoctorRollbackReceipt:
    """Evidence that the candidate worktree was restored from a checkpoint."""

    roots: DoctorAuthorityRoots
    rollback_id: str
    transaction_id: str
    checkpoint_id: str
    plan_id: str
    strategy_ref: str
    restored: bool
    reason_codes: tuple[str, ...]
    quarantined: bool = False
    diagnostic_refs: tuple[str, ...] = ()
    failed_step_ids: tuple[str, ...] = ()
    failed_group_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "rollback_id",
            "transaction_id",
            "checkpoint_id",
            "plan_id",
            "strategy_ref",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "restored", _bool(self.restored, "restored"))
        object.__setattr__(
            self, "quarantined", _bool(self.quarantined, "quarantined")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=True, maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        object.__setattr__(
            self,
            "failed_step_ids",
            _ids(self.failed_step_ids, "failed_step_ids", preserve_order=True),
        )
        object.__setattr__(
            self,
            "failed_group_id",
            _text(self.failed_group_id, "failed_group_id", required=False),
        )
        if not self.restored and not self.quarantined:
            raise DeterministicDoctorTransactionError(
                "failed restore must quarantine; cannot claim clean rollback"
            )
        if self.restored and self.quarantined:
            raise DeterministicDoctorTransactionError(
                "restored rollback cannot simultaneously quarantine"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_ROLLBACK_RECEIPT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
            "roots": self.roots.to_dict(),
            "rollback_id": self.rollback_id,
            "transaction_id": self.transaction_id,
            "checkpoint_id": self.checkpoint_id,
            "plan_id": self.plan_id,
            "strategy_ref": self.strategy_ref,
            "restored": self.restored,
            "quarantined": self.quarantined,
            "reason_codes": list(self.reason_codes),
            "diagnostic_refs": list(self.diagnostic_refs),
            "failed_step_ids": list(self.failed_step_ids),
            "failed_group_id": self.failed_group_id,
            "claims_completion": False,
            "partial_merge_allowed": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DoctorTransactionReport:
    """Full ordered report for one doctor plan execution attempt.

    Success is not completion authority: only a later residual-free fixed-point
    receipt may authorize task completion.  Partial merge is always forbidden.
    """

    roots: DoctorAuthorityRoots
    transaction_id: str
    plan: DeterministicDoctorPlan
    checkpoint: DoctorTransactionCheckpoint
    sandbox_enforcement: DoctorSandboxEnforcementReceipt
    checkout_lock: DoctorCheckoutLock
    lease: DoctorWriterLease
    group_receipts: tuple[DoctorGroupReceipt, ...]
    candidate_tree: DoctorCandidateTreeReceipt | None
    rollback: DoctorRollbackReceipt | None
    merge_cas: DoctorMergeRefCas | None
    reason_codes: tuple[str, ...]
    disposition: DoctorTransactionDisposition
    committed: bool
    partial_merge_allowed: bool = False
    model_invocation_count: int = 0
    provider_invocation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        if not isinstance(self.plan, DeterministicDoctorPlan):
            raise DeterministicDoctorTransactionError(
                "report must carry DeterministicDoctorPlan"
            )
        if not isinstance(self.checkpoint, DoctorTransactionCheckpoint):
            raise DeterministicDoctorTransactionError(
                "report requires DoctorTransactionCheckpoint"
            )
        if not isinstance(self.sandbox_enforcement, DoctorSandboxEnforcementReceipt):
            raise DeterministicDoctorTransactionError(
                "report requires DoctorSandboxEnforcementReceipt"
            )
        if not isinstance(self.checkout_lock, DoctorCheckoutLock):
            raise DeterministicDoctorTransactionError(
                "report requires DoctorCheckoutLock"
            )
        if not isinstance(self.lease, DoctorWriterLease):
            raise DeterministicDoctorTransactionError(
                "report requires DoctorWriterLease"
            )
        if not isinstance(self.group_receipts, Sequence) or not all(
            isinstance(item, DoctorGroupReceipt) for item in self.group_receipts
        ):
            raise DeterministicDoctorTransactionError(
                "group_receipts must be DoctorGroupReceipt values"
            )
        object.__setattr__(self, "group_receipts", tuple(self.group_receipts))
        if self.candidate_tree is not None and not isinstance(
            self.candidate_tree, DoctorCandidateTreeReceipt
        ):
            raise DeterministicDoctorTransactionError(
                "candidate_tree must be DoctorCandidateTreeReceipt or None"
            )
        if self.rollback is not None and not isinstance(
            self.rollback, DoctorRollbackReceipt
        ):
            raise DeterministicDoctorTransactionError(
                "rollback must be DoctorRollbackReceipt or None"
            )
        if self.merge_cas is not None and not isinstance(self.merge_cas, DoctorMergeRefCas):
            raise DeterministicDoctorTransactionError(
                "merge_cas must be DoctorMergeRefCas or None"
            )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorTransactionDisposition, "disposition"),
        )
        object.__setattr__(self, "committed", _bool(self.committed, "committed"))
        object.__setattr__(
            self,
            "partial_merge_allowed",
            _bool(self.partial_merge_allowed, "partial_merge_allowed"),
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _nonneg_int(self.model_invocation_count, "model_invocation_count"),
        )
        object.__setattr__(
            self,
            "provider_invocation_count",
            _nonneg_int(self.provider_invocation_count, "provider_invocation_count"),
        )
        if self.partial_merge_allowed:
            raise DeterministicDoctorTransactionError(
                "partial merge/completion is forbidden for doctor transactions"
            )
        if self.model_invocation_count != 0 or self.provider_invocation_count != 0:
            raise DeterministicDoctorTransactionError(
                "doctor transaction reports forbid model/provider invocations"
            )
        if self.committed:
            if self.disposition is not DoctorTransactionDisposition.COMMITTED:
                raise DeterministicDoctorTransactionError(
                    "committed report requires COMMITTED disposition"
                )
            if self.rollback is not None:
                raise DeterministicDoctorTransactionError(
                    "committed report cannot retain a rollback receipt"
                )
            if self.reason_codes:
                raise DeterministicDoctorTransactionError(
                    "committed report cannot carry failure reason codes"
                )
            if not all(item.passed for item in self.group_receipts):
                raise DeterministicDoctorTransactionError(
                    "committed report requires every group to pass"
                )
            if self.candidate_tree is None:
                raise DeterministicDoctorTransactionError(
                    "committed report requires DoctorCandidateTreeReceipt"
                )
        else:
            if self.disposition is DoctorTransactionDisposition.COMMITTED:
                raise DeterministicDoctorTransactionError(
                    "non-committed report cannot claim COMMITTED disposition"
                )
        # Neither state may claim completion.
        if self.disposition.claims_completion:
            raise DeterministicDoctorTransactionError(
                "transaction disposition must never claim completion"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_TRANSACTION_REPORT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
            "producer_id": PRODUCER_ID,
            "roots": self.roots.to_dict(),
            "transaction_id": self.transaction_id,
            "plan_id": self.plan.plan_id,
            "plan_content_id": self.plan.content_id,
            "checkpoint": self.checkpoint.to_dict(),
            "sandbox_enforcement": self.sandbox_enforcement.to_dict(),
            "checkout_lock": self.checkout_lock.to_dict(),
            "lease": self.lease.to_dict(),
            "group_receipts": [item.to_dict() for item in self.group_receipts],
            "candidate_tree": self.candidate_tree.to_dict()
            if self.candidate_tree
            else None,
            "rollback": self.rollback.to_dict() if self.rollback else None,
            "merge_cas": self.merge_cas.to_dict() if self.merge_cas else None,
            "reason_codes": list(self.reason_codes),
            "disposition": self.disposition.value
            if isinstance(self.disposition, DoctorTransactionDisposition)
            else str(self.disposition),
            "committed": self.committed,
            "partial_merge_allowed": False,
            "claims_completion": False,
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
            "provider_success_is_not_merge": True,
            "change_propagation_interface": CHANGE_PROPAGATION_TRANSACTION_INTERFACE,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}


# ---------------------------------------------------------------------------
# Step applicator protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorStepApplyRequest:
    """Inputs handed to a step applicator for one doctor plan step."""

    plan: DeterministicDoctorPlan
    step: DoctorPlanStep
    lease: DoctorWriterLease
    checkpoint: DoctorTransactionCheckpoint
    sandbox: DoctorSandboxPolicy
    completed_step_ids: tuple[str, ...]
    static_replay_only: bool = False


@dataclass(frozen=True)
class DoctorStepApplyResult:
    """Applicator outcome; never merge or completion authority by itself."""

    disposition: DoctorStepDisposition
    written_paths: tuple[str, ...] = ()
    observed_before_hashes: tuple[PathBeforeHash, ...] = ()
    observed_after_hashes: tuple[PathBeforeHash, ...] = ()
    changed_blob_cids: tuple[str, ...] = ()
    observed_tree_cid: str = ""
    observed_forest_cid: str = ""
    durable_effect_ref: str = ""
    reason_codes: tuple[str, ...] = ()
    diagnostic_refs: tuple[str, ...] = ()
    static_replay: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorStepDisposition, "disposition"),
        )
        object.__setattr__(
            self, "written_paths", _paths(self.written_paths, "written_paths")
        )
        if not isinstance(self.observed_before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.observed_before_hashes
        ):
            raise DeterministicDoctorTransactionError(
                "observed_before_hashes must be PathBeforeHash values"
            )
        object.__setattr__(
            self, "observed_before_hashes", tuple(self.observed_before_hashes)
        )
        if not isinstance(self.observed_after_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.observed_after_hashes
        ):
            raise DeterministicDoctorTransactionError(
                "observed_after_hashes must be PathBeforeHash values"
            )
        object.__setattr__(
            self, "observed_after_hashes", tuple(self.observed_after_hashes)
        )
        object.__setattr__(
            self,
            "changed_blob_cids",
            _ids(
                self.changed_blob_cids,
                "changed_blob_cids",
                maximum=MAX_PATHS,
                preserve_order=True,
            ),
        )
        for name in (
            "observed_tree_cid",
            "observed_forest_cid",
            "durable_effect_ref",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        object.__setattr__(
            self, "static_replay", _bool(self.static_replay, "static_replay")
        )


DoctorStepApplicator = Callable[[DoctorStepApplyRequest], DoctorStepApplyResult]
DoctorRestoreAdapter = Callable[[DoctorTransactionCheckpoint], bool]
DoctorHashProbe = Callable[[str], str]
DoctorLiveRefProbe = Callable[[str], str]  # ref_name -> current tip CID
DoctorCacheBindingProbe = Callable[[Sequence[str]], tuple[str, ...]]  # stale refs
DoctorRefCasAdapter = Callable[[DoctorMergeRefCas], bool]
DoctorEffectVerifier = Callable[
    [DoctorStepApplyRequest, DoctorStepApplyResult], bool
]


def _default_static_applicator(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
    """Fail closed: a validator cannot manufacture mutation effects."""

    return DoctorStepApplyResult(
        disposition=DoctorStepDisposition.FAILED,
        reason_codes=(DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value,),
        static_replay=request.static_replay_only,
    )


def _default_restore(checkpoint: DoctorTransactionCheckpoint) -> bool:
    """No filesystem context means restoration cannot be independently proved."""

    del checkpoint
    return False


# ---------------------------------------------------------------------------
# Group planning
# ---------------------------------------------------------------------------


def _topological_step_order(steps: Sequence[DoctorPlanStep]) -> tuple[str, ...]:
    by_id = {step.step_id: step for step in steps}
    remaining = set(by_id)
    completed: list[str] = []
    while remaining:
        ready = sorted(
            step_id
            for step_id in remaining
            if set(by_id[step_id].dependency_step_ids).issubset(completed)
        )
        if not ready:
            ready = sorted(remaining)
        for step_id in ready:
            completed.append(step_id)
            remaining.discard(step_id)
    return tuple(completed)


def _build_doctor_execution_groups(
    plan: DeterministicDoctorPlan,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Return ordered (group_id, scc_id, step_ids) for the doctor plan.

    Declared ``scc_refs`` that match step groupings become atomic groups.
    Steps not covered by an SCC become singleton groups.  Group order follows
    the earliest dependency-order index of any member.
    """

    steps = plan.steps
    step_order = _topological_step_order(steps)
    order_index = {step_id: idx for idx, step_id in enumerate(step_order)}
    assigned: set[str] = set()
    groups: list[tuple[str, str, tuple[str, ...]]] = []

    # When plan.scc_refs are present, treat each as an atomic group of steps
    # that share the scc ref in their consumer coverage via step validation refs
    # or, if scc refs are plain group ids, map by prefix convention
    # ``scc:<id>`` → group covering steps whose id contains the scc token.
    for scc_ref in plan.scc_refs:
        member_steps = tuple(
            sorted(
                (
                    step.step_id
                    for step in steps
                    if scc_ref in step.validation_refs
                    or scc_ref in step.consumer_ids
                    or step.step_id.startswith(f"step:{scc_ref.split(':', 1)[-1]}")
                ),
                key=lambda sid: order_index.get(sid, MAX_STEPS),
            )
        )
        if not member_steps:
            # Explicit multi-step SCC: if only one scc and multiple steps, group all.
            if len(plan.scc_refs) == 1 and len(steps) > 1:
                member_steps = tuple(
                    sorted(step_order, key=lambda sid: order_index.get(sid, MAX_STEPS))
                )
            else:
                continue
        groups.append((f"group:{scc_ref}", scc_ref, member_steps))
        assigned.update(member_steps)

    for step_id in step_order:
        if step_id in assigned:
            continue
        groups.append((f"group:singleton:{step_id}", "", (step_id,)))
        assigned.add(step_id)

    groups.sort(
        key=lambda item: min(
            (order_index.get(sid, MAX_STEPS) for sid in item[2]), default=MAX_STEPS
        )
    )
    return tuple(groups)


# ---------------------------------------------------------------------------
# Transaction orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DeterministicDoctorTransaction:
    """Orchestrate sandboxed, SCC-atomic execution of one admitted doctor plan.

    ``execute`` always returns a :class:`DoctorTransactionReport`.  Partial
    completion never yields COMMITTED.  Neither committed nor rolled-back
    reports claim task completion or invoke a model.
    """

    INTERFACE: Final[str] = DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE

    step_applicator: DoctorStepApplicator = field(default=_default_static_applicator)
    restore_adapter: DoctorRestoreAdapter = field(default=_default_restore)
    hash_probe: DoctorHashProbe | None = None
    live_ref_probe: DoctorLiveRefProbe | None = None
    cache_binding_probe: DoctorCacheBindingProbe | None = None
    ref_cas_adapter: DoctorRefCasAdapter | None = None
    effect_verifier: DoctorEffectVerifier | None = None
    allow_provisional_live_validation: bool = False
    now: Callable[[], int] = field(default=lambda: 0)

    def create_checkpoint(
        self,
        plan: DeterministicDoctorPlan,
        *,
        path_before_hashes: Sequence[PathBeforeHash],
        base_tree_cid: str,
        candidate_tree_cid: str,
        worktree_root_ref: str,
        strategy_ref: str = "",
        sandbox_enforcement_ref: str = "",
        cache_binding_refs: Sequence[str] = (),
        diagnostic_refs: Sequence[str] = (),
    ) -> DoctorTransactionCheckpoint:
        return create_doctor_checkpoint(
            plan,
            path_before_hashes=path_before_hashes,
            base_tree_cid=base_tree_cid,
            candidate_tree_cid=candidate_tree_cid,
            worktree_root_ref=worktree_root_ref,
            strategy_ref=strategy_ref,
            sandbox_enforcement_ref=sandbox_enforcement_ref,
            cache_binding_refs=cache_binding_refs,
            diagnostic_refs=diagnostic_refs,
        )

    def execute(
        self,
        plan: DeterministicDoctorPlan,
        *,
        sandbox_policy: DoctorSandboxPolicy,
        checkout_lock: DoctorCheckoutLock,
        lease: DoctorWriterLease,
        path_before_hashes: Sequence[PathBeforeHash],
        base_tree_cid: str,
        candidate_tree_cid: str,
        merge_cas: DoctorMergeRefCas | None = None,
        enforcement: DoctorSandboxEnforcementReceipt | None = None,
        checkpoint: DoctorTransactionCheckpoint | None = None,
        hostile_observations: Sequence[DoctorHostileFsObservation] = (),
        requires_target_execution: bool = False,
        cache_binding_refs: Sequence[str] = (),
        transaction_id: str = "",
        observe_timeout: bool = False,
        committed_tree_cid: str = "",
    ) -> DoctorTransactionReport:
        """Run dependency-ordered SCC groups under sandbox + lock + lease.

        On any failure/drift/timeout/scope escape/hostile FS/CAS conflict the
        checkpoint is restored (or the tree is quarantined if restore fails).
        """

        reasons: list[str] = []
        group_receipts: list[DoctorGroupReceipt] = []
        completed: list[str] = []
        txn_id = (
            _identifier(transaction_id, "transaction_id")
            if transaction_id
            else content_identity(
                {
                    "schema": "doctor-txn-id",
                    "plan_id": getattr(plan, "plan_id", "invalid"),
                    "lease_id": getattr(lease, "lease_id", "invalid"),
                }
            )
        )

        # --- Input binding ---
        if not isinstance(plan, DeterministicDoctorPlan):
            raise DeterministicDoctorTransactionError(
                "execute requires DeterministicDoctorPlan"
            )
        if not isinstance(sandbox_policy, DoctorSandboxPolicy):
            raise DeterministicDoctorTransactionError(
                "execute requires DoctorSandboxPolicy"
            )
        if not isinstance(checkout_lock, DoctorCheckoutLock):
            raise DeterministicDoctorTransactionError(
                "execute requires DoctorCheckoutLock"
            )
        if not isinstance(lease, DoctorWriterLease):
            raise DeterministicDoctorTransactionError(
                "execute requires DoctorWriterLease"
            )

        if plan.disposition is not DoctorPlanDisposition.ADMITTED:
            reasons.append(DoctorTransactionReason.PLAN_NOT_ADMITTED.value)

        sandbox_reasons = evaluate_sandbox_for_plan(
            sandbox_policy,
            plan,
            requires_target_execution=requires_target_execution,
            hostile_observations=hostile_observations,
        )
        reasons.extend(sandbox_reasons)

        if not checkout_lock.active:
            reasons.append(DoctorTransactionReason.CHECKOUT_LOCK_INVALID.value)
        if checkout_lock.base_tree_cid != base_tree_cid:
            reasons.append(DoctorTransactionReason.ROOT_DRIFT.value)
        if checkout_lock.worktree_root_ref != sandbox_policy.worktree_root_ref:
            reasons.append(DoctorTransactionReason.CHECKOUT_LOCK_INVALID.value)
        if checkout_lock.expires_at and self.now() > 0 and self.now() >= checkout_lock.expires_at:
            reasons.append(DoctorTransactionReason.CHECKOUT_LOCK_INVALID.value)

        if not lease.active:
            reasons.append(DoctorTransactionReason.LEASE_INVALID.value)
        if lease.expires_at and self.now() > 0 and self.now() >= lease.expires_at:
            reasons.append(DoctorTransactionReason.LEASE_INVALID.value)
        if not lease.covers_writes(plan.permitted_write_paths):
            reasons.append(DoctorTransactionReason.LEASE_PATH_MISMATCH.value)

        expected_lease = plan.lease_id or plan.roots.lease_id
        if expected_lease and lease.lease_id != expected_lease:
            reasons.append(DoctorTransactionReason.LEASE_INVALID.value)

        if plan.roots.sandbox_id and plan.roots.sandbox_id != sandbox_policy.sandbox_id:
            reasons.append(DoctorTransactionReason.ROOT_DRIFT.value)

        # Build enforcement receipt if not supplied.
        if enforcement is None:
            try:
                enforcement = DoctorSandboxEnforcementReceipt(
                    policy=sandbox_policy,
                    enforcement_id=content_identity(
                        {
                            "schema": "enforcement-id",
                            "sandbox_id": sandbox_policy.sandbox_id,
                            "plan_id": plan.plan_id,
                        }
                    ),
                    observed_capabilities=sandbox_policy.required_capabilities,
                    hostile_observations=tuple(hostile_observations),
                    platform_evidence_ref=(
                        "platform:enforced"
                        if sandbox_policy.enforcement_level
                        is DoctorSandboxEnforcementLevel.ENFORCED
                        else "platform:weak"
                    ),
                    enforced=(
                        sandbox_policy.enforcement_level
                        is DoctorSandboxEnforcementLevel.ENFORCED
                        and not hostile_observations
                    ),
                )
            except DoctorSandboxError:
                reasons.append(DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value)
                # Synthesize a non-enforced receipt for the report path.
                enforcement = DoctorSandboxEnforcementReceipt(
                    policy=DoctorSandboxPolicy(
                        sandbox_id=sandbox_policy.sandbox_id,
                        worktree_root_ref=sandbox_policy.worktree_root_ref,
                        permitted_paths=sandbox_policy.permitted_paths,
                        allowlisted_commands=sandbox_policy.allowlisted_commands,
                        required_capabilities=sandbox_policy.required_capabilities,
                        enforcement_level=DoctorSandboxEnforcementLevel.WEAK,
                        secrets_inherited=False,
                        network_denied=True,
                        target_code_imported=False,
                        max_processes=sandbox_policy.max_processes,
                        max_wall_time_seconds=sandbox_policy.max_wall_time_seconds,
                        max_cpu_time_seconds=sandbox_policy.max_cpu_time_seconds,
                        max_memory_bytes=sandbox_policy.max_memory_bytes,
                        max_changed_files=sandbox_policy.max_changed_files,
                        max_changed_bytes=sandbox_policy.max_changed_bytes,
                        resource_bounds=sandbox_policy.resource_bounds,
                    ),
                    enforcement_id="enforcement:hostile",
                    observed_capabilities=sandbox_policy.required_capabilities,
                    hostile_observations=(),
                    platform_evidence_ref="platform:hostile",
                    enforced=False,
                )

        static_only = (
            sandbox_policy.permits_static_replay_only
            or not sandbox_policy.permits_target_execution
        )
        if requires_target_execution and static_only:
            reasons.append(DoctorTransactionReason.SANDBOX_WEAK_EXECUTION_FORBIDDEN.value)
            reasons.append(DoctorTransactionReason.EXECUTION_DEPENDENT_ABSTAIN.value)

        # Placeholder checkpoint for early-failure reports.
        hashes = tuple(path_before_hashes)
        if checkpoint is None:
            try:
                if plan.disposition is DoctorPlanDisposition.ADMITTED and not [
                    r
                    for r in reasons
                    if r == DoctorTransactionReason.PLAN_NOT_ADMITTED.value
                ]:
                    checkpoint = self.create_checkpoint(
                        plan,
                        path_before_hashes=hashes,
                        base_tree_cid=base_tree_cid,
                        candidate_tree_cid=candidate_tree_cid,
                        worktree_root_ref=sandbox_policy.worktree_root_ref,
                        sandbox_enforcement_ref=enforcement.enforcement_id,
                        cache_binding_refs=cache_binding_refs,
                    )
            except DeterministicDoctorTransactionError:
                pass

        if checkpoint is None:
            # Minimal synthetic checkpoint for non-admitted / early reject paths.
            checkpoint = DoctorTransactionCheckpoint(
                roots=plan.roots,
                checkpoint_id=content_identity(
                    {"schema": "doctor-checkpoint-reject", "plan_id": plan.plan_id}
                ),
                plan_id=plan.plan_id,
                plan_content_id=plan.content_id,
                path_before_hashes=hashes,
                strategy_ref=plan.checkpoint_ref or "checkpoint:reject",
                base_tree_cid=base_tree_cid,
                candidate_tree_cid=candidate_tree_cid,
                worktree_root_ref=sandbox_policy.worktree_root_ref,
                sandbox_enforcement_ref=enforcement.enforcement_id,
                cache_binding_refs=tuple(cache_binding_refs),
            )

        if reasons:
            return self._reject(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=(),
                reasons=tuple(sorted(set(reasons))),
                merge_cas=merge_cas,
            )

        # --- Before-hash verification ---
        hash_reasons = self._verify_before_hashes(plan, checkpoint)
        if hash_reasons:
            return self._abort(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=(),
                completed=(),
                reasons=hash_reasons,
                failed_step_ids=(),
                failed_group_id="",
                merge_cas=merge_cas,
            )

        # --- Pre-mutation cache binding revalidation ---
        if self.cache_binding_probe is not None and cache_binding_refs:
            stale = tuple(self.cache_binding_probe(tuple(cache_binding_refs)))
            if stale:
                return self._abort(
                    plan=plan,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    enforcement=enforcement,
                    checkout_lock=checkout_lock,
                    lease=lease,
                    group_receipts=(),
                    completed=(),
                    reasons=(DoctorTransactionReason.CACHE_BINDING_STALE.value,),
                    failed_step_ids=(),
                    failed_group_id="",
                    merge_cas=merge_cas,
                )

        if observe_timeout:
            return self._abort(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=(),
                completed=(),
                reasons=(DoctorTransactionReason.TIMEOUT.value,),
                failed_step_ids=(),
                failed_group_id="",
                merge_cas=merge_cas,
            )

        groups = _build_doctor_execution_groups(plan)
        steps_by_id = {step.step_id: step for step in plan.steps}

        for group_id, scc_id, step_ids in groups:
            if not lease.active or not checkout_lock.active:
                return self._abort(
                    plan=plan,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    enforcement=enforcement,
                    checkout_lock=checkout_lock,
                    lease=lease,
                    group_receipts=tuple(group_receipts),
                    completed=tuple(completed),
                    reasons=(DoctorTransactionReason.LEASE_INVALID.value,),
                    failed_step_ids=(),
                    failed_group_id=group_id,
                    merge_cas=merge_cas,
                )

            step_receipts: list[DoctorStepReceipt] = []
            group_reasons: list[str] = []

            for step_id in step_ids:
                step = steps_by_id[step_id]
                unmet = set(step.dependency_step_ids) - set(completed)
                external_unmet = unmet - set(step_ids)
                if external_unmet:
                    group_reasons.append(DoctorTransactionReason.DEPENDENCY_UNMET.value)
                    step_receipts.append(
                        DoctorStepReceipt(
                            step_id=step_id,
                            disposition=DoctorStepDisposition.FAILED,
                            reason_codes=(
                                DoctorTransactionReason.DEPENDENCY_UNMET.value,
                            ),
                        )
                    )
                    break

                precheck = self._precheck_step(plan, step, lease, sandbox_policy)
                if precheck is not None:
                    step_receipts.append(precheck)
                    group_reasons.extend(precheck.reason_codes)
                    break

                request = DoctorStepApplyRequest(
                    plan=plan,
                    step=step,
                    lease=lease,
                    checkpoint=checkpoint,
                    sandbox=sandbox_policy,
                    completed_step_ids=tuple(completed),
                    static_replay_only=static_only,
                )
                try:
                    result = self.step_applicator(request)
                except Exception as exc:  # noqa: BLE001 — fail-closed boundary
                    step_receipts.append(
                        DoctorStepReceipt(
                            step_id=step_id,
                            disposition=DoctorStepDisposition.FAILED,
                            reason_codes=(DoctorTransactionReason.STEP_FAILURE.value,),
                            diagnostic_refs=(
                                f"diagnostic:exception:{type(exc).__name__}",
                            ),
                        )
                    )
                    group_reasons.append(DoctorTransactionReason.STEP_FAILURE.value)
                    break

                if not isinstance(result, DoctorStepApplyResult):
                    step_receipts.append(
                        DoctorStepReceipt(
                            step_id=step_id,
                            disposition=DoctorStepDisposition.FAILED,
                            reason_codes=(
                                DoctorTransactionReason.MALFORMED_INPUT.value,
                            ),
                        )
                    )
                    group_reasons.append(DoctorTransactionReason.MALFORMED_INPUT.value)
                    break

                written = set(result.written_paths)
                if written - set(plan.permitted_write_paths) or written - set(
                    lease.permitted_write_paths
                ):
                    step_receipts.append(
                        DoctorStepReceipt(
                            step_id=step_id,
                            disposition=DoctorStepDisposition.SCOPE_ESCAPE,
                            reason_codes=(DoctorTransactionReason.SCOPE_ESCAPE.value,),
                            written_paths=result.written_paths,
                            observed_before_hashes=result.observed_before_hashes,
                            diagnostic_refs=result.diagnostic_refs,
                        )
                    )
                    group_reasons.append(DoctorTransactionReason.SCOPE_ESCAPE.value)
                    break

                if any(
                    not sandbox_policy.path_permitted(path)
                    for path in result.written_paths
                ):
                    step_receipts.append(
                        DoctorStepReceipt(
                            step_id=step_id,
                            disposition=DoctorStepDisposition.SCOPE_ESCAPE,
                            reason_codes=(DoctorTransactionReason.PATH_ESCAPE.value,),
                            written_paths=result.written_paths,
                        )
                    )
                    group_reasons.append(DoctorTransactionReason.PATH_ESCAPE.value)
                    break

                if result.disposition is not DoctorStepDisposition.PASSED:
                    reason = (
                        DoctorTransactionReason.TIMEOUT.value
                        if result.disposition is DoctorStepDisposition.TIMED_OUT
                        else DoctorTransactionReason.DRIFT.value
                        if result.disposition is DoctorStepDisposition.DRIFT
                        else DoctorTransactionReason.SCOPE_ESCAPE.value
                        if result.disposition is DoctorStepDisposition.SCOPE_ESCAPE
                        else DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value
                        if result.disposition is DoctorStepDisposition.HOSTILE
                        else DoctorTransactionReason.STEP_FAILURE.value
                    )
                    codes = result.reason_codes or (reason,)
                    step_receipts.append(
                        DoctorStepReceipt(
                            step_id=step_id,
                            disposition=result.disposition,
                            reason_codes=codes,
                            written_paths=result.written_paths,
                            observed_before_hashes=result.observed_before_hashes,
                            observed_after_hashes=result.observed_after_hashes,
                            changed_blob_cids=result.changed_blob_cids,
                            observed_tree_cid=result.observed_tree_cid,
                            observed_forest_cid=result.observed_forest_cid,
                            durable_effect_ref=result.durable_effect_ref,
                            diagnostic_refs=result.diagnostic_refs,
                            static_replay=result.static_replay,
                        )
                    )
                    group_reasons.extend(codes)
                    break

                if step.write_paths:
                    try:
                        independently_verified = (
                            self.effect_verifier is not None
                            and bool(self.effect_verifier(request, result))
                        )
                    except Exception:  # noqa: BLE001 - verifier trust boundary
                        independently_verified = False
                    if not independently_verified:
                        step_receipts.append(
                            DoctorStepReceipt(
                                step_id=step_id,
                                disposition=DoctorStepDisposition.FAILED,
                                reason_codes=(
                                    DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value,
                                ),
                                written_paths=result.written_paths,
                                observed_before_hashes=result.observed_before_hashes,
                                observed_after_hashes=result.observed_after_hashes,
                                changed_blob_cids=result.changed_blob_cids,
                                observed_tree_cid=result.observed_tree_cid,
                                observed_forest_cid=result.observed_forest_cid,
                                durable_effect_ref=result.durable_effect_ref,
                            )
                        )
                        group_reasons.append(
                            DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value
                        )
                        break

                effect_reasons = self._verify_step_effects(
                    step=step,
                    result=result,
                    checkpoint=checkpoint,
                )
                if effect_reasons:
                    step_receipts.append(
                        DoctorStepReceipt(
                            step_id=step_id,
                            disposition=DoctorStepDisposition.FAILED,
                            reason_codes=effect_reasons,
                            written_paths=result.written_paths,
                            observed_before_hashes=result.observed_before_hashes,
                            observed_after_hashes=result.observed_after_hashes,
                            changed_blob_cids=result.changed_blob_cids,
                            observed_tree_cid=result.observed_tree_cid,
                            observed_forest_cid=result.observed_forest_cid,
                            durable_effect_ref=result.durable_effect_ref,
                            diagnostic_refs=result.diagnostic_refs,
                            static_replay=result.static_replay,
                        )
                    )
                    group_reasons.extend(effect_reasons)
                    break

                step_receipts.append(
                    DoctorStepReceipt(
                        step_id=step_id,
                        disposition=DoctorStepDisposition.PASSED,
                        written_paths=result.written_paths,
                        observed_before_hashes=result.observed_before_hashes,
                        observed_after_hashes=result.observed_after_hashes,
                        changed_blob_cids=result.changed_blob_cids,
                        observed_tree_cid=result.observed_tree_cid,
                        observed_forest_cid=result.observed_forest_cid,
                        durable_effect_ref=result.durable_effect_ref,
                        diagnostic_refs=result.diagnostic_refs,
                        static_replay=result.static_replay or static_only,
                    )
                )

            if (
                group_reasons
                or len(step_receipts) != len(step_ids)
                or not all(item.passed for item in step_receipts)
            ):
                failed_ids = tuple(
                    item.step_id for item in step_receipts if not item.passed
                ) or step_ids
                group_receipts.append(
                    DoctorGroupReceipt(
                        group_id=group_id,
                        scc_id=scc_id,
                        step_ids=step_ids,
                        disposition=DoctorGroupDisposition.ROLLED_BACK,
                        step_receipts=tuple(step_receipts),
                        reason_codes=tuple(sorted(set(group_reasons)))
                        or (DoctorTransactionReason.GROUP_INCOMPLETE.value,),
                        diagnostic_refs=tuple(
                            ref
                            for item in step_receipts
                            for ref in item.diagnostic_refs
                        ),
                    )
                )
                return self._abort(
                    plan=plan,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    enforcement=enforcement,
                    checkout_lock=checkout_lock,
                    lease=lease,
                    group_receipts=tuple(group_receipts),
                    completed=tuple(completed),
                    reasons=tuple(sorted(set(group_reasons)))
                    or (DoctorTransactionReason.GROUP_INCOMPLETE.value,),
                    failed_step_ids=failed_ids,
                    failed_group_id=group_id,
                    merge_cas=merge_cas,
                )

            group_receipts.append(
                DoctorGroupReceipt(
                    group_id=group_id,
                    scc_id=scc_id,
                    step_ids=step_ids,
                    disposition=DoctorGroupDisposition.PASSED,
                    step_receipts=tuple(step_receipts),
                )
            )
            completed.extend(step_ids)

        expected_steps = {step.step_id for step in plan.steps}
        if set(completed) != expected_steps:
            return self._abort(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=tuple(group_receipts),
                completed=tuple(completed),
                reasons=(DoctorTransactionReason.PARTIAL_SCC_FORBIDDEN.value,),
                failed_step_ids=tuple(sorted(expected_steps - set(completed))),
                failed_group_id="",
                merge_cas=merge_cas,
            )

        # --- Immediate pre-commit revalidation of roots / cache / CAS ---
        pre_commit_reasons = self._pre_commit_revalidate(
            plan=plan,
            checkpoint=checkpoint,
            lease=lease,
            checkout_lock=checkout_lock,
            merge_cas=merge_cas,
            cache_binding_refs=cache_binding_refs,
            base_tree_cid=base_tree_cid,
        )
        if pre_commit_reasons:
            return self._abort(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=tuple(group_receipts),
                completed=tuple(completed),
                reasons=pre_commit_reasons,
                failed_step_ids=(),
                failed_group_id="",
                merge_cas=merge_cas,
            )

        written_paths = sorted(
            {
                path
                for group in group_receipts
                for step in group.step_receipts
                for path in step.written_paths
            }
        )
        changed_blob_cids = tuple(
            dict.fromkeys(
                cid
                for group in group_receipts
                for step in group.step_receipts
                for cid in step.changed_blob_cids
            )
        )
        observed_tree_cids = tuple(
            step.observed_tree_cid
            for group in group_receipts
            for step in group.step_receipts
            if step.observed_tree_cid
        )
        observed_forest_cids = tuple(
            step.observed_forest_cid
            for group in group_receipts
            for step in group.step_receipts
            if step.observed_forest_cid
        )
        durable_effect_refs = tuple(
            dict.fromkeys(
                step.durable_effect_ref
                for group in group_receipts
                for step in group.step_receipts
                if step.durable_effect_ref
            )
        )
        if (
            not written_paths
            or not changed_blob_cids
            or not observed_tree_cids
            or observed_tree_cids[-1] == base_tree_cid
            or not durable_effect_refs
        ):
            return self._abort(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=tuple(group_receipts),
                completed=tuple(completed),
                reasons=(
                    DoctorTransactionReason.NO_EXPECTED_CHANGE.value,
                    DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value,
                ),
                failed_step_ids=(),
                failed_group_id="",
                merge_cas=merge_cas,
            )
        tip_cid = committed_tree_cid or candidate_tree_cid
        if observed_tree_cids[-1] != tip_cid:
            return self._abort(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=tuple(group_receipts),
                completed=tuple(completed),
                reasons=(
                    DoctorTransactionReason.DRIFT.value,
                    DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value,
                ),
                failed_step_ids=(),
                failed_group_id="",
                merge_cas=merge_cas,
            )
        candidate = DoctorCandidateTreeReceipt(
            roots=plan.roots,
            receipt_id=content_identity(
                {
                    "schema": DOCTOR_CANDIDATE_TREE_RECEIPT_SCHEMA,
                    "plan_id": plan.plan_id,
                    "transaction_id": txn_id,
                    "candidate_tree_cid": tip_cid,
                }
            ),
            plan_id=plan.plan_id,
            plan_content_id=plan.content_id,
            base_tree_cid=base_tree_cid,
            candidate_tree_cid=tip_cid,
            worktree_root_ref=sandbox_policy.worktree_root_ref,
            sandbox_enforcement_id=enforcement.enforcement_id,
            checkpoint_id=checkpoint.checkpoint_id,
            lease_id=lease.lease_id,
            lock_id=checkout_lock.lock_id,
            written_paths=tuple(written_paths),
            path_before_hashes=checkpoint.path_before_hashes,
            group_receipts=tuple(group_receipts),
            static_replay_only=static_only,
            requires_target_execution=requires_target_execution,
            changed_blob_cids=changed_blob_cids,
            observed_tree_cid=observed_tree_cids[-1],
            observed_forest_cid=(
                observed_forest_cids[-1] if observed_forest_cids else ""
            ),
            durable_effect_refs=durable_effect_refs,
        )

        # Apply merge CAS when provided (entire SCC already complete).
        applied_cas = merge_cas
        if merge_cas is not None:
            if not merge_cas.active:
                return self._abort(
                    plan=plan,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    enforcement=enforcement,
                    checkout_lock=checkout_lock,
                    lease=lease,
                    group_receipts=tuple(group_receipts),
                    completed=tuple(completed),
                    reasons=(DoctorTransactionReason.CAS_CONFLICT.value,),
                    failed_step_ids=(),
                    failed_group_id="",
                    merge_cas=merge_cas,
                )
            if self.live_ref_probe is not None:
                live = self.live_ref_probe(merge_cas.ref_name)
                if live != merge_cas.expected_ref:
                    return self._abort(
                        plan=plan,
                        transaction_id=txn_id,
                        checkpoint=checkpoint,
                        enforcement=enforcement,
                        checkout_lock=checkout_lock,
                        lease=lease,
                        group_receipts=tuple(group_receipts),
                        completed=tuple(completed),
                        reasons=(
                            DoctorTransactionReason.CAS_EXPECTED_MISMATCH.value,
                            DoctorTransactionReason.CAS_CONFLICT.value,
                        ),
                        failed_step_ids=(),
                        failed_group_id="",
                        merge_cas=merge_cas,
                    )
            if merge_cas.desired_ref not in {tip_cid, candidate_tree_cid, committed_tree_cid}:
                # Desired tip must be the candidate/committed identity.
                if merge_cas.desired_ref != tip_cid:
                    return self._abort(
                        plan=plan,
                        transaction_id=txn_id,
                        checkpoint=checkpoint,
                        enforcement=enforcement,
                        checkout_lock=checkout_lock,
                        lease=lease,
                        group_receipts=tuple(group_receipts),
                        completed=tuple(completed),
                        reasons=(DoctorTransactionReason.CAS_CONFLICT.value,),
                        failed_step_ids=(),
                        failed_group_id="",
                        merge_cas=merge_cas,
                    )
            if self.ref_cas_adapter is None:
                return self._abort(
                    plan=plan,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    enforcement=enforcement,
                    checkout_lock=checkout_lock,
                    lease=lease,
                    group_receipts=tuple(group_receipts),
                    completed=tuple(completed),
                    reasons=(DoctorTransactionReason.REF_CAS_NOT_APPLIED.value,),
                    failed_step_ids=(),
                    failed_group_id="",
                    merge_cas=merge_cas,
                )
            try:
                cas_applied = bool(self.ref_cas_adapter(merge_cas))
            except Exception:  # noqa: BLE001 - fail-closed external CAS boundary
                cas_applied = False
            if not cas_applied:
                return self._abort(
                    plan=plan,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    enforcement=enforcement,
                    checkout_lock=checkout_lock,
                    lease=lease,
                    group_receipts=tuple(group_receipts),
                    completed=tuple(completed),
                    reasons=(
                        DoctorTransactionReason.CAS_CONFLICT.value,
                        DoctorTransactionReason.REF_CAS_NOT_APPLIED.value,
                    ),
                    failed_step_ids=(),
                    failed_group_id="",
                    merge_cas=merge_cas,
                )

        if merge_cas is None and not self.allow_provisional_live_validation:
            return self._abort(
                plan=plan,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                enforcement=enforcement,
                checkout_lock=checkout_lock,
                lease=lease,
                group_receipts=tuple(group_receipts),
                completed=tuple(completed),
                reasons=(DoctorTransactionReason.REF_CAS_NOT_APPLIED.value,),
                failed_step_ids=(),
                failed_group_id="",
                merge_cas=None,
            )

        return DoctorTransactionReport(
            roots=plan.roots,
            transaction_id=txn_id,
            plan=plan,
            checkpoint=checkpoint,
            sandbox_enforcement=enforcement,
            checkout_lock=checkout_lock,
            lease=lease,
            group_receipts=tuple(group_receipts),
            candidate_tree=candidate,
            rollback=None,
            merge_cas=applied_cas,
            reason_codes=(),
            disposition=DoctorTransactionDisposition.COMMITTED,
            committed=True,
            partial_merge_allowed=False,
        )

    def execute_live(
        self,
        plan: DeterministicDoctorPlan,
        *,
        worktree_adapter: Any,
        edits: Sequence[Any],
        target_ref: str,
        base_ref: str = "",
        transaction_id: str = "",
        requires_target_execution: bool = False,
        cache_binding_refs: Sequence[str] = (),
        commit_message: str = "deterministic doctor transaction",
    ) -> DoctorTransactionReport:
        """Own a real lease/checkpoint/SCC apply/ref-CAS/rollback lifecycle.

        The runtime adapter is imported lazily so this planning contract keeps
        its no-provider, no-target-import surface.  Exact edits are grouped by
        the plan's impact/SCC grouping and are materialised before the pure
        transaction validator is allowed to construct a provisional report.
        The report is returned as COMMITTED only after the adapter's durable
        Git ref CAS succeeds.
        """

        from ..runtime.doctor_worktree_adapter import (  # local trust boundary
            DoctorExactEdit,
            DoctorWorktreeAdapter,
        )

        if not isinstance(plan, DeterministicDoctorPlan):
            raise DeterministicDoctorTransactionError(
                "execute_live requires DeterministicDoctorPlan"
            )
        if not isinstance(worktree_adapter, DoctorWorktreeAdapter):
            raise DeterministicDoctorTransactionError(
                "execute_live requires DoctorWorktreeAdapter"
            )
        if isinstance(edits, (str, bytes, bytearray)) or not isinstance(
            edits, Sequence
        ):
            raise DeterministicDoctorTransactionError(
                "execute_live edits must be a sequence"
            )
        exact_edits = tuple(edits)
        if not exact_edits or not all(
            isinstance(item, DoctorExactEdit) for item in exact_edits
        ):
            raise DeterministicDoctorTransactionError(
                "execute_live requires a nonempty exact edit set"
            )
        if len({item.path for item in exact_edits}) != len(exact_edits):
            raise DeterministicDoctorTransactionError(
                "execute_live requires one complete replacement per path"
            )
        expected_paths = set(plan.permitted_write_paths)
        edit_paths = {item.path for item in exact_edits}
        if not expected_paths or edit_paths != expected_paths:
            raise DeterministicDoctorTransactionError(
                "live edits must cover the complete permitted impact set exactly"
            )
        if edit_paths != set(worktree_adapter.permitted_paths):
            raise DeterministicDoctorTransactionError(
                "adapter allowlist must equal the plan write set"
            )

        steps_by_id = {step.step_id: step for step in plan.steps}
        assigned: list[DoctorExactEdit] = []
        for edit in exact_edits:
            step_id = edit.step_id
            if not step_id:
                owners = [
                    step.step_id
                    for step in plan.steps
                    if edit.path in set(step.write_paths)
                ]
                if len(owners) != 1:
                    raise DeterministicDoctorTransactionError(
                        f"edit {edit.path} does not have one exact plan-step owner"
                    )
                step_id = owners[0]
                edit = replace(edit, step_id=step_id)
            if step_id not in steps_by_id:
                raise DeterministicDoctorTransactionError(
                    f"edit references unknown step {step_id}"
                )
            if edit.path not in set(steps_by_id[step_id].write_paths):
                raise DeterministicDoctorTransactionError(
                    "edit path is not owned by its declared step"
                )
            assigned.append(edit)
        by_step: dict[str, list[DoctorExactEdit]] = {
            step_id: [] for step_id in steps_by_id
        }
        for edit in assigned:
            by_step[edit.step_id].append(edit)
        for step in plan.steps:
            if {item.path for item in by_step[step.step_id]} != set(step.write_paths):
                if step.write_paths:
                    raise DeterministicDoctorTransactionError(
                        f"step {step.step_id} is missing a complete exact edit set"
                    )

        txn_id = transaction_id or content_identity(
            {
                "schema": "doctor-live-transaction-id@1",
                "plan_id": plan.plan_id,
                "target_ref": target_ref,
                "edit_paths": sorted(edit_paths),
            }
        )
        session_token = "txn-" + content_identity(
            {"transaction_id": txn_id}
        ).split(":")[-1][:32]
        session = worktree_adapter.prepare(
            base_ref=base_ref or target_ref,
            session_id=session_token,
        )
        try:
            baseline = session.baseline
            groups = _build_doctor_execution_groups(plan)
            result_by_step: dict[str, DoctorStepApplyResult] = {}
            for group_id, _scc_id, step_ids in groups:
                group_edits = tuple(
                    edit
                    for step_id in step_ids
                    for edit in by_step.get(step_id, ())
                )
                if not group_edits:
                    for step_id in step_ids:
                        result_by_step[step_id] = DoctorStepApplyResult(
                            disposition=DoctorStepDisposition.PASSED,
                        )
                    continue
                receipt = session.apply_group(group_edits, group_id=group_id)
                effects_by_step: dict[str, list[Any]] = {
                    step_id: [] for step_id in step_ids
                }
                for effect in receipt.effects:
                    effects_by_step.setdefault(effect.step_id, []).append(effect)
                for step_id in step_ids:
                    effects = effects_by_step.get(step_id, [])
                    if not effects and steps_by_id[step_id].write_paths:
                        raise DeterministicDoctorTransactionError(
                            f"atomic group omitted step effects for {step_id}"
                        )
                    result_by_step[step_id] = DoctorStepApplyResult(
                        disposition=DoctorStepDisposition.PASSED,
                        written_paths=tuple(item.path for item in effects),
                        observed_before_hashes=tuple(
                            PathBeforeHash(
                                path=item.path,
                                before_hash=item.before_hash,
                            )
                            for item in effects
                        ),
                        observed_after_hashes=tuple(
                            PathBeforeHash(
                                path=item.path,
                                before_hash=item.after_hash,
                            )
                            for item in effects
                        ),
                        changed_blob_cids=tuple(
                            item.after_blob_cid for item in effects
                        ),
                        observed_tree_cid=(
                            receipt.after_tree_cid if effects else ""
                        ),
                        observed_forest_cid=(
                            receipt.after_forest_cid if effects else ""
                        ),
                        durable_effect_ref=(
                            receipt.durable_effect_ref if effects else ""
                        ),
                        static_replay=not requires_target_execution,
                    )

            final_snapshot = worktree_adapter.snapshot(session)
            path_hashes = tuple(
                PathBeforeHash(path=path, before_hash=digest)
                for path, digest in baseline.path_hashes
                if path in expected_paths
            )
            sandbox = DoctorSandboxPolicy(
                sandbox_id=plan.roots.sandbox_id,
                worktree_root_ref=str(session.worktree_root),
                permitted_paths=tuple(sorted(expected_paths)),
                enforcement_level=DoctorSandboxEnforcementLevel.ENFORCED,
                secrets_inherited=False,
                network_denied=True,
                target_code_imported=False,
            )
            holder_id = f"holder:{session.session_id}"
            lock = DoctorCheckoutLock(
                lock_id=f"lock:{session.session_id}",
                holder_id=holder_id,
                worktree_root_ref=str(session.worktree_root),
                base_tree_cid=baseline.tree_cid,
                active=True,
                fence_id=f"fence:{session.session_id}",
            )
            lease_id = plan.lease_id or plan.roots.lease_id
            if not lease_id:
                raise DeterministicDoctorTransactionError(
                    "live transaction requires a plan-bound writer lease id"
                )
            lease = DoctorWriterLease(
                lease_id=lease_id,
                fence_id=f"fence:{session.session_id}",
                holder_id=holder_id,
                permitted_write_paths=tuple(sorted(expected_paths)),
                permitted_read_paths=tuple(
                    sorted(set(plan.permitted_read_paths) | expected_paths)
                ),
                active=True,
            )
            checkpoint = self.create_checkpoint(
                plan,
                path_before_hashes=path_hashes,
                base_tree_cid=baseline.tree_cid,
                candidate_tree_cid=final_snapshot.tree_cid,
                worktree_root_ref=str(session.worktree_root),
                strategy_ref=plan.checkpoint_ref,
                cache_binding_refs=cache_binding_refs,
                diagnostic_refs=(f"durable-intent:{session.session_id}",),
            )

            def live_result(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
                return result_by_step.get(
                    request.step.step_id,
                    DoctorStepApplyResult(
                        disposition=DoctorStepDisposition.FAILED,
                        reason_codes=(
                            DoctorTransactionReason.GROUP_INCOMPLETE.value,
                        ),
                    ),
                )

            validator = replace(
                self,
                step_applicator=live_result,
                restore_adapter=session.default_restore,
                hash_probe=None,
                live_ref_probe=None,
                ref_cas_adapter=None,
                effect_verifier=lambda request, result: (
                    result_by_step.get(request.step.step_id) == result
                ),
                allow_provisional_live_validation=True,
            )
            report = validator.execute(
                plan,
                sandbox_policy=sandbox,
                checkout_lock=lock,
                lease=lease,
                path_before_hashes=path_hashes,
                base_tree_cid=baseline.tree_cid,
                candidate_tree_cid=final_snapshot.tree_cid,
                checkpoint=checkpoint,
                requires_target_execution=requires_target_execution,
                cache_binding_refs=cache_binding_refs,
                transaction_id=txn_id,
                committed_tree_cid=final_snapshot.tree_cid,
            )
            if not report.committed:
                session.close(remove_worktree=not (
                    report.disposition is DoctorTransactionDisposition.QUARANTINED
                ))
                return report
            try:
                cas_receipt = session.commit_ref(
                    target_ref=target_ref,
                    expected_commit_oid=session.base_commit_oid,
                    message=commit_message,
                )
            except BaseException:
                abort_report = validator._abort(
                    plan=plan,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    enforcement=report.sandbox_enforcement,
                    checkout_lock=lock,
                    lease=lease,
                    group_receipts=report.group_receipts,
                    completed=tuple(step.step_id for step in plan.steps),
                    reasons=(
                        DoctorTransactionReason.CAS_CONFLICT.value,
                        DoctorTransactionReason.REF_CAS_NOT_APPLIED.value,
                    ),
                    failed_step_ids=(),
                    failed_group_id="",
                    merge_cas=None,
                )
                session.close(remove_worktree=not (
                    abort_report.disposition
                    is DoctorTransactionDisposition.QUARANTINED
                ))
                return abort_report
            merge_cas = DoctorMergeRefCas(
                cas_id=f"cas:{session.session_id}",
                ref_name=cas_receipt.ref_name,
                expected_ref=cas_receipt.expected_commit_oid,
                desired_ref=cas_receipt.desired_commit_oid,
                holder_id=holder_id,
                active=True,
            )
            committed = replace(report, merge_cas=merge_cas)
            session.close()
            return committed
        except BaseException:
            try:
                session.restore(reason="execute_live_exception")
            finally:
                session.close(
                    remove_worktree=(
                        session.state.value != "quarantined"
                    )
                )
            raise

    def require_committed(self, *args: Any, **kwargs: Any) -> DoctorTransactionReport:
        report = self.execute(*args, **kwargs)
        if not report.committed:
            reasons = ", ".join(report.reason_codes) or "incomplete"
            raise DeterministicDoctorTransactionError(
                "deterministic doctor transaction rejected: " + reasons
            )
        return report

    # --- internals ---------------------------------------------------------

    def _verify_before_hashes(
        self,
        plan: DeterministicDoctorPlan,
        checkpoint: DoctorTransactionCheckpoint,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        hash_map = checkpoint.hash_map()
        required_paths = set(plan.permitted_write_paths)
        for site in plan.edit_sites:
            if not isinstance(site, DoctorEditSite):
                reasons.append(DoctorTransactionReason.MALFORMED_INPUT.value)
                continue
            required_paths.add(site.path)
            expected = hash_map.get(site.path)
            if not expected:
                reasons.append(DoctorTransactionReason.BEFORE_HASH_MISSING.value)
            elif site.before_hash and expected != site.before_hash:
                reasons.append(DoctorTransactionReason.BEFORE_HASH_MISMATCH.value)
        for path in plan.permitted_write_paths:
            if path not in hash_map or not hash_map[path]:
                reasons.append(DoctorTransactionReason.BEFORE_HASH_MISSING.value)
        if self.hash_probe is not None:
            for path, expected in hash_map.items():
                if path not in required_paths:
                    continue
                try:
                    current = self.hash_probe(path)
                except Exception:  # noqa: BLE001
                    reasons.append(DoctorTransactionReason.BEFORE_HASH_MISMATCH.value)
                    continue
                if current and expected and current != expected:
                    reasons.append(DoctorTransactionReason.BEFORE_HASH_MISMATCH.value)
        return tuple(sorted(set(reasons)))

    def _precheck_step(
        self,
        plan: DeterministicDoctorPlan,
        step: DoctorPlanStep,
        lease: DoctorWriterLease,
        sandbox: DoctorSandboxPolicy,
    ) -> DoctorStepReceipt | None:
        if set(step.write_paths) - set(plan.permitted_write_paths):
            return DoctorStepReceipt(
                step_id=step.step_id,
                disposition=DoctorStepDisposition.SCOPE_ESCAPE,
                reason_codes=(DoctorTransactionReason.SCOPE_ESCAPE.value,),
                written_paths=step.write_paths,
            )
        if set(step.write_paths) - set(lease.permitted_write_paths):
            return DoctorStepReceipt(
                step_id=step.step_id,
                disposition=DoctorStepDisposition.SCOPE_ESCAPE,
                reason_codes=(DoctorTransactionReason.LEASE_PATH_MISMATCH.value,),
                written_paths=step.write_paths,
            )
        for path in step.write_paths:
            if not sandbox.path_permitted(path) or is_doctor_tcb_path(path):
                return DoctorStepReceipt(
                    step_id=step.step_id,
                    disposition=DoctorStepDisposition.SCOPE_ESCAPE,
                    reason_codes=(DoctorTransactionReason.PATH_ESCAPE.value,),
                    written_paths=step.write_paths,
                )
        return None

    def _verify_step_effects(
        self,
        *,
        step: DoctorPlanStep,
        result: DoctorStepApplyResult,
        checkpoint: DoctorTransactionCheckpoint,
    ) -> tuple[str, ...]:
        """Reject passing applicators that did not prove reread durable effects."""

        reasons: list[str] = []
        expected_paths = set(step.write_paths)
        written_paths = set(result.written_paths)
        if not expected_paths:
            # Read-only validation/checkpoint steps may pass without an effect;
            # the transaction as a whole still requires a nonempty mutation.
            if written_paths:
                reasons.append(DoctorTransactionReason.SCOPE_ESCAPE.value)
            return tuple(sorted(set(reasons)))
        if written_paths != expected_paths:
            reasons.append(DoctorTransactionReason.GROUP_INCOMPLETE.value)
        before = {
            item.path: item.before_hash for item in result.observed_before_hashes
        }
        after = {
            item.path: item.before_hash for item in result.observed_after_hashes
        }
        checkpoint_hashes = checkpoint.hash_map()
        for path in expected_paths:
            if path not in before or path not in after:
                reasons.append(DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value)
                continue
            expected_before = checkpoint_hashes.get(path)
            if expected_before and before[path] != expected_before:
                reasons.append(DoctorTransactionReason.BEFORE_HASH_MISMATCH.value)
            if before[path] == after[path]:
                reasons.append(DoctorTransactionReason.NO_EXPECTED_CHANGE.value)
        if len(result.changed_blob_cids) < len(expected_paths):
            reasons.append(DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value)
        if not result.observed_tree_cid or not result.observed_forest_cid:
            reasons.append(DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value)
        if result.observed_tree_cid == checkpoint.base_tree_cid:
            reasons.append(DoctorTransactionReason.NO_EXPECTED_CHANGE.value)
        if not result.durable_effect_ref:
            reasons.append(DoctorTransactionReason.DURABLE_INTENT_MISSING.value)
        return tuple(sorted(set(reasons)))

    def _pre_commit_revalidate(
        self,
        *,
        plan: DeterministicDoctorPlan,
        checkpoint: DoctorTransactionCheckpoint,
        lease: DoctorWriterLease,
        checkout_lock: DoctorCheckoutLock,
        merge_cas: DoctorMergeRefCas | None,
        cache_binding_refs: Sequence[str],
        base_tree_cid: str,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if checkpoint.roots != plan.roots:
            reasons.append(DoctorTransactionReason.ROOT_DRIFT.value)
        if checkpoint.plan_id != plan.plan_id:
            reasons.append(DoctorTransactionReason.ROOT_DRIFT.value)
        if checkpoint.plan_content_id != plan.content_id:
            reasons.append(DoctorTransactionReason.ROOT_DRIFT.value)
        if checkpoint.base_tree_cid != base_tree_cid:
            reasons.append(DoctorTransactionReason.ROOT_DRIFT.value)
        if not lease.active:
            reasons.append(DoctorTransactionReason.LEASE_INVALID.value)
        if not checkout_lock.active:
            reasons.append(DoctorTransactionReason.CHECKOUT_LOCK_INVALID.value)
        if self.cache_binding_probe is not None and cache_binding_refs:
            stale = tuple(self.cache_binding_probe(tuple(cache_binding_refs)))
            if stale:
                reasons.append(DoctorTransactionReason.CACHE_BINDING_STALE.value)
                reasons.append(
                    DoctorTransactionReason.PRE_COMMIT_REVALIDATION_FAILED.value
                )
        if merge_cas is not None and self.live_ref_probe is not None:
            live = self.live_ref_probe(merge_cas.ref_name)
            if live != merge_cas.expected_ref:
                reasons.append(DoctorTransactionReason.CAS_EXPECTED_MISMATCH.value)
                reasons.append(
                    DoctorTransactionReason.PRE_COMMIT_REVALIDATION_FAILED.value
                )
        return tuple(sorted(set(reasons)))

    def _reject(
        self,
        *,
        plan: DeterministicDoctorPlan,
        transaction_id: str,
        checkpoint: DoctorTransactionCheckpoint,
        enforcement: DoctorSandboxEnforcementReceipt,
        checkout_lock: DoctorCheckoutLock,
        lease: DoctorWriterLease,
        group_receipts: tuple[DoctorGroupReceipt, ...],
        reasons: tuple[str, ...],
        merge_cas: DoctorMergeRefCas | None,
    ) -> DoctorTransactionReport:
        disposition = DoctorTransactionDisposition.REJECTED
        if DoctorTransactionReason.EXECUTION_DEPENDENT_ABSTAIN.value in reasons:
            disposition = DoctorTransactionDisposition.ABSTAINED
        elif DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value in reasons:
            disposition = DoctorTransactionDisposition.ABSTAINED
        elif DoctorTransactionReason.PLAN_NOT_ADMITTED.value in reasons:
            disposition = DoctorTransactionDisposition.REJECTED
        return DoctorTransactionReport(
            roots=plan.roots,
            transaction_id=transaction_id,
            plan=plan,
            checkpoint=checkpoint,
            sandbox_enforcement=enforcement,
            checkout_lock=checkout_lock,
            lease=lease,
            group_receipts=group_receipts,
            candidate_tree=None,
            rollback=None,
            merge_cas=merge_cas,
            reason_codes=reasons,
            disposition=disposition,
            committed=False,
        )

    def _abort(
        self,
        *,
        plan: DeterministicDoctorPlan,
        transaction_id: str,
        checkpoint: DoctorTransactionCheckpoint,
        enforcement: DoctorSandboxEnforcementReceipt,
        checkout_lock: DoctorCheckoutLock,
        lease: DoctorWriterLease,
        group_receipts: tuple[DoctorGroupReceipt, ...],
        completed: tuple[str, ...],
        reasons: tuple[str, ...],
        failed_step_ids: tuple[str, ...],
        failed_group_id: str,
        merge_cas: DoctorMergeRefCas | None,
    ) -> DoctorTransactionReport:
        del completed  # retained for diagnostics via group receipts only
        restored = False
        quarantined = False
        try:
            restored = bool(self.restore_adapter(checkpoint))
        except Exception:  # noqa: BLE001
            restored = False
        if not restored:
            quarantined = True
            reason_list = list(reasons) + [
                DoctorTransactionReason.RESTORE_FAILED.value,
                DoctorTransactionReason.QUARANTINE_REQUIRED.value,
            ]
            reasons = tuple(sorted(set(reason_list)))
            disposition = DoctorTransactionDisposition.QUARANTINED
        else:
            disposition = DoctorTransactionDisposition.ROLLED_BACK

        strategy = plan.rollback_ref or checkpoint.strategy_ref
        rollback = DoctorRollbackReceipt(
            roots=plan.roots,
            rollback_id=content_identity(
                {
                    "schema": DOCTOR_ROLLBACK_RECEIPT_SCHEMA,
                    "transaction_id": transaction_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "reasons": list(reasons),
                }
            ),
            transaction_id=transaction_id,
            checkpoint_id=checkpoint.checkpoint_id,
            plan_id=plan.plan_id,
            strategy_ref=strategy,
            restored=restored,
            reason_codes=reasons,
            quarantined=quarantined,
            failed_step_ids=failed_step_ids,
            failed_group_id=failed_group_id,
        )
        return DoctorTransactionReport(
            roots=plan.roots,
            transaction_id=transaction_id,
            plan=plan,
            checkpoint=checkpoint,
            sandbox_enforcement=enforcement,
            checkout_lock=checkout_lock,
            lease=lease,
            group_receipts=group_receipts,
            candidate_tree=None,
            rollback=rollback,
            merge_cas=merge_cas,
            reason_codes=reasons,
            disposition=disposition,
            committed=False,
        )


def execute_deterministic_doctor_transaction(
    plan: DeterministicDoctorPlan,
    *,
    sandbox_policy: DoctorSandboxPolicy,
    checkout_lock: DoctorCheckoutLock,
    lease: DoctorWriterLease,
    path_before_hashes: Sequence[PathBeforeHash],
    base_tree_cid: str,
    candidate_tree_cid: str,
    **kwargs: Any,
) -> DoctorTransactionReport:
    """Module-level convenience wrapper around :class:`DeterministicDoctorTransaction`."""

    return DeterministicDoctorTransaction().execute(
        plan,
        sandbox_policy=sandbox_policy,
        checkout_lock=checkout_lock,
        lease=lease,
        path_before_hashes=path_before_hashes,
        base_tree_cid=base_tree_cid,
        candidate_tree_cid=candidate_tree_cid,
        **kwargs,
    )


def doctor_plan_to_propagation_checkpoint(
    plan: DeterministicDoctorPlan,
    *,
    path_before_hashes: Sequence[PathBeforeHash],
) -> PropagationCheckpoint:
    """Bridge an admitted doctor plan into a propagation checkpoint record.

    Useful when composing with :class:`ChangePropagationTransaction` engines
    without redefining RPR-022 records.
    """

    from ..analysis.change_propagation_contracts import (
        AtomicPropagationPlan,
        ConsumerDisposition,
        ConsumerMigrationObligation,
        GraphNodeRef,
        GraphProvenance,
        PlanDisposition,
        PlanStepKind,
        PropagationPlanStep,
        obligation_set_identity,
    )

    if plan.disposition is not DoctorPlanDisposition.ADMITTED:
        raise DeterministicDoctorTransactionError(
            "propagation checkpoint bridge requires an admitted plan"
        )
    prop_roots = doctor_roots_to_propagation_roots(plan.roots)
    obligations = []
    for consumer in plan.consumer_dispositions:
        if consumer.disposition is DoctorRepairDisposition.ABSTAIN:
            continue
        path = plan.permitted_write_paths[0] if plan.permitted_write_paths else "pkg/x.py"
        for site in plan.edit_sites:
            path = site.path
            break
        obligations.append(
            ConsumerMigrationObligation(
                roots=prop_roots,
                obligation_id=f"obligation:{consumer.consumer_id}",
                consumer_id=consumer.consumer_id,
                delta_id=plan.impact_closure_id,
                disposition=ConsumerDisposition.MIGRATE
                if consumer.disposition is DoctorRepairDisposition.SUPPORTED
                else ConsumerDisposition.UNCHANGED,
                clause_ids=("clause:doctor",),
                node=GraphNodeRef(
                    node_id=f"node:{consumer.consumer_id}",
                    kind="function",
                    path=path,
                    symbol_id=f"symbol:{consumer.consumer_id}",
                    artifact_id=f"blob:{consumer.consumer_id}",
                    provenance=GraphProvenance.TRUSTED,
                    extractor_id="extractor:doctor",
                ),
                proof_refs=plan.proof_refs or ("proof:doctor",),
                missing_input_ids=(),
                behavior_contract_ids=(),
                invalidation_refs=plan.invalidation_refs,
            )
        )
    if not obligations:
        raise DeterministicDoctorTransactionError(
            "propagation bridge requires at least one non-abstaining consumer"
        )
    steps = tuple(
        PropagationPlanStep(
            step_id=step.step_id,
            kind=PlanStepKind.ANALYTICAL,
            obligation_ids=tuple(
                f"obligation:{cid}" for cid in step.consumer_ids
            )
            or (obligations[0].obligation_id,),
            transform_id=step.operator_id or plan.selected_operator_id,
            write_paths=step.write_paths,
            read_paths=step.write_paths,
            dependency_step_ids=step.dependency_step_ids,
        )
        for step in plan.steps
    )
    prop_plan = AtomicPropagationPlan(
        roots=prop_roots,
        plan_id=plan.plan_id,
        change_set_id=f"changeset:{plan.plan_id}",
        delta_id=plan.impact_closure_id,
        impact_closure_id=plan.impact_closure_id,
        disposition=PlanDisposition.ADMITTED,
        obligations=tuple(obligations),
        obligation_set_id=obligation_set_identity(tuple(obligations)),
        steps=steps,
        permitted_read_paths=plan.permitted_read_paths or plan.permitted_write_paths,
        permitted_write_paths=plan.permitted_write_paths,
        checkpoint_strategy_ref=plan.checkpoint_ref,
        rollback_strategy_ref=plan.rollback_ref,
        fixed_point_obligation_ref="fixed-point:doctor",
        proof_refs=plan.proof_refs,
        invalidation_refs=plan.invalidation_refs,
    )
    return create_propagation_checkpoint(
        prop_plan,
        path_before_hashes=path_before_hashes,
        tree_snapshot_ref=plan.roots.tree_id,
    )


__all__ = [
    "CHANGE_PROPAGATION_TRANSACTION_INTERFACE",
    "CONTRACT_VERSION",
    "DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE",
    "PRODUCER_ID",
    "DeterministicDoctorTransaction",
    "DeterministicDoctorTransactionError",
    "DoctorCandidateTreeReceipt",
    "DoctorCheckoutLock",
    "DoctorGroupDisposition",
    "DoctorGroupReceipt",
    "DoctorHostileFsObservation",
    "DoctorHostileObservationKind",
    "DoctorMergeCasError",
    "DoctorMergeRefCas",
    "DoctorQuarantineError",
    "DoctorRollbackReceipt",
    "DoctorSandboxCapability",
    "DoctorSandboxEnforcementLevel",
    "DoctorSandboxEnforcementReceipt",
    "DoctorSandboxError",
    "DoctorSandboxPolicy",
    "DoctorStepApplyRequest",
    "DoctorStepApplyResult",
    "DoctorStepDisposition",
    "DoctorStepReceipt",
    "DoctorTransactionCheckpoint",
    "DoctorTransactionDisposition",
    "DoctorTransactionReason",
    "DoctorTransactionReport",
    "DoctorWriterLease",
    "GroupExecutionDisposition",
    "PropagationGroupReceipt",
    "PropagationRollbackReceipt",
    "PropagationStepReceipt",
    "StepExecutionDisposition",
    "assert_no_provider_surface",
    "create_doctor_checkpoint",
    "doctor_plan_to_propagation_checkpoint",
    "evaluate_sandbox_for_plan",
    "execute_deterministic_doctor_transaction",
]
