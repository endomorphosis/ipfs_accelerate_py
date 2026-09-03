"""SPAR-037 subgoal refinement, extraction-wave task synthesis, and backlog repair.

This module extends current supervisor task synthesis with
``RefinedSubgoal@1`` and ``CompiledTask@1``.  It consumes SPAR-036 durable
goals/findings, SPAR-019 exact-scope/rollback contracts, and SPAR-035
identical-failure retry fingerprints, then refines them into bounded
inventory/extraction/state/boundary/façade/validation/repair/rescan/retirement
tasks with exact write paths, evidence, rollback, and deduplicated retries.

The refiner and compiler are nomination-only.  They cannot authorize a
transition, completion, façade retirement, or competing authority.  Vector,
model, and heuristic evidence cannot admit a task.  Generic prompts and
identical retries without new evidence are rejected.  Observational metadata
is excluded from identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .opportunity_detector import (
    ACCEPTANCE_IDS as SPAR036_ACCEPTANCE_IDS,
    AutonomyTier,
    DurableGoal,
    EXISTING_ADAPTER_AUTHORITIES as SPAR036_ADAPTER_AUTHORITIES,
    GOAL_COMPILATION_RECEIPT_INTERFACE as SPAR036_GOAL_RECEIPT_INTERFACE,
    GoalCompilationReceipt,
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
    MonolithFinding,
    OPPORTUNITY_DETECTION_RECEIPT_INTERFACE as SPAR036_DETECTION_RECEIPT_INTERFACE,
    OpportunityDetectionReceipt,
    RiskKind,
    compile_durable_goals,
)
from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_EXCLUDED,
)
from .transformation_packet import (
    DECLARED_ALLOWED_EFFECTS as SPAR019_ALLOWED_EFFECTS,
    DECLARED_FORBIDDEN_EFFECTS as SPAR019_FORBIDDEN_EFFECTS,
    ROLLBACK_MODE as SPAR019_ROLLBACK_MODE,
)


TASK_ID: Final[str] = "SPAR-037"
GOAL_ID: Final[str] = "SPAR-G071"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "task synthesis"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.task_compiler@1"
)
REFINER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring"
    ".partition_subgoal_refiner@1"
)
COMPILER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring"
    ".extraction_wave_task_compiler@1"
)

REFINED_SUBGOAL_INTERFACE: Final[str] = "RefinedSubgoal@1"
COMPILED_TASK_INTERFACE: Final[str] = "CompiledTask@1"
BACKLOG_REPAIR_DECISION_INTERFACE: Final[str] = "BacklogRepairDecision@1"
SUBGOAL_REFINEMENT_RECEIPT_INTERFACE: Final[str] = "SubgoalRefinementReceipt@1"
EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_INTERFACE: Final[str] = (
    "ExtractionWaveTaskCompilationReceipt@1"
)
PARTITION_SUBGOAL_REFINER_INTERFACE: Final[str] = "PartitionSubgoalRefiner@1"
EXTRACTION_WAVE_TASK_COMPILER_INTERFACE: Final[str] = (
    "ExtractionWaveTaskCompiler@1"
)

REFINED_SUBGOAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refined-subgoal@1"
)
COMPILED_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/compiled-task@1"
)
BACKLOG_REPAIR_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backlog-repair-decision@1"
)
SUBGOAL_REFINEMENT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/subgoal-refinement-receipt@1"
)
EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/extraction-wave-task-compilation-receipt@1"
)

TASK_CONTRACT_VERSION: Final[str] = "1"
POLICY_ID: Final[str] = "task-compiler-policy@1"
POLICY_REVISION: Final[str] = "1"

TASK_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
TASK_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
TASK_CAN_CREATE_AUTHORITY: Final[bool] = False
TASK_CAN_RETIRE_FACADE: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
PLAN_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
GENERIC_PROMPT_FORBIDDEN: Final[bool] = True
ONE_PACKET_AT_A_TIME: Final[bool] = True
IDENTICAL_RETRY_WITHOUT_EVIDENCE: Final[bool] = False

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_TASKS: Final[int] = 16_384
MAX_GOALS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_PATH_CHARS: Final[int] = 1_024
MAX_WAVE_TASKS: Final[int] = 64
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_FAILURES: Final[int] = 16_384

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS
assert IDENTITY_EXCLUDED_FIELDS == SPAR013_EXCLUDED

ACCEPTANCE_IDS: Final[tuple[str, ...]] = SPAR036_ACCEPTANCE_IDS
EXISTING_ADAPTER_AUTHORITIES: Final[tuple[str, ...]] = SPAR036_ADAPTER_AUTHORITIES
ROLLBACK_MODE: Final[str] = SPAR019_ROLLBACK_MODE
ALLOWED_EFFECTS: Final[tuple[str, ...]] = tuple(sorted(SPAR019_ALLOWED_EFFECTS))
FORBIDDEN_EFFECTS: Final[tuple[str, ...]] = tuple(sorted(SPAR019_FORBIDDEN_EFFECTS))

_FORBIDDEN_CAPSULE_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FunctionSemanticCapsule",
        "MethodSemanticCapsule",
        "ClassSemanticCapsule",
        "TopLevelBlockCapsule",
        "ModuleSemanticCapsule",
        "PackageSemanticCapsule",
        "CallsiteSemanticCapsule",
        "StateOwnerCapsule",
        "RegistrationCapsule",
        "ResourceLifecycleCapsule",
    }
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_GENERIC_PROMPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prompt",
        "generic_prompt",
        "fix_prompt",
        "prompt_body",
        "prompt_text",
        "task_prose",
    }
)

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "projection_is_authority",
    "can_retire_facade",
)

WAVE_KINDS: Final[tuple[str, ...]] = (
    "extraction",
    "state",
    "boundary",
    "facade",
)

ALWAYS_KINDS: Final[tuple[str, ...]] = (
    "inventory",
    "validation",
    "rescan",
)


class TaskCompilerError(ValueError):
    """Fail-closed violation of a SPAR-037 task-compiler contract."""


class TaskKind(str, Enum):
    INVENTORY = "inventory"
    EXTRACTION = "extraction"
    STATE = "state"
    BOUNDARY = "boundary"
    FACADE = "facade"
    VALIDATION = "validation"
    REPAIR = "repair"
    RESCAN = "rescan"
    RETIREMENT = "retirement"


TASK_KIND_ORDER: Final[tuple[str, ...]] = tuple(kind.value for kind in TaskKind)
DECLARED_TASK_KINDS: Final[frozenset[str]] = frozenset(TASK_KIND_ORDER)

RISK_TO_KINDS: Final[Mapping[str, tuple[str, ...]]] = {
    RiskKind.OVERSIZED_LOC.value: (TaskKind.EXTRACTION.value,),
    RiskKind.OVERSIZED_SCC.value: (TaskKind.EXTRACTION.value,),
    RiskKind.UNIQUE_STATE_OWNER.value: (TaskKind.STATE.value,),
    RiskKind.INITIALIZATION_ORDER.value: (TaskKind.STATE.value,),
    RiskKind.PUBLIC_COMPATIBILITY.value: (TaskKind.FACADE.value,),
    RiskKind.INCOMPLETE_CONTRACT.value: (TaskKind.BOUNDARY.value,),
    RiskKind.UNRESOLVED_FRONTIER.value: (TaskKind.INVENTORY.value,),
    RiskKind.OPAQUE_FRONTIER.value: (TaskKind.INVENTORY.value,),
}


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise TaskCompilerError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise TaskCompilerError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise TaskCompilerError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise TaskCompilerError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise TaskCompilerError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise TaskCompilerError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TaskCompilerError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise TaskCompilerError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise TaskCompilerError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise TaskCompilerError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise TaskCompilerError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise TaskCompilerError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise TaskCompilerError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise TaskCompilerError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise TaskCompilerError(f"{name} does not verify")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise TaskCompilerError(f"unknown {name}: {text}") from exc


def _project(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _project(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _project(to_dict())
    raise TaskCompilerError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise TaskCompilerError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TaskCompilerError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise TaskCompilerError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise TaskCompilerError(f"{name} must not contain duplicates")
    return ordered


def _unique_ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TaskCompilerError(f"{name} must be a list")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if text in seen:
            raise TaskCompilerError(f"{name} must not contain duplicates")
        seen.add(text)
        ordered.append(text)
    if len(ordered) > limit:
        raise TaskCompilerError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered: tuple[str, ...] = ()
    elif not isinstance(values, (list, tuple)):
        raise TaskCompilerError(f"{name} must be a list")
    else:
        ordered = tuple(sorted(_cid(item, name) for item in values))
    if len(ordered) != len(set(ordered)):
        raise TaskCompilerError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise TaskCompilerError(f"{name} exceeds maximum length")
    if required and not ordered:
        raise TaskCompilerError(f"{name} must not be empty")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name)
    if len(raw) > MAX_PATH_CHARS:
        raise TaskCompilerError(f"{name} exceeds path bound")
    normalized = raw.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or normalized in {".", ""}
        or normalized.startswith("./")
        or any(char in normalized for char in "*?[]{}")
        or "//" in normalized
        or normalized.endswith("/")
    ):
        raise TaskCompilerError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise TaskCompilerError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TaskCompilerError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise TaskCompilerError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise TaskCompilerError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TaskCompilerError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise TaskCompilerError(f"{name} exceeds command bound")
        if text in seen:
            raise TaskCompilerError(f"{name} must not contain duplicates")
        seen.add(text)
        ordered.append(text)
    if not ordered:
        raise TaskCompilerError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise TaskCompilerError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _acceptance_ids(values: Any) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered = ACCEPTANCE_IDS
    elif not isinstance(values, (list, tuple)):
        raise TaskCompilerError("acceptance_ids must be a list")
    else:
        ordered = tuple(_text(item, "acceptance_ids") for item in values)
    if tuple(sorted(ordered)) != tuple(sorted(ACCEPTANCE_IDS)):
        raise TaskCompilerError("task must include every acceptance identifier")
    if len(ordered) != len(set(ordered)):
        raise TaskCompilerError("acceptance_ids must not contain duplicates")
    if ordered != ACCEPTANCE_IDS:
        raise TaskCompilerError("acceptance identifiers must remain in canonical order")
    return ordered


def _kinds(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TaskCompilerError(f"{name} must be a list")
    ordered = tuple(_enum(item, TaskKind, name) for item in values)
    if len(ordered) != len(set(ordered)):
        raise TaskCompilerError(f"{name} must not contain duplicates")
    ranked = tuple(kind for kind in TASK_KIND_ORDER if kind in set(ordered))
    if ranked != ordered:
        raise TaskCompilerError(f"{name} must remain in canonical order")
    return ordered


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise TaskCompilerError(f"{name} cannot claim {flag}")


def task_compiler_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _reject_non_admitting(payload: Mapping[str, Any], name: str) -> None:
    present = _NON_ADMITTING_EVIDENCE & set(payload)
    if present:
        raise TaskCompilerError(
            f"{name} cannot admit tasks from {sorted(present)}"
        )
    prompts = _GENERIC_PROMPT_KEYS & set(payload)
    if prompts:
        raise TaskCompilerError(
            f"{name} cannot emit a generic prompt: {sorted(prompts)}"
        )


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


@dataclass(frozen=True, slots=True)
class TaskScope:
    """Exact module scope. Unrestricted or escaped paths fail closed."""

    module_id: str
    write_paths: Sequence[str]
    member_ids: Sequence[str]
    evidence_cids: Sequence[str]
    validation_commands: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "member_ids",
            _unique_sorted_text(list(self.member_ids), "member_ids", limit=MAX_MEMBERS),
        )
        if not self.member_ids:
            raise TaskCompilerError("scope member_ids must not be empty")
        object.__setattr__(
            self,
            "evidence_cids",
            _cids(list(self.evidence_cids), "evidence_cids", required=True),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _commands(list(self.validation_commands), "validation_commands"),
        )


def _coerce_scope(value: Mapping[str, Any] | TaskScope) -> TaskScope:
    if isinstance(value, TaskScope):
        return value
    payload = _mapping(value, "scope")
    return TaskScope(
        module_id=payload["module_id"],
        write_paths=payload["write_paths"],
        member_ids=payload["member_ids"],
        evidence_cids=payload["evidence_cids"],
        validation_commands=payload["validation_commands"],
    )


def _parse_scopes(values: Any) -> dict[str, TaskScope]:
    if not isinstance(values, (list, tuple)):
        raise TaskCompilerError("scopes must be a list")
    if not values:
        raise TaskCompilerError("scopes must not be empty; unrestricted scope is rejected")
    scopes: dict[str, TaskScope] = {}
    for item in values:
        scope = _coerce_scope(item)
        if scope.module_id in scopes:
            raise TaskCompilerError("duplicate scope module_id")
        scopes[scope.module_id] = scope
    return scopes


@dataclass(frozen=True, slots=True)
class PriorFailure:
    """Body-free identical-retry fingerprint from a previous attempt."""

    module_id: str
    kind: TaskKind | str
    write_paths: Sequence[str]
    evidence_cids: Sequence[str]
    failure_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(self, "kind", _enum(self.kind, TaskKind, "kind"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _cids(list(self.evidence_cids), "evidence_cids", required=True),
        )
        object.__setattr__(
            self, "failure_cid", _optional_cid(self.failure_cid, "failure_cid")
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "module_id": self.module_id,
            "write_paths": list(self.write_paths),
            "evidence_cids": list(self.evidence_cids),
        }

    @property
    def fingerprint(self) -> str:
        return cid_for_dag_json(self.fingerprint_payload())


def _parse_prior_failures(values: Any) -> tuple[PriorFailure, ...]:
    if values in (None, (), []):
        return ()
    if not isinstance(values, (list, tuple)):
        raise TaskCompilerError("prior_failures must be a list")
    if len(values) > MAX_FAILURES:
        raise TaskCompilerError("prior_failures exceed maximum length")
    failures = []
    for item in values:
        payload = _mapping(item, "prior_failures")
        failures.append(
            PriorFailure(
                module_id=payload["module_id"],
                kind=payload["kind"],
                write_paths=payload["write_paths"],
                evidence_cids=payload["evidence_cids"],
                failure_cid=payload.get("failure_cid", ""),
            )
        )
    fingerprints = [item.fingerprint for item in failures]
    if len(fingerprints) != len(set(fingerprints)):
        raise TaskCompilerError("prior_failures must not contain duplicates")
    return tuple(failures)


def retry_fingerprint(
    *,
    kind: str,
    module_id: str,
    write_paths: Sequence[str],
    evidence_cids: Sequence[str],
) -> str:
    payload = {
        "kind": kind,
        "module_id": module_id,
        "write_paths": list(write_paths),
        "evidence_cids": list(evidence_cids),
    }
    _require_dag_json(payload, "retry_fingerprint")
    return cid_for_dag_json(payload)


def _kinds_for_risks(risks: Sequence[str]) -> tuple[str, ...]:
    selected: set[str] = set(ALWAYS_KINDS)
    for risk in risks:
        extras = RISK_TO_KINDS.get(risk)
        if extras is None:
            raise TaskCompilerError(f"unknown risk: {risk}")
        selected.update(extras)
    if TaskKind.FACADE.value in selected:
        selected.add(TaskKind.RETIREMENT.value)
    return tuple(kind for kind in TASK_KIND_ORDER if kind in selected)


def _depends_on(kind: str, selected: Sequence[str]) -> tuple[str, ...]:
    selected_set = set(selected)
    predecessors = []
    for item in TASK_KIND_ORDER:
        if item == kind:
            break
        if item in selected_set:
            predecessors.append(item)
    return tuple(predecessors)


def _coerce_goal(value: DurableGoal | Mapping[str, Any]) -> DurableGoal:
    if isinstance(value, DurableGoal):
        return value
    if isinstance(value, Mapping):
        if "goal_cid" in value:
            return DurableGoal.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "goal_cid",
                "generic_prompt_forbidden",
                "can_authorize_transition",
                "can_authorize_completion",
                "can_create_authority",
            }
        }
        return DurableGoal(**payload)
    raise TaskCompilerError("goal must be a DurableGoal")


def _coerce_finding(value: MonolithFinding | Mapping[str, Any]) -> MonolithFinding:
    if isinstance(value, MonolithFinding):
        return value
    if isinstance(value, Mapping):
        if "finding_cid" in value:
            return MonolithFinding.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "finding_cid",
                "can_authorize_transition",
                "can_authorize_completion",
                "can_create_authority",
            }
        }
        return MonolithFinding(**payload)
    raise TaskCompilerError("finding must be a MonolithFinding")


def _coerce_goals(value: Any) -> tuple[DurableGoal, ...]:
    if isinstance(value, GoalCompilationReceipt):
        goals = value.goals
    elif isinstance(value, OpportunityDetectionReceipt):
        if not value.findings:
            raise TaskCompilerError("task compilation requires durable goals")
        goals = compile_durable_goals(value).goals
    elif isinstance(value, Mapping):
        interface = value.get("interface")
        if interface == SPAR036_GOAL_RECEIPT_INTERFACE:
            goals = GoalCompilationReceipt.from_dict(value).goals
        elif interface == SPAR036_DETECTION_RECEIPT_INTERFACE:
            detection = OpportunityDetectionReceipt.from_dict(value)
            if not detection.findings:
                raise TaskCompilerError("task compilation requires durable goals")
            goals = compile_durable_goals(detection).goals
        elif "goals" in value:
            return _coerce_goals(value["goals"])
        else:
            raise TaskCompilerError("task compilation requires durable goals")
    elif isinstance(value, (list, tuple)):
        if not value:
            raise TaskCompilerError("task compilation requires durable goals")
        first = value[0]
        treat_as_findings = isinstance(first, MonolithFinding) or (
            isinstance(first, Mapping)
            and "goal_cid" not in first
            and (
                first.get("interface") == "MonolithFinding@1"
                or "finding_cid" in first
                or "oversized" in first
            )
        )
        if treat_as_findings:
            findings = tuple(_coerce_finding(item) for item in value)
            goals = compile_durable_goals(findings).goals
        else:
            goals = tuple(_coerce_goal(item) for item in value)
    else:
        raise TaskCompilerError("task compilation requires durable goals")
    if not goals:
        raise TaskCompilerError("task compilation requires durable goals")
    if len(goals) > MAX_GOALS:
        raise TaskCompilerError("goals exceed maximum length")
    return tuple(goals)


@dataclass(frozen=True, slots=True)
class RefinedSubgoal:
    """One bounded subgoal refined from a durable goal. Nomination-only."""

    tree_id: str
    goal_cid: str
    finding_cid: str
    module_id: str
    risks: Sequence[str]
    task_kinds: Sequence[str]
    write_paths: Sequence[str]
    member_ids: Sequence[str]
    evidence_cids: Sequence[str]
    autonomy_tier: AutonomyTier | str
    blocked_retry_fingerprints: Sequence[str] = ()
    refiner_id: str = REFINER_ID

    interface: ClassVar[str] = REFINED_SUBGOAL_INTERFACE
    schema: ClassVar[str] = REFINED_SUBGOAL_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "goal_cid",
            "finding_cid",
            "module_id",
            "risks",
            "task_kinds",
            "write_paths",
            "member_ids",
            "evidence_cids",
            "autonomy_tier",
            "blocked_retry_fingerprints",
            "refiner_id",
            "generic_prompt_forbidden",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_retire_facade",
            "subgoal_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "goal_cid", _cid(self.goal_cid, "goal_cid"))
        object.__setattr__(self, "finding_cid", _cid(self.finding_cid, "finding_cid"))
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(
            self,
            "risks",
            tuple(sorted({_enum(item, RiskKind, "risks") for item in self.risks})),
        )
        if not self.risks:
            raise TaskCompilerError("refined subgoal must retain finding risks")
        object.__setattr__(self, "task_kinds", _kinds(list(self.task_kinds), "task_kinds"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "member_ids",
            _unique_sorted_text(list(self.member_ids), "member_ids", limit=MAX_MEMBERS),
        )
        if not self.member_ids:
            raise TaskCompilerError("subgoal member_ids must not be empty")
        object.__setattr__(
            self,
            "evidence_cids",
            _cids(list(self.evidence_cids), "evidence_cids", required=True),
        )
        object.__setattr__(
            self,
            "autonomy_tier",
            _enum(self.autonomy_tier, AutonomyTier, "autonomy_tier"),
        )
        object.__setattr__(
            self,
            "blocked_retry_fingerprints",
            _cids(
                list(self.blocked_retry_fingerprints),
                "blocked_retry_fingerprints",
                required=False,
            ),
        )
        if not self.task_kinds and not self.blocked_retry_fingerprints:
            raise TaskCompilerError("subgoal must emit tasks or blocked retries")
        refiner = _text(self.refiner_id, "refiner_id")
        if refiner != REFINER_ID:
            raise TaskCompilerError("refiner_id must remain the SPAR-037 refiner")
        object.__setattr__(self, "refiner_id", refiner)

    @property
    def generic_prompt_forbidden(self) -> bool:
        return True

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def can_retire_facade(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFINED_SUBGOAL_SCHEMA,
            "interface": REFINED_SUBGOAL_INTERFACE,
            "tree_id": self.tree_id,
            "goal_cid": self.goal_cid,
            "finding_cid": self.finding_cid,
            "module_id": self.module_id,
            "risks": list(self.risks),
            "task_kinds": list(self.task_kinds),
            "write_paths": list(self.write_paths),
            "member_ids": list(self.member_ids),
            "evidence_cids": list(self.evidence_cids),
            "autonomy_tier": self.autonomy_tier,
            "blocked_retry_fingerprints": list(self.blocked_retry_fingerprints),
            "refiner_id": self.refiner_id,
            "generic_prompt_forbidden": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_retire_facade": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def subgoal_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["subgoal_cid"] = self.subgoal_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefinedSubgoal":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("subgoal_cid")
        if payload.pop("schema") != REFINED_SUBGOAL_SCHEMA:
            raise TaskCompilerError("unsupported RefinedSubgoal schema")
        if payload.pop("interface") != REFINED_SUBGOAL_INTERFACE:
            raise TaskCompilerError("unsupported RefinedSubgoal interface")
        if payload.pop("generic_prompt_forbidden") is not True:
            raise TaskCompilerError("subgoal cannot emit a generic prompt")
        _pop_authority_flags(payload, "RefinedSubgoal")
        result = cls(**payload)
        _verify_cid(claimed, result.subgoal_cid, "RefinedSubgoal subgoal_cid")
        return result


def _coerce_subgoal(value: RefinedSubgoal | Mapping[str, Any]) -> RefinedSubgoal:
    if isinstance(value, RefinedSubgoal):
        return value
    if isinstance(value, Mapping):
        if "subgoal_cid" in value:
            return RefinedSubgoal.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "subgoal_cid",
                "generic_prompt_forbidden",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return RefinedSubgoal(**payload)
    raise TaskCompilerError("subgoal must be a RefinedSubgoal")


@dataclass(frozen=True, slots=True)
class BacklogRepairDecision:
    """Evidence-driven repair vs identical retry. Nomination-only."""

    module_id: str
    fingerprint: str
    prior_evidence_cids: Sequence[str]
    current_evidence_cids: Sequence[str]
    retry_blocked: bool
    repair_admitted: bool

    interface: ClassVar[str] = BACKLOG_REPAIR_DECISION_INTERFACE
    schema: ClassVar[str] = BACKLOG_REPAIR_DECISION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "module_id",
            "fingerprint",
            "prior_evidence_cids",
            "current_evidence_cids",
            "retry_blocked",
            "repair_admitted",
            "identical_retry_without_evidence",
            "can_authorize_completion",
            "decision_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(self, "fingerprint", _cid(self.fingerprint, "fingerprint"))
        object.__setattr__(
            self,
            "prior_evidence_cids",
            _cids(list(self.prior_evidence_cids), "prior_evidence_cids"),
        )
        object.__setattr__(
            self,
            "current_evidence_cids",
            _cids(list(self.current_evidence_cids), "current_evidence_cids", required=True),
        )
        retry_blocked = _bool(self.retry_blocked, "retry_blocked")
        repair_admitted = _bool(self.repair_admitted, "repair_admitted")
        if retry_blocked and repair_admitted:
            raise TaskCompilerError("blocked retry cannot admit repair")
        if repair_admitted and self.prior_evidence_cids == self.current_evidence_cids:
            raise TaskCompilerError("repair requires new evidence")
        if (
            retry_blocked
            and self.prior_evidence_cids
            and self.prior_evidence_cids != self.current_evidence_cids
        ):
            raise TaskCompilerError("new evidence cannot be an identical retry")
        object.__setattr__(self, "retry_blocked", retry_blocked)
        object.__setattr__(self, "repair_admitted", repair_admitted)

    @property
    def identical_retry_without_evidence(self) -> bool:
        return self.retry_blocked

    @property
    def can_authorize_completion(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BACKLOG_REPAIR_DECISION_SCHEMA,
            "interface": BACKLOG_REPAIR_DECISION_INTERFACE,
            "module_id": self.module_id,
            "fingerprint": self.fingerprint,
            "prior_evidence_cids": list(self.prior_evidence_cids),
            "current_evidence_cids": list(self.current_evidence_cids),
            "retry_blocked": self.retry_blocked,
            "repair_admitted": self.repair_admitted,
            "identical_retry_without_evidence": self.retry_blocked,
            "can_authorize_completion": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def decision_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["decision_cid"] = self.decision_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BacklogRepairDecision":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("decision_cid")
        if payload.pop("schema") != BACKLOG_REPAIR_DECISION_SCHEMA:
            raise TaskCompilerError("unsupported BacklogRepairDecision schema")
        if payload.pop("interface") != BACKLOG_REPAIR_DECISION_INTERFACE:
            raise TaskCompilerError("unsupported BacklogRepairDecision interface")
        if payload.pop("identical_retry_without_evidence") is not payload["retry_blocked"]:
            raise TaskCompilerError("identical retry flag does not match retry_blocked")
        if payload.pop("can_authorize_completion") is not False:
            raise TaskCompilerError("repair decision cannot claim can_authorize_completion")
        result = cls(**payload)
        _verify_cid(claimed, result.decision_cid, "BacklogRepairDecision decision_cid")
        return result


@dataclass(frozen=True, slots=True)
class CompiledTask:
    """One bounded inventory/extraction/... task. Nomination-only."""

    tree_id: str
    goal_cid: str
    finding_cid: str
    subgoal_cid: str
    module_id: str
    kind: TaskKind | str
    write_paths: Sequence[str]
    member_ids: Sequence[str]
    evidence_cids: Sequence[str]
    validation_commands: Sequence[str]
    depends_on: Sequence[str] = ()
    autonomy_tier: AutonomyTier | str = AutonomyTier.B
    acceptance_ids: Sequence[str] = ACCEPTANCE_IDS
    rollback_mode: str = ROLLBACK_MODE
    allowed_effects: Sequence[str] = ALLOWED_EFFECTS
    forbidden_effects: Sequence[str] = FORBIDDEN_EFFECTS
    compiler_id: str = COMPILER_ID

    interface: ClassVar[str] = COMPILED_TASK_INTERFACE
    schema: ClassVar[str] = COMPILED_TASK_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "goal_cid",
            "finding_cid",
            "subgoal_cid",
            "module_id",
            "kind",
            "write_paths",
            "member_ids",
            "evidence_cids",
            "validation_commands",
            "depends_on",
            "autonomy_tier",
            "acceptance_ids",
            "rollback_mode",
            "allowed_effects",
            "forbidden_effects",
            "compiler_id",
            "generic_prompt_forbidden",
            "one_packet_at_a_time",
            "unrestricted_scope",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_retire_facade",
            "retry_fingerprint",
            "task_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "goal_cid", _cid(self.goal_cid, "goal_cid"))
        object.__setattr__(self, "finding_cid", _cid(self.finding_cid, "finding_cid"))
        object.__setattr__(self, "subgoal_cid", _cid(self.subgoal_cid, "subgoal_cid"))
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(self, "kind", _enum(self.kind, TaskKind, "kind"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "member_ids",
            _unique_sorted_text(list(self.member_ids), "member_ids", limit=MAX_MEMBERS),
        )
        if not self.member_ids:
            raise TaskCompilerError("task member_ids must not be empty")
        object.__setattr__(
            self,
            "evidence_cids",
            _cids(list(self.evidence_cids), "evidence_cids", required=True),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _commands(list(self.validation_commands), "validation_commands"),
        )
        depends = _unique_ordered_text(list(self.depends_on), "depends_on", limit=len(TASK_KIND_ORDER))
        unknown = [item for item in depends if item not in DECLARED_TASK_KINDS]
        if unknown:
            raise TaskCompilerError(f"unknown depends_on kind: {unknown}")
        if self.kind in depends:
            raise TaskCompilerError("task cannot depend on its own kind")
        object.__setattr__(self, "depends_on", depends)
        object.__setattr__(
            self,
            "autonomy_tier",
            _enum(self.autonomy_tier, AutonomyTier, "autonomy_tier"),
        )
        object.__setattr__(self, "acceptance_ids", _acceptance_ids(self.acceptance_ids))
        mode = _text(self.rollback_mode, "rollback_mode")
        if mode != ROLLBACK_MODE:
            raise TaskCompilerError("rollback_mode must restore preimages or discard worktree")
        object.__setattr__(self, "rollback_mode", mode)
        allowed = _unique_sorted_text(
            list(self.allowed_effects), "allowed_effects", limit=len(ALLOWED_EFFECTS)
        )
        if set(allowed) != set(ALLOWED_EFFECTS):
            raise TaskCompilerError("allowed_effects must remain the SPAR-019 set")
        object.__setattr__(self, "allowed_effects", tuple(sorted(allowed)))
        forbidden = _unique_sorted_text(
            list(self.forbidden_effects),
            "forbidden_effects",
            limit=len(FORBIDDEN_EFFECTS),
        )
        if set(forbidden) != set(FORBIDDEN_EFFECTS):
            raise TaskCompilerError("forbidden_effects must remain the SPAR-019 set")
        object.__setattr__(self, "forbidden_effects", tuple(sorted(forbidden)))
        compiler = _text(self.compiler_id, "compiler_id")
        if compiler != COMPILER_ID:
            raise TaskCompilerError(
                "compiler_id must remain the SPAR-037 extraction-wave task compiler"
            )
        object.__setattr__(self, "compiler_id", compiler)
        if self.autonomy_tier == "A":
            raise TaskCompilerError("compiled tasks cannot claim Tier A")

    @property
    def generic_prompt_forbidden(self) -> bool:
        return True

    @property
    def one_packet_at_a_time(self) -> bool:
        return self.kind in WAVE_KINDS

    @property
    def unrestricted_scope(self) -> bool:
        return False

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def can_retire_facade(self) -> bool:
        return False

    @property
    def retry_fingerprint(self) -> str:
        return retry_fingerprint(
            kind=self.kind,
            module_id=self.module_id,
            write_paths=self.write_paths,
            evidence_cids=self.evidence_cids,
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": COMPILED_TASK_SCHEMA,
            "interface": COMPILED_TASK_INTERFACE,
            "tree_id": self.tree_id,
            "goal_cid": self.goal_cid,
            "finding_cid": self.finding_cid,
            "subgoal_cid": self.subgoal_cid,
            "module_id": self.module_id,
            "kind": self.kind,
            "write_paths": list(self.write_paths),
            "member_ids": list(self.member_ids),
            "evidence_cids": list(self.evidence_cids),
            "validation_commands": list(self.validation_commands),
            "depends_on": list(self.depends_on),
            "autonomy_tier": self.autonomy_tier,
            "acceptance_ids": list(self.acceptance_ids),
            "rollback_mode": self.rollback_mode,
            "allowed_effects": list(self.allowed_effects),
            "forbidden_effects": list(self.forbidden_effects),
            "compiler_id": self.compiler_id,
            "generic_prompt_forbidden": True,
            "one_packet_at_a_time": self.one_packet_at_a_time,
            "unrestricted_scope": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_retire_facade": False,
            "retry_fingerprint": self.retry_fingerprint,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def task_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["task_cid"] = self.task_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompiledTask":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("task_cid")
        claimed_fp = payload.pop("retry_fingerprint")
        if payload.pop("schema") != COMPILED_TASK_SCHEMA:
            raise TaskCompilerError("unsupported CompiledTask schema")
        if payload.pop("interface") != COMPILED_TASK_INTERFACE:
            raise TaskCompilerError("unsupported CompiledTask interface")
        if payload.pop("generic_prompt_forbidden") is not True:
            raise TaskCompilerError("task cannot emit a generic prompt")
        one_packet = payload.pop("one_packet_at_a_time")
        if payload.pop("unrestricted_scope") is not False:
            raise TaskCompilerError("task cannot claim unrestricted scope")
        _pop_authority_flags(payload, "CompiledTask")
        result = cls(**payload)
        if bool(one_packet) != result.one_packet_at_a_time:
            raise TaskCompilerError("one_packet_at_a_time does not match task kind")
        _verify_cid(claimed_fp, result.retry_fingerprint, "CompiledTask retry_fingerprint")
        _verify_cid(claimed, result.task_cid, "CompiledTask task_cid")
        return result


def _coerce_task(value: CompiledTask | Mapping[str, Any]) -> CompiledTask:
    if isinstance(value, CompiledTask):
        return value
    if isinstance(value, Mapping):
        if "task_cid" in value:
            return CompiledTask.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "task_cid",
                "retry_fingerprint",
                "generic_prompt_forbidden",
                "one_packet_at_a_time",
                "unrestricted_scope",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return CompiledTask(**payload)
    raise TaskCompilerError("task must be a CompiledTask")


def _refine_one(
    goal: DurableGoal,
    scope: TaskScope,
    prior: Sequence[PriorFailure],
) -> tuple[RefinedSubgoal, tuple[BacklogRepairDecision, ...]]:
    if goal.module_id != scope.module_id:
        raise TaskCompilerError("scope module_id does not match goal")
    kinds = list(_kinds_for_risks(goal.risks))
    prior_for_module = tuple(item for item in prior if item.module_id == goal.module_id)
    prior_by_kind = {item.kind: item for item in prior_for_module}
    identical = {
        item.fingerprint
        for item in prior_for_module
        if tuple(item.write_paths) == tuple(scope.write_paths)
        and tuple(item.evidence_cids) == tuple(scope.evidence_cids)
    }
    emitted: list[str] = []
    blocked: list[str] = []
    decisions: list[BacklogRepairDecision] = []
    for kind in kinds:
        fingerprint = retry_fingerprint(
            kind=kind,
            module_id=goal.module_id,
            write_paths=scope.write_paths,
            evidence_cids=scope.evidence_cids,
        )
        if fingerprint in identical:
            blocked.append(fingerprint)
            continue
        emitted.append(kind)

    has_prior = bool(prior_for_module)
    new_evidence = False
    prior_evidence: tuple[str, ...] = ()
    if has_prior:
        # New evidence relative to any prior attempt on this module/scope.
        new_evidence = any(
            tuple(item.evidence_cids) != tuple(scope.evidence_cids)
            or tuple(item.write_paths) != tuple(scope.write_paths)
            for item in prior_for_module
        )
        prior_evidence = tuple(
            sorted({cid for item in prior_for_module for cid in item.evidence_cids})
        )

    repair_fp = retry_fingerprint(
        kind=TaskKind.REPAIR.value,
        module_id=goal.module_id,
        write_paths=scope.write_paths,
        evidence_cids=scope.evidence_cids,
    )
    retry_blocked = has_prior and not new_evidence
    repair_admitted = has_prior and new_evidence and repair_fp not in identical
    if has_prior:
        decisions.append(
            BacklogRepairDecision(
                module_id=goal.module_id,
                fingerprint=repair_fp,
                prior_evidence_cids=prior_evidence,
                current_evidence_cids=scope.evidence_cids,
                retry_blocked=retry_blocked,
                repair_admitted=repair_admitted,
            )
        )
    if repair_admitted:
        if TaskKind.REPAIR.value not in emitted:
            emitted.append(TaskKind.REPAIR.value)
    elif has_prior and not new_evidence:
        blocked.append(repair_fp)
        for kind, item in prior_by_kind.items():
            fingerprint = retry_fingerprint(
                kind=kind,
                module_id=goal.module_id,
                write_paths=scope.write_paths,
                evidence_cids=scope.evidence_cids,
            )
            if fingerprint not in blocked:
                blocked.append(fingerprint)
            if kind in emitted:
                emitted.remove(kind)
    if (
        TaskKind.RETIREMENT.value in emitted
        and TaskKind.FACADE.value not in emitted
    ):
        emitted.remove(TaskKind.RETIREMENT.value)

    emitted_ordered = tuple(kind for kind in TASK_KIND_ORDER if kind in set(emitted))
    blocked_ordered = tuple(sorted(set(blocked)))
    subgoal = RefinedSubgoal(
        tree_id=goal.tree_id,
        goal_cid=goal.goal_cid,
        finding_cid=goal.finding_cid,
        module_id=goal.module_id,
        risks=goal.risks,
        task_kinds=emitted_ordered,
        write_paths=scope.write_paths,
        member_ids=scope.member_ids,
        evidence_cids=scope.evidence_cids,
        autonomy_tier=goal.autonomy_tier,
        blocked_retry_fingerprints=blocked_ordered,
    )
    return subgoal, tuple(decisions)


@dataclass(frozen=True, slots=True)
class SubgoalRefinementReceipt:
    """Refined subgoals for one tree. Nomination-only."""

    tree_id: str
    subgoals: Sequence[RefinedSubgoal | Mapping[str, Any]]
    repair_decisions: Sequence[BacklogRepairDecision | Mapping[str, Any]] = ()
    refiner_id: str = REFINER_ID

    interface: ClassVar[str] = SUBGOAL_REFINEMENT_RECEIPT_INTERFACE
    schema: ClassVar[str] = SUBGOAL_REFINEMENT_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "subgoals",
            "repair_decisions",
            "refiner_id",
            "subgoal_cids",
            "blocked_retry_fingerprints",
            "generic_prompt_forbidden",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_retire_facade",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        refiner = _text(self.refiner_id, "refiner_id")
        if refiner != REFINER_ID:
            raise TaskCompilerError("refiner_id must remain the SPAR-037 refiner")
        subgoals = tuple(_coerce_subgoal(item) for item in self.subgoals)
        if not subgoals:
            raise TaskCompilerError("refinement requires durable goals")
        if len(subgoals) > MAX_GOALS:
            raise TaskCompilerError("subgoals exceed maximum length")
        tree_id = _tree_id(self.tree_id)
        mismatched = [item.module_id for item in subgoals if item.tree_id != tree_id]
        if mismatched:
            raise TaskCompilerError("subgoal tree_id does not match receipt")
        ids = [item.subgoal_cid for item in subgoals]
        if len(ids) != len(set(ids)):
            raise TaskCompilerError("duplicate subgoal identity")
        goals = [item.goal_cid for item in subgoals]
        if len(goals) != len(set(goals)):
            raise TaskCompilerError("duplicate refined goal identity")
        decisions = tuple(
            item
            if isinstance(item, BacklogRepairDecision)
            else BacklogRepairDecision.from_dict(item)
            if isinstance(item, Mapping) and "decision_cid" in item
            else BacklogRepairDecision(**_project(item))
            for item in self.repair_decisions
        )
        decision_ids = [item.decision_cid for item in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise TaskCompilerError("duplicate repair decision identity")
        subgoals = tuple(sorted(subgoals, key=lambda item: item.subgoal_cid))
        decisions = tuple(sorted(decisions, key=lambda item: item.decision_cid))
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "subgoals", subgoals)
        object.__setattr__(self, "repair_decisions", decisions)
        object.__setattr__(self, "refiner_id", refiner)

    @property
    def subgoal_cids(self) -> tuple[str, ...]:
        return tuple(item.subgoal_cid for item in self.subgoals)

    @property
    def blocked_retry_fingerprints(self) -> tuple[str, ...]:
        fingerprints = [
            fingerprint
            for item in self.subgoals
            for fingerprint in item.blocked_retry_fingerprints
        ]
        return tuple(sorted(set(fingerprints)))

    @property
    def generic_prompt_forbidden(self) -> bool:
        return True

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def can_retire_facade(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SUBGOAL_REFINEMENT_RECEIPT_SCHEMA,
            "interface": SUBGOAL_REFINEMENT_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "subgoals": [item.to_dict() for item in self.subgoals],
            "repair_decisions": [item.to_dict() for item in self.repair_decisions],
            "refiner_id": self.refiner_id,
            "subgoal_cids": list(self.subgoal_cids),
            "blocked_retry_fingerprints": list(self.blocked_retry_fingerprints),
            "generic_prompt_forbidden": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_retire_facade": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubgoalRefinementReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SUBGOAL_REFINEMENT_RECEIPT_SCHEMA:
            raise TaskCompilerError("unsupported SubgoalRefinementReceipt schema")
        if payload.pop("interface") != SUBGOAL_REFINEMENT_RECEIPT_INTERFACE:
            raise TaskCompilerError("unsupported SubgoalRefinementReceipt interface")
        if payload.pop("generic_prompt_forbidden") is not True:
            raise TaskCompilerError("refinement cannot emit a generic prompt")
        _pop_authority_flags(payload, "SubgoalRefinementReceipt")
        payload.pop("subgoal_cids")
        payload.pop("blocked_retry_fingerprints")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "SubgoalRefinementReceipt receipt_cid"
        )
        return result


def refine_partition_subgoals(
    evidence: Mapping[str, Any]
    | GoalCompilationReceipt
    | OpportunityDetectionReceipt,
    *,
    scopes: Sequence[Mapping[str, Any] | TaskScope] | None = None,
    prior_failures: Sequence[Mapping[str, Any]] | None = None,
) -> SubgoalRefinementReceipt:
    """Refine SPAR-036 goals into bounded subgoals. Nomination-only."""

    if isinstance(evidence, (GoalCompilationReceipt, OpportunityDetectionReceipt)):
        payload: dict[str, Any] = {
            "tree_id": evidence.tree_id,
            "goals": evidence,
        }
        if scopes is not None:
            payload["scopes"] = list(scopes)
        if prior_failures is not None:
            payload["prior_failures"] = list(prior_failures)
    elif isinstance(evidence, Mapping):
        payload = _mapping(evidence, "evidence")
        if scopes is not None:
            payload = {**payload, "scopes": list(scopes)}
        if prior_failures is not None:
            payload = {**payload, "prior_failures": list(prior_failures)}
    else:
        raise TaskCompilerError("refinement requires durable goals")
    _reject_non_admitting(payload, "evidence")
    tree_id = _tree_id(payload.get("tree_id", ""))
    goals = _coerce_goals(payload.get("goals", payload))
    mismatched = [item.module_id for item in goals if item.tree_id != tree_id]
    if mismatched:
        raise TaskCompilerError("goal tree_id does not match evidence")
    scope_map = _parse_scopes(payload.get("scopes"))
    missing = sorted({item.module_id for item in goals} - set(scope_map))
    if missing:
        raise TaskCompilerError(
            f"missing exact scope for modules: {missing}; unrestricted scope is rejected"
        )
    extra = sorted(set(scope_map) - {item.module_id for item in goals})
    if extra:
        raise TaskCompilerError(f"unknown scope module_id: {extra}")
    prior = _parse_prior_failures(payload.get("prior_failures"))
    subgoals: list[RefinedSubgoal] = []
    decisions: list[BacklogRepairDecision] = []
    for goal in goals:
        subgoal, goal_decisions = _refine_one(goal, scope_map[goal.module_id], prior)
        subgoals.append(subgoal)
        decisions.extend(goal_decisions)
    return SubgoalRefinementReceipt(
        tree_id=tree_id,
        subgoals=subgoals,
        repair_decisions=decisions,
    )


class PartitionSubgoalRefiner:
    """SPAR-037 subgoal refiner. Nomination-only; cannot mint authority."""

    interface: ClassVar[str] = PARTITION_SUBGOAL_REFINER_INTERFACE

    def refine(
        self,
        evidence: Mapping[str, Any]
        | GoalCompilationReceipt
        | OpportunityDetectionReceipt,
        *,
        scopes: Sequence[Mapping[str, Any] | TaskScope] | None = None,
        prior_failures: Sequence[Mapping[str, Any]] | None = None,
    ) -> SubgoalRefinementReceipt:
        return refine_partition_subgoals(
            evidence, scopes=scopes, prior_failures=prior_failures
        )


def _task_from_subgoal(subgoal: RefinedSubgoal, kind: str, commands: Sequence[str]) -> CompiledTask:
    selected = subgoal.task_kinds
    return CompiledTask(
        tree_id=subgoal.tree_id,
        goal_cid=subgoal.goal_cid,
        finding_cid=subgoal.finding_cid,
        subgoal_cid=subgoal.subgoal_cid,
        module_id=subgoal.module_id,
        kind=kind,
        write_paths=subgoal.write_paths,
        member_ids=subgoal.member_ids,
        evidence_cids=subgoal.evidence_cids,
        validation_commands=commands,
        depends_on=_depends_on(kind, selected),
        autonomy_tier=subgoal.autonomy_tier,
    )


@dataclass(frozen=True, slots=True)
class ExtractionWaveTaskCompilationReceipt:
    """Ordered extraction-wave task synthesis. Nomination-only."""

    tree_id: str
    tasks: Sequence[CompiledTask | Mapping[str, Any]]
    blocked_retry_fingerprints: Sequence[str] = ()
    worktree_id: str = ""
    lease_id: str = ""
    fence_id: str = ""
    compiler_id: str = COMPILER_ID

    interface: ClassVar[str] = EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_INTERFACE
    schema: ClassVar[str] = EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "tasks",
            "blocked_retry_fingerprints",
            "worktree_id",
            "lease_id",
            "fence_id",
            "compiler_id",
            "task_cids",
            "wave_task_cids",
            "rollback_mode",
            "generic_prompt_forbidden",
            "one_packet_at_a_time",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_retire_facade",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        compiler = _text(self.compiler_id, "compiler_id")
        if compiler != COMPILER_ID:
            raise TaskCompilerError(
                "compiler_id must remain the SPAR-037 extraction-wave task compiler"
            )
        tasks = tuple(_coerce_task(item) for item in self.tasks)
        if len(tasks) > MAX_TASKS:
            raise TaskCompilerError("tasks exceed maximum length")
        tree_id = _tree_id(self.tree_id)
        mismatched = [item.module_id for item in tasks if item.tree_id != tree_id]
        if mismatched:
            raise TaskCompilerError("task tree_id does not match receipt")
        ids = [item.task_cid for item in tasks]
        if len(ids) != len(set(ids)):
            raise TaskCompilerError("duplicate task identity")
        fingerprints = [item.retry_fingerprint for item in tasks]
        if len(fingerprints) != len(set(fingerprints)):
            raise TaskCompilerError("duplicate retry fingerprint")
        blocked = _cids(
            list(self.blocked_retry_fingerprints),
            "blocked_retry_fingerprints",
            required=False,
        )
        overlap = set(fingerprints) & set(blocked)
        if overlap:
            raise TaskCompilerError("blocked identical retry was re-emitted")
        ordered = tuple(
            sorted(
                tasks,
                key=lambda item: (
                    TASK_KIND_ORDER.index(item.kind),
                    item.module_id,
                    item.task_cid,
                ),
            )
        )
        wave = tuple(item for item in ordered if item.kind in WAVE_KINDS)
        if len(wave) > MAX_WAVE_TASKS:
            raise TaskCompilerError("wave tasks exceed maximum length")
        worktree = _optional_cid(self.worktree_id, "worktree_id")
        lease = _optional_cid(self.lease_id, "lease_id")
        fence = _optional_cid(self.fence_id, "fence_id")
        bound = [worktree, lease, fence]
        if any(bound) and not all(bound):
            raise TaskCompilerError("wave binding requires worktree, lease, and fence")
        if not ordered and not blocked:
            raise TaskCompilerError("compilation requires tasks or blocked retries")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "tasks", ordered)
        object.__setattr__(self, "blocked_retry_fingerprints", blocked)
        object.__setattr__(self, "worktree_id", worktree)
        object.__setattr__(self, "lease_id", lease)
        object.__setattr__(self, "fence_id", fence)
        object.__setattr__(self, "compiler_id", compiler)

    @property
    def task_cids(self) -> tuple[str, ...]:
        return tuple(item.task_cid for item in self.tasks)

    @property
    def wave_task_cids(self) -> tuple[str, ...]:
        return tuple(item.task_cid for item in self.tasks if item.kind in WAVE_KINDS)

    @property
    def rollback_mode(self) -> str:
        return ROLLBACK_MODE

    @property
    def generic_prompt_forbidden(self) -> bool:
        return True

    @property
    def one_packet_at_a_time(self) -> bool:
        return True

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def can_retire_facade(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_SCHEMA,
            "interface": EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "tasks": [item.to_dict() for item in self.tasks],
            "blocked_retry_fingerprints": list(self.blocked_retry_fingerprints),
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "compiler_id": self.compiler_id,
            "task_cids": list(self.task_cids),
            "wave_task_cids": list(self.wave_task_cids),
            "rollback_mode": ROLLBACK_MODE,
            "generic_prompt_forbidden": True,
            "one_packet_at_a_time": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_retire_facade": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "ExtractionWaveTaskCompilationReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_SCHEMA:
            raise TaskCompilerError(
                "unsupported ExtractionWaveTaskCompilationReceipt schema"
            )
        if payload.pop("interface") != EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_INTERFACE:
            raise TaskCompilerError(
                "unsupported ExtractionWaveTaskCompilationReceipt interface"
            )
        if payload.pop("generic_prompt_forbidden") is not True:
            raise TaskCompilerError("compilation cannot emit a generic prompt")
        if payload.pop("one_packet_at_a_time") is not True:
            raise TaskCompilerError("wave must apply one packet at a time")
        if payload.pop("rollback_mode") != ROLLBACK_MODE:
            raise TaskCompilerError("rollback_mode must restore preimages or discard worktree")
        _pop_authority_flags(payload, "ExtractionWaveTaskCompilationReceipt")
        payload.pop("task_cids")
        payload.pop("wave_task_cids")
        result = cls(**payload)
        _verify_cid(
            claimed,
            result.receipt_cid,
            "ExtractionWaveTaskCompilationReceipt receipt_cid",
        )
        return result


def _commands_by_module(
    scopes: Mapping[str, TaskScope],
) -> dict[str, tuple[str, ...]]:
    return {module: scope.validation_commands for module, scope in scopes.items()}


def compile_extraction_wave_tasks(
    refinement: SubgoalRefinementReceipt | Mapping[str, Any],
    *,
    scopes: Sequence[Mapping[str, Any] | TaskScope] | None = None,
    worktree_id: str = "",
    lease_id: str = "",
    fence_id: str = "",
    validation_commands: Mapping[str, Sequence[str]] | None = None,
) -> ExtractionWaveTaskCompilationReceipt:
    """Compile refined subgoals into one-packet-at-a-time wave tasks."""

    receipt = (
        refinement
        if isinstance(refinement, SubgoalRefinementReceipt)
        else SubgoalRefinementReceipt.from_dict(refinement)
    )
    commands: dict[str, tuple[str, ...]] = {}
    if scopes is not None:
        scope_map = _parse_scopes(scopes)
        commands.update(_commands_by_module(scope_map))
    if validation_commands is not None:
        if not isinstance(validation_commands, Mapping):
            raise TaskCompilerError("validation_commands must be an object")
        for module, items in validation_commands.items():
            commands[_text(module, "module_id")] = _commands(list(items))
    tasks: list[CompiledTask] = []
    for subgoal in receipt.subgoals:
        module_commands = commands.get(subgoal.module_id)
        if module_commands is None:
            raise TaskCompilerError(
                f"missing validation commands for module {subgoal.module_id}"
            )
        for kind in subgoal.task_kinds:
            tasks.append(_task_from_subgoal(subgoal, kind, module_commands))
    return ExtractionWaveTaskCompilationReceipt(
        tree_id=receipt.tree_id,
        tasks=tasks,
        blocked_retry_fingerprints=receipt.blocked_retry_fingerprints,
        worktree_id=worktree_id,
        lease_id=lease_id,
        fence_id=fence_id,
    )


class ExtractionWaveTaskCompiler:
    """SPAR-037 extraction-wave task compiler. Nomination-only."""

    interface: ClassVar[str] = EXTRACTION_WAVE_TASK_COMPILER_INTERFACE

    def compile(
        self,
        refinement: SubgoalRefinementReceipt | Mapping[str, Any],
        *,
        scopes: Sequence[Mapping[str, Any] | TaskScope] | None = None,
        worktree_id: str = "",
        lease_id: str = "",
        fence_id: str = "",
        validation_commands: Mapping[str, Sequence[str]] | None = None,
    ) -> ExtractionWaveTaskCompilationReceipt:
        return compile_extraction_wave_tasks(
            refinement,
            scopes=scopes,
            worktree_id=worktree_id,
            lease_id=lease_id,
            fence_id=fence_id,
            validation_commands=validation_commands,
        )


def compile_backlog(
    evidence: Mapping[str, Any]
    | GoalCompilationReceipt
    | OpportunityDetectionReceipt,
    *,
    scopes: Sequence[Mapping[str, Any] | TaskScope] | None = None,
    prior_failures: Sequence[Mapping[str, Any]] | None = None,
    worktree_id: str = "",
    lease_id: str = "",
    fence_id: str = "",
) -> ExtractionWaveTaskCompilationReceipt:
    """Refine then compile one backlog. Nomination-only."""

    if isinstance(evidence, Mapping):
        payload = _mapping(evidence, "evidence")
        resolved_scopes = scopes if scopes is not None else payload.get("scopes")
        resolved_prior = (
            prior_failures
            if prior_failures is not None
            else payload.get("prior_failures")
        )
        worktree_id = worktree_id or str(payload.get("worktree_id") or "")
        lease_id = lease_id or str(payload.get("lease_id") or "")
        fence_id = fence_id or str(payload.get("fence_id") or "")
    else:
        payload = evidence
        resolved_scopes = scopes
        resolved_prior = prior_failures
    refinement = refine_partition_subgoals(
        payload, scopes=resolved_scopes, prior_failures=resolved_prior
    )
    scope_list = resolved_scopes
    return compile_extraction_wave_tasks(
        refinement,
        scopes=scope_list,
        worktree_id=worktree_id,
        lease_id=lease_id,
        fence_id=fence_id,
    )


def encode_canonical_receipt(
    receipt: SubgoalRefinementReceipt | ExtractionWaveTaskCompilationReceipt,
) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_refinement_receipt(
    payload: Mapping[str, Any],
) -> SubgoalRefinementReceipt:
    return SubgoalRefinementReceipt.from_dict(payload)


def decode_canonical_compilation_receipt(
    payload: Mapping[str, Any],
) -> ExtractionWaveTaskCompilationReceipt:
    return ExtractionWaveTaskCompilationReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise TaskCompilerError(
            f"task compiler must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ACCEPTANCE_IDS",
    "ALLOWED_EFFECTS",
    "ALWAYS_KINDS",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "AutonomyTier",
    "BACKLOG_REPAIR_DECISION_INTERFACE",
    "BacklogRepairDecision",
    "COMPILED_TASK_INTERFACE",
    "COMPILER_ID",
    "CompiledTask",
    "DECLARED_TASK_KINDS",
    "DUCKLAKE_IS_AUTHORITY",
    "EXISTING_ADAPTER_AUTHORITIES",
    "EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_INTERFACE",
    "EXTRACTION_WAVE_TASK_COMPILER_INTERFACE",
    "ExtractionWaveTaskCompilationReceipt",
    "ExtractionWaveTaskCompiler",
    "FORBIDDEN_EFFECTS",
    "GENERIC_PROMPT_FORBIDDEN",
    "GOAL_ID",
    "IDENTICAL_RETRY_WITHOUT_EVIDENCE",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "ONE_PACKET_AT_A_TIME",
    "PARTITION_SUBGOAL_REFINER_INTERFACE",
    "PLAN_IS_NOMINATION_ONLY",
    "POLICY_ID",
    "POLICY_REVISION",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "PartitionSubgoalRefiner",
    "PriorFailure",
    "RAW_SOURCE_REQUIRED",
    "REFINED_SUBGOAL_INTERFACE",
    "REFINER_ID",
    "RISK_TO_KINDS",
    "ROLLBACK_MODE",
    "RefinedSubgoal",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "SUBGOAL_REFINEMENT_RECEIPT_INTERFACE",
    "SubgoalRefinementReceipt",
    "TASK_CAN_AUTHORIZE_COMPLETION",
    "TASK_CAN_AUTHORIZE_TRANSITION",
    "TASK_CAN_CREATE_AUTHORITY",
    "TASK_CAN_RETIRE_FACADE",
    "TASK_CONTRACT_VERSION",
    "TASK_ID",
    "TASK_KIND_ORDER",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TaskCompilerError",
    "TaskKind",
    "TaskScope",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WAVE_KINDS",
    "WORKER_SELF_APPROVAL",
    "assert_not_competing_capsule_family",
    "compile_backlog",
    "compile_extraction_wave_tasks",
    "decode_canonical_compilation_receipt",
    "decode_canonical_refinement_receipt",
    "encode_canonical_receipt",
    "provider_free_exports",
    "refine_partition_subgoals",
    "retry_fingerprint",
    "task_compiler_cid_profile",
]
