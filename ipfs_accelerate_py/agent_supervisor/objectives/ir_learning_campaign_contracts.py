"""Canonical IR learning campaign, role, and board-task contracts.

``IRLearningCampaign@1`` is the versioned operational campaign envelope used
by the supervisor work graph.  It reuses the existing canonical-JSON identity
boundary and does not invent a second agent framework, semantic vocabulary, or
hidden-label channel.

A campaign is admitted only when:

* every independently schedulable work-graph role is declared;
* every task carries the closed board-field set;
* dependency output identities are bound (or explicitly unresolved); and
* a lease is refused while any required ``RESULT(task)`` remains unresolved.

Free-form prose is retained as non-semantic metadata.  Prompt-selected
authority, hidden labels, and producer-claimed promotion are rejected.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ..control.control_contracts import (
    MUTATION_OPERATIONS,
    Operation,
    OperationAuthority,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    _mapping,
    content_identity,
)


IR_LEARNING_CAMPAIGN_CONTRACT_VERSION: Final = 1
CONTRACT_VERSION: Final = IR_LEARNING_CAMPAIGN_CONTRACT_VERSION
SCHEMA_VERSION: Final = IR_LEARNING_CAMPAIGN_CONTRACT_VERSION

IR_LEARNING_CAMPAIGN_SCHEMA: Final = "IRLearningCampaign@1"
IR_LEARNING_CAMPAIGN_SCHEMA_PATH: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-learning-campaign@1"
)
IR_LEARNING_CAMPAIGN_ROLE_SCHEMA: Final = "IRLearningCampaignRole@1"
IR_LEARNING_CAMPAIGN_TASK_SCHEMA: Final = "IRLearningCampaignTask@1"
IR_LEARNING_CAMPAIGN_OPERATION_SCHEMA: Final = "IRLearningCampaignOperation@1"
IR_LEARNING_CAMPAIGN_REVISION_SCHEMA: Final = "IRLearningCampaignTaskRevision@1"
IR_LEARNING_CAMPAIGN_PROJECTION_SCHEMA: Final = (
    "IRLearningCampaignDependencyProjection@1"
)
IR_LEARNING_CAMPAIGN_RECEIPT_SCHEMA: Final = "IRLearningCampaignOperationReceipt@1"

ACCEPTED_CAMPAIGN_SCHEMAS: Final = frozenset(
    {
        IR_LEARNING_CAMPAIGN_SCHEMA,
        IR_LEARNING_CAMPAIGN_SCHEMA_PATH,
    }
)

RESULT_REFERENCE_RE: Final = re.compile(r"^RESULT\(([^)]+)\)$")
RESULT_ALIAS_RE: Final = re.compile(r"^RESULT\(([A-Z0-9][A-Z0-9._:-]*)\)$")

IRLearningCampaignValidationError = ContractValidationError


class CampaignWorkGraphRole(str, Enum):
    """Independently schedulable ownership roles on the campaign work graph."""

    INVENTORY = "inventory"
    CORPUS = "corpus"
    SPLIT = "split"
    LINEAGE = "lineage"
    COMPILER = "compiler"
    DECOMPILER = "decompiler"
    TOKENIZER = "tokenizer"
    CURRICULUM = "curriculum"
    TRAINING_RUN = "training_run"
    PROOF = "proof"
    EVALUATION = "evaluation"
    CHECKPOINT = "checkpoint"
    PROMOTION = "promotion"
    PUBLICATION = "publication"
    RESOURCE = "resource"
    CAMPAIGN_CONTROL = "campaign_control"


REQUIRED_WORK_GRAPH_ROLES: Final[tuple[CampaignWorkGraphRole, ...]] = tuple(
    CampaignWorkGraphRole
)


class CampaignOperationKind(str, Enum):
    """Closed campaign operations declared by ``IRLearningCampaign@1``.

    The O2 public surface adds ``start`` and ``resume`` to the PGIR-060
    create/plan/status/steer/refill/proof-replay/compare/promote/reject/report
    set.  Both new verbs reuse existing control-catalog operations and do not
    expand ``Operation``.
    """

    CREATE = "create"
    PLAN = "plan"
    START = "start"
    RESUME = "resume"
    STATUS = "status"
    STEER = "steer"
    REFILL = "refill"
    PROOF_REPLAY = "proof-replay"
    COMPARE = "compare"
    PROMOTE = "promote"
    REJECT = "reject"
    REPORT = "report"


# PGIR-060 board fixtures declare this set explicitly.  Start/resume are
# published on the O2 surface and are defaulted when a campaign omits
# ``operations``, but they are not required of already-admitted envelopes.
REQUIRED_CAMPAIGN_OPERATIONS: Final[tuple[CampaignOperationKind, ...]] = (
    CampaignOperationKind.CREATE,
    CampaignOperationKind.PLAN,
    CampaignOperationKind.STATUS,
    CampaignOperationKind.STEER,
    CampaignOperationKind.REFILL,
    CampaignOperationKind.PROOF_REPLAY,
    CampaignOperationKind.COMPARE,
    CampaignOperationKind.PROMOTE,
    CampaignOperationKind.REJECT,
    CampaignOperationKind.REPORT,
)

STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS: Final[tuple[CampaignOperationKind, ...]] = (
    CampaignOperationKind.CREATE,
    CampaignOperationKind.PLAN,
    CampaignOperationKind.START,
    CampaignOperationKind.RESUME,
    CampaignOperationKind.STATUS,
    CampaignOperationKind.STEER,
    CampaignOperationKind.REFILL,
    CampaignOperationKind.PROOF_REPLAY,
    CampaignOperationKind.COMPARE,
    CampaignOperationKind.PROMOTE,
    CampaignOperationKind.REJECT,
    CampaignOperationKind.REPORT,
)


class CampaignTaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_GO = "no_go"


class CampaignCompletionKind(str, Enum):
    VALIDATED_IMPLEMENTATION = "validated-implementation"
    SUPERVISOR_EVIDENCE = "supervisor-evidence"
    MANUAL = "manual"
    NONE = "none"


class CampaignPriority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class CampaignOwningRepository(str, Enum):
    IPFS_ACCELERATE_PY = "ipfs_accelerate_py"
    IPFS_DATASETS_PY = "ipfs_datasets_py"


class CampaignResourceProfile(str, Enum):
    CPU_S = "RP-CPU-S"
    CPU_M = "RP-CPU-M"
    IO_PINNED = "RP-IO-PINNED"
    PROVER = "RP-PROVER"
    GPU = "RP-GPU"
    MIXED = "RP-MIXED"


class CampaignTrack(str, Enum):
    INVENTORY = "inventory"
    QUALIFICATION = "qualification"
    COMPILER = "compiler"
    CURRICULUM = "curriculum"
    CAMPAIGN = "campaign"
    EVALUATION = "evaluation"
    API = "api"
    PUBLICATION = "publication"
    SECURITY = "security"
    EXPERIMENT = "experiment"


class CampaignLeaseDecision(str, Enum):
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"


class CampaignOperationStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    DENIED = "denied"
    BLOCKED = "blocked"


# Campaign operations reuse the existing control catalog.  They never expand
# the closed ``Operation`` enum or raise authority above the mapped surface.
CAMPAIGN_OPERATION_CONTROL_MAP: Final[Mapping[CampaignOperationKind, Operation]] = {
    CampaignOperationKind.CREATE: Operation.WORKFLOW_MATERIALIZE,
    CampaignOperationKind.PLAN: Operation.PLAN,
    CampaignOperationKind.START: Operation.START,
    CampaignOperationKind.RESUME: Operation.RESUME,
    CampaignOperationKind.STATUS: Operation.STATUS,
    CampaignOperationKind.STEER: Operation.OBJECTIVE_REFINE,
    CampaignOperationKind.REFILL: Operation.BACKLOG_REFILL,
    CampaignOperationKind.PROOF_REPLAY: Operation.VALIDATION_REPLAY,
    CampaignOperationKind.COMPARE: Operation.RECEIPTS,
    CampaignOperationKind.PROMOTE: Operation.OBJECTIVE_RECONCILE,
    CampaignOperationKind.REJECT: Operation.QUARANTINE,
    CampaignOperationKind.REPORT: Operation.METRICS,
}

CAMPAIGN_OPERATION_AUTHORITY: Final[Mapping[CampaignOperationKind, OperationAuthority]] = {
    kind: CAMPAIGN_OPERATION_CONTROL_MAP[kind].authority for kind in CampaignOperationKind
}

LEASE_REQUIRING_OPERATIONS: Final[frozenset[CampaignOperationKind]] = frozenset(
    {
        CampaignOperationKind.CREATE,
        CampaignOperationKind.START,
        CampaignOperationKind.RESUME,
        CampaignOperationKind.STEER,
        CampaignOperationKind.REFILL,
        CampaignOperationKind.PROOF_REPLAY,
        CampaignOperationKind.PROMOTE,
        CampaignOperationKind.REJECT,
    }
)

BOARD_TASK_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "title",
    "status",
    "completion",
    "is_schedulable",
    "priority",
    "track",
    "parent_goal",
    "subgoal",
    "owning_repository",
    "owned_paths",
    "base_source_revisions",
    "source_dataset_revisions",
    "data_split_identity",
    "compiler_identity",
    "decompiler_identity",
    "model_checkpoint_identity",
    "objective",
    "depends_on",
    "resource_profile",
    "expected_inputs",
    "expected_outputs",
    "allowed_effects",
    "prohibited_effects",
    "acceptance_criteria",
    "required_proof_or_evaluation_evidence",
    "lease_and_checkpoint_policy",
    "rollback_procedure",
    "result_identity",
    "outputs",
    "validation",
    "bundle",
    "parallel_lane",
    "predicted_files",
    "conflict_policy",
    "work_graph_role",
)

REQUIRED_BOARD_TASK_FIELDS: Final[frozenset[str]] = frozenset(BOARD_TASK_FIELDS)

_FORBIDDEN_AUTHORITY_FIELDS: Final = frozenset(
    {
        "hidden_labels",
        "hidden_label",
        "prompt_authority",
        "prompt_selected_authority",
        "self_promote",
        "self_promotion",
        "secret",
        "secrets",
    }
)

_ABSOLUTE_TEXT_BYTES = 64 * 1024


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    max_bytes: int = _ABSOLUTE_TEXT_BYTES,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise ContractValidationError("%s must be a string" % field_name)
    else:
        result = value.strip()
    if required and not result:
        raise ContractValidationError("%s is required" % field_name)
    if "\x00" in result:
        raise ContractValidationError("%s must not contain NUL bytes" % field_name)
    if len(result.encode("utf-8")) > max_bytes:
        raise ContractValidationError(
            "%s exceeds the maximum of %s UTF-8 bytes" % (field_name, max_bytes)
        )
    return result


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        source: Iterable[Any] = ()
    elif isinstance(values, str):
        source = (item.strip() for item in values.split(",") if item.strip())
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray, memoryview)):
        source = values
    else:
        raise ContractValidationError("%s must be a sequence of strings" % field_name)
    result: list[str] = []
    for index, value in enumerate(source):
        item = _text(value, field_name="%s[%s]" % (field_name, index), required=True)
        if item not in result:
            result.append(item)
    if required and not result:
        raise ContractValidationError("%s must not be empty" % field_name)
    return tuple(result if preserve_order else sorted(result))


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw).strip())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractValidationError("%s must be one of: %s" % (field_name, allowed)) from exc


def _bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ContractValidationError("%s must be a boolean" % field_name)


def _schema(payload: Mapping[str, Any], expected: str | frozenset[str]) -> None:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("campaign contract payload must be an object")
    accepted = expected if isinstance(expected, frozenset) else frozenset({expected})
    supplied = payload.get("schema")
    if supplied not in (None, "") and supplied not in accepted:
        raise ContractValidationError(
            "unsupported schema %r; expected %s" % (supplied, sorted(accepted)[0])
        )


def _claimed_identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    for name in ("content_id", "identity", "campaign_revision", "task_revision"):
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual and name in payload:
            if name in {"campaign_revision", "task_revision"} and noun not in name:
                continue
            if name in {"campaign_revision", "task_revision"}:
                continue
            raise ContractValidationError("%s content identity does not match payload" % noun)


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str], *, noun: str) -> None:
    extra = set(payload).difference(allowed)
    if extra:
        raise ContractValidationError(
            "%s contains unsupported fields; rebuild its canonical payload" % noun
        )


def _reject_forbidden_authority(payload: Mapping[str, Any], *, noun: str) -> None:
    keys = {str(key).strip().lower().replace("-", "_") for key in payload}
    if keys.intersection(_FORBIDDEN_AUTHORITY_FIELDS):
        raise ContractValidationError(
            "%s cannot carry prompt-selected authority or hidden labels" % noun
        )
    nested = payload.get("metadata")
    if isinstance(nested, Mapping):
        nested_keys = {str(key).strip().lower().replace("-", "_") for key in nested}
        if nested_keys.intersection(_FORBIDDEN_AUTHORITY_FIELDS):
            raise ContractValidationError(
                "%s metadata cannot carry prompt-selected authority or hidden labels" % noun
            )


def result_reference_task_id(value: str) -> str | None:
    """Return the referenced task id from ``RESULT(task)``, if well formed."""

    match = RESULT_ALIAS_RE.match(str(value or "").strip())
    return match.group(1) if match else None


def collect_result_references(*values: Any) -> tuple[str, ...]:
    found: list[str] = []
    for value in values:
        if isinstance(value, str):
            task_id = result_reference_task_id(value)
            if task_id and task_id not in found:
                found.append(task_id)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                task_id = result_reference_task_id(str(item))
                if task_id and task_id not in found:
                    found.append(task_id)
    return tuple(sorted(found))


def bind_result_identity(task_id: str) -> str:
    normalized = _text(task_id, field_name="task_id", required=True)
    return "RESULT(%s)" % normalized


class CampaignContract(CanonicalContract):
    """Canonical contract pinned to the IR learning campaign schema version."""

    @property
    def schema_version(self) -> int:
        return IR_LEARNING_CAMPAIGN_CONTRACT_VERSION

    def _versioned(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "contract_version": IR_LEARNING_CAMPAIGN_CONTRACT_VERSION,
            **dict(payload),
        }


@dataclass(frozen=True)
class CampaignWorkGraphRoleRecord(CampaignContract):
    """One independently schedulable role declared by a campaign."""

    SCHEMA: ClassVar[str] = IR_LEARNING_CAMPAIGN_ROLE_SCHEMA

    role: CampaignWorkGraphRole
    owner_actor_id: str
    exclusive_lease_key: str
    independently_schedulable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role", _enum(self.role, CampaignWorkGraphRole, field_name="role")
        )
        object.__setattr__(
            self,
            "owner_actor_id",
            _text(self.owner_actor_id, field_name="owner_actor_id", required=True),
        )
        object.__setattr__(
            self,
            "exclusive_lease_key",
            _text(
                self.exclusive_lease_key,
                field_name="exclusive_lease_key",
                required=True,
            ),
        )
        if not isinstance(self.independently_schedulable, bool):
            raise ContractValidationError("independently_schedulable must be a boolean")
        if not self.independently_schedulable:
            raise ContractValidationError(
                "work-graph roles must remain independently schedulable"
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, field_name="metadata"))
        _reject_forbidden_authority(self.metadata, noun="campaign role")

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "role": self.role,
                "owner_actor_id": self.owner_actor_id,
                "exclusive_lease_key": self.exclusive_lease_key,
                "independently_schedulable": self.independently_schedulable,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignWorkGraphRoleRecord":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "content_id",
                "identity",
                "role",
                "owner_actor_id",
                "exclusive_lease_key",
                "independently_schedulable",
                "metadata",
            },
            noun="campaign role",
        )
        _reject_forbidden_authority(payload, noun="campaign role")
        result = cls(
            role=payload.get("role", ""),
            owner_actor_id=payload.get("owner_actor_id", ""),
            exclusive_lease_key=payload.get("exclusive_lease_key", ""),
            independently_schedulable=payload.get("independently_schedulable", True),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "campaign role")
        return result


def default_campaign_roles(
    *,
    owner_actor_id: str = "supervisor",
) -> tuple[CampaignWorkGraphRoleRecord, ...]:
    """Return the complete independently schedulable role set."""

    owner = _text(owner_actor_id, field_name="owner_actor_id", required=True)
    return tuple(
        CampaignWorkGraphRoleRecord(
            role=role,
            owner_actor_id=owner,
            exclusive_lease_key="campaign-role:%s" % role.value,
        )
        for role in REQUIRED_WORK_GRAPH_ROLES
    )


@dataclass(frozen=True)
class CampaignBoardTask(CampaignContract):
    """One campaign task with every field required by the executable board."""

    SCHEMA: ClassVar[str] = IR_LEARNING_CAMPAIGN_TASK_SCHEMA

    task_id: str
    title: str
    status: CampaignTaskStatus
    completion: CampaignCompletionKind
    is_schedulable: bool
    priority: CampaignPriority
    track: CampaignTrack
    parent_goal: str
    subgoal: str
    owning_repository: CampaignOwningRepository
    owned_paths: tuple[str, ...]
    base_source_revisions: str
    source_dataset_revisions: str
    data_split_identity: str
    compiler_identity: str
    decompiler_identity: str
    model_checkpoint_identity: str
    objective: str
    depends_on: tuple[str, ...]
    resource_profile: CampaignResourceProfile
    expected_inputs: str
    expected_outputs: str
    allowed_effects: str
    prohibited_effects: str
    acceptance_criteria: str
    required_proof_or_evaluation_evidence: str
    lease_and_checkpoint_policy: str
    rollback_procedure: str
    result_identity: str
    outputs: tuple[str, ...]
    validation: str
    bundle: str
    parallel_lane: str
    predicted_files: tuple[str, ...]
    conflict_policy: str
    work_graph_role: CampaignWorkGraphRole
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id", _text(self.task_id, field_name="task_id", required=True)
        )
        object.__setattr__(self, "title", _text(self.title, field_name="title", required=True))
        object.__setattr__(
            self, "status", _enum(self.status, CampaignTaskStatus, field_name="status")
        )
        object.__setattr__(
            self,
            "completion",
            _enum(self.completion, CampaignCompletionKind, field_name="completion"),
        )
        object.__setattr__(
            self,
            "is_schedulable",
            _bool(self.is_schedulable, field_name="is_schedulable"),
        )
        object.__setattr__(
            self,
            "priority",
            _enum(self.priority, CampaignPriority, field_name="priority"),
        )
        object.__setattr__(self, "track", _enum(self.track, CampaignTrack, field_name="track"))
        for name in (
            "parent_goal",
            "subgoal",
            "base_source_revisions",
            "source_dataset_revisions",
            "data_split_identity",
            "compiler_identity",
            "decompiler_identity",
            "model_checkpoint_identity",
            "objective",
            "expected_inputs",
            "expected_outputs",
            "allowed_effects",
            "prohibited_effects",
            "acceptance_criteria",
            "required_proof_or_evaluation_evidence",
            "lease_and_checkpoint_policy",
            "rollback_procedure",
            "result_identity",
            "validation",
            "bundle",
            "parallel_lane",
            "conflict_policy",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self,
            "owning_repository",
            _enum(
                self.owning_repository,
                CampaignOwningRepository,
                field_name="owning_repository",
            ),
        )
        object.__setattr__(
            self,
            "resource_profile",
            _enum(
                self.resource_profile,
                CampaignResourceProfile,
                field_name="resource_profile",
            ),
        )
        object.__setattr__(
            self,
            "work_graph_role",
            _enum(self.work_graph_role, CampaignWorkGraphRole, field_name="work_graph_role"),
        )
        for name in ("owned_paths", "depends_on", "outputs", "predicted_files"):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    required=name in {"owned_paths", "outputs", "predicted_files"},
                    preserve_order=name == "depends_on",
                ),
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, field_name="metadata"))
        _reject_forbidden_authority(self.metadata, noun="campaign task")
        if self.task_id in self.depends_on:
            raise ContractValidationError("a campaign task cannot depend on itself")
        expected_result = bind_result_identity(self.task_id)
        if self.result_identity != expected_result:
            raise ContractValidationError(
                "result_identity must be the closed RESULT(%s) binding" % self.task_id
            )
        if "hidden" in self.prohibited_effects.casefold() and "label" in self.prohibited_effects.casefold():
            # Keep the prohibition as data; do not parse it into authority.
            pass

    @property
    def required_result_dependencies(self) -> tuple[str, ...]:
        """Task ids whose ``RESULT(...)`` outputs this task consumes."""

        return collect_result_references(
            *self.depends_on,
            self.source_dataset_revisions,
            self.data_split_identity,
            self.compiler_identity,
            self.decompiler_identity,
            self.model_checkpoint_identity,
            self.expected_inputs,
        )

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "task_id": self.task_id,
                "title": self.title,
                "status": self.status,
                "completion": self.completion,
                "is_schedulable": self.is_schedulable,
                "priority": self.priority,
                "track": self.track,
                "parent_goal": self.parent_goal,
                "subgoal": self.subgoal,
                "owning_repository": self.owning_repository,
                "owned_paths": self.owned_paths,
                "base_source_revisions": self.base_source_revisions,
                "source_dataset_revisions": self.source_dataset_revisions,
                "data_split_identity": self.data_split_identity,
                "compiler_identity": self.compiler_identity,
                "decompiler_identity": self.decompiler_identity,
                "model_checkpoint_identity": self.model_checkpoint_identity,
                "objective": self.objective,
                "depends_on": self.depends_on,
                "resource_profile": self.resource_profile,
                "expected_inputs": self.expected_inputs,
                "expected_outputs": self.expected_outputs,
                "allowed_effects": self.allowed_effects,
                "prohibited_effects": self.prohibited_effects,
                "acceptance_criteria": self.acceptance_criteria,
                "required_proof_or_evaluation_evidence": self.required_proof_or_evaluation_evidence,
                "lease_and_checkpoint_policy": self.lease_and_checkpoint_policy,
                "rollback_procedure": self.rollback_procedure,
                "result_identity": self.result_identity,
                "outputs": self.outputs,
                "validation": self.validation,
                "bundle": self.bundle,
                "parallel_lane": self.parallel_lane,
                "predicted_files": self.predicted_files,
                "conflict_policy": self.conflict_policy,
                "work_graph_role": self.work_graph_role,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignBoardTask":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "content_id",
                "identity",
                *BOARD_TASK_FIELDS,
                "metadata",
            },
            noun="campaign task",
        )
        _reject_forbidden_authority(payload, noun="campaign task")
        missing = sorted(REQUIRED_BOARD_TASK_FIELDS.difference(payload))
        if missing:
            raise ContractValidationError(
                "campaign task is missing required board fields: %s" % ", ".join(missing)
            )
        result = cls(
            task_id=payload.get("task_id", ""),
            title=payload.get("title", ""),
            status=payload.get("status", CampaignTaskStatus.TODO),
            completion=payload.get("completion", CampaignCompletionKind.VALIDATED_IMPLEMENTATION),
            is_schedulable=payload.get("is_schedulable", False),
            priority=payload.get("priority", CampaignPriority.P0),
            track=payload.get("track", CampaignTrack.CAMPAIGN),
            parent_goal=payload.get("parent_goal", ""),
            subgoal=payload.get("subgoal", ""),
            owning_repository=payload.get(
                "owning_repository", CampaignOwningRepository.IPFS_ACCELERATE_PY
            ),
            owned_paths=tuple(payload.get("owned_paths") or ()),
            base_source_revisions=payload.get("base_source_revisions", ""),
            source_dataset_revisions=payload.get("source_dataset_revisions", ""),
            data_split_identity=payload.get("data_split_identity", ""),
            compiler_identity=payload.get("compiler_identity", ""),
            decompiler_identity=payload.get("decompiler_identity", ""),
            model_checkpoint_identity=payload.get("model_checkpoint_identity", ""),
            objective=payload.get("objective", ""),
            depends_on=tuple(payload.get("depends_on") or ()),
            resource_profile=payload.get("resource_profile", CampaignResourceProfile.CPU_M),
            expected_inputs=payload.get("expected_inputs", ""),
            expected_outputs=payload.get("expected_outputs", ""),
            allowed_effects=payload.get("allowed_effects", ""),
            prohibited_effects=payload.get("prohibited_effects", ""),
            acceptance_criteria=payload.get("acceptance_criteria", ""),
            required_proof_or_evaluation_evidence=payload.get(
                "required_proof_or_evaluation_evidence", ""
            ),
            lease_and_checkpoint_policy=payload.get("lease_and_checkpoint_policy", ""),
            rollback_procedure=payload.get("rollback_procedure", ""),
            result_identity=payload.get("result_identity", ""),
            outputs=tuple(payload.get("outputs") or ()),
            validation=payload.get("validation", ""),
            bundle=payload.get("bundle", ""),
            parallel_lane=payload.get("parallel_lane", ""),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            conflict_policy=payload.get("conflict_policy", ""),
            work_graph_role=payload.get(
                "work_graph_role", CampaignWorkGraphRole.CAMPAIGN_CONTROL
            ),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "campaign task")
        return result


@dataclass(frozen=True)
class CampaignTaskRevision(CampaignContract):
    """Task revision that binds unresolved dependency outputs before lease."""

    SCHEMA: ClassVar[str] = IR_LEARNING_CAMPAIGN_REVISION_SCHEMA

    task_id: str
    campaign_id: str
    input_root_cid: str
    task_content_id: str
    dependency_output_bindings: Mapping[str, str]
    unresolved_dependency_outputs: tuple[str, ...]
    lease_decision: CampaignLeaseDecision
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "campaign_id", "input_root_cid", "task_content_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        bindings = _mapping(
            self.dependency_output_bindings, field_name="dependency_output_bindings"
        )
        normalized: dict[str, str] = {}
        for key, value in bindings.items():
            dep = _text(key, field_name="dependency_output_bindings key", required=True)
            bound = _text(value, field_name="dependency_output_bindings value")
            normalized[dep] = bound or "unresolved"
        object.__setattr__(
            self,
            "dependency_output_bindings",
            {key: normalized[key] for key in sorted(normalized)},
        )
        unresolved = _strings(
            self.unresolved_dependency_outputs,
            field_name="unresolved_dependency_outputs",
        )
        derived_unresolved = tuple(
            sorted(
                bind_result_identity(key)
                if not key.startswith("RESULT(")
                else key
                for key, value in self.dependency_output_bindings.items()
                if value == "unresolved"
            )
        )
        if unresolved and unresolved != derived_unresolved:
            raise ContractValidationError(
                "unresolved_dependency_outputs must match unbound RESULT identities"
            )
        object.__setattr__(self, "unresolved_dependency_outputs", derived_unresolved)
        object.__setattr__(
            self,
            "lease_decision",
            _enum(self.lease_decision, CampaignLeaseDecision, field_name="lease_decision"),
        )
        expected = (
            CampaignLeaseDecision.BLOCKED
            if self.unresolved_dependency_outputs
            else CampaignLeaseDecision.ELIGIBLE
        )
        if self.lease_decision is not expected:
            raise ContractValidationError(
                "task revision must block lease while dependency outputs are unresolved"
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, field_name="metadata"))
        _reject_forbidden_authority(self.metadata, noun="campaign task revision")

    @property
    def lease_eligible(self) -> bool:
        return self.lease_decision is CampaignLeaseDecision.ELIGIBLE

    @property
    def task_revision(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "task_id": self.task_id,
                "campaign_id": self.campaign_id,
                "input_root_cid": self.input_root_cid,
                "task_content_id": self.task_content_id,
                "dependency_output_bindings": self.dependency_output_bindings,
                "unresolved_dependency_outputs": self.unresolved_dependency_outputs,
                "lease_decision": self.lease_decision,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignTaskRevision":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "content_id",
                "identity",
                "task_revision",
                "task_id",
                "campaign_id",
                "input_root_cid",
                "task_content_id",
                "dependency_output_bindings",
                "unresolved_dependency_outputs",
                "lease_decision",
                "lease_eligible",
                "metadata",
            },
            noun="campaign task revision",
        )
        _reject_forbidden_authority(payload, noun="campaign task revision")
        result = cls(
            task_id=payload.get("task_id", ""),
            campaign_id=payload.get("campaign_id", ""),
            input_root_cid=payload.get("input_root_cid", ""),
            task_content_id=payload.get("task_content_id", ""),
            dependency_output_bindings=payload.get("dependency_output_bindings") or {},
            unresolved_dependency_outputs=tuple(
                payload.get("unresolved_dependency_outputs") or ()
            ),
            lease_decision=payload.get("lease_decision", CampaignLeaseDecision.BLOCKED),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("task_revision")
        if claimed not in (None, "") and claimed != result.task_revision:
            raise ContractValidationError("campaign task revision identity does not match payload")
        _claimed_identity(payload, result.content_id, "campaign task revision")
        return result


def revise_campaign_task(
    task: CampaignBoardTask,
    *,
    campaign_id: str,
    input_root_cid: str,
    dependency_results: Mapping[str, str] | None = None,
) -> CampaignTaskRevision:
    """Bind ``RESULT(dep)`` identities and refuse lease while any remain open."""

    if not isinstance(task, CampaignBoardTask):
        raise TypeError("task must be a CampaignBoardTask")
    supplied = {
        _text(key, field_name="dependency_results key", required=True): _text(
            value, field_name="dependency_results value"
        )
        for key, value in (dependency_results or {}).items()
    }
    bindings: dict[str, str] = {}
    for dep_id in task.depends_on:
        result_key = result_reference_task_id(dep_id) or dep_id
        bound = supplied.get(result_key) or supplied.get(bind_result_identity(result_key))
        bindings[result_key] = bound or "unresolved"
    for result_task_id in task.required_result_dependencies:
        bound = supplied.get(result_task_id) or supplied.get(bind_result_identity(result_task_id))
        bindings[result_task_id] = bound or bindings.get(result_task_id) or "unresolved"
    unresolved = tuple(
        bind_result_identity(key) for key, value in sorted(bindings.items()) if value == "unresolved"
    )
    return CampaignTaskRevision(
        task_id=task.task_id,
        campaign_id=campaign_id,
        input_root_cid=input_root_cid,
        task_content_id=task.content_id,
        dependency_output_bindings=bindings,
        unresolved_dependency_outputs=unresolved,
        lease_decision=(
            CampaignLeaseDecision.BLOCKED if unresolved else CampaignLeaseDecision.ELIGIBLE
        ),
    )


@dataclass(frozen=True)
class CampaignDependencyProjection(CampaignContract):
    """Deterministic action/effect/dependency projection used at admission."""

    SCHEMA: ClassVar[str] = IR_LEARNING_CAMPAIGN_PROJECTION_SCHEMA

    campaign_id: str
    campaign_revision: str
    input_root_cid: str
    repository_tree_id: str
    actions: tuple[Mapping[str, Any], ...] = ()
    dependency_edges: tuple[Mapping[str, Any], ...] = ()
    role_ids: tuple[str, ...] = ()
    lease_eligible_task_ids: tuple[str, ...] = ()
    blocked_task_ids: tuple[str, ...] = ()
    unresolved_result_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "campaign_id",
            "campaign_revision",
            "input_root_cid",
            "repository_tree_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(self, "actions", _unique_maps(self.actions, "action_id"))
        object.__setattr__(
            self, "dependency_edges", _unique_maps(self.dependency_edges, "dependency_id")
        )
        for name in (
            "role_ids",
            "lease_eligible_task_ids",
            "blocked_task_ids",
            "unresolved_result_ids",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), field_name=name)
            )

    @property
    def projection_id(self) -> str:
        return self.content_id

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(str(item["action_id"]) for item in self.actions)

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "campaign_id": self.campaign_id,
                "campaign_revision": self.campaign_revision,
                "input_root_cid": self.input_root_cid,
                "repository_tree_id": self.repository_tree_id,
                "actions": [dict(item) for item in self.actions],
                "dependency_edges": [dict(item) for item in self.dependency_edges],
                "role_ids": self.role_ids,
                "lease_eligible_task_ids": self.lease_eligible_task_ids,
                "blocked_task_ids": self.blocked_task_ids,
                "unresolved_result_ids": self.unresolved_result_ids,
                "proof_results": [],
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignDependencyProjection":
        _schema(payload, cls.SCHEMA)
        if payload.get("proof_results"):
            raise ContractValidationError("campaign projections cannot carry proof results")
        result = cls(
            campaign_id=payload.get("campaign_id", ""),
            campaign_revision=payload.get("campaign_revision", ""),
            input_root_cid=payload.get("input_root_cid", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            actions=tuple(payload.get("actions") or ()),
            dependency_edges=tuple(payload.get("dependency_edges") or ()),
            role_ids=tuple(payload.get("role_ids") or ()),
            lease_eligible_task_ids=tuple(payload.get("lease_eligible_task_ids") or ()),
            blocked_task_ids=tuple(payload.get("blocked_task_ids") or ()),
            unresolved_result_ids=tuple(payload.get("unresolved_result_ids") or ()),
        )
        claimed = payload.get("projection_id") or payload.get("content_id")
        if claimed not in (None, "") and claimed != result.content_id:
            raise ContractValidationError("campaign projection identity does not match payload")
        return result


def _unique_maps(
    values: Iterable[Mapping[str, Any]], key: str
) -> tuple[Mapping[str, Any], ...]:
    result: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ContractValidationError("%s records must be objects" % key)
        item = _canonical_value(raw)
        if not isinstance(item, dict):
            raise ContractValidationError("%s records must be objects" % key)
        identity = _text(item.get(key), field_name=key, required=True)
        previous = result.get(identity)
        if previous is not None and previous != item:
            raise ContractValidationError("conflicting %s record %s" % (key, identity))
        result[identity] = item
    return tuple(result[item] for item in sorted(result))


def _records(
    values: Any,
    *,
    cls: type[Any],
    key: str,
    field_name: str,
    required: bool = False,
) -> tuple[Any, ...]:
    if values is None:
        values = ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ContractValidationError("%s must be a sequence" % field_name)
    result: list[Any] = []
    for value in values:
        if isinstance(value, cls):
            record = value
        elif isinstance(value, Mapping):
            record = cls.from_dict(value)
        else:
            raise ContractValidationError("%s must contain %s records" % (field_name, cls.__name__))
        result.append(record)
    if required and not result:
        raise ContractValidationError("%s must not be empty" % field_name)
    result.sort(key=lambda item: getattr(item, key))
    keys = [getattr(item, key) for item in result]
    if len(keys) != len(set(keys)):
        raise ContractValidationError("%s identifiers must be unique" % field_name)
    return tuple(result)


def _acyclic(nodes: Iterable[str], dependencies: Mapping[str, tuple[str, ...]]) -> None:
    node_set = set(nodes)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ContractValidationError("campaign task dependencies must be acyclic")
        visiting.add(node)
        for dependency in dependencies.get(node, ()):
            if dependency not in node_set:
                raise ContractValidationError(
                    "campaign task %s has unknown dependency %s" % (node, dependency)
                )
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(node_set):
        visit(node)


@dataclass(frozen=True)
class IRLearningCampaign(CampaignContract):
    """Versioned IR learning campaign with closed roles, tasks, and operations."""

    SCHEMA: ClassVar[str] = IR_LEARNING_CAMPAIGN_SCHEMA

    campaign_id: str
    input_root_cid: str
    repository_tree_id: str
    roles: tuple[CampaignWorkGraphRoleRecord, ...]
    tasks: tuple[CampaignBoardTask, ...]
    operations: tuple[CampaignOperationKind, ...] = STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS
    dependency_results: Mapping[str, str] = field(default_factory=dict)
    owner_actor_id: str = "supervisor"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("campaign_id", "input_root_cid", "repository_tree_id", "owner_actor_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(
            self,
            "roles",
            _records(
                self.roles,
                cls=CampaignWorkGraphRoleRecord,
                key="role",
                field_name="roles",
                required=True,
            ),
        )
        declared = tuple(item.role for item in self.roles)
        missing_roles = [role.value for role in REQUIRED_WORK_GRAPH_ROLES if role not in declared]
        if missing_roles:
            raise ContractValidationError(
                "campaign is missing required work-graph roles: %s" % ", ".join(missing_roles)
            )
        object.__setattr__(
            self,
            "tasks",
            _records(
                self.tasks,
                cls=CampaignBoardTask,
                key="task_id",
                field_name="tasks",
                required=True,
            ),
        )
        operations = tuple(
            _enum(item, CampaignOperationKind, field_name="operations")
            for item in (self.operations or STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS)
        )
        unique_ops = tuple(sorted(set(operations), key=lambda item: item.value))
        missing_ops = [
            item.value for item in REQUIRED_CAMPAIGN_OPERATIONS if item not in unique_ops
        ]
        if missing_ops:
            raise ContractValidationError(
                "campaign is missing required operations: %s" % ", ".join(missing_ops)
            )
        object.__setattr__(self, "operations", unique_ops)
        object.__setattr__(
            self,
            "dependency_results",
            {
                _text(key, field_name="dependency_results key", required=True): _text(
                    value, field_name="dependency_results value", required=True
                )
                for key, value in sorted((_mapping(self.dependency_results, field_name="dependency_results")).items())
            },
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, field_name="metadata"))
        _reject_forbidden_authority(self.metadata, noun="IR learning campaign")
        task_ids = {item.task_id for item in self.tasks}
        _acyclic(
            task_ids,
            {
                item.task_id: tuple(dep for dep in item.depends_on if dep in task_ids)
                for item in self.tasks
            },
        )
        used_roles = {item.work_graph_role for item in self.tasks}
        unknown_roles = used_roles.difference(declared)
        if unknown_roles:
            raise ContractValidationError(
                "campaign tasks reference undeclared work-graph roles"
            )

    @property
    def campaign_revision(self) -> str:
        return self.content_id

    @property
    def task_revisions(self) -> tuple[CampaignTaskRevision, ...]:
        return tuple(
            revise_campaign_task(
                task,
                campaign_id=self.campaign_id,
                input_root_cid=self.input_root_cid,
                dependency_results=self.dependency_results,
            )
            for task in self.tasks
        )

    @property
    def lease_eligible_task_ids(self) -> tuple[str, ...]:
        return tuple(
            item.task_id for item in self.task_revisions if item.lease_eligible
        )

    @property
    def blocked_task_ids(self) -> tuple[str, ...]:
        return tuple(
            item.task_id for item in self.task_revisions if not item.lease_eligible
        )

    def revision_for(self, task_id: str) -> CampaignTaskRevision:
        for item in self.task_revisions:
            if item.task_id == task_id:
                return item
        raise ContractValidationError("unknown campaign task %s" % task_id)

    def task_by_id(self, task_id: str) -> CampaignBoardTask:
        for item in self.tasks:
            if item.task_id == task_id:
                return item
        raise ContractValidationError("unknown campaign task %s" % task_id)

    def project_dependencies(self) -> CampaignDependencyProjection:
        """Return the deterministic admission projection for this campaign."""

        revisions = {item.task_id: item for item in self.task_revisions}
        actions: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        unresolved: list[str] = []
        for task in self.tasks:
            revision = revisions[task.task_id]
            actions.append(
                {
                    "action_id": task.task_id,
                    "task_id": task.task_id,
                    "goal_id": task.parent_goal,
                    "subgoal_id": task.subgoal,
                    "role": task.work_graph_role.value,
                    "depends_on": list(task.depends_on),
                    "result_identity": task.result_identity,
                    "task_revision": revision.task_revision,
                    "lease_eligible": revision.lease_eligible,
                    "unresolved_dependency_outputs": list(
                        revision.unresolved_dependency_outputs
                    ),
                    "owned_paths": list(task.owned_paths),
                    "resource_profile": task.resource_profile.value,
                }
            )
            unresolved.extend(revision.unresolved_dependency_outputs)
            for dependency_id in task.depends_on:
                material = {
                    "action_id": task.task_id,
                    "depends_on_action_id": dependency_id,
                }
                edges.append({"dependency_id": content_identity(material), **material})
        return CampaignDependencyProjection(
            campaign_id=self.campaign_id,
            campaign_revision=self.campaign_revision,
            input_root_cid=self.input_root_cid,
            repository_tree_id=self.repository_tree_id,
            actions=tuple(actions),
            dependency_edges=tuple(edges),
            role_ids=tuple(item.role.value for item in self.roles),
            lease_eligible_task_ids=self.lease_eligible_task_ids,
            blocked_task_ids=self.blocked_task_ids,
            unresolved_result_ids=tuple(sorted(set(unresolved))),
        )

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "campaign_id": self.campaign_id,
                "input_root_cid": self.input_root_cid,
                "repository_tree_id": self.repository_tree_id,
                "owner_actor_id": self.owner_actor_id,
                "roles": [item.to_record() for item in self.roles],
                "tasks": [item.to_record() for item in self.tasks],
                "operations": [item.value for item in self.operations],
                "dependency_results": self.dependency_results,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IRLearningCampaign":
        _schema(payload, ACCEPTED_CAMPAIGN_SCHEMAS)
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "content_id",
                "identity",
                "campaign_revision",
                "campaign_id",
                "input_root_cid",
                "repository_tree_id",
                "owner_actor_id",
                "roles",
                "tasks",
                "operations",
                "dependency_results",
                "metadata",
            },
            noun="IR learning campaign",
        )
        _reject_forbidden_authority(payload, noun="IR learning campaign")
        version = payload.get("contract_version")
        if version not in (None, IR_LEARNING_CAMPAIGN_CONTRACT_VERSION):
            raise ContractValidationError("unsupported IR learning campaign contract version")
        result = cls(
            campaign_id=payload.get("campaign_id", ""),
            input_root_cid=payload.get("input_root_cid", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            owner_actor_id=payload.get("owner_actor_id", "supervisor"),
            roles=tuple(payload.get("roles") or ()),
            tasks=tuple(payload.get("tasks") or ()),
            operations=tuple(
                payload.get("operations") or STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS
            ),
            dependency_results=payload.get("dependency_results") or {},
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("campaign_revision") or payload.get("content_id")
        if claimed not in (None, "") and claimed != result.campaign_revision:
            raise ContractValidationError("IR learning campaign identity does not match payload")
        return result


@dataclass(frozen=True)
class CampaignOperationRequest(CampaignContract):
    """One create/plan/start/resume/status/steer/refill/proof-replay/compare/promote/reject/report call."""

    SCHEMA: ClassVar[str] = IR_LEARNING_CAMPAIGN_OPERATION_SCHEMA

    operation: CampaignOperationKind
    campaign: IRLearningCampaign
    caller: str
    task_id: str = ""
    dry_run: bool = False
    idempotency_key: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation",
            _enum(self.operation, CampaignOperationKind, field_name="operation"),
        )
        if isinstance(self.campaign, Mapping):
            object.__setattr__(self, "campaign", IRLearningCampaign.from_dict(self.campaign))
        if not isinstance(self.campaign, IRLearningCampaign):
            raise ContractValidationError("campaign must be an IRLearningCampaign")
        object.__setattr__(self, "caller", _text(self.caller, field_name="caller", required=True))
        object.__setattr__(self, "task_id", _text(self.task_id, field_name="task_id"))
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, field_name="idempotency_key"),
        )
        if not isinstance(self.dry_run, bool):
            raise ContractValidationError("dry_run must be a boolean")
        object.__setattr__(self, "parameters", _mapping(self.parameters, field_name="parameters"))
        _reject_forbidden_authority(self.parameters, noun="campaign operation")
        if self.task_id:
            self.campaign.task_by_id(self.task_id)
        declared = set(self.campaign.operations)
        if self.operation not in declared:
            if self.operation not in {
                CampaignOperationKind.START,
                CampaignOperationKind.RESUME,
            }:
                raise ContractValidationError("campaign does not declare the requested operation")
            missing = [
                item.value
                for item in REQUIRED_CAMPAIGN_OPERATIONS
                if item not in declared
            ]
            if missing:
                raise ContractValidationError(
                    "campaign does not declare the requested operation"
                )

    @property
    def control_operation(self) -> Operation:
        return CAMPAIGN_OPERATION_CONTROL_MAP[self.operation]

    @property
    def authority(self) -> OperationAuthority:
        if self.dry_run and self.operation in LEASE_REQUIRING_OPERATIONS:
            return OperationAuthority.PROPOSAL
        return CAMPAIGN_OPERATION_AUTHORITY[self.operation]

    @property
    def requires_lease(self) -> bool:
        return (not self.dry_run) and self.operation in LEASE_REQUIRING_OPERATIONS

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "operation": self.operation,
                "campaign_id": self.campaign.campaign_id,
                "campaign_revision": self.campaign.campaign_revision,
                "caller": self.caller,
                "task_id": self.task_id,
                "dry_run": self.dry_run,
                "idempotency_key": self.idempotency_key,
                "parameters": self.parameters,
                "control_operation": self.control_operation.value,
                "authority": self.authority.value,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignOperationRequest":
        _schema(payload, cls.SCHEMA)
        _reject_forbidden_authority(payload, noun="campaign operation")
        campaign = payload.get("campaign")
        if campaign is None:
            raise ContractValidationError("campaign is required")
        return cls(
            operation=payload.get("operation", CampaignOperationKind.STATUS),
            campaign=campaign,
            caller=payload.get("caller", ""),
            task_id=payload.get("task_id", ""),
            dry_run=payload.get("dry_run", False),
            idempotency_key=payload.get("idempotency_key", ""),
            parameters=payload.get("parameters") or {},
        )


@dataclass(frozen=True)
class CampaignOperationReceipt(CampaignContract):
    """Deterministic receipt for one campaign operation."""

    SCHEMA: ClassVar[str] = IR_LEARNING_CAMPAIGN_RECEIPT_SCHEMA

    operation: CampaignOperationKind
    status: CampaignOperationStatus
    campaign_id: str
    campaign_revision: str
    control_operation: Operation
    authority: OperationAuthority
    projection_id: str
    lease_eligible_task_ids: tuple[str, ...] = ()
    blocked_task_ids: tuple[str, ...] = ()
    unresolved_result_ids: tuple[str, ...] = ()
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation",
            _enum(self.operation, CampaignOperationKind, field_name="operation"),
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, CampaignOperationStatus, field_name="status"),
        )
        for name in ("campaign_id", "campaign_revision", "projection_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        if isinstance(self.control_operation, Operation):
            control = self.control_operation
        else:
            control = Operation(str(self.control_operation))
        object.__setattr__(self, "control_operation", control)
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, OperationAuthority, field_name="authority"),
        )
        expected_control = CAMPAIGN_OPERATION_CONTROL_MAP[self.operation]
        if self.control_operation is not expected_control:
            raise ContractValidationError(
                "campaign receipt control operation must match the closed catalog map"
            )
        if self.authority.rank > expected_control.authority.rank:
            raise ContractValidationError(
                "campaign receipt cannot raise authority above the mapped control operation"
            )
        for name in ("lease_eligible_task_ids", "blocked_task_ids", "unresolved_result_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), field_name=name))
        object.__setattr__(self, "message", _text(self.message, field_name="message"))
        object.__setattr__(self, "details", _mapping(self.details, field_name="details"))
        _reject_forbidden_authority(self.details, noun="campaign receipt")

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "operation": self.operation,
                "status": self.status,
                "campaign_id": self.campaign_id,
                "campaign_revision": self.campaign_revision,
                "control_operation": self.control_operation,
                "authority": self.authority,
                "projection_id": self.projection_id,
                "lease_eligible_task_ids": self.lease_eligible_task_ids,
                "blocked_task_ids": self.blocked_task_ids,
                "unresolved_result_ids": self.unresolved_result_ids,
                "message": self.message,
                "details": self.details,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignOperationReceipt":
        _schema(payload, cls.SCHEMA)
        _reject_forbidden_authority(payload, noun="campaign receipt")
        result = cls(
            operation=payload.get("operation", CampaignOperationKind.STATUS),
            status=payload.get("status", CampaignOperationStatus.REJECTED),
            campaign_id=payload.get("campaign_id", ""),
            campaign_revision=payload.get("campaign_revision", ""),
            control_operation=payload.get("control_operation", Operation.STATUS),
            authority=payload.get("authority", OperationAuthority.READ),
            projection_id=payload.get("projection_id", ""),
            lease_eligible_task_ids=tuple(payload.get("lease_eligible_task_ids") or ()),
            blocked_task_ids=tuple(payload.get("blocked_task_ids") or ()),
            unresolved_result_ids=tuple(payload.get("unresolved_result_ids") or ()),
            message=payload.get("message", ""),
            details=payload.get("details") or {},
        )
        _claimed_identity(payload, result.content_id, "campaign receipt")
        return result


def campaign_control_catalog() -> dict[str, Any]:
    """Return the closed campaign-to-control map without expanding Operation."""

    return {
        "schema": IR_LEARNING_CAMPAIGN_OPERATION_SCHEMA,
        "contract_version": IR_LEARNING_CAMPAIGN_CONTRACT_VERSION,
        "stable_operations": [item.value for item in STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS],
        "required_operations": [item.value for item in REQUIRED_CAMPAIGN_OPERATIONS],
        "operations": {
            kind.value: {
                "control_operation": operation.value,
                "authority": operation.authority.value,
                "requires_lease": kind in LEASE_REQUIRING_OPERATIONS,
                "in_read_catalog": operation in READ_OPERATIONS,
                "in_proposal_catalog": operation in PROPOSAL_OPERATIONS,
                "in_mutation_catalog": operation in MUTATION_OPERATIONS,
                "stable": kind in STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS,
            }
            for kind, operation in CAMPAIGN_OPERATION_CONTROL_MAP.items()
        },
        "expands_control_catalog": False,
    }


def assert_campaign_control_parity() -> None:
    """Fail closed if campaign operations would expand or re-rank control ops."""

    mapped = set(CAMPAIGN_OPERATION_CONTROL_MAP.values())
    if not mapped.issubset(set(Operation)):
        raise ContractValidationError("campaign operations must reuse existing control operations")
    for kind in CampaignOperationKind:
        if kind not in CAMPAIGN_OPERATION_CONTROL_MAP:
            raise ContractValidationError("campaign operation %s is unmapped" % kind.value)
        operation = CAMPAIGN_OPERATION_CONTROL_MAP[kind]
        if CAMPAIGN_OPERATION_AUTHORITY[kind] is not operation.authority:
            raise ContractValidationError(
                "campaign operation %s authority drifted from %s" % (kind.value, operation.value)
            )


assert_campaign_control_parity()
