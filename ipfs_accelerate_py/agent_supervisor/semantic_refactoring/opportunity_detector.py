"""SPAR-036 monolith opportunity detection and durable goal compilation.

This module extends current supervisor opportunity-detection with
``OpportunityPolicy@1``, ``MonolithFinding@1``, and ``DurableGoal@1``.
It consumes SPAR-007/008/009/010/011/014 sealed evidence, detects oversized
or structurally overloaded modules under a versioned policy, and compiles
exact findings, risks, autonomy, and acceptance into durable goals.

Findings and goals are nomination-only.  They cannot authorize a transition,
completion, or competing authority.  Vector, model, and heuristic evidence
cannot admit or hide a module.  Observational metadata is excluded from
identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)
from .partition_policy import (
    ANALYZER_ID as SPAR014_ANALYZER_ID,
    PartitionComparisonReceipt,
)


TASK_ID: Final[str] = "SPAR-036"
GOAL_ID: Final[str] = "SPAR-G071"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "opportunity detection"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.opportunity_detector@1"
)
COMPILER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.goal_compiler@1"
)

OPPORTUNITY_POLICY_INTERFACE: Final[str] = "OpportunityPolicy@1"
MONOLITH_FINDING_INTERFACE: Final[str] = "MonolithFinding@1"
OPPORTUNITY_DETECTION_RECEIPT_INTERFACE: Final[str] = (
    "OpportunityDetectionReceipt@1"
)
DURABLE_GOAL_INTERFACE: Final[str] = "DurableGoal@1"
GOAL_COMPILATION_RECEIPT_INTERFACE: Final[str] = "GoalCompilationReceipt@1"
MONOLITH_OPPORTUNITY_DETECTOR_INTERFACE: Final[str] = (
    "MonolithOpportunityDetector@1"
)
GOAL_COMPILER_INTERFACE: Final[str] = "GoalCompiler@1"

OPPORTUNITY_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/opportunity-policy@1"
)
MONOLITH_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/monolith-finding@1"
)
OPPORTUNITY_DETECTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/opportunity-detection-receipt@1"
)
DURABLE_GOAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/durable-goal@1"
)
GOAL_COMPILATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-compilation-receipt@1"
)

OPPORTUNITY_CONTRACT_VERSION: Final[str] = "1"
POLICY_ID: Final[str] = "opportunity-policy@1"
POLICY_REVISION: Final[str] = "1"

OPPORTUNITY_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
OPPORTUNITY_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
OPPORTUNITY_CAN_CREATE_AUTHORITY: Final[bool] = False
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

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_MODULES: Final[int] = 16_384
MAX_FINDINGS: Final[int] = 16_384
MAX_GOALS: Final[int] = 16_384
MAX_VIOLATIONS: Final[int] = 4_096
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_THRESHOLD: Final[int] = 1_000_000
MAX_LOC: Final[int] = 10_000_000
MAX_SCORE: Final[int] = 1_000_000

DEFAULT_LOC_THRESHOLD: Final[int] = 400
DEFAULT_PUBLIC_EXPORT_THRESHOLD: Final[int] = 8
DEFAULT_UNIQUE_STATE_OWNER_THRESHOLD: Final[int] = 2
DEFAULT_SCC_MEMBER_THRESHOLD: Final[int] = 8
DEFAULT_INITIALIZATION_BLOCK_THRESHOLD: Final[int] = 4
DEFAULT_RESPONSIBILITY_THRESHOLD: Final[int] = 3

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

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

THRESHOLD_FAMILIES: Final[tuple[str, ...]] = (
    "loc",
    "public_export",
    "unique_state_owner",
    "scc_member",
    "initialization_block",
    "responsibility",
)

ACCEPTANCE_IDS: Final[tuple[str, ...]] = (
    "exact-current-tree",
    "declared-effects",
    "independent-validation",
    "rollback",
    "authority-separation",
    "no-safety-floor-regression",
)

EXISTING_ADAPTER_AUTHORITIES: Final[tuple[str, ...]] = (
    "datasets_semantic",
    "kit_storage",
    "kit_vfs",
    "accelerator_supervisor",
    "accelerator_runtime",
    "spar_narrow",
)

_UNIQUE_OWNER: Final[str] = "unique"


class OpportunityDetectorError(ValueError):
    """Fail-closed violation of a SPAR-036 opportunity-detection contract."""


class ThresholdFamily(str, Enum):
    LOC = "loc"
    PUBLIC_EXPORT = "public_export"
    UNIQUE_STATE_OWNER = "unique_state_owner"
    SCC_MEMBER = "scc_member"
    INITIALIZATION_BLOCK = "initialization_block"
    RESPONSIBILITY = "responsibility"


class FindingKind(str, Enum):
    OVERSIZED = "oversized"
    STRUCTURALLY_OVERLOADED = "structurally_overloaded"
    MIXED = "mixed"


class RiskKind(str, Enum):
    OVERSIZED_LOC = "oversized_loc"
    OVERSIZED_SCC = "oversized_scc"
    UNIQUE_STATE_OWNER = "unique_state_owner"
    INITIALIZATION_ORDER = "initialization_order"
    PUBLIC_COMPATIBILITY = "public_compatibility"
    UNRESOLVED_FRONTIER = "unresolved_frontier"
    OPAQUE_FRONTIER = "opaque_frontier"
    INCOMPLETE_CONTRACT = "incomplete_contract"


class AutonomyTier(str, Enum):
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class ResponsibilityFamily(str, Enum):
    STATE = "state"
    INITIALIZATION = "initialization"
    PUBLIC = "public"
    SCC = "scc"
    FRONTIER = "frontier"


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise OpportunityDetectorError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise OpportunityDetectorError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise OpportunityDetectorError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise OpportunityDetectorError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise OpportunityDetectorError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise OpportunityDetectorError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise OpportunityDetectorError(f"{name} must be a boolean")
    return value


def _nat(value: Any, name: str, *, limit: int) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise OpportunityDetectorError(f"{name} must be an integer")
    if value < 0:
        raise OpportunityDetectorError(f"{name} must be non-negative")
    if value > limit:
        raise OpportunityDetectorError(f"{name} exceeds maximum")
    return value


def _positive(value: Any, name: str, *, limit: int) -> int:
    number = _nat(value, name, limit=limit)
    if number < 1:
        raise OpportunityDetectorError(f"{name} must be a positive integer")
    return number


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise OpportunityDetectorError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise OpportunityDetectorError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise OpportunityDetectorError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise OpportunityDetectorError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise OpportunityDetectorError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise OpportunityDetectorError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise OpportunityDetectorError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise OpportunityDetectorError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise OpportunityDetectorError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise OpportunityDetectorError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise OpportunityDetectorError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise OpportunityDetectorError(f"unknown {name}: {text}") from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise OpportunityDetectorError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


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
    raise OpportunityDetectorError(
        f"unsupported projected type {type(value).__name__}"
    )


def _sequence_maps(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        projected = _project(value)
        if isinstance(projected, list):
            value = projected
        else:
            raise OpportunityDetectorError(f"{name} must be a list")
    items = []
    for item in value:
        projected = _project(item)
        if not isinstance(projected, dict):
            raise OpportunityDetectorError(f"{name} items must be objects")
        items.append(projected)
    if len(items) > MAX_MEMBERS:
        raise OpportunityDetectorError(f"{name} exceeds maximum length")
    return tuple(items)


def opportunity_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _threshold_table(values: Any) -> tuple[dict[str, int | str], ...]:
    defaults = {
        ThresholdFamily.LOC.value: DEFAULT_LOC_THRESHOLD,
        ThresholdFamily.PUBLIC_EXPORT.value: DEFAULT_PUBLIC_EXPORT_THRESHOLD,
        ThresholdFamily.UNIQUE_STATE_OWNER.value: DEFAULT_UNIQUE_STATE_OWNER_THRESHOLD,
        ThresholdFamily.SCC_MEMBER.value: DEFAULT_SCC_MEMBER_THRESHOLD,
        ThresholdFamily.INITIALIZATION_BLOCK.value: DEFAULT_INITIALIZATION_BLOCK_THRESHOLD,
        ThresholdFamily.RESPONSIBILITY.value: DEFAULT_RESPONSIBILITY_THRESHOLD,
    }
    if values in (None, (), {}):
        return tuple(
            {"family": family, "limit": defaults[family]}
            for family in THRESHOLD_FAMILIES
        )
    if isinstance(values, Mapping):
        missing = [item for item in THRESHOLD_FAMILIES if item not in values]
        extra = [str(key) for key in values if str(key) not in THRESHOLD_FAMILIES]
        if missing:
            raise OpportunityDetectorError(
                f"opportunity policy must include every threshold family: {missing}"
            )
        if extra:
            raise OpportunityDetectorError(f"unknown threshold family: {sorted(extra)}")
        return tuple(
            {
                "family": family,
                "limit": _positive(
                    values[family], f"thresholds.{family}", limit=MAX_THRESHOLD
                ),
            }
            for family in THRESHOLD_FAMILIES
        )
    if not isinstance(values, (list, tuple)):
        raise OpportunityDetectorError("thresholds must be a list or object")
    if len(values) != len(THRESHOLD_FAMILIES):
        raise OpportunityDetectorError(
            "opportunity policy must include every threshold family"
        )
    rows: list[dict[str, int | str]] = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            family = _enum(item.get("family"), ThresholdFamily, "family")
            if family != THRESHOLD_FAMILIES[index]:
                raise OpportunityDetectorError(
                    "threshold families must remain in canonical order"
                )
            rows.append(
                {
                    "family": family,
                    "limit": _positive(
                        item.get("limit"), f"thresholds.{family}", limit=MAX_THRESHOLD
                    ),
                }
            )
        else:
            rows.append(
                {
                    "family": THRESHOLD_FAMILIES[index],
                    "limit": _positive(
                        item, f"thresholds[{index}]", limit=MAX_THRESHOLD
                    ),
                }
            )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class OpportunityPolicy:
    """Complete versioned oversized/overload policy. Not ranking authority."""

    thresholds: Mapping[str, int] | Sequence[int] | Sequence[Mapping[str, Any]] = ()
    vector_may_admit: bool = False
    policy_id: str = POLICY_ID
    policy_revision: str = POLICY_REVISION

    interface: ClassVar[str] = OPPORTUNITY_POLICY_INTERFACE
    schema: ClassVar[str] = OPPORTUNITY_POLICY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "policy_id",
            "policy_revision",
            "thresholds",
            "vector_may_admit",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "soft_signals_cannot_override_hard_constraints",
            "policy_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )
        object.__setattr__(self, "thresholds", _threshold_table(self.thresholds))
        object.__setattr__(
            self, "vector_may_admit", _bool(self.vector_may_admit, "vector_may_admit")
        )
        if self.vector_may_admit:
            raise OpportunityDetectorError("vector scores cannot admit a module")
        if self.policy_id != POLICY_ID:
            raise OpportunityDetectorError("policy_id must remain opportunity-policy@1")
        if self.policy_revision != POLICY_REVISION:
            raise OpportunityDetectorError("policy_revision must remain 1")

    def limit_for(self, family: str) -> int:
        for item in self.thresholds:
            if item["family"] == family:
                return int(item["limit"])
        raise OpportunityDetectorError(f"missing threshold family: {family}")

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
    def soft_signals_cannot_override_hard_constraints(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": OPPORTUNITY_POLICY_SCHEMA,
            "interface": OPPORTUNITY_POLICY_INTERFACE,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "thresholds": [dict(item) for item in self.thresholds],
            "vector_may_admit": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "soft_signals_cannot_override_hard_constraints": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def policy_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["policy_cid"] = self.policy_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpportunityPolicy":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("policy_cid")
        if payload.pop("schema") != OPPORTUNITY_POLICY_SCHEMA:
            raise OpportunityDetectorError("unsupported OpportunityPolicy schema")
        if payload.pop("interface") != OPPORTUNITY_POLICY_INTERFACE:
            raise OpportunityDetectorError("unsupported OpportunityPolicy interface")
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "vector_may_admit",
        ):
            if payload.pop(flag) is not False:
                raise OpportunityDetectorError(f"policy cannot claim {flag}")
        if payload.pop("soft_signals_cannot_override_hard_constraints") is not True:
            raise OpportunityDetectorError(
                "policy must retain soft-cannot-override-hard"
            )
        result = cls(**payload)
        _verify_cid(claimed, result.policy_cid, "OpportunityPolicy policy_cid")
        return result


def default_opportunity_policy() -> OpportunityPolicy:
    return OpportunityPolicy()


def _coerce_policy(
    value: OpportunityPolicy | Mapping[str, Any] | None,
) -> OpportunityPolicy:
    if value is None:
        return default_opportunity_policy()
    if isinstance(value, OpportunityPolicy):
        return value
    if isinstance(value, Mapping):
        if "policy_cid" in value:
            return OpportunityPolicy.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "policy_cid",
                "can_authorize_transition",
                "can_authorize_completion",
                "can_create_authority",
                "soft_signals_cannot_override_hard_constraints",
            }
        }
        return OpportunityPolicy(**payload)
    raise OpportunityDetectorError("policy must be an OpportunityPolicy")


@dataclass(frozen=True, slots=True)
class _ModuleView:
    module_id: str
    loc: int
    member_ids: tuple[str, ...]
    public_export_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SccLite:
    snapshot_cid: str
    node_to_scc: Mapping[str, str]
    members: Mapping[str, tuple[str, ...]]
    cyclic: Mapping[str, bool]
    oversized: Mapping[str, bool]

    def closed_members(self, member_ids: Sequence[str]) -> tuple[str, ...]:
        unknown = [item for item in member_ids if item not in self.node_to_scc]
        if unknown:
            raise OpportunityDetectorError(
                f"unknown SCC member: {sorted(set(unknown))}"
            )
        closed: list[str] = []
        for member in member_ids:
            closed.extend(self.members[self.node_to_scc[member]])
        return tuple(sorted(set(closed)))

    def oversized_in(self, member_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.node_to_scc[item]
                    for item in member_ids
                    if self.oversized.get(self.node_to_scc[item], False)
                }
            )
        )


@dataclass(frozen=True, slots=True)
class OpportunityEvidence:
    """Sealed residual evidence for SPAR-036 detection. Body-free."""

    tree_id: str
    modules: tuple[_ModuleView, ...]
    evidence_cids: tuple[str, ...]
    scc: _SccLite | None = None
    unresolved_subjects: frozenset[str] = frozenset()
    opaque_subjects: frozenset[str] = frozenset()
    unique_owners: Mapping[str, tuple[str, ...]] = ()
    init_blocks: Mapping[str, tuple[str, ...]] = ()
    compatibility_subjects: frozenset[str] = frozenset()
    incomplete_contract_subjects: frozenset[str] = frozenset()
    oversized_partition_subjects: frozenset[str] = frozenset()
    vector_scores: Mapping[str, int] = ()


def _module_view(item: Mapping[str, Any]) -> _ModuleView:
    module_id = _text(item.get("module_id"), "module_id")
    loc = _nat(item.get("loc"), "loc", limit=MAX_LOC)
    member_ids = _unique_sorted_text(
        list(item.get("member_ids") or (module_id,)),
        "member_ids",
        limit=MAX_MEMBERS,
    )
    if not member_ids:
        member_ids = (module_id,)
    public_export_ids = _unique_sorted_text(
        list(item.get("public_export_ids") or ()),
        "public_export_ids",
        limit=MAX_MEMBERS,
    )
    return _ModuleView(
        module_id=module_id,
        loc=loc,
        member_ids=member_ids,
        public_export_ids=public_export_ids,
    )


def _build_scc_lite(snapshot: Mapping[str, Any] | None) -> _SccLite | None:
    if snapshot is None:
        return None
    payload = _mapping(snapshot, "scc_snapshot")
    snapshot_cid = _cid(
        payload.get("snapshot_cid") or payload.get("graph_view_cid"),
        "scc_snapshot.snapshot_cid",
    )
    components = _sequence_maps(payload.get("components"), "scc_snapshot.components")
    if not components:
        raise OpportunityDetectorError("scc_snapshot.components must be nonempty")
    node_to_scc: dict[str, str] = {}
    members: dict[str, tuple[str, ...]] = {}
    cyclic: dict[str, bool] = {}
    oversized: dict[str, bool] = {}
    for item in components:
        scc_id = _cid(item.get("scc_id"), "scc_id")
        member_ids = _unique_sorted_text(
            list(item.get("member_ids") or ()), "member_ids", limit=MAX_MEMBERS
        )
        if not member_ids:
            raise OpportunityDetectorError("SCC component requires members")
        if scc_id in members:
            raise OpportunityDetectorError("duplicate SCC identity")
        for member in member_ids:
            if member in node_to_scc:
                raise OpportunityDetectorError("SCC members must be partitioned")
            node_to_scc[member] = scc_id
        members[scc_id] = member_ids
        cyclic[scc_id] = _bool(item.get("cyclic", False), "cyclic")
        oversized_flag = _bool(item.get("oversized", False), "oversized")
        if oversized_flag and not cyclic[scc_id]:
            raise OpportunityDetectorError(
                "oversized flag applies only to cyclic SCCs"
            )
        oversized[scc_id] = oversized_flag
    return _SccLite(
        snapshot_cid=snapshot_cid,
        node_to_scc=node_to_scc,
        members=members,
        cyclic=cyclic,
        oversized=oversized,
    )


def _unique_owners(state: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    if state is None:
        return {}
    payload = _mapping(state, "state")
    owners: dict[str, tuple[str, ...]] = {}
    for item in _sequence_maps(payload.get("owners"), "owners"):
        uniqueness = _text(item.get("uniqueness", ""), "uniqueness", empty=True)
        if uniqueness != _UNIQUE_OWNER:
            continue
        owner_id = _text(item.get("owner_id"), "owner_id")
        members = _unique_sorted_text(
            list(item.get("member_ids") or ()),
            "owner member_ids",
            limit=MAX_MEMBERS,
        )
        if not members:
            members = (
                _text(
                    item.get("owning_symbol_id") or owner_id, "owning_symbol_id"
                ),
            )
        owners[owner_id] = members
    return owners


def _init_blocks(initialization: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    if initialization is None:
        return {}
    payload = _mapping(initialization, "initialization")
    blocks: dict[str, tuple[str, ...]] = {}
    for item in _sequence_maps(payload.get("blocks"), "blocks"):
        block_id = _text(item.get("block_id"), "block_id")
        members = _unique_sorted_text(
            list(item.get("member_ids") or ()),
            "block member_ids",
            limit=MAX_MEMBERS,
        )
        if not members:
            raise OpportunityDetectorError("initialization block requires members")
        blocks[block_id] = members
    return blocks


def _subject_set(values: Any, name: str) -> frozenset[str]:
    if values in (None, (), []):
        return frozenset()
    return frozenset(_unique_sorted_text(list(values), name, limit=MAX_MEMBERS))


def _compatibility(
    compatibility: Mapping[str, Any] | None,
) -> tuple[frozenset[str], frozenset[str]]:
    if compatibility is None:
        return frozenset(), frozenset()
    payload = _mapping(compatibility, "compatibility")
    obligations = _sequence_maps(payload.get("obligations"), "obligations")
    consumers = _sequence_maps(payload.get("consumers"), "consumers")
    subjects = {
        _text(item.get("subject_id"), "subject_id")
        for item in obligations
        if item.get("subject_id")
    }
    consumer_subjects = {
        _text(item.get("subject_id"), "consumer subject_id")
        for item in consumers
        if item.get("subject_id")
    }
    incomplete = frozenset(subjects - consumer_subjects)
    return frozenset(subjects), incomplete


def _partition_oversized(
    receipt: Mapping[str, Any] | PartitionComparisonReceipt | None,
    *,
    tree_id: str,
) -> frozenset[str]:
    if receipt is None:
        return frozenset()
    if isinstance(receipt, PartitionComparisonReceipt):
        compared = receipt
    elif isinstance(receipt, Mapping):
        compared = PartitionComparisonReceipt.from_dict(receipt)
    else:
        raise OpportunityDetectorError("partition_receipt must be a comparison receipt")
    if compared.tree_id != tree_id:
        raise OpportunityDetectorError("partition_receipt tree_id does not match")
    if compared.analyzer_id != SPAR014_ANALYZER_ID:
        raise OpportunityDetectorError("partition_receipt analyzer must remain SPAR-014")
    subjects: set[str] = set()
    for breakdown in compared.breakdowns:
        scc = next(
            (item for item in breakdown.hard_results if item.family == "scc"),
            None,
        )
        if scc is None or scc.passed:
            continue
        if "oversized_cycle" not in scc.violations:
            continue
        subjects.update(scc.witness_ids)
    return frozenset(subjects)


def _vector_scores(values: Any) -> dict[str, int]:
    if values in (None, (), {}):
        return {}
    if not isinstance(values, Mapping):
        raise OpportunityDetectorError("vector_scores must be an object")
    scores: dict[str, int] = {}
    for key, value in values.items():
        scores[_text(key, "vector_scores key")] = _nat(
            value, "vector_scores value", limit=MAX_SCORE
        )
    return scores


def _evidence_cids(*cids: str) -> tuple[str, ...]:
    ordered = tuple(sorted({item for item in cids if item}))
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise OpportunityDetectorError("evidence_cids exceeds maximum length")
    return ordered


def compile_opportunity_evidence(evidence: Mapping[str, Any] | Any) -> OpportunityEvidence:
    """Compile sealed SPAR-007/008/009/010/011/014 evidence. Fail closed."""

    payload = _mapping(evidence, "opportunity evidence")
    if any(key in payload for key in _NON_ADMITTING_EVIDENCE):
        raise OpportunityDetectorError(
            "vector, model, and heuristic evidence cannot admit a module"
        )
    tree_id = _tree_id(payload.get("tree_id"))
    graph_view = payload.get("graph_view")
    graph_payload = _mapping(graph_view, "graph_view") if graph_view is not None else {}
    module_rows = payload.get("modules")
    if module_rows in (None, ()):
        module_rows = graph_payload.get("modules")
    modules = tuple(_module_view(item) for item in _sequence_maps(module_rows, "modules"))
    if not modules:
        raise OpportunityDetectorError("opportunity detection requires modules")
    if len(modules) > MAX_MODULES:
        raise OpportunityDetectorError("modules exceed maximum length")
    seen = [item.module_id for item in modules]
    if len(seen) != len(set(seen)):
        raise OpportunityDetectorError("duplicate module identity")
    scc = _build_scc_lite(payload.get("scc_snapshot"))
    frontier = payload.get("frontier")
    frontier_payload = _mapping(frontier, "frontier") if frontier is not None else {}
    unresolved = _subject_set(
        frontier_payload.get("unresolved_subject_ids"), "unresolved_subject_ids"
    )
    opaque = _subject_set(
        frontier_payload.get("opaque_subject_ids"), "opaque_subject_ids"
    )
    owners = _unique_owners(payload.get("state") or payload.get("state_ownership"))
    blocks = _init_blocks(payload.get("initialization"))
    compat_subjects, incomplete = _compatibility(payload.get("compatibility"))
    partition_subjects = _partition_oversized(
        payload.get("partition_receipt"), tree_id=tree_id
    )
    scores = _vector_scores(payload.get("vector_scores"))
    cids = _evidence_cids(
        graph_payload.get("graph_view_cid") or "",
        scc.snapshot_cid if scc is not None else "",
        frontier_payload.get("frontier_cid") or "",
        (_mapping(payload.get("state") or payload.get("state_ownership") or {}, "state").get("graph_cid") or "")
        if payload.get("state") or payload.get("state_ownership")
        else "",
        (_mapping(payload.get("initialization") or {}, "initialization").get("graph_cid") or "")
        if payload.get("initialization")
        else "",
        (_mapping(payload.get("compatibility") or {}, "compatibility").get("inventory_cid") or "")
        if payload.get("compatibility")
        else "",
        getattr(payload.get("partition_receipt"), "receipt_cid", "")
        if not isinstance(payload.get("partition_receipt"), Mapping)
        else payload.get("partition_receipt", {}).get("receipt_cid") or "",
    )
    return OpportunityEvidence(
        tree_id=tree_id,
        modules=modules,
        evidence_cids=cids,
        scc=scc,
        unresolved_subjects=unresolved,
        opaque_subjects=opaque,
        unique_owners=owners,
        init_blocks=blocks,
        compatibility_subjects=compat_subjects,
        incomplete_contract_subjects=incomplete,
        oversized_partition_subjects=partition_subjects,
        vector_scores=scores,
    )


def _intersecting(owner_members: Mapping[str, tuple[str, ...]], members: Sequence[str]) -> tuple[str, ...]:
    present = set(members)
    return tuple(
        sorted(
            owner_id
            for owner_id, owned in owner_members.items()
            if present.intersection(owned)
        )
    )


def _subject_hit(subjects: frozenset[str], members: Sequence[str], module_id: str) -> bool:
    if module_id in subjects:
        return True
    return any(item in subjects for item in members)


def _responsibility_count(
    *,
    unique_state_owner_count: int,
    initialization_block_count: int,
    public_export_count: int,
    scc_member_count: int,
    scc_cyclic: bool,
    scc_oversized: bool,
    unresolved: bool,
) -> tuple[int, tuple[str, ...]]:
    families: list[str] = []
    if unique_state_owner_count:
        families.append(ResponsibilityFamily.STATE.value)
    if initialization_block_count:
        families.append(ResponsibilityFamily.INITIALIZATION.value)
    if public_export_count:
        families.append(ResponsibilityFamily.PUBLIC.value)
    if scc_cyclic or scc_oversized:
        families.append(ResponsibilityFamily.SCC.value)
    if unresolved:
        families.append(ResponsibilityFamily.FRONTIER.value)
    ordered = tuple(sorted(set(families)))
    return len(ordered), ordered


def _autonomy_tier(
    *,
    opaque: bool,
    unresolved: bool,
    unique_state_owner_count: int,
    initialization_block_count: int,
    initialization_limit: int,
) -> str:
    if opaque:
        return AutonomyTier.E.value
    if unresolved:
        return AutonomyTier.D.value
    if (
        unique_state_owner_count >= 2
        or initialization_block_count >= initialization_limit
    ):
        return AutonomyTier.C.value
    return AutonomyTier.B.value


def _risks(
    *,
    oversized_loc: bool,
    oversized_scc: bool,
    unique_state_owner: bool,
    initialization_order: bool,
    public_compatibility: bool,
    unresolved: bool,
    opaque: bool,
    incomplete_contract: bool,
) -> tuple[str, ...]:
    risks: list[str] = []
    if oversized_loc:
        risks.append(RiskKind.OVERSIZED_LOC.value)
    if oversized_scc:
        risks.append(RiskKind.OVERSIZED_SCC.value)
    if unique_state_owner:
        risks.append(RiskKind.UNIQUE_STATE_OWNER.value)
    if initialization_order:
        risks.append(RiskKind.INITIALIZATION_ORDER.value)
    if public_compatibility:
        risks.append(RiskKind.PUBLIC_COMPATIBILITY.value)
    if unresolved:
        risks.append(RiskKind.UNRESOLVED_FRONTIER.value)
    if opaque:
        risks.append(RiskKind.OPAQUE_FRONTIER.value)
    if incomplete_contract:
        risks.append(RiskKind.INCOMPLETE_CONTRACT.value)
    return tuple(sorted(set(risks)))


@dataclass(frozen=True, slots=True)
class MonolithFinding:
    """Exact oversized/overload finding. Nomination-only."""

    tree_id: str
    module_id: str
    kind: FindingKind | str
    loc: int
    public_export_count: int
    unique_state_owner_count: int
    scc_member_count: int
    initialization_block_count: int
    responsibility_count: int
    oversized: bool
    structurally_overloaded: bool
    risks: Sequence[str] = ()
    evidence_cids: Sequence[str] = ()
    autonomy_tier: AutonomyTier | str = AutonomyTier.B
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = MONOLITH_FINDING_INTERFACE
    schema: ClassVar[str] = MONOLITH_FINDING_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "module_id",
            "kind",
            "loc",
            "public_export_count",
            "unique_state_owner_count",
            "scc_member_count",
            "initialization_block_count",
            "responsibility_count",
            "oversized",
            "structurally_overloaded",
            "risks",
            "evidence_cids",
            "autonomy_tier",
            "analyzer_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "finding_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        kind = _enum(self.kind, FindingKind, "kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "loc", _nat(self.loc, "loc", limit=MAX_LOC))
        object.__setattr__(
            self,
            "public_export_count",
            _nat(self.public_export_count, "public_export_count", limit=MAX_MEMBERS),
        )
        object.__setattr__(
            self,
            "unique_state_owner_count",
            _nat(
                self.unique_state_owner_count,
                "unique_state_owner_count",
                limit=MAX_MEMBERS,
            ),
        )
        object.__setattr__(
            self,
            "scc_member_count",
            _nat(self.scc_member_count, "scc_member_count", limit=MAX_MEMBERS),
        )
        object.__setattr__(
            self,
            "initialization_block_count",
            _nat(
                self.initialization_block_count,
                "initialization_block_count",
                limit=MAX_MEMBERS,
            ),
        )
        object.__setattr__(
            self,
            "responsibility_count",
            _nat(self.responsibility_count, "responsibility_count", limit=MAX_MEMBERS),
        )
        object.__setattr__(self, "oversized", _bool(self.oversized, "oversized"))
        object.__setattr__(
            self,
            "structurally_overloaded",
            _bool(self.structurally_overloaded, "structurally_overloaded"),
        )
        object.__setattr__(
            self,
            "risks",
            tuple(_enum(item, RiskKind, "risks") for item in self.risks),
        )
        if len(self.risks) != len(set(self.risks)):
            raise OpportunityDetectorError("risks must not contain duplicates")
        object.__setattr__(
            self,
            "risks",
            tuple(sorted(self.risks)),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            tuple(sorted(_cid(item, "evidence_cids") for item in self.evidence_cids)),
        )
        if len(self.evidence_cids) != len(set(self.evidence_cids)):
            raise OpportunityDetectorError("evidence_cids must not contain duplicates")
        object.__setattr__(
            self,
            "autonomy_tier",
            _enum(self.autonomy_tier, AutonomyTier, "autonomy_tier"),
        )
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise OpportunityDetectorError(
                "analyzer_id must remain the SPAR-036 analyzer"
            )
        object.__setattr__(self, "analyzer_id", analyzer)
        if not self.oversized and not self.structurally_overloaded:
            raise OpportunityDetectorError(
                "finding must be oversized or structurally overloaded"
            )
        if self.oversized and self.structurally_overloaded and kind != FindingKind.MIXED.value:
            raise OpportunityDetectorError("mixed findings must use kind mixed")
        if self.oversized and not self.structurally_overloaded and kind != FindingKind.OVERSIZED.value:
            raise OpportunityDetectorError("oversized-only findings must use kind oversized")
        if (
            self.structurally_overloaded
            and not self.oversized
            and kind != FindingKind.STRUCTURALLY_OVERLOADED.value
        ):
            raise OpportunityDetectorError(
                "overload-only findings must use kind structurally_overloaded"
            )
        if not self.risks:
            raise OpportunityDetectorError("finding must retain at least one risk")
        if self.autonomy_tier == "A":
            raise OpportunityDetectorError("opportunity findings cannot claim Tier A")

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MONOLITH_FINDING_SCHEMA,
            "interface": MONOLITH_FINDING_INTERFACE,
            "tree_id": self.tree_id,
            "module_id": self.module_id,
            "kind": self.kind,
            "loc": self.loc,
            "public_export_count": self.public_export_count,
            "unique_state_owner_count": self.unique_state_owner_count,
            "scc_member_count": self.scc_member_count,
            "initialization_block_count": self.initialization_block_count,
            "responsibility_count": self.responsibility_count,
            "oversized": self.oversized,
            "structurally_overloaded": self.structurally_overloaded,
            "risks": list(self.risks),
            "evidence_cids": list(self.evidence_cids),
            "autonomy_tier": self.autonomy_tier,
            "analyzer_id": self.analyzer_id,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def finding_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["finding_cid"] = self.finding_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonolithFinding":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("finding_cid")
        if payload.pop("schema") != MONOLITH_FINDING_SCHEMA:
            raise OpportunityDetectorError("unsupported MonolithFinding schema")
        if payload.pop("interface") != MONOLITH_FINDING_INTERFACE:
            raise OpportunityDetectorError("unsupported MonolithFinding interface")
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
        ):
            if payload.pop(flag) is not False:
                raise OpportunityDetectorError(f"finding cannot claim {flag}")
        result = cls(**payload)
        _verify_cid(claimed, result.finding_cid, "MonolithFinding finding_cid")
        return result


def _evaluate_module(
    module: _ModuleView,
    *,
    facts: OpportunityEvidence,
    policy: OpportunityPolicy,
) -> tuple[MonolithFinding | None, str]:
    members = module.member_ids
    loc_limit = policy.limit_for(ThresholdFamily.LOC.value)
    export_limit = policy.limit_for(ThresholdFamily.PUBLIC_EXPORT.value)
    owner_limit = policy.limit_for(ThresholdFamily.UNIQUE_STATE_OWNER.value)
    scc_limit = policy.limit_for(ThresholdFamily.SCC_MEMBER.value)
    init_limit = policy.limit_for(ThresholdFamily.INITIALIZATION_BLOCK.value)
    responsibility_limit = policy.limit_for(ThresholdFamily.RESPONSIBILITY.value)

    scc_ids: tuple[str, ...] = ()
    if facts.scc is not None:
        closed = facts.scc.closed_members(members)
        scc_member_count = len(closed)
        oversized_sccs = facts.scc.oversized_in(members)
        scc_ids = tuple(sorted({facts.scc.node_to_scc[item] for item in members}))
        scc_oversized = bool(oversized_sccs)
        scc_cyclic = any(
            facts.scc.cyclic.get(facts.scc.node_to_scc[item], False) for item in members
        )
    else:
        scc_member_count = len(members)
        scc_oversized = False
        scc_cyclic = False

    if _subject_hit(facts.oversized_partition_subjects, members, module.module_id) or any(
        item in facts.oversized_partition_subjects for item in scc_ids
    ):
        scc_oversized = True

    owners = _intersecting(facts.unique_owners, members)
    unique_state_owner_count = len(owners)
    init_hits = _intersecting(facts.init_blocks, members)
    initialization_block_count = len(init_hits)
    public_export_ids = set(module.public_export_ids)
    if _subject_hit(facts.compatibility_subjects, members, module.module_id):
        public_export_ids.add(module.module_id)
    public_export_count = len(public_export_ids)
    unresolved = _subject_hit(facts.unresolved_subjects, members, module.module_id)
    opaque = _subject_hit(facts.opaque_subjects, members, module.module_id)
    incomplete = _subject_hit(
        facts.incomplete_contract_subjects, members, module.module_id
    )
    responsibility_count, _families = _responsibility_count(
        unique_state_owner_count=unique_state_owner_count,
        initialization_block_count=initialization_block_count,
        public_export_count=public_export_count,
        scc_member_count=scc_member_count,
        scc_cyclic=scc_cyclic,
        scc_oversized=scc_oversized,
        unresolved=unresolved,
    )

    oversized_loc = module.loc >= loc_limit
    oversized_scc = scc_oversized or scc_member_count >= scc_limit
    oversized = oversized_loc or oversized_scc
    structurally_overloaded = (
        public_export_count >= export_limit
        or unique_state_owner_count >= owner_limit
        or initialization_block_count >= init_limit
        or responsibility_count >= responsibility_limit
    )
    if not oversized and not structurally_overloaded:
        return None, module.module_id

    if oversized and structurally_overloaded:
        kind = FindingKind.MIXED.value
    elif oversized:
        kind = FindingKind.OVERSIZED.value
    else:
        kind = FindingKind.STRUCTURALLY_OVERLOADED.value

    risks = _risks(
        oversized_loc=oversized_loc,
        oversized_scc=oversized_scc,
        unique_state_owner=unique_state_owner_count >= owner_limit,
        initialization_order=initialization_block_count >= init_limit,
        public_compatibility=public_export_count >= export_limit,
        unresolved=unresolved,
        opaque=opaque,
        incomplete_contract=incomplete,
    )
    finding = MonolithFinding(
        tree_id=facts.tree_id,
        module_id=module.module_id,
        kind=kind,
        loc=module.loc,
        public_export_count=public_export_count,
        unique_state_owner_count=unique_state_owner_count,
        scc_member_count=scc_member_count,
        initialization_block_count=initialization_block_count,
        responsibility_count=responsibility_count,
        oversized=oversized,
        structurally_overloaded=structurally_overloaded,
        risks=risks,
        evidence_cids=facts.evidence_cids,
        autonomy_tier=_autonomy_tier(
            opaque=opaque,
            unresolved=unresolved,
            unique_state_owner_count=unique_state_owner_count,
            initialization_block_count=initialization_block_count,
            initialization_limit=init_limit,
        ),
    )
    return finding, module.module_id


@dataclass(frozen=True, slots=True)
class OpportunityDetectionReceipt:
    """Deterministic detection receipt. Nomination-only."""

    tree_id: str
    policy: OpportunityPolicy | Mapping[str, Any]
    findings: Sequence[MonolithFinding | Mapping[str, Any]]
    below_threshold_module_ids: Sequence[str] = ()
    evidence_cids: Sequence[str] = ()
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = OPPORTUNITY_DETECTION_RECEIPT_INTERFACE
    schema: ClassVar[str] = OPPORTUNITY_DETECTION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "policy",
            "findings",
            "below_threshold_module_ids",
            "evidence_cids",
            "analyzer_id",
            "finding_cids",
            "negative_evidence_ids",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "soft_signals_cannot_override_hard_constraints",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise OpportunityDetectorError(
                "analyzer_id must remain the SPAR-036 analyzer"
            )
        policy = _coerce_policy(self.policy)
        findings = tuple(_coerce_finding(item) for item in self.findings)
        if len(findings) > MAX_FINDINGS:
            raise OpportunityDetectorError("findings exceed maximum length")
        tree_id = _tree_id(self.tree_id)
        mismatched = [item.module_id for item in findings if item.tree_id != tree_id]
        if mismatched:
            raise OpportunityDetectorError("finding tree_id does not match receipt")
        seen = [item.finding_cid for item in findings]
        if len(seen) != len(set(seen)):
            raise OpportunityDetectorError("duplicate finding identity")
        modules = [item.module_id for item in findings]
        if len(modules) != len(set(modules)):
            raise OpportunityDetectorError("duplicate finding module identity")
        findings = tuple(sorted(findings, key=lambda item: item.finding_cid))
        below = _unique_sorted_text(
            list(self.below_threshold_module_ids),
            "below_threshold_module_ids",
            limit=MAX_MODULES,
        )
        overlap = set(modules) & set(below)
        if overlap:
            raise OpportunityDetectorError(
                "soft scores never override a hard constraint"
            )
        evidence = tuple(
            sorted(_cid(item, "evidence_cids") for item in self.evidence_cids)
        )
        if len(evidence) != len(set(evidence)):
            raise OpportunityDetectorError("evidence_cids must not contain duplicates")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "below_threshold_module_ids", below)
        object.__setattr__(self, "evidence_cids", evidence)
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def finding_cids(self) -> tuple[str, ...]:
        return tuple(item.finding_cid for item in self.findings)

    @property
    def negative_evidence_ids(self) -> tuple[str, ...]:
        return self.below_threshold_module_ids

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
    def projection_is_authority(self) -> bool:
        return False

    @property
    def soft_signals_cannot_override_hard_constraints(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": OPPORTUNITY_DETECTION_RECEIPT_SCHEMA,
            "interface": OPPORTUNITY_DETECTION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "policy": self.policy.to_dict(),
            "findings": [item.to_dict() for item in self.findings],
            "below_threshold_module_ids": list(self.below_threshold_module_ids),
            "evidence_cids": list(self.evidence_cids),
            "analyzer_id": self.analyzer_id,
            "finding_cids": list(self.finding_cids),
            "negative_evidence_ids": list(self.negative_evidence_ids),
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "soft_signals_cannot_override_hard_constraints": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "OpportunityDetectionReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != OPPORTUNITY_DETECTION_RECEIPT_SCHEMA:
            raise OpportunityDetectorError(
                "unsupported OpportunityDetectionReceipt schema"
            )
        if payload.pop("interface") != OPPORTUNITY_DETECTION_RECEIPT_INTERFACE:
            raise OpportunityDetectorError(
                "unsupported OpportunityDetectionReceipt interface"
            )
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
        ):
            if payload.pop(flag) is not False:
                raise OpportunityDetectorError(f"receipt cannot claim {flag}")
        if payload.pop("soft_signals_cannot_override_hard_constraints") is not True:
            raise OpportunityDetectorError(
                "receipt must retain soft-cannot-override-hard"
            )
        payload.pop("finding_cids")
        payload.pop("negative_evidence_ids")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "OpportunityDetectionReceipt receipt_cid"
        )
        return result


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
    raise OpportunityDetectorError("finding must be a MonolithFinding")


def detect_monolith_opportunities(
    evidence: Mapping[str, Any] | Any,
    *,
    policy: OpportunityPolicy | Mapping[str, Any] | None = None,
) -> OpportunityDetectionReceipt:
    """Detect oversized or structurally overloaded modules under policy."""

    facts = compile_opportunity_evidence(evidence)
    resolved_policy = _coerce_policy(policy)
    findings: list[MonolithFinding] = []
    below: list[str] = []
    for module in facts.modules:
        finding, module_id = _evaluate_module(
            module, facts=facts, policy=resolved_policy
        )
        if finding is None:
            below.append(module_id)
            continue
        findings.append(finding)
    return OpportunityDetectionReceipt(
        tree_id=facts.tree_id,
        policy=resolved_policy,
        findings=tuple(findings),
        below_threshold_module_ids=tuple(below),
        evidence_cids=facts.evidence_cids,
        analyzer_id=ANALYZER_ID,
    )


class MonolithOpportunityDetector:
    """SPAR-036 detector. Nomination-only; cannot mint authority."""

    interface: ClassVar[str] = MONOLITH_OPPORTUNITY_DETECTOR_INTERFACE

    def __init__(
        self, policy: OpportunityPolicy | Mapping[str, Any] | None = None
    ) -> None:
        self.policy = _coerce_policy(policy)

    def detect(
        self, evidence: Mapping[str, Any] | Any
    ) -> OpportunityDetectionReceipt:
        return detect_monolith_opportunities(evidence, policy=self.policy)


def _acceptance_ids(values: Any) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered = ACCEPTANCE_IDS
    else:
        ordered = tuple(_text(item, "acceptance_ids") for item in values)
    if tuple(sorted(ordered)) != tuple(sorted(ACCEPTANCE_IDS)):
        raise OpportunityDetectorError(
            "durable goal must include every acceptance identifier"
        )
    if len(ordered) != len(set(ordered)):
        raise OpportunityDetectorError("acceptance_ids must not contain duplicates")
    if ordered != ACCEPTANCE_IDS:
        raise OpportunityDetectorError(
            "acceptance identifiers must remain in canonical order"
        )
    return ordered


@dataclass(frozen=True, slots=True)
class DurableGoal:
    """Exact finding compiled into a bounded durable goal. Nomination-only."""

    tree_id: str
    finding_cid: str
    module_id: str
    risks: Sequence[str]
    autonomy_tier: AutonomyTier | str
    acceptance_ids: Sequence[str] = ACCEPTANCE_IDS
    compiler_id: str = COMPILER_ID

    interface: ClassVar[str] = DURABLE_GOAL_INTERFACE
    schema: ClassVar[str] = DURABLE_GOAL_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "finding_cid",
            "module_id",
            "risks",
            "autonomy_tier",
            "acceptance_ids",
            "compiler_id",
            "generic_prompt_forbidden",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "goal_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "finding_cid", _cid(self.finding_cid, "finding_cid"))
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(
            self,
            "risks",
            tuple(sorted({_enum(item, RiskKind, "risks") for item in self.risks})),
        )
        if not self.risks:
            raise OpportunityDetectorError("durable goal must retain finding risks")
        object.__setattr__(
            self,
            "autonomy_tier",
            _enum(self.autonomy_tier, AutonomyTier, "autonomy_tier"),
        )
        object.__setattr__(self, "acceptance_ids", _acceptance_ids(self.acceptance_ids))
        compiler = _text(self.compiler_id, "compiler_id")
        if compiler != COMPILER_ID:
            raise OpportunityDetectorError(
                "compiler_id must remain the SPAR-036 goal compiler"
            )
        object.__setattr__(self, "compiler_id", compiler)

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": DURABLE_GOAL_SCHEMA,
            "interface": DURABLE_GOAL_INTERFACE,
            "tree_id": self.tree_id,
            "finding_cid": self.finding_cid,
            "module_id": self.module_id,
            "risks": list(self.risks),
            "autonomy_tier": self.autonomy_tier,
            "acceptance_ids": list(self.acceptance_ids),
            "compiler_id": self.compiler_id,
            "generic_prompt_forbidden": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def goal_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["goal_cid"] = self.goal_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DurableGoal":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("goal_cid")
        if payload.pop("schema") != DURABLE_GOAL_SCHEMA:
            raise OpportunityDetectorError("unsupported DurableGoal schema")
        if payload.pop("interface") != DURABLE_GOAL_INTERFACE:
            raise OpportunityDetectorError("unsupported DurableGoal interface")
        if payload.pop("generic_prompt_forbidden") is not True:
            raise OpportunityDetectorError("durable goal cannot emit a generic prompt")
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
        ):
            if payload.pop(flag) is not False:
                raise OpportunityDetectorError(f"goal cannot claim {flag}")
        result = cls(**payload)
        _verify_cid(claimed, result.goal_cid, "DurableGoal goal_cid")
        return result


def _goal_from_finding(finding: MonolithFinding) -> DurableGoal:
    return DurableGoal(
        tree_id=finding.tree_id,
        finding_cid=finding.finding_cid,
        module_id=finding.module_id,
        risks=finding.risks,
        autonomy_tier=finding.autonomy_tier,
    )


@dataclass(frozen=True, slots=True)
class GoalCompilationReceipt:
    """One durable goal per admitted finding. Nomination-only."""

    tree_id: str
    goals: Sequence[DurableGoal | Mapping[str, Any]]
    compiler_id: str = COMPILER_ID

    interface: ClassVar[str] = GOAL_COMPILATION_RECEIPT_INTERFACE
    schema: ClassVar[str] = GOAL_COMPILATION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "goals",
            "compiler_id",
            "goal_cids",
            "finding_cids",
            "generic_prompt_forbidden",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        compiler = _text(self.compiler_id, "compiler_id")
        if compiler != COMPILER_ID:
            raise OpportunityDetectorError(
                "compiler_id must remain the SPAR-036 goal compiler"
            )
        goals = tuple(_coerce_goal(item) for item in self.goals)
        if not goals:
            raise OpportunityDetectorError("goal compilation requires findings")
        if len(goals) > MAX_GOALS:
            raise OpportunityDetectorError("goals exceed maximum length")
        tree_id = _tree_id(self.tree_id)
        mismatched = [item.module_id for item in goals if item.tree_id != tree_id]
        if mismatched:
            raise OpportunityDetectorError("goal tree_id does not match receipt")
        finding_cids = [item.finding_cid for item in goals]
        if len(finding_cids) != len(set(finding_cids)):
            raise OpportunityDetectorError("duplicate compiled finding identity")
        goal_cids = [item.goal_cid for item in goals]
        if len(goal_cids) != len(set(goal_cids)):
            raise OpportunityDetectorError("duplicate goal identity")
        goals = tuple(sorted(goals, key=lambda item: item.goal_cid))
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "compiler_id", compiler)

    @property
    def goal_cids(self) -> tuple[str, ...]:
        return tuple(item.goal_cid for item in self.goals)

    @property
    def finding_cids(self) -> tuple[str, ...]:
        return tuple(item.finding_cid for item in self.goals)

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": GOAL_COMPILATION_RECEIPT_SCHEMA,
            "interface": GOAL_COMPILATION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "goals": [item.to_dict() for item in self.goals],
            "compiler_id": self.compiler_id,
            "goal_cids": list(self.goal_cids),
            "finding_cids": list(self.finding_cids),
            "generic_prompt_forbidden": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalCompilationReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != GOAL_COMPILATION_RECEIPT_SCHEMA:
            raise OpportunityDetectorError(
                "unsupported GoalCompilationReceipt schema"
            )
        if payload.pop("interface") != GOAL_COMPILATION_RECEIPT_INTERFACE:
            raise OpportunityDetectorError(
                "unsupported GoalCompilationReceipt interface"
            )
        if payload.pop("generic_prompt_forbidden") is not True:
            raise OpportunityDetectorError("compilation cannot emit a generic prompt")
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
        ):
            if payload.pop(flag) is not False:
                raise OpportunityDetectorError(f"receipt cannot claim {flag}")
        payload.pop("goal_cids")
        payload.pop("finding_cids")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "GoalCompilationReceipt receipt_cid"
        )
        return result


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
    raise OpportunityDetectorError("goal must be a DurableGoal")


def compile_durable_goals(
    findings: OpportunityDetectionReceipt
    | Sequence[MonolithFinding | Mapping[str, Any]]
    | Mapping[str, Any],
) -> GoalCompilationReceipt:
    """Compile admitted findings into bounded durable goals."""

    if isinstance(findings, OpportunityDetectionReceipt):
        resolved = findings.findings
        tree_id = findings.tree_id
    elif isinstance(findings, Mapping) and findings.get("interface") == (
        OPPORTUNITY_DETECTION_RECEIPT_INTERFACE
    ):
        receipt = OpportunityDetectionReceipt.from_dict(findings)
        resolved = receipt.findings
        tree_id = receipt.tree_id
    elif isinstance(findings, (list, tuple)):
        resolved = tuple(_coerce_finding(item) for item in findings)
        if not resolved:
            raise OpportunityDetectorError("goal compilation requires findings")
        tree_ids = {item.tree_id for item in resolved}
        if len(tree_ids) != 1:
            raise OpportunityDetectorError("findings must share one tree_id")
        tree_id = next(iter(tree_ids))
    else:
        raise OpportunityDetectorError("goal compilation requires findings")
    if not resolved:
        raise OpportunityDetectorError("goal compilation requires findings")
    goals = tuple(_goal_from_finding(item) for item in resolved)
    return GoalCompilationReceipt(tree_id=tree_id, goals=goals)


class GoalCompiler:
    """SPAR-036 goal compiler. Nomination-only; cannot mint authority."""

    interface: ClassVar[str] = GOAL_COMPILER_INTERFACE

    def compile(
        self,
        findings: OpportunityDetectionReceipt
        | Sequence[MonolithFinding | Mapping[str, Any]]
        | Mapping[str, Any],
    ) -> GoalCompilationReceipt:
        return compile_durable_goals(findings)


def encode_canonical_receipt(
    receipt: OpportunityDetectionReceipt | GoalCompilationReceipt,
) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_detection_receipt(
    payload: Mapping[str, Any],
) -> OpportunityDetectionReceipt:
    return OpportunityDetectionReceipt.from_dict(payload)


def decode_canonical_goal_receipt(
    payload: Mapping[str, Any],
) -> GoalCompilationReceipt:
    return GoalCompilationReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise OpportunityDetectorError(
            f"opportunity detector must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ACCEPTANCE_IDS",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "AutonomyTier",
    "COMPILER_ID",
    "DUCKLAKE_IS_AUTHORITY",
    "DurableGoal",
    "EXISTING_ADAPTER_AUTHORITIES",
    "FindingKind",
    "GENERIC_PROMPT_FORBIDDEN",
    "GOAL_COMPILATION_RECEIPT_INTERFACE",
    "GOAL_COMPILER_INTERFACE",
    "GOAL_ID",
    "GoalCompilationReceipt",
    "GoalCompiler",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "MONOLITH_FINDING_INTERFACE",
    "MONOLITH_OPPORTUNITY_DETECTOR_INTERFACE",
    "MonolithFinding",
    "MonolithOpportunityDetector",
    "OPPORTUNITY_CAN_AUTHORIZE_COMPLETION",
    "OPPORTUNITY_CAN_AUTHORIZE_TRANSITION",
    "OPPORTUNITY_CAN_CREATE_AUTHORITY",
    "OPPORTUNITY_CONTRACT_VERSION",
    "OPPORTUNITY_DETECTION_RECEIPT_INTERFACE",
    "OPPORTUNITY_POLICY_INTERFACE",
    "OpportunityDetectionReceipt",
    "OpportunityDetectorError",
    "OpportunityPolicy",
    "PLAN_IS_NOMINATION_ONLY",
    "POLICY_ID",
    "POLICY_REVISION",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "RiskKind",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "THRESHOLD_FAMILIES",
    "ThresholdFamily",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "assert_not_competing_capsule_family",
    "compile_durable_goals",
    "compile_opportunity_evidence",
    "decode_canonical_detection_receipt",
    "decode_canonical_goal_receipt",
    "default_opportunity_policy",
    "detect_monolith_opportunities",
    "encode_canonical_receipt",
    "opportunity_cid_profile",
    "provider_free_exports",
]
