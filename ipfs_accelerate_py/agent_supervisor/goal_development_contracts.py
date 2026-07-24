"""Canonical, bounded contracts for untrusted goal-development proposals.

Goal development is deliberately separate from formal-plan admission and
proof.  A model may suggest a decomposition, but it cannot mutate the
canonical root objective, add premises, admit work, attest implementation
conformance, or mark a goal complete through any contract in this module.

The request freezes every semantic input by canonical reference.  A draft
contains the exact request and policy used to produce it, so changing the
root, formula, assumptions, evidence requirements, vocabulary, repository
tree, scope, or policy necessarily changes both the request identity and the
draft identity.  Draft limits are enforced from the canonical proposal graph,
not from producer-supplied summaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)
from .proof_context import estimate_context_tokens


GOAL_DEVELOPMENT_CONTRACT_VERSION = 1
CONTRACT_VERSION = GOAL_DEVELOPMENT_CONTRACT_VERSION
SCHEMA_VERSION = GOAL_DEVELOPMENT_CONTRACT_VERSION

GOAL_DEVELOPMENT_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/goal-development-policy@1"
)
GOAL_DEVELOPMENT_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/goal-development-request@1"
)
GOAL_DECOMPOSITION_PROPOSAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/goal-decomposition-proposal@1"
)
GOAL_DECOMPOSITION_DRAFT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/goal-decomposition-draft@1"
)
GOAL_DEVELOPMENT_PROPOSAL_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/goal-development-proposal-receipt@1"
)
GOAL_DEVELOPMENT_ADMISSION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/goal-development-admission-receipt@1"
)

DEFAULT_MAX_DECOMPOSITION_DEPTH = 3
DEFAULT_MAX_DECOMPOSITION_BREADTH = 4
DEFAULT_MAX_DECOMPOSITION_PROPOSALS = 12
DEFAULT_MAX_DECOMPOSITION_BYTES = 256 * 1024
DEFAULT_MAX_DECOMPOSITION_TOKENS = 8192
ABSOLUTE_MAX_GOAL_DEVELOPMENT_TEXT_BYTES = 64 * 1024

GoalDevelopmentValidationError = ContractValidationError


class GoalDevelopmentMode(str, Enum):
    """Policy-controlled effect level for goal development."""

    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTO_SAFE = "auto_safe"
    REPAIR_ONLY = "repair_only"


class GoalDevelopmentTrust(str, Enum):
    """Trust classification attached to a proposal artifact."""

    UNVERIFIED = "unverified"


class GoalDevelopmentAuthority(str, Enum):
    """Closed authority vocabulary for this boundary."""

    NONE = "none"
    DETERMINISTIC_VALIDATOR = "deterministic_validator"
    SUPERVISOR_ADMISSION = "supervisor_admission"


class GoalProposalDecision(str, Enum):
    """Result of deterministic proposal-envelope validation."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class GoalAdmissionDecision(str, Enum):
    """Whether a reviewed proposal may affect the canonical objective graph."""

    NOT_ADMITTED = "not_admitted"
    REVIEW_REQUIRED = "review_required"
    ADMITTED = "admitted"


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    max_bytes: int = ABSOLUTE_MAX_GOAL_DEVELOPMENT_TEXT_BYTES,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise ContractValidationError(f"{field_name} is required")
    if "\x00" in result:
        raise ContractValidationError(f"{field_name} must not contain NUL bytes")
    if len(result.encode("utf-8")) > max_bytes:
        raise ContractValidationError(
            f"{field_name} exceeds the maximum of {max_bytes} UTF-8 bytes"
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
        source = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ContractValidationError(f"{field_name} must be a sequence of strings")
    result: list[str] = []
    for index, value in enumerate(source):
        item = _text(value, field_name=f"{field_name}[{index}]", required=True)
        if item not in result:
            result.append(item)
    if required and not result:
        raise ContractValidationError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _schema_and_version(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            "goal-development contract payload must be an object"
        )
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        raise ContractValidationError(
            f"unsupported schema {schema!r}; expected {expected_schema}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, GOAL_DEVELOPMENT_CONTRACT_VERSION):
        raise ContractValidationError(
            "unsupported goal-development contract version"
        )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], *, artifact_name: str
) -> None:
    if set(payload).difference(allowed):
        raise ContractValidationError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload"
        )


def _claimed_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ContractValidationError(
                f"{artifact_name} content identity does not match payload"
            )


def _validate_false_claims(
    payload: Mapping[str, Any],
    *,
    artifact_name: str,
    allow_admitted: bool = False,
) -> None:
    """Reject producer attempts to smuggle authority through derived fields."""

    expected: dict[str, Any] = {
        "proof_claimed": False,
        "implementation_conformance_claimed": False,
        "completion_claimed": False,
        "implementation_conformant": False,
        "complete": False,
    }
    if not allow_admitted:
        expected.update({"admission_claimed": False, "admitted": False})
    for field_name, expected_value in expected.items():
        if field_name in payload and payload[field_name] is not expected_value:
            raise ContractValidationError(
                f"{artifact_name} cannot claim {field_name.replace('_', ' ')}"
            )


class GoalDevelopmentContract(CanonicalContract):
    """Canonical contract pinned to the goal-development schema version."""

    @property
    def schema_version(self) -> int:
        return GOAL_DEVELOPMENT_CONTRACT_VERSION

    def _versioned(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "contract_version": GOAL_DEVELOPMENT_CONTRACT_VERSION,
            **dict(payload),
        }


@dataclass(frozen=True)
class GoalDevelopmentPolicy(GoalDevelopmentContract):
    """Hard mode, graph, byte, and token limits for one request."""

    SCHEMA: ClassVar[str] = GOAL_DEVELOPMENT_POLICY_SCHEMA

    mode: GoalDevelopmentMode = GoalDevelopmentMode.OFF
    max_depth: int = DEFAULT_MAX_DECOMPOSITION_DEPTH
    max_breadth: int = DEFAULT_MAX_DECOMPOSITION_BREADTH
    max_proposals: int = DEFAULT_MAX_DECOMPOSITION_PROPOSALS
    max_bytes: int = DEFAULT_MAX_DECOMPOSITION_BYTES
    max_tokens: int = DEFAULT_MAX_DECOMPOSITION_TOKENS
    allow_new_assumptions: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, GoalDevelopmentMode, field_name="mode")
        )
        for name in (
            "max_depth",
            "max_breadth",
            "max_proposals",
            "max_bytes",
            "max_tokens",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), field_name=name)
            )
        if not isinstance(self.allow_new_assumptions, bool):
            raise ContractValidationError("allow_new_assumptions must be a boolean")
        if self.allow_new_assumptions:
            raise ContractValidationError(
                "goal development cannot authorize hidden or new assumptions"
            )

    @property
    def policy_digest(self) -> str:
        return self.content_id

    @property
    def digest(self) -> str:
        return self.policy_digest

    @property
    def max_breadth_per_parent(self) -> int:
        return self.max_breadth

    @property
    def max_count(self) -> int:
        return self.max_proposals

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "mode": self.mode,
                "max_depth": self.max_depth,
                "max_breadth": self.max_breadth,
                "max_proposals": self.max_proposals,
                "max_bytes": self.max_bytes,
                "max_tokens": self.max_tokens,
                "allow_new_assumptions": self.allow_new_assumptions,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDevelopmentPolicy":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "mode",
                "max_depth",
                "max_breadth",
                "max_breadth_per_parent",
                "max_proposals",
                "max_count",
                "max_bytes",
                "max_tokens",
                "allow_new_assumptions",
                "policy_digest",
                "content_id",
            },
            artifact_name="goal-development policy",
        )
        result = cls(
            mode=payload.get("mode", GoalDevelopmentMode.OFF),
            max_depth=payload.get("max_depth", DEFAULT_MAX_DECOMPOSITION_DEPTH),
            max_breadth=payload.get(
                "max_breadth",
                payload.get(
                    "max_breadth_per_parent", DEFAULT_MAX_DECOMPOSITION_BREADTH
                ),
            ),
            max_proposals=payload.get(
                "max_proposals",
                payload.get("max_count", DEFAULT_MAX_DECOMPOSITION_PROPOSALS),
            ),
            max_bytes=payload.get(
                "max_bytes", DEFAULT_MAX_DECOMPOSITION_BYTES
            ),
            max_tokens=payload.get(
                "max_tokens", DEFAULT_MAX_DECOMPOSITION_TOKENS
            ),
            allow_new_assumptions=payload.get("allow_new_assumptions", False),
        )
        _claimed_identity(
            payload,
            result.policy_digest,
            names=("policy_digest", "content_id"),
            artifact_name="goal-development policy",
        )
        return result


@dataclass(frozen=True)
class GoalDevelopmentRequest(GoalDevelopmentContract):
    """Immutable semantic envelope supplied to a decomposition producer."""

    SCHEMA: ClassVar[str] = GOAL_DEVELOPMENT_REQUEST_SCHEMA

    root_goal_id: str
    root_goal_content_id: str
    satisfaction_formula_id: str
    assumption_ids: tuple[str, ...]
    evidence_requirement_ids: tuple[str, ...]
    vocabulary_profile_id: str
    vocabulary_version: int
    repository_tree_id: str
    scope_ids: tuple[str, ...]
    policy_digest: str
    mode: GoalDevelopmentMode = GoalDevelopmentMode.OFF
    repair_draft_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "root_goal_id",
            "root_goal_content_id",
            "satisfaction_formula_id",
            "vocabulary_profile_id",
            "repository_tree_id",
            "policy_digest",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self,
            "assumption_ids",
            _strings(self.assumption_ids, field_name="assumption_ids"),
        )
        object.__setattr__(
            self,
            "evidence_requirement_ids",
            _strings(
                self.evidence_requirement_ids,
                field_name="evidence_requirement_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "scope_ids",
            _strings(self.scope_ids, field_name="scope_ids", required=True),
        )
        object.__setattr__(
            self, "mode", _enum(self.mode, GoalDevelopmentMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "repair_draft_id",
            _text(self.repair_draft_id, field_name="repair_draft_id"),
        )
        object.__setattr__(
            self,
            "vocabulary_version",
            _positive_int(self.vocabulary_version, field_name="vocabulary_version"),
        )
        if (
            self.mode is GoalDevelopmentMode.REPAIR_ONLY
            and not self.repair_draft_id
        ):
            raise ContractValidationError(
                "repair_only requests must bind the draft being repaired"
            )
        if (
            self.mode is not GoalDevelopmentMode.REPAIR_ONLY
            and self.repair_draft_id
        ):
            raise ContractValidationError(
                "repair_draft_id is only valid in repair_only mode"
            )

    @property
    def request_id(self) -> str:
        return self.content_id

    @property
    def root_goal_digest(self) -> str:
        return self.root_goal_content_id

    @property
    def repository_tree(self) -> str:
        return self.repository_tree_id

    def require_policy(self, policy: GoalDevelopmentPolicy) -> None:
        if not isinstance(policy, GoalDevelopmentPolicy):
            raise ContractValidationError("policy must be GoalDevelopmentPolicy")
        if self.policy_digest != policy.policy_digest:
            raise ContractValidationError("request policy digest does not match policy")
        if self.mode is not policy.mode:
            raise ContractValidationError("request mode does not match policy")

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "root_goal_id": self.root_goal_id,
                "root_goal_content_id": self.root_goal_content_id,
                "satisfaction_formula_id": self.satisfaction_formula_id,
                "assumption_ids": self.assumption_ids,
                "evidence_requirement_ids": self.evidence_requirement_ids,
                "vocabulary_profile_id": self.vocabulary_profile_id,
                "vocabulary_version": self.vocabulary_version,
                "repository_tree_id": self.repository_tree_id,
                "scope_ids": self.scope_ids,
                "policy_digest": self.policy_digest,
                "mode": self.mode,
                "repair_draft_id": self.repair_draft_id,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDevelopmentRequest":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "root_goal_id",
                "root_goal_content_id",
                "root_goal_digest",
                "satisfaction_formula_id",
                "assumption_ids",
                "evidence_requirement_ids",
                "vocabulary_profile_id",
                "vocabulary_version",
                "repository_tree_id",
                "scope_ids",
                "policy_digest",
                "mode",
                "repair_draft_id",
                "request_id",
                "content_id",
            },
            artifact_name="goal-development request",
        )
        result = cls(
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get(
                "root_goal_content_id", payload.get("root_goal_digest", "")
            ),
            satisfaction_formula_id=payload.get("satisfaction_formula_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            vocabulary_profile_id=payload.get("vocabulary_profile_id", ""),
            vocabulary_version=payload.get("vocabulary_version", 0),
            repository_tree_id=payload.get("repository_tree_id", ""),
            scope_ids=tuple(payload.get("scope_ids") or ()),
            policy_digest=payload.get("policy_digest", ""),
            mode=payload.get("mode", GoalDevelopmentMode.OFF),
            repair_draft_id=payload.get("repair_draft_id", ""),
        )
        _claimed_identity(
            payload,
            result.request_id,
            names=("request_id", "content_id"),
            artifact_name="goal-development request",
        )
        return result


@dataclass(frozen=True)
class GoalDecompositionProposal(GoalDevelopmentContract):
    """One unverified child-goal proposal in a rooted decomposition DAG."""

    SCHEMA: ClassVar[str] = GOAL_DECOMPOSITION_PROPOSAL_SCHEMA

    proposal_id: str
    parent_id: str
    satisfaction_formula_id: str
    evidence_requirement_ids: tuple[str, ...]
    scope_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    title: str = ""

    def __post_init__(self) -> None:
        for name in ("proposal_id", "parent_id", "satisfaction_formula_id"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self,
            "evidence_requirement_ids",
            _strings(
                self.evidence_requirement_ids,
                field_name="evidence_requirement_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "scope_ids",
            _strings(self.scope_ids, field_name="scope_ids", required=True),
        )
        object.__setattr__(
            self,
            "assumption_ids",
            _strings(self.assumption_ids, field_name="assumption_ids"),
        )
        object.__setattr__(
            self,
            "depends_on",
            _strings(self.depends_on, field_name="depends_on"),
        )
        object.__setattr__(
            self,
            "title",
            _text(self.title, field_name="title", max_bytes=16 * 1024),
        )
        if self.proposal_id == self.parent_id:
            raise ContractValidationError("a proposal cannot be its own parent")
        if self.proposal_id in self.depends_on:
            raise ContractValidationError("a proposal cannot depend on itself")

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "proposal_id": self.proposal_id,
                "parent_id": self.parent_id,
                "satisfaction_formula_id": self.satisfaction_formula_id,
                "assumption_ids": self.assumption_ids,
                "evidence_requirement_ids": self.evidence_requirement_ids,
                "scope_ids": self.scope_ids,
                "depends_on": self.depends_on,
                "title": self.title,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDecompositionProposal":
        _schema_and_version(payload, cls.SCHEMA)
        _validate_false_claims(payload, artifact_name="goal decomposition proposal")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "proposal_id",
                "parent_id",
                "satisfaction_formula_id",
                "assumption_ids",
                "evidence_requirement_ids",
                "scope_ids",
                "depends_on",
                "title",
                "content_id",
                "proof_claimed",
                "admission_claimed",
                "implementation_conformance_claimed",
                "completion_claimed",
                "implementation_conformant",
                "complete",
                "admitted",
            },
            artifact_name="goal decomposition proposal",
        )
        result = cls(
            proposal_id=payload.get("proposal_id", ""),
            parent_id=payload.get("parent_id", ""),
            satisfaction_formula_id=payload.get("satisfaction_formula_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            scope_ids=tuple(payload.get("scope_ids") or ()),
            depends_on=tuple(payload.get("depends_on") or ()),
            title=payload.get("title", ""),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id",),
            artifact_name="goal decomposition proposal",
        )
        return result


def _proposals(values: Any) -> tuple[GoalDecompositionProposal, ...]:
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise ContractValidationError("proposals must be a sequence")
    result: list[GoalDecompositionProposal] = []
    for value in values:
        proposal = (
            value
            if isinstance(value, GoalDecompositionProposal)
            else GoalDecompositionProposal.from_dict(value)
        )
        result.append(proposal)
    by_id: dict[str, GoalDecompositionProposal] = {}
    for proposal in result:
        if proposal.proposal_id in by_id:
            raise ContractValidationError("proposal IDs must be unique")
        by_id[proposal.proposal_id] = proposal
    return tuple(by_id[key] for key in sorted(by_id))


@dataclass(frozen=True)
class GoalDecompositionDraft(GoalDevelopmentContract):
    """Content-addressed, explicitly unverified decomposition proposal."""

    SCHEMA: ClassVar[str] = GOAL_DECOMPOSITION_DRAFT_SCHEMA

    request: GoalDevelopmentRequest
    policy: GoalDevelopmentPolicy
    proposals: tuple[GoalDecompositionProposal, ...]
    producer_id: str
    token_count: int = 0

    def __post_init__(self) -> None:
        request = (
            self.request
            if isinstance(self.request, GoalDevelopmentRequest)
            else GoalDevelopmentRequest.from_dict(self.request)
        )
        policy = (
            self.policy
            if isinstance(self.policy, GoalDevelopmentPolicy)
            else GoalDevelopmentPolicy.from_dict(self.policy)
        )
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "policy", policy)
        request.require_policy(policy)
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id, field_name="producer_id", required=True),
        )
        normalized = _proposals(self.proposals)
        object.__setattr__(self, "proposals", normalized)
        supplied_tokens = _nonnegative_int(
            self.token_count, field_name="token_count"
        )
        measured_tokens = estimate_context_tokens(
            canonical_json_bytes([item.to_dict() for item in normalized]).decode(
                "utf-8"
            )
        )
        object.__setattr__(
            self, "token_count", max(supplied_tokens, measured_tokens)
        )
        self._validate_graph_and_bounds()

    def _validate_graph_and_bounds(self) -> None:
        if self.request.mode is GoalDevelopmentMode.OFF and self.proposals:
            raise ContractValidationError("off mode cannot produce proposals")
        if len(self.proposals) > self.policy.max_proposals:
            raise ContractValidationError("draft exceeds max_proposals")

        by_id = {item.proposal_id: item for item in self.proposals}
        known_ids = set(by_id)
        root_id = self.request.root_goal_id
        allowed_parents = known_ids | {root_id}
        for proposal in self.proposals:
            if proposal.proposal_id == root_id:
                raise ContractValidationError(
                    "draft cannot mutate or replace the frozen root goal"
                )
            if proposal.parent_id not in allowed_parents:
                raise ContractValidationError(
                    f"proposal {proposal.proposal_id} has an unknown parent"
                )
            unknown_dependencies = set(proposal.depends_on).difference(known_ids)
            if unknown_dependencies:
                raise ContractValidationError(
                    f"proposal {proposal.proposal_id} has unknown dependencies"
                )
            if not set(proposal.assumption_ids).issubset(
                self.request.assumption_ids
            ):
                raise ContractValidationError(
                    "proposal introduces a hidden or new assumption"
                )
            if not set(proposal.evidence_requirement_ids).issubset(
                self.request.evidence_requirement_ids
            ):
                raise ContractValidationError(
                    "proposal changes the frozen evidence requirements"
                )
            if not set(proposal.scope_ids).issubset(self.request.scope_ids):
                raise ContractValidationError(
                    "proposal escapes the frozen development scope"
                )

        state: dict[str, int] = {}
        depths: dict[str, int] = {}

        def depth(proposal_id: str) -> int:
            marker = state.get(proposal_id, 0)
            if marker == 1:
                raise ContractValidationError("proposal parent graph must be acyclic")
            if marker == 2:
                return depths[proposal_id]
            state[proposal_id] = 1
            proposal = by_id[proposal_id]
            value = (
                1
                if proposal.parent_id == root_id
                else depth(proposal.parent_id) + 1
            )
            state[proposal_id] = 2
            depths[proposal_id] = value
            return value

        for proposal_id in sorted(by_id):
            if depth(proposal_id) > self.policy.max_depth:
                raise ContractValidationError("draft exceeds max_depth")

        dependency_state: dict[str, int] = {}

        def visit_dependency(proposal_id: str) -> None:
            marker = dependency_state.get(proposal_id, 0)
            if marker == 1:
                raise ContractValidationError(
                    "proposal dependency graph must be acyclic"
                )
            if marker == 2:
                return
            dependency_state[proposal_id] = 1
            for dependency in by_id[proposal_id].depends_on:
                visit_dependency(dependency)
            dependency_state[proposal_id] = 2

        for proposal_id in sorted(by_id):
            visit_dependency(proposal_id)

        breadth = Counter(item.parent_id for item in self.proposals)
        if breadth and max(breadth.values()) > self.policy.max_breadth:
            raise ContractValidationError("draft exceeds max_breadth")

        if self.proposal_bytes > self.policy.max_bytes:
            raise ContractValidationError("draft exceeds max_bytes")
        if self.token_count > self.policy.max_tokens:
            raise ContractValidationError("draft exceeds max_tokens")

    @property
    def draft_id(self) -> str:
        return self.content_id

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def root_goal_id(self) -> str:
        return self.request.root_goal_id

    @property
    def satisfaction_formula_id(self) -> str:
        return self.request.satisfaction_formula_id

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        return self.request.assumption_ids

    @property
    def evidence_requirement_ids(self) -> tuple[str, ...]:
        return self.request.evidence_requirement_ids

    @property
    def vocabulary_profile_id(self) -> str:
        return self.request.vocabulary_profile_id

    @property
    def repository_tree_id(self) -> str:
        return self.request.repository_tree_id

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return self.request.scope_ids

    @property
    def policy_digest(self) -> str:
        return self.policy.policy_digest

    @property
    def proposal_bytes(self) -> int:
        return len(canonical_json_bytes([item.to_dict() for item in self.proposals]))

    @property
    def byte_count(self) -> int:
        return self.proposal_bytes

    @property
    def trust(self) -> GoalDevelopmentTrust:
        return GoalDevelopmentTrust.UNVERIFIED

    @property
    def assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    @property
    def authority(self) -> GoalDevelopmentAuthority:
        return GoalDevelopmentAuthority.NONE

    @property
    def verified(self) -> bool:
        return False

    @property
    def admitted(self) -> bool:
        return False

    @property
    def implementation_conformant(self) -> bool:
        return False

    @property
    def complete(self) -> bool:
        return False

    def validate_request(self, request: GoalDevelopmentRequest) -> None:
        if not isinstance(request, GoalDevelopmentRequest):
            raise ContractValidationError("request must be GoalDevelopmentRequest")
        if request.request_id != self.request_id:
            raise ContractValidationError(
                "draft does not match the frozen goal-development request"
            )

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "request": self.request,
                "request_id": self.request_id,
                "policy": self.policy,
                "policy_digest": self.policy_digest,
                "proposals": self.proposals,
                "producer_id": self.producer_id,
                "proposal_bytes": self.proposal_bytes,
                "token_count": self.token_count,
                "trust": self.trust,
                "assurance": self.assurance,
                "authority": self.authority,
                "verified": False,
                "proof_claimed": False,
                "admission_claimed": False,
                "admitted": False,
                "implementation_conformance_claimed": False,
                "implementation_conformant": False,
                "completion_claimed": False,
                "complete": False,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDecompositionDraft":
        _schema_and_version(payload, cls.SCHEMA)
        _validate_false_claims(payload, artifact_name="goal decomposition draft")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "request",
                "request_id",
                "policy",
                "policy_digest",
                "proposals",
                "producer_id",
                "proposal_bytes",
                "byte_count",
                "token_count",
                "trust",
                "assurance",
                "authority",
                "verified",
                "proof_claimed",
                "admission_claimed",
                "admitted",
                "implementation_conformance_claimed",
                "implementation_conformant",
                "completion_claimed",
                "complete",
                "draft_id",
                "content_id",
            },
            artifact_name="goal decomposition draft",
        )
        if payload.get("trust", GoalDevelopmentTrust.UNVERIFIED.value) not in (
            GoalDevelopmentTrust.UNVERIFIED,
            GoalDevelopmentTrust.UNVERIFIED.value,
        ):
            raise ContractValidationError("goal decomposition draft must be unverified")
        if payload.get("assurance", AssuranceLevel.UNVERIFIED.value) not in (
            AssuranceLevel.UNVERIFIED,
            AssuranceLevel.UNVERIFIED.value,
        ):
            raise ContractValidationError("goal decomposition draft cannot claim assurance")
        if payload.get("authority", GoalDevelopmentAuthority.NONE.value) not in (
            GoalDevelopmentAuthority.NONE,
            GoalDevelopmentAuthority.NONE.value,
        ):
            raise ContractValidationError("goal decomposition draft has invalid authority")
        request = GoalDevelopmentRequest.from_dict(payload.get("request") or {})
        policy = GoalDevelopmentPolicy.from_dict(payload.get("policy") or {})
        if payload.get("request_id") not in (None, "", request.request_id):
            raise ContractValidationError("draft request identity does not match payload")
        if payload.get("policy_digest") not in (None, "", policy.policy_digest):
            raise ContractValidationError("draft policy digest does not match payload")
        result = cls(
            request=request,
            policy=policy,
            proposals=tuple(payload.get("proposals") or ()),
            producer_id=payload.get("producer_id", ""),
            token_count=payload.get("token_count", 0),
        )
        if payload.get("proposal_bytes") not in (None, result.proposal_bytes):
            raise ContractValidationError("draft proposal byte count does not match payload")
        if payload.get("byte_count") not in (None, result.proposal_bytes):
            raise ContractValidationError("draft byte count does not match payload")
        _claimed_identity(
            payload,
            result.draft_id,
            names=("draft_id", "content_id"),
            artifact_name="goal decomposition draft",
        )
        return result


@dataclass(frozen=True)
class GoalDevelopmentProposalReceipt(GoalDevelopmentContract):
    """Deterministic envelope-validation receipt with no proof authority."""

    SCHEMA: ClassVar[str] = GOAL_DEVELOPMENT_PROPOSAL_RECEIPT_SCHEMA

    request_id: str
    draft_id: str
    root_goal_id: str
    root_goal_content_id: str
    satisfaction_formula_id: str
    assumption_ids: tuple[str, ...]
    evidence_requirement_ids: tuple[str, ...]
    vocabulary_profile_id: str
    vocabulary_version: int
    repository_tree_id: str
    scope_ids: tuple[str, ...]
    policy_digest: str
    mode: GoalDevelopmentMode
    validator_id: str
    decision: GoalProposalDecision
    proposal_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "draft_id",
            "root_goal_id",
            "root_goal_content_id",
            "satisfaction_formula_id",
            "vocabulary_profile_id",
            "repository_tree_id",
            "policy_digest",
            "validator_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        for name in (
            "assumption_ids",
            "evidence_requirement_ids",
            "scope_ids",
            "proposal_ids",
            "reason_codes",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    required=name in {"evidence_requirement_ids", "scope_ids"},
                ),
            )
        object.__setattr__(
            self,
            "vocabulary_version",
            _positive_int(self.vocabulary_version, field_name="vocabulary_version"),
        )
        object.__setattr__(
            self, "mode", _enum(self.mode, GoalDevelopmentMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "decision",
            _enum(self.decision, GoalProposalDecision, field_name="decision"),
        )
        if self.decision is GoalProposalDecision.ACCEPTED and not self.proposal_ids:
            raise ContractValidationError(
                "accepted proposal receipts must identify proposals"
            )
        if self.decision is GoalProposalDecision.REJECTED and not self.reason_codes:
            raise ContractValidationError(
                "rejected proposal receipts must contain reason codes"
            )

    @classmethod
    def for_draft(
        cls,
        draft: GoalDecompositionDraft,
        *,
        validator_id: str,
        decision: GoalProposalDecision = GoalProposalDecision.ACCEPTED,
        reason_codes: Sequence[str] = (),
    ) -> "GoalDevelopmentProposalReceipt":
        if not isinstance(draft, GoalDecompositionDraft):
            raise ContractValidationError("draft must be GoalDecompositionDraft")
        request = draft.request
        return cls(
            request_id=request.request_id,
            draft_id=draft.draft_id,
            root_goal_id=request.root_goal_id,
            root_goal_content_id=request.root_goal_content_id,
            satisfaction_formula_id=request.satisfaction_formula_id,
            assumption_ids=request.assumption_ids,
            evidence_requirement_ids=request.evidence_requirement_ids,
            vocabulary_profile_id=request.vocabulary_profile_id,
            vocabulary_version=request.vocabulary_version,
            repository_tree_id=request.repository_tree_id,
            scope_ids=request.scope_ids,
            policy_digest=request.policy_digest,
            mode=request.mode,
            validator_id=validator_id,
            decision=decision,
            proposal_ids=tuple(item.proposal_id for item in draft.proposals),
            reason_codes=tuple(reason_codes),
        )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def authority(self) -> GoalDevelopmentAuthority:
        return GoalDevelopmentAuthority.DETERMINISTIC_VALIDATOR

    @property
    def assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    def validate_draft(self, draft: GoalDecompositionDraft) -> None:
        request = draft.request
        expected = (
            request.request_id,
            draft.draft_id,
            request.root_goal_id,
            request.root_goal_content_id,
            request.satisfaction_formula_id,
            request.assumption_ids,
            request.evidence_requirement_ids,
            request.vocabulary_profile_id,
            request.vocabulary_version,
            request.repository_tree_id,
            request.scope_ids,
            request.policy_digest,
            request.mode,
            tuple(item.proposal_id for item in draft.proposals),
        )
        actual = (
            self.request_id,
            self.draft_id,
            self.root_goal_id,
            self.root_goal_content_id,
            self.satisfaction_formula_id,
            self.assumption_ids,
            self.evidence_requirement_ids,
            self.vocabulary_profile_id,
            self.vocabulary_version,
            self.repository_tree_id,
            self.scope_ids,
            self.policy_digest,
            self.mode,
            self.proposal_ids,
        )
        if actual != expected:
            raise ContractValidationError(
                "proposal receipt does not match the frozen draft bindings"
            )

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "request_id": self.request_id,
                "draft_id": self.draft_id,
                "root_goal_id": self.root_goal_id,
                "root_goal_content_id": self.root_goal_content_id,
                "satisfaction_formula_id": self.satisfaction_formula_id,
                "assumption_ids": self.assumption_ids,
                "evidence_requirement_ids": self.evidence_requirement_ids,
                "vocabulary_profile_id": self.vocabulary_profile_id,
                "vocabulary_version": self.vocabulary_version,
                "repository_tree_id": self.repository_tree_id,
                "scope_ids": self.scope_ids,
                "policy_digest": self.policy_digest,
                "mode": self.mode,
                "validator_id": self.validator_id,
                "decision": self.decision,
                "proposal_ids": self.proposal_ids,
                "reason_codes": self.reason_codes,
                "authority": self.authority,
                "assurance": self.assurance,
                "proof_claimed": False,
                "admission_claimed": False,
                "admitted": False,
                "implementation_conformance_claimed": False,
                "implementation_conformant": False,
                "completion_claimed": False,
                "complete": False,
            }
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "GoalDevelopmentProposalReceipt":
        _schema_and_version(payload, cls.SCHEMA)
        _validate_false_claims(payload, artifact_name="goal proposal receipt")
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "request_id",
            "draft_id",
            "root_goal_id",
            "root_goal_content_id",
            "satisfaction_formula_id",
            "assumption_ids",
            "evidence_requirement_ids",
            "vocabulary_profile_id",
            "vocabulary_version",
            "repository_tree_id",
            "scope_ids",
            "policy_digest",
            "mode",
            "validator_id",
            "decision",
            "proposal_ids",
            "reason_codes",
            "authority",
            "assurance",
            "proof_claimed",
            "admission_claimed",
            "admitted",
            "implementation_conformance_claimed",
            "implementation_conformant",
            "completion_claimed",
            "complete",
            "receipt_id",
            "content_id",
        }
        _reject_unknown(payload, allowed, artifact_name="goal proposal receipt")
        if payload.get(
            "authority", GoalDevelopmentAuthority.DETERMINISTIC_VALIDATOR.value
        ) not in (
            GoalDevelopmentAuthority.DETERMINISTIC_VALIDATOR,
            GoalDevelopmentAuthority.DETERMINISTIC_VALIDATOR.value,
        ):
            raise ContractValidationError("goal proposal receipt has invalid authority")
        if payload.get("assurance", AssuranceLevel.UNVERIFIED.value) not in (
            AssuranceLevel.UNVERIFIED,
            AssuranceLevel.UNVERIFIED.value,
        ):
            raise ContractValidationError(
                "goal proposal receipt cannot claim proof assurance"
            )
        result = cls(
            request_id=payload.get("request_id", ""),
            draft_id=payload.get("draft_id", ""),
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            satisfaction_formula_id=payload.get("satisfaction_formula_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            vocabulary_profile_id=payload.get("vocabulary_profile_id", ""),
            vocabulary_version=payload.get("vocabulary_version", 0),
            repository_tree_id=payload.get("repository_tree_id", ""),
            scope_ids=tuple(payload.get("scope_ids") or ()),
            policy_digest=payload.get("policy_digest", ""),
            mode=payload.get("mode", GoalDevelopmentMode.OFF),
            validator_id=payload.get("validator_id", ""),
            decision=payload.get("decision", GoalProposalDecision.REJECTED),
            proposal_ids=tuple(payload.get("proposal_ids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        _claimed_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id"),
            artifact_name="goal proposal receipt",
        )
        return result


@dataclass(frozen=True)
class GoalDevelopmentAdmissionReceipt(GoalDevelopmentContract):
    """Supervisor admission decision; never proof or completion evidence."""

    SCHEMA: ClassVar[str] = GOAL_DEVELOPMENT_ADMISSION_RECEIPT_SCHEMA

    request_id: str
    draft_id: str
    proposal_receipt_id: str
    root_goal_id: str
    root_goal_content_id: str
    satisfaction_formula_id: str
    assumption_ids: tuple[str, ...]
    evidence_requirement_ids: tuple[str, ...]
    vocabulary_profile_id: str
    vocabulary_version: int
    repository_tree_id: str
    scope_ids: tuple[str, ...]
    policy_digest: str
    mode: GoalDevelopmentMode
    admitter_id: str
    decision: GoalAdmissionDecision
    proposal_ids: tuple[str, ...] = ()
    authoritative_receipt_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "draft_id",
            "proposal_receipt_id",
            "root_goal_id",
            "root_goal_content_id",
            "satisfaction_formula_id",
            "vocabulary_profile_id",
            "repository_tree_id",
            "policy_digest",
            "admitter_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        for name in (
            "assumption_ids",
            "evidence_requirement_ids",
            "scope_ids",
            "proposal_ids",
            "authoritative_receipt_ids",
            "reason_codes",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    required=name in {"evidence_requirement_ids", "scope_ids"},
                ),
            )
        object.__setattr__(
            self,
            "vocabulary_version",
            _positive_int(self.vocabulary_version, field_name="vocabulary_version"),
        )
        object.__setattr__(
            self, "mode", _enum(self.mode, GoalDevelopmentMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "decision",
            _enum(self.decision, GoalAdmissionDecision, field_name="decision"),
        )
        if self.decision is GoalAdmissionDecision.ADMITTED:
            if self.mode is not GoalDevelopmentMode.AUTO_SAFE:
                raise ContractValidationError(
                    "only auto_safe mode can produce an admission receipt"
                )
            if not self.proposal_ids:
                raise ContractValidationError(
                    "admission must identify admitted proposals"
                )
            if not self.authoritative_receipt_ids:
                raise ContractValidationError(
                    "auto_safe admission requires authoritative receipts"
                )
            if self.reason_codes:
                raise ContractValidationError(
                    "an admitted decision cannot contain rejection reasons"
                )
        else:
            if self.authoritative_receipt_ids:
                raise ContractValidationError(
                    "non-admission cannot claim authoritative admission receipts"
                )
            if not self.reason_codes:
                raise ContractValidationError(
                    "non-admission decisions must contain reason codes"
                )
        if (
            self.mode
            in {
                GoalDevelopmentMode.OFF,
                GoalDevelopmentMode.SHADOW,
                GoalDevelopmentMode.REPAIR_ONLY,
            }
            and self.decision is GoalAdmissionDecision.REVIEW_REQUIRED
        ):
            raise ContractValidationError(
                f"{self.mode.value} mode cannot request objective admission review"
            )
        if (
            self.mode is GoalDevelopmentMode.ASSIST
            and self.decision is GoalAdmissionDecision.ADMITTED
        ):
            raise ContractValidationError("assist mode cannot admit proposals")

    @classmethod
    def for_proposal(
        cls,
        receipt: GoalDevelopmentProposalReceipt,
        *,
        mode: GoalDevelopmentMode,
        admitter_id: str,
        decision: GoalAdmissionDecision,
        authoritative_receipt_ids: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
    ) -> "GoalDevelopmentAdmissionReceipt":
        if not isinstance(receipt, GoalDevelopmentProposalReceipt):
            raise ContractValidationError(
                "receipt must be GoalDevelopmentProposalReceipt"
            )
        if (
            decision is GoalAdmissionDecision.ADMITTED
            and receipt.decision is not GoalProposalDecision.ACCEPTED
        ):
            raise ContractValidationError("a rejected proposal cannot be admitted")
        normalized_mode = _enum(mode, GoalDevelopmentMode, field_name="mode")
        if normalized_mode is not receipt.mode:
            raise ContractValidationError(
                "admission mode does not match the frozen proposal policy"
            )
        return cls(
            request_id=receipt.request_id,
            draft_id=receipt.draft_id,
            proposal_receipt_id=receipt.receipt_id,
            root_goal_id=receipt.root_goal_id,
            root_goal_content_id=receipt.root_goal_content_id,
            satisfaction_formula_id=receipt.satisfaction_formula_id,
            assumption_ids=receipt.assumption_ids,
            evidence_requirement_ids=receipt.evidence_requirement_ids,
            vocabulary_profile_id=receipt.vocabulary_profile_id,
            vocabulary_version=receipt.vocabulary_version,
            repository_tree_id=receipt.repository_tree_id,
            scope_ids=receipt.scope_ids,
            policy_digest=receipt.policy_digest,
            mode=normalized_mode,
            admitter_id=admitter_id,
            decision=decision,
            proposal_ids=receipt.proposal_ids,
            authoritative_receipt_ids=tuple(authoritative_receipt_ids),
            reason_codes=tuple(reason_codes),
        )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def authority(self) -> GoalDevelopmentAuthority:
        return GoalDevelopmentAuthority.SUPERVISOR_ADMISSION

    @property
    def admitted(self) -> bool:
        return self.decision is GoalAdmissionDecision.ADMITTED

    @property
    def proof_assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    def validate_proposal_receipt(
        self, receipt: GoalDevelopmentProposalReceipt
    ) -> None:
        """Fail unless this decision is bound to the exact proposal receipt."""

        if not isinstance(receipt, GoalDevelopmentProposalReceipt):
            raise ContractValidationError(
                "receipt must be GoalDevelopmentProposalReceipt"
            )
        expected = (
            receipt.request_id,
            receipt.draft_id,
            receipt.receipt_id,
            receipt.root_goal_id,
            receipt.root_goal_content_id,
            receipt.satisfaction_formula_id,
            receipt.assumption_ids,
            receipt.evidence_requirement_ids,
            receipt.vocabulary_profile_id,
            receipt.vocabulary_version,
            receipt.repository_tree_id,
            receipt.scope_ids,
            receipt.policy_digest,
            receipt.mode,
            receipt.proposal_ids,
        )
        actual = (
            self.request_id,
            self.draft_id,
            self.proposal_receipt_id,
            self.root_goal_id,
            self.root_goal_content_id,
            self.satisfaction_formula_id,
            self.assumption_ids,
            self.evidence_requirement_ids,
            self.vocabulary_profile_id,
            self.vocabulary_version,
            self.repository_tree_id,
            self.scope_ids,
            self.policy_digest,
            self.mode,
            self.proposal_ids,
        )
        if actual != expected:
            raise ContractValidationError(
                "admission receipt does not match the frozen proposal bindings"
            )
        if (
            self.admitted
            and receipt.decision is not GoalProposalDecision.ACCEPTED
        ):
            raise ContractValidationError(
                "an admission receipt cannot admit a rejected proposal"
            )

    def _payload(self) -> dict[str, Any]:
        return self._versioned(
            {
                "request_id": self.request_id,
                "draft_id": self.draft_id,
                "proposal_receipt_id": self.proposal_receipt_id,
                "root_goal_id": self.root_goal_id,
                "root_goal_content_id": self.root_goal_content_id,
                "satisfaction_formula_id": self.satisfaction_formula_id,
                "assumption_ids": self.assumption_ids,
                "evidence_requirement_ids": self.evidence_requirement_ids,
                "vocabulary_profile_id": self.vocabulary_profile_id,
                "vocabulary_version": self.vocabulary_version,
                "repository_tree_id": self.repository_tree_id,
                "scope_ids": self.scope_ids,
                "policy_digest": self.policy_digest,
                "mode": self.mode,
                "admitter_id": self.admitter_id,
                "decision": self.decision,
                "proposal_ids": self.proposal_ids,
                "authoritative_receipt_ids": self.authoritative_receipt_ids,
                "reason_codes": self.reason_codes,
                "authority": self.authority,
                "admitted": self.admitted,
                "proof_assurance": self.proof_assurance,
                "proof_claimed": False,
                "implementation_conformance_claimed": False,
                "implementation_conformant": False,
                "completion_claimed": False,
                "complete": False,
            }
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "GoalDevelopmentAdmissionReceipt":
        _schema_and_version(payload, cls.SCHEMA)
        _validate_false_claims(
            payload,
            artifact_name="goal admission receipt",
            allow_admitted=True,
        )
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "request_id",
            "draft_id",
            "proposal_receipt_id",
            "root_goal_id",
            "root_goal_content_id",
            "satisfaction_formula_id",
            "assumption_ids",
            "evidence_requirement_ids",
            "vocabulary_profile_id",
            "vocabulary_version",
            "repository_tree_id",
            "scope_ids",
            "policy_digest",
            "mode",
            "admitter_id",
            "decision",
            "proposal_ids",
            "authoritative_receipt_ids",
            "reason_codes",
            "authority",
            "admitted",
            "proof_assurance",
            "proof_claimed",
            "implementation_conformance_claimed",
            "implementation_conformant",
            "completion_claimed",
            "complete",
            "receipt_id",
            "content_id",
        }
        _reject_unknown(payload, allowed, artifact_name="goal admission receipt")
        if payload.get(
            "authority", GoalDevelopmentAuthority.SUPERVISOR_ADMISSION.value
        ) not in (
            GoalDevelopmentAuthority.SUPERVISOR_ADMISSION,
            GoalDevelopmentAuthority.SUPERVISOR_ADMISSION.value,
        ):
            raise ContractValidationError("goal admission receipt has invalid authority")
        if payload.get("proof_assurance", AssuranceLevel.UNVERIFIED.value) not in (
            AssuranceLevel.UNVERIFIED,
            AssuranceLevel.UNVERIFIED.value,
        ):
            raise ContractValidationError(
                "goal admission receipt cannot claim proof assurance"
            )
        result = cls(
            request_id=payload.get("request_id", ""),
            draft_id=payload.get("draft_id", ""),
            proposal_receipt_id=payload.get("proposal_receipt_id", ""),
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            satisfaction_formula_id=payload.get("satisfaction_formula_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            vocabulary_profile_id=payload.get("vocabulary_profile_id", ""),
            vocabulary_version=payload.get("vocabulary_version", 0),
            repository_tree_id=payload.get("repository_tree_id", ""),
            scope_ids=tuple(payload.get("scope_ids") or ()),
            policy_digest=payload.get("policy_digest", ""),
            mode=payload.get("mode", GoalDevelopmentMode.OFF),
            admitter_id=payload.get("admitter_id", ""),
            decision=payload.get(
                "decision", GoalAdmissionDecision.NOT_ADMITTED
            ),
            proposal_ids=tuple(payload.get("proposal_ids") or ()),
            authoritative_receipt_ids=tuple(
                payload.get("authoritative_receipt_ids") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        if payload.get("admitted") not in (None, result.admitted):
            raise ContractValidationError(
                "goal admission decision does not match admitted claim"
            )
        _claimed_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id"),
            artifact_name="goal admission receipt",
        )
        return result


# Concise compatibility spellings for callers that already establish the
# goal-development namespace at import time.  These are module-local only; the
# package initializer intentionally does not export them.
GoalProposalReceipt = GoalDevelopmentProposalReceipt
ProposalReceipt = GoalDevelopmentProposalReceipt
GoalAdmissionReceipt = GoalDevelopmentAdmissionReceipt
AdmissionReceipt = GoalDevelopmentAdmissionReceipt
GoalDecompositionItem = GoalDecompositionProposal
