"""Provider-free semantic steering contracts and closed intent classification.

ASE-023 defines the durable, body-free boundary for follow-up instructions
against an existing run. Classification is closed and deterministic. Optional
model output may only propose an intent; it never admits authority, effects, or
state mutation. Prompt text cannot select policy, principal, effect ceilings,
or other authority-bearing fields.

Runtime application of admitted deltas is deferred to ASE-024. Concurrent CAS,
leases, and fencing belong to ASE-025.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    MAX_PROMPT_BYTES,
    MAX_QUESTIONS,
    MAX_REASON_CODES,
    MAX_TIMEOUT_MS,
    ContractBoundsError,
    ContractIdentityError,
    EntrypointContractError,
    ExpectedEffect,
    SecretBearingRecordError,
    UnknownContractFieldError,
    _boolean,
    _CanonicalContract,
    _cid,
    _closed,
    _enum,
    _enum_tuple,
    _integer,
    _prompt_cid,
    _reason,
    _reference,
    _reject_embedded_prompt,
    _text,
    _text_tuple,
    cid_for_bytes,
)

STEERING_CONTRACT_REQUIREMENT_ID: Final = (
    "agent_supervisor.entrypoints.steering_contracts.v1"
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
STEERING_REQUEST_SCHEMA: Final = f"{SCHEMA_PREFIX}/steering-request@1"
STEERING_MODEL_PROPOSAL_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/steering-model-proposal@1"
)
STEERING_QUESTION_SCHEMA: Final = f"{SCHEMA_PREFIX}/steering-question@1"
STEERING_EVENT_SCHEMA: Final = f"{SCHEMA_PREFIX}/steering-event@1"
STEERING_RESULT_SCHEMA: Final = f"{SCHEMA_PREFIX}/steering-result@1"
STEERING_CLASSIFICATION_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/steering-classification@1"
)

MAX_AFFECTED_POPULATION: Final = 4_096
MAX_INSTRUCTION_BYTES: Final = MAX_PROMPT_BYTES
MAX_CANDIDATE_INTENTS: Final = 8
MAX_RULE_MATCHES: Final = 16

# Closed steering vocabulary from the prompt-entrypoint plan.
CLOSED_STEERING_INTENT_KINDS: Final[tuple[str, ...]] = (
    "append_requirement",
    "answer_question",
    "narrow_scope",
    "reprioritize",
    "request_replan",
    "pause",
    "resume",
    "cancel",
    "request_status",
)

# Effects a classified intent may request. Never derived from prompt prose.
INTENT_EFFECT_REQUIREMENTS: Final[Mapping[str, frozenset[str]]] = {
    "append_requirement": frozenset({"write_supervisor_state"}),
    "answer_question": frozenset({"write_supervisor_state"}),
    "narrow_scope": frozenset({"write_supervisor_state"}),
    "reprioritize": frozenset({"write_supervisor_state"}),
    "request_replan": frozenset({"write_supervisor_state"}),
    "pause": frozenset({"write_supervisor_state", "launch_local_process"}),
    "resume": frozenset({"write_supervisor_state", "launch_local_process"}),
    "cancel": frozenset({"write_supervisor_state", "launch_local_process"}),
    "request_status": frozenset(),
}

LIFECYCLE_INTENT_KINDS: Final[frozenset[str]] = frozenset(
    {"pause", "resume", "cancel"}
)
READ_ONLY_INTENT_KINDS: Final[frozenset[str]] = frozenset({"request_status"})

# Material mutation families used for ambiguity detection. Two matches in
# different families produce one bounded clarification question.
_MATERIAL_FAMILY: Final[Mapping[str, str]] = {
    "append_requirement": "plan_mutation",
    "answer_question": "plan_mutation",
    "narrow_scope": "plan_mutation",
    "reprioritize": "plan_mutation",
    "request_replan": "plan_mutation",
    "pause": "lifecycle",
    "resume": "lifecycle",
    "cancel": "lifecycle",
    "request_status": "status",
}

# Prompt text must never select these authority/effect/state selectors.
_FORBIDDEN_AUTHORITY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "prompt_selected_authority",
        re.compile(
            r"(?i)\b(?:set|grant|use|assume|select|choose|override)\b.{0,40}\b"
            r"(?:authority|principal|caller|ucan|capability)\b"
        ),
    ),
    (
        "prompt_selected_policy",
        re.compile(
            r"(?i)\b(?:set|grant|use|select|choose|override)\b.{0,40}\b"
            r"(?:policy|effect[_ -]?ceiling|permission)\b"
        ),
    ),
    (
        "prompt_selected_effect",
        re.compile(
            r"(?i)\b(?:allow|authorize|enable|permit)\b.{0,40}\b"
            r"(?:merge|push|deploy|destructive|secret|network)\b"
        ),
    ),
    (
        "prompt_selected_root",
        re.compile(
            r"(?i)\b(?:set|use|select|choose)\b.{0,40}\b"
            r"(?:repository[_ -]?root|state[_ -]?root|checkout)\b"
        ),
    ),
    (
        "prompt_selected_credentials",
        re.compile(
            r"(?i)\b(?:api[_ -]?key|bearer|password|private[_ -]?key|token)\b"
            r".{0,20}[:=]"
        ),
    ),
)

# Deterministic closed rules. Order is stable for replay; material conflicts
# are collected rather than first-match wins when families differ.
_DETERMINISTIC_RULES: Final[
    tuple[tuple[str, str, re.Pattern[str]], ...]
] = (
    (
        "request_status",
        "rule:request_status",
        re.compile(
            r"(?i)\b(?:"
            r"status|progress|how\s+is\s+(?:the\s+)?run|"
            r"what(?:'s|\s+is)\s+(?:the\s+)?(?:state|status)|"
            r"show\s+(?:me\s+)?(?:the\s+)?status|"
            r"current\s+(?:run\s+)?state"
            r")\b"
        ),
    ),
    (
        "answer_question",
        "rule:answer_question",
        re.compile(
            r"(?i)\b(?:"
            r"answer(?:\s+is|\s+the\s+question)?|"
            r"the\s+answer\s+is|"
            r"respond\s+with|"
            r"my\s+answer\s+is|"
            r"clarification\s*:"
            r")\b"
        ),
    ),
    (
        "pause",
        "rule:pause",
        re.compile(
            r"(?i)\b(?:"
            r"pause(?:\s+the\s+run)?|"
            r"hold(?:\s+(?:off|the\s+run))?|"
            r"suspend(?:\s+(?:work|the\s+run))?|"
            r"stop\s+temporarily|"
            r"freeze\s+(?:the\s+)?(?:run|work)"
            r")\b"
        ),
    ),
    (
        "resume",
        "rule:resume",
        re.compile(
            r"(?i)\b(?:"
            r"resume(?:\s+the\s+run)?|"
            r"unpause|"
            r"continue\s+(?:the\s+)?(?:run|work)|"
            r"restart\s+paused"
            r")\b"
        ),
    ),
    (
        "cancel",
        "rule:cancel",
        re.compile(
            r"(?i)\b(?:"
            r"cancel(?:\s+the\s+run)?|"
            r"abort(?:\s+the\s+run)?|"
            r"terminate(?:\s+the\s+run)?|"
            r"stop\s+(?:the\s+run\s+)?permanently|"
            r"kill\s+the\s+run"
            r")\b"
        ),
    ),
    (
        "narrow_scope",
        "rule:narrow_scope",
        re.compile(
            r"(?i)\b(?:"
            r"narrow(?:\s+(?:the\s+)?scope)?|"
            r"limit\s+(?:the\s+)?scope|"
            r"restrict\s+(?:to|scope)|"
            r"only\s+touch|"
            r"keep\s+scope\s+(?:narrow|limited)|"
            # Non-expansion asides ("without broadening") are constraints on
            # another intent, not a standalone narrow_scope classification.
            r"reduce\s+(?:the\s+)?scope|"
            r"shrink\s+(?:the\s+)?scope"
            r")\b"
        ),
    ),
    (
        "reprioritize",
        "rule:reprioritize",
        re.compile(
            r"(?i)\b(?:"
            r"reprioritize|"
            r"prioriti[sz]e|"
            r"raise\s+(?:the\s+)?priority|"
            r"do\s+.+\s+first|"
            r"higher\s+priority|"
            r"prefer\s+.+\s+over"
            r")\b"
        ),
    ),
    (
        "request_replan",
        "rule:request_replan",
        re.compile(
            r"(?i)\b(?:"
            r"replan|"
            r"request\s+replan|"
            r"redo\s+the\s+plan|"
            r"rethink\s+the\s+plan|"
            r"rebuild\s+the\s+plan|"
            r"new\s+plan\s+from\s+scratch"
            r")\b"
        ),
    ),
    (
        "append_requirement",
        "rule:append_requirement",
        re.compile(
            r"(?i)\b(?:"
            r"append\s+(?:a\s+)?requirement|"
            r"add(?:\s+a)?\s+requirement|"
            r"additionally\b|"
            r"also\s+require|"
            r"must\s+also|"
            r"ensure\s+that|"
            r"new\s+requirement\s*:"
            r")\b"
        ),
    ),
)


class SteeringContractError(EntrypointContractError):
    """A steering contract is malformed, ambiguous, or unsafe."""


class SteeringClassificationError(SteeringContractError):
    """Classification cannot admit a closed intent without guessing."""


class SteeringIntentKind(str, Enum):
    """Closed steering vocabulary. Prompt prose cannot invent new kinds."""

    APPEND_REQUIREMENT = "append_requirement"
    ANSWER_QUESTION = "answer_question"
    NARROW_SCOPE = "narrow_scope"
    REPRIORITIZE = "reprioritize"
    REQUEST_REPLAN = "request_replan"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    REQUEST_STATUS = "request_status"


class SteeringClassificationSource(str, Enum):
    """Where the admitted classification came from."""

    STRUCTURED_FIELD = "structured_field"
    DETERMINISTIC_RULE = "deterministic_rule"
    # Model output is recorded for audit but never alone admits classification.
    MODEL_PROPOSAL = "model_proposal"
    NONE = "none"


class SteeringDisposition(str, Enum):
    CLASSIFIED = "classified"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"
    DENIED = "denied"


class SteeringResultStatus(str, Enum):
    CLASSIFIED = "classified"
    NEEDS_INPUT = "needs_input"
    DENIED = "denied"
    REJECTED = "rejected"


class SteeringLifecycleRequest(str, Enum):
    """Lifecycle request named by classification; apply is a separate effect."""

    NONE = "none"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class SteeringProposalTier(str, Enum):
    """Model and free-form suggestions remain non-authoritative."""

    NONE = "none"
    PROPOSAL_ONLY = "proposal_only"


def _intent_kind(value: Any, name: str = "intent_kind") -> SteeringIntentKind:
    return _enum(value, SteeringIntentKind, name)


def _optional_intent_kind(
    value: Any, name: str = "intent_kind_hint"
) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, SteeringIntentKind):
        return value.value
    text = _text(value, name, required=False, maximum=64)
    if not text:
        return ""
    return _intent_kind(text, name).value


def _instruction_body(value: Any) -> bytes | None:
    if value is None:
        return None
    if type(value) is not bytes:
        raise SteeringContractError(
            "transient_instruction_body must be exact bytes"
        )
    if not value:
        raise SteeringContractError(
            "transient_instruction_body must not be empty"
        )
    if len(value) > MAX_INSTRUCTION_BYTES:
        raise ContractBoundsError(
            f"transient_instruction_body exceeds {MAX_INSTRUCTION_BYTES} bytes"
        )
    return value


@dataclass(frozen=True)
class SteeringModelProposal(_CanonicalContract):
    """Optional model-assisted intent suggestion. Always proposal-tier only."""

    SCHEMA: ClassVar[str] = STEERING_MODEL_PROPOSAL_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "proposed_intent_kind",
        "confidence_ppm",
        "rationale_ref",
        "producer_ref",
        "proposal_receipt_cid",
        "tier",
    )

    proposed_intent_kind: SteeringIntentKind
    confidence_ppm: int = 0
    rationale_ref: str = ""
    producer_ref: str = ""
    proposal_receipt_cid: str = ""
    tier: SteeringProposalTier = SteeringProposalTier.PROPOSAL_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposed_intent_kind",
            _intent_kind(self.proposed_intent_kind, "proposed_intent_kind"),
        )
        object.__setattr__(
            self,
            "confidence_ppm",
            _integer(
                self.confidence_ppm,
                "confidence_ppm",
                maximum=1_000_000,
            ),
        )
        object.__setattr__(
            self,
            "rationale_ref",
            _reference(self.rationale_ref, "rationale_ref", required=False),
        )
        object.__setattr__(
            self,
            "producer_ref",
            _reference(self.producer_ref, "producer_ref", required=False),
        )
        object.__setattr__(
            self,
            "proposal_receipt_cid",
            _cid(
                self.proposal_receipt_cid,
                "proposal_receipt_cid",
                required=False,
            ),
        )
        tier = _enum(self.tier, SteeringProposalTier, "tier")
        if tier is not SteeringProposalTier.PROPOSAL_ONLY:
            raise SteeringContractError(
                "model proposals must remain proposal_only tier"
            )
        object.__setattr__(self, "tier", tier)

    @property
    def is_authoritative(self) -> bool:
        """Model proposals never admit effects or authority."""

        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "proposed_intent_kind": self.proposed_intent_kind.value,
            "confidence_ppm": self.confidence_ppm,
            "rationale_ref": self.rationale_ref,
            "producer_ref": self.producer_ref,
            "proposal_receipt_cid": self.proposal_receipt_cid,
            "tier": self.tier.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SteeringModelProposal:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class SteeringQuestion(_CanonicalContract):
    """One bounded typed question when material interpretations conflict."""

    SCHEMA: ClassVar[str] = STEERING_QUESTION_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "question_code",
        "candidate_intents",
        "context_ref",
    )

    question_code: str
    candidate_intents: tuple[SteeringIntentKind, ...]
    context_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "question_code",
            _reason(self.question_code, "question_code"),
        )
        intents = _enum_tuple(
            self.candidate_intents,
            SteeringIntentKind,
            "candidate_intents",
            maximum_items=MAX_CANDIDATE_INTENTS,
            sorted_items=True,
        )
        if len(intents) < 2:
            raise SteeringContractError(
                "clarification questions require at least two candidate intents"
            )
        object.__setattr__(self, "candidate_intents", intents)
        object.__setattr__(
            self,
            "context_ref",
            _reference(self.context_ref, "context_ref", required=False),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "question_code": self.question_code,
            "candidate_intents": [item.value for item in self.candidate_intents],
            "context_ref": self.context_ref,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SteeringQuestion:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(
            question_code=value["question_code"],
            candidate_intents=tuple(value["candidate_intents"]),
            context_ref=value["context_ref"],
        )
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class SteeringRequest(_CanonicalContract):
    """Body-free durable steering request bound to an exact run revision.

    ``transient_instruction_body`` may exist only in process memory. It is
    checked against ``instruction_prompt_cid`` and never appears in equality,
    repr, canonical bytes, JSON, or the request CID.
    """

    SCHEMA: ClassVar[str] = STEERING_REQUEST_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "run_id",
        "expected_run_revision",
        "expected_plan_revision",
        "expected_task_source_revision",
        "instruction_prompt_cid",
        "instruction_prompt_ref",
        "intent_kind_hint",
        "affected_population_bound",
        "deadline_ms",
        "principal_ref",
        "policy_cid",
        "effect_ceiling_cid",
        "allowed_effects",
        "allow_lifecycle_request",
        "idempotency_key",
        "event_cursor",
        "model_proposal_cid",
    )

    run_id: str
    expected_run_revision: str
    expected_plan_revision: str
    expected_task_source_revision: str
    instruction_prompt_cid: str
    instruction_prompt_ref: str
    intent_kind_hint: str = ""
    affected_population_bound: int = 64
    deadline_ms: int = 3_600_000
    principal_ref: str = ""
    policy_cid: str = ""
    effect_ceiling_cid: str = ""
    allowed_effects: tuple[ExpectedEffect, ...] = ()
    allow_lifecycle_request: bool = False
    idempotency_key: str = ""
    event_cursor: str = ""
    model_proposal_cid: str = ""
    transient_instruction_body: bytes | None = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_id", _reference(self.run_id, "run_id")
        )
        object.__setattr__(
            self,
            "expected_run_revision",
            _cid(self.expected_run_revision, "expected_run_revision"),
        )
        object.__setattr__(
            self,
            "expected_plan_revision",
            _cid(self.expected_plan_revision, "expected_plan_revision"),
        )
        object.__setattr__(
            self,
            "expected_task_source_revision",
            _cid(
                self.expected_task_source_revision,
                "expected_task_source_revision",
            ),
        )
        object.__setattr__(
            self,
            "instruction_prompt_cid",
            _prompt_cid(self.instruction_prompt_cid, "instruction_prompt_cid"),
        )
        object.__setattr__(
            self,
            "instruction_prompt_ref",
            _reference(self.instruction_prompt_ref, "instruction_prompt_ref"),
        )
        object.__setattr__(
            self,
            "intent_kind_hint",
            _optional_intent_kind(self.intent_kind_hint),
        )
        object.__setattr__(
            self,
            "affected_population_bound",
            _integer(
                self.affected_population_bound,
                "affected_population_bound",
                minimum=1,
                maximum=MAX_AFFECTED_POPULATION,
            ),
        )
        object.__setattr__(
            self,
            "deadline_ms",
            _integer(
                self.deadline_ms,
                "deadline_ms",
                minimum=1,
                maximum=MAX_TIMEOUT_MS,
            ),
        )
        # Authority and effects are authenticated/profile-bound fields. Empty
        # is allowed only for pure classification previews; mutation admission
        # in later tasks must require them. Prompt text never fills these.
        object.__setattr__(
            self,
            "principal_ref",
            _reference(self.principal_ref, "principal_ref", required=False),
        )
        object.__setattr__(
            self,
            "policy_cid",
            _cid(self.policy_cid, "policy_cid", required=False),
        )
        object.__setattr__(
            self,
            "effect_ceiling_cid",
            _cid(self.effect_ceiling_cid, "effect_ceiling_cid", required=False),
        )
        effects = _enum_tuple(
            self.allowed_effects,
            ExpectedEffect,
            "allowed_effects",
            maximum_items=32,
            sorted_items=True,
        )
        object.__setattr__(self, "allowed_effects", effects)
        object.__setattr__(
            self,
            "allow_lifecycle_request",
            _boolean(self.allow_lifecycle_request, "allow_lifecycle_request"),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _reference(
                self.idempotency_key, "idempotency_key", required=False
            ),
        )
        object.__setattr__(
            self,
            "event_cursor",
            _reference(self.event_cursor, "event_cursor", required=False),
        )
        object.__setattr__(
            self,
            "model_proposal_cid",
            _cid(self.model_proposal_cid, "model_proposal_cid", required=False),
        )
        body = _instruction_body(self.transient_instruction_body)
        object.__setattr__(self, "transient_instruction_body", body)
        if body is not None:
            if cid_for_bytes(body, codec="raw") != self.instruction_prompt_cid:
                raise ContractIdentityError(
                    "transient_instruction_body does not match "
                    "instruction_prompt_cid"
                )
            for name in (
                "run_id",
                "instruction_prompt_ref",
                "principal_ref",
                "idempotency_key",
                "event_cursor",
            ):
                _reject_embedded_prompt(
                    getattr(self, name),
                    prompt_body=body,
                    name=name,
                )

    @classmethod
    def from_instruction(
        cls,
        instruction: str | bytes,
        *,
        instruction_prompt_ref: str,
        run_id: str,
        expected_run_revision: str,
        expected_plan_revision: str,
        expected_task_source_revision: str,
        **values: Any,
    ) -> SteeringRequest:
        if isinstance(instruction, str):
            body = instruction.encode("utf-8")
        elif type(instruction) is bytes:
            body = instruction
        else:
            raise SteeringContractError(
                "instruction must be text or exact bytes"
            )
        return cls(
            run_id=run_id,
            expected_run_revision=expected_run_revision,
            expected_plan_revision=expected_plan_revision,
            expected_task_source_revision=expected_task_source_revision,
            instruction_prompt_cid=cid_for_bytes(body, codec="raw"),
            instruction_prompt_ref=instruction_prompt_ref,
            transient_instruction_body=body,
            **values,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_id": self.run_id,
            "expected_run_revision": self.expected_run_revision,
            "expected_plan_revision": self.expected_plan_revision,
            "expected_task_source_revision": self.expected_task_source_revision,
            "instruction_prompt_cid": self.instruction_prompt_cid,
            "instruction_prompt_ref": self.instruction_prompt_ref,
            "intent_kind_hint": self.intent_kind_hint,
            "affected_population_bound": self.affected_population_bound,
            "deadline_ms": self.deadline_ms,
            "principal_ref": self.principal_ref,
            "policy_cid": self.policy_cid,
            "effect_ceiling_cid": self.effect_ceiling_cid,
            "allowed_effects": [item.value for item in self.allowed_effects],
            "allow_lifecycle_request": self.allow_lifecycle_request,
            "idempotency_key": self.idempotency_key,
            "event_cursor": self.event_cursor,
            "model_proposal_cid": self.model_proposal_cid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SteeringRequest:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(
            run_id=value["run_id"],
            expected_run_revision=value["expected_run_revision"],
            expected_plan_revision=value["expected_plan_revision"],
            expected_task_source_revision=value[
                "expected_task_source_revision"
            ],
            instruction_prompt_cid=value["instruction_prompt_cid"],
            instruction_prompt_ref=value["instruction_prompt_ref"],
            intent_kind_hint=value["intent_kind_hint"],
            affected_population_bound=value["affected_population_bound"],
            deadline_ms=value["deadline_ms"],
            principal_ref=value["principal_ref"],
            policy_cid=value["policy_cid"],
            effect_ceiling_cid=value["effect_ceiling_cid"],
            allowed_effects=tuple(value["allowed_effects"]),
            allow_lifecycle_request=value["allow_lifecycle_request"],
            idempotency_key=value["idempotency_key"],
            event_cursor=value["event_cursor"],
            model_proposal_cid=value["model_proposal_cid"],
        )
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class SteeringClassification(_CanonicalContract):
    """Replayable closed interpretation of one steering instruction."""

    SCHEMA: ClassVar[str] = STEERING_CLASSIFICATION_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "request_cid",
        "disposition",
        "intent_kind",
        "source",
        "matched_rule_ids",
        "candidate_intents",
        "reason_codes",
        "forbidden_selector_codes",
        "required_effects",
        "lifecycle_request",
        "model_proposal_tier",
        "model_proposal_cid",
    )

    request_cid: str
    disposition: SteeringDisposition
    intent_kind: str
    source: SteeringClassificationSource
    matched_rule_ids: tuple[str, ...]
    candidate_intents: tuple[SteeringIntentKind, ...]
    reason_codes: tuple[str, ...]
    forbidden_selector_codes: tuple[str, ...]
    required_effects: tuple[ExpectedEffect, ...]
    lifecycle_request: SteeringLifecycleRequest
    model_proposal_tier: SteeringProposalTier
    model_proposal_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_cid", _cid(self.request_cid, "request_cid")
        )
        disposition = _enum(
            self.disposition, SteeringDisposition, "disposition"
        )
        object.__setattr__(self, "disposition", disposition)
        intent = _optional_intent_kind(self.intent_kind, "intent_kind")
        object.__setattr__(self, "intent_kind", intent)
        source = _enum(self.source, SteeringClassificationSource, "source")
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "matched_rule_ids",
            _text_tuple(
                self.matched_rule_ids,
                "matched_rule_ids",
                maximum_items=MAX_RULE_MATCHES,
                item_kind="reason",
                sorted_items=True,
            ),
        )
        candidates = _enum_tuple(
            self.candidate_intents,
            SteeringIntentKind,
            "candidate_intents",
            maximum_items=MAX_CANDIDATE_INTENTS,
            sorted_items=True,
        )
        object.__setattr__(self, "candidate_intents", candidates)
        object.__setattr__(
            self,
            "reason_codes",
            _text_tuple(
                self.reason_codes,
                "reason_codes",
                maximum_items=MAX_REASON_CODES,
                item_kind="reason",
                sorted_items=True,
            ),
        )
        object.__setattr__(
            self,
            "forbidden_selector_codes",
            _text_tuple(
                self.forbidden_selector_codes,
                "forbidden_selector_codes",
                maximum_items=MAX_REASON_CODES,
                item_kind="reason",
                sorted_items=True,
            ),
        )
        effects = _enum_tuple(
            self.required_effects,
            ExpectedEffect,
            "required_effects",
            maximum_items=32,
            sorted_items=True,
        )
        object.__setattr__(self, "required_effects", effects)
        lifecycle = _enum(
            self.lifecycle_request,
            SteeringLifecycleRequest,
            "lifecycle_request",
        )
        object.__setattr__(self, "lifecycle_request", lifecycle)
        proposal_tier = _enum(
            self.model_proposal_tier,
            SteeringProposalTier,
            "model_proposal_tier",
        )
        object.__setattr__(self, "model_proposal_tier", proposal_tier)
        object.__setattr__(
            self,
            "model_proposal_cid",
            _cid(
                self.model_proposal_cid,
                "model_proposal_cid",
                required=False,
            ),
        )
        if disposition is SteeringDisposition.CLASSIFIED:
            if not intent:
                raise SteeringContractError(
                    "classified disposition requires intent_kind"
                )
            if source in {
                SteeringClassificationSource.NONE,
                SteeringClassificationSource.MODEL_PROPOSAL,
            }:
                raise SteeringContractError(
                    "classified disposition cannot be admitted from "
                    "model_proposal or none source"
                )
            if lifecycle is not SteeringLifecycleRequest.NONE:
                if intent not in LIFECYCLE_INTENT_KINDS:
                    raise SteeringContractError(
                        "lifecycle_request must match a lifecycle intent"
                    )
                if lifecycle.value != intent:
                    raise SteeringContractError(
                        "lifecycle_request must equal classified lifecycle intent"
                    )
            elif intent in LIFECYCLE_INTENT_KINDS:
                raise SteeringContractError(
                    "lifecycle intents require a matching lifecycle_request"
                )
        else:
            if intent:
                raise SteeringContractError(
                    "non-classified disposition cannot carry intent_kind"
                )
            if lifecycle is not SteeringLifecycleRequest.NONE:
                raise SteeringContractError(
                    "non-classified disposition cannot request lifecycle action"
                )
        if (
            disposition is SteeringDisposition.NEEDS_CLARIFICATION
            and len(candidates) < 2
        ):
            raise SteeringContractError(
                "needs_clarification requires at least two candidate intents"
            )
        # Model output is never an admitting source; proposal tier is recorded
        # separately on model_proposal_tier / model_proposal_cid.
        if source is SteeringClassificationSource.MODEL_PROPOSAL:
            raise SteeringContractError(
                "MODEL_PROPOSAL cannot be the admitting classification source"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "request_cid": self.request_cid,
            "disposition": self.disposition.value,
            "intent_kind": self.intent_kind,
            "source": self.source.value,
            "matched_rule_ids": list(self.matched_rule_ids),
            "candidate_intents": [item.value for item in self.candidate_intents],
            "reason_codes": list(self.reason_codes),
            "forbidden_selector_codes": list(self.forbidden_selector_codes),
            "required_effects": [item.value for item in self.required_effects],
            "lifecycle_request": self.lifecycle_request.value,
            "model_proposal_tier": self.model_proposal_tier.value,
            "model_proposal_cid": self.model_proposal_cid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SteeringClassification:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(
            request_cid=value["request_cid"],
            disposition=value["disposition"],
            intent_kind=value["intent_kind"],
            source=value["source"],
            matched_rule_ids=tuple(value["matched_rule_ids"]),
            candidate_intents=tuple(value["candidate_intents"]),
            reason_codes=tuple(value["reason_codes"]),
            forbidden_selector_codes=tuple(value["forbidden_selector_codes"]),
            required_effects=tuple(value["required_effects"]),
            lifecycle_request=value["lifecycle_request"],
            model_proposal_tier=value["model_proposal_tier"],
            model_proposal_cid=value["model_proposal_cid"],
        )
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class SteeringEvent(_CanonicalContract):
    """Append-only steering event binding classification to a run revision.

    Mutation fields (plan_delta_cid, deferred successors) remain empty until a
    later runtime admits and applies a delta. This event never itself alters
    task-source or run state.
    """

    SCHEMA: ClassVar[str] = STEERING_EVENT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "run_id",
        "request_cid",
        "classification_cid",
        "expected_run_revision",
        "expected_plan_revision",
        "expected_task_source_revision",
        "intent_kind",
        "disposition",
        "lifecycle_request",
        "plan_delta_cid",
        "deferred_successor_cids",
        "event_cursor",
        "idempotency_key",
        "reason_codes",
        "state_mutated",
    )

    run_id: str
    request_cid: str
    classification_cid: str
    expected_run_revision: str
    expected_plan_revision: str
    expected_task_source_revision: str
    intent_kind: str
    disposition: SteeringDisposition
    lifecycle_request: SteeringLifecycleRequest
    plan_delta_cid: str = ""
    deferred_successor_cids: tuple[str, ...] = ()
    event_cursor: str = ""
    idempotency_key: str = ""
    reason_codes: tuple[str, ...] = ()
    state_mutated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _reference(self.run_id, "run_id"))
        object.__setattr__(
            self, "request_cid", _cid(self.request_cid, "request_cid")
        )
        object.__setattr__(
            self,
            "classification_cid",
            _cid(self.classification_cid, "classification_cid"),
        )
        object.__setattr__(
            self,
            "expected_run_revision",
            _cid(self.expected_run_revision, "expected_run_revision"),
        )
        object.__setattr__(
            self,
            "expected_plan_revision",
            _cid(self.expected_plan_revision, "expected_plan_revision"),
        )
        object.__setattr__(
            self,
            "expected_task_source_revision",
            _cid(
                self.expected_task_source_revision,
                "expected_task_source_revision",
            ),
        )
        object.__setattr__(
            self,
            "intent_kind",
            _optional_intent_kind(self.intent_kind, "intent_kind"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, SteeringDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "lifecycle_request",
            _enum(
                self.lifecycle_request,
                SteeringLifecycleRequest,
                "lifecycle_request",
            ),
        )
        object.__setattr__(
            self,
            "plan_delta_cid",
            _cid(self.plan_delta_cid, "plan_delta_cid", required=False),
        )
        object.__setattr__(
            self,
            "deferred_successor_cids",
            _text_tuple(
                self.deferred_successor_cids,
                "deferred_successor_cids",
                maximum_items=MAX_AFFECTED_POPULATION,
                item_kind="cid",
                sorted_items=True,
            ),
        )
        object.__setattr__(
            self,
            "event_cursor",
            _reference(self.event_cursor, "event_cursor", required=False),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _reference(
                self.idempotency_key, "idempotency_key", required=False
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _text_tuple(
                self.reason_codes,
                "reason_codes",
                maximum_items=MAX_REASON_CODES,
                item_kind="reason",
                sorted_items=True,
            ),
        )
        object.__setattr__(
            self,
            "state_mutated",
            _boolean(self.state_mutated, "state_mutated"),
        )
        # Classification events never mutate state; apply is ASE-024.
        if self.state_mutated:
            raise SteeringContractError(
                "steering contract events cannot claim state mutation"
            )
        if self.plan_delta_cid or self.deferred_successor_cids:
            raise SteeringContractError(
                "classification-tier steering events cannot carry plan deltas "
                "or deferred successors"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_id": self.run_id,
            "request_cid": self.request_cid,
            "classification_cid": self.classification_cid,
            "expected_run_revision": self.expected_run_revision,
            "expected_plan_revision": self.expected_plan_revision,
            "expected_task_source_revision": self.expected_task_source_revision,
            "intent_kind": self.intent_kind,
            "disposition": self.disposition.value,
            "lifecycle_request": self.lifecycle_request.value,
            "plan_delta_cid": self.plan_delta_cid,
            "deferred_successor_cids": list(self.deferred_successor_cids),
            "event_cursor": self.event_cursor,
            "idempotency_key": self.idempotency_key,
            "reason_codes": list(self.reason_codes),
            "state_mutated": self.state_mutated,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SteeringEvent:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(
            run_id=value["run_id"],
            request_cid=value["request_cid"],
            classification_cid=value["classification_cid"],
            expected_run_revision=value["expected_run_revision"],
            expected_plan_revision=value["expected_plan_revision"],
            expected_task_source_revision=value[
                "expected_task_source_revision"
            ],
            intent_kind=value["intent_kind"],
            disposition=value["disposition"],
            lifecycle_request=value["lifecycle_request"],
            plan_delta_cid=value["plan_delta_cid"],
            deferred_successor_cids=tuple(value["deferred_successor_cids"]),
            event_cursor=value["event_cursor"],
            idempotency_key=value["idempotency_key"],
            reason_codes=tuple(value["reason_codes"]),
            state_mutated=value["state_mutated"],
        )
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class SteeringResult(_CanonicalContract):
    """Outcome of closed intent classification for one steering request."""

    SCHEMA: ClassVar[str] = STEERING_RESULT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "request_cid",
        "status",
        "classification",
        "event",
        "questions",
        "reason_codes",
        "error_code",
        "model_proposal_tier",
    )

    request_cid: str
    status: SteeringResultStatus
    classification: SteeringClassification
    event: SteeringEvent | None
    questions: tuple[SteeringQuestion, ...]
    reason_codes: tuple[str, ...]
    error_code: str
    model_proposal_tier: SteeringProposalTier

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_cid", _cid(self.request_cid, "request_cid")
        )
        status = _enum(self.status, SteeringResultStatus, "status")
        object.__setattr__(self, "status", status)
        classification = self.classification
        if not isinstance(classification, SteeringClassification):
            if isinstance(classification, Mapping):
                classification = SteeringClassification.from_dict(
                    classification
                )
            else:
                raise SteeringContractError(
                    "classification must be a SteeringClassification"
                )
        object.__setattr__(self, "classification", classification)
        if classification.request_cid != self.request_cid:
            raise SteeringContractError(
                "classification.request_cid must match result request_cid"
            )
        event = self.event
        if event is not None and not isinstance(event, SteeringEvent):
            if isinstance(event, Mapping):
                event = SteeringEvent.from_dict(event)
            else:
                raise SteeringContractError("event must be a SteeringEvent")
        object.__setattr__(self, "event", event)
        if isinstance(self.questions, (str, bytes)) or not isinstance(
            self.questions, Sequence
        ):
            raise SteeringContractError("questions must be a sequence")
        if len(self.questions) > MAX_QUESTIONS:
            raise ContractBoundsError(
                f"questions exceeds {MAX_QUESTIONS} items"
            )
        # Acceptance: material ambiguity yields one bounded question.
        if len(self.questions) > 1:
            raise SteeringContractError(
                "materially different interpretations produce at most one "
                "bounded question"
            )
        normalized_questions: list[SteeringQuestion] = []
        for item in self.questions:
            if isinstance(item, SteeringQuestion):
                normalized_questions.append(item)
            elif isinstance(item, Mapping):
                normalized_questions.append(SteeringQuestion.from_dict(item))
            else:
                raise SteeringContractError(
                    "questions must contain SteeringQuestion records"
                )
        object.__setattr__(self, "questions", tuple(normalized_questions))
        object.__setattr__(
            self,
            "reason_codes",
            _text_tuple(
                self.reason_codes,
                "reason_codes",
                maximum_items=MAX_REASON_CODES,
                item_kind="reason",
                sorted_items=True,
            ),
        )
        error = self.error_code
        if error:
            error = _reason(error, "error_code")
        object.__setattr__(self, "error_code", error)
        proposal_tier = _enum(
            self.model_proposal_tier,
            SteeringProposalTier,
            "model_proposal_tier",
        )
        object.__setattr__(self, "model_proposal_tier", proposal_tier)

        if status is SteeringResultStatus.CLASSIFIED:
            if classification.disposition is not SteeringDisposition.CLASSIFIED:
                raise SteeringContractError(
                    "classified results require classified disposition"
                )
            if event is None:
                raise SteeringContractError(
                    "classified results require a steering event"
                )
            if self.questions:
                raise SteeringContractError(
                    "classified results cannot carry questions"
                )
            if error:
                raise SteeringContractError(
                    "classified results cannot carry error_code"
                )
            if event.request_cid != self.request_cid:
                raise SteeringContractError(
                    "event.request_cid must match result request_cid"
                )
            if event.classification_cid != classification.content_id:
                raise SteeringContractError(
                    "event.classification_cid must match classification"
                )
            if event.intent_kind != classification.intent_kind:
                raise SteeringContractError(
                    "event intent must match classification"
                )
            if event.state_mutated:
                raise SteeringContractError(
                    "classification results cannot claim state mutation"
                )
        elif status is SteeringResultStatus.NEEDS_INPUT:
            if (
                classification.disposition
                is not SteeringDisposition.NEEDS_CLARIFICATION
            ):
                raise SteeringContractError(
                    "needs_input requires needs_clarification disposition"
                )
            if not self.questions:
                raise SteeringContractError(
                    "needs_input requires exactly one bounded question"
                )
            if event is not None:
                raise SteeringContractError(
                    "needs_input cannot emit a mutation-bound event"
                )
            if error:
                raise SteeringContractError(
                    "needs_input cannot carry error_code"
                )
        elif status in {
            SteeringResultStatus.DENIED,
            SteeringResultStatus.REJECTED,
        }:
            expected = (
                SteeringDisposition.DENIED
                if status is SteeringResultStatus.DENIED
                else SteeringDisposition.REJECTED
            )
            if classification.disposition is not expected:
                raise SteeringContractError(
                    f"{status.value} requires {expected.value} disposition"
                )
            if not error:
                raise SteeringContractError(
                    f"{status.value} requires error_code"
                )
            if self.questions:
                raise SteeringContractError(
                    f"{status.value} cannot carry questions"
                )
            if event is not None:
                raise SteeringContractError(
                    f"{status.value} cannot emit a steering event"
                )
        if proposal_tier is not classification.model_proposal_tier:
            raise SteeringContractError(
                "result model_proposal_tier must match classification"
            )

    @property
    def admits_runtime_apply(self) -> bool:
        """True only when classification is closed and ready for ASE-024."""

        return (
            self.status is SteeringResultStatus.CLASSIFIED
            and self.event is not None
            and not self.event.state_mutated
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "request_cid": self.request_cid,
            "status": self.status.value,
            "classification": self.classification.to_dict(),
            "event": self.event.to_dict() if self.event is not None else None,
            "questions": [item.to_dict() for item in self.questions],
            "reason_codes": list(self.reason_codes),
            "error_code": self.error_code,
            "model_proposal_tier": self.model_proposal_tier.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SteeringResult:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        event_value = value["event"]
        result = cls(
            request_cid=value["request_cid"],
            status=value["status"],
            classification=SteeringClassification.from_dict(
                value["classification"]
            ),
            event=(
                None
                if event_value is None
                else SteeringEvent.from_dict(event_value)
            ),
            questions=tuple(value["questions"]),
            reason_codes=tuple(value["reason_codes"]),
            error_code=value["error_code"],
            model_proposal_tier=value["model_proposal_tier"],
        )
        return cls._verify_claimed(value, result)


def _instruction_text(request: SteeringRequest) -> str:
    body = request.transient_instruction_body
    if body is None:
        return ""
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SteeringContractError(
            "instruction body must be UTF-8 text for classification"
        ) from exc


def _scan_forbidden_selectors(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    codes: list[str] = []
    for code, pattern in _FORBIDDEN_AUTHORITY_PATTERNS:
        if pattern.search(text):
            codes.append(code)
    return tuple(sorted(set(codes)))


def _deterministic_matches(text: str) -> tuple[tuple[SteeringIntentKind, str], ...]:
    if not text or not text.strip():
        return ()
    matches: list[tuple[SteeringIntentKind, str]] = []
    seen: set[SteeringIntentKind] = set()
    for kind_value, rule_id, pattern in _DETERMINISTIC_RULES:
        if pattern.search(text):
            kind = SteeringIntentKind(kind_value)
            if kind not in seen:
                seen.add(kind)
                matches.append((kind, rule_id))
    return tuple(matches)


def _material_families(
    kinds: Sequence[SteeringIntentKind],
) -> frozenset[str]:
    return frozenset(_MATERIAL_FAMILY[kind.value] for kind in kinds)


def _required_effects_for(kind: SteeringIntentKind) -> tuple[ExpectedEffect, ...]:
    names = INTENT_EFFECT_REQUIREMENTS[kind.value]
    return tuple(
        sorted(
            (ExpectedEffect(name) for name in names),
            key=lambda item: item.value,
        )
    )


def _lifecycle_for(kind: SteeringIntentKind) -> SteeringLifecycleRequest:
    if kind.value in LIFECYCLE_INTENT_KINDS:
        return SteeringLifecycleRequest(kind.value)
    return SteeringLifecycleRequest.NONE


def _effects_permitted(
    kind: SteeringIntentKind,
    allowed: Sequence[ExpectedEffect],
) -> bool:
    required = set(_required_effects_for(kind))
    if not required:
        return True
    allowed_set = set(allowed)
    return required.issubset(allowed_set)


def _normalize_model_proposal(
    proposal: SteeringModelProposal | Mapping[str, Any] | None,
    *,
    request: SteeringRequest,
) -> SteeringModelProposal | None:
    if proposal is None:
        return None
    if isinstance(proposal, Mapping):
        proposal = SteeringModelProposal.from_dict(proposal)
    if not isinstance(proposal, SteeringModelProposal):
        raise SteeringContractError(
            "model_proposal must be a SteeringModelProposal"
        )
    if proposal.is_authoritative:
        raise SteeringContractError(
            "model proposals cannot be marked authoritative"
        )
    # Optional binding: when both the request and proposal declare a proposal
    # identity, they must agree. A free-standing proposal object used only for
    # classify() need not equal a request-side pin.
    if (
        request.model_proposal_cid
        and proposal.proposal_receipt_cid
        and proposal.proposal_receipt_cid != request.model_proposal_cid
        and proposal.content_id != request.model_proposal_cid
    ):
        raise ContractIdentityError(
            "model_proposal does not match request.model_proposal_cid"
        )
    return proposal


def classify_steering_instruction(
    request: SteeringRequest,
    *,
    model_proposal: SteeringModelProposal | Mapping[str, Any] | None = None,
) -> SteeringResult:
    """Classify a steering instruction with closed deterministic rules.

    Optional ``model_proposal`` is recorded as proposal-tier only. It cannot
    alone select the intent, authority, effects, or mutate state. When
    deterministic rules find materially different interpretations, exactly one
    bounded question is returned.
    """

    if not isinstance(request, SteeringRequest):
        raise SteeringContractError("request must be a SteeringRequest")

    proposal = _normalize_model_proposal(model_proposal, request=request)
    proposal_tier = (
        SteeringProposalTier.PROPOSAL_ONLY
        if proposal is not None or request.model_proposal_cid
        else SteeringProposalTier.NONE
    )
    proposal_cid = ""
    if proposal is not None:
        proposal_cid = proposal.content_id
    elif request.model_proposal_cid:
        proposal_cid = request.model_proposal_cid

    text = _instruction_text(request)
    forbidden = _scan_forbidden_selectors(text)
    reason_codes: list[str] = []
    if forbidden:
        reason_codes.append("prompt_cannot_select_authority_or_effects")
        reason_codes.extend(forbidden)

    structured_kind: SteeringIntentKind | None = None
    if request.intent_kind_hint:
        structured_kind = SteeringIntentKind(request.intent_kind_hint)

    matches = _deterministic_matches(text)
    matched_kinds = tuple(kind for kind, _ in matches)
    matched_rules = tuple(rule_id for _, rule_id in matches)

    # Structured field is authoritative for classification when present and
    # valid under policy bounds. Deterministic text must not contradict it with
    # a different material family.
    if structured_kind is not None:
        conflicting = [
            kind
            for kind in matched_kinds
            if _MATERIAL_FAMILY[kind.value]
            != _MATERIAL_FAMILY[structured_kind.value]
        ]
        if conflicting:
            candidates = tuple(
                sorted(
                    {structured_kind, *conflicting},
                    key=lambda item: item.value,
                )
            )
            return _clarification_result(
                request,
                candidates=candidates,
                matched_rule_ids=matched_rules,
                reason_codes=tuple(
                    sorted(
                        {
                            *reason_codes,
                            "structured_and_text_materially_conflict",
                        }
                    )
                ),
                forbidden=forbidden,
                proposal_tier=proposal_tier,
                proposal_cid=proposal_cid,
                question_code="choose_steering_intent",
            )
        return _admit_or_deny(
            request,
            kind=structured_kind,
            source=SteeringClassificationSource.STRUCTURED_FIELD,
            matched_rule_ids=matched_rules,
            reason_codes=tuple(
                sorted({*reason_codes, "structured_intent_kind"})
            ),
            forbidden=forbidden,
            proposal_tier=proposal_tier,
            proposal_cid=proposal_cid,
        )

    if not matched_kinds:
        # Model proposal alone never classifies.
        if proposal is not None:
            reason_codes.append("model_proposal_non_authoritative")
            reason_codes.append("unsupported_without_deterministic_match")
            return _reject_result(
                request,
                error_code="unsupported_instruction",
                disposition=SteeringDisposition.REJECTED,
                status=SteeringResultStatus.REJECTED,
                reason_codes=tuple(sorted(set(reason_codes))),
                forbidden=forbidden,
                proposal_tier=proposal_tier,
                proposal_cid=proposal_cid,
                matched_rule_ids=(),
                candidate_intents=(proposal.proposed_intent_kind,),
            )
        if not text.strip():
            reason_codes.append("empty_instruction")
            error = "empty_instruction"
        else:
            reason_codes.append("unsupported_instruction")
            error = "unsupported_instruction"
        return _reject_result(
            request,
            error_code=error,
            disposition=SteeringDisposition.REJECTED,
            status=SteeringResultStatus.REJECTED,
            reason_codes=tuple(sorted(set(reason_codes))),
            forbidden=forbidden,
            proposal_tier=proposal_tier,
            proposal_cid=proposal_cid,
            matched_rule_ids=(),
            candidate_intents=(),
        )

    families = _material_families(matched_kinds)
    if len(matched_kinds) > 1 and len(families) > 1:
        candidates = tuple(
            sorted(matched_kinds, key=lambda item: item.value)
        )
        return _clarification_result(
            request,
            candidates=candidates,
            matched_rule_ids=matched_rules,
            reason_codes=tuple(
                sorted({*reason_codes, "materially_ambiguous_instruction"})
            ),
            forbidden=forbidden,
            proposal_tier=proposal_tier,
            proposal_cid=proposal_cid,
            question_code="choose_steering_intent",
        )

    # Same material family with multiple kinds: still ask rather than guess.
    if len(matched_kinds) > 1:
        candidates = tuple(
            sorted(matched_kinds, key=lambda item: item.value)
        )
        return _clarification_result(
            request,
            candidates=candidates,
            matched_rule_ids=matched_rules,
            reason_codes=tuple(
                sorted({*reason_codes, "multiple_closed_intents"})
            ),
            forbidden=forbidden,
            proposal_tier=proposal_tier,
            proposal_cid=proposal_cid,
            question_code="choose_steering_intent",
        )

    kind = matched_kinds[0]
    # Model proposal may agree or disagree; disagreement is recorded only.
    if proposal is not None:
        if proposal.proposed_intent_kind is kind:
            reason_codes.append("model_proposal_agrees")
        else:
            reason_codes.append("model_proposal_ignored")
        reason_codes.append("model_proposal_non_authoritative")

    return _admit_or_deny(
        request,
        kind=kind,
        source=SteeringClassificationSource.DETERMINISTIC_RULE,
        matched_rule_ids=matched_rules,
        reason_codes=tuple(sorted(set(reason_codes))),
        forbidden=forbidden,
        proposal_tier=proposal_tier,
        proposal_cid=proposal_cid,
    )


def _admit_or_deny(
    request: SteeringRequest,
    *,
    kind: SteeringIntentKind,
    source: SteeringClassificationSource,
    matched_rule_ids: tuple[str, ...],
    reason_codes: tuple[str, ...],
    forbidden: tuple[str, ...],
    proposal_tier: SteeringProposalTier,
    proposal_cid: str,
) -> SteeringResult:
    codes = list(reason_codes)
    if kind.value in LIFECYCLE_INTENT_KINDS and not request.allow_lifecycle_request:
        codes.append("lifecycle_request_not_permitted")
        return _reject_result(
            request,
            error_code="lifecycle_not_permitted",
            disposition=SteeringDisposition.DENIED,
            status=SteeringResultStatus.DENIED,
            reason_codes=tuple(sorted(set(codes))),
            forbidden=forbidden,
            proposal_tier=proposal_tier,
            proposal_cid=proposal_cid,
            matched_rule_ids=matched_rule_ids,
            candidate_intents=(kind,),
        )
    if not _effects_permitted(kind, request.allowed_effects):
        codes.append("required_effects_exceed_ceiling")
        return _reject_result(
            request,
            error_code="effect_ceiling_exceeded",
            disposition=SteeringDisposition.DENIED,
            status=SteeringResultStatus.DENIED,
            reason_codes=tuple(sorted(set(codes))),
            forbidden=forbidden,
            proposal_tier=proposal_tier,
            proposal_cid=proposal_cid,
            matched_rule_ids=matched_rule_ids,
            candidate_intents=(kind,),
        )
    # Forbidden selectors in prompt text never change authority; classification
    # of a valid closed intent may still proceed with audit codes.
    if forbidden:
        codes.append("prompt_authority_selectors_ignored")

    required = _required_effects_for(kind)
    lifecycle = _lifecycle_for(kind)
    classification = SteeringClassification(
        request_cid=request.content_id,
        disposition=SteeringDisposition.CLASSIFIED,
        intent_kind=kind.value,
        source=source,
        matched_rule_ids=matched_rule_ids,
        candidate_intents=(kind,),
        reason_codes=tuple(sorted(set(codes))),
        forbidden_selector_codes=forbidden,
        required_effects=required,
        lifecycle_request=lifecycle,
        model_proposal_tier=proposal_tier,
        model_proposal_cid=proposal_cid,
    )
    event = SteeringEvent(
        run_id=request.run_id,
        request_cid=request.content_id,
        classification_cid=classification.content_id,
        expected_run_revision=request.expected_run_revision,
        expected_plan_revision=request.expected_plan_revision,
        expected_task_source_revision=request.expected_task_source_revision,
        intent_kind=kind.value,
        disposition=SteeringDisposition.CLASSIFIED,
        lifecycle_request=lifecycle,
        plan_delta_cid="",
        deferred_successor_cids=(),
        event_cursor=request.event_cursor,
        idempotency_key=request.idempotency_key,
        reason_codes=tuple(sorted(set(codes))),
        state_mutated=False,
    )
    return SteeringResult(
        request_cid=request.content_id,
        status=SteeringResultStatus.CLASSIFIED,
        classification=classification,
        event=event,
        questions=(),
        reason_codes=tuple(sorted(set(codes))),
        error_code="",
        model_proposal_tier=proposal_tier,
    )


def _clarification_result(
    request: SteeringRequest,
    *,
    candidates: tuple[SteeringIntentKind, ...],
    matched_rule_ids: tuple[str, ...],
    reason_codes: tuple[str, ...],
    forbidden: tuple[str, ...],
    proposal_tier: SteeringProposalTier,
    proposal_cid: str,
    question_code: str,
) -> SteeringResult:
    ordered = tuple(sorted(candidates, key=lambda item: item.value))
    question = SteeringQuestion(
        question_code=question_code,
        candidate_intents=ordered,
        context_ref=request.instruction_prompt_ref,
    )
    classification = SteeringClassification(
        request_cid=request.content_id,
        disposition=SteeringDisposition.NEEDS_CLARIFICATION,
        intent_kind="",
        source=SteeringClassificationSource.NONE,
        matched_rule_ids=matched_rule_ids,
        candidate_intents=ordered,
        reason_codes=reason_codes,
        forbidden_selector_codes=forbidden,
        required_effects=(),
        lifecycle_request=SteeringLifecycleRequest.NONE,
        model_proposal_tier=proposal_tier,
        model_proposal_cid=proposal_cid,
    )
    return SteeringResult(
        request_cid=request.content_id,
        status=SteeringResultStatus.NEEDS_INPUT,
        classification=classification,
        event=None,
        questions=(question,),
        reason_codes=reason_codes,
        error_code="",
        model_proposal_tier=proposal_tier,
    )


def _reject_result(
    request: SteeringRequest,
    *,
    error_code: str,
    disposition: SteeringDisposition,
    status: SteeringResultStatus,
    reason_codes: tuple[str, ...],
    forbidden: tuple[str, ...],
    proposal_tier: SteeringProposalTier,
    proposal_cid: str,
    matched_rule_ids: tuple[str, ...],
    candidate_intents: tuple[SteeringIntentKind, ...],
) -> SteeringResult:
    classification = SteeringClassification(
        request_cid=request.content_id,
        disposition=disposition,
        intent_kind="",
        source=SteeringClassificationSource.NONE,
        matched_rule_ids=matched_rule_ids,
        candidate_intents=candidate_intents,
        reason_codes=reason_codes,
        forbidden_selector_codes=forbidden,
        required_effects=(),
        lifecycle_request=SteeringLifecycleRequest.NONE,
        model_proposal_tier=proposal_tier,
        model_proposal_cid=proposal_cid,
    )
    return SteeringResult(
        request_cid=request.content_id,
        status=status,
        classification=classification,
        event=None,
        questions=(),
        reason_codes=reason_codes,
        error_code=error_code,
        model_proposal_tier=proposal_tier,
    )


def closed_intent_vocabulary() -> tuple[str, ...]:
    """Return the frozen closed steering intent vocabulary."""

    return CLOSED_STEERING_INTENT_KINDS


def intent_requires_lifecycle_authorization(kind: SteeringIntentKind | str) -> bool:
    value = kind.value if isinstance(kind, SteeringIntentKind) else str(kind)
    return value in LIFECYCLE_INTENT_KINDS


__all__ = [
    "CLOSED_STEERING_INTENT_KINDS",
    "INTENT_EFFECT_REQUIREMENTS",
    "LIFECYCLE_INTENT_KINDS",
    "READ_ONLY_INTENT_KINDS",
    "STEERING_CLASSIFICATION_SCHEMA",
    "STEERING_CONTRACT_REQUIREMENT_ID",
    "STEERING_EVENT_SCHEMA",
    "STEERING_MODEL_PROPOSAL_SCHEMA",
    "STEERING_QUESTION_SCHEMA",
    "STEERING_REQUEST_SCHEMA",
    "STEERING_RESULT_SCHEMA",
    "ContractBoundsError",
    "ContractIdentityError",
    "EntrypointContractError",
    "ExpectedEffect",
    "SecretBearingRecordError",
    "SteeringClassification",
    "SteeringClassificationError",
    "SteeringClassificationSource",
    "SteeringContractError",
    "SteeringDisposition",
    "SteeringEvent",
    "SteeringIntentKind",
    "SteeringLifecycleRequest",
    "SteeringModelProposal",
    "SteeringProposalTier",
    "SteeringQuestion",
    "SteeringRequest",
    "SteeringResult",
    "SteeringResultStatus",
    "UnknownContractFieldError",
    "classify_steering_instruction",
    "closed_intent_vocabulary",
    "intent_requires_lifecycle_authorization",
]
