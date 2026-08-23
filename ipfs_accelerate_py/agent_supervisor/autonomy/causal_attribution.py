# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Evidence-gated causal attribution and bounded shadow-only ablations.

``CausalAttributionEngine`` classifies compact public episodes against the
closed :class:`AttributionCause` vocabulary.  It does not execute ablations,
call a provider, store prompts or source bodies, or admit production
acceptance.  Semantic-governor and adversarial-assurance authorities remain
the executors of any admitted shadow comparison.

Assignment rules
----------------
* A cause is emitted only when discriminating evidence isolates it.
* Observational correlation, frequency, and compact metrics are not causes.
* A model is never blamed while required source is omitted or untested.
* Compression is never credited from one compressed pass.
* Controlled ablations are shadow-only and cannot affect production
  acceptance.  Ablation-backed attributions remain shadow-only.
* Stated confounders that still have supporting evidence force abstention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION,
    canonical_json_bytes,
    content_identity,
)
from .contracts import (
    MAX_CANONICAL_RECORD_BYTES,
    MAX_IDENTIFIER_BYTES,
    MAX_MAPPING_ITEMS,
    MAX_NESTING_DEPTH,
    MAX_SEQUENCE_ITEMS,
    AttributionCause,
    AutonomyContractError,
    CausalAttribution,
    ExperienceEpisode,
    MetaAction,
    TerminalStatus,
)

CAUSAL_ATTRIBUTION_ENGINE_INTERFACE: Final[str] = "CausalAttributionEngine@1"
CAUSAL_ATTRIBUTION_INTERFACE: Final[str] = "CausalAttribution@1"
CAUSAL_ATTRIBUTION_ENGINE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/causal-attribution-engine@1"
)
ATTRIBUTION_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/attribution-observation@1"
)
CONTROLLED_ABLATION_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/controlled-ablation-proposal@1"
)
CAUSAL_ATTRIBUTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/causal-attribution-result@1"
)
COMPRESSION_CREDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/compression-credit@1"
)

MAX_ATTRIBUTION_OBSERVATIONS: Final[int] = MAX_SEQUENCE_ITEMS
MAX_ABLATION_PROPOSALS: Final[int] = 32
CONFIDENCE_WITNESS_BP: Final[int] = 8_500
CONFIDENCE_PAIRED_BP: Final[int] = 9_000
CONFIDENCE_INDEPENDENT_BP: Final[int] = 9_500

_FAILURE_STATUSES: Final[frozenset[TerminalStatus]] = frozenset(
    {
        TerminalStatus.FAILED,
        TerminalStatus.BLOCKED,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.EXHAUSTED,
    }
)
_MODEL_ACTIONS: Final[frozenset[MetaAction]] = frozenset(
    {
        MetaAction.CALL_LOCAL_SMALL_MODEL,
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
)
_CACHE_ACTIONS: Final[frozenset[MetaAction]] = frozenset({MetaAction.READ_CACHED_RECEIPT})
_VALIDATION_ACTIONS: Final[frozenset[MetaAction]] = frozenset(
    {
        MetaAction.RUN_SCHEMA_VALIDATION,
        MetaAction.RUN_TYPE_CHECK,
        MetaAction.RUN_SELECTED_TEST,
        MetaAction.RUN_FULL_VALIDATION,
    }
)
_PROOF_ACTIONS: Final[frozenset[MetaAction]] = frozenset({MetaAction.RUN_SMT_OR_PROVER})
_PLAN_ACTIONS: Final[frozenset[MetaAction]] = frozenset({MetaAction.REPLAN_AFFECTED_SUFFIX})
_DECOMPOSITION_ACTIONS: Final[frozenset[MetaAction]] = frozenset(
    {MetaAction.GENERATE_BOUNDED_REPAIR}
)

# Upstream / isolating order.  Model capability is last so omitted source,
# provider, and environment evidence cannot be displaced by a model guess.
_CAUSE_PRECEDENCE: Final[tuple[AttributionCause, ...]] = (
    AttributionCause.HUMAN_POLICY_BLOCKER,
    AttributionCause.ENVIRONMENT_FAILURE,
    AttributionCause.PROVIDER_FAILURE,
    AttributionCause.CONTEXT_OMISSION,
    AttributionCause.STALE_EVIDENCE,
    AttributionCause.INCORRECT_CACHE_REUSE,
    AttributionCause.BAD_TASK_DECOMPOSITION,
    AttributionCause.BAD_PLAN_BRANCH,
    AttributionCause.VALIDATION_SELECTION_FAILURE,
    AttributionCause.PROOF_SELECTION_FAILURE,
    AttributionCause.MERGE_CONFLICT,
    AttributionCause.MODEL_CAPABILITY_FAILURE,
)

_CONFOUNDER_PAIRS: Final[frozenset[frozenset[AttributionCause]]] = frozenset(
    {
        frozenset(
            {AttributionCause.CONTEXT_OMISSION, AttributionCause.MODEL_CAPABILITY_FAILURE}
        ),
        frozenset({AttributionCause.PROVIDER_FAILURE, AttributionCause.ENVIRONMENT_FAILURE}),
        frozenset({AttributionCause.STALE_EVIDENCE, AttributionCause.INCORRECT_CACHE_REUSE}),
        frozenset(
            {AttributionCause.BAD_TASK_DECOMPOSITION, AttributionCause.BAD_PLAN_BRANCH}
        ),
    }
)

_FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "chain_of_thought",
        "chat_messages",
        "completion",
        "cookie",
        "credential",
        "decoded_source",
        "executable_code",
        "hidden_reasoning",
        "messages",
        "model_output",
        "model_transcript",
        "password",
        "private_key",
        "private_reasoning",
        "prompt",
        "raw_prompt",
        "refresh_token",
        "secret",
        "shell_command",
        "source_body",
        "source_text",
        "transcript",
    }
)

_ALLOWED_OBSERVATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "contract_version",
        "observation_id",
        "kind",
        "evidence_ids",
        "episode_ids",
        "omitted_reference_ids",
        "factor",
        "baseline_terminal_status",
        "contrast_terminal_status",
        "contrast_episode_ids",
        "confounder_ids",
        "shadow_only",
        "production_acceptance",
    }
)


class CausalAttributionError(ValueError):
    """Raised when attribution inputs violate the closed evidence contract."""


class AttributionEvidenceKind(str, Enum):
    """Closed vocabulary of public, identity-only attribution observations."""

    COMPLETENESS_WITNESS = "completeness_witness"
    CONTEXT_SUFFICIENT = "context_sufficient"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    ENVIRONMENT_PROBE_FAILURE = "environment_probe_failure"
    ENVIRONMENT_PROBE_SUCCESS = "environment_probe_success"
    STALE_IDENTITY = "stale_identity"
    CACHE_BINDING_MISMATCH = "cache_binding_mismatch"
    VALIDATION_SELECTOR_MISMATCH = "validation_selector_mismatch"
    PROOF_SELECTOR_MISMATCH = "proof_selector_mismatch"
    MERGE_CONFLICT = "merge_conflict"
    HUMAN_POLICY_BLOCK = "human_policy_block"
    DECOMPOSITION_FAILURE = "decomposition_failure"
    PLAN_BRANCH_FAILURE = "plan_branch_failure"
    PAIRED_COMPARISON = "paired_comparison"
    SINGLE_PASS_SUCCESS = "single_pass_success"


class AblationFactor(str, Enum):
    """One controlled factor a shadow ablation may toggle."""

    CONTEXT_COMPLETENESS = "context_completeness"
    COMPRESSION = "compression"
    PROVIDER = "provider"
    MODEL = "model"
    PLAN_BRANCH = "plan_branch"
    TASK_DECOMPOSITION = "task_decomposition"
    CACHE_REUSE = "cache_reuse"
    EVIDENCE_FRESHNESS = "evidence_freshness"
    VALIDATION_SELECTION = "validation_selection"
    PROOF_SELECTION = "proof_selection"
    MERGE_QUEUE = "merge_queue"
    ENVIRONMENT = "environment"
    HUMAN_POLICY = "human_policy"


class AttributionDisposition(str, Enum):
    """Closed outcome of one attribution attempt."""

    ATTRIBUTED = "attributed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFOUNDER_PRESENT = "confounder_present"
    ABLATION_REQUIRED = "ablation_required"


class SupportLevel(str, Enum):
    """Internal strength of evidence for one closed cause."""

    NONE = "none"
    SUPPORTING = "supporting"
    DISCRIMINATING = "discriminating"


_FACTOR_CAUSE: Final[Mapping[AblationFactor, AttributionCause]] = MappingProxyType(
    {
        AblationFactor.CONTEXT_COMPLETENESS: AttributionCause.CONTEXT_OMISSION,
        AblationFactor.COMPRESSION: AttributionCause.CONTEXT_OMISSION,
        AblationFactor.PROVIDER: AttributionCause.PROVIDER_FAILURE,
        AblationFactor.MODEL: AttributionCause.MODEL_CAPABILITY_FAILURE,
        AblationFactor.PLAN_BRANCH: AttributionCause.BAD_PLAN_BRANCH,
        AblationFactor.TASK_DECOMPOSITION: AttributionCause.BAD_TASK_DECOMPOSITION,
        AblationFactor.CACHE_REUSE: AttributionCause.INCORRECT_CACHE_REUSE,
        AblationFactor.EVIDENCE_FRESHNESS: AttributionCause.STALE_EVIDENCE,
        AblationFactor.VALIDATION_SELECTION: AttributionCause.VALIDATION_SELECTION_FAILURE,
        AblationFactor.PROOF_SELECTION: AttributionCause.PROOF_SELECTION_FAILURE,
        AblationFactor.MERGE_QUEUE: AttributionCause.MERGE_CONFLICT,
        AblationFactor.ENVIRONMENT: AttributionCause.ENVIRONMENT_FAILURE,
        AblationFactor.HUMAN_POLICY: AttributionCause.HUMAN_POLICY_BLOCKER,
    }
)

_CAUSE_CONTRAST_ACTION: Final[Mapping[AttributionCause, MetaAction]] = MappingProxyType(
    {
        AttributionCause.CONTEXT_OMISSION: MetaAction.EXPAND_CONTEXT_REFERENCE,
        AttributionCause.MODEL_CAPABILITY_FAILURE: MetaAction.CALL_LOCAL_SMALL_MODEL,
        AttributionCause.PROVIDER_FAILURE: MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        AttributionCause.ENVIRONMENT_FAILURE: MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        AttributionCause.BAD_TASK_DECOMPOSITION: MetaAction.GENERATE_BOUNDED_REPAIR,
        AttributionCause.BAD_PLAN_BRANCH: MetaAction.REPLAN_AFFECTED_SUFFIX,
        AttributionCause.STALE_EVIDENCE: MetaAction.RUN_SELECTED_TEST,
        AttributionCause.INCORRECT_CACHE_REUSE: MetaAction.RUN_SELECTED_TEST,
        AttributionCause.VALIDATION_SELECTION_FAILURE: MetaAction.RUN_SELECTED_TEST,
        AttributionCause.PROOF_SELECTION_FAILURE: MetaAction.RUN_SMT_OR_PROVER,
        AttributionCause.MERGE_CONFLICT: MetaAction.QUARANTINE_TASK,
        AttributionCause.HUMAN_POLICY_BLOCKER: MetaAction.REQUEST_HUMAN_DECISION,
    }
)

_CAUSE_FACTOR: Final[Mapping[AttributionCause, AblationFactor]] = MappingProxyType(
    {
        AttributionCause.CONTEXT_OMISSION: AblationFactor.CONTEXT_COMPLETENESS,
        AttributionCause.MODEL_CAPABILITY_FAILURE: AblationFactor.MODEL,
        AttributionCause.PROVIDER_FAILURE: AblationFactor.PROVIDER,
        AttributionCause.ENVIRONMENT_FAILURE: AblationFactor.ENVIRONMENT,
        AttributionCause.BAD_TASK_DECOMPOSITION: AblationFactor.TASK_DECOMPOSITION,
        AttributionCause.BAD_PLAN_BRANCH: AblationFactor.PLAN_BRANCH,
        AttributionCause.STALE_EVIDENCE: AblationFactor.EVIDENCE_FRESHNESS,
        AttributionCause.INCORRECT_CACHE_REUSE: AblationFactor.CACHE_REUSE,
        AttributionCause.VALIDATION_SELECTION_FAILURE: AblationFactor.VALIDATION_SELECTION,
        AttributionCause.PROOF_SELECTION_FAILURE: AblationFactor.PROOF_SELECTION,
        AttributionCause.MERGE_CONFLICT: AblationFactor.MERGE_QUEUE,
        AttributionCause.HUMAN_POLICY_BLOCKER: AblationFactor.HUMAN_POLICY,
    }
)


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise CausalAttributionError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CausalAttributionError(f"{name} must be a boolean")
    return value


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise CausalAttributionError(f"{name} must be a string")
    if not result:
        if required:
            raise CausalAttributionError(f"{name} must be a compact bounded identifier")
        return ""
    if (
        len(result.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
    ):
        raise CausalAttributionError(f"{name} must be a compact bounded identifier")
    return result


def _identifiers(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise CausalAttributionError(f"{name} must be a sequence of identifiers")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise CausalAttributionError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        identifier = _identifier(item, name)
        if identifier not in seen:
            seen.add(identifier)
            normalized.append(identifier)
    if required and not normalized:
        raise CausalAttributionError(f"{name} must not be empty")
    return tuple(sorted(normalized))


def _normalize_field_name(key: Any) -> str:
    if not isinstance(key, str):
        raise CausalAttributionError("attribution field names must be strings")
    return key.strip().lower().replace("-", "_")


def field_is_forbidden(key: Any) -> bool:
    """Return whether a mapping key is a secret, prompt, source, or transcript."""

    normalized = _normalize_field_name(key)
    if not normalized:
        return False
    return any(
        normalized == marker or normalized.endswith("_" + marker)
        for marker in _FORBIDDEN_FIELD_MARKERS
    )


def _reject_forbidden_payload(value: Any, name: str, *, depth: int = 0) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise CausalAttributionError(f"{name} exceeds maximum nesting")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        raise CausalAttributionError(f"{name} cannot contain floats")
    if isinstance(value, Enum):
        return
    if isinstance(value, str):
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise CausalAttributionError(f"{name} contains too many entries")
        for raw_key, raw_value in value.items():
            if field_is_forbidden(raw_key):
                raise CausalAttributionError(
                    f"{name} contains forbidden private or executable data"
                )
            _reject_forbidden_payload(raw_value, f"{name}.{raw_key}", depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise CausalAttributionError(f"{name} contains too many items")
        for index, item in enumerate(value):
            _reject_forbidden_payload(item, f"{name}[{index}]", depth=depth + 1)
        return
    raise CausalAttributionError(f"{name} contains unsupported value type {type(value).__name__}")


def _optional_enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if value is None or value == "":
        return None
    return _enum(value, enum_type, name)


def _metric_int(metrics: Mapping[str, Any], key: str) -> int:
    raw = metrics.get(key, 0)
    if raw is None:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise CausalAttributionError(f"context_metrics.{key} must be a compact non-negative integer")
    return raw


def _coerce_episode(value: Any) -> ExperienceEpisode:
    if isinstance(value, ExperienceEpisode):
        _reject_forbidden_payload(value.to_dict(), "episode")
        return value
    if not isinstance(value, Mapping):
        raise CausalAttributionError("episode must be an ExperienceEpisode or mapping")
    _reject_forbidden_payload(value, "episode")
    try:
        return ExperienceEpisode.from_dict(value)
    except AutonomyContractError as exc:
        raise CausalAttributionError(str(exc)) from exc


def _coerce_episodes(values: Any) -> tuple[ExperienceEpisode, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (ExperienceEpisode, Mapping)):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise CausalAttributionError("episodes must be a sequence of ExperienceEpisode values")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise CausalAttributionError("episodes contains too many items")
    episodes = tuple(_coerce_episode(item) for item in raw)
    if not episodes:
        raise CausalAttributionError("episodes must not be empty")
    seen: set[str] = set()
    unique: list[ExperienceEpisode] = []
    for episode in episodes:
        if episode.episode_id not in seen:
            seen.add(episode.episode_id)
            unique.append(episode)
    return tuple(unique)


def _coerce_observations(values: Any) -> tuple["AttributionObservation", ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, AttributionObservation):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise CausalAttributionError(
            "observations must be a sequence of AttributionObservation values"
        )
    if len(raw) > MAX_ATTRIBUTION_OBSERVATIONS:
        raise CausalAttributionError("observations exceeds its bounded size")
    return tuple(AttributionObservation.from_dict(item) for item in raw)


def confounder_pair(left: AttributionCause, right: AttributionCause) -> bool:
    """Return whether two causes cannot be jointly assigned without isolation."""

    if left is right:
        return False
    return frozenset({left, right}) in _CONFOUNDER_PAIRS


def ablation_may_affect_production_acceptance(
    proposal: "ControlledAblationProposal" | Mapping[str, Any] | None = None,
) -> bool:
    """Ablations never participate in production acceptance."""

    if proposal is None:
        return False
    item = ControlledAblationProposal.from_dict(proposal)
    if item.affects_production_acceptance or not item.shadow_only:
        raise CausalAttributionError("controlled ablations cannot affect production acceptance")
    return False


@dataclass(frozen=True)
class AttributionObservation:
    """One typed public observation.  Identities only; never a prompt or body."""

    observation_id: str
    kind: AttributionEvidenceKind
    evidence_ids: tuple[str, ...]
    episode_ids: tuple[str, ...] = ()
    omitted_reference_ids: tuple[str, ...] = ()
    factor: AblationFactor | None = None
    baseline_terminal_status: TerminalStatus | None = None
    contrast_terminal_status: TerminalStatus | None = None
    contrast_episode_ids: tuple[str, ...] = ()
    confounder_ids: tuple[str, ...] = ()
    shadow_only: bool = True
    production_acceptance: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, AttributionEvidenceKind, "kind")
        )
        object.__setattr__(
            self,
            "observation_id",
            _identifier(self.observation_id, "observation_id"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _identifiers(self.evidence_ids, "evidence_ids", required=True),
        )
        object.__setattr__(
            self, "episode_ids", _identifiers(self.episode_ids, "episode_ids")
        )
        object.__setattr__(
            self,
            "omitted_reference_ids",
            _identifiers(self.omitted_reference_ids, "omitted_reference_ids"),
        )
        object.__setattr__(
            self, "factor", _optional_enum(self.factor, AblationFactor, "factor")
        )
        object.__setattr__(
            self,
            "baseline_terminal_status",
            _optional_enum(
                self.baseline_terminal_status, TerminalStatus, "baseline_terminal_status"
            ),
        )
        object.__setattr__(
            self,
            "contrast_terminal_status",
            _optional_enum(
                self.contrast_terminal_status, TerminalStatus, "contrast_terminal_status"
            ),
        )
        object.__setattr__(
            self,
            "contrast_episode_ids",
            _identifiers(self.contrast_episode_ids, "contrast_episode_ids"),
        )
        object.__setattr__(
            self,
            "confounder_ids",
            _identifiers(self.confounder_ids, "confounder_ids"),
        )
        object.__setattr__(self, "shadow_only", _bool(self.shadow_only, "shadow_only"))
        object.__setattr__(
            self,
            "production_acceptance",
            _bool(self.production_acceptance, "production_acceptance"),
        )
        if self.kind is AttributionEvidenceKind.COMPLETENESS_WITNESS:
            if not self.omitted_reference_ids:
                raise CausalAttributionError(
                    "completeness_witness must name omitted_reference_ids"
                )
        if self.kind is AttributionEvidenceKind.CONTEXT_SUFFICIENT:
            if self.omitted_reference_ids:
                raise CausalAttributionError(
                    "context_sufficient cannot name omitted_reference_ids"
                )
        if self.kind is AttributionEvidenceKind.PAIRED_COMPARISON:
            if self.factor is None:
                raise CausalAttributionError("paired_comparison requires a closed ablation factor")
            if self.baseline_terminal_status is None or self.contrast_terminal_status is None:
                raise CausalAttributionError(
                    "paired_comparison requires baseline and contrast terminal statuses"
                )
            if not self.episode_ids or not (
                self.contrast_episode_ids or len(self.episode_ids) >= 2
            ):
                raise CausalAttributionError(
                    "paired_comparison requires baseline and contrast episode identities"
                )
            if not self.shadow_only:
                raise CausalAttributionError("paired comparisons must remain shadow-only")
            if self.production_acceptance:
                raise CausalAttributionError(
                    "controlled ablations cannot affect production acceptance"
                )
        elif self.production_acceptance:
            raise CausalAttributionError(
                "attribution observations cannot declare production acceptance"
            )
        payload = self.to_dict()
        _reject_forbidden_payload(payload, "observation")

    @property
    def is_ablation(self) -> bool:
        return self.kind is AttributionEvidenceKind.PAIRED_COMPARISON

    @property
    def outcome_flipped(self) -> bool:
        if not self.is_ablation:
            return False
        baseline_failed = self.baseline_terminal_status in _FAILURE_STATUSES
        contrast_failed = self.contrast_terminal_status in _FAILURE_STATUSES
        return baseline_failed != contrast_failed

    @property
    def both_failed(self) -> bool:
        if not self.is_ablation:
            return False
        return (
            self.baseline_terminal_status in _FAILURE_STATUSES
            and self.contrast_terminal_status in _FAILURE_STATUSES
        )

    @property
    def both_succeeded(self) -> bool:
        if not self.is_ablation:
            return False
        return (
            self.baseline_terminal_status is TerminalStatus.SUCCEEDED
            and self.contrast_terminal_status is TerminalStatus.SUCCEEDED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ATTRIBUTION_OBSERVATION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "observation_id": self.observation_id,
            "kind": self.kind.value,
            "evidence_ids": list(self.evidence_ids),
            "episode_ids": list(self.episode_ids),
            "omitted_reference_ids": list(self.omitted_reference_ids),
            "factor": None if self.factor is None else self.factor.value,
            "baseline_terminal_status": None
            if self.baseline_terminal_status is None
            else self.baseline_terminal_status.value,
            "contrast_terminal_status": None
            if self.contrast_terminal_status is None
            else self.contrast_terminal_status.value,
            "contrast_episode_ids": list(self.contrast_episode_ids),
            "confounder_ids": list(self.confounder_ids),
            "shadow_only": self.shadow_only,
            "production_acceptance": self.production_acceptance,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | AttributionObservation
    ) -> AttributionObservation:
        if isinstance(payload, AttributionObservation):
            return payload
        if not isinstance(payload, Mapping):
            raise CausalAttributionError("attribution observation must be an object")
        _reject_forbidden_payload(payload, "observation")
        extra = set(payload).difference(_ALLOWED_OBSERVATION_FIELDS)
        if extra:
            if any(field_is_forbidden(key) for key in extra):
                raise CausalAttributionError(
                    "observation contains forbidden private or executable data"
                )
            raise CausalAttributionError("observation contains unsupported fields")
        return cls(
            observation_id=str(payload.get("observation_id", "")),
            kind=payload.get("kind"),  # type: ignore[arg-type]
            evidence_ids=payload.get("evidence_ids") or (),
            episode_ids=payload.get("episode_ids") or (),
            omitted_reference_ids=payload.get("omitted_reference_ids") or (),
            factor=payload.get("factor"),
            baseline_terminal_status=payload.get("baseline_terminal_status"),
            contrast_terminal_status=payload.get("contrast_terminal_status"),
            contrast_episode_ids=payload.get("contrast_episode_ids") or (),
            confounder_ids=payload.get("confounder_ids") or (),
            shadow_only=payload.get("shadow_only", True),
            production_acceptance=payload.get("production_acceptance", False),
        )


@dataclass(frozen=True)
class ControlledAblationProposal:
    """Bounded shadow comparison.  SCG/AAE may execute it; this engine does not."""

    factor: AblationFactor
    hypothesized_cause: AttributionCause
    baseline_episode_ids: tuple[str, ...]
    contrast_action: MetaAction
    expected_evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    shadow_only: bool = True
    affects_production_acceptance: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", _enum(self.factor, AblationFactor, "factor"))
        object.__setattr__(
            self,
            "hypothesized_cause",
            _enum(self.hypothesized_cause, AttributionCause, "hypothesized_cause"),
        )
        object.__setattr__(
            self,
            "baseline_episode_ids",
            _identifiers(self.baseline_episode_ids, "baseline_episode_ids", required=True),
        )
        object.__setattr__(
            self,
            "contrast_action",
            _enum(self.contrast_action, MetaAction, "contrast_action"),
        )
        object.__setattr__(
            self,
            "expected_evidence_ids",
            _identifiers(self.expected_evidence_ids, "expected_evidence_ids"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True),
        )
        object.__setattr__(self, "shadow_only", _bool(self.shadow_only, "shadow_only"))
        object.__setattr__(
            self,
            "affects_production_acceptance",
            _bool(self.affects_production_acceptance, "affects_production_acceptance"),
        )
        if not self.shadow_only or self.affects_production_acceptance:
            raise CausalAttributionError(
                "controlled ablations cannot affect production acceptance"
            )
        encoded = canonical_json_bytes(self.to_dict())
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise CausalAttributionError("ablation proposal exceeds its bounded size")

    @property
    def proposal_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("proposal_id", None)
        return content_identity(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CONTROLLED_ABLATION_PROPOSAL_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "factor": self.factor.value,
            "hypothesized_cause": self.hypothesized_cause.value,
            "baseline_episode_ids": list(self.baseline_episode_ids),
            "contrast_action": self.contrast_action.value,
            "expected_evidence_ids": list(self.expected_evidence_ids),
            "reason_codes": list(self.reason_codes),
            "shadow_only": True,
            "affects_production_acceptance": False,
        }
        payload["proposal_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "proposal_id"}
        )
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | ControlledAblationProposal
    ) -> ControlledAblationProposal:
        if isinstance(payload, ControlledAblationProposal):
            return payload
        if not isinstance(payload, Mapping):
            raise CausalAttributionError("ablation proposal must be an object")
        _reject_forbidden_payload(payload, "ablation")
        item = cls(
            factor=payload.get("factor"),  # type: ignore[arg-type]
            hypothesized_cause=payload.get("hypothesized_cause"),  # type: ignore[arg-type]
            baseline_episode_ids=payload.get("baseline_episode_ids") or (),
            contrast_action=payload.get("contrast_action"),  # type: ignore[arg-type]
            expected_evidence_ids=payload.get("expected_evidence_ids") or (),
            reason_codes=payload.get("reason_codes") or (),
            shadow_only=payload.get("shadow_only", True),
            affects_production_acceptance=payload.get(
                "affects_production_acceptance", False
            ),
        )
        claimed = payload.get("proposal_id")
        if claimed not in (None, "", item.proposal_id):
            raise CausalAttributionError("ablation proposal_id does not match canonical identity")
        return item


@dataclass(frozen=True)
class CompressionCredit:
    """Whether compression may be credited as causally supported savings."""

    credited: bool
    reason_codes: tuple[str, ...]
    paired_episode_ids: tuple[str, ...] = ()
    proposed_ablations: tuple[ControlledAblationProposal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "credited", _bool(self.credited, "credited"))
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True),
        )
        object.__setattr__(
            self,
            "paired_episode_ids",
            _identifiers(self.paired_episode_ids, "paired_episode_ids"),
        )
        proposals = tuple(
            ControlledAblationProposal.from_dict(item) for item in self.proposed_ablations
        )
        if len(proposals) > MAX_ABLATION_PROPOSALS:
            raise CausalAttributionError("compression credit proposes too many ablations")
        object.__setattr__(self, "proposed_ablations", proposals)
        if self.credited and "single_pass_insufficient" in self.reason_codes:
            raise CausalAttributionError("compression is not credited from one pass")
        if any(not item.shadow_only or item.affects_production_acceptance for item in proposals):
            raise CausalAttributionError(
                "controlled ablations cannot affect production acceptance"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPRESSION_CREDIT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "credited": self.credited,
            "reason_codes": list(self.reason_codes),
            "paired_episode_ids": list(self.paired_episode_ids),
            "proposed_ablations": [item.to_dict() for item in self.proposed_ablations],
            "affects_production_acceptance": False,
        }


@dataclass(frozen=True)
class CausalAttributionResult:
    """Typed attribution outcome.  Absence of a cause is an ordinary result."""

    disposition: AttributionDisposition
    reason_codes: tuple[str, ...]
    episode_ids: tuple[str, ...]
    attribution: CausalAttribution | None = None
    proposed_ablations: tuple[ControlledAblationProposal, ...] = ()
    confounder_ids: tuple[str, ...] = ()
    competing_causes: tuple[AttributionCause, ...] = ()
    affects_production_acceptance: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, AttributionDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True),
        )
        object.__setattr__(
            self, "episode_ids", _identifiers(self.episode_ids, "episode_ids", required=True)
        )
        if self.attribution is not None and not isinstance(self.attribution, CausalAttribution):
            raise CausalAttributionError("attribution must be a CausalAttribution")
        proposals = tuple(
            ControlledAblationProposal.from_dict(item) for item in self.proposed_ablations
        )
        if len(proposals) > MAX_ABLATION_PROPOSALS:
            raise CausalAttributionError("attribution result proposes too many ablations")
        object.__setattr__(self, "proposed_ablations", proposals)
        object.__setattr__(
            self, "confounder_ids", _identifiers(self.confounder_ids, "confounder_ids")
        )
        object.__setattr__(
            self,
            "competing_causes",
            tuple(
                _enum(item, AttributionCause, "competing_causes")
                for item in self.competing_causes
            ),
        )
        object.__setattr__(
            self,
            "affects_production_acceptance",
            _bool(self.affects_production_acceptance, "affects_production_acceptance"),
        )
        if self.affects_production_acceptance:
            raise CausalAttributionError(
                "controlled ablations cannot affect production acceptance"
            )
        if self.disposition is AttributionDisposition.ATTRIBUTED:
            if self.attribution is None:
                raise CausalAttributionError("attributed results require a CausalAttribution")
        elif self.attribution is not None:
            raise CausalAttributionError("no cause is assigned without discriminating evidence")
        if self.attribution is not None:
            if self.attribution.controlled_ablation_ids and not self.attribution.shadow_only:
                raise CausalAttributionError(
                    "ablation-backed attributions must remain shadow-only"
                )
        if any(not item.shadow_only or item.affects_production_acceptance for item in proposals):
            raise CausalAttributionError(
                "controlled ablations cannot affect production acceptance"
            )

    @property
    def result_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("result_id", None)
        return content_identity(payload)

    @property
    def primary_cause(self) -> AttributionCause | None:
        if self.attribution is None:
            return None
        return self.attribution.primary_cause

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CAUSAL_ATTRIBUTION_RESULT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "episode_ids": list(self.episode_ids),
            "attribution": None if self.attribution is None else self.attribution.to_dict(),
            "proposed_ablations": [item.to_dict() for item in self.proposed_ablations],
            "confounder_ids": list(self.confounder_ids),
            "competing_causes": [item.value for item in self.competing_causes],
            "affects_production_acceptance": False,
        }
        payload["result_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "result_id"}
        )
        return payload


@dataclass(frozen=True)
class _CauseSupport:
    cause: AttributionCause
    level: SupportLevel
    evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    ablation_ids: tuple[str, ...]
    counterexample_ids: tuple[str, ...] = ()

    @property
    def discriminating(self) -> bool:
        return self.level is SupportLevel.DISCRIMINATING


def _failed_episodes(episodes: Sequence[ExperienceEpisode]) -> tuple[ExperienceEpisode, ...]:
    failed = tuple(item for item in episodes if item.terminal_status in _FAILURE_STATUSES)
    return failed or tuple(episodes)


def _observations_for(
    observations: Sequence[AttributionObservation],
    kind: AttributionEvidenceKind,
) -> tuple[AttributionObservation, ...]:
    return tuple(item for item in observations if item.kind is kind)


def _paired_for(
    observations: Sequence[AttributionObservation],
    factor: AblationFactor,
) -> tuple[AttributionObservation, ...]:
    return tuple(
        item
        for item in observations
        if item.kind is AttributionEvidenceKind.PAIRED_COMPARISON and item.factor is factor
    )


def _ids_from(observations: Sequence[AttributionObservation]) -> tuple[str, ...]:
    collected: list[str] = []
    seen: set[str] = set()
    for item in observations:
        for evidence_id in item.evidence_ids:
            if evidence_id not in seen:
                seen.add(evidence_id)
                collected.append(evidence_id)
    return tuple(sorted(collected))


def _observation_ids(observations: Sequence[AttributionObservation]) -> tuple[str, ...]:
    return tuple(sorted({item.observation_id for item in observations}))


def _merge_ids(*groups: Sequence[str]) -> tuple[str, ...]:
    collected: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                collected.append(item)
    return tuple(sorted(collected))


def _support(
    cause: AttributionCause,
    level: SupportLevel,
    observations: Sequence[AttributionObservation],
    *reason_codes: str,
    extra_evidence: Sequence[str] = (),
    extra_ablations: Sequence[str] = (),
    extra_counterexamples: Sequence[str] = (),
) -> _CauseSupport:
    ablation_obs = tuple(item for item in observations if item.is_ablation)
    return _CauseSupport(
        cause=cause,
        level=level,
        evidence_ids=_merge_ids(_ids_from(observations), extra_evidence),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        ablation_ids=_merge_ids(_observation_ids(ablation_obs), extra_ablations),
        counterexample_ids=_merge_ids(extra_counterexamples),
    )


def _none(cause: AttributionCause) -> _CauseSupport:
    return _CauseSupport(
        cause=cause,
        level=SupportLevel.NONE,
        evidence_ids=(),
        reason_codes=(),
        ablation_ids=(),
    )


def _omission_named(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> bool:
    if _observations_for(observations, AttributionEvidenceKind.COMPLETENESS_WITNESS):
        return True
    return any(_metric_int(item.context_metrics, "omitted_reference_count") > 0 for item in episodes)


def _has_model_episode(episodes: Sequence[ExperienceEpisode]) -> bool:
    return any(item.selected_action in _MODEL_ACTIONS for item in episodes)


def _evaluate_context_omission(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    witnesses = _observations_for(observations, AttributionEvidenceKind.COMPLETENESS_WITNESS)
    paired = _paired_for(observations, AblationFactor.CONTEXT_COMPLETENESS)
    compression_paired = _paired_for(observations, AblationFactor.COMPRESSION)
    isolating = tuple(
        item
        for item in paired
        if item.outcome_flipped
        and item.baseline_terminal_status in _FAILURE_STATUSES
        and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    compression_omission = tuple(
        item
        for item in compression_paired
        if item.outcome_flipped
        and item.baseline_terminal_status in _FAILURE_STATUSES
        and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    if witnesses:
        return _support(
            AttributionCause.CONTEXT_OMISSION,
            SupportLevel.DISCRIMINATING,
            witnesses,
            "context_omission_witness",
        )
    if isolating:
        return _support(
            AttributionCause.CONTEXT_OMISSION,
            SupportLevel.DISCRIMINATING,
            isolating,
            "paired_context_restoration",
        )
    if compression_omission:
        return _support(
            AttributionCause.CONTEXT_OMISSION,
            SupportLevel.DISCRIMINATING,
            compression_omission,
            "compressed_failure_expanded_success",
        )
    if _omission_named(episodes, observations):
        hints = tuple(
            item
            for item in episodes
            if _metric_int(item.context_metrics, "omitted_reference_count") > 0
        )
        extra = tuple(item.episode_id for item in hints)
        return _CauseSupport(
            cause=AttributionCause.CONTEXT_OMISSION,
            level=SupportLevel.SUPPORTING,
            evidence_ids=extra,
            reason_codes=("omitted_reference_metric",),
            ablation_ids=(),
        )
    return _none(AttributionCause.CONTEXT_OMISSION)


def _evaluate_model_failure(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    if _omission_named(episodes, observations):
        return _none(AttributionCause.MODEL_CAPABILITY_FAILURE)
    if _observations_for(observations, AttributionEvidenceKind.COMPLETENESS_WITNESS):
        return _none(AttributionCause.MODEL_CAPABILITY_FAILURE)
    sufficient = _observations_for(observations, AttributionEvidenceKind.CONTEXT_SUFFICIENT)
    provider_fail = _observations_for(
        observations, AttributionEvidenceKind.PROVIDER_UNAVAILABLE
    ) + _observations_for(observations, AttributionEvidenceKind.PROVIDER_ERROR)
    if provider_fail:
        return _none(AttributionCause.MODEL_CAPABILITY_FAILURE)
    model_episodes = tuple(item for item in episodes if item.selected_action in _MODEL_ACTIONS)
    failed_models = tuple(
        item for item in model_episodes if item.terminal_status is TerminalStatus.FAILED
    )
    paired = _paired_for(observations, AblationFactor.MODEL)
    isolating = tuple(
        item
        for item in paired
        if item.outcome_flipped
        and item.baseline_terminal_status in _FAILURE_STATUSES
        and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    complete_still_failed = tuple(
        item
        for item in _paired_for(observations, AblationFactor.CONTEXT_COMPLETENESS)
        if item.both_failed
    )
    if failed_models and sufficient and (complete_still_failed or isolating or sufficient):
        used = sufficient + isolating + complete_still_failed
        extra = tuple(
            evidence_id for item in failed_models for evidence_id in item.evidence_ids
        )
        return _support(
            AttributionCause.MODEL_CAPABILITY_FAILURE,
            SupportLevel.DISCRIMINATING,
            used,
            "complete_context_model_failure",
            extra_evidence=extra,
        )
    if failed_models:
        extra = tuple(item.episode_id for item in failed_models)
        return _CauseSupport(
            cause=AttributionCause.MODEL_CAPABILITY_FAILURE,
            level=SupportLevel.SUPPORTING,
            evidence_ids=extra,
            reason_codes=("model_action_correlation",),
            ablation_ids=(),
        )
    return _none(AttributionCause.MODEL_CAPABILITY_FAILURE)


def _evaluate_provider(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    unavailable = _observations_for(observations, AttributionEvidenceKind.PROVIDER_UNAVAILABLE)
    errors = _observations_for(observations, AttributionEvidenceKind.PROVIDER_ERROR)
    env_fail = _observations_for(observations, AttributionEvidenceKind.ENVIRONMENT_PROBE_FAILURE)
    env_ok = _observations_for(observations, AttributionEvidenceKind.ENVIRONMENT_PROBE_SUCCESS)
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.PROVIDER)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    hits = unavailable + errors
    provider_episodes = tuple(
        item
        for item in episodes
        if item.provider_id and item.terminal_status is TerminalStatus.UNAVAILABLE
    )
    if hits and env_fail and not paired and not env_ok:
        return _support(
            AttributionCause.PROVIDER_FAILURE,
            SupportLevel.SUPPORTING,
            hits + env_fail,
            "provider_environment_confounded",
        )
    if hits and (env_ok or not env_fail or paired):
        return _support(
            AttributionCause.PROVIDER_FAILURE,
            SupportLevel.DISCRIMINATING,
            hits + env_ok + paired,
            "provider_unavailable_isolated",
        )
    if paired:
        return _support(
            AttributionCause.PROVIDER_FAILURE,
            SupportLevel.DISCRIMINATING,
            paired,
            "provider_unavailable_isolated",
        )
    if provider_episodes:
        return _CauseSupport(
            cause=AttributionCause.PROVIDER_FAILURE,
            level=SupportLevel.SUPPORTING,
            evidence_ids=tuple(item.episode_id for item in provider_episodes),
            reason_codes=("unavailable_provider_correlation",),
            ablation_ids=(),
        )
    return _none(AttributionCause.PROVIDER_FAILURE)


def _evaluate_environment(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    env_fail = _observations_for(observations, AttributionEvidenceKind.ENVIRONMENT_PROBE_FAILURE)
    env_ok = _observations_for(observations, AttributionEvidenceKind.ENVIRONMENT_PROBE_SUCCESS)
    provider_hits = _observations_for(
        observations, AttributionEvidenceKind.PROVIDER_UNAVAILABLE
    ) + _observations_for(observations, AttributionEvidenceKind.PROVIDER_ERROR)
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.ENVIRONMENT)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    if env_ok and not env_fail:
        return _none(AttributionCause.ENVIRONMENT_FAILURE)
    if env_fail and provider_hits and not paired:
        return _support(
            AttributionCause.ENVIRONMENT_FAILURE,
            SupportLevel.SUPPORTING,
            env_fail + provider_hits,
            "provider_environment_confounded",
        )
    if env_fail and (not provider_hits or paired):
        return _support(
            AttributionCause.ENVIRONMENT_FAILURE,
            SupportLevel.DISCRIMINATING,
            env_fail + paired,
            "environment_probe_isolated",
        )
    if paired:
        return _support(
            AttributionCause.ENVIRONMENT_FAILURE,
            SupportLevel.DISCRIMINATING,
            paired,
            "environment_probe_isolated",
        )
    return _none(AttributionCause.ENVIRONMENT_FAILURE)


def _evaluate_stale(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    stale = _observations_for(observations, AttributionEvidenceKind.STALE_IDENTITY)
    cache_mismatch = _observations_for(
        observations, AttributionEvidenceKind.CACHE_BINDING_MISMATCH
    )
    cache_used = any(item.selected_action in _CACHE_ACTIONS for item in episodes)
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.EVIDENCE_FRESHNESS)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    if stale and cache_used and cache_mismatch and not paired:
        return _support(
            AttributionCause.STALE_EVIDENCE,
            SupportLevel.SUPPORTING,
            stale + cache_mismatch,
            "stale_cache_confounded",
        )
    if stale and (not cache_used or not cache_mismatch or paired):
        return _support(
            AttributionCause.STALE_EVIDENCE,
            SupportLevel.DISCRIMINATING,
            stale + paired,
            "stale_identity_isolated",
        )
    if paired:
        return _support(
            AttributionCause.STALE_EVIDENCE,
            SupportLevel.DISCRIMINATING,
            paired,
            "stale_identity_isolated",
        )
    return _none(AttributionCause.STALE_EVIDENCE)


def _evaluate_cache(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    mismatch = _observations_for(observations, AttributionEvidenceKind.CACHE_BINDING_MISMATCH)
    stale = _observations_for(observations, AttributionEvidenceKind.STALE_IDENTITY)
    cache_episodes = tuple(item for item in episodes if item.selected_action in _CACHE_ACTIONS)
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.CACHE_REUSE)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    if mismatch and cache_episodes and stale and not paired:
        return _support(
            AttributionCause.INCORRECT_CACHE_REUSE,
            SupportLevel.SUPPORTING,
            mismatch + stale,
            "stale_cache_confounded",
        )
    if mismatch and cache_episodes and (not stale or paired):
        return _support(
            AttributionCause.INCORRECT_CACHE_REUSE,
            SupportLevel.DISCRIMINATING,
            mismatch + paired,
            "incorrect_cache_reuse_isolated",
            extra_evidence=tuple(
                evidence_id for item in cache_episodes for evidence_id in item.evidence_ids
            ),
        )
    if paired and cache_episodes:
        return _support(
            AttributionCause.INCORRECT_CACHE_REUSE,
            SupportLevel.DISCRIMINATING,
            paired,
            "incorrect_cache_reuse_isolated",
        )
    if cache_episodes and any(item.terminal_status in _FAILURE_STATUSES for item in cache_episodes):
        return _CauseSupport(
            cause=AttributionCause.INCORRECT_CACHE_REUSE,
            level=SupportLevel.SUPPORTING,
            evidence_ids=tuple(item.episode_id for item in cache_episodes),
            reason_codes=("cache_action_correlation",),
            ablation_ids=(),
        )
    return _none(AttributionCause.INCORRECT_CACHE_REUSE)


def _evaluate_decomposition(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    hits = _observations_for(observations, AttributionEvidenceKind.DECOMPOSITION_FAILURE)
    branch = _observations_for(observations, AttributionEvidenceKind.PLAN_BRANCH_FAILURE)
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.TASK_DECOMPOSITION)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    if hits and branch and not paired:
        return _support(
            AttributionCause.BAD_TASK_DECOMPOSITION,
            SupportLevel.SUPPORTING,
            hits + branch,
            "plan_decomposition_confounded",
        )
    if hits and (not branch or paired):
        return _support(
            AttributionCause.BAD_TASK_DECOMPOSITION,
            SupportLevel.DISCRIMINATING,
            hits + paired,
            "decomposition_isolated",
        )
    if paired:
        return _support(
            AttributionCause.BAD_TASK_DECOMPOSITION,
            SupportLevel.DISCRIMINATING,
            paired,
            "decomposition_isolated",
        )
    if any(item.selected_action in _DECOMPOSITION_ACTIONS for item in episodes):
        return _CauseSupport(
            cause=AttributionCause.BAD_TASK_DECOMPOSITION,
            level=SupportLevel.SUPPORTING,
            evidence_ids=tuple(
                item.episode_id
                for item in episodes
                if item.selected_action in _DECOMPOSITION_ACTIONS
            ),
            reason_codes=("decomposition_action_correlation",),
            ablation_ids=(),
        )
    return _none(AttributionCause.BAD_TASK_DECOMPOSITION)


def _evaluate_plan_branch(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    hits = _observations_for(observations, AttributionEvidenceKind.PLAN_BRANCH_FAILURE)
    decomposition = _observations_for(
        observations, AttributionEvidenceKind.DECOMPOSITION_FAILURE
    )
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.PLAN_BRANCH)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    if hits and decomposition and not paired:
        return _support(
            AttributionCause.BAD_PLAN_BRANCH,
            SupportLevel.SUPPORTING,
            hits + decomposition,
            "plan_decomposition_confounded",
        )
    if hits and (not decomposition or paired):
        return _support(
            AttributionCause.BAD_PLAN_BRANCH,
            SupportLevel.DISCRIMINATING,
            hits + paired,
            "plan_branch_isolated",
        )
    if paired:
        return _support(
            AttributionCause.BAD_PLAN_BRANCH,
            SupportLevel.DISCRIMINATING,
            paired,
            "plan_branch_isolated",
        )
    if any(item.selected_action in _PLAN_ACTIONS for item in episodes):
        return _CauseSupport(
            cause=AttributionCause.BAD_PLAN_BRANCH,
            level=SupportLevel.SUPPORTING,
            evidence_ids=tuple(
                item.episode_id for item in episodes if item.selected_action in _PLAN_ACTIONS
            ),
            reason_codes=("plan_action_correlation",),
            ablation_ids=(),
        )
    return _none(AttributionCause.BAD_PLAN_BRANCH)


def _evaluate_validation(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    hits = _observations_for(
        observations, AttributionEvidenceKind.VALIDATION_SELECTOR_MISMATCH
    )
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.VALIDATION_SELECTION)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    validation_episodes = tuple(
        item for item in episodes if item.selected_action in _VALIDATION_ACTIONS
    )
    if hits:
        return _support(
            AttributionCause.VALIDATION_SELECTION_FAILURE,
            SupportLevel.DISCRIMINATING,
            hits + paired,
            "validation_selector_mismatch",
            extra_evidence=tuple(
                receipt_id
                for item in validation_episodes
                for receipt_id in item.validation_receipt_ids
            ),
        )
    if paired:
        return _support(
            AttributionCause.VALIDATION_SELECTION_FAILURE,
            SupportLevel.DISCRIMINATING,
            paired,
            "validation_selector_mismatch",
        )
    return _none(AttributionCause.VALIDATION_SELECTION_FAILURE)


def _evaluate_proof(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    hits = _observations_for(observations, AttributionEvidenceKind.PROOF_SELECTOR_MISMATCH)
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.PROOF_SELECTION)
        if item.outcome_flipped and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
    )
    proof_episodes = tuple(item for item in episodes if item.selected_action in _PROOF_ACTIONS)
    if hits:
        return _support(
            AttributionCause.PROOF_SELECTION_FAILURE,
            SupportLevel.DISCRIMINATING,
            hits + paired,
            "proof_selector_mismatch",
            extra_evidence=tuple(
                receipt_id for item in proof_episodes for receipt_id in item.proof_receipt_ids
            ),
        )
    if paired:
        return _support(
            AttributionCause.PROOF_SELECTION_FAILURE,
            SupportLevel.DISCRIMINATING,
            paired,
            "proof_selector_mismatch",
        )
    return _none(AttributionCause.PROOF_SELECTION_FAILURE)


def _evaluate_merge(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    hits = _observations_for(observations, AttributionEvidenceKind.MERGE_CONFLICT)
    paired = tuple(
        item
        for item in _paired_for(observations, AblationFactor.MERGE_QUEUE)
        if item.outcome_flipped
    )
    if hits:
        extra = tuple(
            receipt_id for item in episodes for receipt_id in item.merge_receipt_ids
        )
        return _support(
            AttributionCause.MERGE_CONFLICT,
            SupportLevel.DISCRIMINATING,
            hits + paired,
            "merge_conflict_receipt",
            extra_evidence=extra,
        )
    if paired and any(item.merge_receipt_ids for item in episodes):
        return _support(
            AttributionCause.MERGE_CONFLICT,
            SupportLevel.DISCRIMINATING,
            paired,
            "merge_conflict_receipt",
        )
    return _none(AttributionCause.MERGE_CONFLICT)


def _evaluate_human_policy(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> _CauseSupport:
    hits = _observations_for(observations, AttributionEvidenceKind.HUMAN_POLICY_BLOCK)
    if hits:
        extra = tuple(
            intervention_id
            for item in episodes
            for intervention_id in item.human_intervention_ids
        )
        return _support(
            AttributionCause.HUMAN_POLICY_BLOCKER,
            SupportLevel.DISCRIMINATING,
            hits,
            "human_policy_block",
            extra_evidence=extra,
        )
    blocked = tuple(
        item
        for item in episodes
        if item.terminal_status is TerminalStatus.BLOCKED and item.human_intervention_ids
    )
    if blocked:
        return _CauseSupport(
            cause=AttributionCause.HUMAN_POLICY_BLOCKER,
            level=SupportLevel.SUPPORTING,
            evidence_ids=tuple(item.episode_id for item in blocked),
            reason_codes=("human_intervention_correlation",),
            ablation_ids=(),
        )
    return _none(AttributionCause.HUMAN_POLICY_BLOCKER)


_CAUSE_EVALUATORS = (
    _evaluate_human_policy,
    _evaluate_environment,
    _evaluate_provider,
    _evaluate_context_omission,
    _evaluate_stale,
    _evaluate_cache,
    _evaluate_decomposition,
    _evaluate_plan_branch,
    _evaluate_validation,
    _evaluate_proof,
    _evaluate_merge,
    _evaluate_model_failure,
)


def _collect_support(
    episodes: Sequence[ExperienceEpisode],
    observations: Sequence[AttributionObservation],
) -> dict[AttributionCause, _CauseSupport]:
    collected: dict[AttributionCause, _CauseSupport] = {}
    for evaluator in _CAUSE_EVALUATORS:
        support = evaluator(episodes, observations)
        if support.level is not SupportLevel.NONE:
            collected[support.cause] = support
    # Model capability cannot survive omitted source even if a caller claimed it.
    if AttributionCause.CONTEXT_OMISSION in collected:
        collected.pop(AttributionCause.MODEL_CAPABILITY_FAILURE, None)
    return collected


def _explicit_confounders(
    observations: Sequence[AttributionObservation],
    support: Mapping[AttributionCause, _CauseSupport],
) -> tuple[str, ...]:
    named: list[str] = []
    seen: set[str] = set()
    for item in observations:
        for raw in item.confounder_ids:
            try:
                cause = AttributionCause(raw)
            except ValueError:
                identifier = raw
                if identifier not in seen:
                    seen.add(identifier)
                    named.append(identifier)
                continue
            other = support.get(cause)
            if other is not None and other.level is not SupportLevel.NONE:
                if cause.value not in seen:
                    seen.add(cause.value)
                    named.append(cause.value)
    return tuple(sorted(named))


def _confounded_causes(
    support: Mapping[AttributionCause, _CauseSupport],
) -> tuple[AttributionCause, ...]:
    present = [cause for cause, item in support.items() if item.level is not SupportLevel.NONE]
    confounded: list[AttributionCause] = []
    seen: set[AttributionCause] = set()
    for left in present:
        for right in present:
            if left is right or not confounder_pair(left, right):
                continue
            left_item = support[left]
            right_item = support[right]
            if left_item.discriminating and not right_item.discriminating:
                continue
            if right_item.discriminating and not left_item.discriminating:
                continue
            if left_item.discriminating and right_item.discriminating:
                if left not in seen:
                    seen.add(left)
                    confounded.append(left)
                if right not in seen:
                    seen.add(right)
                    confounded.append(right)
                continue
            if left not in seen:
                seen.add(left)
                confounded.append(left)
            if right not in seen:
                seen.add(right)
                confounded.append(right)
    return tuple(sorted(confounded, key=lambda item: _CAUSE_PRECEDENCE.index(item)))


def _propose(
    cause: AttributionCause,
    episodes: Sequence[ExperienceEpisode],
    *reason_codes: str,
    extra_evidence: Sequence[str] = (),
) -> ControlledAblationProposal:
    baseline = _failed_episodes(episodes)
    return ControlledAblationProposal(
        factor=_CAUSE_FACTOR[cause],
        hypothesized_cause=cause,
        baseline_episode_ids=tuple(item.episode_id for item in baseline),
        contrast_action=_CAUSE_CONTRAST_ACTION[cause],
        expected_evidence_ids=tuple(extra_evidence),
        reason_codes=reason_codes or ("ablation_required",),
    )


def _proposals_from_support(
    episodes: Sequence[ExperienceEpisode],
    support: Mapping[AttributionCause, _CauseSupport],
    *,
    include_model_guard: bool = True,
) -> tuple[ControlledAblationProposal, ...]:
    proposals: list[ControlledAblationProposal] = []
    seen: set[str] = set()

    def add(proposal: ControlledAblationProposal) -> None:
        if proposal.proposal_id in seen:
            return
        if len(proposals) >= MAX_ABLATION_PROPOSALS:
            return
        seen.add(proposal.proposal_id)
        proposals.append(proposal)

    for cause in _CAUSE_PRECEDENCE:
        item = support.get(cause)
        if item is None or item.discriminating:
            continue
        if item.level is SupportLevel.SUPPORTING:
            add(
                _propose(
                    cause,
                    episodes,
                    "ablation_required",
                    *item.reason_codes,
                    extra_evidence=item.evidence_ids[:8],
                )
            )
    if include_model_guard and _has_model_episode(episodes):
        if AttributionCause.CONTEXT_OMISSION not in support or not support[
            AttributionCause.CONTEXT_OMISSION
        ].discriminating:
            if AttributionCause.MODEL_CAPABILITY_FAILURE not in support or not support[
                AttributionCause.MODEL_CAPABILITY_FAILURE
            ].discriminating:
                add(
                    _propose(
                        AttributionCause.CONTEXT_OMISSION,
                        episodes,
                        "model_not_blamed_for_omitted_source",
                        "ablation_required",
                    )
                )
    compressed = any(
        _metric_int(item.context_metrics, "compressed") > 0
        or _metric_int(item.context_metrics, "prefix_reused_tokens") > 0
        for item in episodes
    )
    if compressed:
        add(
            ControlledAblationProposal(
                factor=AblationFactor.COMPRESSION,
                hypothesized_cause=AttributionCause.CONTEXT_OMISSION,
                baseline_episode_ids=tuple(item.episode_id for item in episodes),
                contrast_action=MetaAction.EXPAND_CONTEXT_REFERENCE,
                expected_evidence_ids=(),
                reason_codes=("single_pass_insufficient", "compression_not_credited"),
            )
        )
    return tuple(proposals)


def _confidence(support: _CauseSupport, independent: bool) -> int:
    if independent:
        return CONFIDENCE_INDEPENDENT_BP
    if support.ablation_ids:
        return CONFIDENCE_PAIRED_BP
    return CONFIDENCE_WITNESS_BP


def _build_attribution(
    episodes: Sequence[ExperienceEpisode],
    primary: _CauseSupport,
    secondary: Sequence[_CauseSupport],
) -> CausalAttribution:
    episode_ids = tuple(item.episode_id for item in _failed_episodes(episodes))
    evidence_ids = primary.evidence_ids
    if not evidence_ids:
        raise CausalAttributionError("no cause is assigned without discriminating evidence")
    independent = bool(secondary) and all(item.discriminating for item in secondary)
    counterexample_ids = _merge_ids(
        primary.counterexample_ids,
        *(item.counterexample_ids for item in episodes),
    )
    return CausalAttribution(
        episode_ids=episode_ids,
        primary_cause=primary.cause,
        evidence_ids=evidence_ids,
        confidence_bp=_confidence(primary, independent),
        secondary_causes=tuple(item.cause for item in secondary),
        controlled_ablation_ids=primary.ablation_ids,
        counterexample_ids=counterexample_ids,
        shadow_only=True,
    )


class CausalAttributionEngine:
    """Classify evidence-supported causes and emit shadow-only ablation proposals."""

    INTERFACE = CAUSAL_ATTRIBUTION_ENGINE_INTERFACE
    SCHEMA = CAUSAL_ATTRIBUTION_ENGINE_SCHEMA
    ATTRIBUTION_INTERFACE = CAUSAL_ATTRIBUTION_INTERFACE

    def attribute(
        self,
        episodes: Sequence[ExperienceEpisode | Mapping[str, Any]] | ExperienceEpisode,
        observations: Sequence[AttributionObservation | Mapping[str, Any]] = (),
        *,
        claimed_cause: AttributionCause | str | None = None,
    ) -> CausalAttributionResult:
        """Return a cause only when discriminating evidence isolates it."""

        bound_episodes = _coerce_episodes(episodes)
        bound_observations = _coerce_observations(observations)
        claimed = (
            None
            if claimed_cause is None
            else _enum(claimed_cause, AttributionCause, "claimed_cause")
        )
        support = _collect_support(bound_episodes, bound_observations)
        episode_ids = tuple(item.episode_id for item in _failed_episodes(bound_episodes))
        proposals = _proposals_from_support(bound_episodes, support)
        explicit = _explicit_confounders(bound_observations, support)
        confounded = _confounded_causes(support)
        competing = tuple(
            cause
            for cause in _CAUSE_PRECEDENCE
            if cause in support and support[cause].level is not SupportLevel.NONE
        )

        if claimed is AttributionCause.MODEL_CAPABILITY_FAILURE and (
            AttributionCause.CONTEXT_OMISSION in support
            or _omission_named(bound_episodes, bound_observations)
        ):
            claimed = AttributionCause.CONTEXT_OMISSION

        discriminating = tuple(
            support[cause]
            for cause in _CAUSE_PRECEDENCE
            if cause in support and support[cause].discriminating and cause not in confounded
        )

        if explicit:
            supported_explicit = tuple(
                item
                for item in explicit
                if item in {cause.value for cause in competing}
            )
            if supported_explicit and not (
                len(discriminating) == 1
                and discriminating[0].cause.value not in supported_explicit
                and not any(
                    confounder_pair(discriminating[0].cause, AttributionCause(item))
                    for item in supported_explicit
                    if item in {cause.value for cause in AttributionCause}
                )
            ):
                return CausalAttributionResult(
                    disposition=AttributionDisposition.CONFOUNDER_PRESENT,
                    reason_codes=("confounder_unresolved", "insufficient_discriminating_evidence"),
                    episode_ids=episode_ids,
                    proposed_ablations=proposals,
                    confounder_ids=explicit,
                    competing_causes=competing,
                )

        if confounded and not discriminating:
            return CausalAttributionResult(
                disposition=AttributionDisposition.CONFOUNDER_PRESENT,
                reason_codes=("confounder_unresolved", "insufficient_discriminating_evidence"),
                episode_ids=episode_ids,
                proposed_ablations=proposals,
                confounder_ids=tuple(item.value for item in confounded),
                competing_causes=competing,
            )

        if discriminating:
            primary = discriminating[0]
            if claimed is not None:
                for item in discriminating:
                    if item.cause is claimed:
                        primary = item
                        break
            secondary = tuple(item for item in discriminating if item.cause is not primary.cause)
            attribution = _build_attribution(bound_episodes, primary, secondary)
            reason_codes = list(primary.reason_codes)
            if (
                primary.cause is AttributionCause.CONTEXT_OMISSION
                or _omission_named(bound_episodes, bound_observations)
            ):
                reason_codes.append("model_not_blamed_for_omitted_source")
            reason_codes.append("shadow_only_enforced")
            reason_codes.append("production_acceptance_unaffected")
            return CausalAttributionResult(
                disposition=AttributionDisposition.ATTRIBUTED,
                reason_codes=tuple(reason_codes),
                episode_ids=episode_ids,
                attribution=attribution,
                proposed_ablations=(),
                confounder_ids=explicit,
                competing_causes=tuple(item.cause for item in secondary),
            )

        supporting = tuple(
            support[cause]
            for cause in _CAUSE_PRECEDENCE
            if cause in support and support[cause].level is SupportLevel.SUPPORTING
        )
        if all(item.terminal_status is TerminalStatus.SUCCEEDED for item in bound_episodes):
            return CausalAttributionResult(
                disposition=AttributionDisposition.INSUFFICIENT_EVIDENCE,
                reason_codes=("insufficient_discriminating_evidence", "no_failure_to_attribute"),
                episode_ids=episode_ids,
                proposed_ablations=(),
                confounder_ids=explicit,
                competing_causes=(),
            )
        if supporting or proposals:
            return CausalAttributionResult(
                disposition=AttributionDisposition.ABLATION_REQUIRED,
                reason_codes=("ablation_required", "insufficient_discriminating_evidence"),
                episode_ids=episode_ids,
                proposed_ablations=proposals,
                confounder_ids=explicit or tuple(item.value for item in confounded),
                competing_causes=competing,
            )
        return CausalAttributionResult(
            disposition=AttributionDisposition.INSUFFICIENT_EVIDENCE,
            reason_codes=("insufficient_discriminating_evidence",),
            episode_ids=episode_ids,
            proposed_ablations=(),
            confounder_ids=explicit,
            competing_causes=competing,
        )

    def propose_ablations(
        self,
        episodes: Sequence[ExperienceEpisode | Mapping[str, Any]] | ExperienceEpisode,
        observations: Sequence[AttributionObservation | Mapping[str, Any]] = (),
    ) -> tuple[ControlledAblationProposal, ...]:
        """Return bounded shadow-only comparisons that would isolate remaining causes."""

        result = self.attribute(episodes, observations)
        if result.disposition is AttributionDisposition.ATTRIBUTED:
            return ()
        return result.proposed_ablations

    def compression_credit(
        self,
        episodes: Sequence[ExperienceEpisode | Mapping[str, Any]] | ExperienceEpisode,
        observations: Sequence[AttributionObservation | Mapping[str, Any]] = (),
    ) -> CompressionCredit:
        """Credit compression only from a paired equivalent comparison, never one pass."""

        bound_episodes = _coerce_episodes(episodes)
        bound_observations = _coerce_observations(observations)
        paired = _paired_for(bound_observations, AblationFactor.COMPRESSION)
        equivalent = tuple(
            item
            for item in paired
            if item.both_succeeded and item.shadow_only and not item.production_acceptance
        )
        both_failed = tuple(item for item in paired if item.both_failed)
        omission = tuple(
            item
            for item in paired
            if item.outcome_flipped
            and item.baseline_terminal_status in _FAILURE_STATUSES
            and item.contrast_terminal_status is TerminalStatus.SUCCEEDED
        )
        episode_ids = tuple(item.episode_id for item in bound_episodes)
        uncompressed = ControlledAblationProposal(
            factor=AblationFactor.COMPRESSION,
            hypothesized_cause=AttributionCause.CONTEXT_OMISSION,
            baseline_episode_ids=episode_ids,
            contrast_action=MetaAction.EXPAND_CONTEXT_REFERENCE,
            expected_evidence_ids=(),
            reason_codes=("single_pass_insufficient", "compression_not_credited"),
        )
        if equivalent:
            paired_ids = _merge_ids(
                *(item.episode_ids + item.contrast_episode_ids for item in equivalent)
            )
            return CompressionCredit(
                credited=True,
                reason_codes=("paired_compression_equivalent", "production_acceptance_unaffected"),
                paired_episode_ids=paired_ids,
            )
        if both_failed:
            return CompressionCredit(
                credited=False,
                reason_codes=(
                    "compression_not_credited",
                    "both_failed_cannot_blame_compression",
                    "production_acceptance_unaffected",
                ),
                paired_episode_ids=_merge_ids(
                    *(item.episode_ids + item.contrast_episode_ids for item in both_failed)
                ),
            )
        if omission:
            return CompressionCredit(
                credited=False,
                reason_codes=(
                    "compression_not_credited",
                    "compressed_failure_expanded_success",
                    "production_acceptance_unaffected",
                ),
                paired_episode_ids=_merge_ids(
                    *(item.episode_ids + item.contrast_episode_ids for item in omission)
                ),
            )
        return CompressionCredit(
            credited=False,
            reason_codes=("single_pass_insufficient", "compression_not_credited"),
            paired_episode_ids=episode_ids if len(bound_episodes) == 1 else (),
            proposed_ablations=(uncompressed,),
        )


__all__ = [
    "AblationFactor",
    "AttributionDisposition",
    "AttributionEvidenceKind",
    "AttributionObservation",
    "CAUSAL_ATTRIBUTION_ENGINE_INTERFACE",
    "CAUSAL_ATTRIBUTION_INTERFACE",
    "CausalAttributionEngine",
    "CausalAttributionError",
    "CausalAttributionResult",
    "CompressionCredit",
    "CONFIDENCE_INDEPENDENT_BP",
    "CONFIDENCE_PAIRED_BP",
    "CONFIDENCE_WITNESS_BP",
    "CONTROLLED_ABLATION_PROPOSAL_SCHEMA",
    "ControlledAblationProposal",
    "SupportLevel",
    "ablation_may_affect_production_acceptance",
    "confounder_pair",
    "field_is_forbidden",
]
