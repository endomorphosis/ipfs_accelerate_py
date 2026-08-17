"""Bounded campaign refill scoring, curriculum priority, and loop limits.

Refill is a control surface, not an unbounded work generator.  Every trigger
has a hard count, every pass has a task ceiling, and a repeated no-progress
identity is refused instead of being replayed.  Scoring is a deterministic
total order so two supervisors observing the same residual set emit the same
admission decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from ..runtime.learning_checkpoint import CAMPAIGN_DURABILITY_REQUIREMENT_ID


CAMPAIGN_REFILL_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-refill-policy@1"
)
CAMPAIGN_REFILL_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-refill-decision@1"
)
CAMPAIGN_REFILL_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-refill-candidate@1"
)

# Hard safety ceilings.  Callers may tighten these but must not raise them.
MAX_REFILL_ROUNDS: Final = 8
MAX_NO_PROGRESS_ROUNDS: Final = 3
MAX_CURRICULUM_REPETITIONS: Final = 2
MAX_TASKS_PER_REFILL: Final = 8
MAX_OPEN_WORK: Final = 24
MAX_TRIGGER_FIRINGS: Final = 8


class CampaignRefillError(ValueError):
    """Malformed refill policy, candidate, or history."""


class RefillTrigger(str, Enum):
    NO_PROGRESS = "no_progress"
    CURRICULUM_GAP = "curriculum_gap"
    PROOF_RESIDUAL = "proof_residual"
    EVALUATION_RESIDUAL = "evaluation_residual"
    VALIDATION_FAILURE = "validation_failure"
    TOKEN_BUDGET = "token_budget"
    ROUND_BOUND = "round_bound"


# Lower rank is admitted first.  Proof and evaluation residuals outrank
# speculative curriculum expansion; no-progress is last and still bounded.
TRIGGER_PRIORITY: Final[Mapping[RefillTrigger, int]] = {
    RefillTrigger.PROOF_RESIDUAL: 0,
    RefillTrigger.EVALUATION_RESIDUAL: 1,
    RefillTrigger.CURRICULUM_GAP: 2,
    RefillTrigger.VALIDATION_FAILURE: 3,
    RefillTrigger.TOKEN_BUDGET: 4,
    RefillTrigger.NO_PROGRESS: 5,
    RefillTrigger.ROUND_BOUND: 6,
}

TRIGGER_BOUNDS: Final[Mapping[RefillTrigger, int]] = {
    RefillTrigger.NO_PROGRESS: MAX_NO_PROGRESS_ROUNDS,
    RefillTrigger.CURRICULUM_GAP: MAX_TRIGGER_FIRINGS,
    RefillTrigger.PROOF_RESIDUAL: MAX_TRIGGER_FIRINGS,
    RefillTrigger.EVALUATION_RESIDUAL: MAX_TRIGGER_FIRINGS,
    RefillTrigger.VALIDATION_FAILURE: 4,
    RefillTrigger.TOKEN_BUDGET: 2,
    RefillTrigger.ROUND_BOUND: 1,
}


class RefillDisposition(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    NO_PROGRESS_BOUNDED = "no_progress_bounded"
    TRIGGER_BOUNDED = "trigger_bounded"
    OPEN_WORK_BOUNDED = "open_work_bounded"
    ROUND_BOUNDED = "round_bounded"
    REPETITION_BOUNDED = "repetition_bounded"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not str(value).strip():
        raise CampaignRefillError(f"{name} must be a non-empty string")
    text = str(value).strip()
    if "\x00" in text:
        raise CampaignRefillError(f"{name} must not contain NUL")
    return text


def _required_int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignRefillError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise CampaignRefillError(f"{name} is outside its bound")
    return value


@dataclass(frozen=True)
class CampaignRefillPolicy:
    """Closed numeric bounds for one campaign refill controller."""

    max_refill_rounds: int = MAX_REFILL_ROUNDS
    max_no_progress_rounds: int = MAX_NO_PROGRESS_ROUNDS
    max_curriculum_repetitions: int = MAX_CURRICULUM_REPETITIONS
    max_tasks_per_refill: int = MAX_TASKS_PER_REFILL
    max_open_work: int = MAX_OPEN_WORK
    cooldown_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_refill_rounds",
            _required_int(
                self.max_refill_rounds,
                "max_refill_rounds",
                minimum=1,
                maximum=MAX_REFILL_ROUNDS,
            ),
        )
        object.__setattr__(
            self,
            "max_no_progress_rounds",
            _required_int(
                self.max_no_progress_rounds,
                "max_no_progress_rounds",
                minimum=1,
                maximum=MAX_NO_PROGRESS_ROUNDS,
            ),
        )
        object.__setattr__(
            self,
            "max_curriculum_repetitions",
            _required_int(
                self.max_curriculum_repetitions,
                "max_curriculum_repetitions",
                minimum=1,
                maximum=MAX_CURRICULUM_REPETITIONS,
            ),
        )
        object.__setattr__(
            self,
            "max_tasks_per_refill",
            _required_int(
                self.max_tasks_per_refill,
                "max_tasks_per_refill",
                minimum=1,
                maximum=MAX_TASKS_PER_REFILL,
            ),
        )
        object.__setattr__(
            self,
            "max_open_work",
            _required_int(
                self.max_open_work, "max_open_work", minimum=1, maximum=MAX_OPEN_WORK
            ),
        )
        object.__setattr__(
            self,
            "cooldown_ms",
            _required_int(self.cooldown_ms, "cooldown_ms", minimum=0),
        )

    @property
    def policy_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_REFILL_POLICY_SCHEMA,
            "max_refill_rounds": self.max_refill_rounds,
            "max_no_progress_rounds": self.max_no_progress_rounds,
            "max_curriculum_repetitions": self.max_curriculum_repetitions,
            "max_tasks_per_refill": self.max_tasks_per_refill,
            "max_open_work": self.max_open_work,
            "cooldown_ms": self.cooldown_ms,
        }

    def trigger_bound(self, trigger: RefillTrigger) -> int:
        if trigger is RefillTrigger.NO_PROGRESS:
            return self.max_no_progress_rounds
        return TRIGGER_BOUNDS[trigger]


@dataclass(frozen=True)
class CampaignRefillCandidate:
    """One residual item considered for a refill pass."""

    candidate_id: str
    trigger: RefillTrigger
    residual_count: int = 1
    curriculum_key: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _required_text(self.candidate_id, "candidate_id")
        )
        selected = (
            self.trigger if isinstance(self.trigger, RefillTrigger) else RefillTrigger(str(self.trigger))
        )
        object.__setattr__(self, "trigger", selected)
        object.__setattr__(
            self,
            "residual_count",
            _required_int(self.residual_count, "residual_count", minimum=1),
        )
        object.__setattr__(self, "curriculum_key", str(self.curriculum_key or "").strip())
        if not isinstance(self.metadata, Mapping):
            raise CampaignRefillError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def progress_identity(self) -> str:
        return self.curriculum_key or self.candidate_id

    def score(self) -> tuple[int, int, str]:
        """Deterministic total order: priority, residual, then identity."""

        return (
            TRIGGER_PRIORITY[self.trigger],
            -int(self.residual_count),
            self.candidate_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_REFILL_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "trigger": self.trigger.value,
            "residual_count": self.residual_count,
            "curriculum_key": self.curriculum_key,
            "progress_identity": self.progress_identity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CampaignRefillHistory:
    """Observed refill firings and progress identities for one campaign."""

    refill_rounds: int = 0
    open_work: int = 0
    trigger_counts: Mapping[str, int] = field(default_factory=dict)
    progress_identities: tuple[str, ...] = ()
    last_progress_identity: str = ""
    no_progress_streak: int = 0
    curriculum_repetitions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "refill_rounds", _required_int(self.refill_rounds, "refill_rounds")
        )
        object.__setattr__(self, "open_work", _required_int(self.open_work, "open_work"))
        counts = {
            _required_text(key, "trigger"): _required_int(value, "trigger count")
            for key, value in dict(self.trigger_counts).items()
        }
        object.__setattr__(self, "trigger_counts", counts)
        identities = tuple(
            _required_text(item, "progress identity") for item in self.progress_identities
        )
        object.__setattr__(self, "progress_identities", identities)
        object.__setattr__(
            self, "last_progress_identity", str(self.last_progress_identity or "").strip()
        )
        object.__setattr__(
            self,
            "no_progress_streak",
            _required_int(self.no_progress_streak, "no_progress_streak"),
        )
        repetitions = {
            _required_text(key, "curriculum key"): _required_int(value, "curriculum repetition")
            for key, value in dict(self.curriculum_repetitions).items()
        }
        object.__setattr__(self, "curriculum_repetitions", repetitions)

    def count_for(self, trigger: RefillTrigger) -> int:
        return int(self.trigger_counts.get(trigger.value, 0))


@dataclass(frozen=True)
class CampaignRefillDecision:
    """Exact, bounded outcome of one refill admission."""

    disposition: RefillDisposition
    policy_id: str
    admitted: tuple[CampaignRefillCandidate, ...] = ()
    rejected: tuple[CampaignRefillCandidate, ...] = ()
    reason_code: str = ""
    trigger_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selected = (
            self.disposition
            if isinstance(self.disposition, RefillDisposition)
            else RefillDisposition(str(self.disposition))
        )
        object.__setattr__(self, "disposition", selected)
        object.__setattr__(self, "policy_id", _required_text(self.policy_id, "policy_id"))
        object.__setattr__(self, "admitted", tuple(self.admitted))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "reason_code", str(self.reason_code or selected.value))
        object.__setattr__(
            self,
            "trigger_counts",
            {
                _required_text(key, "trigger"): _required_int(value, "trigger count")
                for key, value in dict(self.trigger_counts).items()
            },
        )

    @property
    def changed(self) -> bool:
        return bool(self.admitted) and self.disposition is RefillDisposition.ADMITTED

    @property
    def bounded(self) -> bool:
        return self.disposition is not RefillDisposition.ADMITTED or not self.admitted

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CAMPAIGN_REFILL_DECISION_SCHEMA,
            "requirement_id": CAMPAIGN_DURABILITY_REQUIREMENT_ID,
            "disposition": self.disposition.value,
            "policy_id": self.policy_id,
            "reason_code": self.reason_code,
            "admitted": [item.to_dict() for item in self.admitted],
            "rejected": [item.to_dict() for item in self.rejected],
            "trigger_counts": dict(self.trigger_counts),
            "changed": self.changed,
        }
        payload["decision_id"] = content_identity(payload)
        return payload


class CampaignRefillController:
    """Score and admit a finite residual set under the campaign policy."""

    def __init__(self, policy: CampaignRefillPolicy | None = None) -> None:
        self.policy = policy or CampaignRefillPolicy()

    def rank(
        self, candidates: Iterable[CampaignRefillCandidate | Mapping[str, Any]]
    ) -> tuple[CampaignRefillCandidate, ...]:
        normalized = tuple(
            item if isinstance(item, CampaignRefillCandidate) else CampaignRefillCandidate(
                candidate_id=str(item.get("candidate_id") or ""),
                trigger=item.get("trigger") or RefillTrigger.CURRICULUM_GAP,  # type: ignore[arg-type]
                residual_count=int(item.get("residual_count") or 1),
                curriculum_key=str(item.get("curriculum_key") or ""),
                metadata=item.get("metadata") or {},
            )
            for item in candidates
        )
        return tuple(sorted(normalized, key=lambda item: item.score()))

    def decide(
        self,
        candidates: Sequence[CampaignRefillCandidate | Mapping[str, Any]],
        *,
        history: CampaignRefillHistory | None = None,
        cursor_advanced: bool = True,
        progress_identity: str = "",
    ) -> CampaignRefillDecision:
        policy = self.policy
        observed = history or CampaignRefillHistory()
        ranked = self.rank(candidates)
        rejected: list[CampaignRefillCandidate] = []
        if observed.refill_rounds >= policy.max_refill_rounds:
            return CampaignRefillDecision(
                disposition=RefillDisposition.ROUND_BOUNDED,
                policy_id=policy.policy_id,
                rejected=ranked,
                reason_code="max_refill_rounds",
                trigger_counts=observed.trigger_counts,
            )
        if observed.open_work >= policy.max_open_work:
            return CampaignRefillDecision(
                disposition=RefillDisposition.OPEN_WORK_BOUNDED,
                policy_id=policy.policy_id,
                rejected=ranked,
                reason_code="max_open_work",
                trigger_counts=observed.trigger_counts,
            )
        identity = str(progress_identity or "").strip()
        no_progress_streak = observed.no_progress_streak
        if identity and identity == observed.last_progress_identity and not cursor_advanced:
            no_progress_streak += 1
        elif not cursor_advanced:
            no_progress_streak += 1
        else:
            no_progress_streak = 0
        if no_progress_streak >= policy.max_no_progress_rounds:
            return CampaignRefillDecision(
                disposition=RefillDisposition.NO_PROGRESS_BOUNDED,
                policy_id=policy.policy_id,
                rejected=ranked,
                reason_code="no_progress_bound",
                trigger_counts=observed.trigger_counts,
            )

        admitted: list[CampaignRefillCandidate] = []
        local_counts = dict(observed.trigger_counts)
        local_repetitions = dict(observed.curriculum_repetitions)
        for candidate in ranked:
            if len(admitted) >= policy.max_tasks_per_refill:
                rejected.append(candidate)
                continue
            if observed.open_work + len(admitted) >= policy.max_open_work:
                rejected.append(candidate)
                continue
            trigger_count = int(local_counts.get(candidate.trigger.value, 0))
            if trigger_count >= policy.trigger_bound(candidate.trigger):
                rejected.append(candidate)
                continue
            curriculum_key = candidate.curriculum_key
            if curriculum_key:
                seen = int(local_repetitions.get(curriculum_key, 0))
                if seen >= policy.max_curriculum_repetitions:
                    rejected.append(candidate)
                    continue
                local_repetitions[curriculum_key] = seen + 1
            local_counts[candidate.trigger.value] = trigger_count + 1
            admitted.append(candidate)

        if not admitted:
            if rejected and any(
                int(local_counts.get(item.trigger.value, 0))
                >= policy.trigger_bound(item.trigger)
                or (
                    item.curriculum_key
                    and int(local_repetitions.get(item.curriculum_key, 0))
                    >= policy.max_curriculum_repetitions
                )
                for item in rejected
            ):
                if any(
                    item.curriculum_key
                    and int(observed.curriculum_repetitions.get(item.curriculum_key, 0))
                    >= policy.max_curriculum_repetitions
                    for item in rejected
                ):
                    disposition = RefillDisposition.REPETITION_BOUNDED
                    reason = "curriculum_repetition_bound"
                else:
                    disposition = RefillDisposition.TRIGGER_BOUNDED
                    reason = "trigger_bound"
            else:
                disposition = RefillDisposition.REJECTED
                reason = "no_admissible_candidates"
            return CampaignRefillDecision(
                disposition=disposition,
                policy_id=policy.policy_id,
                rejected=tuple(rejected),
                reason_code=reason,
                trigger_counts=local_counts,
            )
        return CampaignRefillDecision(
            disposition=RefillDisposition.ADMITTED,
            policy_id=policy.policy_id,
            admitted=tuple(admitted),
            rejected=tuple(rejected),
            reason_code="admitted",
            trigger_counts=local_counts,
        )


def all_refill_triggers_are_bounded(policy: CampaignRefillPolicy | None = None) -> bool:
    """Return True when every closed trigger has a positive finite bound."""

    selected = policy or CampaignRefillPolicy()
    return all(selected.trigger_bound(trigger) >= 1 for trigger in RefillTrigger)


__all__ = (
    "CAMPAIGN_REFILL_DECISION_SCHEMA",
    "CAMPAIGN_REFILL_POLICY_SCHEMA",
    "MAX_CURRICULUM_REPETITIONS",
    "MAX_NO_PROGRESS_ROUNDS",
    "MAX_OPEN_WORK",
    "MAX_REFILL_ROUNDS",
    "MAX_TASKS_PER_REFILL",
    "TRIGGER_BOUNDS",
    "TRIGGER_PRIORITY",
    "CampaignRefillCandidate",
    "CampaignRefillController",
    "CampaignRefillDecision",
    "CampaignRefillError",
    "CampaignRefillHistory",
    "CampaignRefillPolicy",
    "RefillDisposition",
    "RefillTrigger",
    "all_refill_triggers_are_bounded",
)
