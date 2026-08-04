"""Launch guard for agent supervisor effect boundaries.

Every effectful facade must pass through this guard. Stale or incomplete
inputs fail before effects; exact replay adopts or returns the prior
result; crashes at intent/effect/receipt boundaries have one deterministic
continuation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable


class LaunchGuardError(Exception):
    """Base error for launch guard failures."""


class IncompleteLaunchPlanError(LaunchGuardError):
    """Raised when a LaunchPlan is missing required fields."""


class StaleLaunchPlanError(LaunchGuardError):
    """Raised when a LaunchPlan fails revalidation (stale inputs)."""


class BoundaryCrashError(LaunchGuardError):
    """Raised when a crash is detected at an intent/effect/receipt boundary."""


class BoundaryPhase(str, Enum):
    """Deterministic phases at every intent/effect/receipt boundary."""

    INTENT = "intent"
    EFFECT = "effect"
    RECEIPT = "receipt"


class BoundaryContinuation(str, Enum):
    """One deterministic continuation after a boundary crash."""

    RETRY_INTENT = "retry_intent"
    RETRY_EFFECT = "retry_effect"
    RETRY_RECEIPT = "retry_receipt"
    ADOPT_PRIOR = "adopt_prior"
    FAIL_CLOSED = "fail_closed"


# Required fields for a complete LaunchPlan.
REQUIRED_LAUNCH_PLAN_FIELDS = frozenset(
    {
        "plan_id",
        "intent_hash",
        "effect_spec",
        "created_at",
        "schema_version",
    }
)

# Optional but recommended fields used for revalidation / replay.
OPTIONAL_LAUNCH_PLAN_FIELDS = frozenset(
    {
        "content_hash",
        "prior_receipt",
        "metadata",
        "ttl_seconds",
        "source",
    }
)


def _canonical_json(value: Any) -> str:
    """Stable JSON encoding for hashing and equality."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def compute_content_hash(plan: Mapping[str, Any]) -> str:
    """Compute a deterministic content hash for a LaunchPlan body.

    Excludes volatile fields (prior_receipt) so replay identity is stable.
    """
    body = {
        k: v
        for k, v in plan.items()
        if k not in ("prior_receipt", "content_hash")
    }
    digest = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def validate_launch_plan_complete(plan: Any) -> Dict[str, Any]:
    """Validate that *plan* is a complete LaunchPlan mapping.

    Raises:
        IncompleteLaunchPlanError: missing required fields or wrong type.
    """
    if not isinstance(plan, Mapping):
        raise IncompleteLaunchPlanError(
            f"LaunchPlan must be a mapping, got {type(plan).__name__}"
        )
    missing = sorted(REQUIRED_LAUNCH_PLAN_FIELDS - set(plan.keys()))
    if missing:
        raise IncompleteLaunchPlanError(
            f"LaunchPlan missing required fields: {', '.join(missing)}"
        )
    for key in ("plan_id", "intent_hash", "schema_version"):
        val = plan.get(key)
        if not isinstance(val, str) or not val.strip():
            raise IncompleteLaunchPlanError(
                f"LaunchPlan field {key!r} must be a non-empty string"
            )
    if not isinstance(plan.get("effect_spec"), Mapping):
        raise IncompleteLaunchPlanError(
            "LaunchPlan field 'effect_spec' must be a mapping"
        )
    created_at = plan.get("created_at")
    if not isinstance(created_at, (int, float, str)):
        raise IncompleteLaunchPlanError(
            "LaunchPlan field 'created_at' must be a number or string timestamp"
        )
    return dict(plan)


def revalidate_launch_plan(
    plan: Mapping[str, Any],
    *,
    expected_schema_version: Optional[str] = None,
    now: Optional[float] = None,
    max_age_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Revalidate a complete LaunchPlan before any effect.

    Checks completeness, optional schema pin, content_hash consistency,
    and optional TTL staleness.

    Raises:
        IncompleteLaunchPlanError: incomplete plan.
        StaleLaunchPlanError: stale or inconsistent plan.
    """
    complete = validate_launch_plan_complete(plan)

    if expected_schema_version is not None:
        if complete["schema_version"] != expected_schema_version:
            raise StaleLaunchPlanError(
                f"LaunchPlan schema_version {complete['schema_version']!r} "
                f"!= expected {expected_schema_version!r}"
            )

    declared_hash = complete.get("content_hash")
    if declared_hash is not None:
        actual = compute_content_hash(complete)
        if declared_hash != actual:
            raise StaleLaunchPlanError(
                f"LaunchPlan content_hash mismatch: declared={declared_hash!r} "
                f"actual={actual!r}"
            )

    # TTL / age check when both now and either max_age or plan.ttl_seconds present.
    ttl = complete.get("ttl_seconds")
    age_limit = max_age_seconds if max_age_seconds is not None else ttl
    if age_limit is not None and now is not None:
        created = complete["created_at"]
        try:
            created_ts = float(created)
        except (TypeError, ValueError) as exc:
            raise StaleLaunchPlanError(
                f"cannot parse created_at for staleness check: {created!r}"
            ) from exc
        age = float(now) - created_ts
        if age > float(age_limit):
            raise StaleLaunchPlanError(
                f"LaunchPlan is stale: age={age}s > limit={age_limit}s"
            )
        if age < 0:
            raise StaleLaunchPlanError(
                f"LaunchPlan created_at is in the future: age={age}s"
            )

    return complete


@runtime_checkable
class LaunchReceiptStore(Protocol):
    """Storage for prior launch receipts used for exact replay."""

    def get_receipt(self, plan_id: str, intent_hash: str) -> Optional[Mapping[str, Any]]:
        """Return prior receipt for (plan_id, intent_hash) or None."""
        ...

    def put_receipt(
        self,
        plan_id: str,
        intent_hash: str,
        receipt: Mapping[str, Any],
    ) -> None:
        """Persist a receipt for exact replay."""
        ...


class InMemoryReceiptStore:
    """Simple in-memory LaunchReceiptStore for tests and local use."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _key(plan_id: str, intent_hash: str) -> str:
        return f"{plan_id}::{intent_hash}"

    def get_receipt(
        self, plan_id: str, intent_hash: str
    ) -> Optional[Mapping[str, Any]]:
        return self._store.get(self._key(plan_id, intent_hash))

    def put_receipt(
        self,
        plan_id: str,
        intent_hash: str,
        receipt: Mapping[str, Any],
    ) -> None:
        self._store[self._key(plan_id, intent_hash)] = dict(receipt)


@dataclass
class GuardDecision:
    """Outcome of a launch guard check before effects."""

    allowed: bool
    plan: Dict[str, Any]
    adopted_receipt: Optional[Dict[str, Any]] = None
    reason: str = ""
    phase: BoundaryPhase = BoundaryPhase.INTENT

    @property
    def is_replay(self) -> bool:
        return self.adopted_receipt is not None


@dataclass
class BoundaryState:
    """Tracks crash/continuation state across intent/effect/receipt."""

    plan_id: str
    intent_hash: str
    last_completed_phase: Optional[BoundaryPhase] = None
    crashed_at: Optional[BoundaryPhase] = None
    continuation: Optional[BoundaryContinuation] = None
    receipt: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def continuation_for_crash(phase: BoundaryPhase) -> BoundaryContinuation:
    """Map a crash phase to exactly one deterministic continuation."""
    if phase is BoundaryPhase.INTENT:
        return BoundaryContinuation.RETRY_INTENT
    if phase is BoundaryPhase.EFFECT:
        return BoundaryContinuation.RETRY_EFFECT
    if phase is BoundaryPhase.RECEIPT:
        return BoundaryContinuation.RETRY_RECEIPT
    return BoundaryContinuation.FAIL_CLOSED


class LaunchGuard:
    """Gate every effectful facade on a complete revalidated LaunchPlan.

    Contract:
    - No effectful facade can bypass the guard (use :meth:`run_effect`).
    - Stale or incomplete inputs fail before effects.
    - Exact replay adopts or returns the prior result.
    - Crashes at every intent/effect/receipt boundary have one
      deterministic continuation.
    """

    def __init__(
        self,
        *,
        receipt_store: Optional[LaunchReceiptStore] = None,
        expected_schema_version: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._receipt_store: LaunchReceiptStore = (
            receipt_store if receipt_store is not None else InMemoryReceiptStore()
        )
        self._expected_schema_version = expected_schema_version
        self._max_age_seconds = max_age_seconds
        self._clock = clock
        self._boundary_states: Dict[str, BoundaryState] = {}

    def _state_key(self, plan_id: str, intent_hash: str) -> str:
        return f"{plan_id}::{intent_hash}"

    def _now(self) -> Optional[float]:
        if self._clock is None:
            return None
        return float(self._clock())

    def check(self, plan: Any) -> GuardDecision:
        """Validate and revalidate *plan*; adopt prior receipt on exact replay.

        Does not execute effects. Raises on incomplete/stale inputs.
        """
        complete = revalidate_launch_plan(
            plan,
            expected_schema_version=self._expected_schema_version,
            now=self._now(),
            max_age_seconds=self._max_age_seconds,
        )
        plan_id = complete["plan_id"]
        intent_hash = complete["intent_hash"]

        prior = self._receipt_store.get_receipt(plan_id, intent_hash)
        if prior is not None:
            # Exact replay: adopt/return prior result; do not re-effect.
            return GuardDecision(
                allowed=True,
                plan=complete,
                adopted_receipt=dict(prior),
                reason="exact_replay_adopt_prior",
                phase=BoundaryPhase.RECEIPT,
            )

        # Also honor prior_receipt embedded on the plan (caller-supplied).
        embedded = complete.get("prior_receipt")
        if isinstance(embedded, Mapping) and embedded:
            return GuardDecision(
                allowed=True,
                plan=complete,
                adopted_receipt=dict(embedded),
                reason="exact_replay_embedded_prior",
                phase=BoundaryPhase.RECEIPT,
            )

        return GuardDecision(
            allowed=True,
            plan=complete,
            adopted_receipt=None,
            reason="validated_ready",
            phase=BoundaryPhase.INTENT,
        )

    def begin_boundary(self, plan: Mapping[str, Any]) -> BoundaryState:
        """Open intent phase for a validated plan (after :meth:`check`)."""
        complete = revalidate_launch_plan(
            plan,
            expected_schema_version=self._expected_schema_version,
            now=self._now(),
            max_age_seconds=self._max_age_seconds,
        )
        key = self._state_key(complete["plan_id"], complete["intent_hash"])
        existing = self._boundary_states.get(key)
        if existing is not None and existing.crashed_at is not None:
            # Deterministic continuation from prior crash.
            cont = continuation_for_crash(existing.crashed_at)
            existing.continuation = cont
            existing.crashed_at = None
            return existing

        state = BoundaryState(
            plan_id=complete["plan_id"],
            intent_hash=complete["intent_hash"],
            last_completed_phase=None,
        )
        self._boundary_states[key] = state
        return state

    def mark_phase_complete(
        self,
        state: BoundaryState,
        phase: BoundaryPhase,
        *,
        receipt: Optional[Mapping[str, Any]] = None,
    ) -> BoundaryState:
        """Record successful completion of a boundary phase."""
        state.last_completed_phase = phase
        state.crashed_at = None
        state.continuation = None
        if receipt is not None:
            state.receipt = dict(receipt)
        key = self._state_key(state.plan_id, state.intent_hash)
        self._boundary_states[key] = state
        if phase is BoundaryPhase.RECEIPT and state.receipt is not None:
            self._receipt_store.put_receipt(
                state.plan_id, state.intent_hash, state.receipt
            )
        return state

    def mark_crash(
        self,
        state: BoundaryState,
        phase: BoundaryPhase,
    ) -> BoundaryContinuation:
        """Record a crash at *phase* and return the sole continuation."""
        cont = continuation_for_crash(phase)
        state.crashed_at = phase
        state.continuation = cont
        key = self._state_key(state.plan_id, state.intent_hash)
        self._boundary_states[key] = state
        return cont

    def resolve_crash_continuation(
        self,
        plan_id: str,
        intent_hash: str,
    ) -> BoundaryContinuation:
        """Return the deterministic continuation for a crashed boundary."""
        key = self._state_key(plan_id, intent_hash)
        state = self._boundary_states.get(key)
        if state is None or state.crashed_at is None:
            # No crash recorded: fail closed rather than invent a path.
            return BoundaryContinuation.FAIL_CLOSED
        return continuation_for_crash(state.crashed_at)

    def run_effect(
        self,
        plan: Any,
        effect_fn: Callable[[Mapping[str, Any]], Any],
        *,
        receipt_builder: Optional[Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Only path for effectful work: check → intent → effect → receipt.

        - Incomplete/stale plans raise before *effect_fn* runs.
        - Exact replay returns/adopts prior receipt without calling *effect_fn*.
        - Crashes at each phase yield one deterministic continuation.

        Returns a result dict with keys: status, plan, receipt, replayed,
        continuation (if crashed).
        """
        decision = self.check(plan)
        if decision.is_replay:
            return {
                "status": "adopted_prior",
                "plan": decision.plan,
                "receipt": decision.adopted_receipt,
                "replayed": True,
                "continuation": None,
            }

        state = self.begin_boundary(decision.plan)

        # INTENT
        try:
            self.mark_phase_complete(state, BoundaryPhase.INTENT)
        except Exception:
            cont = self.mark_crash(state, BoundaryPhase.INTENT)
            return {
                "status": "crashed",
                "plan": decision.plan,
                "receipt": None,
                "replayed": False,
                "continuation": cont.value,
                "crashed_at": BoundaryPhase.INTENT.value,
            }

        # EFFECT
        try:
            effect_result = effect_fn(decision.plan)
            self.mark_phase_complete(state, BoundaryPhase.EFFECT)
        except Exception:
            cont = self.mark_crash(state, BoundaryPhase.EFFECT)
            return {
                "status": "crashed",
                "plan": decision.plan,
                "receipt": None,
                "replayed": False,
                "continuation": cont.value,
                "crashed_at": BoundaryPhase.EFFECT.value,
            }

        # RECEIPT
        try:
            if receipt_builder is not None:
                receipt = dict(receipt_builder(effect_result, decision.plan))
            else:
                receipt = {
                    "plan_id": decision.plan["plan_id"],
                    "intent_hash": decision.plan["intent_hash"],
                    "effect_result": effect_result,
                    "status": "ok",
                }
            self.mark_phase_complete(state, BoundaryPhase.RECEIPT, receipt=receipt)
        except Exception:
            cont = self.mark_crash(state, BoundaryPhase.RECEIPT)
            return {
                "status": "crashed",
                "plan": decision.plan,
                "receipt": None,
                "replayed": False,
                "continuation": cont.value,
                "crashed_at": BoundaryPhase.RECEIPT.value,
            }

        return {
            "status": "completed",
            "plan": decision.plan,
            "receipt": state.receipt,
            "replayed": False,
            "continuation": None,
            "effect_result": effect_result,
        }


def require_launch_plan(plan: Any, **revalidate_kwargs: Any) -> Dict[str, Any]:
    """Public helper: fail closed unless *plan* is complete and revalidated."""
    return revalidate_launch_plan(plan, **revalidate_kwargs)
