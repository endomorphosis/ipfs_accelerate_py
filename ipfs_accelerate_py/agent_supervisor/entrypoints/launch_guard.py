"""Launch guard: revalidate complete LaunchPlan at every effect boundary.

No effectful facade may bypass this guard. Stale or incomplete inputs fail
before effects; exact replay adopts or returns the prior result; crashes at
every intent/effect/receipt boundary have one deterministic continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, MutableMapping, Optional


class LaunchGuardError(Exception):
    """Raised when a launch cannot proceed past a guarded boundary."""

    def __init__(self, code: str, message: str, *, boundary: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.boundary = boundary


class BoundaryKind(str, Enum):
    INTENT = "intent"
    EFFECT = "effect"
    RECEIPT = "receipt"


class ContinuationAction(str, Enum):
    """Deterministic continuation after a crash or incomplete boundary."""

    FAIL_INCOMPLETE = "fail_incomplete"
    FAIL_STALE = "fail_stale"
    ADOPT_PRIOR = "adopt_prior"
    RETURN_PRIOR = "return_prior"
    PROCEED = "proceed"
    REPLAY_RECEIPT = "replay_receipt"


# Required LaunchPlan fields that must be present and non-empty for effect boundaries.
REQUIRED_LAUNCH_PLAN_FIELDS: tuple[str, ...] = (
    "plan_id",
    "intent_id",
    "created_at",
    "config_fingerprint",
    "effect_sequence",
    "payload",
)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and not isinstance(value, (str, bytes))


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _stable_fingerprint(plan: Mapping[str, Any]) -> str:
    """Compute a stable fingerprint for exact-replay comparison.

    Uses config_fingerprint when present and valid; otherwise derives from
    plan_id + intent_id + effect_sequence length for deterministic identity.
    """
    fp = plan.get("config_fingerprint")
    if _nonempty_str(fp):
        return str(fp)
    plan_id = plan.get("plan_id", "")
    intent_id = plan.get("intent_id", "")
    seq = plan.get("effect_sequence") or []
    seq_len = len(seq) if isinstance(seq, (list, tuple)) else 0
    return f"{plan_id}:{intent_id}:{seq_len}"


def validate_launch_plan_complete(
    plan: Any,
    *,
    require_revalidated: bool = True,
    boundary: str | BoundaryKind | None = None,
) -> Mapping[str, Any]:
    """Validate that *plan* is a complete, revalidated LaunchPlan.

    Fails before any effect when inputs are stale or incomplete.

    Raises:
        LaunchGuardError: if plan is missing, incomplete, or not revalidated.
    """
    boundary_s = boundary.value if isinstance(boundary, BoundaryKind) else boundary

    if plan is None:
        raise LaunchGuardError(
            "missing_launch_plan",
            "LaunchPlan is required at effect boundary; got None",
            boundary=boundary_s,
        )

    if not _is_mapping(plan):
        raise LaunchGuardError(
            "invalid_launch_plan_type",
            f"LaunchPlan must be a mapping; got {type(plan).__name__}",
            boundary=boundary_s,
        )

    missing: list[str] = []
    for key in REQUIRED_LAUNCH_PLAN_FIELDS:
        if key not in plan:
            missing.append(key)
            continue
        val = plan[key]
        if key == "payload":
            if val is None:
                missing.append(key)
        elif key == "effect_sequence":
            if not isinstance(val, (list, tuple)):
                missing.append(key)
        elif not _nonempty_str(val) and val is not None and not isinstance(val, (int, float, bool)):
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(key)
        elif val is None or (isinstance(val, str) and not str(val).strip()):
            missing.append(key)

    # Stricter string fields
    for key in ("plan_id", "intent_id", "created_at", "config_fingerprint"):
        if key in plan and not _nonempty_str(plan[key]):
            if key not in missing:
                missing.append(key)

    if missing:
        raise LaunchGuardError(
            "incomplete_launch_plan",
            f"LaunchPlan incomplete at boundary; missing or empty: {sorted(set(missing))}",
            boundary=boundary_s,
        )

    if require_revalidated:
        revalidated = plan.get("revalidated")
        revalidated_at = plan.get("revalidated_at")
        # Accept either explicit revalidated flag or revalidated_at timestamp
        ok_flag = revalidated is True or revalidated == 1 or revalidated == "true"
        ok_ts = _nonempty_str(revalidated_at) if revalidated_at is not None else False
        if not ok_flag and not ok_ts:
            raise LaunchGuardError(
                "not_revalidated",
                "LaunchPlan must be revalidated before effect boundary "
                "(set revalidated=True or revalidated_at)",
                boundary=boundary_s,
            )

    stale = plan.get("stale")
    if stale is True or stale == 1 or stale == "true":
        raise LaunchGuardError(
            "stale_launch_plan",
            "LaunchPlan is marked stale; refuse effect",
            boundary=boundary_s,
        )

    # Optional explicit validity window
    if plan.get("invalid") is True:
        raise LaunchGuardError(
            "invalid_launch_plan",
            "LaunchPlan marked invalid; refuse effect",
            boundary=boundary_s,
        )

    return plan


@dataclass
class BoundaryReceipt:
    """Immutable receipt recorded at an intent/effect/receipt boundary."""

    boundary: str
    plan_id: str
    intent_id: str
    fingerprint: str
    status: str
    result: Any = None
    error_code: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "plan_id": self.plan_id,
            "intent_id": self.intent_id,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "result": self.result,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass
class LaunchGuard:
    """Guards effectful facades so no path bypasses LaunchPlan revalidation.

    Exact replay of a prior successful boundary adopts or returns the prior
    result. Crashes at intent/effect/receipt boundaries yield one deterministic
    continuation via :meth:`continuation_for_crash`.
    """

    _receipts: MutableMapping[str, BoundaryReceipt] = field(default_factory=dict)
    _prior_results: MutableMapping[str, Any] = field(default_factory=dict)

    def _receipt_key(self, boundary: str, plan: Mapping[str, Any]) -> str:
        fp = _stable_fingerprint(plan)
        plan_id = str(plan.get("plan_id", ""))
        return f"{boundary}:{plan_id}:{fp}"

    def validate_at_boundary(
        self,
        plan: Any,
        boundary: str | BoundaryKind,
        *,
        require_revalidated: bool = True,
    ) -> Mapping[str, Any]:
        """Validate complete revalidated LaunchPlan at a named boundary."""
        kind = boundary.value if isinstance(boundary, BoundaryKind) else str(boundary)
        return validate_launch_plan_complete(
            plan,
            require_revalidated=require_revalidated,
            boundary=kind,
        )

    def check_replay(
        self,
        plan: Mapping[str, Any],
        boundary: str | BoundaryKind,
    ) -> tuple[bool, Any | None]:
        """If exact prior receipt exists for this plan+boundary, return it.

        Returns:
            (True, prior_result) when exact replay should adopt/return prior.
            (False, None) when no prior receipt; caller may proceed.
        """
        kind = boundary.value if isinstance(boundary, BoundaryKind) else str(boundary)
        key = self._receipt_key(kind, plan)
        receipt = self._receipts.get(key)
        if receipt is None:
            return False, None
        if receipt.status == "success":
            return True, receipt.result
        if receipt.status == "failed":
            # Deterministic: re-raise same failure for exact replay of failed boundary
            raise LaunchGuardError(
                receipt.error_code or "prior_failure",
                receipt.error_message or "Prior boundary failed",
                boundary=kind,
            )
        return False, None

    def record_receipt(
        self,
        plan: Mapping[str, Any],
        boundary: str | BoundaryKind,
        *,
        status: str,
        result: Any = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> BoundaryReceipt:
        kind = boundary.value if isinstance(boundary, BoundaryKind) else str(boundary)
        receipt = BoundaryReceipt(
            boundary=kind,
            plan_id=str(plan.get("plan_id", "")),
            intent_id=str(plan.get("intent_id", "")),
            fingerprint=_stable_fingerprint(plan),
            status=status,
            result=result,
            error_code=error_code,
            error_message=error_message,
        )
        key = self._receipt_key(kind, plan)
        self._receipts[key] = receipt
        if status == "success":
            self._prior_results[key] = result
        return receipt

    def run_guarded(
        self,
        plan: Any,
        boundary: str | BoundaryKind,
        effect: Callable[[Mapping[str, Any]], Any],
        *,
        require_revalidated: bool = True,
        adopt_prior_on_replay: bool = True,
    ) -> Any:
        """Run *effect* only after complete LaunchPlan revalidation.

        Exact replay adopts or returns the prior result without re-executing
        the effect when a successful receipt exists.
        """
        kind = boundary.value if isinstance(boundary, BoundaryKind) else str(boundary)
        validated = self.validate_at_boundary(
            plan,
            kind,
            require_revalidated=require_revalidated,
        )

        if adopt_prior_on_replay:
            is_replay, prior = self.check_replay(validated, kind)
            if is_replay:
                return prior

        try:
            result = effect(validated)
        except LaunchGuardError:
            raise
        except Exception as exc:  # noqa: BLE001 — record then re-raise
            self.record_receipt(
                validated,
                kind,
                status="failed",
                error_code="effect_exception",
                error_message=str(exc),
            )
            raise

        self.record_receipt(validated, kind, status="success", result=result)
        return result

    def continuation_for_crash(
        self,
        plan: Any,
        boundary: str | BoundaryKind,
        *,
        crash_phase: str = "during_effect",
    ) -> ContinuationAction:
        """Single deterministic continuation after a crash at a boundary.

        Rules (ordered):
        1. Incomplete or missing plan -> FAIL_INCOMPLETE (no effect).
        2. Stale / not revalidated -> FAIL_STALE (no effect).
        3. Exact successful prior receipt -> ADOPT_PRIOR or RETURN_PRIOR.
        4. Crash during receipt recording after success -> REPLAY_RECEIPT.
        5. Otherwise -> PROCEED only if plan fully validates; else FAIL_INCOMPLETE.
        """
        kind = boundary.value if isinstance(boundary, BoundaryKind) else str(boundary)

        try:
            validated = validate_launch_plan_complete(
                plan,
                require_revalidated=True,
                boundary=kind,
            )
        except LaunchGuardError as exc:
            if exc.code in ("stale_launch_plan", "not_revalidated", "invalid_launch_plan"):
                return ContinuationAction.FAIL_STALE
            return ContinuationAction.FAIL_INCOMPLETE

        key = self._receipt_key(kind, validated)
        receipt = self._receipts.get(key)

        if receipt is not None and receipt.status == "success":
            if crash_phase in ("after_effect", "during_receipt"):
                return ContinuationAction.REPLAY_RECEIPT
            return ContinuationAction.RETURN_PRIOR

        if receipt is not None and receipt.status == "failed":
            return ContinuationAction.FAIL_STALE

        if crash_phase == "before_effect":
            return ContinuationAction.PROCEED

        if crash_phase == "during_effect":
            # No receipt: effect may or may not have applied; deterministic
            # policy is to not re-enter blindly — require revalidation path.
            return ContinuationAction.PROCEED

        if crash_phase in ("after_effect", "during_receipt"):
            return ContinuationAction.REPLAY_RECEIPT

        return ContinuationAction.PROCEED

    def assert_no_bypass(
        self,
        plan: Any,
        boundary: str | BoundaryKind = BoundaryKind.EFFECT,
    ) -> Mapping[str, Any]:
        """Public facade entry: every effectful path must call this first."""
        return self.validate_at_boundary(plan, boundary, require_revalidated=True)


# Module-level default guard for shared facades.
_default_guard: LaunchGuard | None = None


def get_launch_guard() -> LaunchGuard:
    global _default_guard
    if _default_guard is None:
        _default_guard = LaunchGuard()
    return _default_guard


def reset_launch_guard() -> LaunchGuard:
    """Replace the process-wide guard (tests / clean runtime composition)."""
    global _default_guard
    _default_guard = LaunchGuard()
    return _default_guard


def require_launch_plan(
    plan: Any,
    *,
    boundary: str | BoundaryKind = BoundaryKind.EFFECT,
    guard: LaunchGuard | None = None,
) -> Mapping[str, Any]:
    """Require a complete revalidated LaunchPlan; used by effectful facades."""
    g = guard if guard is not None else get_launch_guard()
    return g.assert_no_bypass(plan, boundary)
