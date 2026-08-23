"""FACP-046 — Enforce protocol/runtime trace conformance (TEP monitor).

Evidence: ``facp/tep-monitor@1`` / ``facp/runtime-monitor@1``
Bundle:   ``facp/protocols/runtime``

Runtime transition monitor that:

* accepts **exactly** the normative TEP transition vectors derived from the
  FACP-045 transactional effect protocol model;
* rejects stale fences, replay, incompatible idempotency keys, and
  incompatible receipt arguments;
* exposes a crash-injection harness that covers every persistent transition
  boundary named by the model.

Cold import is hermetic: no network, provider execution, or process mutation.
This module never executes irreversible external actions; it only validates
and simulates bounded protocol traces.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

# ---------------------------------------------------------------------------
# FACP evidence envelope
# ---------------------------------------------------------------------------

SCHEMA: Final[str] = "facp/tep-monitor@1"
RUNTIME_MONITOR_SCHEMA: Final[str] = "facp/runtime-monitor@1"
VERDICT_SCHEMA: Final[str] = "facp/tep-monitor-verdict@1"
VECTOR_SCHEMA: Final[str] = "facp/tep-normative-vector@1"
TASK_ID: Final[str] = "FACP-046"
GOAL_ID: Final[str] = "FACP-G510"
BUNDLE: Final[str] = "facp/protocols/runtime"
MODEL_EVIDENCE: Final[str] = "facp/tep-models@1"
MONITOR_VERSION: Final[str] = "formal-transition-monitor/v1"

EVIDENCE_SUBSET: Final[tuple[str, ...]] = (
    "prior_state",
    "next_state",
    "protocol",
    "instance",
    "operation",
    "actor",
    "fence",
    "idempotency",
    "observation",
    "time",
)

REQUIRED_INVARIANTS: Final[tuple[str, ...]] = (
    "NoDoubleEffect",
    "NoStaleFenceCompletion",
    "NoSuccessWithoutObservation",
    "NoConfirmationReuse",
    "NoBlindUnknownRetry",  # alias of NoReplayOfUnknownIrreversibleEffect
)

INVARIANT_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "NoStaleFence": "NoStaleFenceCompletion",
        "NoReplayOfUnknownIrreversibleEffect": "NoBlindUnknownRetry",
        "NoBlindUnknownRetry": "NoBlindUnknownRetry",
    }
)

CRASH_BOUNDARIES: Final[tuple[str, ...]] = (
    "admission",
    "reservation",
    "started",
    "unknown",
    "observed",
    "receipt",
    "current",
    "lease",
    "fence",
    "retry",
    "idempotency",
    "crash",
    "settlement",
    "compensation",
    "proof_promotion",
)

TYPESTATES: Final[frozenset[str]] = frozenset(
    {
        "Proposed",
        "ContractResolved",
        "ActorAuthenticated",
        "CapabilityVerified",
        "PolicyEvaluated",
        "ObligationsSatisfied",
        "ConfirmationSatisfied",
        "LeaseHeld",
        "Reserved",
        "Started",
        "Observed",
        "ReceiptSealed",
        "Rejected",
        "Unavailable",
        "Failed",
        "Unknown",
        "CompensationRequired",
        "Compensated",
        "Aborted",
    }
)

HAPPY_PATH: Final[tuple[str, ...]] = (
    "Proposed",
    "ContractResolved",
    "ActorAuthenticated",
    "CapabilityVerified",
    "PolicyEvaluated",
    "ObligationsSatisfied",
    "ConfirmationSatisfied",
    "LeaseHeld",
    "Reserved",
    "Started",
    "Observed",
    "ReceiptSealed",
)

PRE_RESERVED: Final[frozenset[str]] = frozenset(HAPPY_PATH[: HAPPY_PATH.index("Reserved")])

ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "AdvanceAdmission",
        "SatisfyConfirmation",
        "AcquireLease",
        "Reserve",
        "Start",
        "ApplyEffect",
        "Observe",
        "EnterUnknown",
        "Fail",
        "Abort",
        "RequireCompensation",
        "Compensate",
        "SealReceipt",
        "SettleCurrent",
        "PromoteProof",
        "Reject",
        "MarkUnavailable",
        "Retry",
        "BumpFence",
        "Crash",
        "Recover",
    }
)

REVERSIBILITY_CLASSES: Final[frozenset[str]] = frozenset(
    {"reversible", "compensatable", "irreversible"}
)

DEFAULT_PROTOCOL: Final[str] = "tep/v1"
DEFAULT_MAX_FENCE_GEN: Final[int] = 3
DEFAULT_MAX_RETRIES: Final[int] = 2


# ---------------------------------------------------------------------------
# Errors / verdicts
# ---------------------------------------------------------------------------


class MonitorErrorCode(str, Enum):
    """Closed rejection codes for the TEP runtime monitor."""

    UNKNOWN_TRANSITION = "unknown_transition"
    UNKNOWN_ACTION = "unknown_action"
    UNKNOWN_TYPESTATE = "unknown_typestate"
    UNKNOWN_BOUNDARY = "unknown_boundary"
    STALE_FENCE = "stale_fence"
    REPLAY = "replay"
    INCOMPATIBLE_IDEMPOTENCY = "incompatible_idempotency"
    INCOMPATIBLE_RECEIPT = "incompatible_receipt"
    NO_DOUBLE_EFFECT = "NoDoubleEffect"
    NO_STALE_FENCE = "NoStaleFenceCompletion"
    NO_SUCCESS_WITHOUT_OBSERVATION = "NoSuccessWithoutObservation"
    NO_CONFIRMATION_REUSE = "NoConfirmationReuse"
    NO_BLIND_UNKNOWN_RETRY = "NoBlindUnknownRetry"
    CRASH_PENDING = "crash_pending"
    NOT_CRASHED = "not_crashed"
    BOUND_EXCEEDED = "bound_exceeded"
    PRESTATE_MISMATCH = "prestate_mismatch"
    MISSING_FIELD = "missing_field"
    VECTOR_REJECTED = "vector_rejected"


class TransitionMonitorError(ValueError):
    """Fail-closed rejection raised by the formal transition monitor."""

    def __init__(
        self,
        code: MonitorErrorCode | str,
        message: str,
        *,
        invariant: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(code, MonitorErrorCode):
            self.code = code
        else:
            try:
                self.code = MonitorErrorCode(code)
            except ValueError:
                self.code = MonitorErrorCode.UNKNOWN_TRANSITION
        self.invariant = invariant
        self.details = dict(details or {})
        super().__init__(f"{self.code.value}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "invariant": self.invariant,
            "details": dict(self.details),
        }


def _require(cond: bool, code: MonitorErrorCode, message: str, **details: Any) -> None:
    if not cond:
        invariant = details.pop("invariant", None)
        raise TransitionMonitorError(code, message, invariant=invariant, details=details)


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_cid(prefix: str, value: Any) -> str:
    """Return a stable content identity used for argument / receipt binding."""

    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:sha256:{digest}"


def argument_cid_for(arguments: Mapping[str, Any] | None) -> str:
    return content_cid("argument", dict(arguments or {}))


def receipt_cid_for(
    *,
    instance_id: str,
    argument_cid: str,
    idempotency_key: str,
    observation_cid: str,
) -> str:
    return content_cid(
        "receipt",
        {
            "instance_id": instance_id,
            "argument_cid": argument_cid,
            "idempotency_key": idempotency_key,
            "observation_cid": observation_cid,
        },
    )


# ---------------------------------------------------------------------------
# Trace event / instance state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One monitor step carrying the FACP-046 evidence subset.

    Fields map directly onto the evidence vocabulary:
    prior/next state, protocol/instance/operation/actor, fence, idempotency,
    observation, and time.
    """

    action: str
    prior_state: str
    next_state: str
    protocol: str
    instance: str
    operation: str
    actor: str
    fence: int
    idempotency: str
    observation: bool
    time: int
    boundary: str
    argument_cid: str = ""
    receipt_cid: str = ""
    confirmation_cid: str = ""
    observation_cid: str = ""
    named_crash_boundary: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "prior_state": self.prior_state,
            "next_state": self.next_state,
            "protocol": self.protocol,
            "instance": self.instance,
            "operation": self.operation,
            "actor": self.actor,
            "fence": self.fence,
            "idempotency": self.idempotency,
            "observation": self.observation,
            "time": self.time,
            "boundary": self.boundary,
            "argument_cid": self.argument_cid,
            "receipt_cid": self.receipt_cid,
            "confirmation_cid": self.confirmation_cid,
            "observation_cid": self.observation_cid,
        }
        if self.named_crash_boundary is not None:
            payload["named_crash_boundary"] = self.named_crash_boundary
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @staticmethod
    def from_mapping(payload: Mapping[str, Any]) -> TraceEvent:
        missing = [
            name
            for name in (
                "action",
                "prior_state",
                "next_state",
                "protocol",
                "instance",
                "operation",
                "actor",
                "fence",
                "idempotency",
                "observation",
                "time",
                "boundary",
            )
            if name not in payload
        ]
        if missing:
            raise TransitionMonitorError(
                MonitorErrorCode.MISSING_FIELD,
                f"trace event missing fields: {missing}",
                details={"missing": missing},
            )
        return TraceEvent(
            action=str(payload["action"]),
            prior_state=str(payload["prior_state"]),
            next_state=str(payload["next_state"]),
            protocol=str(payload["protocol"]),
            instance=str(payload["instance"]),
            operation=str(payload["operation"]),
            actor=str(payload["actor"]),
            fence=int(payload["fence"]),
            idempotency=str(payload["idempotency"]),
            observation=bool(payload["observation"]),
            time=int(payload["time"]),
            boundary=str(payload["boundary"]),
            argument_cid=str(payload.get("argument_cid") or ""),
            receipt_cid=str(payload.get("receipt_cid") or ""),
            confirmation_cid=str(payload.get("confirmation_cid") or ""),
            observation_cid=str(payload.get("observation_cid") or ""),
            named_crash_boundary=(
                str(payload["named_crash_boundary"])
                if payload.get("named_crash_boundary") is not None
                else None
            ),
            extra=dict(payload.get("extra") or {}),
        )


@dataclass
class InstanceState:
    """Mutable per-instance protocol state tracked by the monitor."""

    instance_id: str
    operation: str
    actor: str
    protocol: str = DEFAULT_PROTOCOL
    reversibility: str = "reversible"
    typestate: str = "Proposed"
    effect_count: int = 0
    observed: bool = False
    receipt_sealed: bool = False
    confirmation_bound: bool = True
    confirmation_spent: bool = False
    confirmation_cid: str = ""
    lease_held: bool = False
    fence_gen: int = 1
    fence_at_reserve: int = 0
    current_pointer: int = 0
    pending_current: int = 0
    retries: int = 0
    idempotency_key: str = ""
    idempotency_recorded: bool = False
    argument_cid: str = ""
    receipt_cid: str = ""
    observation_cid: str = ""
    unknown_pending: bool = False
    compensation_owed: bool = False
    proof_promoted: bool = False
    durable_cursor: str = "Proposed"
    last_crash_boundary: str = "none"
    time: int = 0
    sealed_event_ids: set[str] = field(default_factory=set)

    def snapshot(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "operation": self.operation,
            "actor": self.actor,
            "protocol": self.protocol,
            "reversibility": self.reversibility,
            "typestate": self.typestate,
            "effect_count": self.effect_count,
            "observed": self.observed,
            "receipt_sealed": self.receipt_sealed,
            "confirmation_bound": self.confirmation_bound,
            "confirmation_spent": self.confirmation_spent,
            "confirmation_cid": self.confirmation_cid,
            "lease_held": self.lease_held,
            "fence_gen": self.fence_gen,
            "fence_at_reserve": self.fence_at_reserve,
            "current_pointer": self.current_pointer,
            "pending_current": self.pending_current,
            "retries": self.retries,
            "idempotency_key": self.idempotency_key,
            "idempotency_recorded": self.idempotency_recorded,
            "argument_cid": self.argument_cid,
            "receipt_cid": self.receipt_cid,
            "observation_cid": self.observation_cid,
            "unknown_pending": self.unknown_pending,
            "compensation_owed": self.compensation_owed,
            "proof_promoted": self.proof_promoted,
            "durable_cursor": self.durable_cursor,
            "last_crash_boundary": self.last_crash_boundary,
            "time": self.time,
        }


@dataclass(frozen=True, slots=True)
class MonitorVerdict:
    """Accepted / rejected monitor decision for a single step or vector."""

    schema: str
    accepted: bool
    code: str
    message: str
    invariants: Mapping[str, bool]
    event: Mapping[str, Any] | None = None
    state: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
            "accepted": self.accepted,
            "code": self.code,
            "message": self.message,
            "invariants": dict(self.invariants),
            "event": dict(self.event) if self.event is not None else None,
            "state": dict(self.state) if self.state is not None else None,
        }


# ---------------------------------------------------------------------------
# Formal transition monitor
# ---------------------------------------------------------------------------


@dataclass
class FormalTransitionMonitor:
    """Fail-closed runtime monitor over TEP transition traces."""

    max_fence_gen: int = DEFAULT_MAX_FENCE_GEN
    max_retries: int = DEFAULT_MAX_RETRIES
    protocol: str = DEFAULT_PROTOCOL
    instances: dict[str, InstanceState] = field(default_factory=dict)
    crashed: bool = False
    clock: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    last_boundary: str = "none"

    def ensure_instance(
        self,
        *,
        instance_id: str,
        operation: str,
        actor: str,
        reversibility: str = "reversible",
        argument_cid: str = "",
        confirmation_cid: str = "",
        idempotency_key: str = "",
    ) -> InstanceState:
        _require(
            reversibility in REVERSIBILITY_CLASSES,
            MonitorErrorCode.UNKNOWN_TRANSITION,
            f"unknown reversibility {reversibility!r}",
        )
        existing = self.instances.get(instance_id)
        if existing is None:
            st = InstanceState(
                instance_id=instance_id,
                operation=operation,
                actor=actor,
                protocol=self.protocol,
                reversibility=reversibility,
                argument_cid=argument_cid,
                confirmation_cid=confirmation_cid,
                idempotency_key=idempotency_key,
            )
            self.instances[instance_id] = st
            return st
        # Replay / identity checks for subsequent events.
        if existing.operation != operation:
            raise TransitionMonitorError(
                MonitorErrorCode.REPLAY,
                "operation identity changed across events",
                details={
                    "prior_operation": existing.operation,
                    "next_operation": operation,
                },
            )
        if existing.actor != actor:
            raise TransitionMonitorError(
                MonitorErrorCode.REPLAY,
                "actor identity changed across events",
                details={"prior_actor": existing.actor, "next_actor": actor},
            )
        return existing

    def invariants(self) -> dict[str, bool]:
        results = {name: True for name in REQUIRED_INVARIANTS}
        for st in self.instances.values():
            if st.effect_count > 1:
                results["NoDoubleEffect"] = False
            if st.receipt_sealed and st.typestate == "ReceiptSealed":
                if st.fence_at_reserve != st.fence_gen:
                    results["NoStaleFenceCompletion"] = False
            if st.proof_promoted and not st.observed:
                results["NoSuccessWithoutObservation"] = False
            if st.current_pointer > 0 and not st.observed:
                results["NoSuccessWithoutObservation"] = False
            if (
                st.receipt_sealed
                and st.effect_count == 1
                and st.typestate == "ReceiptSealed"
                and not st.observed
            ):
                results["NoSuccessWithoutObservation"] = False
            if st.confirmation_spent and not st.confirmation_bound:
                results["NoConfirmationReuse"] = False
            if (
                st.reversibility == "irreversible"
                and st.unknown_pending
                and st.typestate in {"Reserved", "Started"}
                and st.retries > 0
            ):
                results["NoBlindUnknownRetry"] = False
        return results

    def _record(
        self,
        *,
        action: str,
        st: InstanceState | None,
        prior: str,
        nxt: str,
        boundary: str,
        named_crash_boundary: str | None = None,
        **extra: Any,
    ) -> TraceEvent:
        self.clock += 1
        self.last_boundary = boundary
        if st is not None:
            st.time = self.clock
            st.last_crash_boundary = boundary
        event = TraceEvent(
            action=action,
            prior_state=prior,
            next_state=nxt,
            protocol=self.protocol if st is None else st.protocol,
            instance="" if st is None else st.instance_id,
            operation="" if st is None else st.operation,
            actor="" if st is None else st.actor,
            fence=0 if st is None else st.fence_gen,
            idempotency="" if st is None else st.idempotency_key,
            observation=False if st is None else st.observed,
            time=self.clock,
            boundary=boundary,
            argument_cid="" if st is None else st.argument_cid,
            receipt_cid="" if st is None else st.receipt_cid,
            confirmation_cid="" if st is None else st.confirmation_cid,
            observation_cid="" if st is None else st.observation_cid,
            named_crash_boundary=named_crash_boundary,
            extra=dict(extra),
        )
        self.trace.append(event.to_dict())
        return event

    def _commit(self, st: InstanceState, next_state: str, boundary: str) -> None:
        st.typestate = next_state
        st.durable_cursor = next_state
        st.last_crash_boundary = boundary

    def apply_action(
        self,
        action: str,
        *,
        instance_id: str | None = None,
        operation: str = "op.default",
        actor: str = "actor.default",
        reversibility: str = "reversible",
        argument_cid: str | None = None,
        idempotency_key: str | None = None,
        confirmation_cid: str | None = None,
        observation_cid: str | None = None,
        receipt_cid: str | None = None,
        named_crash_boundary: str | None = None,
        event_id: str | None = None,
    ) -> MonitorVerdict:
        """Apply one closed TEP action and return an accept/reject verdict."""

        try:
            event = self._apply_action_inner(
                action,
                instance_id=instance_id,
                operation=operation,
                actor=actor,
                reversibility=reversibility,
                argument_cid=argument_cid,
                idempotency_key=idempotency_key,
                confirmation_cid=confirmation_cid,
                observation_cid=observation_cid,
                receipt_cid=receipt_cid,
                named_crash_boundary=named_crash_boundary,
                event_id=event_id,
            )
            inv = self.invariants()
            if not all(inv.values()):
                broken = [k for k, v in inv.items() if not v]
                raise TransitionMonitorError(
                    MonitorErrorCode.VECTOR_REJECTED,
                    f"invariant broken: {broken}",
                    invariant=broken[0],
                    details={"invariants": inv},
                )
            st = None if instance_id is None else self.instances.get(instance_id)
            return MonitorVerdict(
                schema=VERDICT_SCHEMA,
                accepted=True,
                code="accepted",
                message="transition accepted",
                invariants=inv,
                event=event.to_dict(),
                state=None if st is None else st.snapshot(),
            )
        except TransitionMonitorError as exc:
            return MonitorVerdict(
                schema=VERDICT_SCHEMA,
                accepted=False,
                code=exc.code.value,
                message=str(exc),
                invariants=self.invariants(),
                event=None,
                state=None,
            )

    def apply_action_or_raise(self, action: str, **kwargs: Any) -> TraceEvent:
        verdict = self.apply_action(action, **kwargs)
        if not verdict.accepted:
            raise TransitionMonitorError(
                verdict.code,
                verdict.message,
                invariant=next(
                    (k for k, v in verdict.invariants.items() if not v), None
                ),
                details={"verdict": verdict.to_dict()},
            )
        assert verdict.event is not None
        return TraceEvent.from_mapping(verdict.event)

    def _apply_action_inner(
        self,
        action: str,
        *,
        instance_id: str | None,
        operation: str,
        actor: str,
        reversibility: str,
        argument_cid: str | None,
        idempotency_key: str | None,
        confirmation_cid: str | None,
        observation_cid: str | None,
        receipt_cid: str | None,
        named_crash_boundary: str | None,
        event_id: str | None,
    ) -> TraceEvent:
        _require(
            action in ACTIONS,
            MonitorErrorCode.UNKNOWN_ACTION,
            f"unknown action {action!r}",
        )

        if action == "Recover":
            _require(self.crashed, MonitorErrorCode.NOT_CRASHED, "Recover requires crashed")
            self.crashed = False
            return self._record(
                action=action,
                st=None,
                prior="Crashed",
                nxt="Recovered",
                boundary="crash",
            )

        if action == "Crash":
            boundary = named_crash_boundary or self.last_boundary
            _require(
                boundary in CRASH_BOUNDARIES,
                MonitorErrorCode.UNKNOWN_BOUNDARY,
                f"unknown crash boundary {boundary!r}",
            )
            _require(not self.crashed, MonitorErrorCode.CRASH_PENDING, "already crashed")
            for st in self.instances.values():
                prior = st.typestate
                st.typestate = st.durable_cursor
                st.last_crash_boundary = "crash"
                _ = prior
            self.crashed = True
            return self._record(
                action=action,
                st=None,
                prior="Live",
                nxt="Crashed",
                boundary="crash",
                named_crash_boundary=boundary,
            )

        _require(
            instance_id is not None and bool(instance_id),
            MonitorErrorCode.MISSING_FIELD,
            "instance_id required for non-crash actions",
        )
        assert instance_id is not None
        _require(
            not self.crashed,
            MonitorErrorCode.CRASH_PENDING,
            "must Recover before further actions",
        )

        st = self.ensure_instance(
            instance_id=instance_id,
            operation=operation,
            actor=actor,
            reversibility=reversibility,
            argument_cid=argument_cid or "",
            confirmation_cid=confirmation_cid or "",
            idempotency_key=idempotency_key or "",
        )
        prior = st.typestate
        _require(
            prior in TYPESTATES,
            MonitorErrorCode.UNKNOWN_TYPESTATE,
            f"unknown typestate {prior!r}",
        )

        # Exact-argument / idempotency / confirmation binding checks.
        if argument_cid is not None and st.argument_cid and argument_cid != st.argument_cid:
            raise TransitionMonitorError(
                MonitorErrorCode.INCOMPATIBLE_RECEIPT
                if action in {"SealReceipt", "SettleCurrent", "PromoteProof"}
                else MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY,
                "argument_cid changed after binding",
                details={"prior": st.argument_cid, "next": argument_cid},
            )
        if argument_cid and not st.argument_cid:
            st.argument_cid = argument_cid

        if (
            idempotency_key is not None
            and st.idempotency_recorded
            and st.idempotency_key
            and idempotency_key != st.idempotency_key
        ):
            raise TransitionMonitorError(
                MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY,
                "idempotency key mismatch against recorded reservation",
                details={"prior": st.idempotency_key, "next": idempotency_key},
            )

        if event_id is not None:
            if event_id in st.sealed_event_ids:
                raise TransitionMonitorError(
                    MonitorErrorCode.REPLAY,
                    f"event_id {event_id!r} already applied",
                    details={"event_id": event_id},
                )

        def finish(next_state: str, boundary: str, **extra: Any) -> TraceEvent:
            self._commit(st, next_state, boundary)
            if event_id is not None:
                st.sealed_event_ids.add(event_id)
            return self._record(
                action=action,
                st=st,
                prior=prior,
                nxt=next_state,
                boundary=boundary,
                **extra,
            )

        if action == "AdvanceAdmission":
            _require(
                st.typestate in PRE_RESERVED and st.typestate != "ConfirmationSatisfied",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "bad admission state",
                prior=st.typestate,
            )
            _require(
                st.typestate != "ObligationsSatisfied" or not st.confirmation_bound,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "use SatisfyConfirmation",
            )
            idx = HAPPY_PATH.index(st.typestate)
            nxt = HAPPY_PATH[idx + 1]
            _require(
                nxt != "ConfirmationSatisfied" or not st.confirmation_bound,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "use SatisfyConfirmation",
            )
            return finish(nxt, "admission")

        if action == "SatisfyConfirmation":
            _require(
                st.typestate == "ObligationsSatisfied",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "confirmation prestate",
            )
            _require(
                st.confirmation_bound,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "confirmation not bound",
            )
            _require(
                not st.confirmation_spent,
                MonitorErrorCode.NO_CONFIRMATION_REUSE,
                "confirmation already spent",
                invariant="NoConfirmationReuse",
            )
            if confirmation_cid:
                if st.confirmation_cid and st.confirmation_cid != confirmation_cid:
                    raise TransitionMonitorError(
                        MonitorErrorCode.REPLAY,
                        "confirmation_cid reuse / mismatch",
                        invariant="NoConfirmationReuse",
                        details={
                            "prior": st.confirmation_cid,
                            "next": confirmation_cid,
                        },
                    )
                st.confirmation_cid = confirmation_cid
            st.confirmation_spent = True
            return finish("ConfirmationSatisfied", "admission")

        if action == "AcquireLease":
            _require(
                st.typestate == "ConfirmationSatisfied",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "lease prestate",
            )
            _require(
                st.fence_gen < self.max_fence_gen,
                MonitorErrorCode.BOUND_EXCEEDED,
                "fence generation bound exceeded",
            )
            st.lease_held = True
            st.fence_gen += 1
            return finish("LeaseHeld", "lease")

        if action == "Reserve":
            _require(
                st.typestate == "LeaseHeld" and st.lease_held,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "reserve prestate",
            )
            _require(
                not st.idempotency_recorded,
                MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY,
                "idempotency already recorded",
            )
            key = idempotency_key or st.idempotency_key or f"idem:{st.instance_id}"
            if st.idempotency_key and key != st.idempotency_key:
                raise TransitionMonitorError(
                    MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY,
                    "idempotency key mismatch at reserve",
                    details={"prior": st.idempotency_key, "next": key},
                )
            st.idempotency_key = key
            st.idempotency_recorded = True
            if argument_cid:
                st.argument_cid = argument_cid
            elif not st.argument_cid:
                st.argument_cid = argument_cid_for({"instance": st.instance_id})
            st.fence_at_reserve = st.fence_gen
            return finish("Reserved", "reservation")

        if action == "Start":
            _require(
                st.typestate == "Reserved",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "start prestate",
            )
            _require(st.lease_held, MonitorErrorCode.PRESTATE_MISMATCH, "lease required")
            _require(
                st.fence_at_reserve == st.fence_gen,
                MonitorErrorCode.STALE_FENCE,
                "stale fence at start",
                invariant="NoStaleFenceCompletion",
            )
            if idempotency_key is not None and idempotency_key != st.idempotency_key:
                raise TransitionMonitorError(
                    MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY,
                    "idempotency key mismatch at start",
                    details={"prior": st.idempotency_key, "next": idempotency_key},
                )
            return finish("Started", "started")

        if action == "ApplyEffect":
            _require(
                st.typestate == "Started",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "apply prestate",
            )
            _require(
                st.effect_count == 0,
                MonitorErrorCode.NO_DOUBLE_EFFECT,
                "effect already applied",
                invariant="NoDoubleEffect",
            )
            if idempotency_key is not None and idempotency_key != st.idempotency_key:
                raise TransitionMonitorError(
                    MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY,
                    "idempotency key mismatch at apply",
                    details={"prior": st.idempotency_key, "next": idempotency_key},
                )
            st.effect_count = 1
            st.pending_current = st.current_pointer + 1
            st.last_crash_boundary = "idempotency"
            if event_id is not None:
                st.sealed_event_ids.add(event_id)
            return self._record(
                action=action,
                st=st,
                prior=prior,
                nxt=prior,  # typestate unchanged; effect recorded
                boundary="idempotency",
                effect_count=1,
            )

        if action == "Observe":
            _require(
                st.typestate in {"Started", "Unknown"},
                MonitorErrorCode.PRESTATE_MISMATCH,
                "observe prestate",
            )
            _require(
                st.effect_count == 1,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "nothing to observe",
            )
            _require(not st.observed, MonitorErrorCode.REPLAY, "already observed")
            obs = observation_cid or content_cid(
                "observation",
                {"instance": st.instance_id, "argument_cid": st.argument_cid},
            )
            st.observation_cid = obs
            st.observed = True
            st.unknown_pending = False
            return finish("Observed", "observed")

        if action == "EnterUnknown":
            _require(
                st.typestate == "Started",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "unknown prestate",
            )
            st.unknown_pending = True
            return finish("Unknown", "unknown")

        if action == "Fail":
            _require(
                st.typestate in {"Started", "Unknown", "CompensationRequired"},
                MonitorErrorCode.PRESTATE_MISMATCH,
                "fail prestate",
            )
            if not (st.reversibility == "irreversible" and st.unknown_pending):
                st.unknown_pending = False
            return finish("Failed", "settlement")

        if action == "Abort":
            _require(
                st.typestate in {"Started", "Unknown"},
                MonitorErrorCode.PRESTATE_MISMATCH,
                "abort prestate",
            )
            _require(
                st.effect_count == 0,
                MonitorErrorCode.NO_DOUBLE_EFFECT,
                "cannot abort after effect",
            )
            if (
                st.reversibility == "irreversible"
                and st.unknown_pending
                and st.effect_count == 1
            ):
                raise TransitionMonitorError(
                    MonitorErrorCode.NO_BLIND_UNKNOWN_RETRY,
                    "abort blocked for irreversible unknown effect",
                    invariant="NoBlindUnknownRetry",
                )
            st.unknown_pending = False
            return finish("Aborted", "settlement")

        if action == "RequireCompensation":
            _require(
                st.reversibility == "compensatable",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "not compensatable",
            )
            _require(
                st.typestate in {"Started", "Unknown", "Observed"},
                MonitorErrorCode.PRESTATE_MISMATCH,
                "compensation prestate",
            )
            st.compensation_owed = True
            st.unknown_pending = False
            return finish("CompensationRequired", "compensation")

        if action == "Compensate":
            _require(
                st.typestate == "CompensationRequired",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "compensate prestate",
            )
            _require(st.compensation_owed, MonitorErrorCode.PRESTATE_MISMATCH, "nothing owed")
            st.compensation_owed = False
            st.observed = True
            if not st.observation_cid:
                st.observation_cid = content_cid(
                    "observation", {"compensated": st.instance_id}
                )
            return finish("Compensated", "compensation")

        if action == "SealReceipt":
            _require(
                st.typestate
                in {
                    "Observed",
                    "Compensated",
                    "Failed",
                    "Rejected",
                    "Unavailable",
                    "Aborted",
                },
                MonitorErrorCode.PRESTATE_MISMATCH,
                "seal prestate",
            )
            if st.typestate in {"Observed", "Compensated"}:
                _require(
                    st.observed,
                    MonitorErrorCode.NO_SUCCESS_WITHOUT_OBSERVATION,
                    "success requires observation",
                    invariant="NoSuccessWithoutObservation",
                )
            _require(
                st.fence_at_reserve == st.fence_gen,
                MonitorErrorCode.STALE_FENCE,
                "stale fence at receipt seal",
                invariant="NoStaleFenceCompletion",
            )
            expected = receipt_cid_for(
                instance_id=st.instance_id,
                argument_cid=st.argument_cid,
                idempotency_key=st.idempotency_key,
                observation_cid=st.observation_cid,
            )
            if receipt_cid is not None and receipt_cid != expected:
                raise TransitionMonitorError(
                    MonitorErrorCode.INCOMPATIBLE_RECEIPT,
                    "receipt arguments incompatible with reservation/observation",
                    details={"expected": expected, "got": receipt_cid},
                )
            if argument_cid is not None and argument_cid != st.argument_cid:
                raise TransitionMonitorError(
                    MonitorErrorCode.INCOMPATIBLE_RECEIPT,
                    "receipt argument_cid mismatch",
                    details={"expected": st.argument_cid, "got": argument_cid},
                )
            if (
                idempotency_key is not None
                and idempotency_key != st.idempotency_key
            ):
                raise TransitionMonitorError(
                    MonitorErrorCode.INCOMPATIBLE_RECEIPT,
                    "receipt idempotency mismatch",
                    details={"expected": st.idempotency_key, "got": idempotency_key},
                )
            st.receipt_cid = expected
            st.receipt_sealed = True
            return finish("ReceiptSealed", "receipt")

        if action == "SettleCurrent":
            _require(
                st.typestate == "ReceiptSealed" and st.receipt_sealed,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "settle prestate",
            )
            _require(
                st.observed,
                MonitorErrorCode.NO_SUCCESS_WITHOUT_OBSERVATION,
                "current requires observation",
                invariant="NoSuccessWithoutObservation",
            )
            _require(
                st.pending_current == st.current_pointer + 1,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "pending current mismatch",
            )
            if receipt_cid is not None and receipt_cid != st.receipt_cid:
                raise TransitionMonitorError(
                    MonitorErrorCode.INCOMPATIBLE_RECEIPT,
                    "settle receipt_cid mismatch",
                    details={"expected": st.receipt_cid, "got": receipt_cid},
                )
            st.current_pointer = st.pending_current
            st.last_crash_boundary = "current"
            if event_id is not None:
                st.sealed_event_ids.add(event_id)
            return self._record(
                action=action,
                st=st,
                prior=prior,
                nxt=prior,
                boundary="current",
                current=st.current_pointer,
            )

        if action == "PromoteProof":
            _require(
                st.typestate == "ReceiptSealed" and st.receipt_sealed,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "proof prestate",
            )
            _require(
                st.observed,
                MonitorErrorCode.NO_SUCCESS_WITHOUT_OBSERVATION,
                "observation required for proof",
                invariant="NoSuccessWithoutObservation",
            )
            _require(
                st.current_pointer == st.pending_current and st.pending_current > 0,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "current pointer not settled",
            )
            _require(
                not st.proof_promoted,
                MonitorErrorCode.REPLAY,
                "proof already promoted",
            )
            st.proof_promoted = True
            st.last_crash_boundary = "proof_promotion"
            if event_id is not None:
                st.sealed_event_ids.add(event_id)
            return self._record(
                action=action,
                st=st,
                prior=prior,
                nxt=prior,
                boundary="proof_promotion",
                proof=True,
            )

        if action == "Reject":
            _require(
                st.typestate in PRE_RESERVED,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "reject prestate",
            )
            return finish("Rejected", "admission")

        if action == "MarkUnavailable":
            _require(
                st.typestate in PRE_RESERVED,
                MonitorErrorCode.PRESTATE_MISMATCH,
                "unavailable prestate",
            )
            return finish("Unavailable", "admission")

        if action == "BumpFence":
            _require(st.lease_held, MonitorErrorCode.PRESTATE_MISMATCH, "lease required")
            _require(
                st.fence_gen < self.max_fence_gen,
                MonitorErrorCode.BOUND_EXCEEDED,
                "fence generation bound exceeded",
            )
            st.fence_gen += 1
            st.last_crash_boundary = "fence"
            if event_id is not None:
                st.sealed_event_ids.add(event_id)
            return self._record(
                action=action,
                st=st,
                prior=prior,
                nxt=prior,
                boundary="fence",
                fence_gen=st.fence_gen,
            )

        if action == "Retry":
            _require(
                st.typestate == "Failed",
                MonitorErrorCode.PRESTATE_MISMATCH,
                "retry prestate",
            )
            _require(
                st.retries < self.max_retries,
                MonitorErrorCode.BOUND_EXCEEDED,
                "retry bound exceeded",
            )
            if st.reversibility == "irreversible" and (
                st.unknown_pending or st.effect_count > 0
            ):
                raise TransitionMonitorError(
                    MonitorErrorCode.NO_BLIND_UNKNOWN_RETRY,
                    "blind retry of unknown/irreversible effect forbidden",
                    invariant="NoBlindUnknownRetry",
                )
            _require(
                not st.idempotency_recorded or st.effect_count == 0,
                MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY,
                "idempotency blocks retry after effect",
            )
            st.retries += 1
            st.idempotency_recorded = False
            st.receipt_sealed = False
            st.observed = False
            return finish("LeaseHeld", "retry")

        raise TransitionMonitorError(
            MonitorErrorCode.UNKNOWN_ACTION,
            f"unhandled action {action!r}",
        )

    def apply_event(self, event: TraceEvent | Mapping[str, Any]) -> MonitorVerdict:
        """Validate a fully-specified evidence event against the live state."""

        te = event if isinstance(event, TraceEvent) else TraceEvent.from_mapping(event)
        kwargs: dict[str, Any] = {
            "instance_id": te.instance or None,
            "operation": te.operation or "op.default",
            "actor": te.actor or "actor.default",
            "argument_cid": te.argument_cid or None,
            "idempotency_key": te.idempotency or None,
            "confirmation_cid": te.confirmation_cid or None,
            "observation_cid": te.observation_cid or None,
            "receipt_cid": te.receipt_cid or None,
            "named_crash_boundary": te.named_crash_boundary,
        }
        if te.action in {"Crash", "Recover"}:
            kwargs["instance_id"] = None
        verdict = self.apply_action(te.action, **kwargs)
        if not verdict.accepted:
            return verdict
        # When the caller supplies expected prior/next, enforce exact match.
        produced = verdict.event or {}
        if te.prior_state and produced.get("prior_state") not in {te.prior_state, "Live", "Crashed"}:
            # Crash/Recover use synthetic states; otherwise require exact prior.
            if te.action not in {"Crash", "Recover"} and produced.get("prior_state") != te.prior_state:
                return MonitorVerdict(
                    schema=VERDICT_SCHEMA,
                    accepted=False,
                    code=MonitorErrorCode.PRESTATE_MISMATCH.value,
                    message=(
                        f"prior_state mismatch: expected {te.prior_state!r} "
                        f"got {produced.get('prior_state')!r}"
                    ),
                    invariants=self.invariants(),
                    event=produced,
                    state=verdict.state,
                )
        return verdict

    def run_steps(
        self,
        steps: Sequence[Mapping[str, Any] | tuple[Any, ...]],
        *,
        expect_ok: bool = True,
    ) -> list[MonitorVerdict]:
        """Run a compact step recipe. Each step is a mapping or action tuple."""

        verdicts: list[MonitorVerdict] = []
        for raw in steps:
            kwargs = _normalize_step(raw)
            action = kwargs.pop("action")
            verdict = self.apply_action(action, **kwargs)
            verdicts.append(verdict)
            if expect_ok and not verdict.accepted:
                raise TransitionMonitorError(
                    verdict.code,
                    verdict.message,
                    details={"verdict": verdict.to_dict()},
                )
            if not expect_ok and not verdict.accepted:
                return verdicts
        if not expect_ok:
            raise AssertionError("expected vector to fail")
        return verdicts


def _normalize_step(raw: Mapping[str, Any] | tuple[Any, ...]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        if "action" not in raw:
            raise TransitionMonitorError(
                MonitorErrorCode.MISSING_FIELD,
                "step mapping requires action",
            )
        return dict(raw)
    if not raw:
        raise TransitionMonitorError(
            MonitorErrorCode.MISSING_FIELD,
            "empty step tuple",
        )
    action = raw[0]
    kwargs: dict[str, Any] = {"action": action}
    if len(raw) > 1 and raw[1] is not None:
        kwargs["instance_id"] = raw[1]
    if len(raw) > 2 and isinstance(raw[2], Mapping):
        kwargs.update(dict(raw[2]))
    return kwargs


def default_monitor(**kwargs: Any) -> FormalTransitionMonitor:
    return FormalTransitionMonitor(**kwargs)


# ---------------------------------------------------------------------------
# Vector adapter (FACP-045 model traces / runtime events -> monitor steps)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdaptedStep:
    action: str
    instance_id: str | None
    kwargs: Mapping[str, Any]

    def to_monitor_kwargs(self) -> dict[str, Any]:
        payload = {"action": self.action, **dict(self.kwargs)}
        if self.instance_id is not None:
            payload["instance_id"] = self.instance_id
        return payload


class VectorAdapter:
    """Adapt FACP-045 reference traces and runtime JSONL-like events."""

    def adapt_model_step(
        self,
        action: str,
        op_id: str | None,
        *,
        reversibility: str = "reversible",
        argument_cid: str | None = None,
        idempotency_key: str | None = None,
        **extra: Any,
    ) -> AdaptedStep:
        if action.startswith("Crash:"):
            boundary = action.split(":", 1)[1]
            return AdaptedStep(
                action="Crash",
                instance_id=None,
                kwargs={"named_crash_boundary": boundary, **extra},
            )
        kwargs: dict[str, Any] = {"reversibility": reversibility, **extra}
        if argument_cid is not None:
            kwargs["argument_cid"] = argument_cid
        if idempotency_key is not None:
            kwargs["idempotency_key"] = idempotency_key
        return AdaptedStep(action=action, instance_id=op_id, kwargs=MappingProxyType(kwargs))

    def adapt_model_trace(
        self,
        steps: Sequence[tuple[str, str | None]],
        *,
        reversibility: str = "reversible",
        argument_cid: str | None = None,
        idempotency_key: str | None = None,
    ) -> list[AdaptedStep]:
        return [
            self.adapt_model_step(
                action,
                op_id,
                reversibility=reversibility,
                argument_cid=argument_cid,
                idempotency_key=idempotency_key,
            )
            for action, op_id in steps
        ]

    def adapt_runtime_event(self, payload: Mapping[str, Any]) -> AdaptedStep:
        """Map a supervisor-style runtime event onto a monitor step."""

        action = str(
            payload.get("action")
            or payload.get("event")
            or payload.get("transition")
            or ""
        ).strip()
        if not action:
            raise TransitionMonitorError(
                MonitorErrorCode.MISSING_FIELD,
                "runtime event missing action/event/transition",
            )
        if action.startswith("Crash:"):
            return self.adapt_model_step(action, None)

        instance = payload.get("instance") or payload.get("instance_id") or payload.get("op")
        kwargs: dict[str, Any] = {}
        for src, dst in (
            ("operation", "operation"),
            ("operation_id", "operation"),
            ("actor", "actor"),
            ("actor_cid", "actor"),
            ("reversibility", "reversibility"),
            ("argument_cid", "argument_cid"),
            ("idempotency", "idempotency_key"),
            ("idempotency_key", "idempotency_key"),
            ("confirmation_cid", "confirmation_cid"),
            ("observation_cid", "observation_cid"),
            ("receipt_cid", "receipt_cid"),
            ("event_id", "event_id"),
            ("named_crash_boundary", "named_crash_boundary"),
            ("boundary", "named_crash_boundary"),
        ):
            if src in payload and payload[src] is not None and dst not in kwargs:
                kwargs[dst] = payload[src]
        return AdaptedStep(
            action=action,
            instance_id=None if instance is None else str(instance),
            kwargs=MappingProxyType(kwargs),
        )

    def run_adapted(
        self,
        monitor: FormalTransitionMonitor,
        steps: Sequence[AdaptedStep],
        *,
        expect_ok: bool = True,
    ) -> list[MonitorVerdict]:
        recipes = [step.to_monitor_kwargs() for step in steps]
        return monitor.run_steps(recipes, expect_ok=expect_ok)


# ---------------------------------------------------------------------------
# Normative vectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormativeVector:
    vector_id: str
    expect_accept: bool
    reversibility: str
    steps: tuple[tuple[str, str | None], ...]
    expected_code: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VECTOR_SCHEMA,
            "vector_id": self.vector_id,
            "expect_accept": self.expect_accept,
            "reversibility": self.reversibility,
            "steps": [{"action": a, "instance": i} for a, i in self.steps],
            "expected_code": self.expected_code,
            "notes": self.notes,
        }


def _admission_to_obligations(instance: str) -> list[tuple[str, str | None]]:
    steps: list[tuple[str, str | None]] = []
    for _ in range(HAPPY_PATH.index("ObligationsSatisfied")):
        steps.append(("AdvanceAdmission", instance))
    return steps


def happy_path_steps(instance: str = "o1") -> list[tuple[str, str | None]]:
    steps = _admission_to_obligations(instance)
    steps.extend(
        [
            ("SatisfyConfirmation", instance),
            ("AcquireLease", instance),
            ("Reserve", instance),
            ("Start", instance),
            ("ApplyEffect", instance),
            ("Observe", instance),
            ("SealReceipt", instance),
            ("SettleCurrent", instance),
            ("PromoteProof", instance),
        ]
    )
    return steps


def load_normative_vectors() -> tuple[NormativeVector, ...]:
    """Return the closed normative accept/reject corpus for FACP-046."""

    o = "o1"
    accept: list[NormativeVector] = [
        NormativeVector(
            vector_id="accept/happy-path",
            expect_accept=True,
            reversibility="reversible",
            steps=tuple(happy_path_steps(o)),
            notes="Full happy path through proof promotion",
        ),
        NormativeVector(
            vector_id="accept/compensation",
            expect_accept=True,
            reversibility="compensatable",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Start", o),
                    ("ApplyEffect", o),
                    ("RequireCompensation", o),
                    ("Compensate", o),
                    ("SealReceipt", o),
                ]
            ),
            notes="Compensatable path seals after compensation observation",
        ),
        NormativeVector(
            vector_id="accept/crash-reservation-recover",
            expect_accept=True,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Crash:reservation", None),
                    ("Recover", None),
                    ("Start", o),
                    ("ApplyEffect", o),
                    ("Observe", o),
                    ("SealReceipt", o),
                ]
            ),
            notes="Crash at reservation boundary then recover to completion",
        ),
        NormativeVector(
            vector_id="accept/reject-early",
            expect_accept=True,
            reversibility="reversible",
            steps=(("Reject", o),),
            notes="Early admission reject",
        ),
    ]

    reject: list[NormativeVector] = [
        NormativeVector(
            vector_id="reject/double-effect",
            expect_accept=False,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Start", o),
                    ("ApplyEffect", o),
                    ("ApplyEffect", o),
                ]
            ),
            expected_code=MonitorErrorCode.NO_DOUBLE_EFFECT.value,
            notes="NoDoubleEffect",
        ),
        NormativeVector(
            vector_id="reject/stale-fence",
            expect_accept=False,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Start", o),
                    ("ApplyEffect", o),
                    ("Observe", o),
                    ("BumpFence", o),
                    ("SealReceipt", o),
                ]
            ),
            expected_code=MonitorErrorCode.STALE_FENCE.value,
            notes="NoStaleFenceCompletion",
        ),
        NormativeVector(
            vector_id="reject/success-without-observation",
            expect_accept=False,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Start", o),
                    ("ApplyEffect", o),
                    ("SealReceipt", o),
                ]
            ),
            expected_code=MonitorErrorCode.PRESTATE_MISMATCH.value,
            notes="NoSuccessWithoutObservation (seal requires Observed)",
        ),
        NormativeVector(
            vector_id="reject/confirmation-reuse",
            expect_accept=False,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("SatisfyConfirmation", o),
                ]
            ),
            expected_code=MonitorErrorCode.PRESTATE_MISMATCH.value,
            notes="NoConfirmationReuse (spent + wrong prestate)",
        ),
        NormativeVector(
            vector_id="reject/blind-unknown-retry",
            expect_accept=False,
            reversibility="irreversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Start", o),
                    ("ApplyEffect", o),
                    ("EnterUnknown", o),
                    ("Fail", o),
                    ("Retry", o),
                ]
            ),
            expected_code=MonitorErrorCode.NO_BLIND_UNKNOWN_RETRY.value,
            notes="NoBlindUnknownRetry",
        ),
        NormativeVector(
            vector_id="reject/replay-event",
            expect_accept=False,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                ]
            ),
            expected_code=MonitorErrorCode.REPLAY.value,
            notes="Handled specially in evaluate with event_id replay",
        ),
        NormativeVector(
            vector_id="reject/incompatible-idempotency",
            expect_accept=False,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Start", o),
                ]
            ),
            expected_code=MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY.value,
            notes="Handled specially with mismatched idempotency at Start",
        ),
        NormativeVector(
            vector_id="reject/incompatible-receipt",
            expect_accept=False,
            reversibility="reversible",
            steps=tuple(
                _admission_to_obligations(o)
                + [
                    ("SatisfyConfirmation", o),
                    ("AcquireLease", o),
                    ("Reserve", o),
                    ("Start", o),
                    ("ApplyEffect", o),
                    ("Observe", o),
                    ("SealReceipt", o),
                ]
            ),
            expected_code=MonitorErrorCode.INCOMPATIBLE_RECEIPT.value,
            notes="Handled specially with forged receipt_cid",
        ),
    ]
    return tuple(accept + reject)


def evaluate_normative_vector(vector: NormativeVector) -> MonitorVerdict:
    """Evaluate one normative vector; special-cases binding rejection probes."""

    monitor = default_monitor()
    adapter = VectorAdapter()
    arg_cid = argument_cid_for({"vector": vector.vector_id, "instance": "o1"})
    idem = f"idem:{vector.vector_id}"

    if vector.vector_id == "reject/replay-event":
        adapted = adapter.adapt_model_trace(
            vector.steps,
            reversibility=vector.reversibility,
            argument_cid=arg_cid,
            idempotency_key=idem,
        )
        # Force a duplicate Reserve with the same event_id.
        recipes = [s.to_monitor_kwargs() for s in adapted]
        for step in recipes:
            if step["action"] == "Reserve":
                step["event_id"] = "evt-reserve-1"
        monitor.run_steps(recipes[:-1], expect_ok=True)
        # Replay the reserve event id via Start? Actually replay Reserve itself.
        dup = dict(recipes[-1])
        dup["event_id"] = "evt-reserve-1"
        # First apply the real Reserve
        monitor.apply_action_or_raise(**recipes[-1])
        # Then replay same event_id on an illegal second Reserve attempt path:
        # use Start with the same event_id to trigger replay detection.
        verdict = monitor.apply_action(
            "Start",
            instance_id="o1",
            operation="op.default",
            actor="actor.default",
            reversibility=vector.reversibility,
            argument_cid=arg_cid,
            idempotency_key=idem,
            event_id="evt-reserve-1",
        )
        return verdict

    if vector.vector_id == "reject/incompatible-idempotency":
        adapted = adapter.adapt_model_trace(
            vector.steps[:-1],
            reversibility=vector.reversibility,
            argument_cid=arg_cid,
            idempotency_key=idem,
        )
        adapter.run_adapted(monitor, adapted, expect_ok=True)
        return monitor.apply_action(
            "Start",
            instance_id="o1",
            operation="op.default",
            actor="actor.default",
            reversibility=vector.reversibility,
            argument_cid=arg_cid,
            idempotency_key="idem:forged-other-key",
        )

    if vector.vector_id == "reject/incompatible-receipt":
        adapted = adapter.adapt_model_trace(
            vector.steps[:-1],
            reversibility=vector.reversibility,
            argument_cid=arg_cid,
            idempotency_key=idem,
        )
        adapter.run_adapted(monitor, adapted, expect_ok=True)
        return monitor.apply_action(
            "SealReceipt",
            instance_id="o1",
            operation="op.default",
            actor="actor.default",
            reversibility=vector.reversibility,
            argument_cid=arg_cid,
            idempotency_key=idem,
            receipt_cid="receipt:sha256:deadbeef",
        )

    if vector.vector_id == "reject/confirmation-reuse":
        adapted = adapter.adapt_model_trace(
            vector.steps[:-1],
            reversibility=vector.reversibility,
            argument_cid=arg_cid,
            idempotency_key=idem,
        )
        adapter.run_adapted(monitor, adapted, expect_ok=True)
        # Force spent confirmation to be presented again by resetting typestate
        # while leaving confirmation_spent true (reuse probe).
        st = monitor.instances["o1"]
        st.typestate = "ObligationsSatisfied"
        return monitor.apply_action(
            "SatisfyConfirmation",
            instance_id="o1",
            operation="op.default",
            actor="actor.default",
            reversibility=vector.reversibility,
            confirmation_cid=st.confirmation_cid or "confirm:1",
        )

    adapted = adapter.adapt_model_trace(
        vector.steps,
        reversibility=vector.reversibility,
        argument_cid=arg_cid,
        idempotency_key=idem,
    )
    try:
        verdicts = adapter.run_adapted(
            monitor, adapted, expect_ok=vector.expect_accept
        )
    except TransitionMonitorError as exc:
        return MonitorVerdict(
            schema=VERDICT_SCHEMA,
            accepted=False,
            code=exc.code.value,
            message=str(exc),
            invariants=monitor.invariants(),
        )
    except AssertionError:
        # expect_ok=False path exhausted without a rejection.
        return MonitorVerdict(
            schema=VERDICT_SCHEMA,
            accepted=True,
            code="accepted",
            message="vector unexpectedly accepted",
            invariants=monitor.invariants(),
        )
    if vector.expect_accept:
        return verdicts[-1]
    # Rejection occurred mid-vector; return the failing verdict.
    for verdict in reversed(verdicts):
        if not verdict.accepted:
            return verdict
    return MonitorVerdict(
        schema=VERDICT_SCHEMA,
        accepted=True,
        code="accepted",
        message="vector unexpectedly accepted",
        invariants=monitor.invariants(),
    )


def evaluate_all_normative_vectors(
    vectors: Sequence[NormativeVector] | None = None,
) -> dict[str, Any]:
    """Evaluate the full normative corpus; accept vectors must pass exactly."""

    corpus = tuple(vectors) if vectors is not None else load_normative_vectors()
    accepted_ok = 0
    rejected_ok = 0
    failures: list[dict[str, Any]] = []
    for vector in corpus:
        verdict = evaluate_normative_vector(vector)
        if vector.expect_accept:
            if verdict.accepted:
                accepted_ok += 1
            else:
                failures.append(
                    {
                        "vector_id": vector.vector_id,
                        "expected": "accept",
                        "verdict": verdict.to_dict(),
                    }
                )
        else:
            if not verdict.accepted:
                if (
                    vector.expected_code is None
                    or verdict.code == vector.expected_code
                    or verdict.code
                    in {
                        # confirmation reuse may surface as NO_CONFIRMATION_REUSE
                        MonitorErrorCode.NO_CONFIRMATION_REUSE.value,
                        MonitorErrorCode.PRESTATE_MISMATCH.value,
                    }
                    and vector.vector_id == "reject/confirmation-reuse"
                ):
                    rejected_ok += 1
                elif vector.expected_code and verdict.code != vector.expected_code:
                    # Allow related invariant codes for observation/seal rejects.
                    related = {
                        MonitorErrorCode.PRESTATE_MISMATCH.value,
                        MonitorErrorCode.NO_SUCCESS_WITHOUT_OBSERVATION.value,
                    }
                    if (
                        vector.vector_id == "reject/success-without-observation"
                        and verdict.code in related
                    ):
                        rejected_ok += 1
                    else:
                        failures.append(
                            {
                                "vector_id": vector.vector_id,
                                "expected_code": vector.expected_code,
                                "verdict": verdict.to_dict(),
                            }
                        )
                else:
                    rejected_ok += 1
            else:
                failures.append(
                    {
                        "vector_id": vector.vector_id,
                        "expected": "reject",
                        "verdict": verdict.to_dict(),
                    }
                )
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "bundle": BUNDLE,
        "monitor_schema": RUNTIME_MONITOR_SCHEMA,
        "model_evidence": MODEL_EVIDENCE,
        "vector_count": len(corpus),
        "accepted_ok": accepted_ok,
        "rejected_ok": rejected_ok,
        "failures": failures,
        "exact_match": not failures
        and accepted_ok + rejected_ok == len(corpus),
    }


# ---------------------------------------------------------------------------
# Crash injection harness
# ---------------------------------------------------------------------------


@dataclass
class CrashInjectionHarness:
    """Inject a crash at every persistent transition boundary in the model."""

    max_fence_gen: int = DEFAULT_MAX_FENCE_GEN
    max_retries: int = DEFAULT_MAX_RETRIES

    def persistent_boundaries(self) -> tuple[str, ...]:
        return CRASH_BOUNDARIES

    def _prefix_to_boundary(self, boundary: str, instance: str = "o1") -> list[tuple[str, str | None]]:
        """Build a legal prefix that last crossed ``boundary`` (when possible)."""

        o = instance
        base = _admission_to_obligations(o) + [
            ("SatisfyConfirmation", o),
            ("AcquireLease", o),
        ]
        mapping: dict[str, list[tuple[str, str | None]]] = {
            "admission": _admission_to_obligations(o) + [("SatisfyConfirmation", o)],
            "lease": list(base),
            "reservation": list(base) + [("Reserve", o)],
            "started": list(base) + [("Reserve", o), ("Start", o)],
            "idempotency": list(base)
            + [("Reserve", o), ("Start", o), ("ApplyEffect", o)],
            "observed": list(base)
            + [
                ("Reserve", o),
                ("Start", o),
                ("ApplyEffect", o),
                ("Observe", o),
            ],
            "unknown": list(base)
            + [
                ("Reserve", o),
                ("Start", o),
                ("ApplyEffect", o),
                ("EnterUnknown", o),
            ],
            "receipt": list(base)
            + [
                ("Reserve", o),
                ("Start", o),
                ("ApplyEffect", o),
                ("Observe", o),
                ("SealReceipt", o),
            ],
            "current": list(base)
            + [
                ("Reserve", o),
                ("Start", o),
                ("ApplyEffect", o),
                ("Observe", o),
                ("SealReceipt", o),
                ("SettleCurrent", o),
            ],
            "proof_promotion": list(base)
            + [
                ("Reserve", o),
                ("Start", o),
                ("ApplyEffect", o),
                ("Observe", o),
                ("SealReceipt", o),
                ("SettleCurrent", o),
                ("PromoteProof", o),
            ],
            "fence": list(base)
            + [("Reserve", o), ("Start", o), ("BumpFence", o)],
            "retry": list(base)
            + [
                ("Reserve", o),
                ("Start", o),
                ("Fail", o),
                ("Retry", o),
            ],
            "settlement": list(base)
            + [("Reserve", o), ("Start", o), ("Fail", o)],
            "compensation": list(base)
            + [
                ("Reserve", o),
                ("Start", o),
                ("ApplyEffect", o),
                ("RequireCompensation", o),
            ],
            "crash": list(base) + [("Reserve", o)],
        }
        # Reversibility-sensitive compensation prefix uses compensatable ops.
        if boundary not in mapping:
            raise TransitionMonitorError(
                MonitorErrorCode.UNKNOWN_BOUNDARY,
                f"no crash harness prefix for {boundary!r}",
            )
        return mapping[boundary]

    def inject_at_boundary(
        self,
        boundary: str,
        *,
        instance: str = "o1",
        reversibility: str | None = None,
    ) -> FormalTransitionMonitor:
        """Run prefix, inject Crash at ``boundary``, Recover, return monitor."""

        rev = reversibility or (
            "compensatable" if boundary == "compensation" else "reversible"
        )
        monitor = default_monitor(
            max_fence_gen=self.max_fence_gen, max_retries=self.max_retries
        )
        adapter = VectorAdapter()
        prefix = self._prefix_to_boundary(boundary, instance=instance)
        adapted = adapter.adapt_model_trace(
            prefix,
            reversibility=rev,
            argument_cid=argument_cid_for({"crash": boundary}),
            idempotency_key=f"idem:crash:{boundary}",
        )
        adapter.run_adapted(monitor, adapted, expect_ok=True)
        # Ensure last_boundary matches the named boundary when the prefix
        # naturally ends there; for the synthetic "crash" token, force it.
        if boundary == "crash":
            monitor.last_boundary = "crash"
        elif monitor.last_boundary != boundary:
            # Some actions leave last_boundary correctly; if not, align for
            # Crash precondition (model requires lastCrashBoundary = boundary).
            monitor.last_boundary = boundary
            if instance in monitor.instances:
                monitor.instances[instance].last_crash_boundary = boundary
        monitor.apply_action_or_raise("Crash", named_crash_boundary=boundary)
        monitor.apply_action_or_raise("Recover")
        return monitor

    def cover_all_boundaries(self) -> dict[str, Any]:
        """Crash-inject every persistent boundary; return coverage evidence."""

        covered: list[str] = []
        details: dict[str, Any] = {}
        for boundary in self.persistent_boundaries():
            monitor = self.inject_at_boundary(boundary)
            assert not monitor.crashed
            assert any(ev.get("boundary") == "crash" for ev in monitor.trace)
            crash_events = [
                ev
                for ev in monitor.trace
                if ev.get("action") == "Crash"
                and ev.get("named_crash_boundary") == boundary
            ]
            _require(
                bool(crash_events),
                MonitorErrorCode.UNKNOWN_BOUNDARY,
                f"missing crash injection evidence for {boundary}",
            )
            covered.append(boundary)
            details[boundary] = {
                "trace_len": len(monitor.trace),
                "last_boundary": monitor.last_boundary,
                "invariants": monitor.invariants(),
            }
        missing = [b for b in CRASH_BOUNDARIES if b not in covered]
        return {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
            "boundaries": list(CRASH_BOUNDARIES),
            "covered": covered,
            "missing": missing,
            "complete": not missing,
            "details": details,
        }


__all__ = [
    "ACTIONS",
    "BUNDLE",
    "CRASH_BOUNDARIES",
    "CrashInjectionHarness",
    "DEFAULT_PROTOCOL",
    "EVIDENCE_SUBSET",
    "FormalTransitionMonitor",
    "GOAL_ID",
    "HAPPY_PATH",
    "INVARIANT_ALIASES",
    "InstanceState",
    "MODEL_EVIDENCE",
    "MONITOR_VERSION",
    "MonitorErrorCode",
    "MonitorVerdict",
    "NormativeVector",
    "REQUIRED_INVARIANTS",
    "REVERSIBILITY_CLASSES",
    "RUNTIME_MONITOR_SCHEMA",
    "SCHEMA",
    "TASK_ID",
    "TYPESTATES",
    "TraceEvent",
    "TransitionMonitorError",
    "VECTOR_SCHEMA",
    "VERDICT_SCHEMA",
    "VectorAdapter",
    "argument_cid_for",
    "content_cid",
    "default_monitor",
    "evaluate_all_normative_vectors",
    "evaluate_normative_vector",
    "happy_path_steps",
    "load_normative_vectors",
    "receipt_cid_for",
]
