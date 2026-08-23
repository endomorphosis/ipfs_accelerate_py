"""Supervisor and subagent crash recovery with cursor replay and effect reconciliation.

A crashed owner cannot be revived by replaying its consumed identity. Recovery
mints a fresh subject, process-birth, and lease, increments the fencing epoch,
and replays from the durable event cursor. Unknown in-flight effects must be
reconciled before retry. Irreversible effects stay put. Stale fences, DuckLake,
and process-exit completion claims fail closed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    FederationLifecycleState,
    _identifier,
    _integer,
    _strings,
)
from .events import EventEffectClass
from .lifecycle import assert_transition
from .merge import MergeStore
from .parallel_frontier import IRREVERSIBLE_EFFECT_CLASSES
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority

IRREVERSIBLE_VALUES = frozenset(item.value for item in IRREVERSIBLE_EFFECT_CLASSES)
CLOSED_EFFECT_CLASSES = frozenset(item.value for item in EventEffectClass)
SUBJECT_KINDS = frozenset({"supervisor", "subagent"})
EFFECT_DISPOSITIONS = frozenset(
    {"observed", "absent", "irreversible", "unknown", "conflict"}
)
RECONCILED_DISPOSITIONS = frozenset({"observed", "absent", "irreversible"})
RECOVERY_OUTCOMES = frozenset({"recovered", "failed", "blocked"})
MAX_RECOVERY_EFFECTS = 10_000
MAX_RECOVERY_ATTEMPTS = 10_000


class RecoveryError(CausalGraphError):
    """Base typed crash-recovery failure."""


class RecoveryAuthorityError(FederationAuthorityError, RecoveryError):
    """An attempt to revive a stale identity, rewind a cursor, or retry unknown effects."""


def refuse_ducklake_recovery_authority(receipt: Mapping[str, Any] | None) -> None:
    if not receipt:
        return
    if (
        receipt.get("authoritative") is True
        or receipt.get("schedules") is True
        or receipt.get("steals") is True
        or receipt.get("rebalances") is True
        or receipt.get("merges") is True
        or receipt.get("recovers") is True
    ):
        raise RecoveryAuthorityError("DuckLake cannot admit crash recovery")


def _effect_class(value: EventEffectClass | str, name: str = "effect_class") -> str:
    if isinstance(value, EventEffectClass):
        return value.value
    text = _identifier(value, name)
    if text not in CLOSED_EFFECT_CLASSES:
        raise FederationContractError("effect_class is not closed")
    return text


def _lifecycle_state(value: FederationLifecycleState | str) -> FederationLifecycleState:
    if isinstance(value, FederationLifecycleState):
        return value
    try:
        return FederationLifecycleState(value)
    except ValueError as exc:
        raise FederationContractError("lifecycle_state is not closed") from exc


@dataclass(frozen=True)
class InFlightEffect:
    """One effect observed or still unknown at the crash boundary."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/in-flight-effect@1"
    )

    effect_id: str
    effect_class: str
    attempt_id: str
    task_id: str
    disposition: str = "unknown"
    observation_ref: str = ""

    def __post_init__(self) -> None:
        for name in ("effect_id", "attempt_id", "task_id"):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "effect_class", _effect_class(self.effect_class))
        disposition = _identifier(self.disposition, "disposition")
        if disposition not in EFFECT_DISPOSITIONS:
            raise FederationContractError("effect disposition is not closed")
        object.__setattr__(self, "disposition", disposition)
        _identifier(self.observation_ref, "observation_ref", required=False)

    @property
    def irreversible(self) -> bool:
        return self.effect_class in IRREVERSIBLE_VALUES or self.disposition == "irreversible"


@dataclass(frozen=True)
class EffectReconciliation:
    """Exact observation that disposes an unknown in-flight effect."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/effect-reconciliation@1"
    )

    effect_id: str
    disposition: str
    observation_ref: str
    evidence_ref: str

    def __post_init__(self) -> None:
        _identifier(self.effect_id, "effect_id")
        disposition = _identifier(self.disposition, "disposition")
        if disposition not in EFFECT_DISPOSITIONS:
            raise FederationContractError("effect disposition is not closed")
        object.__setattr__(self, "disposition", disposition)
        _identifier(self.observation_ref, "observation_ref")
        _identifier(self.evidence_ref, "evidence_ref")


@dataclass(frozen=True)
class CrashSnapshot:
    """Durable owner state retained across a crash, including the event cursor."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/crash-snapshot@1"
    )

    subject_kind: str
    subject_id: str
    process_birth_id: str
    lease_id: str
    fencing_epoch: int
    event_cursor: int
    lifecycle_state: str
    in_flight_effects: tuple[InFlightEffect, ...] = ()
    preserved_attempt_ids: tuple[str, ...] = ()
    checkpoint_ref: str = ""
    process_exited: bool = False
    claimed_complete: bool = False

    def __post_init__(self) -> None:
        kind = _identifier(self.subject_kind, "subject_kind")
        if kind not in SUBJECT_KINDS:
            raise FederationContractError("subject_kind is not closed")
        object.__setattr__(self, "subject_kind", kind)
        for name in ("subject_id", "process_birth_id", "lease_id"):
            _identifier(getattr(self, name), name)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _integer(self.event_cursor, "event_cursor")
        state = _lifecycle_state(self.lifecycle_state)
        object.__setattr__(self, "lifecycle_state", state.value)
        if not isinstance(self.in_flight_effects, tuple) or not all(
            isinstance(item, InFlightEffect) for item in self.in_flight_effects
        ):
            raise FederationContractError("in_flight_effects must be InFlightEffect records")
        if len(self.in_flight_effects) > MAX_RECOVERY_EFFECTS:
            raise FederationContractError("crash snapshot exceeds in-flight effect ceiling")
        seen = [item.effect_id for item in self.in_flight_effects]
        if len(seen) != len(set(seen)):
            raise FederationContractError("in_flight_effects contains duplicate identities")
        _strings(
            self.preserved_attempt_ids,
            "preserved_attempt_ids",
            maximum=MAX_RECOVERY_ATTEMPTS,
        )
        _identifier(self.checkpoint_ref, "checkpoint_ref", required=False)
        if type(self.process_exited) is not bool or type(self.claimed_complete) is not bool:
            raise FederationContractError("process_exited and claimed_complete must be boolean")


@dataclass(frozen=True)
class CompiledRecoveryPlan:
    """Fresh-identity recovery bound to the durable cursor and reconciled effects."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/compiled-recovery-plan@1"
    )

    subject_kind: str
    previous_subject_id: str
    recovered_subject_id: str
    previous_process_birth_id: str
    recovered_process_birth_id: str
    previous_lease_id: str
    recovered_lease_id: str
    previous_fencing_epoch: int
    fencing_epoch: int
    replay_cursor: int
    recovered_lifecycle: str
    preserved_attempt_ids: tuple[str, ...]
    reconciled_effect_ids: tuple[str, ...]
    replay_effect_ids: tuple[str, ...]
    checkpoint_ref: str

    def __post_init__(self) -> None:
        kind = _identifier(self.subject_kind, "subject_kind")
        if kind not in SUBJECT_KINDS:
            raise FederationContractError("subject_kind is not closed")
        object.__setattr__(self, "subject_kind", kind)
        for name in (
            "previous_subject_id",
            "recovered_subject_id",
            "previous_process_birth_id",
            "recovered_process_birth_id",
            "previous_lease_id",
            "recovered_lease_id",
        ):
            _identifier(getattr(self, name), name)
        if self.recovered_subject_id == self.previous_subject_id:
            raise RecoveryAuthorityError("crash recovery requires a fresh subject identity")
        if self.recovered_process_birth_id == self.previous_process_birth_id:
            raise RecoveryAuthorityError("crash recovery requires a fresh process birth")
        if self.recovered_lease_id == self.previous_lease_id:
            raise RecoveryAuthorityError("crash recovery requires a fresh lease")
        _integer(self.previous_fencing_epoch, "previous_fencing_epoch", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if self.fencing_epoch <= self.previous_fencing_epoch:
            raise RecoveryAuthorityError("recovery must increment the fencing epoch")
        _integer(self.replay_cursor, "replay_cursor")
        state = _lifecycle_state(self.recovered_lifecycle)
        if state is not FederationLifecycleState.RECOVERING:
            raise RecoveryAuthorityError("compiled recovery must enter RECOVERING")
        object.__setattr__(self, "recovered_lifecycle", state.value)
        _strings(
            self.preserved_attempt_ids,
            "preserved_attempt_ids",
            maximum=MAX_RECOVERY_ATTEMPTS,
        )
        _strings(
            self.reconciled_effect_ids,
            "reconciled_effect_ids",
            maximum=MAX_RECOVERY_EFFECTS,
        )
        _strings(self.replay_effect_ids, "replay_effect_ids", maximum=MAX_RECOVERY_EFFECTS)
        extra = set(self.replay_effect_ids) - set(self.reconciled_effect_ids)
        if extra:
            raise RecoveryAuthorityError("replay effects must be reconciled")
        _identifier(self.checkpoint_ref, "checkpoint_ref", required=False)

    @property
    def plan_id(self) -> str:
        return "recovery-plan:" + self.cid

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "subject_kind": self.subject_kind,
                "previous_subject_id": self.previous_subject_id,
                "recovered_subject_id": self.recovered_subject_id,
                "previous_fencing_epoch": self.previous_fencing_epoch,
                "fencing_epoch": self.fencing_epoch,
                "replay_cursor": self.replay_cursor,
                "recovered_process_birth_id": self.recovered_process_birth_id,
            }
        )


@dataclass(frozen=True)
class RecoveryReceipt:
    """Evidence that a crashed owner reconnected with replay and fence CAS."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/recovery-receipt@1"
    )

    recovery_plan_id: str
    subject_kind: str
    recovered_subject_id: str
    recovered_process_birth_id: str
    recovered_lease_id: str
    previous_fencing_epoch: int
    fencing_epoch: int
    replay_cursor: int
    outcome: str
    preserved_attempt_ids: tuple[str, ...]
    replay_effect_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "recovery_plan_id",
            "recovered_subject_id",
            "recovered_process_birth_id",
            "recovered_lease_id",
        ):
            _identifier(getattr(self, name), name)
        kind = _identifier(self.subject_kind, "subject_kind")
        if kind not in SUBJECT_KINDS:
            raise FederationContractError("subject_kind is not closed")
        object.__setattr__(self, "subject_kind", kind)
        _integer(self.previous_fencing_epoch, "previous_fencing_epoch", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if self.fencing_epoch <= self.previous_fencing_epoch:
            raise RecoveryAuthorityError("recovery must increment the fencing epoch")
        _integer(self.replay_cursor, "replay_cursor")
        outcome = _identifier(self.outcome, "outcome")
        if outcome not in RECOVERY_OUTCOMES:
            raise RecoveryAuthorityError("recovery outcome is outside its closed vocabulary")
        _strings(
            self.preserved_attempt_ids,
            "preserved_attempt_ids",
            maximum=MAX_RECOVERY_ATTEMPTS,
        )
        _strings(self.replay_effect_ids, "replay_effect_ids", maximum=MAX_RECOVERY_EFFECTS)

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "recovery_plan_id": self.recovery_plan_id,
                "recovered_subject_id": self.recovered_subject_id,
                "recovered_process_birth_id": self.recovered_process_birth_id,
                "fencing_epoch": self.fencing_epoch,
                "replay_cursor": self.replay_cursor,
                "outcome": self.outcome,
            }
        )


def _apply_reconciliations(
    effects: Sequence[InFlightEffect],
    reconciliations: Sequence[EffectReconciliation],
) -> tuple[InFlightEffect, ...]:
    if not isinstance(reconciliations, Sequence) or isinstance(reconciliations, (str, bytes)):
        raise FederationContractError("reconciliations must be an array")
    by_id = {item.effect_id: item for item in effects}
    seen: set[str] = set()
    for item in reconciliations:
        if not isinstance(item, EffectReconciliation):
            raise FederationContractError("reconciliations must be EffectReconciliation records")
        if item.effect_id in seen:
            raise RecoveryAuthorityError("effect reconciliation contains duplicate identities")
        seen.add(item.effect_id)
        if item.effect_id not in by_id:
            raise RecoveryAuthorityError("effect reconciliation does not match in-flight work")
        if item.disposition == "conflict":
            raise RecoveryAuthorityError("conflicting effect observations cannot be retried")
        if item.disposition == "unknown":
            raise RecoveryAuthorityError("unknown effects must reconcile before retry")
        current = by_id[item.effect_id]
        by_id[item.effect_id] = InFlightEffect(
            effect_id=current.effect_id,
            effect_class=current.effect_class,
            attempt_id=current.attempt_id,
            task_id=current.task_id,
            disposition=item.disposition,
            observation_ref=item.observation_ref,
        )
    return tuple(by_id[item.effect_id] for item in effects)


def compile_recovery(
    snapshot: CrashSnapshot,
    *,
    binding: FederationBinding,
    recovered_subject_id: str,
    recovered_process_birth_id: str,
    recovered_lease_id: str,
    expected_fence: int,
    reconciliations: Sequence[EffectReconciliation] = (),
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> CompiledRecoveryPlan:
    """Mint a fresh identity that replays from the durable cursor after reconciliation."""

    if not isinstance(snapshot, CrashSnapshot):
        raise FederationContractError("crash snapshot is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_recovery_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise RecoveryAuthorityError("retrieval cannot mint crash recovery")
    if snapshot.process_exited and snapshot.claimed_complete:
        raise RecoveryAuthorityError("process exit cannot complete federation work")
    if expected_fence != snapshot.fencing_epoch:
        raise RecoveryAuthorityError("source fencing epoch is stale")
    try:
        assert_transition(snapshot.lifecycle_state, FederationLifecycleState.RECOVERING)
    except FederationAuthorityError as exc:
        raise RecoveryAuthorityError(str(exc)) from exc
    recovered = _apply_reconciliations(snapshot.in_flight_effects, reconciliations)
    unknown = tuple(item.effect_id for item in recovered if item.disposition == "unknown")
    if unknown:
        raise RecoveryAuthorityError("unknown effects must reconcile before retry")
    replay = tuple(
        item.effect_id
        for item in recovered
        if item.disposition == "absent" and not item.irreversible
    )
    if any(item.irreversible and item.disposition == "absent" for item in recovered):
        raise RecoveryAuthorityError("active irreversible effects cannot replay")
    attempts = tuple(
        dict.fromkeys(
            (
                *snapshot.preserved_attempt_ids,
                *(item.attempt_id for item in snapshot.in_flight_effects),
            )
        )
    )
    return CompiledRecoveryPlan(
        subject_kind=snapshot.subject_kind,
        previous_subject_id=snapshot.subject_id,
        recovered_subject_id=recovered_subject_id,
        previous_process_birth_id=snapshot.process_birth_id,
        recovered_process_birth_id=recovered_process_birth_id,
        previous_lease_id=snapshot.lease_id,
        recovered_lease_id=recovered_lease_id,
        previous_fencing_epoch=snapshot.fencing_epoch,
        fencing_epoch=snapshot.fencing_epoch + 1,
        replay_cursor=snapshot.event_cursor,
        recovered_lifecycle=FederationLifecycleState.RECOVERING.value,
        preserved_attempt_ids=attempts,
        reconciled_effect_ids=tuple(
            item.effect_id for item in recovered if item.disposition in RECONCILED_DISPOSITIONS
        ),
        replay_effect_ids=replay,
        checkpoint_ref=snapshot.checkpoint_ref,
    )


def recover(
    plan: CompiledRecoveryPlan,
    *,
    binding: FederationBinding,
    expected_fence: int,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> RecoveryReceipt:
    """Activate a compiled recovery plan. The current fence wins."""

    if not isinstance(plan, CompiledRecoveryPlan):
        raise FederationContractError("compiled recovery plan is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_recovery_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise RecoveryAuthorityError("retrieval cannot mint crash recovery")
    if expected_fence != plan.previous_fencing_epoch:
        raise RecoveryAuthorityError("source fencing epoch is stale")
    return RecoveryReceipt(
        recovery_plan_id=plan.plan_id,
        subject_kind=plan.subject_kind,
        recovered_subject_id=plan.recovered_subject_id,
        recovered_process_birth_id=plan.recovered_process_birth_id,
        recovered_lease_id=plan.recovered_lease_id,
        previous_fencing_epoch=plan.previous_fencing_epoch,
        fencing_epoch=plan.fencing_epoch,
        replay_cursor=plan.replay_cursor,
        outcome="recovered",
        preserved_attempt_ids=plan.preserved_attempt_ids,
        replay_effect_ids=plan.replay_effect_ids,
    )


class FederationRecoveryCoordinator:
    """Reconnect crashed supervisors/subagents, reconcile effects, and replay."""

    def compile(
        self,
        snapshot: CrashSnapshot,
        *,
        binding: FederationBinding,
        recovered_subject_id: str,
        recovered_process_birth_id: str,
        recovered_lease_id: str,
        expected_fence: int,
        reconciliations: Sequence[EffectReconciliation] = (),
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> CompiledRecoveryPlan:
        return compile_recovery(
            snapshot,
            binding=binding,
            recovered_subject_id=recovered_subject_id,
            recovered_process_birth_id=recovered_process_birth_id,
            recovered_lease_id=recovered_lease_id,
            expected_fence=expected_fence,
            reconciliations=reconciliations,
            ducklake_receipt=ducklake_receipt,
        )

    def recover(
        self,
        plan: CompiledRecoveryPlan,
        *,
        binding: FederationBinding,
        expected_fence: int,
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> RecoveryReceipt:
        return recover(
            plan,
            binding=binding,
            expected_fence=expected_fence,
            ducklake_receipt=ducklake_receipt,
        )


def _recovery_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_recovery_action",
            """
            INSERT INTO recovery_actions (
                action_id, subject_kind, subject_id, task_cid, action_kind,
                decided_at, status, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "action_id",
                "subject_kind",
                "subject_id",
                "task_cid",
                "action_kind",
                "decided_at",
                "status",
                "body_json",
            ),
        ),
        _template(
            "casf_select_recovery_action",
            """
            SELECT action_id, subject_kind, subject_id, action_kind, status, body_json
            FROM recovery_actions
            WHERE action_id = ?
            LIMIT 1
            """,
            ("action_id",),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_fencing_epoch",
            """
            INSERT INTO fencing_epochs (
                fencing_epoch_id, tenant_id, federation_id, subject_kind,
                subject_id, lease_id, epoch, status, causation_event_id,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fencing_epoch_id",
                "tenant_id",
                "federation_id",
                "subject_kind",
                "subject_id",
                "lease_id",
                "epoch",
                "status",
                "causation_event_id",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_fencing_epoch",
            """
            SELECT fencing_epoch_id, subject_kind, subject_id, lease_id, epoch, status
            FROM fencing_epochs
            WHERE fencing_epoch_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("fencing_epoch_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_effect_observation",
            """
            INSERT INTO federation_effect_observations (
                effect_observation_id, tenant_id, federation_id,
                effect_reservation_id, task_cid, attempt_id, fencing_epoch,
                disposition, observation_ref, evidence_ref, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "effect_observation_id",
                "tenant_id",
                "federation_id",
                "effect_reservation_id",
                "task_cid",
                "attempt_id",
                "fencing_epoch",
                "disposition",
                "observation_ref",
                "evidence_ref",
                "observed_at",
            ),
        ),
        _template(
            "casf_select_effect_observation",
            """
            SELECT effect_observation_id, disposition, observation_ref, evidence_ref
            FROM federation_effect_observations
            WHERE effect_observation_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("effect_observation_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class RecoveryStore(MergeStore):
    """Persist recovery actions, fencing epochs, and effect observations."""

    INTERFACE = "RecoveryStore@1"

    def __init__(
        self,
        client: QuackStateClient,
        *,
        event_notifier: Callable[[int], None] | None = None,
        outbox_notifier: Callable[[int], None] | None = None,
        test_failure_hook: Callable[[str], None] | None = None,
        require_quack_authority: bool = False,
    ) -> None:
        if isinstance(client, (str, bytes, Path)):
            raise RecoveryError("recovery store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise RecoveryError(
                "recovery store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name for template in _recovery_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise RecoveryError("recovery templates are absent from the sealed catalog")
        else:
            for template in _recovery_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_recovery(
        self,
        plan: CompiledRecoveryPlan,
        receipt: RecoveryReceipt,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        reconciliations: Sequence[EffectReconciliation] = (),
    ) -> CausalGraphCommit:
        if not isinstance(plan, CompiledRecoveryPlan):
            raise FederationContractError("compiled recovery plan is required")
        if not isinstance(receipt, RecoveryReceipt):
            raise FederationContractError("recovery receipt is required")
        if receipt.recovery_plan_id != plan.plan_id:
            raise RecoveryAuthorityError("receipt plan identity differs from the compiled plan")
        action_id = "recovery-action:" + receipt.cid
        return self._commit_fact(
            operation="federation.recovery.record",
            fact_id=action_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((plan.plan_id, action_id, plan.recovered_subject_id))
            ),
            payload_ref=receipt.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_recovery(
                plan,
                receipt,
                action_id=action_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
                reconciliations=reconciliations,
            ),
        )

    def load_action(self, *, action_id: str) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_recovery_action",
            {"action_id": _identifier(action_id, "action_id")},
        )
        if len(rows) != 1:
            raise RecoveryError("recovery action is absent")
        return dict(rows[0])

    def load_epoch(
        self,
        *,
        epoch_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_fencing_epoch",
            {
                "fencing_epoch_id": _identifier(epoch_id, "epoch_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise RecoveryError("fencing epoch is absent")
        return dict(rows[0])

    def load_observation(
        self,
        *,
        observation_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_effect_observation",
            {
                "effect_observation_id": _identifier(observation_id, "observation_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise RecoveryError("effect observation is absent")
        return dict(rows[0])

    def _insert_recovery(
        self,
        plan: CompiledRecoveryPlan,
        receipt: RecoveryReceipt,
        *,
        action_id: str,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
        reconciliations: Sequence[EffectReconciliation],
    ) -> None:
        self._client.execute(
            "casf_insert_recovery_action",
            {
                "action_id": action_id,
                "subject_kind": plan.subject_kind,
                "subject_id": plan.recovered_subject_id,
                "task_cid": "",
                "action_kind": "reconnect",
                "decided_at": recorded_at,
                "status": receipt.outcome,
                "body_json": json.dumps(
                    {
                        "content_ref": receipt.cid,
                        "replay_cursor": plan.replay_cursor,
                        "previous_subject_id": plan.previous_subject_id,
                    },
                    separators=(",", ":"),
                ),
            },
        )
        epoch_id = "fencing-epoch:" + content_identity(
            {
                "subject_id": plan.recovered_subject_id,
                "epoch": plan.fencing_epoch,
                "lease_id": plan.recovered_lease_id,
            }
        )
        self._client.execute(
            "casf_insert_fencing_epoch",
            {
                "fencing_epoch_id": epoch_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "subject_kind": plan.subject_kind,
                "subject_id": plan.recovered_subject_id,
                "lease_id": plan.recovered_lease_id,
                "epoch": plan.fencing_epoch,
                "status": "active",
                "causation_event_id": "causation:recovery",
                "recorded_at": recorded_at,
            },
        )
        for item in reconciliations:
            self._client.execute(
                "casf_insert_effect_observation",
                {
                    "effect_observation_id": "effect-observation:"
                    + content_identity(
                        {
                            "effect_id": item.effect_id,
                            "disposition": item.disposition,
                            "observation_ref": item.observation_ref,
                        }
                    ),
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "effect_reservation_id": item.effect_id,
                    "task_cid": item.effect_id,
                    "attempt_id": item.evidence_ref,
                    "fencing_epoch": plan.fencing_epoch,
                    "disposition": item.disposition,
                    "observation_ref": item.observation_ref,
                    "evidence_ref": item.evidence_ref,
                    "observed_at": recorded_at,
                },
            )


__all__ = (
    "CrashSnapshot",
    "CompiledRecoveryPlan",
    "EffectReconciliation",
    "FederationRecoveryCoordinator",
    "InFlightEffect",
    "RECOVERY_OUTCOMES",
    "RecoveryAuthorityError",
    "RecoveryError",
    "RecoveryReceipt",
    "RecoveryStore",
    "compile_recovery",
    "recover",
    "refuse_ducklake_recovery_authority",
)
