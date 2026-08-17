"""Evidence-driven, bounded refill orchestration.

This entrypoint deliberately does not own a board or a scheduler.  It turns a
small, current-tree observation into a deterministic refill decision and asks
the existing objective/refinery path to append work through its revision CAS.
Keeping those effects behind callbacks makes queue emptiness and stale task
status insufficient to either generate work or close a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


BOUNDED_RESIDUAL_REFILL_REQUIREMENT_ID = "prompt_v3_refill.BOUNDED_RESIDUAL_REFILL_REQUIREMENT_ID"
REFILL_RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/bounded-residual-refill-receipt@1"


class RefillTrigger(str, Enum):
    LOW_WATER = "low_water"
    DRAINED_OPEN_GOAL = "drained_open_goal"
    COMPLETION_REJECTED = "completion_rejected"
    ACTIONABLE_DRIFT = "actionable_drift"
    RETRY_EXHAUSTED = "retry_exhausted"
    ROLLOUT_THRESHOLD_MISSED = "rollout_threshold_missed"


class RefillDisposition(str, Enum):
    NO_REFILL = "no_refill"
    REFILLED = "refilled"
    REOPEN_CONVERGENCE = "reopen_convergence"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"
    CAS_CONFLICT = "cas_conflict"


@dataclass(frozen=True)
class RefillPolicy:
    """Immutable caps; refill output can never enlarge these values."""

    low_water_mark: int = 1
    max_findings_per_scan: int = 5
    max_new_work_per_epoch: int = 5
    max_refinement_depth: int = 3
    max_epochs: int = 8
    cooldown_epochs: int = 0
    max_unchanged_epochs: int = 2

    def __post_init__(self) -> None:
        for name in (
            "low_water_mark", "max_findings_per_scan", "max_new_work_per_epoch",
            "max_refinement_depth", "max_epochs", "max_unchanged_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.cooldown_epochs, bool) or self.cooldown_epochs < 0:
            raise ValueError("cooldown_epochs must be a non-negative integer")


@dataclass(frozen=True)
class RefillEpochCAS:
    """Fence supplied to the canonical task-source append operation."""

    plan_root_cid: str
    expected_revision: int
    epoch: int

    def __post_init__(self) -> None:
        if not self.plan_root_cid or self.expected_revision < 1 or self.epoch < 1:
            raise ValueError("refill CAS requires a plan root and positive revision/epoch")


@dataclass(frozen=True)
class CompletionAuthorityDecision:
    """Current-tree completion proof, never inferred from a drained queue."""

    root_evidence_complete: bool
    evidence_current: bool = True
    accepted_commits_reachable: bool = True
    final_forced_scan_clean: bool = False

    @property
    def authorized(self) -> bool:
        return bool(
            self.root_evidence_complete
            and self.evidence_current
            and self.accepted_commits_reachable
            and self.final_forced_scan_clean
        )


@dataclass(frozen=True)
class ResidualGap:
    """Smallest actionable unit emitted by the evidence evaluator."""

    goal_cid: str
    evidence_cid: str
    scope_cid: str
    lineage_goal_cids: tuple[str, ...]
    depth: int
    scheduler_metadata: Mapping[str, Any]
    kind: str = "task"

    @property
    def identity(self) -> str:
        payload = [self.goal_cid, self.evidence_cid, self.scope_cid]
        return "residual:sha256:" + hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def validate(self) -> None:
        if not self.goal_cid or not self.evidence_cid or not self.scope_cid:
            raise ValueError("residual gaps require goal, evidence, and scope identities")
        if not self.lineage_goal_cids or self.lineage_goal_cids[0] != self.goal_cid:
            raise ValueError("residual gap lineage must begin with its goal")
        if self.depth < 0:
            raise ValueError("residual gap depth cannot be negative")
        required = ("priority", "track", "parallel_lane", "resource_class")
        if any(not self.scheduler_metadata.get(name) for name in required):
            raise ValueError("residual gap is missing scheduler metadata")


@dataclass(frozen=True)
class ResidualEvidence:
    repository_tree_id: str
    gaps: tuple[ResidualGap, ...] = ()
    completion: CompletionAuthorityDecision = field(
        default_factory=lambda: CompletionAuthorityDecision(False, False, False, False)
    )


class ResidualEvidenceEvaluator(Protocol):
    def __call__(self, observation: "RefillObservation", *, force_final_scan: bool) -> ResidualEvidence: ...


class ProductionSelfImprovementHook(Protocol):
    def __call__(self, observation: "RefillObservation", evidence: ResidualEvidence) -> None: ...


@dataclass(frozen=True)
class RefillObservation:
    plan_root_cid: str
    revision: int
    ready_tasks: int = 0
    active_tasks: int = 0
    open_goals: int = 0
    validation_rejected: bool = False
    review_rejected: bool = False
    merge_rejected: bool = False
    stale_evidence: bool = False
    branch_only_completion: bool = False
    actionable_drift: bool = False
    retry_exhausted_with_refinement: bool = False
    rollout_threshold_missed: bool = False

    def __post_init__(self) -> None:
        if not self.plan_root_cid or self.revision < 1:
            raise ValueError("observation requires plan root and positive revision")
        if any(value < 0 for value in (self.ready_tasks, self.active_tasks, self.open_goals)):
            raise ValueError("work counts cannot be negative")


@dataclass(frozen=True)
class RefillDecision:
    disposition: RefillDisposition
    triggers: tuple[RefillTrigger, ...]
    epoch: int
    gap_identities: tuple[str, ...] = ()
    appended_count: int = 0
    reason: str = ""
    cas: RefillEpochCAS | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REFILL_RECEIPT_SCHEMA,
            "disposition": self.disposition.value,
            "triggers": [item.value for item in self.triggers],
            "epoch": self.epoch,
            "gap_identities": list(self.gap_identities),
            "appended_count": self.appended_count,
            "reason": self.reason,
            "cas": None if self.cas is None else {
                "plan_root_cid": self.cas.plan_root_cid,
                "expected_revision": self.cas.expected_revision,
                "epoch": self.cas.epoch,
            },
        }


AppendRefillWork = Callable[[Sequence[ResidualGap], RefillEpochCAS], bool]


def refill_triggers(observation: RefillObservation, policy: RefillPolicy) -> tuple[RefillTrigger, ...]:
    """Return the complete, stable trigger set in enum declaration order."""
    selected: list[RefillTrigger] = []
    if observation.ready_tasks + observation.active_tasks < policy.low_water_mark:
        selected.append(RefillTrigger.LOW_WATER)
    if observation.ready_tasks + observation.active_tasks == 0 and observation.open_goals:
        selected.append(RefillTrigger.DRAINED_OPEN_GOAL)
    if any((observation.validation_rejected, observation.review_rejected, observation.merge_rejected,
            observation.stale_evidence, observation.branch_only_completion)):
        selected.append(RefillTrigger.COMPLETION_REJECTED)
    if observation.actionable_drift:
        selected.append(RefillTrigger.ACTIONABLE_DRIFT)
    if observation.retry_exhausted_with_refinement:
        selected.append(RefillTrigger.RETRY_EXHAUSTED)
    if observation.rollout_threshold_missed:
        selected.append(RefillTrigger.ROLLOUT_THRESHOLD_MISSED)
    return tuple(selected)


class RefillController:
    """Stateful circuit breaker around current-tree residual evaluation."""

    def __init__(self, evaluator: ResidualEvidenceEvaluator, append: AppendRefillWork, *, policy: RefillPolicy | None = None,
                 production_hooks: Iterable[ProductionSelfImprovementHook] = ()) -> None:
        self.evaluator = evaluator
        self.append = append
        self.policy = policy or RefillPolicy()
        self.production_hooks = tuple(production_hooks)
        self._epoch = 0
        self._trigger_ticks = 0
        self._last_refill_tick: int | None = None
        self._last_gap_set: tuple[str, ...] = ()
        self._unchanged_epochs = 0
        self._seen_gap_ids: set[str] = set()

    def decide(self, observation: RefillObservation) -> RefillDecision:
        triggers = refill_triggers(observation, self.policy)
        if not triggers:
            return RefillDecision(RefillDisposition.NO_REFILL, (), self._epoch, reason="no_refill_trigger")
        self._trigger_ticks += 1
        routine_triggers = {RefillTrigger.LOW_WATER, RefillTrigger.DRAINED_OPEN_GOAL}
        if (
            self._last_refill_tick is not None
            and set(triggers).issubset(routine_triggers)
            and self._trigger_ticks - self._last_refill_tick <= self.policy.cooldown_epochs
        ):
            return RefillDecision(RefillDisposition.NO_REFILL, triggers, self._epoch, reason="cooldown")
        if self._epoch >= self.policy.max_epochs:
            return RefillDecision(RefillDisposition.BLOCKED, triggers, self._epoch, reason="epoch_budget_exhausted")

        evidence = self.evaluator(observation, force_final_scan=True)
        if not evidence.repository_tree_id:
            return RefillDecision(RefillDisposition.BLOCKED, triggers, self._epoch, reason="missing_current_tree_evidence")
        for hook in self.production_hooks:
            hook(observation, evidence)

        # A current-tree rejection (including branch-only or stale evidence)
        # must reopen convergence even if there is no immediately derivable task.
        if observation.branch_only_completion or observation.stale_evidence or not evidence.completion.evidence_current or not evidence.completion.accepted_commits_reachable:
            return RefillDecision(RefillDisposition.REOPEN_CONVERGENCE, triggers, self._epoch, reason="completion_not_authorized")

        candidates: list[ResidualGap] = []
        selected_ids: set[str] = set()
        # Evaluators may aggregate independent scanners.  Canonical identity
        # ordering and local deduplication keep their arrival order from
        # changing the first bounded batch.
        for gap in sorted(evidence.gaps, key=lambda item: item.identity)[: self.policy.max_findings_per_scan]:
            gap.validate()
            if (
                gap.depth >= self.policy.max_refinement_depth
                or gap.identity in self._seen_gap_ids
                or gap.identity in selected_ids
            ):
                continue
            candidates.append(gap)
            selected_ids.add(gap.identity)
            if len(candidates) == self.policy.max_new_work_per_epoch:
                break
        identities = tuple(gap.identity for gap in candidates)
        if identities == self._last_gap_set:
            self._unchanged_epochs += 1
        else:
            self._unchanged_epochs = 0
        self._last_gap_set = identities
        if self._unchanged_epochs >= self.policy.max_unchanged_epochs:
            return RefillDecision(RefillDisposition.BLOCKED, triggers, self._epoch, identities, reason="unchanged_residual_circuit_breaker")
        if not candidates:
            disposition = RefillDisposition.NO_REFILL if evidence.completion.authorized else RefillDisposition.BLOCKED
            reason = "final_scan_evidence_complete" if disposition is RefillDisposition.NO_REFILL else "no_novel_actionable_residual"
            return RefillDecision(disposition, triggers, self._epoch, reason=reason)

        epoch = self._epoch + 1
        cas = RefillEpochCAS(observation.plan_root_cid, observation.revision, epoch)
        if not self.append(tuple(candidates), cas):
            return RefillDecision(RefillDisposition.CAS_CONFLICT, triggers, self._epoch, identities, reason="revision_cas_rejected", cas=cas)
        self._epoch = epoch
        self._last_refill_tick = self._trigger_ticks
        self._seen_gap_ids.update(identities)
        return RefillDecision(RefillDisposition.REFILLED, triggers, epoch, identities, len(candidates), cas=cas)



# ---------------------------------------------------------------------------
# ASE3-021 production durable refill runtime (dormant until ASE3-026).
# ---------------------------------------------------------------------------

# ASE3-021 modules are imported lazily inside ProductionRefillRuntime to
# avoid import cycles with refill_adapters.


@dataclass(frozen=True)
class ProductionRefillRuntimeReceipt:
    disposition: str
    phase: str
    logical_attempt_id: str
    epoch: int
    dormant: bool
    triggers: tuple[str, ...] = ()
    gap_identities: tuple[str, ...] = ()
    cursor_cid: str = ""
    append_receipt_cid: str = ""
    plan_invalidation_cid: str = ""
    recompile_cid: str = ""
    dispatch_cid: str = ""
    reason: str = ""
    winner: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "phase": self.phase,
            "logical_attempt_id": self.logical_attempt_id,
            "epoch": self.epoch,
            "dormant": self.dormant,
            "triggers": list(self.triggers),
            "gap_identities": list(self.gap_identities),
            "cursor_cid": self.cursor_cid,
            "append_receipt_cid": self.append_receipt_cid,
            "plan_invalidation_cid": self.plan_invalidation_cid,
            "recompile_cid": self.recompile_cid,
            "dispatch_cid": self.dispatch_cid,
            "reason": self.reason,
            "winner": self.winner,
        }


class ProductionRefillRuntime:
    """Drive the durable refill saga against production adapters.

    The path remains dormant unless ``policy.activation_authorized`` is true
    (ASE3-026). When dormant, evaluation still records that no effect ran.
    """

    def __init__(
        self,
        *,
        store: RefillStore,
        policy: SignedRefillPolicy,
        evaluator: ResidualEvidenceEvaluator,
        event_adapter: Any = None,
        controller_policy: RefillPolicy | None = None,
    ) -> None:
        from .refill_event_adapter import ProductionRefillEventAdapter
        from .refill_store import RefillStore, SignedRefillPolicy

        if not isinstance(store, RefillStore):
            raise TypeError("store must be a RefillStore")
        if not isinstance(policy, SignedRefillPolicy):
            raise TypeError("policy must be a SignedRefillPolicy")
        self.store = store
        self.policy = policy
        self.evaluator = evaluator
        self.event_adapter = event_adapter or ProductionRefillEventAdapter()
        self.controller_policy = controller_policy or RefillPolicy(
            max_epochs=policy.max_epochs,
            max_new_work_per_epoch=policy.max_new_work_per_epoch,
            max_unchanged_epochs=policy.max_unchanged_epochs,
        )

    def run_once(
        self,
        observation: RefillObservation,
        *,
        tree_id: str,
        logical_attempt_id: str | None = None,
        now_ms: int | None = None,
        phase_budget_ms: int = 60_000,
    ) -> ProductionRefillRuntimeReceipt:
        """Execute or adopt one durable refill saga attempt."""

        import time as _time

        from .refill_adapters import (
            dispatch_identity,
            invalidate_active_plan,
            recompile_plan_identity,
        )
        from .refill_store import (
            REFILL_APPEND_RECEIPT_SCHEMA,
            DurableRefillState,
            RefillAppendReceipt,
            RefillSagaPhase,
        )

        clock = int(now_ms if now_ms is not None else int(_time.time() * 1000))
        triggers = refill_triggers(observation, self.controller_policy)
        attempt_id = logical_attempt_id or (
            f"refill:{observation.plan_root_cid}:r{observation.revision}:e"
            f"{self.store.load_state(observation.plan_root_cid).epoch + 1 if self.store.load_state(observation.plan_root_cid) else 1}"
        )

        if not self.policy.activation_authorized:
            return ProductionRefillRuntimeReceipt(
                disposition="dormant",
                phase="",
                logical_attempt_id=attempt_id,
                epoch=0,
                dormant=True,
                triggers=tuple(item.value for item in triggers),
                reason="awaiting_ase3_026_activation_authorization",
            )

        # Exact attempt adoption/resume is authoritative over residual re-eval.
        if logical_attempt_id is not None:
            existing_cursor = self.store.load_cursor(logical_attempt_id)
            if existing_cursor is not None:
                return self._resume(
                    existing_cursor,
                    tree_id=tree_id,
                    now_ms=clock,
                    phase_budget_ms=phase_budget_ms,
                )

        state = self.store.load_state(observation.plan_root_cid)
        if state is None:
            state = DurableRefillState(
                schema="ipfs_accelerate_py/agent-supervisor/durable-refill-state@1",
                plan_root_cid=observation.plan_root_cid,
                tree_id=tree_id,
                activation_authorized=True,
            )
        if state.tree_id and state.tree_id != tree_id:
            return ProductionRefillRuntimeReceipt(
                disposition="blocked",
                phase="",
                logical_attempt_id=attempt_id,
                epoch=state.epoch,
                dormant=False,
                reason="tree_id_mismatch",
            )

        # Adopt incomplete cursor if present.
        if state.active_cursor is not None:
            cursor = self.store.load_cursor(state.active_cursor.logical_attempt_id)
            if cursor is not None and cursor.phase not in {
                RefillSagaPhase.ADOPTED.value,
                RefillSagaPhase.EXHAUSTED.value,
            }:
                return self._resume(cursor, tree_id=tree_id, now_ms=clock, phase_budget_ms=phase_budget_ms)

        if not triggers:
            return ProductionRefillRuntimeReceipt(
                disposition="no_refill",
                phase="",
                logical_attempt_id=attempt_id,
                epoch=state.epoch,
                dormant=False,
                reason="no_refill_trigger",
            )

        epoch = state.epoch + 1
        if epoch > self.policy.max_epochs:
            return ProductionRefillRuntimeReceipt(
                disposition="blocked",
                phase="",
                logical_attempt_id=attempt_id,
                epoch=state.epoch,
                dormant=False,
                triggers=tuple(item.value for item in triggers),
                reason="epoch_budget_exhausted",
            )

        evidence = self.evaluator(observation, force_final_scan=True)
        if evidence.repository_tree_id != tree_id:
            return ProductionRefillRuntimeReceipt(
                disposition="blocked",
                phase="",
                logical_attempt_id=attempt_id,
                epoch=state.epoch,
                dormant=False,
                reason="missing_or_mismatched_current_tree_evidence",
            )

        candidates: list[ResidualGap] = []
        selected: set[str] = set()
        seen = set(state.seen_gap_ids)
        for gap in sorted(evidence.gaps, key=lambda item: item.identity)[
            : self.controller_policy.max_findings_per_scan
        ]:
            gap.validate()
            if gap.identity in seen or gap.identity in selected:
                continue
            if gap.depth >= self.controller_policy.max_refinement_depth:
                continue
            candidates.append(gap)
            selected.add(gap.identity)
            if len(candidates) >= self.policy.max_new_work_per_epoch:
                break
        identities = tuple(gap.identity for gap in candidates)
        if not candidates:
            if evidence.completion.authorized:
                return ProductionRefillRuntimeReceipt(
                    disposition="no_refill",
                    phase="",
                    logical_attempt_id=attempt_id,
                    epoch=state.epoch,
                    dormant=False,
                    triggers=tuple(item.value for item in triggers),
                    reason="final_scan_evidence_complete",
                )
            # Exhaust unchanged residuals
            if identities == state.last_gap_set:
                state.unchanged_epochs += 1
            if state.unchanged_epochs >= self.policy.max_unchanged_epochs:
                cursor, created, _ = self.store.begin_or_adopt(
                    logical_attempt_id=attempt_id,
                    plan_root_cid=observation.plan_root_cid,
                    tree_id=tree_id,
                    epoch=epoch,
                    gap_identities=(),
                    phase_budget_ms=phase_budget_ms,
                    now_ms=clock,
                    activation_authorized=True,
                )
                cursor = self.store.advance(
                    attempt_id,
                    fence_token=cursor.fence_token,
                    next_phase=RefillSagaPhase.EXHAUSTED.value,
                    tree_id=tree_id,
                    now_ms=clock,
                    phase_budget_ms=phase_budget_ms,
                )
                state.epoch = epoch
                state.active_cursor = cursor
                self.store.save_state(state)
                return ProductionRefillRuntimeReceipt(
                    disposition="exhausted",
                    phase=cursor.phase,
                    logical_attempt_id=attempt_id,
                    epoch=epoch,
                    dormant=False,
                    triggers=tuple(item.value for item in triggers),
                    cursor_cid=cursor.phase_cid,
                    reason="unchanged_residual_circuit_breaker",
                )
            return ProductionRefillRuntimeReceipt(
                disposition="blocked",
                phase="",
                logical_attempt_id=attempt_id,
                epoch=state.epoch,
                dormant=False,
                reason="no_novel_actionable_residual",
            )

        # Full saga path.
        cursor, created, adoption = self.store.begin_or_adopt(
            logical_attempt_id=attempt_id,
            plan_root_cid=observation.plan_root_cid,
            tree_id=tree_id,
            epoch=epoch,
            gap_identities=identities,
            phase_budget_ms=phase_budget_ms,
            now_ms=clock,
            activation_authorized=True,
        )
        if not created and adoption is not None:
            return ProductionRefillRuntimeReceipt(
                disposition="adopted",
                phase=adoption.phase,
                logical_attempt_id=attempt_id,
                epoch=adoption.epoch,
                dormant=False,
                winner=False,
                append_receipt_cid=adoption.append_receipt_cid,
                dispatch_cid=adoption.dispatch_cid,
                reason="adopted_existing_terminal",
            )
        if not created:
            return self._resume(cursor, tree_id=tree_id, now_ms=clock, phase_budget_ms=phase_budget_ms)

        fence = cursor.fence_token
        # APPEND_RESERVED
        reservation_id = "sha256:" + hashlib.sha256(
            f"reserve:{attempt_id}:{epoch}".encode()
        ).hexdigest()
        cursor = self.store.advance(
            attempt_id,
            fence_token=fence,
            next_phase=RefillSagaPhase.APPEND_RESERVED.value,
            tree_id=tree_id,
            now_ms=clock,
            phase_budget_ms=phase_budget_ms,
            reservation_id=reservation_id,
            gap_identities=identities,
        )
        # APPENDED
        append_receipt = RefillAppendReceipt(
            schema=REFILL_APPEND_RECEIPT_SCHEMA,
            logical_attempt_id=attempt_id,
            plan_root_cid=observation.plan_root_cid,
            tree_id=tree_id,
            epoch=epoch,
            gap_identities=identities,
            expected_revision=observation.revision,
            append_cid="sha256:" + hashlib.sha256(
                json.dumps(list(identities), separators=(",", ":")).encode()
            ).hexdigest(),
            created_at_ms=clock,
        )
        cursor = self.store.advance(
            attempt_id,
            fence_token=fence,
            next_phase=RefillSagaPhase.APPENDED.value,
            tree_id=tree_id,
            now_ms=clock,
            phase_budget_ms=phase_budget_ms,
            append_receipt_cid=append_receipt.content_id,
        )
        # PLAN_INVALIDATED
        invalidation = invalidate_active_plan(
            logical_attempt_id=attempt_id,
            plan_root_cid=observation.plan_root_cid,
            previous_revision=observation.revision,
            now_ms=clock,
        )
        cursor = self.store.advance(
            attempt_id,
            fence_token=fence,
            next_phase=RefillSagaPhase.PLAN_INVALIDATED.value,
            tree_id=tree_id,
            now_ms=clock,
            phase_budget_ms=phase_budget_ms,
            plan_invalidation_cid=invalidation.content_id,
        )
        # RECOMPILED
        recompile_cid = recompile_plan_identity(
            plan_root_cid=observation.plan_root_cid,
            tree_id=tree_id,
            epoch=epoch,
            gap_identities=identities,
        )
        cursor = self.store.advance(
            attempt_id,
            fence_token=fence,
            next_phase=RefillSagaPhase.RECOMPILED.value,
            tree_id=tree_id,
            now_ms=clock,
            phase_budget_ms=phase_budget_ms,
            recompile_cid=recompile_cid,
        )
        # DISPATCHED
        dispatch_cid = dispatch_identity(
            recompile_cid=recompile_cid,
            plan_root_cid=observation.plan_root_cid,
            epoch=epoch,
        )
        cursor = self.store.advance(
            attempt_id,
            fence_token=fence,
            next_phase=RefillSagaPhase.DISPATCHED.value,
            tree_id=tree_id,
            now_ms=clock,
            phase_budget_ms=phase_budget_ms,
            dispatch_cid=dispatch_cid,
        )
        adoption = self.store.adopt_terminal(attempt_id, now_ms=clock)

        state.epoch = epoch
        state.tree_id = tree_id
        state.last_gap_set = identities
        state.seen_gap_ids = tuple(sorted(set(state.seen_gap_ids) | set(identities)))
        state.unchanged_epochs = 0
        state.activation_authorized = True
        state.active_cursor = self.store.load_cursor(attempt_id)
        history = list(state.history)
        history.append(
            {
                "epoch": epoch,
                "attempt_id": attempt_id,
                "append_receipt_cid": append_receipt.content_id,
                "dispatch_cid": dispatch_cid,
            }
        )
        state.history = tuple(history[-32:])
        self.store.save_state(state)

        return ProductionRefillRuntimeReceipt(
            disposition="refilled",
            phase=adoption.phase,
            logical_attempt_id=attempt_id,
            epoch=epoch,
            dormant=False,
            triggers=tuple(item.value for item in triggers),
            gap_identities=identities,
            cursor_cid=cursor.phase_cid,
            append_receipt_cid=append_receipt.content_id,
            plan_invalidation_cid=invalidation.content_id,
            recompile_cid=recompile_cid,
            dispatch_cid=dispatch_cid,
            reason="durable_saga_completed",
            winner=adoption.winner,
        )

    def _resume(
        self,
        cursor,
        *,
        tree_id: str,
        now_ms: int,
        phase_budget_ms: int,
    ) -> ProductionRefillRuntimeReceipt:
        """Resume from the first incomplete phase without replaying effects."""

        from .refill_store import RefillSagaPhase

        phase = RefillSagaPhase(cursor.phase)
        if phase is RefillSagaPhase.DISPATCHED:
            adoption = self.store.adopt_terminal(
                cursor.logical_attempt_id, now_ms=now_ms
            )
            return ProductionRefillRuntimeReceipt(
                disposition="adopted",
                phase=adoption.phase,
                logical_attempt_id=cursor.logical_attempt_id,
                epoch=cursor.epoch,
                dormant=False,
                winner=adoption.winner,
                append_receipt_cid=cursor.append_receipt_cid,
                dispatch_cid=cursor.dispatch_cid,
                reason="resumed_dispatch_to_adopted",
            )
        if phase is RefillSagaPhase.ADOPTED:
            return ProductionRefillRuntimeReceipt(
                disposition="adopted",
                phase=cursor.phase,
                logical_attempt_id=cursor.logical_attempt_id,
                epoch=cursor.epoch,
                dormant=False,
                winner=False,
                cursor_cid=cursor.phase_cid,
                append_receipt_cid=cursor.append_receipt_cid,
                plan_invalidation_cid=cursor.plan_invalidation_cid,
                recompile_cid=cursor.recompile_cid,
                dispatch_cid=cursor.dispatch_cid,
                gap_identities=cursor.gap_identities,
                reason="adopted_existing_terminal",
            )
        if phase is RefillSagaPhase.EXHAUSTED:
            return ProductionRefillRuntimeReceipt(
                disposition="exhausted",
                phase=cursor.phase,
                logical_attempt_id=cursor.logical_attempt_id,
                epoch=cursor.epoch,
                dormant=False,
                winner=False,
                cursor_cid=cursor.phase_cid,
                reason="adopted_exhausted",
            )
        # For non-terminal incomplete phases, adopt the existing reservation
        # without re-entering provider/append effects.
        return ProductionRefillRuntimeReceipt(
            disposition="adopted",
            phase=cursor.phase,
            logical_attempt_id=cursor.logical_attempt_id,
            epoch=cursor.epoch,
            dormant=False,
            winner=False,
            cursor_cid=cursor.phase_cid,
            append_receipt_cid=cursor.append_receipt_cid,
            plan_invalidation_cid=cursor.plan_invalidation_cid,
            recompile_cid=cursor.recompile_cid,
            dispatch_cid=cursor.dispatch_cid,
            gap_identities=cursor.gap_identities,
            reason=f"resumed_incomplete_phase_{cursor.phase}",
        )



__all__ = (
    "BOUNDED_RESIDUAL_REFILL_REQUIREMENT_ID", "REFILL_RECEIPT_SCHEMA", "AppendRefillWork",
    "CompletionAuthorityDecision", "ProductionSelfImprovementHook", "RefillController",
    "RefillDecision", "RefillDisposition", "RefillEpochCAS", "RefillObservation", "RefillPolicy",
    "RefillTrigger", "ResidualEvidence", "ResidualEvidenceEvaluator", "ResidualGap", "refill_triggers",
)
