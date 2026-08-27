"""Federation-level fixed-point detection over an exact world snapshot.

A true fixed point is a conjunctive predicate: zero outstanding required work,
effects, proofs, merges, human reviews, recovering owners, and unprocessed
events; a fresh semantic root; and an event watermark that matches the bound
world snapshot. A quiet queue, completed board, or process exit cannot
establish completion. DuckLake never admits a fixed point. The current fence
wins.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    FederationWorldSnapshot,
    UnknownNormativeFieldError,
    _identifier,
    _integer,
)
from .recovery import RecoveryStore
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority

CONJUNCTIVE_PREDICATES = (
    "outstanding_required_work",
    "outstanding_effects",
    "open_proofs",
    "pending_merges",
    "pending_human_reviews",
    "recovering_owners",
    "unprocessed_events",
    "semantic_freshness",
    "event_watermark",
)
FALSE_COMPLETION_PREDICATES = (
    "quiet_queue",
    "board_completed",
    "claimed_complete",
    "process_exit",
)
FIXED_POINT_OUTCOMES = frozenset({"fixed_point"})


class FixedPointError(CausalGraphError):
    """Base typed federation fixed-point failure."""


class FixedPointAuthorityError(FederationAuthorityError, FixedPointError):
    """An attempt to declare fixed point from a quiet queue, board, or process exit."""


def refuse_ducklake_fixed_point_authority(receipt: Mapping[str, Any] | None) -> None:
    if not receipt:
        return
    if (
        receipt.get("authoritative") is True
        or receipt.get("schedules") is True
        or receipt.get("steals") is True
        or receipt.get("rebalances") is True
        or receipt.get("merges") is True
        or receipt.get("recovers") is True
        or receipt.get("fixed_points") is True
    ):
        raise FixedPointAuthorityError("DuckLake cannot admit a federation fixed point")


@dataclass(frozen=True)
class FixedPointObservation:
    """Exact counts and flags observed against one bound world snapshot."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/fixed-point-observation@1"
    )

    snapshot: FederationWorldSnapshot
    event_watermark: int
    fencing_epoch: int
    outstanding_required_work: int = 0
    outstanding_effects: int = 0
    open_proofs: int = 0
    pending_merges: int = 0
    pending_human_reviews: int = 0
    recovering_owners: int = 0
    unprocessed_events: int = 0
    semantic_fresh: bool = True
    board_quiet: bool = False
    board_completed: bool = False
    process_exited: bool = False
    claimed_complete: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, FederationWorldSnapshot):
            raise FederationContractError("fixed-point observation requires FederationWorldSnapshot")
        _integer(self.event_watermark, "event_watermark")
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        for name in (
            "outstanding_required_work",
            "outstanding_effects",
            "open_proofs",
            "pending_merges",
            "pending_human_reviews",
            "recovering_owners",
            "unprocessed_events",
        ):
            _integer(getattr(self, name), name)
        for name in (
            "semantic_fresh",
            "board_quiet",
            "board_completed",
            "process_exited",
            "claimed_complete",
        ):
            if type(getattr(self, name)) is not bool:
                raise FederationContractError(f"{name} must be boolean")


@dataclass(frozen=True)
class FixedPointDiagnostics:
    """Closed reasons a candidate observation is not a true fixed point."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/fixed-point-diagnostics@1"
    )

    failed_predicates: tuple[str, ...]
    event_watermark: int
    snapshot_watermark: int
    outstanding_required_work: int
    world_snapshot_ref: str

    def __post_init__(self) -> None:
        closed = set(CONJUNCTIVE_PREDICATES) | set(FALSE_COMPLETION_PREDICATES)
        failed = _identifier_tuple(self.failed_predicates, "failed_predicates")
        unknown = set(failed) - closed
        if unknown:
            raise FederationContractError(
                "fixed-point diagnostics contain values outside the closed vocabulary"
            )
        object.__setattr__(self, "failed_predicates", failed)
        _integer(self.event_watermark, "event_watermark")
        _integer(self.snapshot_watermark, "snapshot_watermark")
        _integer(self.outstanding_required_work, "outstanding_required_work")
        _identifier(self.world_snapshot_ref, "world_snapshot_ref")

    @property
    def at_fixed_point(self) -> bool:
        return not self.failed_predicates

    @property
    def false_quiet(self) -> bool:
        return bool(set(self.failed_predicates) & set(FALSE_COMPLETION_PREDICATES))


@dataclass(frozen=True)
class FixedPointReceipt:
    """Canonical evidence that the conjunctive predicate holds at one watermark."""

    LEGACY_SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/fixed-point-receipt@1"
    )
    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/fixed-point-receipt@2"
    )

    world_snapshot_ref: str
    event_watermark: int
    outstanding_required_work: int
    fencing_epoch: int
    outcome: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "world_snapshot_ref",
            _identifier(self.world_snapshot_ref, "world_snapshot_ref"),
        )
        _integer(self.event_watermark, "event_watermark")
        _integer(self.outstanding_required_work, "outstanding_required_work")
        if self.outstanding_required_work != 0:
            raise FixedPointAuthorityError(
                "fixed-point outcome cannot carry outstanding required work"
            )
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        outcome = _identifier(self.outcome, "outcome")
        if outcome not in FIXED_POINT_OUTCOMES:
            raise FixedPointAuthorityError("fixed-point outcome is outside its closed vocabulary")
        object.__setattr__(self, "outcome", outcome)
        refs = _identifier_tuple(self.evidence_refs, "evidence_refs")
        if not refs:
            raise FederationContractError("fixed-point receipt requires evidence")
        if len(refs) > 4_096:
            raise FederationContractError("fixed-point receipt evidence exceeds its bound")
        if len(refs) != len(set(refs)):
            raise FederationContractError("fixed-point receipt evidence contains duplicates")
        object.__setattr__(self, "evidence_refs", refs)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        """Return the one canonical JSON wire representation of this receipt."""

        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "world_snapshot_ref": self.world_snapshot_ref,
            "event_watermark": self.event_watermark,
            "outstanding_required_work": self.outstanding_required_work,
            "fencing_epoch": self.fencing_epoch,
            "outcome": self.outcome,
            "evidence_refs": list(self.evidence_refs),
        }
        if include_identity:
            payload["receipt_id"] = self.cid
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FixedPointReceipt:
        """Decode only the exact canonical wire contract and verify its identity."""

        if type(payload) is not dict:
            raise FederationContractError(
                "FixedPointReceipt payload must be an exact JSON object"
            )
        if any(type(key) is not str for key in payload):
            raise FederationContractError(
                "FixedPointReceipt field names must be exact strings"
            )
        fields = {
            "schema",
            "world_snapshot_ref",
            "event_watermark",
            "outstanding_required_work",
            "fencing_epoch",
            "outcome",
            "evidence_refs",
            "receipt_id",
        }
        unknown = set(payload) - fields
        if unknown:
            raise UnknownNormativeFieldError(
                f"FixedPointReceipt has unknown fields: {sorted(unknown)}"
            )
        missing = fields - set(payload)
        if missing:
            raise FederationContractError(
                f"FixedPointReceipt is missing fields: {sorted(missing)}"
            )
        schema = payload.get("schema")
        if type(schema) is not str:
            raise FederationContractError(
                "FixedPointReceipt.schema must be an exact string"
            )
        if schema == cls.LEGACY_SCHEMA:
            raise FixedPointAuthorityError(
                "legacy fixed-point receipt v1 is audit-only; re-observation is required"
            )
        if schema != cls.SCHEMA:
            raise FederationContractError(
                f"FixedPointReceipt.schema must equal {cls.SCHEMA!r}"
            )
        evidence_refs = payload.get("evidence_refs")
        if type(evidence_refs) is not list:
            raise FederationContractError(
                "FixedPointReceipt.evidence_refs must be a canonical array"
            )
        receipt = cls(
            world_snapshot_ref=payload.get("world_snapshot_ref"),
            event_watermark=payload.get("event_watermark"),
            outstanding_required_work=payload.get("outstanding_required_work"),
            fencing_epoch=payload.get("fencing_epoch"),
            outcome=payload.get("outcome"),
            evidence_refs=tuple(evidence_refs),
        )
        receipt_id = payload.get("receipt_id")
        if type(receipt_id) is not str or receipt_id != receipt.cid:
            raise FederationContractError("FixedPointReceipt identity mismatches its payload")
        return receipt

    @property
    def cid(self) -> str:
        return content_identity(self.to_dict(include_identity=False))


def _identifier_tuple(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise FederationContractError(f"{name} must be an array")
    return tuple(_identifier(item, name) for item in value)


def _failed_predicates(observation: FixedPointObservation) -> tuple[str, ...]:
    failed: list[str] = []
    counts = (
        ("outstanding_required_work", observation.outstanding_required_work),
        ("outstanding_effects", observation.outstanding_effects),
        ("open_proofs", observation.open_proofs),
        ("pending_merges", observation.pending_merges),
        ("pending_human_reviews", observation.pending_human_reviews),
        ("recovering_owners", observation.recovering_owners),
        ("unprocessed_events", observation.unprocessed_events),
    )
    for name, count in counts:
        if count != 0:
            failed.append(name)
    if observation.semantic_fresh is not True:
        failed.append("semantic_freshness")
    if observation.event_watermark != observation.snapshot.event_watermark:
        failed.append("event_watermark")
    if observation.board_completed:
        failed.append("board_completed")
    if observation.claimed_complete:
        failed.append("claimed_complete")
    if observation.process_exited and observation.claimed_complete:
        failed.append("process_exit")
    obligation_failed = tuple(
        name for name in failed if name in CONJUNCTIVE_PREDICATES or name == "claimed_complete"
    )
    if observation.board_quiet and obligation_failed:
        failed.append("quiet_queue")
    return tuple(dict.fromkeys(failed))


def diagnose_fixed_point(
    observation: FixedPointObservation,
    *,
    binding: FederationBinding,
    expected_fence: int,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> FixedPointDiagnostics:
    """Return closed diagnostics. False-quiet cases never become a fixed point."""

    if not isinstance(observation, FixedPointObservation):
        raise FederationContractError("fixed-point observation is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_fixed_point_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise FixedPointAuthorityError("retrieval cannot mint a federation fixed point")
    if expected_fence != observation.fencing_epoch:
        raise FixedPointAuthorityError("source fencing epoch is stale")
    if observation.snapshot.binding.tenant_id != binding.tenant_id:
        raise FixedPointAuthorityError("world snapshot is not bound to this federation")
    if observation.snapshot.semantic_roots != binding.semantic_state_roots:
        raise FixedPointAuthorityError("world snapshot semantic roots are not tree-bound")
    return FixedPointDiagnostics(
        failed_predicates=_failed_predicates(observation),
        event_watermark=observation.event_watermark,
        snapshot_watermark=observation.snapshot.event_watermark,
        outstanding_required_work=observation.outstanding_required_work,
        world_snapshot_ref=observation.snapshot.record_id,
    )


def detect_fixed_point(
    observation: FixedPointObservation,
    *,
    binding: FederationBinding,
    expected_fence: int,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> FixedPointReceipt:
    """Admit a fixed point only when the conjunctive predicate holds."""

    diagnostics = diagnose_fixed_point(
        observation,
        binding=binding,
        expected_fence=expected_fence,
        ducklake_receipt=ducklake_receipt,
    )
    if diagnostics.false_quiet:
        raise FixedPointAuthorityError(
            "quiet queue, board status, or process exit cannot complete federation work"
        )
    if not diagnostics.at_fixed_point:
        raise FixedPointAuthorityError(
            "conjunctive fixed-point predicate failed: " + ",".join(diagnostics.failed_predicates)
        )
    return FixedPointReceipt(
        world_snapshot_ref=observation.snapshot.record_id,
        event_watermark=observation.event_watermark,
        outstanding_required_work=0,
        fencing_epoch=observation.fencing_epoch,
        outcome="fixed_point",
        evidence_refs=(
            observation.snapshot.record_id,
            "watermark:" + str(observation.event_watermark),
        ),
    )


class FederationFixedPointDetector:
    """Diagnose false-quiet candidates and admit only a true conjunctive fixed point."""

    def diagnose(
        self,
        observation: FixedPointObservation,
        *,
        binding: FederationBinding,
        expected_fence: int,
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> FixedPointDiagnostics:
        return diagnose_fixed_point(
            observation,
            binding=binding,
            expected_fence=expected_fence,
            ducklake_receipt=ducklake_receipt,
        )

    def detect(
        self,
        observation: FixedPointObservation,
        *,
        binding: FederationBinding,
        expected_fence: int,
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> FixedPointReceipt:
        return detect_fixed_point(
            observation,
            binding=binding,
            expected_fence=expected_fence,
            ducklake_receipt=ducklake_receipt,
        )


def _fixed_point_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_fixed_point_receipt",
            """
            INSERT INTO federation_receipts (
                federation_receipt_id, tenant_id, federation_id, receipt_kind,
                federation_revision, control_plane_generation, event_watermark,
                issuer_id, content_ref, recorded_at
            ) VALUES (?, ?, ?, 'fixed_point', ?, ?, ?, ?, ?, ?)
            """,
            (
                "federation_receipt_id",
                "tenant_id",
                "federation_id",
                "federation_revision",
                "control_plane_generation",
                "event_watermark",
                "issuer_id",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_fixed_point_receipt",
            """
            SELECT federation_receipt_id, receipt_kind, event_watermark, content_ref
            FROM federation_receipts
            WHERE federation_receipt_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("federation_receipt_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class FixedPointStore(RecoveryStore):
    """Persist admitted fixed-point receipts through the sealed state owner."""

    INTERFACE = "FixedPointStore@1"

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
            raise FixedPointError("fixed-point store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise FixedPointError(
                "fixed-point store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _fixed_point_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise FixedPointError("fixed-point templates are absent from the sealed catalog")
        else:
            for template in _fixed_point_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_fixed_point(
        self,
        receipt: FixedPointReceipt,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(receipt, FixedPointReceipt):
            raise FederationContractError("fixed-point receipt is required")
        receipt_id = "federation-receipt:" + receipt.cid
        return self._commit_fact(
            operation="federation.fixed_point.record",
            fact_id=receipt_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((receipt_id, receipt.world_snapshot_ref, *receipt.evidence_refs))
            ),
            payload_ref=receipt.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_fixed_point(
                receipt,
                receipt_id=receipt_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                generation=binding.control_plane_generation,
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def load_fixed_point(
        self,
        *,
        receipt_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_fixed_point_receipt",
            {
                "federation_receipt_id": _identifier(receipt_id, "receipt_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise FixedPointError("fixed-point receipt is absent")
        return dict(rows[0])

    def _insert_fixed_point(
        self,
        receipt: FixedPointReceipt,
        *,
        receipt_id: str,
        federation_id: str,
        tenant_id: str,
        generation: int,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_fixed_point_receipt",
            {
                "federation_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "federation_revision": graph_revision,
                "control_plane_generation": generation,
                "event_watermark": receipt.event_watermark,
                "issuer_id": "fixed-point",
                "content_ref": receipt.cid,
                "recorded_at": recorded_at,
            },
        )


__all__ = (
    "CONJUNCTIVE_PREDICATES",
    "FALSE_COMPLETION_PREDICATES",
    "FIXED_POINT_OUTCOMES",
    "FederationFixedPointDetector",
    "FixedPointAuthorityError",
    "FixedPointDiagnostics",
    "FixedPointError",
    "FixedPointObservation",
    "FixedPointReceipt",
    "FixedPointStore",
    "detect_fixed_point",
    "diagnose_fixed_point",
    "refuse_ducklake_fixed_point_authority",
)
