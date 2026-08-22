"""Federation world snapshots over one tree-bound semantic root.

``SupervisorWorldSnapshot@1`` remains the semantic-state builder's report.  This
module admits the operational ``FederationWorldSnapshot`` used by CASF: event
watermark, task/claim/merge/proof refs, tree-bound semantic roots, and the
compiled causal frontier.  DuckLake projections are observed only and never
admit, schedule, or complete.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..semantic_state.world_snapshot_builder import (
    WorldSnapshotAdmissionError,
    project_casf_world_inputs,
    refuse_ducklake_world_authority,
)
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_frontier import CausalFrontierStore, CompiledFrontier
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    FederationWorldSnapshot,
    _identifier,
    _integer,
    _strings,
)
from .registry import _template


class WorldSnapshotError(CausalGraphError):
    """Base typed federation world-snapshot failure."""


class WorldSnapshotAuthorityError(FederationAuthorityError, WorldSnapshotError):
    """An attempt to admit a stale, unbound, or DuckLake-derived snapshot."""


def assemble_federation_world_snapshot(
    *,
    binding: FederationBinding,
    event_watermark: int,
    task_population_ref: str,
    claim_population_ref: str,
    merge_state_ref: str,
    proof_state_ref: str,
    causal_frontier_ref: str,
    semantic_roots: Sequence[str] | None = None,
    graph_revision: int | None = None,
    builder_result: Mapping[str, Any] | None = None,
    ducklake_receipt: Mapping[str, Any] | None = None,
    record_id: str = "world-snapshot:current",
    revision: int = 1,
) -> FederationWorldSnapshot:
    """Admit one tree-bound operational world snapshot."""

    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    try:
        refuse_ducklake_world_authority(ducklake_receipt)
    except WorldSnapshotAdmissionError as exc:
        raise WorldSnapshotAuthorityError(str(exc)) from exc
    roots = (
        _strings(semantic_roots, "semantic_roots", maximum=256, required=True)
        if semantic_roots is not None
        else binding.semantic_state_roots
    )
    if roots != binding.semantic_state_roots:
        raise WorldSnapshotAuthorityError(
            "semantic roots must remain tree-bound to the federation binding"
        )
    if (
        graph_revision is not None
        and int(graph_revision) != binding.causal_graph_revision
    ):
        raise WorldSnapshotAuthorityError("world snapshot graph revision is stale")
    _integer(event_watermark, "event_watermark")
    if builder_result is not None:
        projected = project_casf_world_inputs(builder_result)
        if projected["ducklake_authoritative"] is True:
            raise WorldSnapshotAuthorityError("DuckLake cannot admit a world snapshot")
        if projected["schedulable"] is not True:
            raise WorldSnapshotAuthorityError(
                "unschedulable supervisor snapshot cannot admit federation state"
            )
        semantic_root = str(projected["datasets_semantic_state_root"] or "")
        if semantic_root and semantic_root not in roots:
            raise WorldSnapshotAuthorityError(
                "builder semantic root is not bound to this federation tree"
            )
        task_ref = str(projected["task_population"] or "")
        if task_ref and task_ref != task_population_ref:
            raise WorldSnapshotAuthorityError(
                "task population disagrees with the admitted supervisor snapshot"
            )
    return FederationWorldSnapshot(
        record_id=_identifier(record_id, "record_id"),
        revision=revision,
        binding=replace(
            binding,
            causal_graph_revision=(
                binding.causal_graph_revision
                if graph_revision is None
                else int(graph_revision)
            ),
        ),
        event_watermark=event_watermark,
        task_population_ref=_identifier(task_population_ref, "task_population_ref"),
        claim_population_ref=_identifier(claim_population_ref, "claim_population_ref"),
        merge_state_ref=_identifier(merge_state_ref, "merge_state_ref"),
        proof_state_ref=_identifier(proof_state_ref, "proof_state_ref"),
        semantic_roots=roots,
        causal_frontier_ref=_identifier(causal_frontier_ref, "causal_frontier_ref"),
    )


def snapshot_from_frontier(
    compiled: CompiledFrontier,
    *,
    binding: FederationBinding,
    event_watermark: int,
    task_population_ref: str,
    claim_population_ref: str,
    merge_state_ref: str,
    proof_state_ref: str,
    **kwargs: Any,
) -> FederationWorldSnapshot:
    """Bind a compiled frontier into a federation world snapshot."""

    if not isinstance(compiled, CompiledFrontier):
        raise FederationContractError("compiled frontier is required")
    return assemble_federation_world_snapshot(
        binding=binding,
        event_watermark=event_watermark,
        task_population_ref=task_population_ref,
        claim_population_ref=claim_population_ref,
        merge_state_ref=merge_state_ref,
        proof_state_ref=proof_state_ref,
        causal_frontier_ref="frontier:" + compiled.cid,
        graph_revision=compiled.graph_revision,
        **kwargs,
    )


def _snapshot_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_world_snapshot_reference",
            """
            INSERT INTO world_snapshot_references (
                world_snapshot_reference_id, tenant_id, federation_id,
                repository_id, tree_id, control_plane_generation,
                causal_graph_revision, semantic_state_root, event_watermark,
                owner_id, source_root, content_ref, freshness_state, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "world_snapshot_reference_id",
                "tenant_id",
                "federation_id",
                "repository_id",
                "tree_id",
                "control_plane_generation",
                "causal_graph_revision",
                "semantic_state_root",
                "event_watermark",
                "owner_id",
                "source_root",
                "content_ref",
                "freshness_state",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_world_snapshot_reference",
            """
            SELECT world_snapshot_reference_id, semantic_state_root,
                   event_watermark, causal_graph_revision, content_ref,
                   freshness_state
            FROM world_snapshot_references
            WHERE world_snapshot_reference_id = ? AND tenant_id = ?
              AND federation_id = ?
            LIMIT 1
            """,
            (
                "world_snapshot_reference_id",
                "tenant_id",
                "federation_id",
            ),
            kind=StatementKind.QUERY,
        ),
    )


class WorldSnapshotStore(CausalFrontierStore):
    """Persist admitted federation world snapshots through the state owner."""

    INTERFACE = "WorldSnapshotStore@1"

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
            raise WorldSnapshotError("world snapshot store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise WorldSnapshotError(
                "world snapshot store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _snapshot_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise WorldSnapshotError(
                    "world snapshot templates are absent from the sealed catalog"
                )
        else:
            for template in _snapshot_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_snapshot(
        self,
        snapshot: FederationWorldSnapshot,
        *,
        federation_id: str,
        expected_graph_revision: int,
        idempotency_key: str,
        owner_id: str,
        source_root: str,
        freshness_state: str = "current",
    ) -> CausalGraphCommit:
        if not isinstance(snapshot, FederationWorldSnapshot):
            raise FederationContractError("snapshot must be a FederationWorldSnapshot")
        if freshness_state not in {"current", "stale", "unavailable"}:
            raise WorldSnapshotError("freshness_state is not closed")
        if snapshot.binding.causal_graph_revision != expected_graph_revision:
            raise WorldSnapshotAuthorityError("world snapshot graph revision is stale")
        return self._commit_fact(
            operation="federation.world.snapshot.record",
            fact_id=snapshot.record_id,
            federation_id=federation_id,
            binding=snapshot.binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(snapshot.record_id, snapshot.causal_frontier_ref),
            payload_ref=snapshot.cid,
            prepare_fact=lambda: self._prepare_snapshot(
                snapshot.record_id,
                tenant_id=snapshot.binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_snapshot(
                snapshot,
                federation_id=federation_id,
                graph_revision=revision,
                recorded_at=recorded_at,
                owner_id=owner_id,
                source_root=source_root,
                freshness_state=freshness_state,
            ),
        )

    def load_snapshot(
        self,
        *,
        snapshot_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_world_snapshot_reference",
            {
                "world_snapshot_reference_id": _identifier(snapshot_id, "snapshot_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise WorldSnapshotError("world snapshot is absent")
        return rows[0]

    def _prepare_snapshot(
        self, snapshot_id: str, *, tenant_id: str, federation_id: str
    ) -> None:
        existing = self._client.execute(
            "casf_select_world_snapshot_reference",
            {
                "world_snapshot_reference_id": snapshot_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise WorldSnapshotError("world snapshot identity is already bound")

    def _insert_snapshot(
        self,
        snapshot: FederationWorldSnapshot,
        *,
        federation_id: str,
        graph_revision: int,
        recorded_at: str,
        owner_id: str,
        source_root: str,
        freshness_state: str,
    ) -> None:
        del graph_revision
        self._client.execute(
            "casf_insert_world_snapshot_reference",
            {
                "world_snapshot_reference_id": snapshot.record_id,
                "tenant_id": snapshot.binding.tenant_id,
                "federation_id": federation_id,
                "repository_id": snapshot.binding.repository_ids[0],
                "tree_id": snapshot.binding.repository_tree_ids[0],
                "control_plane_generation": snapshot.binding.control_plane_generation,
                "causal_graph_revision": snapshot.binding.causal_graph_revision,
                "semantic_state_root": snapshot.semantic_roots[0],
                "event_watermark": snapshot.event_watermark,
                "owner_id": _identifier(owner_id, "owner_id"),
                "source_root": _identifier(source_root, "source_root"),
                "content_ref": snapshot.cid,
                "freshness_state": freshness_state,
                "recorded_at": recorded_at,
            },
        )


__all__ = (
    "WorldSnapshotAuthorityError",
    "WorldSnapshotError",
    "WorldSnapshotStore",
    "assemble_federation_world_snapshot",
    "snapshot_from_frontier",
)
