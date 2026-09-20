"""SAWM-044 current-tree release, migration, and rollback evidence.

This report is not extra-gate completion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SemanticWorldRollbackTarget:
    extra_gate_generation: int
    store: str
    note: str = "keep live extra-gate; do not treat overlay files as completion"


@dataclass(frozen=True, slots=True)
class SemanticWorldMigrationReceipt:
    from_generation: int
    to_generation: int | None
    migrated: bool
    completion_authority: bool = False


@dataclass(frozen=True, slots=True)
class SemanticWorldReleaseReport:
    schema: str = "SemanticWorldReleaseReport@1"
    native_completed: int = 20
    remaining_todo: int = 25
    generation_published: bool = False
    completion_authority: bool = False
    cas_completed: bool = False
    released: bool = False
    safety_floor_violations: int = 0
    blockers: tuple[str, ...] = ()
    rollback: SemanticWorldRollbackTarget | None = None
    migration: SemanticWorldMigrationReceipt | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "native_completed": self.native_completed,
            "remaining_todo": self.remaining_todo,
            "generation_published": self.generation_published,
            "completion_authority": self.completion_authority,
            "cas_completed": self.cas_completed,
            "released": self.released,
            "safety_floor_violations": self.safety_floor_violations,
            "blockers": list(self.blockers),
            "rollback_target": None
            if self.rollback is None
            else {
                "extra_gate_generation": self.rollback.extra_gate_generation,
                "store": self.rollback.store,
                "note": self.rollback.note,
            },
            "migration": None
            if self.migration is None
            else {
                "from_generation": self.migration.from_generation,
                "to_generation": self.migration.to_generation,
                "migrated": self.migration.migrated,
                "completion_authority": self.migration.completion_authority,
            },
        }


def build_current_tree_release_report() -> SemanticWorldReleaseReport:
    from benchmarks.agent_supervisor.semantic_addressed_world_model.ablation import (
        run_frozen_semantic_world_ablation,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
        evaluate_program_world_reuse,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_service import (
        ProgramWorldService,
        _reuse_key,
    )

    ablation = run_frozen_semantic_world_ablation()
    safety = ablation["safety"]
    violations = int(safety.test_weakening) + int(safety.protected_path_hits)
    similarity = evaluate_program_world_reuse(
        _reuse_key({}),
        similarity_candidates=({"score": 0.99},),
    )
    index = ProgramWorldService().operation("index")
    blockers = [
        "live extra-gate cannot admit remaining todos",
        "overlay tests are not DuckDB completion evidence",
        "generation root was not published",
    ]
    if ablation.get("vector_backend") == "vector_backend_unavailable":
        blockers.append("vector_backend_unavailable")
    if index.get("reason_code") == "ann_index_unavailable":
        blockers.append("ann_index_unavailable")
    if similarity.ann_authoritative or similarity.admitted:
        blockers.append("similarity_reuse_was_not_closed")
    if violations:
        blockers.append("safety_floor_violations")
    released = (
        violations == 0
        and not similarity.admitted
        and False  # remaining todos and unpublished generation forbid release
    )
    report = SemanticWorldReleaseReport(
        released=released,
        safety_floor_violations=violations,
        blockers=tuple(blockers),
        rollback=SemanticWorldRollbackTarget(
            extra_gate_generation=48,
            store="semantic-addressed-world-model-v1/run-r2-m27",
        ),
        migration=SemanticWorldMigrationReceipt(
            from_generation=48,
            to_generation=None,
            migrated=False,
        ),
    )
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        mirror_work_record(
            catalog_kind="world_model",
            record_kind="world_release",
            record_ref="SemanticWorldReleaseReport@1",
            subject_kind="record_cid",
            subject_ref="SemanticWorldReleaseReport@1",
        )
    except Exception:
        pass
    return report
