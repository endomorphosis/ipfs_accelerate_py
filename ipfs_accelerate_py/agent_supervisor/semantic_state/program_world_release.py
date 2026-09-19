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
    return SemanticWorldReleaseReport(
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
