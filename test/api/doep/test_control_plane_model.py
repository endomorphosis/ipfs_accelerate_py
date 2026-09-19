"""DOEP-046 model-based and temporal invariants for the control plane."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


class ControlPlaneModelError(AssertionError):
    """A temporal or model invariant failed."""


@dataclass
class ControlPlaneModel:
    """Minimal generation/status model. Completions are never forged here."""

    generation: int = 1
    tasks: dict[str, str] = field(default_factory=dict)
    owner_live: bool = True

    def restart_same_generation(self) -> None:
        if not self.owner_live:
            raise ControlPlaneModelError("cannot restart a lost owner in place")
        self.owner_live = True

    def lose_owner(self) -> None:
        self.owner_live = False

    def successor(self) -> None:
        if self.owner_live:
            raise ControlPlaneModelError("live owner cannot mint a successor")
        self.generation += 1
        self.owner_live = True

    def set_status(self, task_id: str, status: str) -> None:
        previous = self.tasks.get(task_id)
        if previous == "completed" and status == "completed":
            return
        if previous == "completed" and status not in {"todo", "retrying"}:
            raise ControlPlaneModelError("completed work cannot be rewritten as success")
        if status == "completed" and previous not in {None, "todo", "retrying", "in_progress", "blocked"}:
            raise ControlPlaneModelError("illegal completion transition")
        self.tasks[task_id] = status


def test_generation_never_decreases() -> None:
    model = ControlPlaneModel(generation=7)
    model.lose_owner()
    model.successor()
    assert model.generation == 8
    model.lose_owner()
    model.successor()
    assert model.generation == 9


def test_live_owner_cannot_mint_successor() -> None:
    model = ControlPlaneModel()
    with pytest.raises(ControlPlaneModelError, match="live owner"):
        model.successor()


def test_completed_cannot_be_rewritten_as_success() -> None:
    model = ControlPlaneModel()
    model.set_status("DOEP-001", "completed")
    with pytest.raises(ControlPlaneModelError, match="rewritten"):
        model.set_status("DOEP-001", "in_progress")
    model.set_status("DOEP-001", "todo")
    assert model.tasks["DOEP-001"] == "todo"


def test_exclusive_owner_loss_then_same_generation_restart() -> None:
    model = ControlPlaneModel(generation=48)
    model.restart_same_generation()
    assert model.generation == 48
    model.lose_owner()
    with pytest.raises(ControlPlaneModelError, match="lost owner"):
        model.restart_same_generation()

