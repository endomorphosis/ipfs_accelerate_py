"""Disposable native Quack owner fixtures independent of retired inbox protocols."""
from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import open_intent_repository


def _seed(database: Path) -> None:
    repo = open_intent_repository(database, owner_id="seed")
    try:
        repo.upsert_objective(
            objective_id="objective:test", objective_alias="O", title="Objective"
        )
        repo.upsert_goal(
            goal_cid="goal:test",
            goal_alias="G",
            title="Goal",
            objective_id="objective:test",
        )
        repo.upsert_plan(
            plan_cid="plan:test",
            goal_cid="goal:test",
            plan_alias="P",
        )
        repo.upsert_task(
            task_cid="task:test",
            task_alias="T",
            goal_cid="goal:test",
            plan_cid="plan:test",
            objective_id="objective:test",
            ordinal=1,
            status="ready",
        )
    finally:
        repo.close()
