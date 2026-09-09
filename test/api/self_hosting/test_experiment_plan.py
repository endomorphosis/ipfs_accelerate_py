from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.self_hosting.experiment import (
    EVIDENCE_KINDS,
    ExperimentPlan,
    SelfHostingTask,
)
from ipfs_accelerate_py.proof_context.bootstrap import RUNTIME_CID


def _plan(**changes):
    payload = {
        "engine_id": RUNTIME_CID,
        "package_id": "ipfs-accelerate-py",
        "package_identity": "sha256:" + "1" * 64,
        "repository_id": "example/self-hosting",
        "repository_state_cid": "bafkreigh2akiscaildc6ic4z2zgw6eo6wgbdbr7r7z5t2wea5l5lyz66ri",
        "configuration_id": "current-head-a",
        "configuration_cid": "bafkreigh2akiscaildc6ic4z2zgw6eo6wgbdbr7r7z5t2wea5l5lyz66ri",
        "evidence_kind": "live",
        "tasks": [{"task_id": "task-1", "task_specification_cid": "bafkreigh2akiscaildc6ic4z2zgw6eo6wgbdbr7r7z5t2wea5l5lyz66ri", "proposal": {"files": {"src/demo/value.py": "VALUE = 2\n"}, "declared_files": ["src/demo/value.py"]}}],
    }
    payload.update(changes)
    return ExperimentPlan.from_mapping(payload)


def test_plan_binds_all_inputs_and_is_deterministic():
    first = _plan()
    second = _plan()
    assert first.plan_id == second.plan_id
    assert first.to_mapping()["plan_id"] == first.plan_id
    assert set(EVIDENCE_KINDS) == {"live", "replayed", "simulated"}


def test_plan_rejects_duplicate_task_and_unbound_replay():
    with pytest.raises(ValueError, match="unique"):
        _plan(tasks=[_plan().tasks[0].to_mapping(), _plan().tasks[0].to_mapping()])
    with pytest.raises(ValueError, match="replay_record"):
        _plan(evidence_kind="replayed")


def test_task_rejects_empty_identity():
    with pytest.raises(ValueError, match="task_id"):
        SelfHostingTask(task_id="", task_specification_cid="bafkreigh2akiscaildc6ic4z2zgw6eo6wgbdbr7r7z5t2wea5l5lyz66ri")


def test_frozen_fixture_plan_is_consumable():
    fixture = Path(__file__).parents[2] / "fixtures" / "proof_context_self_hosting" / "live-plan.fixture"
    plan = ExperimentPlan.from_mapping(json.loads(fixture.read_text(encoding="utf-8")))
    assert plan.evidence_kind == "live"
    assert plan.tasks[0].task_id == "fixture-task"
