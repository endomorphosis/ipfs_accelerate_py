"""Stopped policy migration retains actual task and watchdog history."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import (
    CampaignConfig, atomic_json, build_population, identity, observed_forest, read_json,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog_policy import (
    ADDED_PATHS, migrate, validate_scope_transition,
)
from test.api.test_worldwide_legal_search_runtime import plan_fixture


def command(root, *argv):
    return subprocess.run(["git", "-C", str(root), *argv], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def campaign(tmp_path):
    site, framework, state = [tmp_path / name for name in ("site", "framework", "state")]
    for root in (site, framework):
        root.mkdir()
        command(root, "init", "-q", "-b", "campaign")
        command(root, "config", "user.name", "Fixture")
        command(root, "config", "user.email", "fixture@example.invalid")
    (framework / "fixture.py").write_text("# actual source\n")
    command(framework, "add", ".")
    command(framework, "commit", "-qm", "Source fixture")
    plan = plan_fixture()
    config = CampaignConfig(site, framework, state, site / "implementation_plan/plan.json", tmp_path / "config.json")
    atomic_json(config.plan_path, plan)
    before = {"schema": "worldwide-legal-search.watchdog-policy@1", "policy_id": "fixture-policy",
        "max_campaign_actions": 128, "max_campaign_repairs": 8,
        "repair_scope": {"allowed_paths": ["ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search*.py"],
            "protected_policy_or_completion_criteria_changes": False}}
    path = config.plan_path.with_name("watchdog.policy.json")
    atomic_json(path, before)
    command(site, "add", ".")
    command(site, "commit", "-qm", "Existing admitted policy")
    predecessor = command(site, "rev-parse", "HEAD")
    after = copy.deepcopy(before)
    after["repair_scope"]["allowed_paths"].extend(ADDED_PATHS)
    atomic_json(path, after)
    command(site, "add", ".")
    command(site, "commit", "-qm", "Reviewed exact source paths")
    watchdog = {"schema": "wwl.watchdog.state@1", "run_id": config.run_id,
        "policy_digest": identity(before).removeprefix("sha256:"), "actions": 14, "repairs": 1,
        "required_task_ids": ["WWL-000", "WWL-001"], "admitted_plan_revision": "plan:existing",
        "incidents": {"retained": {"reconcile": 2, "recover": 2, "repair": 1}},
        "outbox": {"action_id": "old:unknown", "kind": "reconcile"},
        "inflight": {"old:running": {"deadline_at": 100}}, "uncertain_actions": {"old:unknown": {"paid": True}}}
    atomic_json(state / "watchdog/state.json", watchdog)
    return config, predecessor, before, after, watchdog


def stopped(_config):
    return {"both_stopped": True, "supervisor": {"main_pid": 0, "cgroup_empty": True}, "watchdog": {"MainPID": "0"}}


def test_reviewed_scope_transition_rejects_budget_criteria_and_wildcard_changes(campaign):
    _, _, before, after, _ = campaign
    validate_scope_transition(before, after)
    for mutation in ("budget", "criteria", "wildcard", "omission", "old_path"):
        bad = copy.deepcopy(after)
        if mutation == "budget": bad["max_campaign_actions"] += 1
        elif mutation == "criteria": bad["repair_scope"]["protected_policy_or_completion_criteria_changes"] = True
        elif mutation == "wildcard": bad["repair_scope"]["allowed_paths"][-1] += "*"
        elif mutation == "omission": bad["repair_scope"]["allowed_paths"].pop()
        else: bad["repair_scope"]["allowed_paths"].pop(0)
        with pytest.raises(ValueError, match="only the exact"):
            validate_scope_transition(before, bad)


def test_policy_migration_preserves_every_counter_and_unsettled_action(campaign):
    config, predecessor, _, after, original = campaign
    before_bytes = (config.state_root / "watchdog/state.json").read_bytes()
    preview = migrate(config, previous_site_revision=predecessor, fence_probe=stopped)
    assert preview["applied"] is False
    assert (config.state_root / "watchdog/state.json").read_bytes() == before_bytes
    assert not (config.state_root / "control/policy-migrations").exists()
    result = migrate(config, previous_site_revision=predecessor, apply=True, fence_probe=stopped)
    actual = read_json(config.state_root / "watchdog/state.json")
    assert actual == {**original, "policy_digest": identity(after).removeprefix("sha256:")}
    assert result["watchdog_budgets_and_task_set_preserved"] is True
    assert result["native_policy_revision"]["required"] is False
    replay = migrate(config, previous_site_revision=predecessor, apply=True, fence_probe=stopped)
    assert replay["admission_id"] == result["admission_id"]
    assert read_json(config.state_root / "watchdog/state.json") == actual


def test_policy_migration_refuses_live_services_stale_policy_and_uncommitted_input(campaign):
    config, predecessor, _, _, original = campaign
    with pytest.raises(ValueError, match="both registered"):
        migrate(config, previous_site_revision=predecessor, apply=True, fence_probe=lambda c: {"both_stopped": False})
    assert read_json(config.state_root / "watchdog/state.json") == original
    atomic_json(config.state_root / "watchdog/state.json", {**original, "policy_digest": "unrelated-policy"})
    with pytest.raises(ValueError, match="predecessor run or policy"):
        migrate(config, previous_site_revision=predecessor, apply=True, fence_probe=stopped)
    atomic_json(config.state_root / "watchdog/state.json", original)
    path = config.plan_path.with_name("watchdog.policy.json")
    policy = read_json(path)
    policy["max_campaign_actions"] += 1
    atomic_json(path, policy)
    with pytest.raises(ValueError, match="committed"):
        migrate(config, previous_site_revision=predecessor, apply=True, fence_probe=stopped)
    assert read_json(config.state_root / "watchdog/state.json") == original


def test_native_policy_admission_uses_real_cas_and_leaves_tasks_exact(campaign):
    from dataclasses import replace
    from test.api.test_agent_supervisor_plan_revision_store import _revision, _roots
    from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import PlanPopulationDigest, PopulationKind
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import PlanRevisionStore, PlanRevisionApplyRequest
    config, predecessor, before, after, original = campaign
    population = build_population(read_json(config.plan_path), source_forest=observed_forest(config))
    with DatabaseTaskSource(config.database, owner_id="fixture-setup") as source:
        source.materialize(population)
        projection = copy.deepcopy(dict(source.plan_projection()))
    root = population["plan_root_cid"]
    tasks = tuple(t["task_cid"] for t in projection["tasks"])
    goals = tuple(g["goal_cid"] for g in projection["goals"])
    roots = _roots(policy_root=identity(before))
    revision = _revision(plan_root_cid=root, roots=roots,
        task_population=PlanPopulationDigest(kind=PopulationKind.RETAINED, member_cids=tasks),
        retained_population=PlanPopulationDigest(kind=PopulationKind.RETAINED, member_cids=tasks),
        goal_population=PlanPopulationDigest(kind=PopulationKind.RETAINED, member_cids=goals))
    store = PlanRevisionStore(config.state_root / "plan-revisions")
    store.apply(PlanRevisionApplyRequest(revision=revision, observed_roots=roots,
        idempotency_key="fixture-existing-native-plan", expected_effects=("adopt_existing_plan_metadata",)))
    old_active = store.get_active()
    result = migrate(config, previous_site_revision=predecessor, apply=True, fence_probe=stopped)
    new_active = store.get_active()
    assert new_active.semantic_revision == old_active.semantic_revision + 1
    assert store.load_revision(new_active.revision_cid).roots.policy_root == identity(after)
    assert result["native_policy_revision"]["task_specifications_unchanged"] is True
    with DatabaseTaskSource(config.database, owner_id="fixture-verify") as source:
        final = dict(source.plan_projection())
    for key in ("tasks", "goals", "goal_edges"):
        assert final[key] == projection[key]
    assert read_json(config.state_root / "watchdog/state.json") == {**original, "policy_digest": identity(after).removeprefix("sha256:")}
    replay = migrate(config, previous_site_revision=predecessor, apply=True, fence_probe=stopped)
    assert store.get_active().revision_cid == new_active.revision_cid
    assert replay["admission_id"] == result["admission_id"]
