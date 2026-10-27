"""Native campaign ingestion, repository routing and completion boundaries."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import (
    CampaignConfig, CampaignRuntime, atomic_json, build_population, identity,
    read_json, topological_tasks, validate_goal,
)


def plan_fixture():
    tasks = []
    for number in (0, 1):
        task_id = f"WWL-{number:03d}"
        tasks.append({"id": task_id, "title": "Fixture task " + str(number), "depends_on": [] if number == 0 else ["WWL-000"],
            "repository": "site" if number == 0 else "accelerate", "goal_id": "WWL-G010", "subgoal_id": "WWL-G010",
            "outputs": [f"fixture-{number}.json"], "evidence_output": f"evidence-{number}.json", "acceptance": "Actual validated fixture output",
            "validation_argv": [["python3", "-c", "print('validated fixture')"]],
            "bounds": {"max_attempts": 3, "wall_seconds_per_attempt": 5400}})
    return {"schema": "worldwide-legal-search.plan@1", "namespace": "fixture-worldwide",
        "goals": [{"id": "WWL-G000", "title": "Fixture root", "parent_ids": [], "task_ids": ["WWL-000", "WWL-001"], "acceptance": "Native receipts"},
                  {"id": "WWL-G010", "title": "Fixture child", "parent_ids": ["WWL-G000"], "task_ids": ["WWL-000", "WWL-001"], "acceptance": "Native receipts"}],
        "tasks": tasks, "execution": {"protected_acceptance_inputs": []}}


FOREST = {"site": {"head": "1" * 40, "tree": "2" * 40}, "accelerate": {"head": "3" * 40, "tree": "4" * 40}}


class RuntimeBoundaries(unittest.TestCase):
    def test_dependency_order_is_derived_not_file_order(self):
        plan = plan_fixture()
        plan["tasks"].reverse()
        self.assertEqual([task["id"] for task in topological_tasks(plan)], ["WWL-000", "WWL-001"])
        plan["tasks"][1]["depends_on"] = ["WWL-001"]
        with self.assertRaises(ValueError):
            topological_tasks(plan)

    def test_population_retains_distinct_owning_repositories_and_validation(self):
        population = build_population(plan_fixture(), source_forest=FOREST)
        tasks = population["tasks"]
        self.assertEqual(tasks[0]["base_revision"], FOREST["site"]["head"])
        self.assertEqual(tasks[1]["base_revision"], FOREST["accelerate"]["head"])
        self.assertEqual(tasks[1]["depends_on"], [tasks[0]["task_cid"]])
        self.assertEqual(tasks[0]["completion"], "auto")
        self.assertTrue(tasks[0]["validations"])
        self.assertNotIn("completion_receipt", tasks[0])

    def test_native_source_ingests_exact_dependency_frontier(self):
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
        population = build_population(plan_fixture(), source_forest=FOREST)
        with tempfile.TemporaryDirectory() as directory:
            with DatabaseTaskSource(Path(directory) / "control.duckdb", owner_id="wwl-fixture-owner") as source:
                receipt = source.materialize(population)
                self.assertEqual(receipt["task_count"], 2)
                self.assertEqual(receipt["goal_count"], 2)
                self.assertEqual([task.task_alias for task in source.ready_tasks().tasks], ["WWL-000"])
                projection = source.intent.completion_evidence_projection()
                self.assertEqual(projection["completion_receipts"], [])
                self.assertFalse(source.snapshot().terminal)

    def test_multi_repository_bridge_routes_all_three_native_gates(self):
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_args
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import bind_database_portal_execution_from_args
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            site, framework = base / "site", base / "framework"
            for root, branch in ((site, "site-campaign"), (framework, "framework-campaign")):
                root.mkdir()
                subprocess.run(["git", "init", "-q", "-b", branch, str(root)], check=True)
            parsed = parse_args(["--implement", "--state-dir", str(base / "state"), "--state-prefix", "wwl",
                "--merge-target-branch", "site-campaign", "--worktree-root", str(base / "worktrees")])
            captured = {}
            daemon = SimpleNamespace(task_source=object(), authorize_superseded_portal_attempt_binding=lambda *_args: None,
                bind_execution_callbacks=lambda **callbacks: captured.update(callbacks))

            class NativeBridgeFixture:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.attempt_root = kwargs["attempt_root"]
                    self.max_task_attempts = 3
                def run_provider(self, _attempt):
                    return ("provider", self.kwargs["repo_root"], self.kwargs["merge_target_branch"])
                def apply_effect(self, _attempt, _result):
                    return ("effect", self.kwargs["repo_root"], self.kwargs["merge_target_branch"])
                def validate_effect(self, _attempt, _result):
                    return ("validation", self.kwargs["repo_root"], self.kwargs["merge_target_branch"])

            with patch("ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge.DatabasePortalExecutionBridge", NativeBridgeFixture):
                bind_database_portal_execution_from_args(daemon, parsed, repo_root=site, portal_daemon_class=object,
                    task_repo_root_resolver=lambda alias: site if alias == "WWL-000" else framework)
                for alias, expected, branch in (("WWL-000", site, "site-campaign"), ("WWL-002", framework, "framework-campaign")):
                    attempt = SimpleNamespace(task_alias=alias)
                    self.assertEqual(captured["provider_fn"](attempt), ("provider", expected, branch))
                    self.assertEqual(captured["effect_fn"](attempt, {}), ("effect", expected, branch))
                    self.assertEqual(captured["validation_fn"](attempt, {}), ("validation", expected, branch))

    def test_unknown_repository_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            atomic_json(plan_path, plan_fixture())
            config = CampaignConfig(root / "site", root / "framework", root / "state", plan_path, root / "config.json")
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            self.assertEqual(runtime.repository_for_task("WWL-001"), root / "framework")
            with self.assertRaises(ValueError):
                runtime.repository_for_task("WWL-999")

    def test_goal_cannot_complete_from_counts_without_native_receipt_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = CampaignConfig(root / "site", root / "framework", root / "state", root / "plan.json", root / "config.json")
            atomic_json(config.plan_path, plan_fixture())
            atomic_json(config.state_root / "campaign-status.json", {"verified_task_ids": ["WWL-000", "WWL-001"], "native_completion_receipts": {}})
            with self.assertRaises(ValueError):
                validate_goal(config, "WWL-G000")

    def test_failed_pass_is_counted_and_pre_provider_failure_is_known(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            atomic_json(plan_path, plan_fixture())
            config = CampaignConfig(root / "site", root / "framework", root / "state", plan_path, root / "config.json")
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime._daemon = SimpleNamespace(list_running_attempts=lambda: [], run_once=lambda: (_ for _ in ()).throw(RuntimeError("pre-provider reproducer")))
            with patch.object(runtime, "capture"), patch("ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime.git", return_value="source-head"):
                runtime._run_pass()
            self.assertEqual(runtime._passes, 1)
            self.assertFalse(runtime._last_failure["effects_unknown"])
            self.assertFalse(runtime._last_failure["provider_callback_entered"])

    def test_retained_attempt_failure_never_claims_pre_effect_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            atomic_json(plan_path, plan_fixture())
            config = CampaignConfig(root / "site", root / "framework", root / "state", plan_path, root / "config.json")
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime._daemon = SimpleNamespace(list_running_attempts=lambda: [object()], run_once=lambda: (_ for _ in ()).throw(RuntimeError("retained unknown effect")))
            with patch.object(runtime, "capture"), patch("ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime.git", return_value="source-head"):
                runtime._run_pass()
            self.assertTrue(runtime._last_failure["effects_unknown"])


if __name__ == "__main__":
    unittest.main()
