"""Native campaign ingestion, repository routing and completion boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import (
    CampaignConfig, CampaignRuntime, atomic_json, build_population, identity,
    read_json, topological_tasks, validate_goal, admit_repair, REPAIR_VALIDATOR,
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


class RepairAdmissionBoundaries(unittest.TestCase):
    def fixture(self, root):
        site, framework, state = root / "site", root / "framework", root / "state"
        for repository in (site, framework):
            repository.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "campaign", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "config", "user.email", "fixture@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repository), "config", "user.name", "Fixture"], check=True)
        relative = "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_runtime.py"
        target = framework / relative
        target.parent.mkdir(parents=True)
        target.write_text("def affected():\n    raise RuntimeError('native fixture defect')\n")
        (site / "fixture").write_text("site")
        for repository in (site, framework):
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repository), "commit", "-qm", "Fixture"], check=True)
        config = CampaignConfig(site, framework, state, site / "plan.json", root / "config.json")
        atomic_json(config.plan_path, plan_fixture())
        atomic_json(config.plan_path.with_name("watchdog.policy.json"), {"policy_id": "fixture-policy", "repair_scope": {"allowed_paths": ["ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search*.py", "test/api/test_worldwide_legal_search*.py"]}})
        from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import git, observed_forest
        failure = {"schema": "wwl.native-failure@1", "run_id": config.run_id, "effects_unknown": False,
            "source_revision": git(framework, "rev-parse", "HEAD"), "traceback": "Native fixture traceback", "reproduce_argv": ["native-exclusive-fixture"],
            "phase": "preselection", "source_frames": [{"path": relative, "line": 2, "source_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}]}
        failure["receipt_id"] = identity(failure)
        atomic_json(state / "failures/fixture.json", failure)
        request = {"schema": "wwl.watchdog.action@1", "action_id": "a" * 64, "kind": "repair", "run_id": config.run_id,
            "policy_id": "fixture-policy", "accepted_tree": identity(observed_forest(config)), "reproducer_receipt": failure["receipt_id"], "issued_at": time.time(), "authority_revision": "fixture-revision"}
        atomic_json(state / "campaign-status.json", {"run_id": config.run_id, "effects_unknown": False, "hold": False,
            "authority_revision": request["authority_revision"], "native_error": failure})
        path = state / "control/requests" / (request["action_id"] + ".json")
        atomic_json(path, request)
        return config, path, failure, target

    def test_repair_binds_exact_current_traced_source_and_new_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            config, request, failure, target = self.fixture(Path(directory))
            admission = admit_repair(config, request)
            self.assertEqual(admission["source_paths"], [target.relative_to(config.framework_root).as_posix()])
            self.assertEqual(admission["failure"], failure)
            self.assertIn("repair_aaaaaaaaaaaaaaaa.py", admission["regression_path"])
            self.assertTrue(all(not path.endswith("/") for path in admission["allowed_paths"]))

    def test_repair_rejects_missing_and_tampered_reproducer(self):
        with tempfile.TemporaryDirectory() as directory:
            config, request, failure, _target = self.fixture(Path(directory))
            record = config.state_root / "failures/fixture.json"
            record.unlink()
            with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                admit_repair(config, request)
            atomic_json(record, {**failure, "traceback": "tampered failure evidence"})
            with self.assertRaisesRegex(ValueError, "integrity"):
                admit_repair(config, request)

    def test_repair_rejects_archived_failure_and_stale_owner_authority(self):
        for mutation in ("resolved", "authority"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                config, request, _failure, _target = self.fixture(Path(directory))
                status_path = config.state_root / "campaign-status.json"
                status = read_json(status_path)
                if mutation == "resolved":
                    status["native_error"] = {}
                else:
                    status["authority_revision"] = "new-native-revision"
                atomic_json(status_path, status)
                with self.assertRaises(ValueError):
                    admit_repair(config, request)

    def test_repair_rejects_changed_source_and_protected_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            config, request, _failure, target = self.fixture(Path(directory))
            target.write_text(target.read_text() + "# drift\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                admit_repair(config, request)
        with tempfile.TemporaryDirectory() as directory:
            config, request, _failure, target = self.fixture(Path(directory))
            plan = read_json(config.plan_path)
            plan["execution"]["protected_acceptance_inputs"] = [target.relative_to(config.framework_root).as_posix()]
            atomic_json(config.plan_path, plan)
            with self.assertRaisesRegex(ValueError, "outside the admitted"):
                admit_repair(config, request)

    def test_repair_rejects_unknown_effects_and_outside_policy_cause(self):
        for mutation in ("effects_unknown", "outside"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                config, request, failure, _target = self.fixture(Path(directory))
                if mutation == "effects_unknown":
                    failure["effects_unknown"] = True
                else:
                    policy_path = config.plan_path.with_name("watchdog.policy.json")
                    policy = read_json(policy_path)
                    policy["repair_scope"]["allowed_paths"] = ["unrelated.py"]
                    atomic_json(policy_path, policy)
                failure.pop("receipt_id")
                failure["receipt_id"] = identity(failure)
                atomic_json(config.state_root / "failures/fixture.json", failure)
                action = read_json(request)
                action["reproducer_receipt"] = failure["receipt_id"]
                atomic_json(request, action)
                status = read_json(config.state_root / "campaign-status.json")
                status["native_error"] = failure
                atomic_json(config.state_root / "campaign-status.json", status)
                with self.assertRaises(ValueError):
                    admit_repair(config, request)

    def test_immutable_repair_validator_requires_real_baseline_failure_and_green_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "repair-fixture", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "fixture@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Fixture"], check=True)
            module = root / "fixture_module.py"
            module.write_text("VALUE = 1\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "Reproduced baseline"], check=True)
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            module.write_text("VALUE = 2\n")
            regression = root / "test_repair_regression.py"
            regression.write_text("from fixture_module import VALUE\n\ndef test_defect():\n    assert VALUE == 2\n")
            binding = {"action_id": "a" * 64, "source_revision": revision, "regression_path": regression.name,
                "allowed_paths": [module.name, regression.name], "protected_hashes": {}, "focused_tests": [], "baseline_inputs": [module.name]}
            result = subprocess.run([sys.executable, "-I", "-c", REPAIR_VALIDATOR, json.dumps(binding)], cwd=root, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["baseline_failure_verified"])
            regression.write_text("def test_placeholder():\n    assert True\n")
            result = subprocess.run([sys.executable, "-I", "-c", REPAIR_VALIDATOR, json.dumps(binding)], cwd=root, capture_output=True, text=True, timeout=180)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("did not reproduce", result.stderr)


if __name__ == "__main__":
    unittest.main()
