"""Native campaign ingestion, repository routing and completion boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import threading
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import (
    CampaignConfig, CampaignRuntime, atomic_json, build_population, identity,
    read_json, topological_tasks, validate_goal, admit_repair, REPAIR_VALIDATOR,
    repair_pipeline_binding, verified_repair_canary,
    wait_for_storage, bootstrap,
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
    def test_uncertain_reconciliation_stays_nonterminal_and_does_not_crash_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = CampaignConfig(root / 'site', root / 'framework', root / 'state', root / 'plan.json', root / 'config.json')
            atomic_json(config.plan_path, plan_fixture())
            old = {'action_id': 'a' * 64, 'status': 'blocked', 'effects_verified': False, 'receipt': 'native:unknown'}
            ledger = {'a' * 64: {'request': {'kind': 'reconcile'}, 'result': {'status': 'accepted'}, 'observation': old}}
            atomic_json(config.state_root / 'control/action-ledger.json', ledger)
            runtime = CampaignRuntime(config, {'admitted_plan_revision': 'plan', 'receipt_id': 'receipt'})
            action = runtime._actions['a' * 64]
            self.assertEqual(action['observation']['status'], 'unknown')
            self.assertEqual(action['previous_unverified_terminal_observation'], old)
            runtime._daemon = SimpleNamespace(**{name: (lambda: []) for name in (
                'reconcile_prepared_task_completions', 'reconcile_terminal_portal_failures',
                'reconcile_expired_running_attempts', 'reconcile_recoverable_portal_failure_rearms')})
            runtime._last_failure = {'effects_unknown': True}
            with patch.object(runtime, 'capture'):
                runtime._reconcile_controls()
            self.assertEqual(action['observation']['status'], 'unknown')
            self.assertFalse(action['observation']['effects_verified'])
            # Genuine later native reconciliation may settle the same action;
            # it never creates a new action ID or replenishes the host budget.
            runtime._last_failure = {}
            runtime._cached = {'native_completion_projection': 'native:completed-projection'}
            runtime._pending_controls.append('a' * 64)
            with patch.object(runtime, 'capture'):
                runtime._reconcile_controls()
            self.assertEqual(action['observation']['status'], 'succeeded')
            self.assertTrue(action['observation']['effects_verified'])
            self.assertEqual(action['reconciliation_evidence']['native_completion_projection'], 'native:completed-projection')

    def test_pending_candidate_survives_restart_and_rejects_changed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = CampaignConfig(root / 'site', root / 'framework', root / 'state', root / 'plan.json', root / 'config.json')
            atomic_json(config.plan_path, plan_fixture())
            request = config.state_root / 'maintenance/candidate.json'
            atomic_json(request, {'run_id': config.run_id, 'task_id': 'WWL-001'})
            atomic_json(config.state_root / 'control/pending-candidate-continuation.json', {
                'schema': 'wwl.pending-candidate-continuation@1', 'run_id': config.run_id,
                'request_path': str(request), 'request_sha256': hashlib.sha256(request.read_bytes()).hexdigest()})
            runtime = CampaignRuntime(config, {'admitted_plan_revision': 'plan', 'receipt_id': 'receipt'})
            self.assertEqual(runtime._candidate_request_path, request)
            atomic_json(request, {'run_id': config.run_id, 'task_id': 'WWL-000'})
            with self.assertRaisesRegex(ValueError, 'pending candidate continuation custody differs'):
                runtime._read_candidate_request()
            with self.assertRaisesRegex(ValueError, 'pending candidate continuation custody differs'):
                CampaignRuntime(config, {'admitted_plan_revision': 'plan', 'receipt_id': 'receipt'})

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
        self.assertEqual(tasks[0]["validations"][0]["argv"], plan_fixture()["tasks"][0]["validation_argv"][0])
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
                self.assertEqual(list(source.get_task("WWL-000").validations[0]["argv"]), plan_fixture()["tasks"][0]["validation_argv"][0])
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

    def test_restarted_repair_is_never_dispatched_twice_and_reconcile_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config, request_path, _failure, _target = self.fixture(Path(directory))
            request = read_json(request_path)
            action = {"request": request, "result": {"status": "accepted", "deadline_at": time.time() + 120},
                "observation": {"status": "running"}, "repair_started": time.time()}
            reconcile = {"request": {"kind": "reconcile"}, "result": {"status": "accepted"}, "observation": {"status": "running"}}
            atomic_json(config.state_root / "control/action-ledger.json", {request["action_id"]: action, "reconcile-action": reconcile})
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            self.assertEqual(runtime._pending_controls, ["reconcile-action"])
            runtime.settle_controls()
            self.assertEqual(runtime._actions[request["action_id"]]["observation"]["status"], "unknown")
            self.assertIsNone(runtime._repair_thread)

    def test_dispatch_exception_and_exit_zero_without_native_receipt_remain_unknown(self):
        for mode in ("exception", "exit-zero"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config, request_path, _failure, _target = self.fixture(Path(directory))
                request = read_json(request_path)
                runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
                action = {"request": request, "result": {"status": "accepted", "deadline_at": time.time() + 120},
                    "observation": {"status": "running"}, "repair_started": time.time()}
                runtime._actions[request["action_id"]] = action
                if mode == "exception":
                    with patch.object(runtime, "_dispatch_repair", side_effect=RuntimeError("owned dispatcher fixture fault")):
                        runtime._execute_repair(request["action_id"], action)
                else:
                    with patch("ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher.run_worker", return_value={"outcome": "awaiting_observation", "exit_code": 0}):
                        runtime._execute_repair(request["action_id"], action)
                self.assertEqual(action["observation"]["status"], "unknown")
                self.assertFalse(action["observation"]["effects_verified"])
                self.assertFalse(runtime.cancel.is_set())

    def test_promoted_handoff_metadata_survives_reloaded_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            config, request_path, _failure, _target = self.fixture(Path(directory))
            request = read_json(request_path)
            observation = {"status": "running", "requires_fenced_reload": True, "expected_framework_revision": "new-native-head",
                "original_failure_receipt": request["reproducer_receipt"]}
            action = {"request": request, "result": {"status": "accepted"}, "observation": observation, "repair_started": time.time()}
            atomic_json(config.state_root / "control/action-ledger.json", {request["action_id"]: action})
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime.settle_controls()
            self.assertEqual(runtime._actions[request["action_id"]]["observation"], observation)
            self.assertIsNone(runtime._repair_thread)
            self.assertTrue(runtime._repair_requires_reconciliation())
            runtime._loaded_framework_revision = observation["expected_framework_revision"]
            self.assertFalse(runtime._repair_requires_reconciliation())
            observation["status"] = "unknown"
            runtime._actions[request["action_id"]]["observation"] = observation
            self.assertTrue(runtime._repair_requires_reconciliation())

    def test_canary_binding_survives_unrelated_commit_but_rejects_pipeline_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request_path, _failure, target = self.fixture(Path(directory))
            for relative in ("project_adapters/worldwide_legal_search_continuation.py", "project_adapters/worldwide_legal_search_provider.py", "project_adapters/worldwide_legal_search_goals.py",
                             "todo_daemon/implementation_daemon_runner.py", "todo_daemon/implementation_daemon.py", "todo_daemon/database_portal_bridge.py", "todo_daemon/dependency_preflight_retry.py", "rescue/repair_dispatcher.py",
                             "task_sources/board_control_plane.py", "task_sources/database_task_source.py", "task_sources/intent_repository.py",
                             "validation/validation_runtime.py", "validation/validation_commands.py", "validation/project_dependency_preflight.py",
                             "todo_daemon/supervisor_runtime.py", "grok_cli_runner.py", "runtime/grok_cli_runner.py", "provider_fallback_runner.py"):
                path = config.framework_root / "ipfs_accelerate_py/agent_supervisor" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# pipeline fixture\n")
            receipt = {"schema": "wwl.native-repair-canary@1", "pipeline_binding": repair_pipeline_binding(config),
                "baseline_failure_verified": True, "candidate_regressions_verified": True, "validated_promotion": True,
                "native_completion_receipt": "native-fixture-proof", "native_projection": "native-fixture-projection",
                "source_provenance": {"fixture": True}}
            receipt["receipt_id"] = identity(receipt)
            atomic_json(config.state_root / "repair-canary.json", receipt)
            self.assertIsNotNone(verified_repair_canary(config))
            (config.framework_root / "unrelated.txt").write_text("unrelated commit\n")
            subprocess.run(["git", "-C", str(config.framework_root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(config.framework_root), "commit", "-qm", "Unrelated source commit"], check=True)
            self.assertIsNotNone(verified_repair_canary(config))
            target.write_text(target.read_text() + "# pipeline changed\n")
            self.assertIsNone(verified_repair_canary(config))

    def test_blocked_native_result_is_not_success_or_an_autonomous_repair_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime._last_failure = {}
            runtime._daemon = SimpleNamespace(list_running_attempts=lambda: [], run_once=lambda: {
                "active_task_id": "WWL-000", "implementation_result": {"status": "blocked", "reason": "implementation_protected_path_mutated"}})
            with patch.object(runtime, "capture"):
                runtime._run_pass()
            self.assertEqual(runtime._last_successful_pass, {})
            self.assertEqual(runtime.publish()["blocked_kind"], "permission")
            self.assertEqual(runtime._last_failure, {})
            runtime._cached["task_statuses"] = {"WWL-000": "blocked"}
            runtime._daemon.run_once = lambda: {"implementation_result": None, "selection_idle_reason": "no_ready_tasks"}
            with patch.object(runtime, "capture"):
                runtime._run_pass()
            self.assertEqual(runtime._native_terminal_result["reason"], "implementation_protected_path_mutated")

    def test_canonical_portal_factory_keeps_native_exit_attestation_and_narrows_repair_scope(self):
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon, PortalTask
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime.plan["operational_repair_binding"] = {"allowed_paths": ["allowed.py"]}
            board = config.site_root / "fixture.todo.md"
            board.write_text("# Fixture\n\n## WWL-000 Fixture\n- Status: ready\n- Outputs: allowed.py\n")
            native = runtime._make_native_portal(todo_path=board, state_path=config.state_root / "portal.json",
                strategy_path=config.state_root / "strategy.json", events_path=config.state_root / "events.jsonl", repo_root=config.site_root)
            try:
                self.assertIs(type(native), PortalImplementationDaemon)
                task = PortalTask("WWL-000", "Fixture", "ready", "auto", "P1", "implementation", outputs=["allowed.py"],
                    metadata={"allowed paths": "forbidden.py", "predicted files": "also-forbidden.py"})
                self.assertEqual(native._proposal_scope_paths(task), ("allowed.py",))
                self.assertFalse(native._implementation_cancel_requested())
                runtime.cancel.set()
                self.assertTrue(native._implementation_cancel_requested())
                bridge = DatabasePortalExecutionBridge.__new__(DatabasePortalExecutionBridge)
                paths = SimpleNamespace(state=native.state_path, events=native.events_path)
                with patch.object(native, "_take_database_provider_exit", return_value=None) as taken:
                    bridge._native_provider_exit_failure(attempt=None, paths=paths, binding={}, daemon=native, authority={}, implementation={}, summary={})
                    taken.assert_called_once()
            finally:
                native.close_event_runtime()

    def test_fresh_owner_reconstructs_native_cleanup_blocker_without_unblocking(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime._last_failure = {}
            attempt = "attempt:retained-provider"
            root = config.state_root / "native/wwl_database_portal_attempts" / hashlib.sha256(attempt.encode()).hexdigest()[:24]
            atomic_json(root / "database-attempt-binding.json", {"attempt_id": attempt, "task_alias": "WWL-000", "task_cid": "native-task", "binding_id": "native-binding"})
            event = {"type": "implementation_finished", "task_id": "WWL-000", "event_id": "native-event", "provider_dispatched": True,
                "returncode": 1, "exception_result": {"exception_type": "ProcessGroupCleanupUnverified"}}
            (root / "portal-events.jsonl").write_text(json.dumps(event) + "\n")
            task = SimpleNamespace(task_alias="WWL-000", task_cid="native-task", revision=4, status="blocked",
                body={"completion_receipt": {"attempt_id": attempt, "task_cid": "native-task", "settlement_id": "native-failure"}})
            source = SimpleNamespace(intent=SimpleNamespace(read_session=lambda: contextlib.nullcontext(),
                completion_evidence_projection=lambda: {"completion_receipts": [], "projection_cid": "native-projection"}),
                snapshot=lambda: SimpleNamespace(to_dict=lambda: {"revision": 9}), list_tasks=lambda **_kwargs: SimpleNamespace(tasks=[task]),
                ready_tasks=lambda **_kwargs: SimpleNamespace(tasks=[]), plan_projection=lambda: {"plans": []})
            runtime._daemon = SimpleNamespace(task_source=source, list_running_attempts=lambda: [])
            runtime.capture()
            snapshot = runtime.publish()
            self.assertTrue(snapshot["effects_unknown"])
            self.assertEqual(snapshot["native_blocked_tasks"][0]["reason"], "native_process_cleanup_unverified")
            self.assertEqual(snapshot["blocked_kind"], "permission")
            self.assertFalse(snapshot["completion_verified"])

    def test_only_bound_completed_tools_renew_progress_and_no_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            binding = {"run_id": config.run_id, "task_id": "WWL-000", "task_cid": "native-task", "portal_task_cid": "portal-task",
                "native_attempt_id": "attempt:active", "portal_attempt_number": 1, "binding_id": "owner-binding", "capability_nonce": "a" * 64,
                "issued_at": time.time() - 10, "activity_path": str(config.state_root / "provider-activity/active.json")}
            runtime._provider_activity_binding = binding
            runtime._cached = {"stage": "provider", "active_attempt_id": binding["native_attempt_id"]}
            activity = {"schema": "wwl.native-provider-activity@1", **{key: binding[key] for key in
                ("run_id", "task_id", "task_cid", "portal_task_cid", "native_attempt_id", "portal_attempt_number", "binding_id", "capability_nonce")},
                "sequence": 1, "event_kind": "thought", "observed_at": time.time(), "completion_authority": False}
            previous = runtime._progress_at
            atomic_json(Path(binding["activity_path"]), activity)
            runtime._consume_provider_activity()
            self.assertEqual(runtime._provider_activity_sequence, 0)
            activity["event_kind"] = "tool_completed"
            atomic_json(Path(binding["activity_path"]), activity)
            runtime._consume_provider_activity()
            self.assertEqual(runtime._provider_activity_sequence, 1)
            self.assertGreaterEqual(runtime._progress_at, previous)
            progress = runtime._progress_at
            activity["observed_at"] += 1
            atomic_json(Path(binding["activity_path"]), activity)
            runtime._consume_provider_activity()
            self.assertEqual(runtime._progress_at, progress)
            activity.update(sequence=2, native_attempt_id="attempt:prior")
            atomic_json(Path(binding["activity_path"]), activity)
            runtime._consume_provider_activity()
            self.assertEqual(runtime._provider_activity_sequence, 1)
            self.assertNotIn("verified_task_ids", runtime._cached)

    def test_only_accepted_changed_framework_sources_request_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime._last_failure = {}
            old = {"source_sha256": {"pipeline.py": "old"}}
            new = {"source_sha256": {"pipeline.py": "new"}}
            runtime._loaded_pipeline = old
            with patch("ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime.repair_pipeline_binding", return_value=new):
                runtime._check_accepted_source_reload("WWL-001")
                self.assertFalse(runtime.stop.is_set())
                runtime._cached.update(verified_task_ids=["WWL-001"], native_completion_receipts_by_alias={"WWL-001": "native-acceptance"}, native_completion_projection="native-projection")
                runtime._last_failure = {"effects_unknown": True}
                runtime._check_accepted_source_reload("WWL-001")
                self.assertFalse(runtime.stop.is_set())
                runtime._last_failure = {}
                runtime._check_accepted_source_reload("WWL-001")
                self.assertTrue(runtime.stop.is_set())
                self.assertEqual(runtime._reload_request["exit_code"], 75)
                self.assertEqual(runtime._reload_request["native_completion_receipt"], "native-acceptance")
            runtime.stop.clear()
            runtime._reload_request = {}
            with patch("ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime.repair_pipeline_binding", return_value=old):
                runtime._check_accepted_source_reload("WWL-001")
                self.assertFalse(runtime.stop.is_set())

    def test_resource_wait_is_stoppable_and_resumes_without_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            stop = threading.Event()
            blocked = {"schema": "wwl.resource-observation@1", "blocked": True, "paths": []}
            sufficient = {**blocked, "blocked": False}
            with patch("ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime.storage_observation", side_effect=[blocked, sufficient]), patch.object(stop, "wait", return_value=False) as waited:
                self.assertFalse(wait_for_storage(config, stop)["blocked"])
                waited.assert_called_once()
            stop.set()
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime.stop.set()
            with patch("ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime.storage_observation", return_value=blocked), patch.object(runtime, "open") as opened:
                self.assertEqual(runtime.run()["schema"], "wwl.native-runtime-stopped@1")
                opened.assert_not_called()
            with patch("ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime.storage_observation", return_value=blocked):
                with self.assertRaises(OSError) as raised:
                    bootstrap(config)
                self.assertEqual(raised.exception.errno, 28)
            self.assertFalse(config.database.exists())
            self.assertFalse((config.state_root / "native-owner.lock").exists())


    def test_terminalized_live_skip_is_unknown_without_a_native_terminal_event(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            attempt = "attempt:still-inflight"
            root = config.state_root / "native/wwl_database_portal_attempts" / hashlib.sha256(attempt.encode()).hexdigest()[:24]
            atomic_json(root / "database-attempt-binding.json", {"attempt_id": attempt, "task_alias": "WWL-000", "task_cid": "native-task", "binding_id": "native-binding"})
            events = [{"type": "implementation_started", "task_id": "WWL-000"},
                {"type": "implementation_skipped", "task_id": "WWL-000", "reason": "inflight_process", "event_id": "native-event"}]
            (root / "portal-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
            task = SimpleNamespace(task_alias="WWL-000", task_cid="native-task", revision=7, status="blocked", body={"completion_receipt": {"attempt_id": attempt, "task_cid": "native-task", "settlement_id": "native-failure"}})
            observation = runtime._native_blocker(task)
            self.assertTrue(observation["effects_unknown"])
            self.assertEqual(observation["reason"], "native_inflight_callback_terminalized")
            self.assertFalse(observation["completion_authority"])

    def test_repeated_provider_observation_reuses_exact_live_activity_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            attempt = SimpleNamespace(attempt_id="attempt:live", task_alias="WWL-000", task_cid="sha256:" + "a" * 64,
                started_at_ms=int((time.time() - 10) * 1000))
            inner = SimpleNamespace(task_source=object(), board_namespace="fixture", _record_for_attempt=lambda *args: object(),
                _paths=lambda _: SimpleNamespace(task_projection=config.state_root / "projection.md"),
                _render_projection=lambda *args: "projection", _projection_task_identity=lambda *args, **kwargs: {"canonical_task_cid": "sha256:" + "b" * 64})
            runtime._bridge = SimpleNamespace(_for_attempt=lambda _: inner)
            runtime._bind_provider_activity(attempt)
            path = config.state_root / "provider-activity-binding.json"
            original = path.read_bytes()
            self.assertTrue(runtime._provider_activity_binding)
            runtime._provider_activity_sequence = 7
            runtime._bind_provider_activity(attempt)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(runtime._provider_activity_sequence, 7)
            changed = read_json(path)
            changed["portal_task_cid"] = "sha256:" + "c" * 64
            atomic_json(path, changed)
            tampered = path.read_bytes()
            runtime._bind_provider_activity(attempt)
            self.assertEqual(path.read_bytes(), tampered)
            self.assertEqual(runtime._provider_activity_sequence, 7)

    def test_routed_native_provider_persists_intent_before_canonical_callback(self):
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import _RepositoryRoutedPortalBridge
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
        with tempfile.TemporaryDirectory() as directory:
            config, _request, _failure, _target = self.fixture(Path(directory))
            with DatabaseTaskSource(config.database, owner_id="fixture") as source:
                source.materialize(build_population(plan_fixture(), source_forest=FOREST))
            runtime = CampaignRuntime(config, {"admitted_plan_revision": "plan", "receipt_id": "receipt"})
            runtime.open()
            try:
                self.assertIs(type(runtime._daemon._provider_fn.__self__), _RepositoryRoutedPortalBridge)
                attempt = runtime._daemon.claim_next()
                seen = []
                def native_callback(bridge, current):
                    self.assertIs(type(bridge), DatabasePortalExecutionBridge)
                    receipt = runtime._daemon.provider_invocation_recorded(current.attempt_id, idempotency_key=f"provider:{current.attempt_id}")
                    self.assertEqual(receipt["callback_state"], "started_outcome_unknown")
                    seen.append(receipt)
                    raise RuntimeError("fixture stops before any provider effect")
                with patch.object(DatabasePortalExecutionBridge, "run_provider", native_callback):
                    # Rebind the exact official router method, never a generic observer wrapper.
                    with self.assertRaisesRegex(RuntimeError, "fixture stops"):
                        runtime._daemon.run_provider(attempt)
                self.assertEqual(len(seen), 1)
                class GenericWrapper:
                    def run_provider(self, current):
                        raise AssertionError("must not execute")
                wrapper = GenericWrapper().run_provider
                self.assertEqual(DatabaseImplementationDaemon._resolve_native_portal_provider_callback(wrapper, attempt), wrapper)
            finally:
                runtime._daemon.close()


if __name__ == "__main__":
    unittest.main()
