"""The serialization fix uses native steer and preserves admitted task history."""
import hashlib
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import CampaignConfig, atomic_json, build_population
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_migration import migrate
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from test.api.test_worldwide_legal_search_runtime import plan_fixture, FOREST


class ValidationMigration(unittest.TestCase):
    def fixture(self, root):
        site, framework, state = root / "site", root / "framework", root / "state"
        for repository in (site, framework):
            repository.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "fixture", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "-c", "user.email=fixture@example.invalid", "-c", "user.name=Fixture", "commit", "--allow-empty", "-qm", "Fixture"], check=True)
        state.mkdir()
        plan = plan_fixture()
        atomic_json(site / "plan.json", plan)
        atomic_json(site / "watchdog.policy.json", {"policy_id": "fixture", "repair_scope": {"allowed_paths": []}})
        config = CampaignConfig(site, framework, state, site / "plan.json", root / "config.json", run_id="fixture-worldwide")
        population = build_population(plan, source_forest=FOREST)
        for task in population["tasks"]:
            task["validations"] = [shlex.join(argv) for argv in next(row for row in plan["tasks"] if row["id"] == task["task_alias"])["validation_argv"]]
        with DatabaseTaskSource(config.database, owner_id="fixture") as source:
            source.materialize(population)
        return config, plan

    def test_native_steer_corrects_argv_without_lifecycle_or_budget_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            config, plan = self.fixture(Path(directory))
            fence = lambda _config: {"owner_alive": False, "main_pid": 0, "cgroup_empty": True, "test_fixture": True, "observed_at": 1.25}
            atomic_json(config.state_root / "bootstrap-receipt.json", {"created_at": 1.25, "native_bootstrap_fixture": True})
            with DatabaseTaskSource(config.database, owner_id="fixture-before") as source:
                before = source.plan_projection()
            report = migrate(config, apply=True, fence_probe=fence)
            self.assertTrue(report["applied"])
            self.assertEqual(report["native_apply_receipt"]["state"], "committed")
            with DatabaseTaskSource(config.database, owner_id="fixture-after") as source:
                after = source.plan_projection()
                self.assertEqual({row["task_cid"] for row in before["tasks"]}, {row["task_cid"] for row in after["tasks"]})
                for task in after["tasks"]:
                    old = next(row for row in before["tasks"] if row["task_cid"] == task["task_cid"])
                    self.assertEqual(task["status"], old["status"])
                    self.assertEqual(task["body"], old["body"])
                    self.assertEqual([row["argv"] for row in task["validations"]], next(row["validation_argv"] for row in plan["tasks"] if row["id"] == task["task_alias"]))
            self.assertFalse(migrate(config, apply=True, fence_probe=fence)["changed"])

    def test_live_owner_is_refused_before_native_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _plan = self.fixture(Path(directory))
            digest = hashlib.sha256(config.database.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                migrate(config, apply=True, fence_probe=lambda _config: {"owner_alive": True, "main_pid": 42, "cgroup_empty": False})
            self.assertEqual(hashlib.sha256(config.database.read_bytes()).hexdigest(), digest)
