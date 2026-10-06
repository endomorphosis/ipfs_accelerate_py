"""Run with Harbor's Python via unittest; no provider or Docker calls."""
import unittest
import json
import tempfile
import shlex
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from pathlib import Path
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest

from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import config_for
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent, measured_usage, security_asset_arguments
from benchmarks.agent_supervisor.container_coding.terminal_deployment import (
    ROOT, SECURITY_INITIALIZER_PATH, SECURITY_INITIALIZER_DESCRIPTOR, CANONICAL_CVE_PATH,
    SECURITY_CHECKPOINT_PATH, SECURITY_CHECKPOINT_HUB,
)
from benchmarks.agent_supervisor.container_coding.native_codex_baseline import config_for as baseline_config


class ReceiptTests(unittest.TestCase):
    def test_only_full_arm_receives_frozen_model_and_pinned_hub_provenance(self):
        checkpoint = {"path": SECURITY_CHECKPOINT_PATH, "manifest_sha256": "a" * 64,
            "descriptor": {"output": ROOT + "/" + SECURITY_CHECKPOINT_PATH,
                "manifest_sha256": "a" * 64, "checkpoint_sha256": "b" * 64},
            "hub": {"revision": "c" * 40}, "hub_descriptor_path": SECURITY_CHECKPOINT_HUB,
            "mode": "frozen_inference", "runtime_training_steps": 0, "runtime_download_calls": 0}
        manifest = {"security_checkpoint": checkpoint}
        args, observation = security_asset_arguments(manifest, "full")
        self.assertEqual(args, ["--security-checkpoint", ROOT + "/" + SECURITY_CHECKPOINT_PATH,
            "--security-checkpoint-manifest-sha256", "a" * 64,
            "--security-checkpoint-hub-descriptor", ROOT + "/" + SECURITY_CHECKPOINT_HUB])
        self.assertEqual(observation["security_checkpoint"]["hub"], checkpoint["hub"])
        self.assertEqual(security_asset_arguments(manifest, "no-index"), ([], {}))
        with self.assertRaises(ValueError):
            security_asset_arguments({**manifest, "security_initializer": {}}, "full")

    def test_only_full_arm_receives_explicit_portable_training_assets(self):
        manifest = {
            'security_initializer': {'path': SECURITY_INITIALIZER_PATH,
                'descriptor_path': SECURITY_INITIALIZER_DESCRIPTOR, 'source_checkpoint_included': False,
                'descriptor': {'output': ROOT + '/' + SECURITY_INITIALIZER_PATH,
                    'manifest_sha256': 'a' * 64, 'initializer_sha256': 'b' * 64,
                    'source_checkpoint_sha256': 'c' * 64}},
            'canonical_cve_training': {'path': CANONICAL_CVE_PATH, 'raw_source_included': False,
                'manifest_sha256': 'd' * 64, 'canonical_dataset_cid': 'authored-cid',
                'canonical_record_count': 5, 'training_pair_count': 2},
        }
        args, observation = security_asset_arguments(manifest, 'full')
        self.assertEqual(args, ['--security-initializer', ROOT + '/' + SECURITY_INITIALIZER_DESCRIPTOR,
            '--canonical-cve-export', ROOT + '/' + CANONICAL_CVE_PATH,
            '--canonical-cve-manifest-sha256', 'd' * 64])
        self.assertEqual(observation['security_initializer']['source_checkpoint_sha256'], 'c' * 64)
        self.assertEqual(security_asset_arguments(manifest, 'no-index'), ([], {}))
        self.assertEqual(security_asset_arguments({}, 'full'), ([], {}))

    def test_nonportable_asset_binding_is_rejected_before_run(self):
        with self.assertRaises(ValueError):
            security_asset_arguments({'canonical_cve_training': {
                'path': '../raw-originals', 'raw_source_included': True}}, 'full')
        with self.assertRaises(ValueError):
            security_asset_arguments({'security_initializer': {
                'path': SECURITY_INITIALIZER_PATH, 'descriptor_path': SECURITY_INITIALIZER_DESCRIPTOR,
                'source_checkpoint_included': True}}, 'full')

    def test_failed_and_successful_calls_both_count_cached_input_once(self):
        rows = []
        for identity, status, complete in (("planner", "provider_returned", True), ("coding", "failed", False)):
            rows.append({"invocation_id": identity, "status": status, "native_rollout_usage": {
                "usage": {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10, "total_tokens": 110},
                "task_complete_observed": complete}})
        value = measured_usage({"provider_invocations": [*rows, rows[0]]})
        self.assertEqual(value["input_tokens"], 200)
        self.assertEqual(value["cached_input_tokens"], 160)
        self.assertEqual(value["total_tokens"], 220)
        self.assertEqual(value["provider_calls"], 2)
        self.assertFalse(value["observed_complete_sessions"])
        self.assertIsNone(value["dollar_cost"])

    def test_absent_or_partial_usage_stays_unknown(self):
        self.assertIsNone(measured_usage({})["input_tokens"])
        rows = [{"invocation_id": "a", "native_rollout_usage": {"usage": {"input_tokens": 2}}},
                {"invocation_id": "b", "usage": {"prompt_tokens": 200}}]
        self.assertIsNone(measured_usage({"provider_invocations": rows})["input_tokens"])
        self.assertIsNone(measured_usage({"provider_invocations": rows[:1],
                                         "unreceipted_provider_attempt": {"phase": "coding"}})["input_tokens"])

    def test_conflicting_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            measured_usage({"provider_invocations": [{"invocation_id": "x"}, {"invocation_id": "x", "status": "failed"}]})

    def test_all_arms_keep_original_task_resources_verifier_and_agent_budget(self):
        from harbor.models.job.config import JobConfig
        base = baseline_config(Path("/dataset"), Path("/output"))
        for arm in ("full", "no-index"):
            value = config_for(Path("/dataset"), Path("/output"), Path("/archive"), arm)
            JobConfig.model_validate(value, extra="forbid")
            for key in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials", "retry", "timeout_multiplier"):
                self.assertEqual(value[key], base[key])
            for key in ("model_name", "override_timeout_sec", "max_timeout_sec"):
                self.assertEqual(value["agents"][0][key], base["agents"][0][key])
            self.assertEqual(value["agents"][0]["kwargs"]["arm"], arm)


class ExportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = 'benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent.'
        self.before = AsyncMock(return_value={'status': 'captured'})
        self.after = AsyncMock(return_value={'status': 'captured'})
        for name, mock in (('capture_public_inputs', self.before), ('export_public_outputs', self.after)):
            scope = patch(module + name, mock)
            scope.start()
            self.addCleanup(scope.stop)

    async def test_symbolic_contract_uses_existing_option_and_preserves_strategy_identity(self):
        from harbor.models.agent.context import AgentContext
        from benchmarks.agent_supervisor.container_coding.test_terminal_intent_requirement_planning import _requirements
        from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import _intent_selection
        commands, uploads = [], []
        class Environment:
            async def upload_file(self, source, target):
                uploads.append(target)
            async def exec(self, **kwargs):
                commands.append(shlex.split(kwargs["command"]))
                return SimpleNamespace(stdout="", stderr="", return_code=0)
            async def download_file(self, source, target):
                raise FileNotFoundError(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(_supervisor_manifest({"archive_sha256": "authored", "learned_requirements": []})))
            instruction = root / "public.md"
            instruction.write_text("Inspect bottle.py, repair problems and write report.jsonl with file_path and cwe_id fields.")
            contract = _requirements(instruction, symbolic=True)
            agent = FullSupervisorAgent(logs_dir=root / "logs", model_name="gpt-6.1-sol",
                runtime_archive=str(root), arm="no-index", intent_requirement_contract=contract)
            context = AgentContext()
            await agent.run(instruction.read_text(), Environment(), context)
            assert context.metadata["planning_strategy"] == "intent_symbolic"
            for key, value in _intent_selection(contract).items():
                assert context.metadata[key] == value
            assert commands[0].count("--intent-requirement-contract") == 1
            assert ROOT + "/intent-requirements.json" in uploads
            assert json.loads((root / "logs/intent-requirements.json").read_text()) == contract
            assert context.metadata["task_completed"] is False

    async def test_driver_failure_still_exports_outputs_and_capture_failure_does_not_mask_it(self):
        from harbor.models.agent.context import AgentContext
        events = []
        async def before(*args):
            events.append('before')
            raise ValueError('authored capture failure')
        async def after(*args):
            events.append('after')
            return {'status': 'captured'}
        self.before.side_effect = before
        self.after.side_effect = after
        class Environment:
            async def upload_file(self, source, target):
                pass
            async def exec(self, **kwargs):
                events.append('driver')
                raise RuntimeError('authored driver failure')
            async def download_file(self, source, target):
                events.append('public-download')
                raise FileNotFoundError(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(_supervisor_manifest({'archive_sha256': 'authored', 'learned_requirements': []})))
            agent = FullSupervisorAgent(logs_dir=root / 'logs', model_name='gpt-6.1-sol',
                runtime_archive=str(root), arm='no-index')
            context = AgentContext()
            with self.assertRaisesRegex(RuntimeError, 'driver failure'):
                await agent.run('Public instruction', Environment(), context)
            self.assertEqual(events, ['before', 'driver', 'public-download', 'public-download', 'after'])
            self.assertEqual(context.metadata['public_input_capture_error'], 'ValueError')
            self.assertEqual(context.metadata['public_output_evidence'], {'status': 'captured'})

    async def test_asset_flags_reach_full_execution_and_are_absent_from_no_index(self):
        from harbor.models.agent.context import AgentContext
        calls = []
        report = {'task_completed': True, 'seconds': 1, 'phases': {}, 'provider_invocations': []}

        class Environment:
            async def upload_file(self, source, target):
                pass

            async def exec(self, **kwargs):
                calls.append(shlex.split(kwargs['command']))
                return SimpleNamespace(return_code=0, stdout='', stderr='')

            async def download_file(self, source, target):
                Path(target).write_text(json.dumps(report if source.endswith('-result.json') else {}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'archive_sha256': 'authored', 'learned_requirements': [],
                'security_initializer': {'path': SECURITY_INITIALIZER_PATH,
                    'descriptor_path': SECURITY_INITIALIZER_DESCRIPTOR, 'source_checkpoint_included': False,
                    'descriptor': {'output': ROOT + '/' + SECURITY_INITIALIZER_PATH,
                        'manifest_sha256': 'a' * 64, 'initializer_sha256': 'b' * 64,
                        'source_checkpoint_sha256': 'c' * 64}},
                'canonical_cve_training': {'path': CANONICAL_CVE_PATH, 'raw_source_included': False,
                    'manifest_sha256': 'd' * 64, 'canonical_dataset_cid': 'authored-cid',
                    'canonical_record_count': 5, 'training_pair_count': 2}}
            (root / 'manifest.json').write_text(json.dumps(_supervisor_manifest(manifest)))
            for arm in ('full', 'no-index'):
                agent = FullSupervisorAgent(logs_dir=root / arm, model_name='gpt-6.1-sol',
                    runtime_archive=str(root), arm=arm)
                context = AgentContext()
                await agent.run('Authored public instruction', Environment(), context)
                self.assertEqual(bool(context.metadata['security_training_assets']), arm == 'full')
            self.assertIn('--security-initializer', calls[0])
            self.assertIn(ROOT + '/' + SECURITY_INITIALIZER_DESCRIPTOR, calls[0])
            self.assertIn('--canonical-cve-export', calls[0])
            self.assertIn('--canonical-cve-manifest-sha256', calls[0])
            self.assertNotIn('--security-initializer', calls[1])
            self.assertNotIn('--canonical-cve-export', calls[1])
            self.assertNotIn('--canonical-cve-manifest-sha256', calls[1])

    async def test_failed_run_preserves_admission_without_exporting_owner_state(self):
        from harbor.models.agent.context import AgentContext
        requested = []
        report = {"task_completed": False, "seconds": 1, "phases": {},
                  "error": {"type": "ContractError"}, "provider_invocations": []}

        class Environment:
            async def upload_file(self, source, target):
                pass

            async def exec(self, **kwargs):
                return SimpleNamespace(return_code=1, stdout="", stderr="")

            async def download_file(self, source, target):
                requested.append(source)
                Path(target).write_text(json.dumps(report if source.endswith("-result.json") else
                                                   {"graph": {}, "receipt": {}, "manifest": {}}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(_supervisor_manifest({"archive_sha256": "test", "learned_requirements": []})))
            agent = FullSupervisorAgent(logs_dir=root / "logs", model_name="gpt-6.1-sol",
                                        runtime_archive=str(root), arm="no-index")
            context = AgentContext()
            await agent.run("Public benchmark instruction", Environment(), context)
            self.assertTrue(context.metadata["signed_admission_exported"])
            self.assertFalse(context.metadata["task_completed"])
            self.assertIsNone(context.n_input_tokens)
            self.assertEqual(requested, ["/opt/ipfs-supervisor/state/benchmark-result.json",
                                         "/opt/ipfs-supervisor/state/benchmark/admission.json"])
            self.before.assert_awaited_once()
            self.after.assert_awaited_once()
            self.assertEqual(context.metadata['public_output_evidence'], {'status': 'captured'})


if __name__ == "__main__":
    unittest.main()
