"""Admitted actual public-parser regression; no Git/native/provider claims.

Root must run this test under genuine finite admission with bytecode disabled.
It imports the actual deployed daemon after exact source-pin verification and
parses the independently frozen actual64-task board. It does not fake a native
owner, invoke Git, dispatch a provider, accept a task or mutate task counters.
"""
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("admitted_v03_actual_public_parser_binding", Path(__file__).with_name("real_candidate_binding.py"))
binding = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = binding
spec.loader.exec_module(binding)

OBSERVER_SHA256 = '0f1b6a583f5b21afca2f47c4d88a399aa6c2273b7ee3616956823a8788947ba2'
PINNED_SOURCE_SHA256 = {'/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller/ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py': '23addb0dfc23e2cb093e7f74abbb7dc84a5308c24259e1eb0e25d721c0a80bed', '/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller/ipfs_accelerate_py/agent_supervisor/task_sources/task_identity.py': '4168c617ba079ca8fbeb2346931d4c7fcfb6516a09e5ddc7079434fb3541654e', '/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py': 'b9c8ca9f889366df93fd82887b7d6fcbfda9edee12a58a5667b485fe2c8a54ed', '/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py': '7c889bc858ff092e18bd2e97ff5a6407973e226eae33cddc9e5eef1d21f35986', '/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/existing_resource_scheduler.py': 'e19dbd913e30c6960d4a848bc2b2f15236944e55bc7718598e047fbdf664f18d', '/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py': 'd23e7f8a55d2563897b558502f05b11f89daa10f0ee6ff7dd2f142966ef7bce9', '/usr/bin/python3.12': '1a301bb1763139d48ae638d97b11edf56de6cd185e1b054eae6dc28c271c0c5f', '/usr/bin/git': 'aa6540695d076182256dd6e96c8b302e4d56381e3000bbfd5c71bbdfe94a4942', '/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/acceptance-adapter/adapter-v02/tests/fixtures/bootstrap/tasks.todo.md': '5244288f69562663f44b00d27add1ee6ef338f977fed1c0e1c8855725ff372c7', '/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/acceptance-adapter/adapter-v02/tests/fixtures/bootstrap/authoring/tasks.json': '66ebc6a95adde4fc717dd19ffaf5f03387f909662d1a47eccbb5c2a51857e9e6'}


class ActualPublicParserRegression(unittest.TestCase):
    def test_actual_public_parser_preserves_independently_frozen_nonsequential_plan(self):
        pins = dict(PINNED_SOURCE_SHA256)
        pins[str(Path(binding.__file__).absolute())] = OBSERVER_SHA256
        binding.verify_source_pins(pins)
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1",
            IPFS_ACCEL_SKIP_CORE="1", IPFS_AUTO_INSTALL="false", IPFS_DATASETS_AUTO_INSTALL="false")
        for path in (binding.ACCELERATOR, binding.DATASETS):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        daemon = importlib.import_module("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon")
        identity = importlib.import_module("ipfs_accelerate_py.agent_supervisor.task_sources.task_identity")
        self.assertEqual(Path(daemon.__file__).absolute(), binding.DAEMON)
        self.assertEqual(Path(identity.__file__).absolute(), binding.IDENTITY)
        self.assertEqual(Path(daemon.parse_task_text.__code__.co_filename).absolute(), binding.DAEMON)
        self.assertIs(daemon.parse_task_text.__globals__["canonical_task_identity"], identity.canonical_task_identity)
        raw = binding.read_pinned_source(binding.ORIGINAL_BOARD, binding.ORIGINAL_BOARD_SHA256)
        tasks = daemon.parse_task_text(raw.decode("utf-8"), path=binding.ORIGINAL_BOARD, task_header_prefix="## LIR-")
        self.assertEqual(tuple(task.task_id for task in tasks), binding.verify_original_inventory())
        result = binding.derive_task_binding(raw, binding.ORIGINAL_BOARD, tasks,
            identity.canonical_task_identity, daemon.task_declared_output_paths)
        self.assertEqual(result["task_id"], "LIR-9001")
        self.assertEqual(result["task_revision"], tasks[0].canonical_task_cid)
        self.assertEqual(result["expected_board_sha256"], binding.ORIGINAL_BOARD_SHA256)
        self.assertEqual(tasks[4].task_id, "LIR-0101")
        self.assertEqual(tasks[-1].task_id, "LIR-9108")
        self.assertEqual(sum(task.status == "blocked" for task in tasks), 63)
        self.assertEqual(tasks[0].status, "todo")
        self.assertEqual(tasks[0].completion, "manual")
        binding.verify_source_pins(pins)


if __name__ == "__main__":
    unittest.main()
