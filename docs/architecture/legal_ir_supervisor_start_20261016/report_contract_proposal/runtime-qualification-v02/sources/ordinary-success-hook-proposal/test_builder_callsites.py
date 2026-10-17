"""Actual proposed-builder assembly controls, never provider/native qualification.

These tests use actual ordinary builder functions and regular temporary files.
They neither start Docker/model processes nor fabricate native owners. Temporary
files must live in a unique genuinely precharged external qualification root.
The active context is explicitly a normative pure-assembly context, incapable
of performing dispatch or minting host/native acceptance authority.
"""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
BASE = Path(__file__).parent
PROPOSED = BASE / "proposed/ipfs_accelerate_py/agent_supervisor/runtime"
ACCELERATOR = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller")
SOURCE_FIXTURES = Path("/home/barberb/lift_coding/artifacts/legal-ir-worker-image-20261016-01/integration/test/api/test_agent_supervisor_ordinary_worker_image.py")
sys.path.insert(0, str(ACCELERATOR))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

H = module("ipfs_accelerate_py.agent_supervisor.runtime.ordinary_success_handoff", PROPOSED / "ordinary_success_handoff.py")
G = module("controlled_ordinary_success_runner", PROPOSED / "grok_cli_runner.py")
# Reuse existing actual source fixture constructors byte-for-byte. These only
# create inert regular test inputs; no old test assertions or native owners run.
fixture_tree = ast.parse(SOURCE_FIXTURES.read_text())
fixture_nodes = [node for node in fixture_tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_grok_inputs", "_codex_inputs"}]
fixture_namespace = {"G": G, "Path": Path}
exec(compile(ast.Module(body=fixture_nodes, type_ignores=[]), str(SOURCE_FIXTURES), "exec"), fixture_namespace)


class ActualBuilderAssemblyControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ordinary-handoff-builder-")
        self.base = Path(self.temporary.name).absolute()
        self.image = G._ordinary_task_toolchain_image_pin()
        self.no_process = patch.object(G.subprocess, "run", side_effect=AssertionError("assembly control must not invoke any process"))
        self.no_process.start()
        self.no_spawn = patch.object(G.subprocess, "Popen", side_effect=AssertionError("assembly control must not spawn any model or process"))
        self.no_spawn.start()

    def tearDown(self):
        self.no_spawn.stop()
        self.no_process.stop()
        self.temporary.cleanup()

    def data(self, provider):
        return fixture_namespace["_" + provider + "_inputs"](self.base, self.image)

    def normative_active(self):
        # No owner/process/native objects are inserted or mocked. Only pure
        # argv transformation is enabled; any productive hook would fail.
        return SimpleNamespace(task_active=True, selected_lease=object(), gated_create=None)

    def assert_gate(self, command, original):
        index = command.index(self.image)
        self.assertEqual(command[index + 1:index + 4], ["-I", "-c", H._START_GATE])
        self.assertEqual(command[index + 4], original)
        self.assertIn("--entrypoint=" + str(G._CODEX_TASK_TOOLCHAIN_PYTHON), command[:index])
        self.assertTrue({"--pull=never", "--read-only", "--interactive", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--network=bridge", "--runtime=runc"} <= set(command))
        self.assertNotIn(G._CODEX_TASK_TOOLCHAIN_IMAGE_ID, command)
        # Trusted host Docker control uses its local socket. The model may not
        # receive that socket through any other generated argument/mount.
        socket_control = "--host=unix:///var/run/docker.sock"
        self.assertIn(socket_control, command)
        self.assertFalse(any("docker.sock" in item for item in command if item != socket_control))

    def test_actual_grok_builder_none_context_preserves_bootstrap(self):
        command = G._docker_ordinary_grok_command(**self.data("grok"), task_execution=True)
        index = command.index(self.image)
        self.assertEqual(command[index + 1:index + 3], ["-I", str(G._GROK_CONTAINER_BOOTSTRAP)])
        self.assertNotIn(H._START_GATE, command)

    def test_actual_codex_builder_none_context_preserves_environment_entry(self):
        command = G._docker_ordinary_codex_fallback_command(**self.data("codex"))
        self.assertEqual(command[command.index(self.image) + 1], "-i")
        self.assertIn("--entrypoint=/usr/bin/env", command)
        self.assertNotIn(H._START_GATE, command)

    def test_actual_grok_builder_opt_in_wraps_its_generated_entrypoint(self):
        context = self.normative_active()
        token = H._ACTIVE.set(context)
        try:
            command = G._docker_ordinary_grok_command(**self.data("grok"), task_execution=True)
            self.assert_gate(command, str(G._CODEX_TASK_TOOLCHAIN_PYTHON))
            self.assertEqual(command[command.index(self.image) + 5:][:2], ["-I", str(G._GROK_CONTAINER_BOOTSTRAP)])
            self.assertEqual(context.gated_create, tuple(command))
        finally:
            H._ACTIVE.reset(token)

    def test_actual_codex_builder_opt_in_wraps_its_generated_environment_entry(self):
        context = self.normative_active()
        token = H._ACTIVE.set(context)
        try:
            command = G._docker_ordinary_codex_fallback_command(**self.data("codex"))
            self.assert_gate(command, "/usr/bin/env")
            self.assertEqual(command[command.index(self.image) + 5], "-i")
            self.assertEqual(context.gated_create, tuple(command))
        finally:
            H._ACTIVE.reset(token)

    def test_actual_generated_transform_preserves_all_non_entrypoint_options(self):
        command = G._docker_ordinary_codex_fallback_command(**self.data("codex"))
        index = command.index(self.image)
        transformed = H.gated_create_command(command, image=self.image, original_entrypoint="/usr/bin/env", python_entrypoint=str(G._CODEX_TASK_TOOLCHAIN_PYTHON))
        expected_prefix = list(command[:index])
        expected_prefix[expected_prefix.index("--entrypoint=/usr/bin/env")] = "--entrypoint=" + str(G._CODEX_TASK_TOOLCHAIN_PYTHON)
        self.assertEqual(transformed[:index], expected_prefix)
        self.assertEqual(transformed[index + 5:], command[index + 1:])
        self.assertEqual(command[index + 1], "-i")  # Input list was not mutated.

    def test_preflight_assembly_does_not_gain_task_gate(self):
        context = SimpleNamespace(task_active=False)
        token = H._ACTIVE.set(context)
        try:
            command = G._docker_ordinary_grok_command(**self.data("grok"), task_execution=True)
            self.assertNotIn(H._START_GATE, command)
        finally:
            H._ACTIVE.reset(token)

    def test_opt_in_transport_kwargs_require_actual_generated_gate(self):
        context = self.normative_active()
        token = H._ACTIVE.set(context)
        try:
            with self.assertRaisesRegex(H.OrdinaryHandoffRefusal, "actual_gated_create_before_transport_required"):
                H.ordinary_transport_kwargs()
            G._docker_ordinary_codex_fallback_command(**self.data("codex"))
            self.assertEqual(H.ordinary_transport_kwargs(), {"start_new_session": True, "stdin": subprocess.PIPE})
            self.assertEqual(H.ordinary_transport_kwargs(stdin_already_present=True), {"start_new_session": True})
        finally:
            H._ACTIVE.reset(token)

    def test_protected_builder_does_not_accept_ordinary_image(self):
        with self.assertRaisesRegex(ValueError, "pinned task-toolchain"):
            G._docker_grok_command(**self.data("grok"), task_execution=True)

    def test_create_transform_refuses_noninteractive_actual_command(self):
        command = G._docker_ordinary_codex_fallback_command(**self.data("codex"))
        command.remove("--interactive")
        with self.assertRaisesRegex(H.OrdinaryHandoffRefusal, "actual_interactive_inert_ordinary_create_required"):
            H.gated_create_command(command, image=self.image, original_entrypoint="/usr/bin/env", python_entrypoint=str(G._CODEX_TASK_TOOLCHAIN_PYTHON))


if __name__ == "__main__":
    unittest.main()
