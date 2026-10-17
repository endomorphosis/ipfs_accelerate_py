"""Pure namespace propagation tests; no database, provider, or launch effects."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def _load_native_modules():
    # The supervisor captures a read-only Git source snapshot on import. This
    # unit fixture deliberately makes subprocess capture unavailable rather
    # than permitting Git, Docker or any provider process to run.
    with patch.object(subprocess, "run", side_effect=OSError("unit subprocess disabled")):
        from ipfs_accelerate_py.agent_supervisor.runtime import configured_board_scheduler
        from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor
    return configured_board_scheduler, implementation_supervisor, multi_supervisor_runner


class ConfiguredBoardNamespaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheduler, cls.supervisor, cls.runner = _load_native_modules()

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ))
        # Isolate only ambient program/provider choices. Preserve the native
        # owner's no-install flags, package pins, CUDA, thread limits and scratch.
        for name in (*self.runner.DATABASE_PROGRAM_ENV_NAMES,
                     *self.runner.STATE_CREDENTIAL_ENV_NAMES,
                     *self.scheduler.SCHEDULER_PROVIDER_ENV_NAMES):
            os.environ.pop(name, None)
        self.stack.enter_context(patch.object(subprocess, "run", side_effect=AssertionError("unmocked subprocess forbidden")))
        self.stack.enter_context(patch.object(subprocess, "Popen", side_effect=AssertionError("unmocked subprocess forbidden")))
        self.temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temporary).resolve()

    def board(self, branch="agent/legal-ir-grok-parallel-20261016", *, explicit_program=True):
        return self.scheduler.ConfiguredBoard(
            config_path=self.root / "scheduler.json", repo_root=self.root,
            payload={
                "stale_seconds": 1800, "check_interval_seconds": 30,
                "watchdog_startup_grace_seconds": 300, "max_restarts": 3,
                "max_task_attempts": 3, "daemon_interval_seconds": 60,
                "implementation_timeout_seconds": 7200,
                "implementation_max_timeout_seconds": 21600,
                "implementation_log_stall_seconds": 1200,
                "implementation_retry_budget": 3,
                "validation_retry_budget": 3, "merge_retry_budget": 3,
            },
            configuration_root="unit-only", configuration_revision="unit-only",
            taskboard_path="tasks.todo.md", objectives_path="objectives.md",
            plan_path="plan.md", validator_path="validate.py",
            dependency_validator_path="", dependency_seal_path="",
            task_prefix="LIR", board_namespace="legal-ir-span-graph-lean-refinement-20261016",
            merge_target_branch=branch, max_lanes=3, strict_task_sharding=True,
            idle_lane_work_stealing="", worktree_submodule_paths=(),
            protected_paths=(), live_capsule_control_paths=(),
            runtime_paths={"worktrees": "worktrees", "merge_queue": "merge-queue"},
            database_program=(self.runner.DatabaseProgramConfig.explicit_legacy_markdown()
                              if explicit_program else None),
        )

    def parsed_config(self, board):
        common = self.scheduler.configured_board_common_args(board, implement=False)
        parsed = self.supervisor.parse_args(list(common))
        return self.supervisor.supervisor_config_from_args(parsed, repo_root=self.root)

    def pure_supervisor(self, config):
        # Enter only the existing non-mutating database-authority constructor
        # branch. No connection or authority is constructed by this stand-in.
        config = replace(config, database_program=SimpleNamespace(
            authority_mode="quack", task_source_kind="duckdb"))
        with (patch.object(self.supervisor, "_control_plane_probe_root", return_value=self.root),
              patch.object(self.supervisor, "_captured_supervisor_reload_arguments", return_value=None)):
            return self.supervisor.PortalImplementationSupervisor(config)

    def test_common_args_preserve_declared_namespace_for_nonshared_branches(self):
        for branch in ("main", "agent/legal-ir-grok-parallel-20261016", "automation/another-board"):
            for explicit_program in (False, True):
                with self.subTest(branch=branch, explicit_program=explicit_program):
                    board = self.board(branch, explicit_program=explicit_program)
                    args = self.scheduler.configured_board_common_args(board, implement=False)
                    self.assertEqual(args.count("--board-namespace"), 1)
                    self.assertEqual(args[args.index("--board-namespace") + 1], board.board_namespace)
                    self.assertEqual(args[args.index("--merge-target-branch") + 1], branch)
                    self.assertEqual(args[args.index("--todo-path") + 1], str(board.path(board.taskboard_path)))

    def test_child_parser_and_config_preserve_explicit_namespace(self):
        board = self.board()
        config = self.parsed_config(board)
        self.assertEqual(config.board_namespace, board.board_namespace)
        self.assertEqual(config.merge_target_branch, board.merge_target_branch)
        self.assertEqual(config.todo_path, board.path(board.taskboard_path))

    def test_runner_common_arg_roundtrip_has_one_namespace(self):
        board = self.board()
        for defaults in (False, True):
            for implement in (False, True):
                with self.subTest(defaults=defaults, implement=implement):
                    common = self.scheduler.configured_board_common_args(board, implement=implement)
                    runner = self.runner.build_configured_multi_supervisor_cli_runner(
                        repo_root=self.root, stamp="unit-only", common_args=common,
                        implementation_supervisor_defaults=defaults)
                    parsed_runner = self.runner.build_arg_parser().parse_args(runner.args())
                    restored = self.runner.common_args_from_parsed_args(parsed_runner)
                    self.assertEqual(restored.count("--board-namespace"), 1)
                    self.assertEqual(restored.count("--implement"), int(implement))
                    self.assertEqual(restored.count("--no-implement"), int(not implement))
                    parsed_child = self.supervisor.parse_args(restored)
                    self.assertEqual(parsed_child.board_namespace, board.board_namespace)
                    self.assertEqual(parsed_child.implement, implement)

    def test_runner_keeps_default_mode_when_explicit_mode_is_absent(self):
        for defaults in (False, True):
            with self.subTest(defaults=defaults):
                runner = self.runner.build_configured_multi_supervisor_cli_runner(
                    repo_root=self.root, stamp="unit-only", common_args=("--once",),
                    implementation_supervisor_defaults=defaults)
                parsed = self.runner.build_arg_parser().parse_args(runner.args())
                restored = self.runner.common_args_from_parsed_args(parsed)
                self.assertEqual(restored.count("--implement"), int(defaults))
                self.assertEqual(restored.count("--no-implement"), 0)
                self.assertEqual(self.supervisor.parse_args(restored).implement, defaults)

    def test_runner_preserves_all_other_default_arguments(self):
        runner = self.runner.build_configured_multi_supervisor_cli_runner(
            repo_root=self.root, stamp="unit-only",
            implementation_supervisor_defaults=True,
            implementation_supervisor_command="unit-implementation-command")
        parsed = self.runner.build_arg_parser().parse_args(runner.args())
        baseline = self.runner.common_args_from_parsed_args(parsed)
        self.assertEqual(baseline[0], "--implement")
        parsed.common_arg = ["--no-implement", "--once"]
        restored = self.runner.common_args_from_parsed_args(parsed)
        self.assertEqual(restored, baseline[1:] + parsed.common_arg)
        self.assertIn("--objective-refill-scan", restored)
        self.assertIn("--codebase-refill-scan", restored)
        self.assertEqual(restored[restored.index("--implementation-command") + 1], "unit-implementation-command")

    def test_runner_does_not_normalize_conflicting_explicit_modes(self):
        for defaults in (False, True):
            for explicit in (("--implement", "--no-implement"), ("--no-implement", "--implement")):
                with self.subTest(defaults=defaults, explicit=explicit):
                    runner = self.runner.build_configured_multi_supervisor_cli_runner(
                        repo_root=self.root, stamp="unit-only", common_args=explicit,
                        implementation_supervisor_defaults=defaults)
                    parsed = self.runner.build_arg_parser().parse_args(runner.args())
                    restored = self.runner.common_args_from_parsed_args(parsed)
                    self.assertEqual(restored[-2:], list(explicit))
                    self.assertEqual(restored.count("--implement"), 1)
                    self.assertEqual(restored.count("--no-implement"), 1)
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as rejected:
                            self.supervisor.parse_args(restored)
                    self.assertEqual(rejected.exception.code, 2)

    def test_runner_preserves_repeated_explicit_modes(self):
        for defaults in (False, True):
            for flag in ("--implement", "--no-implement"):
                with self.subTest(defaults=defaults, flag=flag):
                    explicit = (flag, flag)
                    runner = self.runner.build_configured_multi_supervisor_cli_runner(
                        repo_root=self.root, stamp="unit-only", common_args=explicit,
                        implementation_supervisor_defaults=defaults)
                    parsed = self.runner.build_arg_parser().parse_args(runner.args())
                    restored = self.runner.common_args_from_parsed_args(parsed)
                    self.assertEqual(restored[-2:], list(explicit))
                    self.assertEqual(restored.count(flag), 2)
                    self.assertEqual(self.supervisor.parse_args(restored).implement, flag == "--implement")

    def test_duplicate_namespace_options_cannot_silently_override(self):
        namespace = self.board().board_namespace
        for second in (namespace, "different-board"):
            for tokenized in (("--board-namespace", second), ("--board-namespace=" + second,)):
                with self.subTest(second=second, tokenized=tokenized):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as rejected:
                            self.supervisor.parse_args(["--board-namespace", namespace, *tokenized])
                    self.assertEqual(rejected.exception.code, 2)

    def test_initializer_uses_explicit_namespace_and_keeps_legacy_inference(self):
        board = self.board()
        explicit = self.pure_supervisor(self.parsed_config(board))
        self.assertEqual(explicit.board_namespace, board.board_namespace)
        self.assertEqual(explicit.board_control_plane_status["board_namespace"], board.board_namespace)
        inferred = self.pure_supervisor(replace(self.parsed_config(board), board_namespace=""))
        self.assertEqual(inferred.board_namespace, "legal-ir-grok-parallel-20261016")

    def test_checkout_lock_fallback_retains_explicit_namespace(self):
        board = self.board()
        instance = self.supervisor.PortalImplementationSupervisor.__new__(self.supervisor.PortalImplementationSupervisor)
        instance.config = self.parsed_config(board)
        expected = self.root / "unit-lock"
        with patch.object(self.supervisor, "board_scoped_checkout_mutation_lock_path", return_value=expected) as observe:
            self.assertEqual(instance._repo_merge_lock_path(), expected)
        observe.assert_called_once_with(self.root, board.board_namespace)

    def test_daemon_command_retains_the_same_namespace_once(self):
        board = self.board()
        config = self.parsed_config(board)
        instance = self.pure_supervisor(config)
        instance.config = config
        command = instance._build_daemon_command()
        self.assertEqual(command.count("--board-namespace"), 1)
        self.assertEqual(command[command.index("--board-namespace") + 1], board.board_namespace)
        self.assertEqual(command[command.index("--merge-target-branch") + 1], board.merge_target_branch)

    def test_appended_runner_user_arg_cannot_replace_configured_namespace(self):
        board = self.board()
        common = self.scheduler.configured_board_common_args(board, implement=False)
        for defaults in (False, True):
            with self.subTest(defaults=defaults):
                runner = self.runner.build_configured_multi_supervisor_cli_runner(
                    repo_root=self.root, stamp="unit-only", common_args=common,
                    implementation_supervisor_defaults=defaults)
                parsed_runner = self.runner.build_arg_parser().parse_args([
                    *runner.args(), "--common-arg=--board-namespace=other-board"])
                restored = self.runner.common_args_from_parsed_args(parsed_runner)
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as rejected:
                        self.supervisor.parse_args(restored)
                self.assertEqual(rejected.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
