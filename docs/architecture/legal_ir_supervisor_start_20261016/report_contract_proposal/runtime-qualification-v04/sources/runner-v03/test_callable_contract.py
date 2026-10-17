"""Pure input/parser and source-callsite controls; no project import or grants.

Definitions are extracted from exact authored source through AST. Synthetic
fd numbers below are constructor-only data; no fake native owner or process is
used as admission. Actual FD/owner/gate/runtime qualification is separate.
"""
import argparse
import ast
import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import io
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "proposed/ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py"
BASELINE = Path("/home/barberb/lift_coding/artifacts/legal-ir-real-handoff-20261017-01/ordinary-success-hook-v01/proposed/ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py")
SOURCE = RUNNER.read_text()
TREE = ast.parse(SOURCE)
NAMES = {"OrdinaryCallableRefusal", "_callable_need", "OrdinaryCallableInvocation",
         "check_callable_options", "_EffectFreeRunnerParser", "build_runner_parser", "parse_runner_arguments"}
CONSTANTS = {"MAX_CALLABLE_PROMPT_BYTES", "MAX_CALLABLE_ENVIRONMENT_BYTES", "MAX_CALLABLE_OUTPUT_BYTES",
             "DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT", "CODEX_QUOTA_FALLBACK_REASONING_EFFORTS",
             "CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG", "GROK_INVOCATION_BINDING_FLAG", "GROK_INVOCATION_ID_FLAG"}
NS = dict(__name__=__name__, __file__=str(RUNNER), argparse=argparse, dataclass=dataclass, field=field,
          Mapping=Mapping, Sequence=Sequence, MappingProxyType=MappingProxyType, Path=Path,
          re=re, os=os, sys=sys, builtins=builtins)
SELECTED = [node for node in TREE.body if
            isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in NAMES or
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in CONSTANTS
                                                for target in node.targets)]
exec(compile(ast.Module(body=SELECTED, type_ignores=[]), str(RUNNER), "exec"), NS)
Invocation, Refusal = NS["OrdinaryCallableInvocation"], NS["OrdinaryCallableRefusal"]


def valid(**changes):
    values = dict(prompt="a bounded legal task", cwd=Path("/fixture/workspace"),
                  environment={"HOME": "/fixture/account", "PATH": "/usr/bin:/bin",
                               "TMPDIR": "/fixture/temp", "TMP": "/fixture/temp", "TEMP": "/fixture/temp"},
                  temporary_root=Path("/fixture/temp"), stdout_path=Path("/fixture/output/stdout"),
                  stderr_path=Path("/fixture/output/stderr"), stdout_fd=10, stderr_fd=11, output_limit_bytes=1024)
    values.update(changes)
    return Invocation(**values)


def args(**changes):
    value = NS["parse_runner_arguments"](["--workspace", "/fixture/workspace", "--grok-bin", "/usr/bin/grok", "--model", "grok-4.7"])
    for key, item in changes.items():
        setattr(value, key, item)
    return value


class CallableInputControls(unittest.TestCase):
    def test_valid_inputs_are_immutable_snapshot(self):
        env = dict(valid().environment)
        item = valid(environment=env)
        env["HOME"] = "/changed"
        self.assertEqual(item.environment["HOME"], "/fixture/account")
        with self.assertRaises(TypeError): item.environment["HOME"] = "/changed"

    def test_prompt_and_environment_are_not_in_repr(self):
        item = valid(prompt="secret-prompt", environment={**dict(valid().environment), "XAI_API_KEY": "secret-auth"})
        self.assertNotIn("secret-prompt", repr(item))
        self.assertNotIn("secret-auth", repr(item))

    def test_constructor_is_not_serializable(self):
        with self.assertRaisesRegex(Refusal, "not_serializable"): valid().__reduce__()

    def test_empty_prompt_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_prompt_invalid"): valid(prompt=" ")

    def test_nul_prompt_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_prompt_invalid"): valid(prompt="a\0b")

    def test_large_prompt_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_prompt_invalid"): valid(prompt="a" * (2 * 1024 * 1024 + 1))

    def test_relative_cwd_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_absolute_paths_required"): valid(cwd=Path("relative"))

    def test_parent_path_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_absolute_paths_required"): valid(stdout_path=Path("/fixture/../output"))

    def test_relative_peer_environment_path_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_relative_environment_path_refused"): valid(environment={**dict(valid().environment), "GROK_HOME": "relative"})

    def test_missing_home_refuses(self):
        env = dict(valid().environment); del env["HOME"]
        with self.assertRaisesRegex(Refusal, "callable_explicit_home_and_path_required"): valid(environment=env)

    def test_relative_path_entry_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_absolute_home_and_path_required"): valid(environment={**dict(valid().environment), "PATH": "/usr/bin:relative"})

    def test_empty_path_entry_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_absolute_home_and_path_required"): valid(environment={**dict(valid().environment), "PATH": "/usr/bin:"})

    def test_ambient_temp_fallback_refuses(self):
        env = dict(valid().environment); del env["TEMP"]
        with self.assertRaisesRegex(Refusal, "callable_explicit_temporary_environment_required"): valid(environment=env)

    def test_invalid_env_key_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_environment_invalid"): valid(environment={**dict(valid().environment), "BAD=KEY": "value"})

    def test_large_env_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_environment_too_large"): valid(environment={**dict(valid().environment), "LARGE": "x" * (256 * 1024)})

    def test_global_stream_fd_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_borrowed_output_fds_required"): valid(stdout_fd=1)

    def test_shared_fd_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_borrowed_output_fds_required"): valid(stderr_fd=10)

    def test_boolean_fd_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_borrowed_output_fds_required"): valid(stdout_fd=True)

    def test_unbounded_output_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_output_bound_invalid"): valid(output_limit_bytes=16 * 1024 * 1024 + 1)

    def test_valid_primary_options(self):
        NS["check_callable_options"](args(), -1, valid(), ambient_environment={})

    def test_fallback_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_primary_only_scope_required"):
            NS["check_callable_options"](args(codex_fallback_command_json='["codex"]'), -1, valid())

    def test_preflight_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_primary_only_scope_required"):
            NS["check_callable_options"](args(canonical_legacy_preflight_route=True), -1, valid())

    def test_real_parser_invocation_id_flag_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_primary_only_scope_required"):
            NS["check_callable_options"](args(invocation_id="not-canonical"), -1, valid())

    def test_receipt_fd_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_receipt_dispatch_refused"):
            NS["check_callable_options"](args(), 3, valid())

    def test_ambient_receipt_env_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_receipt_environment_refused"):
            NS["check_callable_options"](args(), -1, valid(), ambient_environment={"IPFS_ACCELERATE_GROK_TERMINAL_RECEIPT_FD": ""})

    def test_workspace_drift_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_explicit_workspace_model_binary_required"):
            NS["check_callable_options"](args(workspace=Path("/other")), -1, valid())

    def test_implicit_binary_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_explicit_workspace_model_binary_required"):
            NS["check_callable_options"](args(grok_bin="grok"), -1, valid())

    def test_chat_scope_refuses(self):
        with self.assertRaisesRegex(Refusal, "callable_agent_mode_required"):
            NS["check_callable_options"](args(mode="chat"), -1, valid())


class ParserAndCallsiteControls(unittest.TestCase):
    def test_parser_does_not_read_environment_or_provider(self):
        class NoAmbientEnvironment:
            def __getattr__(self, name):
                raise AssertionError("ambient environment or OS operation: " + name)
        with mock.patch.dict(NS, os=NoAmbientEnvironment()):
            parsed = NS["parse_runner_arguments"](["--workspace", "/fixture/workspace"])
        self.assertFalse(parsed.receipt_fd_declared)

    def test_parser_help_and_invalid_emit_no_global_output(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            for raw in (["--help"], [], ["--unknown"]):
                with self.assertRaises(Refusal): NS["parse_runner_arguments"](raw)
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))

    def test_parser_requires_explicit_sequence(self):
        with self.assertRaisesRegex(Refusal, "explicit_callable_argv_required"): NS["parse_runner_arguments"]("--workspace /tmp")

    def test_parser_bounds_explicit_argv(self):
        with self.assertRaisesRegex(Refusal, "bounded_callable_argv_required"): NS["parse_runner_arguments"](["a"] * 129)

    def test_cli_parser_definition_preserved(self):
        old = ast.parse(BASELINE.read_text())
        main = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        prefix = []
        for node in main.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "raw_argv" for t in node.targets): break
            prefix.append(node)
        builder = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "build_runner_parser")
        actual = ast.parse(ast.unparse(ast.Module(body=builder.body[:-1], type_ignores=[])))
        actual.body[0].value.func = ast.Attribute(value=ast.Name(id="argparse", ctx=ast.Load()), attr="ArgumentParser", ctx=ast.Load())
        self.assertEqual(ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False), ast.dump(actual, include_attributes=False))

    def test_protected_route_local_is_not_callable_parameter(self):
        run = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run")
        self.assertIn("callable_invocation", [a.arg for a in run.args.kwonlyargs])
        self.assertNotIn("invocation", [a.arg for a in run.args.kwonlyargs])
        self.assertIn("invocation = route_plan.invocation_binding", ast.get_source_segment(SOURCE, run))
        self.assertNotIn("fingeremit", SOURCE)

    def test_callable_transport_uses_explicit_cwd_and_two_pipes(self):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_run_callable_transport")
        popen = next(n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen")
        values = {k.arg: ast.unparse(k.value) for k in popen.keywords if k.arg}
        self.assertEqual(values["cwd"], "invocation.cwd")
        self.assertEqual(values["stdout"], "subprocess.PIPE")
        self.assertEqual(values["stderr"], "subprocess.PIPE")
        self.assertTrue(any(isinstance(n, ast.Try) and n.finalbody for n in node.body))

    def test_guarded_mux_and_actual_returned_hooks_present(self):
        body = ast.get_source_segment(SOURCE, next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_run_callable_transport"))
        self.assertIn("ordinary_provider_guard()", body)
        self.assertIn("select.select(tuple(streams), (), (), 0.5)", body)
        self.assertIn("ordinary_provider_returned(process, returncode)", body)
        self.assertIn("callable_received_output_limit_exceeded", body)

    def test_create_intent_and_known_cid_order(self):
        body = ast.get_source_segment(SOURCE, next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_create_grok_container_and_build_start_command"))
        self.assertLess(body.index("ordinary_container_creation_started"), body.index("subprocess.run"))
        self.assertLess(body.index("_require_startable_docker_container"), body.index("ordinary_container_created"))

    def test_pending_cleanup_gates_all_private_paths(self):
        body = ast.get_source_segment(SOURCE, next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_run"))
        self.assertIn("ordinary_private_allocations_created(isolated_home, Path(prompt_path))", body)
        self.assertIn("ordinary_allocation_close_permitted(docker_lease)", body)
        self.assertIn("if isolated_home is not None and allocation_close_permitted:", body)
        self.assertIn("if prompt_path and allocation_close_permitted:", body)

    def test_callable_mount_join_is_only_exact_readonly_grok_destination(self):
        hooks = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "ordinary_callable_binary_mount_allowed"]
        self.assertEqual(len(hooks), 1)
        hook = hooks[0]
        self.assertEqual(ast.unparse(hook.args[0]), "resolved_source")
        guarded = [n for n in ast.walk(TREE) if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And)
                   and any(part is hook for part in ast.walk(n))]
        self.assertTrue(any('provider == \'grok\'' in ast.unparse(n)
            and "destination == Path('/opt/ipfs-accelerate/grok')" in ast.unparse(n)
            and "_resolve_trusted_grok_bin" in ast.unparse(n) for n in guarded))
        original = ast.parse(BASELINE.read_text())
        old_resolver = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_trusted_grok_bin")
        new_resolver = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_trusted_grok_bin")
        self.assertEqual(ast.dump(old_resolver, include_attributes=False), ast.dump(new_resolver, include_attributes=False))

    def test_callable_actual_trusted_resolver_and_host_join_precede_route(self):
        run = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
        body = ast.get_source_segment(SOURCE, run)
        self.assertIn("if (codex_fallback_command or callable_invocation is not None) and protected_recovery_reservation is None:", body)
        resolver = body.index("grok_bin = _resolve_trusted_grok_bin(")
        refused = body.index("if not grok_bin:", resolver)
        hook = body.index("ordinary_trusted_binary_resolved(grok_bin)", refused)
        route = body.index("_ordinary_handoff.select_ordinary_route", hook)
        temporary = body.index("tempfile.NamedTemporaryFile", route)
        self.assertLess(resolver, refused)
        self.assertLess(refused, hook)
        self.assertLess(hook, route)
        self.assertLess(route, temporary)
        # Legacy CLI selects the same old condition when no callable exists.
        self.assertTrue("codex_fallback_command" in body[resolver - 180:resolver])

    def test_callable_guard_joins_actual_invocation_identity(self):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_callable_run")
        self.assertIn("ordinary_callable_context_required(invocation)", ast.get_source_segment(SOURCE, node))

    def test_public_validator_and_run_share_actual_preeffect_validator(self):
        validator = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "validate_ordinary_callable_invocation")
        run = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
        self.assertIn("_validate_callable_run(args, receipt_fd, invocation)", ast.get_source_segment(SOURCE, validator))
        self.assertIn("_validate_callable_run(args, receipt_fd, callable_invocation)", ast.get_source_segment(SOURCE, run))


if __name__ == "__main__":
    unittest.main()
