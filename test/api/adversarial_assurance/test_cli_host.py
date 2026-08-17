"""Host registration tests for the AAE-056 assurance CLI group.

Acceptance covered here:

* Installed ``ipfs-accelerate assurance …`` host reaches mutate
  plan/run/target/explain and report.
* Registration is parser-only / cold-safe.
* Typed discovery surface is closed and stable.
* No arbitrary external repository root flags are published.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import cli as assurance_cli

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli.py"
)
HOST_CLI_PATH = REPO_ROOT / "ipfs_accelerate_py/cli.py"


# ---------------------------------------------------------------------------
# Discovery / registration
# ---------------------------------------------------------------------------


def test_discovery_manifest_is_static_and_closed() -> None:
    manifest = assurance_cli.assurance_cli_discovery_manifest()
    assert manifest["group"] == "assurance"
    assert manifest["interface"] == "AssuranceCLI@1"
    assert manifest["evidence"] == "aae/cli@1"
    assert manifest["console_entry"] == "ipfs-accelerate"
    assert manifest["cold_help"] is True
    assert manifest["side_effect_free_parse"] is True
    assert manifest["lazy_dispatch"] is True
    assert manifest["production_policy_change"] is False
    assert manifest["arbitrary_external_repositories"] is False
    assert manifest["explicit_run_authority_required"] is True
    assert manifest["honors_cancellation"] is True
    assert manifest["honors_resources"] is True
    assert set(manifest["commands"]) == {"mutate", "report"}
    assert set(manifest["mutate_commands"]) == {"plan", "run", "target", "explain"}
    assert set(manifest["campaign_commands"]) == {
        "mutate.plan",
        "mutate.run",
        "mutate.target",
        "mutate.explain",
        "report",
    }


def test_register_assurance_cli_is_parser_only() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    group = assurance_cli.register_assurance_cli(sub)
    assert "assurance" in group.prog

    help_io = io.StringIO()
    group.print_help(help_io)
    text = help_io.getvalue()
    assert "mutate" in text
    assert "report" in text

    # Nested mutate vocabulary.
    args = parser.parse_args(
        [
            "assurance",
            "mutate",
            "plan",
            "--repository-state-json",
            "state.json",
            "--manifest-json",
            "m.json",
            "--policy-json",
            "p.json",
            "--resource-budget-json",
            "b.json",
        ]
    )
    assert args.command == "assurance"
    assert args.assurance_command == "mutate"
    assert args.assurance_mutate_command == "plan"


def test_host_registers_all_required_commands() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    assurance_cli.register_assurance_cli(sub)

    for mutate_cmd in ("plan", "run", "target", "explain"):
        # parse known args enough to prove the path exists
        argv = ["assurance", "mutate", mutate_cmd, "--help"]
        with pytest.raises(SystemExit) as exited:
            parser.parse_args(argv)
        assert exited.value.code == 0

    with pytest.raises(SystemExit) as exited:
        parser.parse_args(["assurance", "report", "--help"])
    assert exited.value.code == 0


def test_run_requires_authorize_run_flag_in_parser() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    assurance_cli.register_assurance_cli(sub)
    args = parser.parse_args(
        [
            "assurance",
            "mutate",
            "run",
            "--plan-json",
            "plan.json",
            "--verification-policy-json",
            "vp.json",
            "--precomputed-reports-json",
            "reports.json",
            "--authorize-run",
        ]
    )
    assert args.authorize_run is True


def test_host_parser_rejects_unknown_assurance_command() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    assurance_cli.register_assurance_cli(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["assurance", "not-a-command"])


def test_no_arbitrary_external_repository_root_flags() -> None:
    """Host surface must not publish free-form external repo root options."""

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    group = assurance_cli.register_assurance_cli(sub)
    help_io = io.StringIO()
    group.print_help(help_io)
    text = help_io.getvalue().lower()
    # Free-form filesystem root options are forbidden on the public surface.
    for banned in (
        "--repository ",
        "--repository-root",
        "--repo-root",
        "--workdir",
        "--worktree",
    ):
        assert banned not in text


def test_product_cli_source_registers_assurance_group() -> None:
    source = HOST_CLI_PATH.read_text(encoding="utf-8")
    assert "register_assurance_cli" in source
    assert "run_assurance_cli" in source
    assert "assurance" in source
    # Sole AAE-056 owner: one import and one registration call.
    assert source.count("register_assurance_cli") == 2
    assert source.count("register_assurance_cli(subparsers)") == 1
    assert source.count("run_assurance_cli(args)") == 1


def test_cli_module_cold_import_is_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.cli"
    sys.modules.pop(module_name, None)

    before_threads = {t.ident for t in threading.enumerate()}
    started: list[str] = []
    real_start = threading.Thread.start

    def guarded_start(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        started.append(self.name)
        return real_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", guarded_start)

    mod = importlib.import_module(module_name)
    assert mod.ASSURANCE_CLI_INTERFACE == "AssuranceCLI@1"
    assert started == []
    after = {t.ident for t in threading.enumerate()}
    assert after - before_threads == set()


def test_cli_source_has_no_module_level_io() -> None:
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    forbidden_calls = {"open", "urlopen", "Popen", "run", "connect"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_calls:
                # Allow only inside functions (dispatch / loaders), not module body.
                # Module-level check: parent should not be Module.
                # ast.walk loses parents; approximate by checking top-level Expr/Assign.
                pass
    # Stronger: top-level statements must not be Call expressions that open resources.
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            pytest.fail("module-level call expression is not cold-safe")
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_calls:
                pytest.fail(f"module-level I/O call: {name}")


def test_cold_help_via_product_cli_subprocess() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.cli_entry",
            "assurance",
            "--help",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    combined = completed.stdout + completed.stderr
    assert "mutate" in combined
    assert "report" in combined


def test_cold_mutate_help_via_product_cli_subprocess() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.cli_entry",
            "assurance",
            "mutate",
            "--help",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    combined = completed.stdout + completed.stderr
    for name in ("plan", "run", "target", "explain"):
        assert name in combined


def test_resolve_dispatch_command_mapping() -> None:
    assert (
        assurance_cli.resolve_dispatch_command(
            SimpleNamespace(assurance_command="report")
        )
        == "report"
    )
    assert (
        assurance_cli.resolve_dispatch_command(
            SimpleNamespace(
                assurance_command="mutate",
                assurance_mutate_command="plan",
            )
        )
        == "mutate.plan"
    )
    with pytest.raises(assurance_cli.AssuranceCLIUsageError):
        assurance_cli.resolve_dispatch_command(
            SimpleNamespace(assurance_command="mutate", assurance_mutate_command=None)
        )


def test_run_assurance_cli_missing_command_is_usage() -> None:
    out = io.StringIO()
    err = io.StringIO()
    code = assurance_cli.run_assurance_cli(
        SimpleNamespace(assurance_command=None, output_human=False),
        stdout=out,
        stderr=err,
    )
    assert code == assurance_cli.EXIT_USAGE
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    assert payload["reason_code"] == "missing_assurance_command"


def test_path_helpers_reject_absolute_host_paths() -> None:
    assert assurance_cli.looks_like_host_path("/home/user/repo") is True
    assert assurance_cli.looks_like_host_path("C:\\Users\\x") is True
    assert assurance_cli.looks_like_host_path("src/mod.py") is False
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        assurance_cli.reject_path_exposure(
            {"repository_path": "/tmp/external-repo"},
            path="test",
        )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        assurance_cli.repo_relative_path("/etc/passwd", "source_path")
    assert assurance_cli.repo_relative_path("pkg/mod.py", "source_path") == "pkg/mod.py"


def test_emit_json_is_deterministic_and_sorted() -> None:
    out = io.StringIO()
    payload = {
        "z": 1,
        "a": 2,
        "ok": True,
        "command": "report",
        "status": "ok",
        "schema": assurance_cli.ASSURANCE_CLI_RESULT_SCHEMA,
        "interface": assurance_cli.ASSURANCE_CLI_INTERFACE,
        "evidence": assurance_cli.AAE_CLI_EVIDENCE,
        "exit_code": 0,
        "production_policy_change": False,
        "path_exposure": False,
        "side_effects": {
            "network": False,
            "process_spawn": False,
            "key_generation": False,
            "production_policy_change": False,
        },
    }
    assurance_cli.emit(payload, output_json=True, stream=out)
    first = out.getvalue()
    out2 = io.StringIO()
    assurance_cli.emit(payload, output_json=True, stream=out2)
    assert out2.getvalue() == first
    # sort_keys ensures key order independent of dict insertion.
    assert first.index('"a"') < first.index('"z"')


def test_human_emit_is_bounded() -> None:
    out = io.StringIO()
    payload = assurance_cli.envelope(
        ok=True,
        command="mutate.plan",
        status="planned",
        result={
            "plan_id": "p1",
            "plan_cid": "cid:plan",
            "summary": "planned",
            "reason_codes": ["no_production_policy_change"],
        },
    )
    assurance_cli.emit(payload, output_json=False, stream=out)
    text = out.getvalue()
    assert "mutate.plan" in text
    assert "plan_id=p1" in text
    assert text.count("\n") <= assurance_cli.MAX_HUMAN_LINES + 1
