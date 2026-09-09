"""PCCE-040: CLI parser/context plus init, scan, status, and plan commands."""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.proof_context.cli.app import (
    EXIT_CODES,
    INFERS_CURRENT_DIRECTORY,
    OUTPUT_MODES,
    PROG,
    PROVIDER_BOUND,
    REQUIRED_ARGUMENTS,
    SIBLING_LAYOUT_REQUIRED,
    STATE_COMMANDS,
    USAGE_COMMAND,
    USAGE_CORRELATION,
    USAGE_REPOSITORY,
    USAGE_TASK,
    CliContext,
    build_parser,
    exit_code_for,
    main,
)
from ipfs_accelerate_py.proof_context.cli.state_commands import (
    COMMANDS,
    cmd_init,
    cmd_plan,
    cmd_scan,
    cmd_status,
)
from ipfs_accelerate_py.proof_context.facade import OPERATION_CONTRACTS
from ipfs_accelerate_py.proof_context.policy import LIVE_MODES, MODES, STATUSES
from ipfs_accelerate_py.proof_context.results import is_success

APP_PATH = Path(inspect.getfile(main))
STATE_PATH = Path(inspect.getfile(cmd_init))
PACKAGE_ROOT = Path(__file__).resolve().parents[3]

PROMOTION_ENV_VARS = (
    "PCCE_MODE",
    "PCCE_RUNTIME_MODE",
    "IPFS_ACCELERATE_PCCE_MODE",
    "PROOF_CONTEXT_MODE",
    "PROOF_CONTEXT_RUNTIME_MODE",
)


def _invoke(argv: Sequence[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(list(argv), stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def _payload(stdout: str) -> dict[str, Any]:
    data = json.loads(stdout)
    assert isinstance(data, dict)
    return data


def _args(
    command: str,
    repository: Path,
    *,
    policy: str = "production",
    task: str = "PCCE-040",
    correlation: str = "corr-pcce-040",
    output_mode: str = "json",
    extra: Sequence[str] = (),
) -> list[str]:
    values = [
        "--repository",
        str(repository),
        "--policy",
        policy,
        "--task",
        task,
        "--correlation",
        correlation,
        "--output-mode",
        output_mode,
        command,
        *extra,
    ]
    return values


def _module_level_call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()

    def _name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Expr):
            value = node.value
        elif isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if isinstance(value, ast.Call):
            name = _name(value.func)
            if name:
                names.add(name)
    return names


def test_import_app_creates_no_files_and_starts_no_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    banned = {
        "openai",
        "anthropic",
        "ipfs_datasets_py.proof_context.provider",
        "ipfs_kit_py.proof_context.state_store",
        "ipfs_accelerate_py.proof_context.bootstrap",
        "ipfs_accelerate_py.proof_context.cli.state_commands",
    }
    for name in ("ipfs_accelerate_py.proof_context.cli.app", *banned):
        sys.modules.pop(name, None)
    before = {name: sys.modules.get(name) for name in banned}
    before_files = set(tmp_path.rglob("*"))
    imported = importlib.import_module("ipfs_accelerate_py.proof_context.cli.app")
    after_files = set(tmp_path.rglob("*"))
    assert after_files == before_files
    for name in banned:
        assert sys.modules.get(name) is before.get(name)
    assert imported.INFERS_CURRENT_DIRECTORY is False
    assert imported.PROVIDER_BOUND is False
    assert imported.SIBLING_LAYOUT_REQUIRED is False
    assert imported.main is not None


def test_import_state_commands_does_not_open_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    banned = {
        "ipfs_accelerate_py.proof_context.bootstrap",
        "openai",
        "anthropic",
    }
    for name in (
        "ipfs_accelerate_py.proof_context.cli.state_commands",
        *banned,
    ):
        sys.modules.pop(name, None)
    before = {name: sys.modules.get(name) for name in banned}
    before_files = set(tmp_path.rglob("*"))
    imported = importlib.import_module(
        "ipfs_accelerate_py.proof_context.cli.state_commands"
    )
    after_files = set(tmp_path.rglob("*"))
    assert after_files == before_files
    for name in banned:
        assert sys.modules.get(name) is before.get(name)
    assert imported.COMMANDS == STATE_COMMANDS


def test_cold_import_subprocess_is_hermetic(tmp_path: Path) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PACKAGE_ROOT), existing) if part
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HOME"] = str(tmp_path)
    env["GIT_TERMINAL_PROMPT"] = "0"
    before = set(tmp_path.rglob("*"))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ipfs_accelerate_py.proof_context.cli.app as app; "
                "import ipfs_accelerate_py.proof_context.cli.state_commands as state; "
                "print(app.PROG); "
                "print(','.join(state.COMMANDS)); "
                "print(app.INFERS_CURRENT_DIRECTORY)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    after = set(tmp_path.rglob("*"))
    assert after == before
    lines = completed.stdout.splitlines()
    assert lines[0] == PROG
    assert lines[1] == "init,scan,status,plan"
    assert lines[2] == "False"


def test_modules_do_not_start_work_at_import() -> None:
    forbidden = {
        "open_runtime",
        "open_engine",
        "create_ordinary_python_repository",
        "main",
        "dispatch",
        "cmd_init",
        "cmd_scan",
        "cmd_status",
        "cmd_plan",
        "mkdir",
        "Popen",
        "urlopen",
    }
    assert _module_level_call_names(APP_PATH).isdisjoint(forbidden)
    assert _module_level_call_names(STATE_PATH).isdisjoint(forbidden)
    app_source = APP_PATH.read_text(encoding="utf-8")
    state_source = STATE_PATH.read_text(encoding="utf-8")
    assert "os.getcwd" not in app_source
    assert "Path.cwd()" not in app_source
    assert "os.chdir" not in app_source
    assert "create_ordinary_python_repository" in state_source
    assert "engine.scan()" in state_source
    assert "engine.status()" in state_source
    assert "engine.plan()" in state_source
    assert "admit_mode" in state_source
    assert "open_runtime" in state_source
    assert "skip_stages" not in state_source
    assert "PatchLifecycle(" not in state_source


def test_parser_exposes_explicit_arguments_and_state_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert PROG in help_text
    for argument in REQUIRED_ARGUMENTS:
        assert f"--{argument}" in help_text
    for command in STATE_COMMANDS:
        assert command in help_text
    assert "current directory is never inferred" in help_text
    actions = {action.dest: action for action in parser._actions}
    assert "repository" in actions
    assert actions["repository"].default is None
    assert tuple(actions["policy"].choices) == MODES
    assert tuple(actions["output_mode"].choices) == OUTPUT_MODES
    option_strings = {flag for action in parser._actions for flag in action.option_strings}
    assert "--force" not in option_strings
    assert "--skip-policy" not in option_strings
    assert "--approve" not in option_strings
    assert INFERS_CURRENT_DIRECTORY is False
    assert PROVIDER_BOUND is False
    assert SIBLING_LAYOUT_REQUIRED is False
    assert COMMANDS == STATE_COMMANDS


def test_help_is_stable_and_zero_exit() -> None:
    code, stdout, stderr = _invoke(["--help"])
    assert code == 0
    assert stderr == ""
    assert f"usage: {PROG}" in stdout
    for argument in REQUIRED_ARGUMENTS:
        assert f"--{argument}" in stdout
    for command in STATE_COMMANDS:
        assert command in stdout
    init_code, init_help, init_err = _invoke(["init", "--help"])
    assert init_code == 0
    assert init_err == ""
    assert "ordinary Python Git repository" in init_help


def test_argument_errors_are_stable_and_typed() -> None:
    missing_command, stdout, stderr = _invoke(
        [
            "--repository",
            "/tmp/unused-pcce-040",
            "--task",
            "PCCE-040",
            "--correlation",
            "corr",
        ]
    )
    payload = _payload(stdout)
    assert missing_command == EXIT_CODES["invalid"]
    assert payload["status"] == "invalid"
    assert payload["error"] == "malformed"
    assert payload["payload"]["argument_error"] is True
    assert USAGE_COMMAND in stderr
    assert payload["exit_code"] == EXIT_CODES["invalid"]

    missing_repo, stdout, stderr = _invoke(
        ["init", "--task", "PCCE-040", "--correlation", "corr-pcce-040"]
    )
    payload = _payload(stdout)
    assert missing_repo == EXIT_CODES["invalid"]
    assert payload["status"] == "invalid"
    assert USAGE_REPOSITORY in stderr
    assert "current directory is not inferred" in payload["payload"]["reason"]

    missing_task, stdout, stderr = _invoke(
        ["scan", "--repository", "/tmp/unused-pcce-040", "--correlation", "corr"]
    )
    payload = _payload(stdout)
    assert missing_task == EXIT_CODES["invalid"]
    assert USAGE_TASK in stderr

    missing_corr, stdout, stderr = _invoke(
        ["status", "--repository", "/tmp/unused-pcce-040", "--task", "PCCE-040"]
    )
    payload = _payload(stdout)
    assert missing_corr == EXIT_CODES["invalid"]
    assert USAGE_CORRELATION in stderr

    unknown, stdout, stderr = _invoke(
        _args("explode", Path("/tmp/unused-pcce-040"))
    )
    payload = _payload(stdout)
    assert unknown == EXIT_CODES["invalid"]
    assert payload["status"] == "invalid"
    assert "explode" in stderr or USAGE_COMMAND in stderr

    bad_policy, stdout, stderr = _invoke(
        _args("init", Path("/tmp/unused-pcce-040"), policy="shadow")
    )
    payload = _payload(stdout)
    assert bad_policy == EXIT_CODES["invalid"]
    assert payload["status"] == "invalid"
    assert "shadow" in stderr or "policy" in stderr.lower()

    bad_output, stdout, stderr = _invoke(
        _args("init", Path("/tmp/unused-pcce-040"), output_mode="yaml")
    )
    payload = _payload(stdout)
    assert bad_output == EXIT_CODES["invalid"]
    assert payload["status"] == "invalid"


def test_missing_repository_does_not_infer_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "pyproject.toml").write_text("[project]\nname='decoy'\n", encoding="utf-8")
    monkeypatch.chdir(decoy)
    code, stdout, stderr = _invoke(
        ["scan", "--task", "PCCE-040", "--correlation", "corr-pcce-040"]
    )
    payload = _payload(stdout)
    assert code == EXIT_CODES["invalid"]
    assert payload["status"] == "invalid"
    assert USAGE_REPOSITORY in stderr
    assert not (decoy / ".git").exists()
    assert list(decoy.rglob("*")) == [decoy / "pyproject.toml"]


def test_init_scan_status_plan_are_typed_runtime_calls(tmp_path: Path) -> None:
    repo = tmp_path / "ordinary-python"
    outside = set(tmp_path.iterdir())

    init_code, init_out, init_err = _invoke(_args("init", repo))
    init_payload = _payload(init_out)
    assert init_err == ""
    assert init_code == 0
    assert init_payload["status"] == "succeeded"
    assert init_payload["command"] == "init"
    assert init_payload["schema"].endswith("/cli-result")
    assert init_payload["policy"] == "production"
    assert init_payload["correlation_id"] == "corr-pcce-040"
    assert init_payload["identities"]["task_id"] == "PCCE-040"
    assert init_payload["identities"]["trace_id"] == "corr-pcce-040"
    assert str(init_payload["artifact_cid"]).startswith("b")
    assert init_payload["payload"]["ordinary_python_git_repository"] is True
    assert init_payload["payload"]["initialized"] is True
    assert (repo / "pyproject.toml").is_file()
    assert (repo / ".git").exists()
    assert (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (repo / "ipfs_datasets_py").exists()
    assert not (repo / "ipfs_kit_py").exists()
    assert set(tmp_path.iterdir()) == outside | {repo}

    scan_code, scan_out, scan_err = _invoke(_args("scan", repo))
    scan_payload = _payload(scan_out)
    assert scan_err == ""
    assert scan_code == 0
    assert scan_payload["status"] == "succeeded"
    assert scan_payload["command"] == "scan"
    assert scan_payload["contract"] == OPERATION_CONTRACTS["scan"]
    assert scan_payload["contract"] == "pcce/proof-context/v0.1/repository-state"
    assert str(scan_payload["artifact_cid"]).startswith("b")
    assert scan_payload["provenance"] == "live"
    assert is_success(scan_payload["status"], provenance=scan_payload["provenance"])
    files = scan_payload["payload"]["files"]
    assert "pyproject.toml" in files
    assert "src/demo/__init__.py" in files
    assert scan_payload["payload"]["datasets_port"] == (
        "ipfs_datasets_py.proof_context.provider"
    )

    status_code, status_out, status_err = _invoke(_args("status", repo))
    status_payload = _payload(status_out)
    assert status_err == ""
    assert status_code == 0
    assert status_payload["status"] in STATUSES
    assert status_payload["status"] == "succeeded"
    assert status_payload["command"] == "status"
    assert status_payload["contract"] == OPERATION_CONTRACTS["status"]
    assert "ok" not in status_payload
    assert "success" not in status_payload
    assert status_payload["identities"]["task_id"] == "PCCE-040"
    assert status_payload["payload"]["opened"] is True
    assert status_payload["payload"]["canonical_head"]

    plan_code, plan_out, plan_err = _invoke(_args("plan", repo))
    plan_payload = _payload(plan_out)
    assert plan_err == ""
    assert plan_code == 0
    assert plan_payload["status"] == "succeeded"
    assert plan_payload["command"] == "plan"
    assert plan_payload["contract"] == OPERATION_CONTRACTS["plan"]
    assert plan_payload["contract"] == "pcce/proof-context/v0.1/invalidation-plan"
    assert str(plan_payload["artifact_cid"]).startswith("b")
    assert plan_payload["payload"]["planner_authority"] == "canonical"
    assert "src/demo/__init__.py" in plan_payload["payload"]["invalidated"]
    assert (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_human_output_mode_is_stable(tmp_path: Path) -> None:
    repo = tmp_path / "human-repo"
    init_code, _, _ = _invoke(_args("init", repo))
    assert init_code == 0
    code, stdout, stderr = _invoke(_args("status", repo, output_mode="human"))
    assert code == 0
    assert stderr == ""
    assert "command: status" in stdout
    assert "status: succeeded" in stdout
    assert "exit_code: 0" in stdout
    assert "correlation_id: corr-pcce-040" in stdout
    assert "task_id: PCCE-040" in stdout


def test_scan_without_repository_is_typed_failure(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    code, stdout, stderr = _invoke(_args("scan", missing))
    payload = _payload(stdout)
    assert code != 0
    assert payload["status"] in STATUSES
    assert payload["status"] != "succeeded"
    assert payload["error"]
    assert payload["exit_code"] == code
    assert stderr == ""


def test_policy_cannot_be_bypassed_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "policy-repo"
    for key in PROMOTION_ENV_VARS:
        monkeypatch.setenv(key, "simulation")
    code, stdout, stderr = _invoke(_args("init", repo, policy="production"))
    payload = _payload(stdout)
    assert stderr == ""
    assert code == 0
    assert payload["policy"] == "production"
    assert payload["provenance"] != "simulated"
    assert payload["status"] == "succeeded"

    scan_code, scan_out, _ = _invoke(_args("scan", repo, policy="production"))
    scan_payload = _payload(scan_out)
    assert scan_code == 0
    assert scan_payload["policy"] == "production"
    assert scan_payload["provenance"] == "live"
    assert scan_payload["status"] != "simulated"


def test_simulation_policy_does_not_claim_live_success(tmp_path: Path) -> None:
    repo = tmp_path / "sim-repo"
    init_code, _, _ = _invoke(_args("init", repo, policy="simulation"))
    assert init_code in {0, EXIT_CODES["simulated"]}
    code, stdout, stderr = _invoke(_args("scan", repo, policy="simulation"))
    payload = _payload(stdout)
    assert stderr == ""
    assert payload["policy"] == "simulation"
    assert payload["provenance"] == "simulated"
    assert payload["status"] in STATUSES
    assert code == exit_code_for(payload["status"], provenance=payload["provenance"])
    assert code != 0
    assert not is_success(payload["status"], provenance=payload["provenance"])


def test_live_modes_are_the_closed_production_set() -> None:
    assert LIVE_MODES == frozenset({"production", "supervised"})
    assert "shadow" not in MODES
    assert exit_code_for("succeeded", provenance="live") == 0
    assert exit_code_for("succeeded", provenance="simulated") == EXIT_CODES["simulated"]
    assert exit_code_for("unavailable") == EXIT_CODES["unavailable"]
    assert exit_code_for("stale") == EXIT_CODES["stale"]
    assert exit_code_for("rejected") == EXIT_CODES["rejected"]
    assert exit_code_for("invalid") == EXIT_CODES["invalid"]


def test_direct_command_functions_require_explicit_context(tmp_path: Path) -> None:
    repo = tmp_path / "direct-repo"
    context = CliContext(
        command="init",
        repository=repo,
        policy="production",
        task_id="PCCE-040",
        correlation_id="corr-direct",
        output_mode="json",
    )
    result = cmd_init(context)
    assert result.status == "succeeded"
    assert result.artifact_cid
    assert result.payload["ordinary_python_git_repository"] is True
    scanned = cmd_scan(context)
    assert scanned.status == "succeeded"
    assert scanned.contract == "pcce/proof-context/v0.1/repository-state"
    typed_status = cmd_status(context)
    assert typed_status.status == "succeeded"
    planned = cmd_plan(context)
    assert planned.status == "succeeded"
    assert planned.payload["planner_authority"] == "canonical"


def test_state_dir_escape_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "bounded-repo"
    init_code, _, _ = _invoke(_args("init", repo))
    assert init_code == 0
    outside = tmp_path / "outside-state"
    code, stdout, stderr = _invoke(
        _args("scan", repo, extra=("--state-dir", str(outside)))
    )
    payload = _payload(stdout)
    assert code != 0
    assert payload["status"] in {"rejected", "invalid"}
    assert payload["error"] in {"boundary_violation", "malformed"}
    assert not outside.exists()
    assert stderr == ""
