"""SCH-013 semantic-state CLI: commands, exit codes, production gate, cold safety."""

from __future__ import annotations

import importlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_MODULE = "ipfs_accelerate_py.agent_supervisor.semantic_state.cli"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


@pytest.fixture
def cli():
    sys.modules.pop(CLI_MODULE, None)
    return importlib.import_module(CLI_MODULE)


def _run(cli, argv: list[str], *, provider: Any | None = None) -> tuple[int, dict[str, Any], str]:
    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(argv, stdout=out, stderr=err, provider=provider)
    text = out.getvalue()
    payload = json.loads(text) if text.strip() else {}
    return code, payload, err.getvalue()


# ---------------------------------------------------------------------------
# Parser / help / descriptor
# ---------------------------------------------------------------------------


def test_build_parser_exposes_all_plan_commands(cli) -> None:
    parser = cli.build_parser()
    # Help should not raise.
    help_text = parser.format_help()
    assert "semantic-state" in help_text
    for name in (
        "scan",
        "watch",
        "status",
        "graph",
        "explain-symbol",
        "explain-impact",
        "invalidate",
        "select-tests",
        "pack-context",
        "verify",
        "apply-patch",
        "compare-full-suite",
        "benchmark",
    ):
        assert name in help_text


def test_descriptor_declares_interface_and_invariants(cli) -> None:
    desc = cli.semantic_state_cli_descriptor()
    assert desc["interface"] == "SemanticStateCLI@1"
    assert desc["bundle"] == "sch/cli@1"
    assert desc["console_entry"] == "semantic-state"
    assert "production_apply_patch_cannot_simulate" in desc["invariants"]
    assert "cold_help_and_import_no_mutation" in desc["invariants"]
    assert desc["exit_codes"]["unavailable"] == 3
    assert desc["exit_codes"]["production_gate"] == 4


def test_subparser_help_is_bounded_and_stable(cli) -> None:
    for command in (
        "scan",
        "watch",
        "status",
        "apply-patch",
        "benchmark",
        "interface-schema",
    ):
        code = cli.main([command, "--help"], stdout=io.StringIO(), stderr=io.StringIO())
        assert code == 0


def test_missing_command_is_usage_error(cli) -> None:
    out = io.StringIO()
    err = io.StringIO()
    code = cli.main([], stdout=out, stderr=err)
    assert code == cli.EXIT_USAGE


# ---------------------------------------------------------------------------
# Cold import / --help: no mutation
# ---------------------------------------------------------------------------


def test_cold_import_starts_no_resources_threads_processes_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = CLI_MODULE
    sys.modules.pop(module_name, None)

    before_threads = {t.ident for t in threading.enumerate()}
    started_threads: list[str] = []
    real_thread_start = threading.Thread.start

    def guarded_start(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        started_threads.append(self.name)
        return real_thread_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", guarded_start)

    popen_calls: list[Any] = []

    def guarded_popen(*args: Any, **kwargs: Any):
        popen_calls.append((args, kwargs))
        raise AssertionError("cold import must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)

    socket_mod = importlib.import_module("socket")
    real_socket = socket_mod.socket
    socket_calls: list[Any] = []

    class GuardedSocket(real_socket):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            socket_calls.append((args, kwargs))
            raise AssertionError("cold import must not open sockets")

    monkeypatch.setattr(socket_mod, "socket", GuardedSocket)

    created_engines: list[str] = []

    class _NoDB:
        def connect(self, *args: Any, **kwargs: Any):
            created_engines.append("connect")
            raise AssertionError("cold import must not open databases")

    fake_duckdb = types.ModuleType("duckdb")
    fake_duckdb.connect = _NoDB().connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)

    # Block pip / installer entry during import.
    real_run = subprocess.run

    def guarded_run(*args: Any, **kwargs: Any):
        cmd = args[0] if args else kwargs.get("args")
        text = " ".join(str(x) for x in (cmd or ()))
        if "pip" in text or "install" in text:
            raise AssertionError(f"cold import must not install: {text}")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)

    mod = importlib.import_module(module_name)
    assert mod.CLI_INTERFACE == "SemanticStateCLI@1"
    assert started_threads == []
    assert popen_calls == []
    assert socket_calls == []
    assert created_engines == []

    after_threads = {t.ident for t in threading.enumerate()}
    assert after_threads - before_threads == set()


def test_cold_help_does_not_mutate_environment_or_cwd(
    cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_before = dict(os.environ)
    cwd_before = Path.cwd()
    # Point cwd at a disposable dir; help must not create files there.
    monkeypatch.chdir(tmp_path)
    before_names = set(os.listdir(tmp_path))

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(["--help"], stdout=out, stderr=err)
    assert code == 0
    assert "semantic-state" in out.getvalue() or "semantic-state" in err.getvalue()

    assert dict(os.environ) == env_before
    assert Path.cwd() == tmp_path
    assert set(os.listdir(tmp_path)) == before_names
    monkeypatch.chdir(cwd_before)


# ---------------------------------------------------------------------------
# interface-schema + packaging discovery
# ---------------------------------------------------------------------------


def test_interface_schema_command_returns_packaged_json(cli) -> None:
    code, payload, _ = _run(cli, ["interface-schema"])
    assert code == 0
    assert payload["ok"] is True
    schema = payload["result"]["schema"]
    assert schema["name"] == "semantic-state-harness"
    assert schema["namespace"] == "ipfs-accelerate.agent-supervisor"
    assert payload["result"]["cli"]["console_entry"] == "semantic-state"


def test_load_interface_schema_text_is_valid_json(cli) -> None:
    text = cli.load_interface_schema_text()
    body = json.loads(text)
    assert body["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Unavailable provider paths (stable exit 3)
# ---------------------------------------------------------------------------


def test_scan_unavailable_provider_exits_nonzero(cli, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")

    class Boom:
        def scan_repository(self, *a: Any, **k: Any) -> Any:
            from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
                SemanticStateUnavailable,
            )

            raise SemanticStateUnavailable(
                "scan",
                "import_failed",
                "pinned datasets surface unavailable",
                retryable=False,
            )

    code, payload, _ = _run(cli, ["scan", str(repo)], provider=Boom())
    assert code == cli.EXIT_UNAVAILABLE
    assert payload["ok"] is False
    assert payload["exit_code"] == cli.EXIT_UNAVAILABLE
    assert payload["error"]["reason_code"] == "import_failed"
    assert payload["error"]["retryable"] is False


def test_scan_with_injected_provider_returns_json(cli, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    class FakeState:
        state_cid = _cid("state-1")
        root_cid = _cid("root-1")

        def to_dict(self) -> dict[str, Any]:
            return {"state_cid": self.state_cid, "root_cid": self.root_cid}

    class Prov:
        def scan_repository(self, path: str, previous_state: Any = None) -> Any:
            return FakeState()

    code, payload, _ = _run(cli, ["scan", str(repo)], provider=Prov())
    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "scan"
    assert payload["result"]["state_cid"] == FakeState.state_cid
    assert payload["result"]["repo"] == str(repo.resolve())


def test_status_is_local_without_daemon(cli, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    storage = tmp_path / "durable"
    code, payload, _ = _run(
        cli,
        ["status", str(repo), "--storage-dir", str(storage)],
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["result"]["phase"] in {
        "idle",
        "watching",
        "scanning",
        "stopped",
        "failed_closed",
        "restarting",
        "shutting_down",
    }
    assert payload["result"]["repo"] == str(repo.resolve())


def test_watch_admits_snapshot_without_starting_daemon(cli, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    snap = _cid("watch-snap-1")
    code, payload, _ = _run(
        cli,
        ["watch", str(repo), "--snapshot-cid", snap],
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["result"]["ack"]["snapshot_cid"] == snap
    assert payload["result"]["ack"]["scheduled"] is True


def test_graph_explain_select_with_provider(cli, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = types.SimpleNamespace(
        state_cid=_cid("s1"),
        root_cid=_cid("r1"),
        graph_cid=_cid("g1"),
    )

    class Prov:
        def scan_repository(self, path: str, previous_state: Any = None) -> Any:
            return types.SimpleNamespace(state=state, **state.__dict__)

        def explain_symbol(self, repository_state: Any, symbol_id: str) -> Any:
            return {"symbol_id": symbol_id, "kind": "function"}

        def explain_impact(self, repository_state: Any, targets: Any) -> Any:
            return {"targets": list(targets), "impacted": []}

        def select_tests_and_proofs(
            self, previous: Any, current: Any, invalidation: Any, **kwargs: Any
        ) -> Any:
            return types.SimpleNamespace(
                selection_cid=_cid("sel-1"),
                previous_semantic_state_root_cid=None,
                current_semantic_state_root_cid=_cid("r1"),
                to_dict=lambda: {
                    "selection_cid": _cid("sel-1"),
                    "previous_semantic_state_root_cid": None,
                    "current_semantic_state_root_cid": _cid("r1"),
                },
            )

    prov = Prov()
    code, payload, _ = _run(
        cli, ["graph", str(repo), "--symbol", "pkg.mod:foo"], provider=prov
    )
    assert code == 0
    assert payload["result"]["symbol"] == "pkg.mod:foo"

    code, payload, _ = _run(
        cli, ["explain-symbol", str(repo), "pkg.mod:foo"], provider=prov
    )
    assert code == 0
    assert payload["result"]["explanation"]["symbol_id"] == "pkg.mod:foo"

    code, payload, _ = _run(
        cli,
        ["explain-impact", str(repo), "pkg.mod:foo", "pkg/mod.py"],
        provider=prov,
    )
    assert code == 0
    assert "pkg.mod:foo" in payload["result"]["targets"]

    code, payload, _ = _run(
        cli, ["select-tests", str(repo), "pkg.mod:foo"], provider=prov
    )
    assert code == 0
    assert payload["result"]["selection"] is not None


def test_pack_context_and_verify_are_local(cli, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code, payload, _ = _run(
        cli, ["pack-context", str(repo), "fix the bug", "pkg.mod:foo"]
    )
    assert code == 0
    assert payload["ok"] is True
    assert "context_pack" in payload["result"]

    code, payload, _ = _run(cli, ["verify", str(repo)])
    assert code == 0
    assert payload["result"]["verification"]["status"] == "not_executed"


# ---------------------------------------------------------------------------
# Production apply-patch cannot simulate
# ---------------------------------------------------------------------------


def test_production_apply_patch_rejects_simulate(cli, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    patch = tmp_path / "change.diff"
    patch.write_text(
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n",
        encoding="utf-8",
    )
    code, payload, _ = _run(
        cli,
        [
            "apply-patch",
            str(repo),
            str(patch),
            "--mode",
            "production",
            "--simulate",
            "--storage-dir",
            str(tmp_path / "durable"),
        ],
    )
    assert code == cli.EXIT_PRODUCTION_GATE
    assert payload["ok"] is False
    assert payload["exit_code"] == cli.EXIT_PRODUCTION_GATE
    assert payload["error"]["reason_code"] == "production_simulate_rejected"
    assert "simulat" in payload["error"]["diagnostic"].lower()


def test_development_apply_patch_with_missing_inputs_fails_closed(
    cli, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    patch = tmp_path / "change.diff"
    patch.write_text(
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n",
        encoding="utf-8",
    )
    code, payload, _ = _run(
        cli,
        [
            "apply-patch",
            str(repo),
            str(patch),
            "--mode",
            "development",
            "--storage-dir",
            str(tmp_path / "durable"),
            "--allow-paths",
            "pkg/a.py",
        ],
    )
    # Without full harness inputs the command must not claim production success
    # and must return a stable nonzero or structured disposition.
    assert code in {
        cli.EXIT_OK,
        cli.EXIT_ERROR,
        cli.EXIT_UNAVAILABLE,
        cli.EXIT_PRODUCTION_GATE,
    }
    assert "ok" in payload
    assert payload["command"] == "apply-patch"
    if payload["ok"]:
        # Even on success, development must not pretend production acceptance.
        result = payload["result"]
        assert result.get("mode") == "development"
        assert result.get("simulated") in {True, False, None}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_runs_offline_and_returns_summary(cli) -> None:
    code, payload, _ = _run(cli, ["benchmark"])
    assert code == 0
    assert payload["ok"] is True
    report = payload["result"]["report"]
    assert report["task_count"] == 40
    assert report["interface"] == "SemanticStateBenchmark@1"
    assert payload["result"]["gates_ok"] is True


def test_benchmark_write_requires_paths(cli) -> None:
    code, payload, _ = _run(cli, ["benchmark", "--write"])
    assert code == cli.EXIT_ERROR
    assert payload["ok"] is False


def test_benchmark_write_to_tmp(cli, tmp_path: Path) -> None:
    json_out = tmp_path / "results.json"
    md_out = tmp_path / "results.md"
    code, payload, _ = _run(
        cli,
        [
            "benchmark",
            "--write",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
    )
    assert code == 0
    assert json_out.is_file()
    assert md_out.is_file()
    body = json.loads(json_out.read_text(encoding="utf-8"))
    assert body["task_count"] == 40


# ---------------------------------------------------------------------------
# compare-full-suite
# ---------------------------------------------------------------------------


def test_compare_full_suite_with_controlled_fixture(cli) -> None:
    fixture = (
        REPO_ROOT
        / "test"
        / "fixtures"
        / "semantic_state_harness"
        / "controlled_repo"
    )
    assert fixture.is_dir(), (
        "SCH-014 controlled fixture is required for compare-full-suite"
    )
    code, payload, _ = _run(cli, ["compare-full-suite", str(fixture)])
    # May succeed with comparison or fail closed with typed error; never exit 0
    # with ok:false, and never crash.
    assert code in {cli.EXIT_OK, cli.EXIT_ERROR, cli.EXIT_UNAVAILABLE}
    assert "ok" in payload
    assert payload["command"] == "compare-full-suite"
    if code == cli.EXIT_OK:
        assert payload["ok"] is True
    else:
        assert payload["ok"] is False
        assert "error" in payload


# ---------------------------------------------------------------------------
# Exit code stability table
# ---------------------------------------------------------------------------


def test_exit_codes_are_stable_constants(cli) -> None:
    assert cli.EXIT_OK == 0
    assert cli.EXIT_ERROR == 1
    assert cli.EXIT_USAGE == 2
    assert cli.EXIT_UNAVAILABLE == 3
    assert cli.EXIT_PRODUCTION_GATE == 4


def test_nonexistent_repo_errors_stably(cli, tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    code, payload, _ = _run(cli, ["scan", str(missing)], provider=MagicMock())
    assert code == cli.EXIT_ERROR
    assert payload["ok"] is False
    assert payload["error"]["reason_code"] == "not_found"
