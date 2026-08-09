"""Regression coverage for implementation daemon cutover surfaces (DQP-018).

Implied validation target for the database-authoritative daemon cutover.
Keeps legacy defaults intact while proving authority CLI/env wiring and
database-mode gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_AUTHORITY_MODES,
    is_database_authority_mode,
    parse_args,
    resolve_daemon_database_program,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
    implementation_state_paths,
)


def test_parse_args_defaults_remain_legacy_markdown() -> None:
    parsed = parse_args(["--once"])
    assert parsed.task_source_kind == "legacy-markdown"
    assert parsed.authority_mode == ""
    assert parsed.explicit_legacy_task_source is False
    assert resolve_daemon_database_program(args=parsed) is None


def test_is_database_authority_mode_closed_population() -> None:
    assert DATABASE_AUTHORITY_MODES == {
        "quack",
        "embedded",
        "embedded_exclusive",
    }
    for mode in DATABASE_AUTHORITY_MODES:
        assert is_database_authority_mode(mode) is True
    assert is_database_authority_mode("legacy_markdown") is False
    assert is_database_authority_mode("markdown") is False


def test_implementation_state_paths_shape(tmp_path: Path) -> None:
    class _Args:
        state_dir = tmp_path
        state_prefix = "portal"

    paths = implementation_state_paths(_Args())  # type: ignore[arg-type]
    assert paths["state_path"] == tmp_path / "portal_task_state.json"
    assert paths["strategy_path"] == tmp_path / "portal_strategy.json"
    assert paths["events_path"] == tmp_path / "portal_events.jsonl"


def test_runner_builds_legacy_daemon_without_database(
    tmp_path: Path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "DQP-018"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "dqp018@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    markdown = repo / "board.md"
    markdown.write_text("## T1\n\n- Status: todo\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    parsed = parse_args(
        [
            "--todo-path",
            str(markdown),
            "--state-dir",
            str(state_dir),
            "--state-prefix",
            "portal",
            "--once",
        ]
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        parsed,
        repo_root=repo,
    )
    assert getattr(daemon, "is_database_authority")() is False
    assert getattr(daemon, "database_implementation") is None
    assert context.state_path == state_dir / "portal_task_state.json"
    assert context.events_path == state_dir / "portal_events.jsonl"


def test_resolve_daemon_database_program_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    payload = {
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "failover_policy": "fail_closed",
        "explicit_legacy": False,
    }
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        json.dumps(payload, sort_keys=True),
    )

    class _Args:
        authority_mode = ""
        task_source_kind = ""
        endpoint_secret_handle = ""
        state_store_id = ""
        state_store_generation = ""
        state_schema_revision = ""
        event_store_path = ""
        runtime_registry_path = ""
        export_profile = ""
        state_failover_policy = ""
        explicit_legacy_task_source = False
        worktree_root = None

    program = resolve_daemon_database_program(args=_Args())
    assert program is not None
    assert program.authority_mode == "embedded"
    assert program.task_source_kind == "duckdb"
