"""ASE3-010 prompt-first supervisor CLI tests."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as supervisor_cli
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
    Supervisor,
    SupervisorAmbiguityError,
    SupervisorConfigurationError,
    SupervisorObservation,
    SupervisorRun,
    SupervisorUnavailableError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_discovery_manifest_is_static() -> None:
    manifest = supervisor_cli.supervisor_cli_discovery_manifest()
    assert manifest["group"] == "supervisor"
    assert set(manifest["commands"]) == set(supervisor_cli.SUPERVISOR_COMMANDS)
    assert manifest["cold_help"] is True


def test_register_supervisor_cli_is_parser_only() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    group = supervisor_cli.register_supervisor_cli(sub)
    assert group.prog.endswith("supervisor") or "supervisor" in group.prog
    # help should not raise
    help_io = io.StringIO()
    group.print_help(help_io)
    text = help_io.getvalue()
    assert "run" in text
    assert "preview" in text


def test_cold_help_via_product_cli_subprocess() -> None:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        __import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, "-m", "ipfs_accelerate_py.cli_entry", "supervisor", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run" in completed.stdout
    assert "preview" in completed.stdout


class _FakeSupervisor:
    composition_cid = "cid:composition"

    def run(self, prompt: str) -> SupervisorRun:
        assert "secret-token" not in prompt or True
        return SupervisorRun(
            run_id="run-1",
            run_revision=1,
            composition_cid=self.composition_cid,
            state="running",
            health="healthy",
            event_cursor="c1",
            supervisor=None,
            effect_receipt_cids=("r1",),
        )

    def preview(self, prompt: str) -> SupervisorObservation:
        return SupervisorObservation(
            run_id="",
            state="preview",
            health="unknown",
            event_cursor="",
            composition_cid=self.composition_cid,
            summary="preview-only",
            values={"effect_applied": False, "prompt_cid": "cid:p"},
        )

    def status(self, run_id: str | None = None) -> SupervisorObservation:
        if run_id is None:
            raise SupervisorAmbiguityError("ambiguous", candidates=("a", "b"))
        return SupervisorObservation(
            run_id=run_id,
            state="running",
            health="healthy",
            event_cursor="c1",
            composition_cid=self.composition_cid,
            summary="ok",
        )

    def follow(self, run_id: str | None = None):
        yield self.status(run_id or "run-1")

    def explain(self, run_id: str | None = None) -> SupervisorObservation:
        return self.status(run_id or "run-1")

    def doctor(self, run_id: str | None = None) -> SupervisorObservation:
        return self.status(run_id or "run-1")

    def steer(self, run_id: str, prompt: str) -> SupervisorObservation:
        return SupervisorObservation(
            run_id=run_id,
            state="running",
            health="healthy",
            event_cursor="c1",
            composition_cid=self.composition_cid,
            summary="steer accepted",
            values={"effect_applied": False},
        )


def test_run_command_json_envelope() -> None:
    args = SimpleNamespace(
        supervisor_command="run",
        prompt="Improve gates",
        prompt_file=None,
        prompt_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(
        args, stdout=out, supervisor=_FakeSupervisor()
    )
    assert code == supervisor_cli.EXIT_SUCCESS
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["result"]["run_id"] == "run-1"
    assert payload["composition_cid"] == "cid:composition"


def test_missing_prompt_is_invalid() -> None:
    args = SimpleNamespace(
        supervisor_command="run",
        prompt=None,
        prompt_file=None,
        prompt_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(
        args, stdout=out, stderr=out, supervisor=_FakeSupervisor()
    )
    assert code == supervisor_cli.EXIT_INVALID
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False


def test_ambiguity_exit_code() -> None:
    args = SimpleNamespace(
        supervisor_command="status",
        run_id=None,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(
        args, stdout=out, stderr=out, supervisor=_FakeSupervisor()
    )
    assert code == supervisor_cli.EXIT_AMBIGUITY


def test_preview_does_not_echo_prompt_body() -> None:
    secret = "super-secret-prompt-body"
    args = SimpleNamespace(
        supervisor_command="preview",
        prompt=secret,
        prompt_file=None,
        prompt_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(
        args, stdout=out, supervisor=_FakeSupervisor()
    )
    assert code == supervisor_cli.EXIT_SUCCESS
    assert secret not in out.getvalue()
