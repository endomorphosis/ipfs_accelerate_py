from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control_cli import (
    AGENT_CLI_EXIT_INVALID,
    COMMAND_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    ControlBounds,
    EffectKind,
    ExpectedEffect,
    Operation,
    OperationRequest,
    OperationResult,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    InMemoryControlStateStore,
    SupervisorControlService,
)


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:fixture",
        "tree_id": "tree:abc",
        "objective_id": "ASI-G103",
        "objective_revision": "objective:1",
        "policy_id": "policy:control",
        "policy_revision": "policy:1",
        "caller": "operator:test",
    }


def _request(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    parameters: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> OperationRequest:
    effects = ()
    if operation.mutating:
        effects = (
            ExpectedEffect(
                effect_id=f"{operation.value}:fixture",
                kind=EffectKind.WRITE_STATE,
                resource="supervisor:fixture",
                paths=("supervisor.json",),
                description="Fixture state transition",
            ),
        )
    return OperationRequest(
        operation=operation,
        **_binding(repo_root, state_root),
        parameters=parameters or {},
        expected_effects=effects,
        dry_run=dry_run,
        bounds=ControlBounds(max_items=16, max_paths=16, max_effects=16),
    )


def _service(repo_root: Path, state_root: Path) -> SupervisorControlService:
    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={
            Operation.STATUS: lambda _request: {
                "state": "healthy",
                "phase": "idle",
            },
            Operation.PLAN: lambda _request: {"steps": ["inspect", "validate"]},
        },
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )


def _invoke(
    capsys: pytest.CaptureFixture[str],
    service: SupervisorControlService,
    command: str,
    request: OperationRequest,
) -> tuple[int, dict[str, Any]]:
    code = cli.main(
        [
            "agent",
            command,
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    return int(code), json.loads(captured.out)


def test_agent_group_covers_the_closed_operation_vocabulary() -> None:
    assert set(COMMAND_OPERATIONS.values()) == set(Operation)
    assert len(COMMAND_OPERATIONS) == len(Operation)


def test_cli_read_result_is_exactly_the_python_service_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    request = _request(repo_root, state_root, Operation.STATUS)

    python_record = service.execute(request).to_record()
    code, cli_record = _invoke(capsys, service, "status", request)

    assert code == 0
    assert cli_record == python_record
    assert OperationResult.from_dict(cli_record).result_id == python_record["content_id"]


def test_cli_proposal_and_dry_run_use_the_same_result_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    plan = _request(repo_root, state_root, Operation.PLAN)
    pause = _request(
        repo_root,
        state_root,
        Operation.PAUSE,
        parameters={
            "target_id": "supervisor:fixture",
            "reason": "maintenance",
            "requested_state": "paused",
        },
        dry_run=True,
    )

    plan_expected = service.execute(plan).to_record()
    pause_expected = service.execute(pause).to_record()
    assert _invoke(capsys, service, "plan", plan) == (0, plan_expected)
    code, pause_record = _invoke(capsys, service, "pause", pause)

    assert code == 0
    assert pause_record == pause_expected
    assert pause_record["authority"] == "proposal"
    assert pause_record["preview"]["would_change"] is True
    assert pause_record["effects"] == []


def test_cli_rejects_ambiguous_roots_and_non_dry_run_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["agent", "status", "--output-json"])
    missing = capsys.readouterr()
    assert code == AGENT_CLI_EXIT_INVALID
    assert json.loads(missing.err)["status"] == "invalid_request"
    assert "explicit target bindings" in missing.err

    binding = _binding(tmp_path / "repo", tmp_path / "state")
    argv = ["agent", "pause"]
    for name, value in binding.items():
        argv.extend(["--" + name.replace("_", "-"), value])
    argv.extend(["--target-id", "supervisor:fixture", "--output-json"])
    code = cli.main(argv)
    unsafe = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_INVALID
    assert "real mutations require a complete" in unsafe.err


def test_cli_watch_is_bounded_and_emits_one_canonical_record_per_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    request = _request(
        repo_root,
        state_root,
        Operation.HEALTH,
        parameters={"health_path": "missing-health.json"},
    )

    code = cli.main(
        [
            "agent",
            "health",
            "--request-json",
            request.to_json(),
            "--watch-count",
            "2",
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]

    assert code == 4  # both bounded reads consistently report not_found
    assert len(records) == 2
    assert records[0] == records[1]
    assert all(record["schema"].endswith("operation-result@1") for record in records)


def test_cli_rejects_unbounded_watch_before_dispatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    request = _request(repo_root, state_root, Operation.STATUS)

    code = cli.main(
        [
            "agent",
            "status",
            "--request-json",
            request.to_json(),
            "--watch-count",
            "101",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_INVALID
    assert captured.out == ""
    assert "watch-count" in captured.err
