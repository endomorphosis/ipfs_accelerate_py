"""ASI-152: CLI and python -m prompt-workflow entry surface tests."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    COMMAND_OPERATIONS,
    agent_cli_discovery_manifest,
    validate_agent_cli_catalog,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PROMPT_WORKFLOW_CLI_COMMANDS,
    PROMPT_WORKFLOW_CLI_EXIT_INVALID,
    build_prompt_workflow_arg_parser,
    run_prompt_workflow_cli,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_WRAPPER = (
    REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "prompt_workflow.py"
)


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:prompt",
        "tree_id": "tree:current",
        "objective_id": "ASI-152",
        "objective_revision": "objective:1",
        "policy_id": "policy:prompt-control",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


def _request(
    operation: Operation,
    repository_root: Path,
    state_root: Path,
    **parameters: Any,
) -> OperationRequest:
    return OperationRequest(
        operation=operation,
        **_binding(repository_root, state_root),
        parameters=parameters,
        dry_run=operation
        in {Operation.WORKFLOW_PREVIEW, Operation.RESCUE_PREVIEW},
    )


def test_cli_commands_cover_prompt_workflow_operations() -> None:
    assert COMMAND_OPERATIONS["workflow-preview"] is Operation.WORKFLOW_PREVIEW
    assert COMMAND_OPERATIONS["workflow-create"] is Operation.WORKFLOW_MATERIALIZE
    assert COMMAND_OPERATIONS["restart"] is Operation.RESTART
    assert COMMAND_OPERATIONS["rescue-preview"] is Operation.RESCUE_PREVIEW
    assert COMMAND_OPERATIONS["rescue"] is Operation.RESCUE
    assert set(COMMAND_OPERATIONS.values()) == set(Operation)
    validate_agent_cli_catalog()
    assert set(agent_cli_discovery_manifest().operations) == set(Operation)
    assert PROMPT_WORKFLOW_CLI_COMMANDS == (
        "workflow-preview",
        "workflow-create",
        "restart",
        "rescue-preview",
        "rescue",
    )


def test_help_and_import_are_side_effect_free(capsys: pytest.CaptureFixture[str]) -> None:
    # Import already happened at module load; re-import must not start providers.
    __import__("ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow")
    parser = build_prompt_workflow_arg_parser()
    with pytest.raises(SystemExit) as exited:
        parser.parse_args(["--help"])
    assert exited.value.code == 0
    code = run_prompt_workflow_cli([])
    assert code == PROMPT_WORKFLOW_CLI_EXIT_INVALID
    captured = capsys.readouterr()
    assert "workflow-preview" in captured.out
    assert "usage" in captured.out.lower() or "prompt-workflow" in captured.out


def test_agent_cli_workflow_preview_matches_python_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(
        Operation.WORKFLOW_PREVIEW,
        repository_root,
        state_root,
        directory=str(repository_root),
        prompt_source={"kind": "inline", "content_cid": "prompt:one"},
        output_mode="both",
    )
    calls: list[OperationRequest] = []

    def handler(incoming: OperationRequest) -> BackendResponse:
        calls.append(incoming)
        return BackendResponse(
            data={"proposal_root": "plan:one"},
            changed=False,
            checks=("schema",),
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={Operation.WORKFLOW_PREVIEW: handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )
    python_result = service.workflow_preview(request)

    exit_status = cli.main(
        [
            "agent",
            "workflow-preview",
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    assert exit_status == 0
    assert captured.err == ""
    cli_result = json.loads(captured.out)
    assert cli_result["status"] == OperationStatus.SUCCEEDED.value
    assert cli_result == python_result.to_record()
    assert len(calls) == 2


def test_module_entry_and_ops_wrapper_match_agent_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(
        Operation.WORKFLOW_PREVIEW,
        repository_root,
        state_root,
        directory=str(repository_root),
        prompt_source={"kind": "inline", "content_cid": "prompt:module"},
        output_mode="markdown",
    )

    def handler(_request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={"proposal_root": "plan:module"},
            changed=False,
            checks=("schema",),
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={Operation.WORKFLOW_PREVIEW: handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 2_000,
    )
    python_result = service.workflow_preview(request)

    module_out = io.StringIO()
    module_err = io.StringIO()
    module_code = run_prompt_workflow_cli(
        ["workflow-preview", "--request-json", request.to_json()],
        stdout_stream=module_out,
        stderr_stream=module_err,
        control_service=service,
    )
    assert module_code == 0
    assert module_err.getvalue() == ""
    module_record = json.loads(module_out.getvalue())
    assert module_record == python_result.to_record()

    exit_status = cli.main(
        [
            "agent",
            "workflow-preview",
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    assert exit_status == 0
    assert json.loads(captured.out) == python_result.to_record()

    assert OPS_WRAPPER.is_file()
    completed = subprocess.run(
        [sys.executable, str(OPS_WRAPPER), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "workflow-preview" in completed.stdout


def test_prompt_source_mutual_exclusion(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("hello", encoding="utf-8")
    out = io.StringIO()
    err = io.StringIO()
    code = run_prompt_workflow_cli(
        [
            "workflow-preview",
            "--prompt",
            "inline",
            "--prompt-file",
            str(prompt_file),
        ],
        stdout_stream=out,
        stderr_stream=err,
    )
    assert code == PROMPT_WORKFLOW_CLI_EXIT_INVALID
