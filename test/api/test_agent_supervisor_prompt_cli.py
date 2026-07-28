"""ASI-152: CLI and python -m prompt-workflow entry surface tests."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    AGENT_CLI_EXIT_INVALID,
    AGENT_CLI_EXIT_SUCCESS,
    COMMAND_OPERATIONS,
    PROMPT_CLI_REQUIREMENT_ID,
    PROMPT_WORKFLOW_CLI_COMMANDS,
    agent_cli_discovery_manifest,
    build_agent_request,
    register_agent_cli,
    run_agent_cli,
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
    PROMPT_WORKFLOW_CLI_COMMANDS as MODULE_PROMPT_COMMANDS,
    PROMPT_WORKFLOW_CLI_EXIT_INVALID,
    PromptSource,
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


def _service(
    repository_root: Path,
    state_root: Path,
    *,
    operation: Operation,
    handler: Any,
    clock_ms: int = 1_000,
) -> SupervisorControlService:
    return SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={operation: handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: clock_ms,
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
    assert MODULE_PROMPT_COMMANDS == PROMPT_WORKFLOW_CLI_COMMANDS
    assert PROMPT_CLI_REQUIREMENT_ID.startswith("requirement:")


def test_help_and_import_are_side_effect_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Import already happened at module load; re-import must not start providers.
    __import__("ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow")
    __import__("ipfs_accelerate_py.agent_supervisor.prompt_workflow")
    __import__("ipfs_accelerate_py.agent_supervisor.control_cli")
    parser = build_prompt_workflow_arg_parser()
    with pytest.raises(SystemExit) as exited:
        parser.parse_args(["--help"])
    assert exited.value.code == 0
    code = run_prompt_workflow_cli([])
    assert code == PROMPT_WORKFLOW_CLI_EXIT_INVALID
    captured = capsys.readouterr()
    assert "workflow-preview" in captured.out
    assert "usage" in captured.out.lower() or "prompt-workflow" in captured.out

    # Agent CLI registration/discovery must remain provider- and process-free.
    root = argparse.ArgumentParser()
    sub = root.add_subparsers()
    agent = register_agent_cli(sub)
    for command in PROMPT_WORKFLOW_CLI_COMMANDS:
        assert command in COMMAND_OPERATIONS
    # Parser construction alone is enough; the group is registered.
    assert "agent" in agent.prog


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

    service = _service(
        repository_root,
        state_root,
        operation=Operation.WORKFLOW_PREVIEW,
        handler=handler,
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

    service = _service(
        repository_root,
        state_root,
        operation=Operation.WORKFLOW_PREVIEW,
        handler=handler,
        clock_ms=2_000,
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


def test_agent_cli_prompt_file_convenience_is_body_free(
    tmp_path: Path,
) -> None:
    repository_root = (tmp_path / "repo").resolve()
    state_root = (tmp_path / "state").resolve()
    repository_root.mkdir()
    state_root.mkdir()
    secret = "super-secret-prompt-body-do-not-log"
    prompt_file = tmp_path / "request.md"
    prompt_file.write_text(secret, encoding="utf-8")
    expected_cid = PromptSource.file(prompt_file.name, text=secret).prompt_cid

    captured: list[OperationRequest] = []

    def handler(incoming: OperationRequest) -> BackendResponse:
        captured.append(incoming)
        return BackendResponse(
            data={"proposal_root": "plan:file"},
            changed=False,
            checks=("schema",),
        )

    service = _service(
        repository_root,
        state_root,
        operation=Operation.WORKFLOW_PREVIEW,
        handler=handler,
        clock_ms=3_000,
    )
    binding = _binding(repository_root, state_root)
    argv_ns = argparse.Namespace(
        agent_command="workflow-preview",
        agent_operation=Operation.WORKFLOW_PREVIEW.value,
        request_json=None,
        request_file=None,
        parameters_json=None,
        **binding,
        path=None,
        limit=None,
        offset=None,
        cursor=None,
        event_cursor=None,
        task_header_prefix=None,
        target_id=None,
        service_id=None,
        task_id=None,
        bundle_id=None,
        lane_id=None,
        stream_id=None,
        receipt_id=None,
        cache_namespace=None,
        artifact_id=None,
        validation_id=None,
        reason=None,
        requested_state=None,
        expected_effects_json=None,
        idempotency_key=None,
        authorization_json=None,
        authorization_file=None,
        lease_id=None,
        fencing_epoch=None,
        dry_run=True,
        max_items=None,
        max_bytes=None,
        max_text_bytes=None,
        timeout_ms=None,
        watch_count=1,
        watch_interval_ms=0,
        output_json=True,
        human=False,
        directory=str(repository_root),
        prompt=None,
        prompt_file=prompt_file,
        prompt_stdin=False,
        output_mode="both",
        markdown_path="plan.todo.md",
        duckdb_path="plan.duckdb",
        start_after=False,
        allow_llm_fallback=False,
        max_actions=None,
        budget_json=None,
    )
    out = io.StringIO()
    err = io.StringIO()
    code = run_agent_cli(argv_ns, service=service, stdout=out, stderr=err)
    assert code == AGENT_CLI_EXIT_SUCCESS
    assert err.getvalue() == ""
    record = json.loads(out.getvalue())
    assert record["status"] == OperationStatus.SUCCEEDED.value
    assert len(captured) == 1
    source = captured[0].parameters["prompt_source"]
    assert source == {
        "kind": "file",
        "content_cid": expected_cid,
        "artifact_ref": prompt_file.name,
    }
    # No raw prompt or absolute path retained on durable request params.
    serialized = json.dumps(captured[0].to_record())
    assert secret not in serialized
    assert secret not in out.getvalue()
    assert str(prompt_file.resolve()) not in serialized
    assert captured[0].parameters["output_mode"] == "both"
    assert captured[0].parameters["markdown_path"] == "plan.todo.md"
    assert captured[0].parameters["duckdb_path"] == "plan.duckdb"
    assert captured[0].dry_run is True


def test_agent_cli_rejects_mutation_without_authority(tmp_path: Path) -> None:
    repository_root = (tmp_path / "repo").resolve()
    state_root = (tmp_path / "state").resolve()
    repository_root.mkdir()
    state_root.mkdir()
    binding = _binding(repository_root, state_root)
    ns = argparse.Namespace(
        agent_command="rescue",
        agent_operation=Operation.RESCUE.value,
        request_json=None,
        request_file=None,
        parameters_json=json.dumps(
            {
                "incident_cid": "incident:one",
                "incident_root": "incident-root:one",
                "incident_repository_id": binding["repository_id"],
                "incident_tree_id": binding["tree_id"],
                "incident_objective_id": binding["objective_id"],
                "incident_objective_revision": binding["objective_revision"],
                "incident_policy_id": binding["policy_id"],
                "incident_policy_revision": binding["policy_revision"],
                "rescue_plan_cid": "plan:one",
                "rescue_plan_root": "plan-root:one",
                "rescue_plan_incident_cid": "incident:one",
                "rescue_plan_tree_id": binding["tree_id"],
            }
        ),
        **binding,
        path=None,
        limit=None,
        offset=None,
        cursor=None,
        event_cursor=None,
        task_header_prefix=None,
        target_id=None,
        service_id=None,
        task_id=None,
        bundle_id=None,
        lane_id=None,
        stream_id=None,
        receipt_id=None,
        cache_namespace=None,
        artifact_id=None,
        validation_id=None,
        reason=None,
        requested_state=None,
        expected_effects_json=None,
        idempotency_key=None,
        authorization_json=None,
        authorization_file=None,
        lease_id=None,
        fencing_epoch=None,
        dry_run=False,
        max_items=None,
        max_bytes=None,
        max_text_bytes=None,
        timeout_ms=None,
        watch_count=1,
        watch_interval_ms=0,
        output_json=True,
        human=False,
        allow_llm_fallback=True,
        max_actions=2,
        budget_json=None,
        directory=None,
        prompt=None,
        prompt_file=None,
        prompt_stdin=False,
        output_mode=None,
        markdown_path=None,
        duckdb_path=None,
        start_after=False,
    )
    with pytest.raises(Exception) as raised:
        build_agent_request(ns)
    message = str(raised.value)
    assert "authorization" in message.lower() or "mutation" in message.lower()


def test_agent_cli_rescue_preview_budgets_and_human_output(
    tmp_path: Path,
) -> None:
    repository_root = (tmp_path / "repo").resolve()
    state_root = (tmp_path / "state").resolve()
    repository_root.mkdir()
    state_root.mkdir()
    binding = _binding(repository_root, state_root)
    parameters = {
        "incident_cid": "incident:one",
        "incident_root": "incident-root:one",
        "incident_repository_id": binding["repository_id"],
        "incident_tree_id": binding["tree_id"],
        "incident_objective_id": binding["objective_id"],
        "incident_objective_revision": binding["objective_revision"],
        "incident_policy_id": binding["policy_id"],
        "incident_policy_revision": binding["policy_revision"],
    }
    captured: list[OperationRequest] = []

    def handler(incoming: OperationRequest) -> BackendResponse:
        captured.append(incoming)
        return BackendResponse(
            data={
                "rescue_plan_root": "plan:rescue",
                "next_event_cursor": "cursor:abc",
            },
            changed=False,
            checks=("schema",),
        )

    service = _service(
        repository_root,
        state_root,
        operation=Operation.RESCUE_PREVIEW,
        handler=handler,
        clock_ms=4_000,
    )
    ns = argparse.Namespace(
        agent_command="rescue-preview",
        agent_operation=Operation.RESCUE_PREVIEW.value,
        request_json=None,
        request_file=None,
        parameters_json=json.dumps(parameters),
        **binding,
        path=None,
        limit=None,
        offset=None,
        cursor=None,
        event_cursor=None,
        task_header_prefix=None,
        target_id=None,
        service_id=None,
        task_id=None,
        bundle_id=None,
        lane_id=None,
        stream_id=None,
        receipt_id=None,
        cache_namespace=None,
        artifact_id=None,
        validation_id=None,
        reason=None,
        requested_state=None,
        expected_effects_json=None,
        idempotency_key=None,
        authorization_json=None,
        authorization_file=None,
        lease_id=None,
        fencing_epoch=None,
        dry_run=False,
        max_items=None,
        max_bytes=None,
        max_text_bytes=None,
        timeout_ms=30_000,
        watch_count=1,
        watch_interval_ms=0,
        output_json=False,
        human=True,
        allow_llm_fallback=True,
        max_actions=3,
        budget_json=None,
        directory=None,
        prompt=None,
        prompt_file=None,
        prompt_stdin=False,
        output_mode=None,
        markdown_path=None,
        duckdb_path=None,
        start_after=False,
    )
    out = io.StringIO()
    err = io.StringIO()
    code = run_agent_cli(ns, service=service, stdout=out, stderr=err)
    assert code == AGENT_CLI_EXIT_SUCCESS
    assert err.getvalue() == ""
    text = out.getvalue()
    assert "status=succeeded" in text
    assert "event_cursor=cursor:abc" in text
    assert len(captured) == 1
    assert captured[0].parameters["allow_llm_fallback"] is True
    assert captured[0].parameters["max_actions"] == 3
    assert captured[0].dry_run is True
    assert captured[0].bounds.timeout_ms == 30_000


def test_prompt_source_normalization_strips_module_shape(tmp_path: Path) -> None:
    repository_root = (tmp_path / "repo").resolve()
    state_root = (tmp_path / "state").resolve()
    repository_root.mkdir()
    state_root.mkdir()
    secret = "module-style-secret-body"
    source = PromptSource.inline(secret)
    module_shape = {
        key: value
        for key, value in source.to_record().items()
        if key not in {"schema", "contract_version", "content_id"}
    }
    binding = _binding(repository_root, state_root)
    ns = argparse.Namespace(
        agent_command="workflow-preview",
        agent_operation=Operation.WORKFLOW_PREVIEW.value,
        request_json=None,
        request_file=None,
        parameters_json=json.dumps(
            {
                "directory": str(repository_root),
                "prompt_source": module_shape,
                "output_mode": "markdown",
            }
        ),
        **binding,
        path=None,
        limit=None,
        offset=None,
        cursor=None,
        event_cursor=None,
        task_header_prefix=None,
        target_id=None,
        service_id=None,
        task_id=None,
        bundle_id=None,
        lane_id=None,
        stream_id=None,
        receipt_id=None,
        cache_namespace=None,
        artifact_id=None,
        validation_id=None,
        reason=None,
        requested_state=None,
        expected_effects_json=None,
        idempotency_key=None,
        authorization_json=None,
        authorization_file=None,
        lease_id=None,
        fencing_epoch=None,
        dry_run=True,
        max_items=None,
        max_bytes=None,
        max_text_bytes=None,
        timeout_ms=None,
        watch_count=1,
        watch_interval_ms=0,
        output_json=True,
        human=False,
        directory=None,
        prompt=None,
        prompt_file=None,
        prompt_stdin=False,
        output_mode=None,
        markdown_path=None,
        duckdb_path=None,
        start_after=False,
        allow_llm_fallback=False,
        max_actions=None,
        budget_json=None,
    )
    request = build_agent_request(ns)
    assert request.parameters["prompt_source"] == {
        "kind": "inline",
        "content_cid": source.prompt_cid,
    }
    assert secret not in json.dumps(request.to_record())


def test_path_escape_and_request_overlay_rejected(tmp_path: Path) -> None:
    repository_root = (tmp_path / "repo").resolve()
    state_root = (tmp_path / "state").resolve()
    repository_root.mkdir()
    state_root.mkdir()
    binding = _binding(repository_root, state_root)
    request = _request(
        Operation.WORKFLOW_PREVIEW,
        repository_root,
        state_root,
        directory=str(repository_root),
        prompt_source={"kind": "inline", "content_cid": "prompt:x"},
    )
    ns = argparse.Namespace(
        agent_command="workflow-preview",
        agent_operation=Operation.WORKFLOW_PREVIEW.value,
        request_json=request.to_json(),
        request_file=None,
        parameters_json=None,
        **binding,
        path=None,
        limit=None,
        offset=None,
        cursor=None,
        event_cursor=None,
        task_header_prefix=None,
        target_id=None,
        service_id=None,
        task_id=None,
        bundle_id=None,
        lane_id=None,
        stream_id=None,
        receipt_id=None,
        cache_namespace=None,
        artifact_id=None,
        validation_id=None,
        reason=None,
        requested_state=None,
        expected_effects_json=None,
        idempotency_key=None,
        authorization_json=None,
        authorization_file=None,
        lease_id=None,
        fencing_epoch=None,
        dry_run=False,
        max_items=None,
        max_bytes=None,
        max_text_bytes=None,
        timeout_ms=None,
        watch_count=1,
        watch_interval_ms=0,
        output_json=True,
        human=False,
        directory=str(repository_root),
        prompt=None,
        prompt_file=None,
        prompt_stdin=False,
        output_mode=None,
        markdown_path=None,
        duckdb_path=None,
        start_after=False,
        allow_llm_fallback=False,
        max_actions=None,
        budget_json=None,
    )
    with pytest.raises(Exception) as raised:
        build_agent_request(ns)
    assert "cannot be combined" in str(raised.value)

    # Absolute artifact_ref with parent traversal is rejected.
    bad = argparse.Namespace(**vars(ns))
    bad.request_json = None
    bad.directory = None
    bad.parameters_json = json.dumps(
        {
            "directory": str(repository_root),
            "prompt_source": {
                "kind": "file",
                "content_cid": "prompt:escape",
                "artifact_ref": "../secrets",
            },
        }
    )
    with pytest.raises(Exception) as escape:
        build_agent_request(bad)
    assert "escape" in str(escape.value).lower()


def test_module_python_m_discovery_lists_all_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.prompt_workflow",
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    for command in PROMPT_WORKFLOW_CLI_COMMANDS:
        assert command in completed.stdout
    assert "prompt-file" in completed.stdout or "--prompt-file" in completed.stdout


def test_invalid_request_uses_stable_exit_code(tmp_path: Path) -> None:
    out = io.StringIO()
    err = io.StringIO()
    code = run_agent_cli(
        argparse.Namespace(
            agent_command="workflow-preview",
            agent_operation=Operation.WORKFLOW_PREVIEW.value,
            request_json="{not-json",
            request_file=None,
            parameters_json=None,
            repository_root=None,
            state_root=None,
            repository_id=None,
            tree_id=None,
            objective_id=None,
            objective_revision=None,
            policy_id=None,
            policy_revision=None,
            caller=None,
            path=None,
            limit=None,
            offset=None,
            cursor=None,
            event_cursor=None,
            task_header_prefix=None,
            target_id=None,
            service_id=None,
            task_id=None,
            bundle_id=None,
            lane_id=None,
            stream_id=None,
            receipt_id=None,
            cache_namespace=None,
            artifact_id=None,
            validation_id=None,
            reason=None,
            requested_state=None,
            expected_effects_json=None,
            idempotency_key=None,
            authorization_json=None,
            authorization_file=None,
            lease_id=None,
            fencing_epoch=None,
            dry_run=False,
            max_items=None,
            max_bytes=None,
            max_text_bytes=None,
            timeout_ms=None,
            watch_count=1,
            watch_interval_ms=0,
            output_json=True,
            human=False,
        ),
        stdout=out,
        stderr=err,
    )
    assert code == AGENT_CLI_EXIT_INVALID
    payload = json.loads(err.getvalue())
    assert payload["status"] == "invalid_request"
