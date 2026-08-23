"""LGCVF-102: Python / CLI / MCP projections of one semantic service.

Required evidence: transport parity across Python, CLI, and MCP, plus
preview-no-write coverage for mutation operations.  Wrappers perform no
independent semantics and do not publish an MCP++ profile.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    SEMANTIC_MUTATION_OPERATIONS,
    SEMANTIC_SERVICE_OPERATIONS,
    IpfsDatasetsLogicProvider,
    SemanticService,
    SemanticServiceError,
    SemanticServiceMode,
    SemanticServiceOperation,
    SemanticServiceRequest,
    SemanticServiceResult,
    create_semantic_service,
    semantic_service_capability_report,
    semantic_service_main,
)


REQUIRED_OPERATIONS = (
    "capability",
    "snapshot",
    "impact",
    "contracts",
    "abstract",
    "discharge",
    "verify",
    "prove",
    "counterexample",
    "interpolate",
    "synthesize",
    "repair",
    "context",
    "benchmark",
    "explain",
    "replay",
)


def _request(
    operation: SemanticServiceOperation | str,
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "operation": getattr(operation, "value", operation),
        "repository_id": "repository:lgcvf-102",
        "tree_id": "tree:parity",
        "objective_id": "LGCVF-102",
        "policy_id": "policy:transport-parity",
    }
    payload.update(overrides)
    return payload


def _records(
    service: SemanticService,
    request: dict[str, object] | SemanticServiceRequest | list[str] | str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    python_result = service.python(request)
    cli_result = service.cli(request)
    mcp_result = service.mcp(request)
    return python_result.to_record(), cli_result.to_record(), mcp_result.to_record()


def test_catalog_lists_every_required_operation() -> None:
    assert tuple(item.value for item in SEMANTIC_SERVICE_OPERATIONS) == (
        REQUIRED_OPERATIONS
    )
    report = semantic_service_capability_report()
    assert report["operation_ids"] == list(REQUIRED_OPERATIONS)
    assert report["mutation_default_mode"] == "preview"
    assert report["wrappers_have_independent_semantics"] is False
    assert report["mcp_plus_plus_profile"] is False
    assert set(report["surfaces"]) == {"python", "cli", "mcp"}
    assert SemanticService.discovery() == report


def test_python_cli_mcp_are_the_same_typed_entrypoint() -> None:
    service = create_semantic_service()
    assert service.python is service.execute
    assert service.cli is service.execute
    assert service.mcp is service.execute
    assert service.python is service.cli is service.mcp


@pytest.mark.parametrize("operation", list(SEMANTIC_SERVICE_OPERATIONS))
def test_python_cli_mcp_canonical_records_match(
    operation: SemanticServiceOperation,
) -> None:
    service = create_semantic_service()
    request = _request(operation)
    if operation is SemanticServiceOperation.REPLAY:
        original = service.execute(_request(SemanticServiceOperation.SNAPSHOT))
        request = _request(operation, parameters={"result_id": original.result_id})
    python_record, cli_record, mcp_record = _records(service, request)
    assert python_record == cli_record == mcp_record
    assert python_record["operation"] == operation.value
    assert python_record["transport"] == "shared"
    assert python_record["mcp_plus_plus_profile"] is False
    assert python_record["completion_authority"] is False
    assert python_record["non_authoritative"] is True
    if operation in SEMANTIC_MUTATION_OPERATIONS:
        assert python_record["preview"] is True
        assert python_record["mutated"] is False
        assert python_record["wrote_effects"] == []
        assert python_record["mode"] == SemanticServiceMode.PREVIEW.value
    else:
        assert python_record["read_only"] is True
        assert python_record["mutated"] is False


def test_cli_argv_and_mcp_tool_call_match_python_mapping() -> None:
    service = create_semantic_service()
    mapping = _request(
        SemanticServiceOperation.REPAIR,
        parameters={"path": "module.py", "contents": "preview-only"},
    )
    python_record = service.python(mapping).to_record()
    cli_record = service.cli(
        [
            "repair",
            "--repository-id",
            "repository:lgcvf-102",
            "--tree-id",
            "tree:parity",
            "--objective-id",
            "LGCVF-102",
            "--policy-id",
            "policy:transport-parity",
            "--parameter",
            "path=module.py",
            "--parameter",
            "contents=preview-only",
        ]
    ).to_record()
    mcp_record = service.mcp(
        {
            "name": "lgcvf_semantic_repair",
            "arguments": mapping,
        }
    ).to_record()
    assert python_record == cli_record == mcp_record
    assert python_record["preview"] is True
    assert python_record["wrote_effects"] == []


def test_mutation_defaults_to_preview_even_when_apply_is_set() -> None:
    service = create_semantic_service()
    result = service.execute(
        _request(
            SemanticServiceOperation.REPAIR,
            apply=True,
            parameters={"path": "module.py", "contents": "should-not-write"},
        )
    )
    assert result.preview is True
    assert result.mode is SemanticServiceMode.PREVIEW
    assert result.mutated is False
    assert result.wrote_effects == ()
    assert service.write_log == ()


def test_preview_does_not_write_workspace_bytes(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("original\n", encoding="utf-8")
    service = create_semantic_service(workspace=tmp_path)
    for operation in (
        SemanticServiceOperation.REPAIR,
        SemanticServiceOperation.SYNTHESIZE,
    ):
        result = service.execute(
            _request(
                operation,
                parameters={"path": "module.py", "contents": "mutated\n"},
            )
        )
        assert result.preview is True
        assert result.mutated is False
        assert result.wrote_effects == ()
        assert result.proposed_effects
        assert target.read_text(encoding="utf-8") == "original\n"
    assert service.write_log == ()
    assert service.artifacts.get("module.py") != "mutated\n"


def test_explicit_apply_writes_once_and_is_transport_identical(
    tmp_path: Path,
) -> None:
    target = tmp_path / "module.py"
    target.write_text("original\n", encoding="utf-8")
    python_service = create_semantic_service(workspace=tmp_path / "python")
    cli_service = create_semantic_service(workspace=tmp_path / "cli")
    mcp_service = create_semantic_service(workspace=tmp_path / "mcp")
    request = _request(
        SemanticServiceOperation.REPAIR,
        apply=True,
        dry_run=False,
        mode="apply",
        parameters={"path": "module.py", "contents": "repaired\n"},
    )
    python_result = python_service.python(request)
    cli_result = cli_service.cli(
        [
            "repair",
            "--apply",
            "--request-json",
            json.dumps(
                {
                    "repository_id": "repository:lgcvf-102",
                    "tree_id": "tree:parity",
                    "objective_id": "LGCVF-102",
                    "policy_id": "policy:transport-parity",
                    "parameters": {
                        "path": "module.py",
                        "contents": "repaired\n",
                    },
                }
            ),
        ]
    )
    mcp_result = mcp_service.mcp(
        {
            "name": "lgcvf_semantic_repair",
            "arguments": request,
        }
    )
    python_record = dict(python_result.to_record())
    cli_record = dict(cli_result.to_record())
    mcp_record = dict(mcp_result.to_record())
    for record in (python_record, cli_record, mcp_record):
        record.pop("request_id")
        record.pop("result_id")
        record["data"].pop("replay_parameters", None)
    assert python_record == cli_record == mcp_record
    for service in (python_service, cli_service, mcp_service):
        written = (service._workspace / "module.py").read_text(encoding="utf-8")
        assert written == "repaired\n"
        assert service.write_log
        assert service.write_log[0]["path"] == "module.py"
    assert python_result.preview is False
    assert python_result.mutated is True
    assert python_result.wrote_effects
    assert target.read_text(encoding="utf-8") == "original\n"


def test_replay_of_mutation_defaults_to_preview(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = create_semantic_service(workspace=workspace)
    applied = service.execute(
        _request(
            SemanticServiceOperation.SYNTHESIZE,
            apply=True,
            dry_run=False,
            parameters={"path": "generated.py", "contents": "candidate\n"},
        )
    )
    assert (workspace / "generated.py").read_text(encoding="utf-8") == "candidate\n"
    (workspace / "generated.py").write_text("changed-after-apply\n", encoding="utf-8")
    replayed = service.execute(
        _request(
            SemanticServiceOperation.REPLAY,
            parameters={"result_id": applied.result_id},
        )
    )
    assert replayed.preview is True
    assert replayed.mutated is False
    assert replayed.wrote_effects == ()
    assert (workspace / "generated.py").read_text(encoding="utf-8") == (
        "changed-after-apply\n"
    )


def test_request_and_result_round_trip_without_surface_fields() -> None:
    request = SemanticServiceRequest.from_dict(
        _request(SemanticServiceOperation.CONTEXT, parameters={"token_budget": 32})
    )
    restored = SemanticServiceRequest.from_json(request.to_json())
    assert restored.to_record() == request.to_record()
    assert "python" not in request.to_json()
    assert "cli" not in request.to_json()
    service = create_semantic_service()
    result = service.execute(request)
    assert SemanticServiceResult.from_dict(result.to_dict()).to_record() == (
        result.to_record()
    )


def test_mcp_tool_catalog_is_not_an_mcp_plus_plus_profile() -> None:
    service = create_semantic_service()
    tools = service.mcp_tools()
    assert [tool["name"] for tool in tools] == [
        f"lgcvf_semantic_{item}" for item in REQUIRED_OPERATIONS
    ]
    for tool in tools:
        assert tool["preview_default"] is True
        assert tool["mcp_plus_plus_profile"] is False
        assert "mcp++" not in json.dumps(tool)
        assert tool["input_schema"]["properties"]["mode"]["default"] == "preview"


def test_semantic_service_main_prints_shared_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = semantic_service_main(["capability"])
    captured = capsys.readouterr()
    assert status == 0
    record = json.loads(captured.out)
    assert record["operation"] == "capability"
    assert record["transport"] == "shared"
    assert record["data"]["operation_ids"] == list(REQUIRED_OPERATIONS)
    buffer = io.StringIO()
    semantic_service_main(["explain"], stdout=buffer)
    explained = json.loads(buffer.getvalue())
    assert explained["operation"] == "explain"
    assert "preview=True" in explained["data"]["explanation"]


def test_unknown_operation_and_escaped_write_fail_closed(tmp_path: Path) -> None:
    service = create_semantic_service(workspace=tmp_path)
    with pytest.raises(SemanticServiceError, match="unknown semantic service"):
        service.execute("not-an-operation")
    with pytest.raises(SemanticServiceError, match="escapes"):
        service.execute(
            _request(
                SemanticServiceOperation.REPAIR,
                apply=True,
                dry_run=False,
                parameters={
                    "path": "../outside.py",
                    "contents": "nope",
                },
            )
        )
    assert list(tmp_path.iterdir()) == []


def test_provider_bound_service_stays_non_authoritative() -> None:
    class _Provider:
        provider_id = "hammer"

    service = SemanticService(provider=_Provider())
    prove = service.execute("prove")
    verify = service.execute("verify")
    assert prove.data["provider_id"] == "hammer"
    assert verify.data["provider_id"] == "hammer"
    assert prove.data["proof_success"] is False
    assert verify.data["proof_success"] is False
    assert prove.data["candidate_authoritative"] is False


def test_logic_provider_exposes_the_shared_semantic_service() -> None:
    provider = IpfsDatasetsLogicProvider()
    service = provider.semantic_service()
    assert service.python is service.cli is service.mcp is service.execute
    python_record, cli_record, mcp_record = _records(service, "capability")
    assert python_record == cli_record == mcp_record
    assert python_record["operation"] == "capability"
    assert python_record["mcp_plus_plus_profile"] is False
    assert python_record["data"]["mutation_default_mode"] == "preview"
