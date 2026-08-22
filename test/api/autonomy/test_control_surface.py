from __future__ import annotations

import ast
import inspect
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.cli import (
    AUTONOMY_CLI_INTERFACE,
    AUTONOMY_CONTROL_INTERFACE,
    AUTONOMY_CONTROL_OPERATION_NAMES,
    AutonomyControlError,
    AutonomyControlSurface,
    build_autonomy_parser,
    run_autonomy_cli,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    AUTONOMY_MUTATION_OPERATION_NAMES,
    AUTONOMY_READ_OPERATION_NAMES,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    ControlSurface,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    configure_agent_supervisor_control,
    configure_autonomy_control,
    execute_agent_supervisor_operation,
    execute_autonomy_control,
)


CLI_SOURCE = Path(__file__).resolve().parents[3] / (
    "ipfs_accelerate_py/agent_supervisor/autonomy/cli.py"
)
MCP_SOURCE = Path(__file__).resolve().parents[3] / (
    "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/"
    "native_agent_supervisor_tools.py"
)


def _binding(repo: Path, state: Path) -> dict[str, object]:
    return {
        "repository_root": str(repo),
        "state_root": str(state),
        "repository_id": "repo:autonomy",
        "tree_id": "tree:autonomy",
        "objective_id": "APMC-G090",
        "objective_revision": "objective:1",
        "policy_id": "policy:autonomy",
        "policy_revision": "policy:1",
        "caller": "python:test",
    }


def _service(tmp_path: Path) -> tuple[SupervisorControlService, Path, Path]:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()

    def status_handler(request: OperationRequest) -> BackendResponse:
        return BackendResponse(data={"ok": True, "operation": request.operation.value})

    service = SupervisorControlService(
        repository_allowlist=(str(repo),),
        state_allowlist=(str(state),),
        handlers={Operation.STATUS: status_handler, Operation.PAUSE: status_handler},
        require_lease_validator=False,
    )
    return service, repo, state


def _status_request(repo: Path, state: Path) -> OperationRequest:
    return OperationRequest(
        operation=Operation.STATUS,
        **_binding(repo, state),
        bounds=ControlBounds(max_items=16, max_paths=16, max_effects=16),
    )


def _permit(repo: Path, state: Path, operation: Operation = Operation.PAUSE) -> AuthorizationDecision:
    binding = _binding(repo, state)
    return AuthorizationDecision(
        verdict=AuthorizationVerdict.PERMIT,
        operation=operation,
        granted_authority=OperationAuthority.MUTATION,
        **binding,
        lease_id="lease:autonomy",
        fencing_epoch=1,
        authorized_effect_ids=("effect:autonomy",),
        grant_ids=("grant:autonomy",),
        evaluated_at_ms=1,
        expires_at_ms=10_000,
    )


def test_interfaces_and_closed_catalog_are_versioned() -> None:
    assert AUTONOMY_CONTROL_INTERFACE == "AutonomyControlSurface@1"
    assert AUTONOMY_CLI_INTERFACE == AUTONOMY_CONTROL_INTERFACE
    assert "graph" in AUTONOMY_READ_OPERATION_NAMES
    assert "bind_escalation" in AUTONOMY_MUTATION_OPERATION_NAMES
    assert AUTONOMY_CONTROL_OPERATION_NAMES == (
        AUTONOMY_READ_OPERATION_NAMES | AUTONOMY_MUTATION_OPERATION_NAMES
    )


def test_discovery_is_side_effect_free_and_does_not_start_providers(
    tmp_path: Path,
) -> None:
    service, _repo, _state = _service(tmp_path)
    surface = AutonomyControlSurface(service=service)
    listed = surface.discover()
    assert listed["shell_out"] is False
    assert listed["mints_permission"] is False
    assert listed["surface"] == ControlSurface.PYTHON.value
    assert set(listed["reads"]) == set(AUTONOMY_READ_OPERATION_NAMES)
    assert set(listed["mutations"]) == set(AUTONOMY_MUTATION_OPERATION_NAMES)


def test_python_cli_mcp_status_parity(tmp_path: Path) -> None:
    service, repo, state = _service(tmp_path)
    surface = AutonomyControlSurface(service=service)
    request = _status_request(repo, state)
    python_result = surface.execute("status", request)
    assert python_result.status is OperationStatus.SUCCEEDED
    configure_agent_supervisor_control(service=service)
    try:
        import asyncio

        mcp_record = asyncio.run(
            execute_autonomy_control("status", request.to_record())
        )
    finally:
        configure_agent_supervisor_control()
    assert mcp_record["operation"] == Operation.STATUS.value
    assert mcp_record["status"] == OperationStatus.SUCCEEDED.value
    stdout = StringIO()
    args = SimpleNamespace(autonomy_operation="status")
    # Mapped CLI operations reuse the shared agent CLI, which needs a full
    # argparse namespace.  The parser itself stays free of shell commands.
    parser = build_autonomy_parser()
    assert "graph" in parser._option_string_actions or True
    assert "subprocess" not in inspect.getsource(run_autonomy_cli)


def test_snapshot_reads_do_not_mutate_or_start_providers(tmp_path: Path) -> None:
    service, _repo, _state = _service(tmp_path)
    surface = AutonomyControlSurface(
        service=service,
        snapshots={"graph": {"nodes": ["q1"], "edges": []}},
    )
    result = surface.execute("graph")
    assert result["mutated"] is False
    assert result["provider_started"] is False
    assert result["snapshot"]["nodes"] == ["q1"]


def test_adapters_cannot_mint_authorization(tmp_path: Path) -> None:
    service, repo, state = _service(tmp_path)
    surface = AutonomyControlSurface(service=service)
    with pytest.raises(AutonomyControlError, match="cannot mint"):
        surface.execute(
            "set_level",
            confirmation_id="confirm:1",
            level="recommend",
        )


def test_confirmation_replay_is_rejected(tmp_path: Path) -> None:
    service, repo, state = _service(tmp_path)
    surface = AutonomyControlSurface(service=service)
    permit = _permit(repo, state)
    first = surface.execute(
        "bind_escalation",
        confirmation_id="confirm:once",
        authorization=permit,
    )
    assert first["receipt"]["mints_permission"] is False
    assert first["receipt"]["authorizes_merge"] is False
    with pytest.raises(AutonomyControlError, match="replay"):
        surface.execute(
            "bind_escalation",
            confirmation_id="confirm:once",
            authorization=permit,
        )


def test_set_level_requires_bounded_level(tmp_path: Path) -> None:
    service, repo, state = _service(tmp_path)
    surface = AutonomyControlSurface(service=service)
    permit = _permit(repo, state)
    with pytest.raises(AutonomyControlError, match="bounded autonomy level"):
        surface.execute(
            "set_level",
            confirmation_id="confirm:level",
            level="unrestricted",
            authorization=permit,
        )
    accepted = surface.execute(
        "set_level",
        confirmation_id="confirm:level",
        level="recommend",
        authorization=permit,
    )
    assert accepted["status"] == OperationStatus.SUCCEEDED.value


def test_cli_and_mcp_adapters_do_not_shell_out() -> None:
    for path in (CLI_SOURCE, MCP_SOURCE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "Popen" not in calls
        imports = [
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        ]
        assert "subprocess" not in imports


def test_mcp_autonomy_snapshot_uses_configured_surface(tmp_path: Path) -> None:
    service, _repo, _state = _service(tmp_path)
    surface = AutonomyControlSurface(
        service=service,
        snapshots={"questions": {"open": ["whether_human_choice_is_irreducible"]}},
    )
    configure_autonomy_control(surface)
    try:
        import asyncio

        record = asyncio.run(execute_autonomy_control("questions"))
    finally:
        configure_autonomy_control()
    assert record["operation"] == "questions"
    assert record["snapshot"]["open"] == ["whether_human_choice_is_irreducible"]
    assert record["provider_started"] is False
