"""PCPC-028 public control surface parity and mutation safety tests."""

from __future__ import annotations

from io import StringIO
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    PROCEDURE_MUTATION_OPERATIONS,
    PROCEDURE_READ_OPERATIONS,
    ProcedureControlRequest,
    ProcedureControlServiceAdapter,
    ProcedureMCPAdapter,
    ProcedureOperation,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.cli import ProcedureCLI


def _permit(_request):
    return {"allowed": True, "authorization_id": "auth-current"}


def _mutation(operation: ProcedureOperation, **extra):
    return ProcedureControlRequest(
        operation=operation,
        target={"procedure_id": "procedure-1"},
        authorization={"external": "evidence"},
        idempotency_key="key-1",
        lease_fence={"lease_id": "lease-1", "fencing_epoch": 7},
        request_id="request-1",
        **extra,
    )


def test_every_required_procedure_operation_is_discoverable_and_typed():
    service = ProcedureControlServiceAdapter()
    assert set(service.discover()) == set(ProcedureOperation)
    assert set(PROCEDURE_READ_OPERATIONS) | set(PROCEDURE_MUTATION_OPERATIONS) == set(ProcedureOperation)
    assert all(item.value.startswith("procedures.") for item in service.discover())
    assert set(ProcedureMCPAdapter(service).list_tools()) == {item.value for item in ProcedureOperation}


def test_mutation_requires_independent_authorization_fence_and_idempotency():
    calls = []
    service = ProcedureControlServiceAdapter(
        handlers={ProcedureOperation.PROMOTE: lambda request: calls.append(request) or {"promoted": True}},
        authorization_validator=_permit,
        lease_fence_validator=lambda _request: True,
    )
    request = _mutation(ProcedureOperation.PROMOTE)
    result = service.execute(request)
    assert result.successful and result.audit is not None
    assert result.audit.authorization_id == "auth-current"
    assert calls == [request]
    assert service.execute(request).replayed
    assert calls == [request]
    with pytest.raises(Exception):
        service.execute(_mutation(ProcedureOperation.PROMOTE, idempotency_key=""))


def test_dry_run_is_authorized_audited_and_never_dispatches():
    calls = []
    service = ProcedureControlServiceAdapter(
        handlers={ProcedureOperation.REVOKE: lambda request: calls.append(request) or {}},
        authorization_validator=_permit,
        lease_fence_validator=lambda _request: True,
    )
    result = service.execute(_mutation(ProcedureOperation.REVOKE, dry_run=True))
    assert result.successful
    assert result.data["dry_run"] is True
    assert result.audit is not None and result.audit.dry_run
    assert calls == []


def test_cli_and_mcp_call_the_same_service_without_shelling():
    service = ProcedureControlServiceAdapter(
        handlers={ProcedureOperation.GET: lambda _request: {"procedure_id": "procedure-1"}},
    )
    mcp = ProcedureMCPAdapter(service)
    assert mcp.call_tool("procedures.get", {"target": {"procedure_id": "procedure-1"}}).successful
    stdout = StringIO()
    assert ProcedureCLI(service).run(["get", "--target-json", '{"procedure_id":"procedure-1"}'], stdout=stdout) == 0
    assert json.loads(stdout.getvalue())["operation"] == "procedures.get"
