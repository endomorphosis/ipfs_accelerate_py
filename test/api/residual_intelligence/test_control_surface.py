from __future__ import annotations

from io import StringIO
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    EXPERT_MUTATION_OPERATIONS,
    EXPERT_READ_OPERATIONS,
    ExpertControlStateStore,
    JsonExpertControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.cli import (
    ALL_OPERATIONS,
    MUTATION_OPERATIONS,
    READ_OPERATIONS,
    ExpertControlAuthorization,
    ExpertControlBudget,
    ExpertControlRequest,
    ResidualExpertControlBackend,
    mcp_dispatch,
    register_expert_operations,
    run_expert_cli,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
)


def _service(tmp_path, store: ExpertControlStateStore | None = None) -> SupervisorControlService:
    return SupervisorControlService(
        repository_allowlist=[tmp_path],
        state_allowlist=[tmp_path],
        handlers={},
        expert_state_store=store,
    )


def _mutation(
    operation: str,
    *,
    key: str = "idempotency-1",
    fence: int = 1,
    dry_run: bool = False,
    admission=None,
) -> ExpertControlRequest:
    return ExpertControlRequest(
        operation=operation,
        expert_id="expert:one",
        dry_run=dry_run,
        idempotency_key=key,
        authorization=ExpertControlAuthorization(
            subject="operator:one", permitted=True, scopes=("experts.mutate",)
        ),
        lease_id="lease:one",
        fencing_epoch=fence,
        budget=ExpertControlBudget(max_units=4, requested_units=1),
        admission=admission,
    )


def test_exact_catalog_and_direct_python_cli_mcp_parity(tmp_path) -> None:
    catalog = register_expert_operations()
    assert catalog == {"read": READ_OPERATIONS, "mutation": MUTATION_OPERATIONS}
    assert READ_OPERATIONS == EXPERT_READ_OPERATIONS
    assert MUTATION_OPERATIONS == EXPERT_MUTATION_OPERATIONS
    assert ALL_OPERATIONS == READ_OPERATIONS + MUTATION_OPERATIONS

    backend = ResidualExpertControlBackend(_service(tmp_path))
    request = ExpertControlRequest(operation="experts.list")
    python = backend.execute(request)
    mcp = mcp_dispatch(backend, request)
    output = StringIO()
    status = run_expert_cli(
        ["agent", "experts", "list", "--request-json", json.dumps(request.to_dict())],
        backend,
        stdout=output,
    )
    cli = json.loads(output.getvalue())
    assert status == 0
    assert python.to_dict() == mcp.to_dict() == cli
    assert backend.service.expert_operation_catalog() == catalog


def test_mutation_authorization_idempotency_fence_dry_run_budget_and_audit(tmp_path) -> None:
    store = ExpertControlStateStore()
    backend = ResidualExpertControlBackend(_service(tmp_path, store))

    with pytest.raises(ResidualIntelligenceError, match="authorization"):
        ExpertControlRequest(
            operation="experts.revoke", idempotency_key="key", lease_id="lease", fencing_epoch=1,
            budget=ExpertControlBudget(max_units=1, requested_units=1),
        )
    with pytest.raises(ResidualIntelligenceError, match="exceeds"):
        ExpertControlBudget(max_units=0, requested_units=1)

    dry = backend.execute(_mutation("experts.revoke", dry_run=True))
    assert dry.status == "dry_run"
    assert dry.payload["applied"] is False
    assert backend.execute(ExpertControlRequest(operation="experts.get", expert_id="expert:one")).payload["state"] == "candidate"

    applied = backend.execute(_mutation("experts.revoke"))
    replay = backend.execute(_mutation("experts.revoke"))
    assert applied.ok and applied.audit_id
    assert replay.to_dict() == {**applied.to_dict(), "idempotent_replay": True}
    assert len(store.audits) >= 3

    stale = backend.execute(_mutation("experts.demote", key="idempotency-2", fence=1))
    assert stale.status == "conflict"
    assert stale.payload["reason_code"] == "expert_stale_fence"


def test_training_requires_admission_and_state_survives_service_reconstruction(tmp_path) -> None:
    store = ExpertControlStateStore()
    first = ResidualExpertControlBackend(_service(tmp_path, store))
    unavailable = first.execute(_mutation("experts.start_training"))
    assert unavailable.status == "training_unavailable"
    assert unavailable.payload["candidate_only"] is True

    promoted = first.execute(_mutation("experts.promote", key="idempotency-3", fence=2))
    assert promoted.payload["state"] == "promoted"
    recovered = ResidualExpertControlBackend(_service(tmp_path, store))
    assert recovered.execute(ExpertControlRequest(operation="experts.get", expert_id="expert:one")).payload["state"] == "promoted"


def test_durable_store_recovers_idempotency_and_expert_state(tmp_path) -> None:
    path = tmp_path / "expert-control.json"
    first = ResidualExpertControlBackend(_service(tmp_path, JsonExpertControlStateStore(path)))
    applied = first.execute(_mutation("experts.set_shadow", key="durable-key", fence=7))
    assert path.is_file()

    recovered = ResidualExpertControlBackend(_service(tmp_path, JsonExpertControlStateStore(path)))
    replay = recovered.execute(_mutation("experts.set_shadow", key="durable-key", fence=7))
    assert replay.idempotent_replay is True
    assert replay.audit_id == applied.audit_id
    assert recovered.execute(ExpertControlRequest(operation="experts.get", expert_id="expert:one")).payload["state"] == "shadow"
