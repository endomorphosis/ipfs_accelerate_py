from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.cli import (
    ALL_OPERATIONS,
    MUTATION_OPERATIONS,
    READ_OPERATIONS,
    ExpertControlRequest,
    ResidualExpertControlBackend,
    mcp_dispatch,
    register_expert_operations,
)
from .helpers import admission


def test_catalog_cli_mcp_parity_and_training_unavailable() -> None:
    catalog = register_expert_operations()
    assert catalog["read"] == READ_OPERATIONS
    assert catalog["mutation"] == MUTATION_OPERATIONS
    backend = ResidualExpertControlBackend()
    dry = backend.execute(ExpertControlRequest(operation="revoke_expert", expert_id="exp:1", dry_run=True))
    assert dry.status == "dry_run"
    mcp = mcp_dispatch(backend, ExpertControlRequest(operation="list_experts"))
    cli = backend.execute(ExpertControlRequest(operation="list_experts"))
    assert mcp.operation == cli.operation
    denied = backend.execute(ExpertControlRequest(operation="start_training"))
    assert denied.status == "training_unavailable"
    record, _ = admission(admitted=True)
    admitted = backend.execute(
        ExpertControlRequest(operation="start_training", admission=record, idempotency_key="k1")
    )
    assert admitted.ok is True
    assert set(ALL_OPERATIONS) == set(READ_OPERATIONS + MUTATION_OPERATIONS)
