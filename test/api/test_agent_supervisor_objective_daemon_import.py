"""Regression coverage for the objective-daemon import dependency graph."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    DEFAULT_MCP_CONTRACT_CATALOG,
    MCP_CONTRACT_CATALOG_INTERFACE,
    McpClaimFamily,
    McpContractCatalog,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_invocation_trace import (
    MCP_INVOCATION_TRACE_INTERFACE,
    InvocationTerminalState,
    McpInvocationTrace,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    default_objective_path,
)
from ipfs_accelerate_py.agent_supervisor.proof import multi_prover_router


def test_objective_daemon_imports_reviewed_mcp_contract_dependencies() -> None:
    """The supervisor's objective read path must import without substitutions."""

    assert callable(default_objective_path)
    assert isinstance(DEFAULT_MCP_CONTRACT_CATALOG, McpContractCatalog)
    assert MCP_CONTRACT_CATALOG_INTERFACE == "McpContractCatalog@1"
    assert McpClaimFamily.NO_DYNAMIC_AUTHORITY.value == "NoDynamicAuthority"
    assert MCP_INVOCATION_TRACE_INTERFACE == "McpInvocationTrace@1"
    assert InvocationTerminalState.NOT_MEASURED.value == "not_measured"
    assert McpInvocationTrace.__module__.endswith(".mcp_invocation_trace")
    assert multi_prover_router.AUTHORITY_LATTICE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/authority-lattice@1"
    )
    assert multi_prover_router.HAMMER_TRACE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/hammer-trace@1"
    )
    assert multi_prover_router.AUTHORITATIVE_DISPOSITION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/authoritative-disposition@1"
    )
