"""
MCP++ (MCP Plus Plus) - Trio-based MCP + P2P implementation

Module name: ipfs_accelerate_py.mcplusplus_module

This module provides a Trio-native implementation of the Model Context Protocol (MCP)
with P2P (peer-to-peer) capabilities, following the MCP++ blueprint from:
https://github.com/endomorphosis/Mcp-Plus-Plus

The MCP++ module implements:
- Content-addressed interface contracts (MCP-IDL)
- Immutable execution envelopes and receipts
- Capability delegation chains (UCAN)
- Temporal deontic policy evaluation
- Event DAG provenance and ordering
- P2P transport bindings (libp2p)

Architecture:
------------
The module is organized into the following submodules:

- trio/: Trio-native MCP server and client implementations
- p2p/: P2P networking layer using libp2p with Trio
- tools/: MCP tools for P2P taskqueue and workflow orchestration
- tests/: Test infrastructure for validating MCP++ implementation

Key differences from existing MCP implementation:
-------------------------------------------------
1. Trio-first: All async operations use Trio nurseries, cancel scopes, etc.
2. No bridging: Direct Trio execution without asyncio-to-Trio bridges
3. Unified P2P: libp2p operations run natively in the Trio event loop
4. Content-addressed: Uses CIDs for interface contracts and execution envelopes

Usage:
------
For running a Trio-backed MCP server with P2P capabilities:

    import trio
    from ipfs_accelerate_py.mcplusplus_module import TrioMCPServer
    
    async def main():
        server = TrioMCPServer(name="my-p2p-server")
        await server.run()
    
    if __name__ == "__main__":
        trio.run(main)

For more information, see:
- docs/MCP_TRIO_ROADMAP.md - Roadmap for Trio-based MCP implementation
- ipfs_accelerate_py/mcplusplus/README.md - MCP++ specification
- ipfs_accelerate_py/mcplusplus/docs/ARCHITECTURE.md - Architecture details
"""

__version__ = "0.1.0"
__author__ = "endomorphosis"

from ipfs_accelerate_py.mcp_server.compatibility import (
    _build_peer_registration_record,
    _create_storage_wrapper,
    _detect_public_ip,
    _detect_runner_name,
    _missing_dependency_stub,
    _resolve_storage_wrapper_factory,
)

# Import Profile modules (A through E)
try:
    from .interface_descriptor import (
        InterfaceDescriptor,
        MethodDescriptor,
        InterfaceRepository,
        get_interface_repository,
    )
except ImportError:
    InterfaceDescriptor = _missing_dependency_stub("InterfaceDescriptor")
    MethodDescriptor = _missing_dependency_stub("MethodDescriptor")
    InterfaceRepository = _missing_dependency_stub("InterfaceRepository")
    get_interface_repository = _missing_dependency_stub("get_interface_repository")

try:
    from .cid_ucan import (
        IntentObject,
        DecisionObject,
        ReceiptObject,
        ExecutionEnvelope,
        Delegation,
        Capability,
        DelegationEvaluator,
        EventDAG,
        DAGEvent,
        execute_with_envelope,
        get_evaluator,
        get_event_dag,
        compute_cid,
    )
except ImportError:
    IntentObject = _missing_dependency_stub("IntentObject")
    DecisionObject = _missing_dependency_stub("DecisionObject")
    ReceiptObject = _missing_dependency_stub("ReceiptObject")
    ExecutionEnvelope = _missing_dependency_stub("ExecutionEnvelope")
    Delegation = _missing_dependency_stub("Delegation")
    Capability = _missing_dependency_stub("Capability")
    DelegationEvaluator = _missing_dependency_stub("DelegationEvaluator")
    EventDAG = _missing_dependency_stub("EventDAG")
    DAGEvent = _missing_dependency_stub("DAGEvent")
    execute_with_envelope = _missing_dependency_stub("execute_with_envelope")
    get_evaluator = _missing_dependency_stub("get_evaluator")
    get_event_dag = _missing_dependency_stub("get_event_dag")
    compute_cid = _missing_dependency_stub("compute_cid")

try:
    from .temporal_policy import (
        PolicyClause,
        PolicyObject,
        PolicyDecision,
        PolicyEvaluator,
        get_policy_evaluator,
        make_permission_policy,
        make_prohibition_policy,
    )
except ImportError:
    PolicyClause = _missing_dependency_stub("PolicyClause")
    PolicyObject = _missing_dependency_stub("PolicyObject")
    PolicyDecision = _missing_dependency_stub("PolicyDecision")
    PolicyEvaluator = _missing_dependency_stub("PolicyEvaluator")
    get_policy_evaluator = _missing_dependency_stub("get_policy_evaluator")
    make_permission_policy = _missing_dependency_stub("make_permission_policy")
    make_prohibition_policy = _missing_dependency_stub("make_prohibition_policy")

try:
    from .p2p_transport import (
        P2PMessage,
        MCPp2pNode,
        PeerInfo,
        MCP_P2P_PROTOCOL,
        get_p2p_node,
    )
except ImportError:
    P2PMessage = _missing_dependency_stub("P2PMessage")
    MCPp2pNode = _missing_dependency_stub("MCPp2pNode")
    PeerInfo = _missing_dependency_stub("PeerInfo")
    MCP_P2P_PROTOCOL = "/mcp+p2p/1.0.0"
    get_p2p_node = _missing_dependency_stub("get_p2p_node")

# Import key components
try:
    from .trio import (
        TrioMCPServer,
        ServerConfig,
        create_app,
        TrioMCPClient,
        ClientConfig,
        call_tool,
    )
except ImportError:
    TrioMCPServer = _missing_dependency_stub("TrioMCPServer")
    ServerConfig = _missing_dependency_stub("ServerConfig")
    create_app = _missing_dependency_stub("create_app")
    TrioMCPClient = _missing_dependency_stub("TrioMCPClient")
    ClientConfig = _missing_dependency_stub("ClientConfig")
    call_tool = _missing_dependency_stub("call_tool")

try:
    from .p2p import P2PTaskQueue, P2PWorkflowScheduler
except ImportError:
    P2PTaskQueue = _missing_dependency_stub("P2PTaskQueue")
    P2PWorkflowScheduler = _missing_dependency_stub("P2PWorkflowScheduler")

__all__ = [
    "__version__",
    "__author__",
    # Profile A: MCP-IDL
    "InterfaceDescriptor",
    "MethodDescriptor",
    "InterfaceRepository",
    "get_interface_repository",
    # Profile B: CID-Native Execution
    "IntentObject",
    "DecisionObject",
    "ReceiptObject",
    "ExecutionEnvelope",
    "EventDAG",
    "DAGEvent",
    "execute_with_envelope",
    "get_event_dag",
    "compute_cid",
    # Profile C: UCAN Delegation
    "Delegation",
    "Capability",
    "DelegationEvaluator",
    "get_evaluator",
    # Profile D: Temporal Deontic Policy
    "PolicyClause",
    "PolicyObject",
    "PolicyDecision",
    "PolicyEvaluator",
    "get_policy_evaluator",
    "make_permission_policy",
    "make_prohibition_policy",
    # Profile E: P2P Transport
    "P2PMessage",
    "MCPp2pNode",
    "PeerInfo",
    "MCP_P2P_PROTOCOL",
    "get_p2p_node",
    # Trio server/client
    "TrioMCPServer",
    "ServerConfig",
    "create_app",
    "TrioMCPClient",
    "ClientConfig",
    "call_tool",
    "P2PTaskQueue",
    "P2PWorkflowScheduler",
]
