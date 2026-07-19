"""P2P networking compatibility layer for MCP++.

This package preserves the historical ``mcplusplus_module.p2p`` import surface,
while delegating to canonical implementations in
``ipfs_accelerate_py.mcp_server.mcplusplus`` where available.
"""

from .. import _missing_dependency_stub

# Task queue compatibility layer (canonical implementation lives under
# ipfs_accelerate_py.mcp_server.mcplusplus.task_queue).
try:
    from .libp2p_runtime import (
        LIBP2P_COMPAT_ERROR,
        LIBP2P_INSTALL_HINT,
        PY_LIBP2P_EXTRA_PACKAGES,
        PY_LIBP2P_MAIN_SPEC,
        PY_LIBP2P_PROTOBUF_SPEC,
        create_libp2p_key_pair,
        ensure_libp2p_compatible,
        ensure_libp2p_runtime,
        get_background_trio_service,
        get_libp2p_protocol_type,
        get_libp2p_stream_interface,
        get_stream_eof_exceptions,
        have_libp2p_runtime,
        install_libp2p_runtime,
        install_libp2p_runtime_async,
        make_circuit_relay_v2,
        make_dcutr_protocol,
        make_kad_dht,
        make_libp2p_resource_manager,
        make_mdns_discovery,
        make_multiaddr,
        make_rendezvous_client,
        make_rendezvous_service,
        new_libp2p_host,
        peer_id_from_base58,
        peer_id_text,
        peerinfo_from_multiaddr,
        require_libp2p_runtime,
        running_libp2p_host,
    )
except ImportError:
    LIBP2P_COMPAT_ERROR = "MCP++ libp2p runtime unavailable"
    LIBP2P_INSTALL_HINT = "pip install 'ipfs_accelerate_py[mcp-p2p]'"
    PY_LIBP2P_EXTRA_PACKAGES = ()
    PY_LIBP2P_MAIN_SPEC = ""
    PY_LIBP2P_PROTOBUF_SPEC = ""
    create_libp2p_key_pair = _missing_dependency_stub("create_libp2p_key_pair")
    ensure_libp2p_compatible = _missing_dependency_stub("ensure_libp2p_compatible")
    ensure_libp2p_runtime = _missing_dependency_stub("ensure_libp2p_runtime")
    get_background_trio_service = _missing_dependency_stub("get_background_trio_service")
    get_libp2p_protocol_type = _missing_dependency_stub("get_libp2p_protocol_type")
    get_libp2p_stream_interface = _missing_dependency_stub("get_libp2p_stream_interface")
    get_stream_eof_exceptions = _missing_dependency_stub("get_stream_eof_exceptions")
    have_libp2p_runtime = _missing_dependency_stub("have_libp2p_runtime")
    install_libp2p_runtime = _missing_dependency_stub("install_libp2p_runtime")
    install_libp2p_runtime_async = _missing_dependency_stub("install_libp2p_runtime_async")
    make_circuit_relay_v2 = _missing_dependency_stub("make_circuit_relay_v2")
    make_dcutr_protocol = _missing_dependency_stub("make_dcutr_protocol")
    make_kad_dht = _missing_dependency_stub("make_kad_dht")
    make_libp2p_resource_manager = _missing_dependency_stub("make_libp2p_resource_manager")
    make_mdns_discovery = _missing_dependency_stub("make_mdns_discovery")
    make_multiaddr = _missing_dependency_stub("make_multiaddr")
    make_rendezvous_client = _missing_dependency_stub("make_rendezvous_client")
    make_rendezvous_service = _missing_dependency_stub("make_rendezvous_service")
    new_libp2p_host = _missing_dependency_stub("new_libp2p_host")
    peer_id_from_base58 = _missing_dependency_stub("peer_id_from_base58")
    peer_id_text = _missing_dependency_stub("peer_id_text")
    peerinfo_from_multiaddr = _missing_dependency_stub("peerinfo_from_multiaddr")
    require_libp2p_runtime = _missing_dependency_stub("require_libp2p_runtime")
    running_libp2p_host = _missing_dependency_stub("running_libp2p_host")

try:
    from .taskqueue import P2PTaskQueue, RemoteQueue
except ImportError:
    P2PTaskQueue = _missing_dependency_stub("P2PTaskQueue")
    RemoteQueue = _missing_dependency_stub("RemoteQueue")

try:
    from .workflow import (
        P2PWorkflowScheduler,
        P2PTask,
        WorkflowTag,
        MerkleClock,
        get_scheduler,
        HAVE_P2P_SCHEDULER,
    )
except ImportError:
    P2PWorkflowScheduler = _missing_dependency_stub("P2PWorkflowScheduler")
    P2PTask = _missing_dependency_stub("P2PTask")
    WorkflowTag = _missing_dependency_stub("WorkflowTag")
    MerkleClock = _missing_dependency_stub("MerkleClock")

    def get_scheduler():
        return None

    HAVE_P2P_SCHEDULER = False

try:
    from .peer_registry import P2PPeerRegistry
except ImportError:
    P2PPeerRegistry = _missing_dependency_stub("P2PPeerRegistry")

try:
    from .bootstrap import SimplePeerBootstrap
except ImportError:
    SimplePeerBootstrap = _missing_dependency_stub("SimplePeerBootstrap")

try:
    from .connectivity import ConnectivityConfig, UniversalConnectivity

    # Backward-compatible aliases used by older call sites.
    DiscoveryConfig = ConnectivityConfig
    ConnectivityHelper = UniversalConnectivity
except ImportError:
    ConnectivityConfig = _missing_dependency_stub("ConnectivityConfig")
    UniversalConnectivity = _missing_dependency_stub("UniversalConnectivity")
    DiscoveryConfig = _missing_dependency_stub("DiscoveryConfig")
    ConnectivityHelper = _missing_dependency_stub("ConnectivityHelper")

__all__ = [
    # Task queue
    "P2PTaskQueue",
    "RemoteQueue",
    # Workflow scheduler
    "P2PWorkflowScheduler",
    "P2PTask",
    "WorkflowTag",
    "MerkleClock",
    "get_scheduler",
    "HAVE_P2P_SCHEDULER",
    # Peer discovery
    "P2PPeerRegistry",
    "SimplePeerBootstrap",
    # Connectivity
    "ConnectivityConfig",
    "UniversalConnectivity",
    "DiscoveryConfig",
    "ConnectivityHelper",
    # libp2p runtime
    "LIBP2P_COMPAT_ERROR",
    "LIBP2P_INSTALL_HINT",
    "PY_LIBP2P_EXTRA_PACKAGES",
    "PY_LIBP2P_MAIN_SPEC",
    "PY_LIBP2P_PROTOBUF_SPEC",
    "create_libp2p_key_pair",
    "ensure_libp2p_compatible",
    "ensure_libp2p_runtime",
    "get_background_trio_service",
    "get_libp2p_protocol_type",
    "get_libp2p_stream_interface",
    "get_stream_eof_exceptions",
    "have_libp2p_runtime",
    "install_libp2p_runtime",
    "install_libp2p_runtime_async",
    "make_circuit_relay_v2",
    "make_dcutr_protocol",
    "make_kad_dht",
    "make_libp2p_resource_manager",
    "make_mdns_discovery",
    "make_multiaddr",
    "make_rendezvous_client",
    "make_rendezvous_service",
    "new_libp2p_host",
    "peer_id_from_base58",
    "peer_id_text",
    "peerinfo_from_multiaddr",
    "require_libp2p_runtime",
    "running_libp2p_host",
]
