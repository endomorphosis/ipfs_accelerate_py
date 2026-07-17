"""Native IPFS network tool implementations for unified mcp_server.

Migrated from ipfs_accelerate_py/mcp/tools/ipfs_network.py.
Operations: ipfs_id, ipfs_swarm_peers, ipfs_swarm_connect,
ipfs_pubsub_pub, ipfs_dht_findpeer, ipfs_dht_findprovs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_ipfs_network_api() -> Dict[str, Any]:
    """Resolve source IPFS network APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.ipfs_tools.ipfs_tools import (  # type: ignore
            ipfs_id as _ipfs_id,
            ipfs_swarm_peers as _ipfs_swarm_peers,
        )

        return {
            "ipfs_id": _ipfs_id,
            "ipfs_swarm_peers": _ipfs_swarm_peers,
        }
    except Exception:
        logger.warning("Source IPFS network API unavailable, using fallback stubs")
        return {}


_API = _load_ipfs_network_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _get_ipfs_client():
    """Resolve IPFS client with best-effort fallback."""
    try:
        from ipfs_accelerate_py.kit.ipfs_files_kit import get_ipfs_files_kit

        kit = get_ipfs_files_kit()
        return getattr(kit, "_client", None) or getattr(kit, "client", None)
    except Exception:
        return None


async def ipfs_node_inventory() -> Dict[str, Any]:
    """Return inventory metadata for IPFS network tools."""
    return _normalize_payload(
        {
            "category": "ipfs_network_tools",
            "tools": [
                "ipfs_id",
                "ipfs_swarm_peers",
                "ipfs_swarm_connect",
                "ipfs_pubsub_pub",
                "ipfs_dht_findpeer",
                "ipfs_dht_findprovs",
            ],
            "description": "IPFS network operations: node identity, swarm, pubsub, DHT",
            "source": "mcp/tools/ipfs_network.py",
        }
    )


async def ipfs_id() -> Dict[str, Any]:
    """Get information about the local IPFS node identity."""
    delegate = _API.get("ipfs_id")
    if callable(delegate):
        try:
            result = await delegate()
            return _normalize_payload(result)
        except Exception as exc:
            logger.warning("ipfs_id delegate failed: %s", exc)

    client = _get_ipfs_client()
    if client is None:
        return _error_result("IPFS client unavailable")
    try:
        import anyio

        result = await anyio.to_thread.run_sync(client.id)
        if isinstance(result, dict):
            return _normalize_payload(
                {
                    "id": result.get("ID", ""),
                    "addresses": result.get("Addresses", []),
                    "agent_version": result.get("AgentVersion", ""),
                    "protocol_version": result.get("ProtocolVersion", ""),
                    "public_key": result.get("PublicKey", ""),
                }
            )
        return _normalize_payload({"raw": str(result)})
    except Exception as exc:
        return _error_result(str(exc))


async def ipfs_swarm_peers() -> Dict[str, Any]:
    """List peers connected to the IPFS swarm."""
    delegate = _API.get("ipfs_swarm_peers")
    if callable(delegate):
        try:
            result = await delegate()
            return _normalize_payload(result)
        except Exception as exc:
            logger.warning("ipfs_swarm_peers delegate failed: %s", exc)

    client = _get_ipfs_client()
    if client is None:
        return _error_result("IPFS client unavailable")
    try:
        import anyio

        result = await anyio.to_thread.run_sync(client.swarm.peers)
        peers = []
        if isinstance(result, dict):
            peers = [
                {
                    "addr": str(p.get("Addr", "")),
                    "peer": str(p.get("Peer", "")),
                    "latency": str(p.get("Latency", "")),
                    "muxer": str(p.get("Muxer", "")),
                }
                for p in result.get("Peers", [])
            ]
        return _normalize_payload({"peers": peers, "count": len(peers)})
    except Exception as exc:
        return _error_result(str(exc))


async def ipfs_swarm_connect(addr: str) -> Dict[str, Any]:
    """Connect to an IPFS peer by multiaddr."""
    if not isinstance(addr, str) or not addr.strip():
        return _error_result("addr must be a non-empty multiaddr string")

    client = _get_ipfs_client()
    if client is None:
        return _error_result("IPFS client unavailable")
    try:
        import anyio

        result = await anyio.to_thread.run_sync(lambda: client.swarm.connect(addr.strip()))
        return _normalize_payload(
            {
                "addr": addr.strip(),
                "strings": result.get("Strings", []) if isinstance(result, dict) else [],
            }
        )
    except Exception as exc:
        return _error_result(str(exc), addr=addr)


async def ipfs_pubsub_pub(topic: str, message: str) -> Dict[str, Any]:
    """Publish a message to an IPFS pubsub topic."""
    if not isinstance(topic, str) or not topic.strip():
        return _error_result("topic must be a non-empty string")
    if not isinstance(message, str):
        return _error_result("message must be a string")

    client = _get_ipfs_client()
    if client is None:
        return _error_result("IPFS client unavailable")
    try:
        import anyio

        await anyio.to_thread.run_sync(
            lambda: client.pubsub.pub(topic=topic.strip(), payload=message)
        )
        return _normalize_payload(
            {"topic": topic.strip(), "message_length": len(message), "published": True}
        )
    except Exception as exc:
        return _error_result(str(exc), topic=topic)


async def ipfs_dht_findpeer(peer_id: str) -> Dict[str, Any]:
    """Find peer information in the IPFS DHT by peer ID."""
    if not isinstance(peer_id, str) or not peer_id.strip():
        return _error_result("peer_id must be a non-empty string")

    client = _get_ipfs_client()
    if client is None:
        return _error_result("IPFS client unavailable")
    try:
        import anyio

        result = await anyio.to_thread.run_sync(
            lambda: client.dht.findpeer(peer_id=peer_id.strip())
        )
        responses = []
        if isinstance(result, dict):
            responses = result.get("Responses", [])
        return _normalize_payload(
            {"peer_id": peer_id.strip(), "responses": responses, "count": len(responses)}
        )
    except Exception as exc:
        return _error_result(str(exc), peer_id=peer_id)


async def ipfs_dht_findprovs(cid: str, num_providers: int = 20) -> Dict[str, Any]:
    """Find providers for a CID in the IPFS DHT."""
    if not isinstance(cid, str) or not cid.strip():
        return _error_result("cid must be a non-empty string")
    if not isinstance(num_providers, int) or num_providers < 1:
        num_providers = 20

    client = _get_ipfs_client()
    if client is None:
        return _error_result("IPFS client unavailable")
    try:
        import anyio

        result = await anyio.to_thread.run_sync(
            lambda: client.dht.findprovs(cid=cid.strip(), num_providers=num_providers)
        )
        providers = []
        if isinstance(result, dict):
            providers = result.get("Responses", [])
        return _normalize_payload(
            {
                "cid": cid.strip(),
                "providers": providers,
                "count": len(providers),
                "num_providers_requested": num_providers,
            }
        )
    except Exception as exc:
        return _error_result(str(exc), cid=cid)


def register_native_ipfs_network_tools(manager: Any) -> None:
    """Register native IPFS network tools in the unified hierarchical manager."""
    manager.register_tool(
        category="ipfs_network_tools",
        name="ipfs_node_inventory",
        func=ipfs_node_inventory,
        description="Return inventory metadata for IPFS network tools.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "ipfs", "network"],
    )
    manager.register_tool(
        category="ipfs_network_tools",
        name="ipfs_id",
        func=ipfs_id,
        description="Get local IPFS node identity (ID, addresses, versions).",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "ipfs", "network"],
    )
    manager.register_tool(
        category="ipfs_network_tools",
        name="ipfs_swarm_peers",
        func=ipfs_swarm_peers,
        description="List peers connected to the IPFS swarm.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "ipfs", "network"],
    )
    manager.register_tool(
        category="ipfs_network_tools",
        name="ipfs_swarm_connect",
        func=ipfs_swarm_connect,
        description="Connect to an IPFS peer by multiaddr.",
        input_schema={
            "type": "object",
            "properties": {
                "addr": {
                    "type": "string",
                    "description": "Multiaddr of the peer to connect to.",
                }
            },
            "required": ["addr"],
        },
        runtime="fastapi",
        tags=["native", "ipfs", "network"],
    )
    manager.register_tool(
        category="ipfs_network_tools",
        name="ipfs_pubsub_pub",
        func=ipfs_pubsub_pub,
        description="Publish a message to an IPFS pubsub topic.",
        input_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Pubsub topic name."},
                "message": {"type": "string", "description": "Message to publish."},
            },
            "required": ["topic", "message"],
        },
        runtime="fastapi",
        tags=["native", "ipfs", "network"],
    )
    manager.register_tool(
        category="ipfs_network_tools",
        name="ipfs_dht_findpeer",
        func=ipfs_dht_findpeer,
        description="Find peer information in the IPFS DHT by peer ID.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string", "description": "Peer ID to locate in the DHT."}
            },
            "required": ["peer_id"],
        },
        runtime="fastapi",
        tags=["native", "ipfs", "network"],
    )
    manager.register_tool(
        category="ipfs_network_tools",
        name="ipfs_dht_findprovs",
        func=ipfs_dht_findprovs,
        description="Find providers for a CID in the IPFS DHT.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "Content identifier to find providers for."},
                "num_providers": {
                    "type": "integer",
                    "description": "Maximum number of providers to return.",
                    "default": 20,
                },
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "ipfs", "network"],
    )
