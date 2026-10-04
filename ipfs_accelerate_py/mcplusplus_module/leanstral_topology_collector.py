"""Bounded external collector for the Leanstral MCP++ P2P topology.

The collector is intentionally inference-free and never starts a model
server.  It wraps an already-running OpenAI-compatible Leanstral endpoint in
one MCP++ P2P service node, starts an independent client node in a subprocess,
and proves that the client can:

* bootstrap to the service's exact advertised multiaddr;
* dial the service directly;
* register and discover through rendezvous mounted on that same service peer;
* call ``model_list_served`` over ``/mcp+p2p/1.0.0``.

Only the strict, path-free topology receipt from :mod:`leanstral_topology` is
emitted.  A clean Git source tree is required by the public collector entry
point so a dirty development tree cannot accidentally produce operational
evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import trio

from ipfs_accelerate_py.mcplusplus_module.leanstral_topology import (
    CapabilityClaim,
    IndependentDialObservation,
    InterfaceAddress,
    LEANSTRAL_HTTP_TRANSPORT,
    LEANSTRAL_LOGICAL_MODEL_ID,
    LEANSTRAL_P2P_LISTEN_ADDR,
    LEANSTRAL_P2P_PORT,
    LEANSTRAL_P2P_PROTOCOL,
    MAX_TOPOLOGY_PROBE_TIMEOUT_S,
    ProbeObservation,
    canonical_json_cid,
    is_leanstral_transport_model_id,
    select_advertised_ipv4,
    validate_leanstral_topology_mapping,
)
from ipfs_accelerate_py.mcplusplus_module.p2p_transport import (
    DEFAULT_BOOTSTRAP_PEERS,
    MCPp2pNode,
)
from ipfs_accelerate_py.utils.llama_cpp import DEFAULT_LEANSTRAL_MODEL_REF


COLLECTOR_FAILURE_SCHEMA = "hssl.leanstral-p2p-collector-failure/v1"
CLIENT_PROBE_SCHEMA = "hssl.leanstral-p2p-independent-client/v1"
RENDEZVOUS_NAMESPACE = "leanstral-local"
DEFAULT_CLIENT_PROCESS_TIMEOUT_S = 45.0
MAX_CLIENT_RECEIPT_BYTES = 1024 * 1024
MAX_CONFIGURED_INTERFACES = 32
MAX_CONFIGURED_BOOTSTRAP_PEERS = 32

_VIRTUAL_INTERFACE_PREFIXES = (
    "br-",
    "cni",
    "docker",
    "flannel",
    "podman",
    "veth",
    "virbr",
)
_FORBIDDEN_IDENTITY_KEYS = {
    "artifact_path",
    "command",
    "cwd",
    "file_path",
    "identity_file",
    "repo_path",
    "source_path",
    "source_root",
    "stderr",
    "stdout",
    "traceback",
    "worktree_path",
}
_NETWORK_IDENTITY_PREFIXES = (
    "/dns/",
    "/dns4/",
    "/dns6/",
    "/dnsaddr/",
    "/ip4/",
    "/ip6/",
    "/mcp+p2p/",
)
_PROBE_FIELDS = {
    "mechanism",
    "target",
    "attempted",
    "success",
    "timeout_s",
    "duration_ms",
    "error",
    "observer_peer_id",
    "namespace",
    "details",
}
_DIAL_FIELDS = {
    "dialer_peer_id",
    "target_peer_id",
    "target_multiaddr",
    "attempted",
    "success",
    "timeout_s",
    "duration_ms",
    "error",
}
_SAFE_FAILED_PROBE_ERRORS = {"connect_failed", "timeout"}


def _is_path_free_token(value: str, *, max_length: int = 128) -> bool:
    return (
        bool(value)
        and len(value) <= max_length
        and all(character.isalnum() or character in "._:-" for character in value)
    )


def _is_supported_peer_multiaddr(value: str) -> bool:
    rendered = str(value or "").strip()
    if not rendered.startswith(_NETWORK_IDENTITY_PREFIXES[:-1]):
        return False
    if rendered.count("/p2p/") != 1:
        return False
    peer_id = rendered.rsplit("/p2p/", 1)[-1]
    return _is_path_free_token(peer_id, max_length=256)


class TopologyCollectionError(RuntimeError):
    """Stable, path-free failure raised when evidence is incomplete."""

    def __init__(self, code: str):
        normalized = str(code or "").strip()
        if (
            not normalized
            or normalized != normalized.casefold()
            or not normalized.replace("_", "").isalnum()
        ):
            raise ValueError("collector error code must be a lowercase identifier")
        self.code = normalized
        super().__init__(normalized)


@dataclass(frozen=True)
class LeanstralCollectorConfig:
    """Configuration for one bounded, inference-free topology collection."""

    endpoint_url: str = "http://127.0.0.1:8080/v1"
    expected_transport_model_id: str = DEFAULT_LEANSTRAL_MODEL_REF
    allowed_interfaces: Tuple[str, ...] = ()
    bootstrap_peers: Tuple[str, ...] = tuple(DEFAULT_BOOTSTRAP_PEERS)
    rendezvous_namespace: str = RENDEZVOUS_NAMESPACE
    probe_timeout_s: float = MAX_TOPOLOGY_PROBE_TIMEOUT_S
    model_timeout_s: float = 2.0
    client_process_timeout_s: float = DEFAULT_CLIENT_PROCESS_TIMEOUT_S

    def __post_init__(self) -> None:
        endpoint = str(self.endpoint_url or "").strip().rstrip("/")
        if len(endpoint) > 2048:
            raise ValueError("endpoint_url is too long")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "endpoint_url must not contain credentials, query parameters, or fragments"
            )
        try:
            endpoint_port = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint_url contains an invalid port") from exc
        if endpoint_port == LEANSTRAL_P2P_PORT:
            raise ValueError("HTTP endpoint must not use the Leanstral P2P port")
        expected_model = str(self.expected_transport_model_id or "").strip()
        if (
            not expected_model
            or len(expected_model) > 1024
            or expected_model == LEANSTRAL_LOGICAL_MODEL_ID
            or not is_leanstral_transport_model_id(expected_model)
            or expected_model.casefold().startswith("file://")
            or Path(expected_model).is_absolute()
            or PureWindowsPath(expected_model).is_absolute()
            or "../" in expected_model
            or "..\\" in expected_model
        ):
            raise ValueError(
                "expected_transport_model_id must be a path-free Leanstral transport ID"
            )

        interfaces = tuple(
            dict.fromkeys(
                str(interface).strip()
                for interface in self.allowed_interfaces
                if str(interface).strip()
            )
        )
        if not interfaces:
            raise ValueError("at least one advertise interface must be explicitly allowed")
        if len(interfaces) > MAX_CONFIGURED_INTERFACES:
            raise ValueError("too many advertise interfaces were configured")
        if not all(_is_path_free_token(interface, max_length=64) for interface in interfaces):
            raise ValueError("advertise interfaces must be path-free interface names")

        bootstraps = tuple(
            dict.fromkeys(str(peer).strip() for peer in self.bootstrap_peers if str(peer).strip())
        )
        if not bootstraps:
            raise ValueError("at least one bootstrap peer must be configured")
        if len(bootstraps) > MAX_CONFIGURED_BOOTSTRAP_PEERS:
            raise ValueError("too many bootstrap peers were configured")
        if not all(_is_supported_peer_multiaddr(peer) for peer in bootstraps):
            raise ValueError("bootstrap peers must be path-free exact peer multiaddrs")

        namespace = str(self.rendezvous_namespace or "").strip()
        if not _is_path_free_token(namespace):
            raise ValueError("rendezvous_namespace must be a path-free identifier")
        if not (0.0 < float(self.probe_timeout_s) <= MAX_TOPOLOGY_PROBE_TIMEOUT_S):
            raise ValueError("probe_timeout_s must be within (0, 10] seconds")
        if not (0.0 < float(self.model_timeout_s) <= MAX_TOPOLOGY_PROBE_TIMEOUT_S):
            raise ValueError("model_timeout_s must be within (0, 10] seconds")
        if not (float(self.client_process_timeout_s) > (float(self.probe_timeout_s) * 4.0)):
            raise ValueError("client_process_timeout_s must exceed four probe timeout windows")

        object.__setattr__(self, "endpoint_url", endpoint)
        object.__setattr__(self, "expected_transport_model_id", expected_model)
        object.__setattr__(self, "allowed_interfaces", interfaces)
        object.__setattr__(self, "bootstrap_peers", bootstraps)
        object.__setattr__(self, "rendezvous_namespace", namespace)


def _with_cid(payload: Mapping[str, Any], *, field_name: str = "receipt_cid") -> Dict[str, Any]:
    identity = dict(payload)
    if field_name in identity:
        raise ValueError(f"{field_name} must not be present before identity calculation")
    return {**identity, field_name: canonical_json_cid(identity)}


def _assert_path_free_identity(value: Any, *, field_name: str = "receipt") -> None:
    """Reject local-path and diagnostic fields from emitted identity JSON."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold()
            if (
                normalized in _FORBIDDEN_IDENTITY_KEYS
                or normalized.endswith("_filesystem_path")
                or normalized.endswith("_local_path")
            ):
                raise TopologyCollectionError("identity_contains_local_path_field")
            _assert_path_free_identity(child, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_path_free_identity(child, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, str):
        normalized = value.casefold()
        if normalized.startswith("file://"):
            raise TopologyCollectionError("identity_contains_file_uri")
        if any(ord(character) < 32 for character in value):
            raise TopologyCollectionError("identity_contains_control_character")
        if PureWindowsPath(value).is_absolute():
            raise TopologyCollectionError("identity_contains_local_path")
        if Path(value).is_absolute() and not value.startswith(_NETWORK_IDENTITY_PREFIXES):
            raise TopologyCollectionError("identity_contains_local_path")
        if "../" in value or "..\\" in value:
            raise TopologyCollectionError("identity_contains_local_path")


def canonical_identity_json(value: Mapping[str, Any]) -> str:
    """Encode one path-free identity object as deterministic JSON."""

    _assert_path_free_identity(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def collector_failure_receipt(code: str) -> Dict[str, Any]:
    """Build a stable CIDv1 failure receipt without leaking diagnostics."""

    error = TopologyCollectionError(code)
    return _with_cid(
        {
            "schema": COLLECTOR_FAILURE_SCHEMA,
            "status": "failed",
            "error_code": error.code,
            "operational_completion": False,
        }
    )


def _find_source_repo() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise TopologyCollectionError("source_repository_not_found")


def require_clean_source_tree() -> str:
    """Require a clean Git worktree and return its exact source commit."""

    repository = _find_source_repo()
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise TopologyCollectionError("source_state_unverifiable") from exc
    if status.stdout.strip():
        raise TopologyCollectionError("source_tree_dirty")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise TopologyCollectionError("source_commit_invalid")
    return commit


def observe_ipv4_interfaces(
    allowed_interfaces: Sequence[str],
) -> Tuple[InterfaceAddress, ...]:
    """Observe active and rejected IPv4 interfaces without recording host paths."""

    try:
        import psutil

        stats = psutil.net_if_stats()
        addresses = psutil.net_if_addrs()
    except Exception as exc:
        raise TopologyCollectionError("interface_inventory_unavailable") from exc

    allowed = {str(interface).strip() for interface in allowed_interfaces if str(interface).strip()}
    observed = []
    for interface in sorted(addresses):
        normalized = interface.casefold()
        if any(normalized.startswith(prefix) for prefix in _VIRTUAL_INTERFACE_PREFIXES):
            scope = "container"
        elif interface in allowed:
            scope = "lan"
        else:
            scope = "unrelated"
        is_up = bool(interface in stats and stats[interface].isup)
        for entry in addresses[interface]:
            if entry.family != socket.AF_INET:
                continue
            observed.append(
                InterfaceAddress(
                    interface=interface,
                    address=str(entry.address),
                    is_up=is_up,
                    scope=scope,
                )
            )
    if not observed:
        raise TopologyCollectionError("ipv4_interface_inventory_empty")
    return tuple(observed)


def _identity_model_record(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a served model to the path-free identity fields in the contract."""

    if not isinstance(model, Mapping):
        raise TopologyCollectionError("served_model_record_invalid")
    raw_model_id = str(model.get("transport_model_id") or "")
    model_name = str(model.get("name") or "")
    model_endpoint = str(model.get("endpoint") or "").rstrip("/")
    if len(raw_model_id) > 1024 or len(model_name) > 256 or len(model_endpoint) > 2048:
        raise TopologyCollectionError("served_model_identity_too_long")
    if (
        raw_model_id.casefold().startswith("file://")
        or model_name.casefold().startswith("file://")
        or Path(raw_model_id).is_absolute()
        or Path(model_name).is_absolute()
        or PureWindowsPath(raw_model_id).is_absolute()
        or PureWindowsPath(model_name).is_absolute()
    ):
        raise TopologyCollectionError("served_model_identity_contains_local_path")
    capabilities = model.get("capabilities")
    if not isinstance(capabilities, (list, tuple)):
        capabilities = []
    if len(capabilities) > 32 or not all(type(item) is str for item in capabilities):
        raise TopologyCollectionError("served_model_capabilities_invalid")
    if not all(_is_path_free_token(item) for item in capabilities):
        raise TopologyCollectionError("served_model_capabilities_invalid")
    return {
        "id": model.get("id"),
        "model_id": model.get("model_id"),
        "name": model_name,
        "logical_model_id": model.get("logical_model_id"),
        "transport_model_id": raw_model_id,
        "provider": model.get("provider"),
        "transport": model.get("transport"),
        "endpoint": model_endpoint,
        "status": model.get("status"),
        "served": model.get("served"),
        "capabilities": list(capabilities),
        # Arbitrary upstream metadata can contain host paths.  It is not part
        # of the model identity needed by this topology receipt.
        "metadata": {},
    }


def _validated_model_listing(
    models: Iterable[Mapping[str, Any]],
    *,
    endpoint_url: str,
    expected_transport_model_id: str = DEFAULT_LEANSTRAL_MODEL_REF,
) -> Tuple[Dict[str, Any], ...]:
    projected = tuple(_identity_model_record(model) for model in models)
    if len(projected) != 1:
        raise TopologyCollectionError("served_model_record_count_not_one")
    model = projected[0]
    if (
        model["id"] != LEANSTRAL_LOGICAL_MODEL_ID
        or model["model_id"] != LEANSTRAL_LOGICAL_MODEL_ID
        or model["name"] != LEANSTRAL_LOGICAL_MODEL_ID
        or model["logical_model_id"] != LEANSTRAL_LOGICAL_MODEL_ID
    ):
        raise TopologyCollectionError("served_model_logical_identity_mismatch")
    raw_model_id = str(model["transport_model_id"] or "")
    if (
        not raw_model_id
        or raw_model_id == LEANSTRAL_LOGICAL_MODEL_ID
        or not is_leanstral_transport_model_id(raw_model_id)
        or raw_model_id != expected_transport_model_id
    ):
        raise TopologyCollectionError("served_model_transport_identity_mismatch")
    if (
        model["provider"] != LEANSTRAL_HTTP_TRANSPORT
        or model["transport"] != LEANSTRAL_HTTP_TRANSPORT
    ):
        raise TopologyCollectionError("served_model_transport_provider_mismatch")
    if model["endpoint"] != endpoint_url.rstrip("/"):
        raise TopologyCollectionError("served_model_endpoint_mismatch")
    if model["status"] != "available" or model["served"] is not True:
        raise TopologyCollectionError("served_model_not_available")
    return projected


async def _discover_models(endpoint_url: str, timeout_s: float) -> Sequence[Mapping[str, Any]]:
    from ipfs_accelerate_py.model_manager import get_default_model_manager

    manager = get_default_model_manager()
    return await trio.to_thread.run_sync(
        lambda: manager.list_served_models(
            endpoint_url=endpoint_url,
            timeout=timeout_s,
        )
    )


@contextmanager
def _temporary_environment(overrides: Mapping[str, str]):
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _probe_receipt(
    *,
    mechanism: str,
    target: str,
    observer_peer_id: str,
    timeout_s: float,
    attempted: bool,
    success: bool,
    duration_ms: float,
    error: Optional[str],
    namespace: str = "",
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return asdict(
        ProbeObservation(
            mechanism=mechanism,
            target=target,
            attempted=attempted,
            success=success,
            timeout_s=float(timeout_s),
            duration_ms=float(duration_ms),
            error=error,
            observer_peer_id=observer_peer_id,
            namespace=namespace,
            details=dict(details or {}),
        )
    )


async def _bounded_connect(
    node: MCPp2pNode,
    target: str,
    *,
    timeout_s: float,
) -> Tuple[bool, float, Optional[str]]:
    started = time.monotonic()
    success = False
    error = None
    with trio.move_on_after(float(timeout_s)) as cancel_scope:
        success = bool(await node._connect_bootstrap(target))
    if cancel_scope.cancelled_caught:
        error = "timeout"
    elif not success:
        error = "connect_failed"
    return (
        bool(success and not cancel_scope.cancelled_caught),
        (time.monotonic() - started) * 1000.0,
        error,
    )


async def _client_probe(
    *,
    target_multiaddr: str,
    endpoint_url: str,
    expected_transport_model_id: str,
    rendezvous_namespace: str,
    probe_timeout_s: float,
    model_timeout_s: float,
) -> Dict[str, Any]:
    """Run the independent subprocess side of collection."""

    client = MCPp2pNode(
        listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        bootstrap_peers=[],
        advertise_addrs=[],
    )
    result: Optional[Dict[str, Any]] = None
    overrides = {
        "MCPPP_P2P_MDNS": "0",
        "MCPPP_P2P_RENDEZVOUS_AUTO": "0",
        "MCPPP_P2P_RENDEZVOUS_SERVICE": "",
        "IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER": target_multiaddr,
        "IPFS_ACCELERATE_P2P_RENDEZVOUS_NS": rendezvous_namespace,
    }
    with _temporary_environment(overrides):
        async with trio.open_nursery() as nursery:
            try:
                await client.start(nursery)
                status = client.to_dict()
                if not status.get("operational") or not client.peer_id:
                    raise TopologyCollectionError("independent_client_not_operational")

                bootstrap_success, bootstrap_ms, bootstrap_error = await _bounded_connect(
                    client,
                    target_multiaddr,
                    timeout_s=probe_timeout_s,
                )
                bootstrap = _probe_receipt(
                    mechanism="bootstrap",
                    target=target_multiaddr,
                    observer_peer_id=client.peer_id,
                    timeout_s=probe_timeout_s,
                    attempted=True,
                    success=bootstrap_success,
                    duration_ms=bootstrap_ms,
                    error=bootstrap_error,
                )
                if not bootstrap_success:
                    raise TopologyCollectionError("bootstrap_exercise_no_success")

                dial_success, dial_ms, dial_error = await _bounded_connect(
                    client,
                    target_multiaddr,
                    timeout_s=probe_timeout_s,
                )
                target_peer_id = target_multiaddr.rsplit("/p2p/", 1)[-1]
                dial = asdict(
                    IndependentDialObservation(
                        dialer_peer_id=client.peer_id,
                        target_peer_id=target_peer_id,
                        target_multiaddr=target_multiaddr,
                        attempted=True,
                        success=dial_success,
                        timeout_s=float(probe_timeout_s),
                        duration_ms=float(dial_ms),
                        error=dial_error,
                    )
                )
                if not dial_success:
                    raise TopologyCollectionError("independent_dial_not_successful")

                rendezvous = await client.exercise_rendezvous(
                    namespace=rendezvous_namespace,
                    timeout=probe_timeout_s,
                )
                details = rendezvous.get("details")
                if (
                    rendezvous.get("success") is not True
                    or not isinstance(details, Mapping)
                    or details.get("registered") is not True
                    or not isinstance(details.get("discovered_peers"), list)
                    or not details.get("discovered_peers")
                ):
                    raise TopologyCollectionError("rendezvous_exercise_incomplete")

                model_listing = await client.call_tool(
                    target_peer_id,
                    "model_list_served",
                    {},
                    timeout=float(probe_timeout_s),
                    max_retries=1,
                )
                if not isinstance(model_listing, Mapping):
                    raise TopologyCollectionError("mcp_model_listing_invalid")
                models = model_listing.get("models")
                if (
                    model_listing.get("status") != "success"
                    or type(model_listing.get("count")) is not int
                    or not isinstance(models, list)
                    or model_listing.get("count") != len(models)
                ):
                    raise TopologyCollectionError("mcp_model_listing_failed")
                projected_models = _validated_model_listing(
                    models,
                    endpoint_url=endpoint_url,
                    expected_transport_model_id=expected_transport_model_id,
                )

                identity = {
                    "schema": CLIENT_PROBE_SCHEMA,
                    "process_role": "independent_client_subprocess",
                    "client_peer_id": client.peer_id,
                    "target_multiaddr": target_multiaddr,
                    "bootstrap_exercises": [bootstrap],
                    "rendezvous_exercises": [dict(rendezvous)],
                    "independent_dial": dial,
                    "model_listing": {
                        "status": "success",
                        "count": len(projected_models),
                        "models": list(projected_models),
                    },
                    "inference_attempted": False,
                }
                result = _with_cid(identity, field_name="client_receipt_cid")
            finally:
                await client.stop()
                nursery.cancel_scope.cancel()
    if result is None:
        raise TopologyCollectionError("independent_client_result_missing")
    return result


def _validate_client_receipt(
    value: Mapping[str, Any],
    *,
    target_multiaddr: str,
    rendezvous_namespace: str,
) -> Mapping[str, Any]:
    required = {
        "schema",
        "process_role",
        "client_peer_id",
        "target_multiaddr",
        "bootstrap_exercises",
        "rendezvous_exercises",
        "independent_dial",
        "model_listing",
        "inference_attempted",
        "client_receipt_cid",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise TopologyCollectionError("independent_client_receipt_shape_invalid")
    if value.get("schema") != CLIENT_PROBE_SCHEMA:
        raise TopologyCollectionError("independent_client_schema_mismatch")
    if value.get("process_role") != "independent_client_subprocess":
        raise TopologyCollectionError("independent_client_process_role_mismatch")
    if value.get("target_multiaddr") != target_multiaddr:
        raise TopologyCollectionError("independent_client_target_mismatch")
    if value.get("inference_attempted") is not False:
        raise TopologyCollectionError("independent_client_attempted_inference")
    _assert_path_free_identity(value, field_name="independent_client_receipt")
    identity = dict(value)
    supplied_cid = identity.pop("client_receipt_cid")
    if type(supplied_cid) is not str or supplied_cid != canonical_json_cid(identity):
        raise TopologyCollectionError("independent_client_receipt_cid_mismatch")

    client_peer_id = value.get("client_peer_id")
    if type(client_peer_id) is not str or not _is_path_free_token(client_peer_id, max_length=256):
        raise TopologyCollectionError("independent_client_peer_id_missing")
    bootstraps = value.get("bootstrap_exercises")
    if (
        not isinstance(bootstraps, list)
        or len(bootstraps) != 1
        or not all(
            isinstance(item, Mapping)
            and set(item) == _PROBE_FIELDS
            and item.get("mechanism") == "bootstrap"
            and item.get("target") == target_multiaddr
            and item.get("observer_peer_id") == client_peer_id
            and item.get("attempted") is True
            and item.get("success") is True
            and item.get("error") is None
            and item.get("namespace") == ""
            and item.get("details") == {}
            for item in bootstraps
        )
    ):
        raise TopologyCollectionError("bootstrap_exercise_incomplete")
    rendezvous = value.get("rendezvous_exercises")
    if (
        not isinstance(rendezvous, list)
        or len(rendezvous) != 1
        or not all(
            isinstance(item, Mapping)
            and set(item) == _PROBE_FIELDS
            and item.get("mechanism") == "rendezvous"
            and item.get("target") == target_multiaddr
            and item.get("observer_peer_id") == client_peer_id
            and item.get("attempted") is True
            and item.get("success") is True
            and item.get("error") is None
            and item.get("namespace") == rendezvous_namespace
            and isinstance(item.get("details"), Mapping)
            and set(item["details"]) == {"registered", "discovered_peers"}
            and item["details"].get("registered") is True
            and isinstance(item["details"].get("discovered_peers"), list)
            and bool(item["details"]["discovered_peers"])
            and all(
                type(peer) is str and _is_supported_peer_multiaddr(peer)
                for peer in item["details"]["discovered_peers"]
            )
            for item in rendezvous
        )
    ):
        raise TopologyCollectionError("rendezvous_exercise_incomplete")
    dial = value.get("independent_dial")
    expected_peer_id = target_multiaddr.rsplit("/p2p/", 1)[-1]
    if (
        not isinstance(dial, Mapping)
        or set(dial) != _DIAL_FIELDS
        or dial.get("dialer_peer_id") != client_peer_id
        or dial.get("target_peer_id") != expected_peer_id
        or dial.get("target_multiaddr") != target_multiaddr
        or dial.get("attempted") is not True
        or dial.get("success") is not True
        or dial.get("error") is not None
    ):
        raise TopologyCollectionError("independent_dial_incomplete")
    listing = value.get("model_listing")
    if (
        not isinstance(listing, Mapping)
        or set(listing) != {"status", "count", "models"}
        or listing.get("status") != "success"
        or type(listing.get("count")) is not int
        or not isinstance(listing.get("models"), list)
        or listing.get("count") != len(listing["models"])
    ):
        raise TopologyCollectionError("mcp_model_listing_invalid")
    return value


def _validated_server_bootstrap_exercises(
    status: Mapping[str, Any],
    *,
    configured_peers: Sequence[str],
    server_peer_id: str,
) -> Tuple[Dict[str, Any], ...]:
    bootstrap = status.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise TopologyCollectionError("p2p_bootstrap_status_missing")
    configured = bootstrap.get("configured_peers")
    if not isinstance(configured, list) or sorted(configured) != sorted(configured_peers):
        raise TopologyCollectionError("p2p_bootstrap_configuration_mismatch")
    attempts = bootstrap.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise TopologyCollectionError("configured_bootstrap_exercises_missing")

    expected_targets = set(configured_peers)
    observed_targets = set()
    exercises = []
    for attempt in attempts:
        if (
            not isinstance(attempt, Mapping)
            or set(attempt) != _PROBE_FIELDS
            or attempt.get("mechanism") != "bootstrap"
            or attempt.get("target") not in expected_targets
            or attempt.get("observer_peer_id") != server_peer_id
            or attempt.get("attempted") is not True
            or attempt.get("namespace") != ""
            or attempt.get("details") != {}
            or type(attempt.get("success")) is not bool
        ):
            raise TopologyCollectionError("configured_bootstrap_exercise_invalid")
        error = attempt.get("error")
        if (attempt["success"] is True and error is not None) or (
            attempt["success"] is False and error not in _SAFE_FAILED_PROBE_ERRORS
        ):
            raise TopologyCollectionError("configured_bootstrap_error_invalid")
        observed_targets.add(attempt["target"])
        exercises.append(dict(attempt))

    if observed_targets != expected_targets:
        raise TopologyCollectionError("configured_bootstrap_exercises_incomplete")
    if not any(attempt["success"] is True for attempt in exercises):
        raise TopologyCollectionError("configured_bootstrap_exercise_no_success")
    return tuple(
        sorted(
            exercises,
            key=lambda attempt: (
                str(attempt["target"]),
                float(attempt["duration_ms"]),
            ),
        )
    )


def _claims_from_server_status(status: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_capabilities = status.get("capabilities")
    if not isinstance(raw_capabilities, Mapping):
        raise TopologyCollectionError("p2p_capabilities_missing")
    policies = {
        "mcp_stream": "required",
        "bootstrap": "required",
        "rendezvous": "same_as_service_peer",
        "pubsub": "disabled_until_implemented",
        "floodsub": "disabled_until_implemented",
    }
    claims = {}
    for name, policy in policies.items():
        raw = raw_capabilities.get(name)
        if not isinstance(raw, Mapping):
            raise TopologyCollectionError(f"p2p_capability_{name}_missing")
        claim = CapabilityClaim(
            configured=raw.get("configured"),
            implemented=raw.get("implemented"),
            advertised=raw.get("advertised"),
            policy=str(raw.get("policy") or policy),
        )
        if any(
            type(value) is not bool
            for value in (claim.configured, claim.implemented, claim.advertised)
        ):
            raise TopologyCollectionError(f"p2p_capability_{name}_invalid")
        claims[name] = asdict(claim)
    return claims


def assemble_topology_receipt(
    *,
    config: LeanstralCollectorConfig,
    interfaces: Sequence[InterfaceAddress],
    server_status: Mapping[str, Any],
    raw_http_models: Sequence[Mapping[str, Any]],
    client_receipt: Mapping[str, Any],
    source_commit: str,
) -> Dict[str, Any]:
    """Join independently gathered observations and run the strict validator."""

    selection = select_advertised_ipv4(
        interfaces,
        allowed_interfaces=config.allowed_interfaces,
    )
    if not selection.selected:
        raise TopologyCollectionError("policy_selected_advertised_addresses_empty")

    if server_status.get("operational") is not True:
        raise TopologyCollectionError("p2p_service_not_operational")
    if server_status.get("protocol") != LEANSTRAL_P2P_PROTOCOL:
        raise TopologyCollectionError("p2p_protocol_mismatch")
    if server_status.get("listen_addrs") != [LEANSTRAL_P2P_LISTEN_ADDR]:
        raise TopologyCollectionError("p2p_listen_address_mismatch")
    if server_status.get("listen_port") != LEANSTRAL_P2P_PORT:
        raise TopologyCollectionError("p2p_listen_port_mismatch")
    peer_id = server_status.get("peer_id")
    if type(peer_id) is not str or not _is_path_free_token(peer_id, max_length=256):
        raise TopologyCollectionError("p2p_service_peer_id_missing")

    expected_multiaddrs = sorted(
        f"/ip4/{address}/tcp/{LEANSTRAL_P2P_PORT}/p2p/{peer_id}" for address in selection.selected
    )
    advertised_multiaddrs = server_status.get("multiaddrs")
    if (
        not isinstance(advertised_multiaddrs, list)
        or sorted(advertised_multiaddrs) != expected_multiaddrs
    ):
        raise TopologyCollectionError("p2p_advertised_multiaddrs_mismatch")

    configured_bootstrap_exercises = _validated_server_bootstrap_exercises(
        server_status,
        configured_peers=config.bootstrap_peers,
        server_peer_id=peer_id,
    )

    rendezvous = server_status.get("rendezvous")
    service = rendezvous.get("service") if isinstance(rendezvous, Mapping) else None
    if (
        not isinstance(service, Mapping)
        or service.get("mode") != "same_as_service_peer"
        or service.get("configured") is not True
        or service.get("implemented") is not True
        or service.get("peer_id") != peer_id
    ):
        raise TopologyCollectionError("same_service_rendezvous_not_mounted")
    if rendezvous.get("namespace") != config.rendezvous_namespace:
        raise TopologyCollectionError("rendezvous_namespace_mismatch")

    _validate_client_receipt(
        client_receipt,
        target_multiaddr=str(client_receipt.get("target_multiaddr") or ""),
        rendezvous_namespace=config.rendezvous_namespace,
    )
    target_multiaddr = client_receipt["target_multiaddr"]
    if target_multiaddr not in advertised_multiaddrs:
        raise TopologyCollectionError("independent_client_target_not_advertised")
    if client_receipt.get("client_peer_id") == peer_id:
        raise TopologyCollectionError("independent_client_peer_not_distinct")

    direct_models = _validated_model_listing(
        raw_http_models,
        endpoint_url=config.endpoint_url,
        expected_transport_model_id=config.expected_transport_model_id,
    )
    model_listing = client_receipt.get("model_listing")
    if not isinstance(model_listing, Mapping):
        raise TopologyCollectionError("mcp_model_listing_invalid")
    p2p_models_value = model_listing.get("models")
    if not isinstance(p2p_models_value, list):
        raise TopologyCollectionError("mcp_model_listing_invalid")
    p2p_models = _validated_model_listing(
        p2p_models_value,
        endpoint_url=config.endpoint_url,
        expected_transport_model_id=config.expected_transport_model_id,
    )
    if p2p_models != direct_models:
        raise TopologyCollectionError("mcp_listing_differs_from_http_transport")

    parsed_endpoint = urlparse(config.endpoint_url)
    http_port = parsed_endpoint.port or (443 if parsed_endpoint.scheme == "https" else 80)
    observation = {
        "p2p_requested": True,
        "p2p_enabled": True,
        "listen_addrs": [LEANSTRAL_P2P_LISTEN_ADDR],
        "peer_id": peer_id,
        "advertised_multiaddrs": advertised_multiaddrs,
        "interfaces": [asdict(interface) for interface in interfaces],
        "advertise_interface_allowlist": list(config.allowed_interfaces),
        "bootstrap_exercises": [
            *configured_bootstrap_exercises,
            *client_receipt["bootstrap_exercises"],
        ],
        "rendezvous_exercises": client_receipt["rendezvous_exercises"],
        "capabilities": _claims_from_server_status(server_status),
        "independent_dial": client_receipt["independent_dial"],
        "served_models": list(p2p_models),
        "server_instance_count": 1,
        "inference_attempted": False,
        "http_port": http_port,
        "notes": [
            "existing_http_model_server_reused",
            "independent_client_subprocess",
            f"independent_client_receipt:{client_receipt['client_receipt_cid']}",
            f"configured_bootstrap_policy:{canonical_json_cid(list(config.bootstrap_peers))}",
            f"source_commit:{source_commit}",
        ],
    }
    try:
        validation = validate_leanstral_topology_mapping(observation)
    except (TypeError, ValueError) as exc:
        raise TopologyCollectionError("topology_observation_decode_failed") from exc
    if not validation.valid:
        raise TopologyCollectionError("topology_validation_failed")
    _assert_path_free_identity(validation.receipt)
    return dict(validation.receipt)


ClientRunner = Callable[..., Awaitable[Mapping[str, Any]]]
InterfaceObserver = Callable[[Sequence[str]], Sequence[InterfaceAddress]]
ModelProvider = Callable[[str, float], Awaitable[Sequence[Mapping[str, Any]]]]


async def _run_independent_client_subprocess(
    *,
    target_multiaddr: str,
    endpoint_url: str,
    expected_transport_model_id: str,
    rendezvous_namespace: str,
    probe_timeout_s: float,
    model_timeout_s: float,
    process_timeout_s: float,
) -> Mapping[str, Any]:
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.mcplusplus_module.leanstral_topology_collector",
        "_client",
        "--target-multiaddr",
        target_multiaddr,
        "--endpoint-url",
        endpoint_url,
        "--expected-transport-model-id",
        expected_transport_model_id,
        "--rendezvous-namespace",
        rendezvous_namespace,
        "--probe-timeout-s",
        str(probe_timeout_s),
        "--model-timeout-s",
        str(model_timeout_s),
    ]
    try:
        with trio.fail_after(float(process_timeout_s)):
            completed = await trio.run_process(
                command,
                capture_stdout=True,
                stderr=subprocess.DEVNULL,
                check=False,
                cwd=str(_find_source_repo()),
            )
    except trio.TooSlowError as exc:
        raise TopologyCollectionError("independent_client_process_timeout") from exc
    except OSError as exc:
        raise TopologyCollectionError("independent_client_process_start_failed") from exc
    if completed.returncode != 0:
        raise TopologyCollectionError("independent_client_process_failed")
    if len(completed.stdout) > MAX_CLIENT_RECEIPT_BYTES:
        raise TopologyCollectionError("independent_client_output_too_large")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TopologyCollectionError("independent_client_output_invalid") from exc
    if not isinstance(payload, Mapping):
        raise TopologyCollectionError("independent_client_output_invalid")
    return payload


async def _wait_for_configured_bootstrap_attempts(
    node: MCPp2pNode,
    configured_peers: Sequence[str],
    *,
    timeout_s: float,
) -> Mapping[str, Any]:
    """Wait boundedly for the node's startup bootstrap probes to be recorded."""

    expected = set(configured_peers)
    status = node.to_dict()
    with trio.move_on_after(float(timeout_s) + 0.25):
        while True:
            bootstrap = status.get("bootstrap")
            attempts = bootstrap.get("attempts") if isinstance(bootstrap, Mapping) else None
            if isinstance(attempts, list):
                attempted_targets = {
                    attempt.get("target")
                    for attempt in attempts
                    if isinstance(attempt, Mapping) and attempt.get("attempted") is True
                }
                if attempted_targets == expected:
                    return status
            await trio.sleep(0.05)
            status = node.to_dict()
    return status


async def _collect_leanstral_topology_impl(
    config: LeanstralCollectorConfig,
    *,
    _require_clean_source: bool = True,
    _source_commit: str = "",
    _node_factory: Callable[..., MCPp2pNode] = MCPp2pNode,
    _client_runner: Optional[ClientRunner] = None,
    _interface_observer: InterfaceObserver = observe_ipv4_interfaces,
    _model_provider: ModelProvider = _discover_models,
) -> Dict[str, Any]:
    """Collect and validate one fresh Leanstral P2P topology receipt."""

    source_commit = (
        require_clean_source_tree() if _require_clean_source else str(_source_commit or ("0" * 40))
    )
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise TopologyCollectionError("source_commit_invalid")

    interfaces = tuple(_interface_observer(config.allowed_interfaces))
    selection = select_advertised_ipv4(
        interfaces,
        allowed_interfaces=config.allowed_interfaces,
    )
    if not selection.selected:
        raise TopologyCollectionError("policy_selected_advertised_addresses_empty")
    advertise_addrs = [f"/ip4/{address}/tcp/{LEANSTRAL_P2P_PORT}" for address in selection.selected]

    try:
        raw_http_models = await _model_provider(
            config.endpoint_url,
            config.model_timeout_s,
        )
    except TopologyCollectionError:
        raise
    except Exception as exc:
        raise TopologyCollectionError("http_model_discovery_failed") from exc
    _validated_model_listing(
        raw_http_models,
        endpoint_url=config.endpoint_url,
        expected_transport_model_id=config.expected_transport_model_id,
    )

    runner = _client_runner
    if runner is None:

        async def runner(**kwargs):
            return await _run_independent_client_subprocess(
                **kwargs,
                process_timeout_s=config.client_process_timeout_s,
            )

    async def model_tool_handler(method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        call_params = dict(params or {})
        call_params.pop("_sender_peer_id", None)
        if method != "model_list_served" or call_params:
            raise ValueError("collector exposes only model_list_served without arguments")
        from ipfs_accelerate_py.mcp_server.tools.model_tools.native_model_tools import (
            model_list_served,
        )

        return await model_list_served(
            endpoint_url=config.endpoint_url,
            timeout=config.model_timeout_s,
        )

    overrides = {
        "MCPPP_P2P_ADVERTISE_INTERFACES": ",".join(config.allowed_interfaces),
        "MCPPP_P2P_BOOTSTRAP_PEERS": ",".join(config.bootstrap_peers),
        "MCPPP_P2P_LISTEN_ADDRS": LEANSTRAL_P2P_LISTEN_ADDR,
        "MCPPP_P2P_MDNS": "0",
        "MCPPP_P2P_RENDEZVOUS_AUTO": "0",
        "MCPPP_P2P_RENDEZVOUS_SERVICE": "same_as_service_peer",
        "IPFS_ACCELERATE_P2P_RENDEZVOUS_NS": config.rendezvous_namespace,
        "IPFS_ACCELERATE_SERVED_MODEL_ENDPOINTS": config.endpoint_url,
    }
    receipt: Optional[Dict[str, Any]] = None
    with _temporary_environment(overrides):
        node = _node_factory(
            listen_addrs=[LEANSTRAL_P2P_LISTEN_ADDR],
            bootstrap_peers=list(config.bootstrap_peers),
            advertise_addrs=advertise_addrs,
        )
        async with trio.open_nursery() as nursery:
            try:
                node.set_tool_handler(model_tool_handler)
                with trio.fail_after(config.probe_timeout_s):
                    await node.start(nursery)
                server_status = node.to_dict()
                if server_status.get("operational") is not True:
                    raise TopologyCollectionError("p2p_service_not_operational")
                advertised = server_status.get("multiaddrs")
                if not isinstance(advertised, list) or not advertised:
                    raise TopologyCollectionError("p2p_advertised_multiaddrs_missing")
                target = sorted(advertised)[0]
                client_receipt = await runner(
                    target_multiaddr=target,
                    endpoint_url=config.endpoint_url,
                    expected_transport_model_id=config.expected_transport_model_id,
                    rendezvous_namespace=config.rendezvous_namespace,
                    probe_timeout_s=config.probe_timeout_s,
                    model_timeout_s=config.model_timeout_s,
                )
                server_status = await _wait_for_configured_bootstrap_attempts(
                    node,
                    config.bootstrap_peers,
                    timeout_s=config.probe_timeout_s,
                )
                receipt = assemble_topology_receipt(
                    config=config,
                    interfaces=interfaces,
                    server_status=server_status,
                    raw_http_models=raw_http_models,
                    client_receipt=client_receipt,
                    source_commit=source_commit,
                )
            finally:
                await node.stop()
                nursery.cancel_scope.cancel()
    if receipt is None:
        raise TopologyCollectionError("topology_receipt_missing")
    if _require_clean_source and require_clean_source_tree() != source_commit:
        raise TopologyCollectionError("source_commit_changed_during_collection")
    return receipt


def _topology_error_codes(error: BaseException) -> set[str]:
    if isinstance(error, TopologyCollectionError):
        return {error.code}
    if isinstance(error, BaseExceptionGroup):
        return {code for child in error.exceptions for code in _topology_error_codes(child)}
    return set()


async def collect_leanstral_topology(
    config: LeanstralCollectorConfig,
    *,
    _require_clean_source: bool = True,
    _source_commit: str = "",
    _node_factory: Callable[..., MCPp2pNode] = MCPp2pNode,
    _client_runner: Optional[ClientRunner] = None,
    _interface_observer: InterfaceObserver = observe_ipv4_interfaces,
    _model_provider: ModelProvider = _discover_models,
) -> Dict[str, Any]:
    """Collect one receipt and preserve stable errors across Trio nurseries."""

    try:
        return await _collect_leanstral_topology_impl(
            config,
            _require_clean_source=_require_clean_source,
            _source_commit=_source_commit,
            _node_factory=_node_factory,
            _client_runner=_client_runner,
            _interface_observer=_interface_observer,
            _model_provider=_model_provider,
        )
    except BaseExceptionGroup as exc:
        codes = _topology_error_codes(exc)
        if len(codes) == 1:
            raise TopologyCollectionError(codes.pop()) from None
        raise


def _csv_or_repeated(values: Sequence[str], env_name: str) -> Tuple[str, ...]:
    selected = []
    for value in values:
        selected.extend(item.strip() for item in str(value).split(",") if item.strip())
    if not selected:
        selected.extend(
            item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()
        )
    return tuple(dict.fromkeys(selected))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a bounded Leanstral MCP++ P2P topology receipt.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get(
            "IPFS_ACCELERATE_SERVED_MODEL_ENDPOINTS",
            "http://127.0.0.1:8080/v1",
        ).split(",", 1)[0],
    )
    parser.add_argument(
        "--interface",
        action="append",
        default=[],
        help="Allowed active LAN interface; repeat or use comma-separated values.",
    )
    parser.add_argument(
        "--expected-transport-model-id",
        default=DEFAULT_LEANSTRAL_MODEL_REF,
        help="Exact Leanstral transport model identity required from /v1/models.",
    )
    parser.add_argument(
        "--bootstrap-peer",
        action="append",
        default=[],
        help="Exact bootstrap peer multiaddr; repeat or use comma-separated values.",
    )
    parser.add_argument(
        "--rendezvous-namespace",
        default=os.environ.get(
            "IPFS_ACCELERATE_P2P_RENDEZVOUS_NS",
            RENDEZVOUS_NAMESPACE,
        ),
    )
    parser.add_argument("--probe-timeout-s", type=float, default=10.0)
    parser.add_argument("--model-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--client-process-timeout-s",
        type=float,
        default=DEFAULT_CLIENT_PROCESS_TIMEOUT_S,
    )
    return parser


def _build_client_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target-multiaddr", required=True)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--expected-transport-model-id", required=True)
    parser.add_argument("--rendezvous-namespace", required=True)
    parser.add_argument("--probe-timeout-s", required=True, type=float)
    parser.add_argument("--model-timeout-s", required=True, type=float)
    return parser


@contextmanager
def _protocol_stdout_logging_boundary():
    """Keep dependency logs off the collector's JSON protocol channel.

    Some optional P2P dependencies call :func:`logging.basicConfig` lazily
    when their transports are imported.  ``basicConfig`` installs its own
    handler only when the root logger has no handlers, so provide a temporary
    stderr handler before any transport startup.  Restore the caller's exact
    root handlers and level afterward because tests and embedding applications
    may invoke :func:`main` in-process.
    """

    root_logger = logging.getLogger()
    original_handlers = tuple(root_logger.handlers)
    original_level = root_logger.level
    protocol_handler = logging.StreamHandler(sys.stderr)
    protocol_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    )
    for handler in original_handlers:
        root_logger.removeHandler(handler)
    root_logger.addHandler(protocol_handler)
    root_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        for handler in tuple(root_logger.handlers):
            root_logger.removeHandler(handler)
        protocol_handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    client_mode = bool(arguments and arguments[0] == "_client")
    parser = _build_client_parser() if client_mode else _build_parser()
    args = parser.parse_args(arguments[1:] if client_mode else arguments)
    try:
        if client_mode:
            payload = trio.run(
                partial(
                    _client_probe,
                    target_multiaddr=args.target_multiaddr,
                    endpoint_url=args.endpoint_url,
                    expected_transport_model_id=args.expected_transport_model_id,
                    rendezvous_namespace=args.rendezvous_namespace,
                    probe_timeout_s=args.probe_timeout_s,
                    model_timeout_s=args.model_timeout_s,
                )
            )
        else:
            interfaces = _csv_or_repeated(
                args.interface,
                "MCPPP_P2P_ADVERTISE_INTERFACES",
            )
            bootstraps = _csv_or_repeated(
                args.bootstrap_peer,
                "MCPPP_P2P_BOOTSTRAP_PEERS",
            ) or tuple(DEFAULT_BOOTSTRAP_PEERS)
            config = LeanstralCollectorConfig(
                endpoint_url=args.endpoint_url,
                expected_transport_model_id=args.expected_transport_model_id,
                allowed_interfaces=interfaces,
                bootstrap_peers=bootstraps,
                rendezvous_namespace=args.rendezvous_namespace,
                probe_timeout_s=args.probe_timeout_s,
                model_timeout_s=args.model_timeout_s,
                client_process_timeout_s=args.client_process_timeout_s,
            )
            payload = trio.run(collect_leanstral_topology, config)
        print(canonical_identity_json(payload))
        return 0
    except TopologyCollectionError as exc:
        code = exc.code
        print(canonical_identity_json(collector_failure_receipt(code)))
        return 1
    except ValueError:
        print(canonical_identity_json(collector_failure_receipt("collector_configuration_invalid")))
        return 1
    except Exception:
        code = "collector_runtime_failed"
        print(canonical_identity_json(collector_failure_receipt(code)))
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    with _protocol_stdout_logging_boundary():
        return _main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())


__all__ = [
    "CLIENT_PROBE_SCHEMA",
    "COLLECTOR_FAILURE_SCHEMA",
    "LeanstralCollectorConfig",
    "TopologyCollectionError",
    "assemble_topology_receipt",
    "canonical_identity_json",
    "collect_leanstral_topology",
    "collector_failure_receipt",
    "main",
    "observe_ipv4_interfaces",
    "require_clean_source_tree",
]
