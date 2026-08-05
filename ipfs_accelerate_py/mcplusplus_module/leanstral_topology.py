"""Source-side contract for serving Leanstral over MCP++ P2P.

This module is deliberately observation-only.  It does not start a server,
dial a peer, or run inference.  Instead, an operator or an external evidence
runner supplies observations made by independent processes and this module
checks that they describe the intended topology.

The contract keeps three identities separate:

* ``leanstral_local`` is the logical model ID exposed to MCP clients.
* ``llamacpp`` is the OpenAI-compatible HTTP transport serving the model.
* ``/mcp+p2p/1.0.0`` is the libp2p protocol used to reach the MCP server.

Keeping those identities separate prevents an HTTP port (historically 8000)
or an upstream llama.cpp model filename from being mistaken for the P2P
listener or the MCP model name.
"""

from __future__ import annotations

import ipaddress
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LEANSTRAL_LOGICAL_MODEL_ID = "leanstral_local"
LEANSTRAL_HTTP_TRANSPORT = "llamacpp"
LEANSTRAL_P2P_PORT = 19001
LEANSTRAL_P2P_LISTEN_ADDR = f"/ip4/0.0.0.0/tcp/{LEANSTRAL_P2P_PORT}"
LEANSTRAL_P2P_PROTOCOL = "/mcp+p2p/1.0.0"
LEANSTRAL_TOPOLOGY_SCHEMA = "hssl.leanstral-p2p-topology/v1"
MAX_TOPOLOGY_PROBE_TIMEOUT_S = 10.0

# These interfaces are host-local overlays, not addresses an independent
# client should be told to dial.  Explicit interface scope observations below
# remain authoritative because container interfaces are sometimes named eth0.
_CONTAINER_INTERFACE_PREFIXES = (
    "br-",
    "cni",
    "docker",
    "flannel",
    "podman",
    "veth",
    "virbr",
)


@dataclass(frozen=True)
class InterfaceAddress:
    """One interface/address observation used by advertisement policy.

    ``scope`` is asserted by the evidence collector and must be one of
    ``"lan"``, ``"container"``, or ``"unrelated"``.  Requiring both an
    allow-list and an explicit scope avoids treating every RFC1918 address as
    a usable LAN address.
    """

    interface: str
    address: str
    is_up: bool = True
    scope: str = "lan"


@dataclass(frozen=True)
class AddressSelection:
    """Result of applying the advertisement policy."""

    selected: Tuple[str, ...]
    rejected: Mapping[str, str]


@dataclass(frozen=True)
class CapabilityClaim:
    """Configured, implemented, and advertised state for one capability."""

    configured: bool
    implemented: bool
    advertised: bool
    policy: str = ""


@dataclass(frozen=True)
class ProbeObservation:
    """Receipt for one bounded bootstrap or rendezvous exercise."""

    mechanism: str
    target: str
    attempted: bool
    success: bool
    timeout_s: float
    duration_ms: float
    error: Optional[str] = None
    observer_peer_id: str = ""
    namespace: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndependentDialObservation:
    """Receipt produced by a process other than the serving process."""

    dialer_peer_id: str
    target_peer_id: str
    target_multiaddr: str
    attempted: bool
    success: bool
    timeout_s: float
    duration_ms: float
    error: Optional[str] = None


@dataclass(frozen=True)
class LeanstralTopologyObservation:
    """Complete, inference-free observation of the Leanstral service."""

    p2p_requested: bool
    p2p_enabled: bool
    listen_addrs: Tuple[str, ...]
    peer_id: str
    advertised_multiaddrs: Tuple[str, ...]
    interfaces: Tuple[InterfaceAddress, ...]
    advertise_interface_allowlist: Tuple[str, ...]
    bootstrap_exercises: Tuple[ProbeObservation, ...]
    rendezvous_exercises: Tuple[ProbeObservation, ...]
    capabilities: Mapping[str, CapabilityClaim]
    independent_dial: IndependentDialObservation
    served_models: Tuple[Mapping[str, Any], ...]
    server_instance_count: int
    inference_attempted: bool
    http_port: Optional[int] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TopologyValidation:
    """Validation result plus a CIDv1 identity for the complete receipt."""

    valid: bool
    errors: Tuple[str, ...]
    receipt_cid: str
    receipt: Mapping[str, Any]


def canonical_json_cid(value: Any) -> str:
    """Return a real CIDv1/raw/sha2-256 identity for canonical JSON bytes.

    The raw codec is intentional: the hashed block is exactly the UTF-8 JSON
    byte sequence below.  There is no SHA-256 text fallback; callers fail
    closed if the multiformats implementation is unavailable.
    """

    try:
        from multiformats import CID, multihash
    except ImportError as exc:  # pragma: no cover - dependency is pinned in CI
        raise RuntimeError(
            "multiformats is required for Leanstral topology receipt identity"
        ) from exc
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest = multihash.digest(payload, "sha2-256")
        return str(CID("base32", 1, "raw", digest))
    except (TypeError, ValueError) as exc:
        raise ValueError("Leanstral topology receipt is not canonical-JSON encodable") from exc


def _interface_is_container(interface: str) -> bool:
    normalized = str(interface or "").strip().casefold()
    return any(normalized.startswith(prefix) for prefix in _CONTAINER_INTERFACE_PREFIXES)


def select_advertised_ipv4(
    interfaces: Iterable[InterfaceAddress],
    *,
    allowed_interfaces: Sequence[str],
) -> AddressSelection:
    """Select active LAN IPv4 addresses from an explicit interface allow-list.

    Loopback, unspecified, link-local, multicast, stale/down, container, and
    unrelated addresses are rejected.  The result is deterministic and
    suitable for constructing advertised multiaddrs.
    """

    allowed = {str(name).strip() for name in allowed_interfaces if str(name).strip()}
    selected: set[str] = set()
    rejected: Dict[str, str] = {}

    for observed in interfaces:
        key = f"{observed.interface}:{observed.address}"
        if observed.is_up is not True:
            rejected[key] = "interface_down_or_stale"
            continue
        if observed.interface not in allowed:
            rejected[key] = "interface_not_allowed"
            continue
        if observed.scope == "container" or _interface_is_container(observed.interface):
            rejected[key] = "container_interface"
            continue
        if observed.scope != "lan":
            rejected[key] = "interface_not_lan"
            continue
        try:
            address = ipaddress.ip_address(observed.address)
        except ValueError:
            rejected[key] = "invalid_ip_address"
            continue
        if address.version != 4:
            rejected[key] = "not_ipv4"
            continue
        if address.is_loopback:
            rejected[key] = "loopback_address"
            continue
        if address.is_unspecified:
            rejected[key] = "unspecified_address"
            continue
        if address.is_link_local:
            rejected[key] = "link_local_address"
            continue
        if address.is_multicast:
            rejected[key] = "multicast_address"
            continue
        selected.add(str(address))

    return AddressSelection(tuple(sorted(selected)), dict(sorted(rejected.items())))


def leanstral_advertised_multiaddrs(
    addresses: Iterable[str],
    *,
    peer_id: str,
) -> Tuple[str, ...]:
    """Render the canonical port and exact peer ID for selected IPv4 addresses."""

    return tuple(
        f"/ip4/{address}/tcp/{LEANSTRAL_P2P_PORT}/p2p/{peer_id}"
        for address in sorted({str(value) for value in addresses})
    )


def is_leanstral_transport_model_id(model_id: str) -> bool:
    """Return whether an explicit transport model ID names Leanstral."""

    normalized = str(model_id or "").strip().casefold().replace("_", "-")
    return "leanstral" in normalized


def normalize_served_model_record(
    *,
    transport_model_id: str,
    endpoint: str,
    owned_by: str = "",
    name: str = "",
    capabilities: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a model-manager record without conflating logical and transport IDs.

    Leanstral is normalized only when the upstream transport explicitly names
    it.  Other model records retain their upstream identity.
    """

    raw_id = str(transport_model_id or "").strip()
    if not raw_id:
        raise ValueError("transport_model_id must be non-empty")

    is_leanstral = is_leanstral_transport_model_id(raw_id)
    logical_id = LEANSTRAL_LOGICAL_MODEL_ID if is_leanstral else raw_id
    transport = LEANSTRAL_HTTP_TRANSPORT if is_leanstral else str(owned_by or "llama_cpp").strip()
    return {
        "id": logical_id,
        "model_id": logical_id,
        "name": logical_id if is_leanstral else str(name or raw_id),
        "logical_model_id": logical_id,
        "transport_model_id": raw_id,
        "provider": transport,
        "transport": transport,
        "endpoint": str(endpoint).rstrip("/"),
        "status": "available",
        "served": True,
        "capabilities": list(capabilities or ["text-generation"]),
        "metadata": dict(metadata or {}),
    }


def default_capability_claims(
    *,
    rendezvous_implemented: bool = False,
) -> Dict[str, CapabilityClaim]:
    """Return truthful defaults for the current MCP++ P2P implementation."""

    return {
        "mcp_stream": CapabilityClaim(True, True, True, "required"),
        "bootstrap": CapabilityClaim(True, True, True, "required"),
        "rendezvous": CapabilityClaim(
            True,
            bool(rendezvous_implemented),
            bool(rendezvous_implemented),
            "best_effort_runtime_probe",
        ),
        # No pubsub router is wired into MCPp2pNode.  Floodsub is therefore
        # policy-disabled too; neither is advertised merely because libp2p is
        # installed.
        "pubsub": CapabilityClaim(False, False, False, "disabled_until_implemented"),
        "floodsub": CapabilityClaim(False, False, False, "disabled_until_implemented"),
    }


def _parse_ip4_tcp_peer(multiaddr: str) -> Optional[Tuple[str, int, str]]:
    parts = [part for part in str(multiaddr or "").split("/") if part]
    try:
        ip_index = parts.index("ip4")
        tcp_index = parts.index("tcp")
        peer_index = parts.index("p2p")
        address = parts[ip_index + 1]
        port = int(parts[tcp_index + 1])
        peer_id = parts[peer_index + 1]
    except (IndexError, TypeError, ValueError):
        return None
    return address, port, peer_id


def _validate_probe_group(
    probes: Sequence[ProbeObservation],
    *,
    mechanism: str,
    errors: List[str],
) -> None:
    if not probes:
        errors.append(f"{mechanism}_exercise_missing")
        return
    if not any(probe.attempted for probe in probes):
        errors.append(f"{mechanism}_exercise_not_attempted")
    if not any(probe.attempted and probe.success for probe in probes):
        errors.append(f"{mechanism}_exercise_no_success")
    for index, probe in enumerate(probes):
        prefix = f"{mechanism}_exercise_{index}"
        if probe.mechanism != mechanism:
            errors.append(f"{prefix}_mechanism_mismatch")
        if not str(probe.target or "").strip():
            errors.append(f"{prefix}_target_missing")
        if not (0.0 < float(probe.timeout_s) <= MAX_TOPOLOGY_PROBE_TIMEOUT_S):
            errors.append(f"{prefix}_timeout_unbounded")
        if float(probe.duration_ms) < 0:
            errors.append(f"{prefix}_duration_invalid")
        if probe.attempted and (
            float(probe.duration_ms) > (float(probe.timeout_s) * 1000.0 + 250.0)
        ):
            errors.append(f"{prefix}_duration_exceeded_bound")
        if probe.success and not probe.attempted:
            errors.append(f"{prefix}_success_without_attempt")


def validate_leanstral_topology(
    observation: LeanstralTopologyObservation,
) -> TopologyValidation:
    """Validate an external, non-inference topology observation.

    A valid result is source-side evidence that the supplied observation
    satisfies the contract.  It is not operational proof by itself: the
    observation must still come from a fresh external runner and include an
    independent successful dial.
    """

    errors: List[str] = []

    if observation.p2p_requested is not True:
        errors.append("p2p_not_requested")
    if observation.p2p_enabled is not True:
        errors.append("p2p_not_enabled")
    if tuple(observation.listen_addrs) != (LEANSTRAL_P2P_LISTEN_ADDR,):
        errors.append("listen_addr_not_exact_wildcard_19001")
    if not str(observation.peer_id or "").strip():
        errors.append("peer_id_missing")

    selection = select_advertised_ipv4(
        observation.interfaces,
        allowed_interfaces=observation.advertise_interface_allowlist,
    )
    if not observation.advertise_interface_allowlist:
        errors.append("advertise_interface_allowlist_missing")
    if not selection.selected:
        errors.append("policy_selected_advertised_addresses_empty")

    expected_multiaddrs = leanstral_advertised_multiaddrs(
        selection.selected,
        peer_id=observation.peer_id,
    )
    if tuple(sorted(observation.advertised_multiaddrs)) != tuple(sorted(expected_multiaddrs)):
        errors.append("advertised_multiaddrs_do_not_match_policy_selection")

    for index, multiaddr in enumerate(observation.advertised_multiaddrs):
        parsed = _parse_ip4_tcp_peer(multiaddr)
        if parsed is None:
            errors.append(f"advertised_multiaddr_{index}_invalid")
            continue
        address, port, peer_id = parsed
        if address in {"0.0.0.0", "127.0.0.1"}:
            errors.append(f"advertised_multiaddr_{index}_not_dialable")
        if port != LEANSTRAL_P2P_PORT:
            errors.append(f"advertised_multiaddr_{index}_wrong_port")
        if port == 8000:
            errors.append(f"advertised_multiaddr_{index}_uses_http_default_port")
        if peer_id != observation.peer_id:
            errors.append(f"advertised_multiaddr_{index}_wrong_peer_id")

    _validate_probe_group(
        observation.bootstrap_exercises,
        mechanism="bootstrap",
        errors=errors,
    )
    _validate_probe_group(
        observation.rendezvous_exercises,
        mechanism="rendezvous",
        errors=errors,
    )
    for index, probe in enumerate(observation.rendezvous_exercises):
        if not probe.observer_peer_id:
            errors.append(f"rendezvous_exercise_{index}_observer_peer_id_missing")
        elif probe.observer_peer_id == observation.peer_id:
            errors.append(f"rendezvous_exercise_{index}_not_independent")
        target_peer_id = (
            probe.target.rsplit("/p2p/", 1)[-1] if "/p2p/" in probe.target else probe.target
        )
        if target_peer_id != observation.peer_id:
            errors.append(f"rendezvous_exercise_{index}_wrong_service_peer")
        if not probe.namespace:
            errors.append(f"rendezvous_exercise_{index}_namespace_missing")

    required_claims = {
        "mcp_stream": (True, True, True),
        "bootstrap": (True, True, True),
        "rendezvous": (True, True, True),
        "pubsub": (False, False, False),
        "floodsub": (False, False, False),
    }
    for name, (configured, implemented, advertised) in required_claims.items():
        claim = observation.capabilities.get(name)
        if claim is None:
            errors.append(f"capability_{name}_missing")
            continue
        if claim.configured is not configured:
            errors.append(f"capability_{name}_configuration_mismatch")
        if claim.implemented is not implemented:
            errors.append(f"capability_{name}_implementation_mismatch")
        if claim.advertised is not advertised:
            errors.append(f"capability_{name}_advertisement_mismatch")
        if claim.advertised is True and claim.implemented is not True:
            errors.append(f"capability_{name}_overclaimed")

    dial = observation.independent_dial
    if dial.attempted is not True:
        errors.append("independent_dial_not_attempted")
    if dial.success is not True:
        errors.append("independent_dial_not_successful")
    if not dial.dialer_peer_id:
        errors.append("independent_dialer_peer_id_missing")
    if dial.dialer_peer_id == observation.peer_id:
        errors.append("independent_dial_not_independent")
    if dial.target_peer_id != observation.peer_id:
        errors.append("independent_dial_wrong_target_peer_id")
    if dial.target_multiaddr not in observation.advertised_multiaddrs:
        errors.append("independent_dial_target_not_advertised")
    if not (0.0 < float(dial.timeout_s) <= MAX_TOPOLOGY_PROBE_TIMEOUT_S):
        errors.append("independent_dial_timeout_unbounded")
    if float(dial.duration_ms) < 0 or float(dial.duration_ms) > (
        float(dial.timeout_s) * 1000.0 + 250.0
    ):
        errors.append("independent_dial_duration_out_of_bounds")
    if dial.success is True and dial.attempted is not True:
        errors.append("independent_dial_success_without_attempt")

    if type(observation.server_instance_count) is not int or observation.server_instance_count != 1:
        errors.append("server_instance_count_not_one")
    if observation.inference_attempted is not False:
        errors.append("inference_was_attempted")

    if len(observation.served_models) != 1:
        errors.append("served_model_record_count_not_one")
    for index, model in enumerate(observation.served_models):
        prefix = f"served_model_{index}"
        if model.get("id") != LEANSTRAL_LOGICAL_MODEL_ID:
            errors.append(f"{prefix}_logical_id_mismatch")
        if model.get("model_id") != LEANSTRAL_LOGICAL_MODEL_ID:
            errors.append(f"{prefix}_model_id_mismatch")
        if model.get("logical_model_id") != LEANSTRAL_LOGICAL_MODEL_ID:
            errors.append(f"{prefix}_explicit_logical_id_mismatch")
        if model.get("transport") != LEANSTRAL_HTTP_TRANSPORT:
            errors.append(f"{prefix}_transport_mismatch")
        if model.get("provider") != LEANSTRAL_HTTP_TRANSPORT:
            errors.append(f"{prefix}_provider_mismatch")
        if not str(model.get("transport_model_id") or "").strip():
            errors.append(f"{prefix}_transport_model_id_missing")
        if not str(model.get("endpoint") or "").startswith(("http://", "https://")):
            errors.append(f"{prefix}_http_endpoint_missing")
        if model.get("served") is not True or model.get("status") != "available":
            errors.append(f"{prefix}_not_available")

    observation_payload = json.loads(
        json.dumps(
            asdict(observation),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    address_selection_payload = json.loads(
        json.dumps(
            asdict(selection),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    receipt: Dict[str, Any] = {
        "schema": LEANSTRAL_TOPOLOGY_SCHEMA,
        "contract": {
            "logical_model_id": LEANSTRAL_LOGICAL_MODEL_ID,
            "http_transport": LEANSTRAL_HTTP_TRANSPORT,
            "p2p_protocol": LEANSTRAL_P2P_PROTOCOL,
            "p2p_port": LEANSTRAL_P2P_PORT,
            "listen_addr": LEANSTRAL_P2P_LISTEN_ADDR,
            "requires_fresh_external_receipt": True,
            "inference_required": False,
        },
        "observation": observation_payload,
        "address_selection": address_selection_payload,
        "validation": {
            "valid": not errors,
            "errors": sorted(set(errors)),
        },
    }
    receipt_cid = canonical_json_cid(receipt)
    return TopologyValidation(
        valid=not errors,
        errors=tuple(sorted(set(errors))),
        receipt_cid=receipt_cid,
        receipt={**receipt, "receipt_cid": receipt_cid},
    )


def _strict_object(
    value: Any,
    name: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Require one JSON object with an exact, string-keyed field contract."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use string field names")
    expected = set(required)
    allowed = expected | set(optional)
    actual = set(value)
    missing = expected - actual
    unknown = actual - allowed
    if missing or unknown:
        raise ValueError(
            f"{name} fields differ (missing={sorted(missing)}, unknown={sorted(unknown)})"
        )
    return value


def _strict_string(value: Any, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _strict_number(value: Any, name: str) -> float | int:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite JSON number")
    return value


def _strict_string_list(value: Any, name: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a list")
    for index, item in enumerate(value):
        _strict_string(item, f"{name}[{index}]")
    return list(value)


def _strict_json_value(value: Any, name: str) -> None:
    """Reject Python coercion targets that cannot originate in strict JSON."""

    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _strict_json_value(item, f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{name} must use string object keys")
            _strict_json_value(item, f"{name}.{key}")
        return
    raise ValueError(f"{name} contains a non-JSON value")


def topology_observation_from_mapping(
    value: Mapping[str, Any],
) -> LeanstralTopologyObservation:
    """Strictly decode a JSON-compatible external observation.

    Evidence is rejected instead of normalized: booleans cannot stand in for
    integers, numbers cannot become strings, and nested receipt records cannot
    silently drop or acquire fields.
    """

    root = _strict_object(
        value,
        "topology observation",
        required=(
            "p2p_requested",
            "p2p_enabled",
            "listen_addrs",
            "peer_id",
            "advertised_multiaddrs",
            "interfaces",
            "advertise_interface_allowlist",
            "bootstrap_exercises",
            "rendezvous_exercises",
            "capabilities",
            "independent_dial",
            "served_models",
            "server_instance_count",
            "inference_attempted",
        ),
        optional=("http_port", "notes"),
    )

    p2p_requested = _strict_bool(root["p2p_requested"], "p2p_requested")
    p2p_enabled = _strict_bool(root["p2p_enabled"], "p2p_enabled")
    inference_attempted = _strict_bool(
        root["inference_attempted"],
        "inference_attempted",
    )
    if type(root["server_instance_count"]) is not int:
        raise ValueError("server_instance_count must be an integer")
    server_instance_count = root["server_instance_count"]
    peer_id = _strict_string(root["peer_id"], "peer_id")
    listen_addrs = _strict_string_list(root["listen_addrs"], "listen_addrs")
    advertised_multiaddrs = _strict_string_list(
        root["advertised_multiaddrs"],
        "advertised_multiaddrs",
    )
    advertise_interface_allowlist = _strict_string_list(
        root["advertise_interface_allowlist"],
        "advertise_interface_allowlist",
    )
    notes = _strict_string_list(root.get("notes", []), "notes")
    http_port_value = root.get("http_port")
    if http_port_value is not None:
        if type(http_port_value) is not int:
            raise ValueError("http_port must be an integer or null")
        if not 1 <= http_port_value <= 65535:
            raise ValueError("http_port must be between 1 and 65535")

    interfaces_value = root["interfaces"]
    if type(interfaces_value) is not list:
        raise ValueError("interfaces must be a list")
    interfaces: list[InterfaceAddress] = []
    for index, item in enumerate(interfaces_value):
        name = f"interfaces[{index}]"
        record = _strict_object(
            item,
            name,
            required=("interface", "address", "is_up", "scope"),
        )
        interfaces.append(
            InterfaceAddress(
                interface=_strict_string(record["interface"], f"{name}.interface"),
                address=_strict_string(record["address"], f"{name}.address"),
                is_up=_strict_bool(record["is_up"], f"{name}.is_up"),
                scope=_strict_string(record["scope"], f"{name}.scope"),
            )
        )

    def decode_probes(field_name: str) -> list[ProbeObservation]:
        items = root[field_name]
        if type(items) is not list:
            raise ValueError(f"{field_name} must be a list")
        decoded: list[ProbeObservation] = []
        for index, item in enumerate(items):
            name = f"{field_name}[{index}]"
            record = _strict_object(
                item,
                name,
                required=(
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
                ),
            )
            error = record["error"]
            if error is not None:
                _strict_string(error, f"{name}.error")
            details = record["details"]
            if type(details) is not dict:
                raise ValueError(f"{name}.details must be an object")
            _strict_json_value(details, f"{name}.details")
            decoded.append(
                ProbeObservation(
                    mechanism=_strict_string(
                        record["mechanism"],
                        f"{name}.mechanism",
                    ),
                    target=_strict_string(record["target"], f"{name}.target"),
                    attempted=_strict_bool(
                        record["attempted"],
                        f"{name}.attempted",
                    ),
                    success=_strict_bool(record["success"], f"{name}.success"),
                    timeout_s=_strict_number(
                        record["timeout_s"],
                        f"{name}.timeout_s",
                    ),
                    duration_ms=_strict_number(
                        record["duration_ms"],
                        f"{name}.duration_ms",
                    ),
                    error=error,
                    observer_peer_id=_strict_string(
                        record["observer_peer_id"],
                        f"{name}.observer_peer_id",
                    ),
                    namespace=_strict_string(
                        record["namespace"],
                        f"{name}.namespace",
                    ),
                    details=dict(details),
                )
            )
        return decoded

    capability_names = (
        "mcp_stream",
        "bootstrap",
        "rendezvous",
        "pubsub",
        "floodsub",
    )
    capabilities_value = _strict_object(
        root["capabilities"],
        "capabilities",
        required=capability_names,
    )
    capabilities: Dict[str, CapabilityClaim] = {}
    for capability_name in capability_names:
        name = f"capabilities.{capability_name}"
        claim = _strict_object(
            capabilities_value[capability_name],
            name,
            required=("configured", "implemented", "advertised", "policy"),
        )
        capabilities[capability_name] = CapabilityClaim(
            configured=_strict_bool(
                claim["configured"],
                f"{name}.configured",
            ),
            implemented=_strict_bool(
                claim["implemented"],
                f"{name}.implemented",
            ),
            advertised=_strict_bool(
                claim["advertised"],
                f"{name}.advertised",
            ),
            policy=_strict_string(claim["policy"], f"{name}.policy"),
        )

    dial = _strict_object(
        root["independent_dial"],
        "independent_dial",
        required=(
            "dialer_peer_id",
            "target_peer_id",
            "target_multiaddr",
            "attempted",
            "success",
            "timeout_s",
            "duration_ms",
            "error",
        ),
    )
    dial_error = dial["error"]
    if dial_error is not None:
        _strict_string(dial_error, "independent_dial.error")
    independent_dial = IndependentDialObservation(
        dialer_peer_id=_strict_string(
            dial["dialer_peer_id"],
            "independent_dial.dialer_peer_id",
        ),
        target_peer_id=_strict_string(
            dial["target_peer_id"],
            "independent_dial.target_peer_id",
        ),
        target_multiaddr=_strict_string(
            dial["target_multiaddr"],
            "independent_dial.target_multiaddr",
        ),
        attempted=_strict_bool(
            dial["attempted"],
            "independent_dial.attempted",
        ),
        success=_strict_bool(dial["success"], "independent_dial.success"),
        timeout_s=_strict_number(
            dial["timeout_s"],
            "independent_dial.timeout_s",
        ),
        duration_ms=_strict_number(
            dial["duration_ms"],
            "independent_dial.duration_ms",
        ),
        error=dial_error,
    )

    served_models_value = root["served_models"]
    if type(served_models_value) is not list:
        raise ValueError("served_models must be a list")
    served_models: list[Mapping[str, Any]] = []
    model_string_fields = (
        "id",
        "model_id",
        "name",
        "logical_model_id",
        "transport_model_id",
        "provider",
        "transport",
        "endpoint",
        "status",
    )
    for index, item in enumerate(served_models_value):
        name = f"served_models[{index}]"
        record = _strict_object(
            item,
            name,
            required=(
                *model_string_fields,
                "served",
                "capabilities",
                "metadata",
            ),
        )
        normalized_model = {
            field_name: _strict_string(
                record[field_name],
                f"{name}.{field_name}",
            )
            for field_name in model_string_fields
        }
        normalized_model["served"] = _strict_bool(
            record["served"],
            f"{name}.served",
        )
        normalized_model["capabilities"] = _strict_string_list(
            record["capabilities"],
            f"{name}.capabilities",
        )
        metadata = record["metadata"]
        if type(metadata) is not dict:
            raise ValueError(f"{name}.metadata must be an object")
        _strict_json_value(metadata, f"{name}.metadata")
        normalized_model["metadata"] = dict(metadata)
        served_models.append(normalized_model)

    return LeanstralTopologyObservation(
        p2p_requested=p2p_requested,
        p2p_enabled=p2p_enabled,
        listen_addrs=tuple(listen_addrs),
        peer_id=peer_id,
        advertised_multiaddrs=tuple(advertised_multiaddrs),
        interfaces=tuple(interfaces),
        advertise_interface_allowlist=tuple(advertise_interface_allowlist),
        bootstrap_exercises=tuple(decode_probes("bootstrap_exercises")),
        rendezvous_exercises=tuple(decode_probes("rendezvous_exercises")),
        capabilities=capabilities,
        independent_dial=independent_dial,
        served_models=tuple(served_models),
        server_instance_count=server_instance_count,
        inference_attempted=inference_attempted,
        http_port=http_port_value,
        notes=tuple(notes),
    )


def validate_leanstral_topology_mapping(
    value: Mapping[str, Any],
) -> TopologyValidation:
    """Decode and validate one JSON-compatible external observation."""

    return validate_leanstral_topology(topology_observation_from_mapping(value))


def leanstral_topology_environment() -> Dict[str, str]:
    """Return the explicit, non-destructive environment contract for a runner."""

    return {
        "MCPPP_P2P_LISTEN_ADDRS": LEANSTRAL_P2P_LISTEN_ADDR,
        "MCPPP_P2P_ADVERTISE_INTERFACES": "<comma-separated host LAN interfaces>",
        "MCPPP_P2P_ADVERTISE_ADDRS": (f"/ip4/<policy-selected-host-ip>/tcp/{LEANSTRAL_P2P_PORT}"),
        "MCPPP_P2P_BOOTSTRAP_PEERS": "<one-or-more exact peer multiaddrs>",
        "MCPPP_P2P_RENDEZVOUS_SERVICE": "same_as_service_peer",
        "IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER": (
            "<external-client-only: exact service peer ID or multiaddr>"
        ),
        "IPFS_ACCELERATE_P2P_RENDEZVOUS_NS": "leanstral-local",
        "IPFS_ACCELERATE_SERVED_MODEL_ENDPOINTS": ("http://127.0.0.1:8080/v1"),
    }


__all__ = [
    "AddressSelection",
    "CapabilityClaim",
    "IndependentDialObservation",
    "InterfaceAddress",
    "LEANSTRAL_HTTP_TRANSPORT",
    "LEANSTRAL_LOGICAL_MODEL_ID",
    "LEANSTRAL_P2P_LISTEN_ADDR",
    "LEANSTRAL_P2P_PORT",
    "LEANSTRAL_P2P_PROTOCOL",
    "LEANSTRAL_TOPOLOGY_SCHEMA",
    "LeanstralTopologyObservation",
    "MAX_TOPOLOGY_PROBE_TIMEOUT_S",
    "ProbeObservation",
    "TopologyValidation",
    "default_capability_claims",
    "canonical_json_cid",
    "is_leanstral_transport_model_id",
    "leanstral_advertised_multiaddrs",
    "leanstral_topology_environment",
    "normalize_served_model_record",
    "select_advertised_ipv4",
    "topology_observation_from_mapping",
    "validate_leanstral_topology",
    "validate_leanstral_topology_mapping",
]
