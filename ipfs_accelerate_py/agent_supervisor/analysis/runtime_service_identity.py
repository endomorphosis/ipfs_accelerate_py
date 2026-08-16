"""DCR-022 read-only runtime service identity receipts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final
from urllib.parse import urlsplit

from ..proof.formal_verification_contracts import content_identity

RUNTIME_SERVICE_IDENTITY_INTERFACE: Final[str] = "RuntimeServiceIdentity@1"


class ServiceIdentityStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INTEGRATION_PENDING = "integration_pending"


def _loopback(endpoint: str) -> bool:
    """Accept only an unadorned, literal loopback HTTP endpoint."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and parsed.username is None
        and parsed.password is None
        and port is not None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


@dataclass(frozen=True)
class RuntimeServiceObservation:
    role: str
    interpreter: str
    module_origin: str
    module_digest: str
    checkout_commit: str
    checkout_tree: str
    overlay_id: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    config_cid: str
    state_cid: str
    transport: str
    endpoint: str
    pid: int
    started_at: str
    # Opaque diagnostic evidence from the observer.  It is never an authority
    # claim; `process_witness_cid` is recomputed from the bound fields below.
    process_identity: str
    observed_port: int

    def __post_init__(self) -> None:
        if not all(
            (
                self.role,
                self.interpreter,
                self.module_origin,
                self.module_digest,
                self.checkout_commit,
                self.checkout_tree,
                self.config_cid,
                self.state_cid,
                self.transport,
                self.endpoint,
                self.process_identity,
            )
        ):
            raise ValueError("runtime observation omits required identity evidence")
        if self.pid <= 0 or self.observed_port <= 0:
            raise ValueError("pid and observed_port must be positive")


@dataclass(frozen=True)
class RuntimeServiceIdentity:
    observation: RuntimeServiceObservation
    status: ServiceIdentityStatus
    reasons: tuple[str, ...]

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_receipt=False))

    @property
    def process_witness_cid(self) -> str:
        o = self.observation
        return content_identity(
            {
                "pid": o.pid,
                "started_at": o.started_at,
                "interpreter": o.interpreter,
                "argv": list(o.argv),
                "module_origin": o.module_origin,
                "module_digest": o.module_digest,
                "checkout_commit": o.checkout_commit,
                "checkout_tree": o.checkout_tree,
                "overlay_id": o.overlay_id,
                "config_cid": o.config_cid,
                "state_cid": o.state_cid,
                "endpoint": o.endpoint,
                "transport": o.transport,
            }
        )

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        o = self.observation
        payload = {
            "interface": RUNTIME_SERVICE_IDENTITY_INTERFACE,
            "role": o.role,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "interpreter": o.interpreter,
            "module_origin": o.module_origin,
            "module_digest": o.module_digest,
            "checkout_commit": o.checkout_commit,
            "checkout_tree": o.checkout_tree,
            "overlay_id": o.overlay_id,
            "argv": list(o.argv),
            "environment": dict(sorted(o.environment.items())),
            "config_cid": o.config_cid,
            "state_cid": o.state_cid,
            "transport": o.transport,
            "endpoint": o.endpoint,
            "pid": o.pid,
            "started_at": o.started_at,
            "process_identity": o.process_identity,
            "process_witness_cid": self.process_witness_cid,
            "completion_authoritative": False,
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


def observe_runtime_service(
    observation: RuntimeServiceObservation,
    *,
    expected_role: str,
    expected_interpreter: str,
    expected_module_digest: str,
    expected_commit: str,
    expected_tree: str,
    expected_config_cid: str,
    expected_state_cid: str,
    expected_transport: str,
    expected_port: int,
    claimed_process_witness_cid: str = "",
    environment_allowlist: frozenset[str],
    identity_vectors_ready: bool = False,
) -> RuntimeServiceIdentity:
    """Validate supplied observations only; never open a connection or inspect os.environ."""
    reasons = []
    o = observation
    if o.role != expected_role:
        reasons.append("role_mismatch")
    if o.interpreter != expected_interpreter:
        reasons.append("interpreter_mismatch")
    if not o.module_origin:
        reasons.append("module_origin_missing")
    if o.module_digest != expected_module_digest:
        reasons.append("module_digest_mismatch")
    if o.checkout_commit != expected_commit or o.checkout_tree != expected_tree:
        reasons.append("checkout_mismatch")
    if o.config_cid != expected_config_cid or o.state_cid != expected_state_cid:
        reasons.append("config_or_state_drift")
    if o.transport != expected_transport:
        reasons.append("transport_mismatch")
    if not _loopback(o.endpoint):
        reasons.append("remote_endpoint")
    if o.observed_port != expected_port:
        reasons.append("port_disagreement")
    if set(o.environment).difference(environment_allowlist):
        reasons.append("environment_not_sealed")
    result = RuntimeServiceIdentity(o, ServiceIdentityStatus.INTEGRATION_PENDING, ())
    if claimed_process_witness_cid and claimed_process_witness_cid != result.process_witness_cid:
        reasons.append("process_replaced_or_reused")
    status = (
        ServiceIdentityStatus.INVALID
        if reasons
        else (
            ServiceIdentityStatus.VALID
            if identity_vectors_ready and claimed_process_witness_cid
            else ServiceIdentityStatus.INTEGRATION_PENDING
        )
    )
    return RuntimeServiceIdentity(o, status, tuple(sorted(reasons)))
