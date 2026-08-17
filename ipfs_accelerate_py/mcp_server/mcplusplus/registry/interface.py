"""Registry@1 — pluggable agent advertisement registry contract.

A registry indexes :class:`AgentAdvertisement@1` records for discovery and
routing. Finding a record is **not** permission to execute: registry presence
is never execution authority (plan KD-14; MCPP-G110). UCAN / policy proofs
authorize invocation; health and load are selection inputs only.

Concrete adapters (static, libp2p, AGNTCY) implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

REGISTRY_INTERFACE = "Registry@1"
AGENT_ADVERTISEMENT_SCHEMA = "mcp++/discovery/agent-advertisement@1"
AGENT_ADVERTISEMENT_INTERFACE = "AgentAdvertisement@1"

# Health ranks for selection: lower is preferred.
HEALTH_RANK: Mapping[str, int] = {
    "healthy": 0,
    "degraded": 1,
    "unknown": 2,
    "unhealthy": 3,
}

# Maximum ttl_ms from agent-advertisement-1.schema.json (7 days).
MAX_TTL_MS = 604_800_000
MIN_TTL_MS = 1


class RegistryError(Exception):
    """Base error for Registry@1 operations."""


class RegistryValidationError(RegistryError, ValueError):
    """Raised when an advertisement fails structural validation."""


class RegistryStaleError(RegistryError):
    """Raised when an advertisement is past TTL/expiry at evaluation time."""


class RegistryNotFoundError(RegistryError, KeyError):
    """Raised when a required identity is not present in the registry."""


class RegistryUnsignedError(RegistryError):
    """Raised when a signed-only registry receives an unsigned advertisement."""


class RegistryAuthorityError(RegistryError):
    """Raised when code treats registry presence as execution authority."""


def is_execution_authority(_advertisement: Mapping[str, Any] | None = None) -> bool:
    """Return False: registry membership never grants execution authority."""

    return False


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{field} must be a mapping")
    return value


def _require_non_empty_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_str_list(value: object, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RegistryValidationError(f"{field} must be a sequence of strings")
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RegistryValidationError(f"{field}[{index}] must be a non-empty string")
        out.append(item.strip())
    return out


def identity_did(advertisement: Mapping[str, Any]) -> str:
    """Return the principal DID from a validated or raw advertisement."""

    identity = advertisement.get("identity")
    if not isinstance(identity, Mapping):
        raise RegistryValidationError("identity must be a mapping")
    return _require_non_empty_str(identity.get("did"), field="identity.did")


def advertisement_expires_at_ms(
    advertisement: Mapping[str, Any],
    *,
    receipt_ms: Optional[int] = None,
) -> int:
    """Compute absolute expiry in epoch milliseconds.

    Preference order:
    1. ``expires_at_ms`` when present
    2. ``published_at_ms + ttl_ms``
    3. ``receipt_ms + ttl_ms`` (caller-supplied receipt / now at publish time)
    """

    expires = advertisement.get("expires_at_ms")
    if expires is not None:
        if not isinstance(expires, int) or isinstance(expires, bool):
            raise RegistryValidationError("expires_at_ms must be an integer")
        if expires < 0:
            raise RegistryValidationError("expires_at_ms must be non-negative")
        return expires

    ttl = advertisement.get("ttl_ms")
    if not isinstance(ttl, int) or isinstance(ttl, bool):
        raise RegistryValidationError("ttl_ms must be an integer")
    if ttl < MIN_TTL_MS or ttl > MAX_TTL_MS:
        raise RegistryValidationError(
            f"ttl_ms must be in [{MIN_TTL_MS}, {MAX_TTL_MS}]"
        )

    published = advertisement.get("published_at_ms")
    if published is not None:
        if not isinstance(published, int) or isinstance(published, bool):
            raise RegistryValidationError("published_at_ms must be an integer")
        if published < 0:
            raise RegistryValidationError("published_at_ms must be non-negative")
        return published + ttl

    if receipt_ms is None:
        raise RegistryValidationError(
            "cannot compute expiry without expires_at_ms, published_at_ms, or receipt_ms"
        )
    if not isinstance(receipt_ms, int) or isinstance(receipt_ms, bool) or receipt_ms < 0:
        raise RegistryValidationError("receipt_ms must be a non-negative integer")
    return receipt_ms + ttl


def is_stale(
    advertisement: Mapping[str, Any],
    *,
    now_ms: int,
    receipt_ms: Optional[int] = None,
) -> bool:
    """Return True when the advertisement is past TTL/expiry at ``now_ms``."""

    if not isinstance(now_ms, int) or isinstance(now_ms, bool) or now_ms < 0:
        raise RegistryValidationError("now_ms must be a non-negative integer")
    expires_at = advertisement_expires_at_ms(advertisement, receipt_ms=receipt_ms)
    return now_ms > expires_at


def validate_agent_advertisement(
    advertisement: Mapping[str, Any],
    *,
    require_signed: bool = False,
    now_ms: Optional[int] = None,
    reject_stale: bool = True,
    receipt_ms: Optional[int] = None,
) -> dict[str, Any]:
    """Structurally validate AgentAdvertisement@1 and return a shallow copy.

    Fail-closed rules mirror the discovery schema for fields this layer needs:
    ``schema``, ``identity.did``, ``ttl_ms``, and ``interface_cids``. Full JSON
    Schema evaluation is left to dedicated schema tooling.
    """

    ad = _require_mapping(advertisement, field="advertisement")

    schema = ad.get("schema")
    if schema is not None and schema != AGENT_ADVERTISEMENT_SCHEMA:
        raise RegistryValidationError(
            f"schema must be {AGENT_ADVERTISEMENT_SCHEMA!r}, got {schema!r}"
        )

    identity = _require_mapping(ad.get("identity"), field="identity")
    did = _require_non_empty_str(identity.get("did"), field="identity.did")
    if not did.startswith("did:"):
        raise RegistryValidationError("identity.did must be a DID (did:…)")

    ttl = ad.get("ttl_ms")
    if not isinstance(ttl, int) or isinstance(ttl, bool):
        raise RegistryValidationError("ttl_ms is required and must be an integer")
    if ttl < MIN_TTL_MS or ttl > MAX_TTL_MS:
        raise RegistryValidationError(
            f"ttl_ms must be in [{MIN_TTL_MS}, {MAX_TTL_MS}]"
        )

    if "interface_cids" not in ad:
        raise RegistryValidationError("interface_cids is required")
    interface_cids = _optional_str_list(ad.get("interface_cids"), field="interface_cids")

    if require_signed:
        signature = ad.get("signature")
        if not isinstance(signature, Mapping):
            raise RegistryUnsignedError(
                "signature is required when the registry requires signed advertisements"
            )
        for key in ("signer_did", "signature_alg", "signature"):
            if not isinstance(signature.get(key), str) or not str(signature.get(key)).strip():
                raise RegistryUnsignedError(
                    f"signature.{key} is required for signed advertisements"
                )

    out = dict(ad)
    if schema is None:
        out["schema"] = AGENT_ADVERTISEMENT_SCHEMA
    out["identity"] = dict(identity)
    out["identity"]["did"] = did
    out["ttl_ms"] = ttl
    out["interface_cids"] = list(interface_cids)

    eval_now = now_ms if now_ms is not None else receipt_ms
    if reject_stale and eval_now is not None:
        if is_stale(out, now_ms=eval_now, receipt_ms=receipt_ms):
            raise RegistryStaleError(
                f"advertisement for {did!r} is stale at now_ms={eval_now}"
            )

    return out


def health_status(advertisement: Mapping[str, Any]) -> str:
    """Return health status string; missing health is treated as ``unknown``."""

    health = advertisement.get("health")
    if not isinstance(health, Mapping):
        return "unknown"
    status = health.get("status")
    if isinstance(status, str) and status in HEALTH_RANK:
        return status
    return "unknown"


def health_rank(advertisement: Mapping[str, Any]) -> int:
    """Return selection rank for health (lower is better)."""

    return int(HEALTH_RANK.get(health_status(advertisement), HEALTH_RANK["unknown"]))


def load_selection_tuple(advertisement: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Return load-related sort keys (lower is better at each position).

    Order: utilization_millionths, -capacity_millionths (as inverted capacity),
    inflight, queue_depth. Missing fields use neutral defaults that do not
    invent trust — only break ties consistently.
    """

    load = advertisement.get("load")
    if not isinstance(load, Mapping):
        # Unknown load ranks after explicit idle/low-load peers but before saturated.
        return (500_000, 500_000, 0, 0)

    util = load.get("utilization_millionths")
    if not isinstance(util, int) or isinstance(util, bool):
        util = 500_000
    capacity = load.get("capacity_millionths")
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        capacity = 500_000
    inflight = load.get("inflight")
    if not isinstance(inflight, int) or isinstance(inflight, bool):
        inflight = 0
    queue = load.get("queue_depth")
    if not isinstance(queue, int) or isinstance(queue, bool):
        queue = 0
    # Invert capacity so higher remaining capacity sorts first (lower tuple).
    inverted_capacity = max(0, 1_000_000 - capacity)
    return (util, inverted_capacity, inflight, queue)


def selection_key(advertisement: Mapping[str, Any]) -> tuple[Any, ...]:
    """Deterministic total order for health-aware selection.

    1. Health rank (healthy preferred)
    2. Load tuple (lower utilization / higher capacity preferred)
    3. Lexicographic identity DID (stable final tie-break)
    """

    did = identity_did(advertisement)
    return (health_rank(advertisement),) + load_selection_tuple(advertisement) + (did,)


def sort_for_selection(
    advertisements: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return advertisements sorted by :func:`selection_key` (best first)."""

    items = [dict(ad) for ad in advertisements]
    items.sort(key=selection_key)
    return items


def select_advertisement(
    advertisements: Iterable[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Pick the best advertisement under the deterministic selection order.

    Returns a shallow copy of the winner, or ``None`` when the candidate set is
    empty. Does not grant execution authority.
    """

    ordered = sort_for_selection(advertisements)
    if not ordered:
        return None
    return ordered[0]


def semantic_capabilities_of(advertisement: Mapping[str, Any]) -> set[str]:
    """Collect semantic capability tokens from skills projections."""

    caps: set[str] = set()
    skills = advertisement.get("skills")
    if not isinstance(skills, Sequence) or isinstance(skills, (str, bytes)):
        return caps
    for skill in skills:
        if not isinstance(skill, Mapping):
            continue
        for key in ("id", "name", "method"):
            value = skill.get(key)
            if isinstance(value, str) and value.strip():
                caps.add(value.strip())
        tags = skill.get("tags")
        if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    caps.add(tag.strip())
    return caps


def policy_languages_of(advertisement: Mapping[str, Any]) -> set[str]:
    """Return the set of policy language ids from the advertisement."""

    return set(_optional_str_list(advertisement.get("policy_languages"), field="policy_languages"))


def proof_systems_of(advertisement: Mapping[str, Any]) -> set[str]:
    """Return the set of proof system ids from the advertisement."""

    return set(_optional_str_list(advertisement.get("proof_systems"), field="proof_systems"))


class Registry(ABC):
    """Abstract Registry@1 contract for agent advertisement discovery.

    Implementations MUST:

    * reject structurally invalid advertisements fail-closed;
    * reject stale records for selection and default lookup paths;
    * when configured to require signatures, reject unsigned records;
    * never treat registry membership as execution authority;
    * apply health-aware selection with a deterministic tie-break.
    """

    @property
    def interface(self) -> str:
        """Wire interface label for this registry family."""

        return REGISTRY_INTERFACE

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Stable backend label (e.g. ``static``, ``libp2p``)."""

    @abstractmethod
    def publish(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Validate and store an advertisement (insert or replace by DID)."""

    @abstractmethod
    def refresh(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Refresh an existing advertisement; fail if the identity is unknown."""

    @abstractmethod
    def withdraw(
        self,
        identity_did: str,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Remove the advertisement for ``identity_did``. Return True if removed."""

    @abstractmethod
    def lookup_by_identity(
        self,
        identity_did: str,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Return the advertisement for ``identity_did``, if present and fresh."""

    @abstractmethod
    def lookup_by_interface_cid(
        self,
        interface_cid: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return non-stale advertisements claiming ``interface_cid``."""

    @abstractmethod
    def lookup_by_semantic_capability(
        self,
        capability: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return non-stale advertisements matching a skill tag/id/name/method."""

    @abstractmethod
    def lookup_by_policy(
        self,
        policy_language: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return non-stale advertisements listing ``policy_language``."""

    @abstractmethod
    def lookup_by_proof(
        self,
        proof_system: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return non-stale advertisements listing ``proof_system``."""

    @abstractmethod
    def select(
        self,
        *,
        interface_cid: Optional[str] = None,
        semantic_capability: Optional[str] = None,
        policy_language: Optional[str] = None,
        proof_system: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        now_ms: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Health-aware deterministic selection among matching advertisements."""

    def list_all(
        self,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        """Return stored advertisements (optional; default empty)."""

        del now_ms, include_stale
        return []

    def stats(self) -> MutableMapping[str, Any]:
        """Return deterministic registry diagnostics."""

        return {
            "interface": self.interface,
            "provider": self.provider_id,
            "execution_authority": False,
        }


__all__ = [
    "AGENT_ADVERTISEMENT_INTERFACE",
    "AGENT_ADVERTISEMENT_SCHEMA",
    "HEALTH_RANK",
    "MAX_TTL_MS",
    "MIN_TTL_MS",
    "REGISTRY_INTERFACE",
    "Registry",
    "RegistryAuthorityError",
    "RegistryError",
    "RegistryNotFoundError",
    "RegistryStaleError",
    "RegistryUnsignedError",
    "RegistryValidationError",
    "advertisement_expires_at_ms",
    "health_rank",
    "health_status",
    "identity_did",
    "is_execution_authority",
    "is_stale",
    "load_selection_tuple",
    "policy_languages_of",
    "proof_systems_of",
    "select_advertisement",
    "selection_key",
    "semantic_capabilities_of",
    "sort_for_selection",
    "validate_agent_advertisement",
]
