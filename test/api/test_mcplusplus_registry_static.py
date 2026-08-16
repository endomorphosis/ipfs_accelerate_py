"""Tests for Registry@1 and StaticRegistry@1 (MCPP-059).

Acceptance:
* Static registry tests cover publish, refresh, withdraw, lookup by
  identity / interface CID / semantic capability / policy / proof,
  health-aware selection, deterministic tie-break, and stale rejection.
* Selection is deterministic for equal health.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
    AGENT_ADVERTISEMENT_SCHEMA,
    HEALTH_RANK,
    REGISTRY_INTERFACE,
    Registry,
    RegistryNotFoundError,
    RegistryStaleError,
    RegistryUnsignedError,
    RegistryValidationError,
    is_execution_authority,
    select_advertisement,
    selection_key,
    validate_agent_advertisement,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.registry.static import (
    STATIC_PROVIDER_ID,
    STATIC_REGISTRY_INTERFACE,
    StaticRegistry,
    create_static_registry,
)

# Fixed epoch used as a synthetic "now" for hermetic tests.
T0 = 1_700_000_000_000
IFACE_A = "bafkreigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
IFACE_B = "bafkreihxs3rpfcxfqeltptfyem7tjye7ro3v2jirue2vipi4un56agm2du"


def _ad(
    did: str,
    *,
    interfaces: Optional[list[str]] = None,
    ttl_ms: int = 60_000,
    published_at_ms: Optional[int] = T0,
    expires_at_ms: Optional[int] = None,
    health: Optional[str] = "healthy",
    utilization: Optional[int] = None,
    capacity: Optional[int] = None,
    skills: Optional[list[dict[str, Any]]] = None,
    policy_languages: Optional[list[str]] = None,
    proof_systems: Optional[list[str]] = None,
    signature: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema": AGENT_ADVERTISEMENT_SCHEMA,
        "identity": {"did": did, "name": did.rsplit(":", 1)[-1]},
        "ttl_ms": ttl_ms,
        "interface_cids": list(interfaces if interfaces is not None else [IFACE_A]),
    }
    if published_at_ms is not None:
        body["published_at_ms"] = published_at_ms
    if expires_at_ms is not None:
        body["expires_at_ms"] = expires_at_ms
    if health is not None:
        body["health"] = {"status": health}
    if utilization is not None or capacity is not None:
        load: Dict[str, Any] = {}
        if utilization is not None:
            load["utilization_millionths"] = utilization
        if capacity is not None:
            load["capacity_millionths"] = capacity
        body["load"] = load
    if skills is not None:
        body["skills"] = skills
    if policy_languages is not None:
        body["policy_languages"] = policy_languages
    if proof_systems is not None:
        body["proof_systems"] = proof_systems
    if signature is not None:
        body["signature"] = signature
    if extra:
        body.update(extra)
    return body


class _Clock:
    def __init__(self, start: int = T0) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, delta_ms: int) -> None:
        self.now += delta_ms


class TestRegistryInterfaceContract:
    def test_interface_constants(self) -> None:
        assert REGISTRY_INTERFACE == "Registry@1"
        assert STATIC_REGISTRY_INTERFACE == "StaticRegistry@1"
        assert STATIC_PROVIDER_ID == "static"
        assert AGENT_ADVERTISEMENT_SCHEMA == "mcp++/discovery/agent-advertisement@1"

    def test_static_registry_is_registry(self) -> None:
        reg = create_static_registry(clock_ms=_Clock())
        assert isinstance(reg, Registry)
        assert isinstance(reg, StaticRegistry)
        assert reg.interface == STATIC_REGISTRY_INTERFACE
        assert reg.family_interface == REGISTRY_INTERFACE
        assert reg.provider_id == STATIC_PROVIDER_ID

    def test_registry_presence_is_not_execution_authority(self) -> None:
        assert is_execution_authority() is False
        ad = _ad("did:web:agent.example")
        assert is_execution_authority(ad) is False
        reg = create_static_registry(clock_ms=_Clock())
        reg.publish(ad)
        stats = reg.stats()
        assert stats["execution_authority"] is False

    def test_validate_requires_identity_ttl_interface_cids(self) -> None:
        with pytest.raises(RegistryValidationError):
            validate_agent_advertisement(
                {"schema": AGENT_ADVERTISEMENT_SCHEMA, "ttl_ms": 1000, "interface_cids": []}
            )
        with pytest.raises(RegistryValidationError):
            validate_agent_advertisement(
                {
                    "schema": AGENT_ADVERTISEMENT_SCHEMA,
                    "identity": {"did": "did:web:x"},
                    "interface_cids": [],
                }
            )
        with pytest.raises(RegistryValidationError):
            validate_agent_advertisement(
                {
                    "schema": AGENT_ADVERTISEMENT_SCHEMA,
                    "identity": {"did": "did:web:x"},
                    "ttl_ms": 1000,
                }
            )

    def test_validate_accepts_minimal_ad(self) -> None:
        out = validate_agent_advertisement(
            {
                "identity": {"did": "did:web:agent.example"},
                "ttl_ms": 300_000,
                "interface_cids": [IFACE_A],
            },
            reject_stale=False,
        )
        assert out["schema"] == AGENT_ADVERTISEMENT_SCHEMA
        assert out["identity"]["did"] == "did:web:agent.example"


class TestPublishRefreshWithdraw:
    def test_publish_and_lookup_by_identity(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        ad = _ad("did:web:alpha")
        stored = reg.publish(ad)
        assert stored["identity"]["did"] == "did:web:alpha"
        found = reg.lookup_by_identity("did:web:alpha")
        assert found is not None
        assert found["identity"]["did"] == "did:web:alpha"
        assert found["interface_cids"] == [IFACE_A]

    def test_publish_replaces_by_identity(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha", interfaces=[IFACE_A]))
        reg.publish(_ad("did:web:alpha", interfaces=[IFACE_B]))
        found = reg.lookup_by_identity("did:web:alpha")
        assert found is not None
        assert found["interface_cids"] == [IFACE_B]
        assert len(reg.list_all()) == 1

    def test_refresh_requires_existing_identity(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        with pytest.raises(RegistryNotFoundError):
            reg.refresh(_ad("did:web:missing"))

    def test_refresh_updates_existing(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha", health="degraded", utilization=900_000))
        refreshed = reg.refresh(
            _ad("did:web:alpha", health="healthy", utilization=100_000)
        )
        assert refreshed["health"]["status"] == "healthy"
        found = reg.lookup_by_identity("did:web:alpha")
        assert found is not None
        assert found["health"]["status"] == "healthy"
        assert reg.stats()["refresh_count"] == 1

    def test_withdraw_removes_record(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha"))
        assert reg.withdraw("did:web:alpha") is True
        assert reg.lookup_by_identity("did:web:alpha") is None
        assert reg.withdraw("did:web:alpha") is False
        assert reg.stats()["withdraw_count"] == 1


class TestLookups:
    def test_lookup_by_interface_cid(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:a", interfaces=[IFACE_A]))
        reg.publish(_ad("did:web:b", interfaces=[IFACE_B]))
        reg.publish(_ad("did:web:c", interfaces=[IFACE_A, IFACE_B]))
        matches = reg.lookup_by_interface_cid(IFACE_A)
        dids = {m["identity"]["did"] for m in matches}
        assert dids == {"did:web:a", "did:web:c"}

    def test_lookup_by_semantic_capability(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(
            _ad(
                "did:web:git",
                skills=[
                    {
                        "id": "repo.status",
                        "name": "Repository status",
                        "tags": ["vcs", "git"],
                        "method": "repo.status",
                    }
                ],
            )
        )
        reg.publish(
            _ad(
                "did:web:other",
                skills=[{"id": "weather.forecast", "tags": ["weather"]}],
            )
        )
        by_tag = reg.lookup_by_semantic_capability("git")
        assert len(by_tag) == 1
        assert by_tag[0]["identity"]["did"] == "did:web:git"
        by_id = reg.lookup_by_semantic_capability("repo.status")
        assert len(by_id) == 1
        assert reg.lookup_by_semantic_capability("missing") == []

    def test_lookup_by_policy_and_proof(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(
            _ad(
                "did:web:policy-agent",
                policy_languages=["temporal-deontic@1"],
                proof_systems=["ucan", "EdDSA"],
            )
        )
        reg.publish(
            _ad(
                "did:web:plain",
                policy_languages=["other@1"],
                proof_systems=["EdDSA"],
            )
        )
        policy_hits = reg.lookup_by_policy("temporal-deontic@1")
        assert [h["identity"]["did"] for h in policy_hits] == ["did:web:policy-agent"]
        proof_hits = reg.lookup_by_proof("ucan")
        assert [h["identity"]["did"] for h in proof_hits] == ["did:web:policy-agent"]
        eddsa = reg.lookup_by_proof("EdDSA")
        assert {h["identity"]["did"] for h in eddsa} == {
            "did:web:policy-agent",
            "did:web:plain",
        }


class TestStaleRejection:
    def test_publish_rejects_already_stale_advertisement(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        stale = _ad(
            "did:web:stale",
            published_at_ms=T0 - 1_000_000,
            ttl_ms=1_000,
        )
        with pytest.raises(RegistryStaleError):
            reg.publish(stale)
        assert reg.stats()["stale_rejects"] == 1

    def test_lookup_hides_stale_records(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha", ttl_ms=5_000, published_at_ms=T0))
        clock.advance(10_000)
        assert reg.lookup_by_identity("did:web:alpha") is None
        stale = reg.lookup_by_identity("did:web:alpha", include_stale=True)
        assert stale is not None
        assert reg.lookup_by_interface_cid(IFACE_A) == []
        assert reg.list_all() == []
        assert len(reg.list_all(include_stale=True)) == 1

    def test_expires_at_ms_drives_staleness(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(
            _ad(
                "did:web:alpha",
                published_at_ms=T0,
                ttl_ms=60_000,
                expires_at_ms=T0 + 2_000,
            )
        )
        assert reg.lookup_by_identity("did:web:alpha") is not None
        clock.advance(3_000)
        assert reg.lookup_by_identity("did:web:alpha") is None


class TestSignedRequirement:
    def test_require_signed_rejects_unsigned(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(require_signed=True, clock_ms=clock)
        with pytest.raises(RegistryUnsignedError):
            reg.publish(_ad("did:web:alpha"))
        assert reg.stats()["unsigned_rejects"] == 1

    def test_require_signed_accepts_signature_block(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(require_signed=True, clock_ms=clock)
        ad = _ad(
            "did:web:alpha",
            signature={
                "signer_did": "did:web:alpha",
                "signature_alg": "EdDSA",
                "signature": "dGVzdA",
            },
        )
        stored = reg.publish(ad)
        assert stored["signature"]["signature_alg"] == "EdDSA"


class TestHealthAwareSelection:
    def test_prefers_healthier_peer(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:degraded", health="degraded", utilization=0))
        reg.publish(_ad("did:web:healthy", health="healthy", utilization=900_000))
        chosen = reg.select(interface_cid=IFACE_A)
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:healthy"

    def test_equal_health_is_deterministic_by_did(self) -> None:
        """Equal health + equal load → lexicographic DID wins every time."""

        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        # Insert in reverse lexicographic order to ensure sort, not insert order.
        reg.publish(
            _ad("did:web:zeta", health="healthy", utilization=100_000, capacity=900_000)
        )
        reg.publish(
            _ad("did:web:alpha", health="healthy", utilization=100_000, capacity=900_000)
        )
        reg.publish(
            _ad("did:web:mu", health="healthy", utilization=100_000, capacity=900_000)
        )
        winners = [reg.select(interface_cid=IFACE_A)["identity"]["did"] for _ in range(5)]
        assert winners == ["did:web:alpha"] * 5

        # Pure helper agrees with registry selection.
        ads = reg.list_all()
        assert select_advertisement(ads)["identity"]["did"] == "did:web:alpha"
        keys = [selection_key(ad) for ad in ads]
        assert keys == sorted(keys)

    def test_equal_health_prefers_lower_load(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(
            _ad("did:web:busy", health="healthy", utilization=800_000, capacity=200_000)
        )
        reg.publish(
            _ad("did:web:idle", health="healthy", utilization=100_000, capacity=900_000)
        )
        chosen = reg.select(interface_cid=IFACE_A)
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:idle"

    def test_select_with_filters_and_empty(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(
            _ad(
                "did:web:git",
                skills=[{"id": "repo.status", "tags": ["git"]}],
                policy_languages=["temporal-deontic@1"],
                proof_systems=["ucan"],
            )
        )
        chosen = reg.select(
            interface_cid=IFACE_A,
            semantic_capability="git",
            policy_language="temporal-deontic@1",
            proof_system="ucan",
        )
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:git"
        assert reg.select(semantic_capability="nope") is None

    def test_select_excludes_stale_candidates(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:old", ttl_ms=1_000, published_at_ms=T0))
        reg.publish(_ad("did:web:fresh", ttl_ms=60_000, published_at_ms=T0))
        clock.advance(5_000)
        chosen = reg.select(interface_cid=IFACE_A)
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:fresh"

    def test_health_rank_ordering_constants(self) -> None:
        assert HEALTH_RANK["healthy"] < HEALTH_RANK["degraded"]
        assert HEALTH_RANK["degraded"] < HEALTH_RANK["unknown"]
        assert HEALTH_RANK["unknown"] < HEALTH_RANK["unhealthy"]
