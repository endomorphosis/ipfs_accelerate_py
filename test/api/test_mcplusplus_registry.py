"""Adversarial Registry@1 abuse-vector tests (MCPP-061).

Interface: RegistryAbuseVector@1

Acceptance:
* Compromised / unsigned records are rejected.
* A matching advertisement without a valid delegation cannot dispatch.
* Expiry, withdrawal, duplicates, and deterministic selection are covered.
* Registry presence is never execution authority (KD-14; MCPP-G110).

Evidence subset: Registry, UCAN verifier.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.delegation import (
    UcanValidationResult,
    validate_raw_delegation_chain,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
    AGENT_ADVERTISEMENT_SCHEMA,
    HEALTH_RANK,
    REGISTRY_INTERFACE,
    RegistryAuthorityError,
    RegistryStaleError,
    RegistryUnsignedError,
    RegistryValidationError,
    is_execution_authority,
    select_advertisement,
    selection_key,
    validate_agent_advertisement,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.registry.libp2p import (
    LIBP2P_PROVIDER_ID,
    create_libp2p_discovery,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.registry.static import (
    STATIC_PROVIDER_ID,
    StaticRegistry,
    create_static_registry,
)

# ---------------------------------------------------------------------------
# Constants (RegistryAbuseVector@1)
# ---------------------------------------------------------------------------

INTERFACE_LABEL = "RegistryAbuseVector@1"

T0 = 1_700_000_000_000
IFACE_A = "bafkreigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
IFACE_B = "bafkreihxs3rpfcxfqeltptfyem7tjye7ro3v2jirue2vipi4un56agm2du"

# Stable fail-closed case ids for reporting (order matters for assertions).
ABUSE_CASE_IDS: tuple[str, ...] = (
    "expired_ttl",
    "expired_absolute",
    "withdrawn_identity",
    "duplicate_identity_replace",
    "unsigned_when_signed_required",
    "partial_signature_block",
    "stale_publish_reject",
    "invalid_identity_shape",
    "selection_equal_health_tiebreak",
    "matching_ad_missing_delegation",
    "matching_ad_invalid_delegation",
    "registry_not_execution_authority",
)


class _Clock:
    def __init__(self, start: int = T0) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, delta_ms: int) -> None:
        self.now += delta_ms


def _signature(
    *,
    signer_did: str = "did:web:alpha",
    alg: str = "EdDSA",
    token: str = "dGVzdA",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "signer_did": signer_did,
        "signature_alg": alg,
        "signature": token,
    }
    if extra:
        body.update(extra)
    return body


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
        "proof_systems": list(proof_systems if proof_systems is not None else ["ucan"]),
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
    if signature is not None:
        body["signature"] = signature
    if extra:
        body.update(extra)
    return body


def _valid_chain(
    *,
    actor: str,
    resource: str = "tool.echo",
    ability: str = "invoke",
) -> List[Dict[str, Any]]:
    """Minimal root→leaf chain that authorizes ``actor`` for ``resource``."""

    return [
        {
            "issuer": "did:web:root",
            "audience": "did:web:mid",
            "capabilities": [{"resource": "*", "ability": ability}],
        },
        {
            "issuer": "did:web:mid",
            "audience": actor,
            "capabilities": [{"resource": resource, "ability": ability}],
        },
    ]


def authorize_dispatch_from_registry(
    registry: Any,
    *,
    resource: str,
    ability: str,
    actor: str,
    raw_chain: Optional[Sequence[Mapping[str, Any]]] = None,
    interface_cid: Optional[str] = None,
    semantic_capability: Optional[str] = None,
    policy_language: Optional[str] = None,
    proof_system: Optional[str] = None,
    now_ms: Optional[int] = None,
    require_signatures: bool = False,
    issuer_public_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fail-closed dispatch gate: registry match is necessary, UCAN is required.

    Finding a matching advertisement never grants execution authority. A valid
    UCAN delegation chain is required before dispatch may proceed.
    """

    selected = registry.select(
        interface_cid=interface_cid,
        semantic_capability=semantic_capability,
        policy_language=policy_language,
        proof_system=proof_system,
        now_ms=now_ms,
    )
    if selected is None:
        return {
            "allowed": False,
            "reason": "no_matching_advertisement",
            "advertisement": None,
            "ucan": None,
            "execution_authority": False,
            "interface": INTERFACE_LABEL,
        }

    # KD-14: registry membership is never execution authority.
    if is_execution_authority(selected):
        raise RegistryAuthorityError(
            "registry presence must never grant execution authority"
        )

    chain = list(raw_chain or [])
    if not chain:
        return {
            "allowed": False,
            "reason": "missing_delegation_chain",
            "advertisement": selected,
            "ucan": UcanValidationResult(
                False, "missing_delegation_chain", 0, [], None
            ).to_dict(),
            "execution_authority": False,
            "interface": INTERFACE_LABEL,
        }

    ucan = validate_raw_delegation_chain(
        raw_chain=chain,
        resource=resource,
        ability=ability,
        actor=actor,
        require_signatures=require_signatures,
        issuer_public_keys=issuer_public_keys,
    )
    return {
        "allowed": bool(ucan.allowed),
        "reason": ucan.reason if not ucan.allowed else "allowed",
        "advertisement": selected,
        "ucan": ucan.to_dict(),
        "execution_authority": False,
        "interface": INTERFACE_LABEL,
    }


# ---------------------------------------------------------------------------
# Contract / interface markers
# ---------------------------------------------------------------------------


class TestRegistryAbuseVectorContract:
    def test_interface_label_and_case_catalog(self) -> None:
        assert INTERFACE_LABEL == "RegistryAbuseVector@1"
        assert len(ABUSE_CASE_IDS) >= 10
        assert "matching_ad_missing_delegation" in ABUSE_CASE_IDS
        assert "unsigned_when_signed_required" in ABUSE_CASE_IDS
        assert REGISTRY_INTERFACE == "Registry@1"

    def test_static_and_libp2p_are_not_execution_authority(self) -> None:
        clock = _Clock()
        static = create_static_registry(clock_ms=clock)
        libp2p = create_libp2p_discovery(clock_ms=_Clock())
        ad = _ad("did:web:alpha")
        static.publish(ad)
        libp2p.publish(ad)
        for reg in (static, libp2p):
            assert is_execution_authority() is False
            assert is_execution_authority(ad) is False
            stats = reg.stats()
            assert stats["execution_authority"] is False
            assert stats["provider"] in {STATIC_PROVIDER_ID, LIBP2P_PROVIDER_ID}
        libp2p.close()


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


class TestRegistrationExpiry:
    def test_ttl_expiry_hides_record_from_lookup_and_select(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha", ttl_ms=5_000, published_at_ms=T0))
        assert reg.lookup_by_identity("did:web:alpha") is not None
        clock.advance(5_001)
        assert reg.lookup_by_identity("did:web:alpha") is None
        assert reg.lookup_by_interface_cid(IFACE_A) == []
        assert reg.select(interface_cid=IFACE_A) is None
        stale = reg.lookup_by_identity("did:web:alpha", include_stale=True)
        assert stale is not None
        assert stale["identity"]["did"] == "did:web:alpha"

    def test_absolute_expires_at_ms_overrides_ttl(self) -> None:
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
        clock.advance(2_001)
        assert reg.lookup_by_identity("did:web:alpha") is None
        assert reg.select(interface_cid=IFACE_A) is None

    def test_publish_rejects_already_expired_advertisement(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        with pytest.raises(RegistryStaleError):
            reg.publish(
                _ad(
                    "did:web:stale",
                    published_at_ms=T0 - 120_000,
                    ttl_ms=1_000,
                    expires_at_ms=T0 - 60_000,
                )
            )
        assert reg.stats()["stale_rejects"] >= 1
        assert reg.list_all() == []

    def test_expired_candidate_cannot_win_selection_over_fresh_peer(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(
            _ad(
                "did:web:old",
                ttl_ms=1_000,
                published_at_ms=T0,
                health="healthy",
                utilization=0,
            )
        )
        reg.publish(
            _ad(
                "did:web:fresh",
                ttl_ms=60_000,
                published_at_ms=T0,
                health="degraded",
                utilization=900_000,
            )
        )
        clock.advance(5_000)
        chosen = reg.select(interface_cid=IFACE_A)
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:fresh"


# ---------------------------------------------------------------------------
# Withdrawal
# ---------------------------------------------------------------------------


class TestRegistrationWithdrawal:
    def test_withdraw_removes_identity_and_is_idempotent(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha"))
        assert reg.withdraw("did:web:alpha") is True
        assert reg.lookup_by_identity("did:web:alpha") is None
        assert reg.select(interface_cid=IFACE_A) is None
        assert reg.withdraw("did:web:alpha") is False
        assert reg.stats()["withdraw_count"] == 1

    def test_withdrawn_identity_cannot_dispatch_even_with_valid_ucan(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        actor = "did:web:alpha"
        reg.publish(_ad(actor, skills=[{"id": "tool.echo", "tags": ["echo"]}]))
        reg.withdraw(actor)
        decision = authorize_dispatch_from_registry(
            reg,
            resource="tool.echo",
            ability="invoke",
            actor=actor,
            raw_chain=_valid_chain(actor=actor, resource="tool.echo"),
            interface_cid=IFACE_A,
        )
        assert decision["allowed"] is False
        assert decision["reason"] == "no_matching_advertisement"
        assert decision["execution_authority"] is False

    def test_libp2p_withdraw_clears_lookup(self) -> None:
        clock = _Clock()
        reg = create_libp2p_discovery(clock_ms=clock)
        reg.publish(_ad("did:web:alpha"))
        assert reg.withdraw("did:web:alpha") is True
        assert reg.lookup_by_identity("did:web:alpha") is None
        reg.close()


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


class TestDuplicateRegistration:
    def test_publish_replaces_same_identity_keeps_single_record(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha", interfaces=[IFACE_A], health="degraded"))
        reg.publish(
            _ad(
                "did:web:alpha",
                interfaces=[IFACE_B],
                health="healthy",
                utilization=10_000,
            )
        )
        all_ads = reg.list_all()
        assert len(all_ads) == 1
        found = reg.lookup_by_identity("did:web:alpha")
        assert found is not None
        assert found["interface_cids"] == [IFACE_B]
        assert found["health"]["status"] == "healthy"
        # Old interface no longer advertised after replace.
        assert reg.lookup_by_interface_cid(IFACE_A) == []
        assert {a["identity"]["did"] for a in reg.lookup_by_interface_cid(IFACE_B)} == {
            "did:web:alpha"
        }

    def test_duplicate_publish_count_increments_without_duplicating_identity(
        self,
    ) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        for _ in range(3):
            reg.publish(_ad("did:web:alpha", utilization=100_000))
        assert len(reg.list_all()) == 1
        assert reg.stats()["publish_count"] == 3

    def test_refresh_of_duplicate_identity_updates_health_for_selection(
        self,
    ) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(_ad("did:web:alpha", health="unhealthy", utilization=0))
        reg.publish(_ad("did:web:beta", health="healthy", utilization=500_000))
        assert reg.select(interface_cid=IFACE_A)["identity"]["did"] == "did:web:beta"
        reg.refresh(_ad("did:web:alpha", health="healthy", utilization=1_000))
        # Equal health: lower utilization wins (alpha after refresh).
        chosen = reg.select(interface_cid=IFACE_A)
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:alpha"


# ---------------------------------------------------------------------------
# Compromise / unsigned / structural abuse
# ---------------------------------------------------------------------------


class TestCompromisedAndUnsignedRecords:
    def test_require_signed_rejects_unsigned_advertisement(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(require_signed=True, clock_ms=clock)
        with pytest.raises(RegistryUnsignedError):
            reg.publish(_ad("did:web:alpha"))
        assert reg.stats()["unsigned_rejects"] >= 1
        assert reg.list_all() == []

    def test_partial_signature_block_is_rejected_as_unsigned(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(require_signed=True, clock_ms=clock)
        # Missing signature material — treated as compromised / unsigned.
        partial = _ad(
            "did:web:alpha",
            signature={"signer_did": "did:web:alpha", "signature_alg": "EdDSA"},
        )
        with pytest.raises(RegistryUnsignedError):
            reg.publish(partial)
        empty_sig = _ad(
            "did:web:beta",
            signature={
                "signer_did": "did:web:beta",
                "signature_alg": "EdDSA",
                "signature": "   ",
            },
        )
        with pytest.raises(RegistryUnsignedError):
            reg.publish(empty_sig)
        assert reg.stats()["unsigned_rejects"] >= 2

    def test_signed_shape_is_accepted_when_complete(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(require_signed=True, clock_ms=clock)
        stored = reg.publish(
            _ad("did:web:alpha", signature=_signature(signer_did="did:web:alpha"))
        )
        assert stored["signature"]["signature_alg"] == "EdDSA"
        assert reg.lookup_by_identity("did:web:alpha") is not None

    def test_invalid_identity_and_schema_are_rejected(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        with pytest.raises(RegistryValidationError):
            reg.publish(
                {
                    "schema": AGENT_ADVERTISEMENT_SCHEMA,
                    "identity": {"did": "not-a-did"},
                    "ttl_ms": 1000,
                    "interface_cids": [IFACE_A],
                }
            )
        with pytest.raises(RegistryValidationError):
            reg.publish(
                {
                    "schema": "mcp++/discovery/agent-advertisement@bogus",
                    "identity": {"did": "did:web:x"},
                    "ttl_ms": 1000,
                    "interface_cids": [IFACE_A],
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
        assert reg.list_all() == []

    def test_libp2p_require_signed_rejects_compromised_unsigned(self) -> None:
        clock = _Clock()
        reg = create_libp2p_discovery(require_signed=True, clock_ms=clock)
        with pytest.raises(RegistryUnsignedError):
            reg.publish(_ad("did:web:alpha"))
        assert reg.stats()["unsigned_rejects"] >= 1
        reg.close()


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------


class TestDeterministicSelection:
    def test_prefers_healthier_then_lower_load_then_lexicographic_did(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        # Insert reverse of expected order to prove sort, not insertion order.
        reg.publish(
            _ad("did:web:zeta", health="healthy", utilization=100_000, capacity=900_000)
        )
        reg.publish(
            _ad("did:web:mu", health="healthy", utilization=100_000, capacity=900_000)
        )
        reg.publish(
            _ad("did:web:alpha", health="healthy", utilization=100_000, capacity=900_000)
        )
        reg.publish(
            _ad("did:web:busy", health="healthy", utilization=800_000, capacity=200_000)
        )
        reg.publish(
            _ad("did:web:sick", health="degraded", utilization=0, capacity=1_000_000)
        )
        winners = [
            reg.select(interface_cid=IFACE_A)["identity"]["did"] for _ in range(5)
        ]
        assert winners == ["did:web:alpha"] * 5
        ads = reg.list_all()
        assert select_advertisement(ads)["identity"]["did"] == "did:web:alpha"
        # list_all is DID-ordered; selection_key is the independent total order.
        ordered_dids = [ad["identity"]["did"] for ad in sorted(ads, key=selection_key)]
        assert ordered_dids == [
            "did:web:alpha",
            "did:web:mu",
            "did:web:zeta",
            "did:web:busy",
            "did:web:sick",
        ]
        assert HEALTH_RANK["healthy"] < HEALTH_RANK["degraded"]

    def test_selection_filters_and_excludes_withdrawn_and_stale(self) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        reg.publish(
            _ad(
                "did:web:git",
                skills=[{"id": "repo.status", "tags": ["git"]}],
                policy_languages=["temporal-deontic@1"],
                proof_systems=["ucan"],
                health="healthy",
            )
        )
        reg.publish(
            _ad(
                "did:web:other",
                skills=[{"id": "weather.forecast", "tags": ["weather"]}],
                health="healthy",
            )
        )
        reg.publish(_ad("did:web:ephemeral", ttl_ms=1_000, published_at_ms=T0))
        clock.advance(5_000)
        chosen = reg.select(
            interface_cid=IFACE_A,
            semantic_capability="git",
            policy_language="temporal-deontic@1",
            proof_system="ucan",
        )
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:git"
        reg.withdraw("did:web:git")
        assert reg.select(semantic_capability="git") is None
        assert reg.select(semantic_capability="weather")["identity"]["did"] == (
            "did:web:other"
        )


# ---------------------------------------------------------------------------
# Dispatch authority: matching ad ≠ permission (UCAN required)
# ---------------------------------------------------------------------------


class TestMatchingAdvertisementDispatchAuthority:
    """Registry match is necessary for routing but never sufficient for dispatch."""

    def test_matching_advertisement_without_delegation_cannot_dispatch(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        actor = "did:web:worker"
        reg.publish(
            _ad(
                actor,
                skills=[{"id": "tool.echo", "tags": ["echo"]}],
                proof_systems=["ucan"],
                signature=_signature(signer_did=actor),
            )
        )
        match = reg.select(interface_cid=IFACE_A, semantic_capability="echo")
        assert match is not None
        assert match["identity"]["did"] == actor
        assert is_execution_authority(match) is False

        decision = authorize_dispatch_from_registry(
            reg,
            resource="tool.echo",
            ability="invoke",
            actor=actor,
            raw_chain=None,
            interface_cid=IFACE_A,
            semantic_capability="echo",
        )
        assert decision["allowed"] is False
        assert decision["reason"] == "missing_delegation_chain"
        assert decision["advertisement"]["identity"]["did"] == actor
        assert decision["execution_authority"] is False
        assert decision["ucan"]["allowed"] is False
        assert decision["interface"] == INTERFACE_LABEL

    def test_matching_advertisement_with_invalid_delegation_cannot_dispatch(
        self,
    ) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        actor = "did:web:worker"
        reg.publish(_ad(actor, skills=[{"id": "tool.echo"}]))
        bad_chain = [
            {
                "issuer": "did:web:root",
                "audience": "did:web:other",
                "capabilities": [{"resource": "tool.echo", "ability": "invoke"}],
            }
        ]
        decision = authorize_dispatch_from_registry(
            reg,
            resource="tool.echo",
            ability="invoke",
            actor=actor,
            raw_chain=bad_chain,
            interface_cid=IFACE_A,
        )
        assert decision["allowed"] is False
        assert decision["reason"] == "actor_mismatch"
        assert decision["advertisement"] is not None
        assert decision["ucan"]["allowed"] is False

    def test_matching_advertisement_with_valid_delegation_may_dispatch(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        actor = "did:web:worker"
        reg.publish(_ad(actor, skills=[{"id": "tool.echo", "tags": ["echo"]}]))
        decision = authorize_dispatch_from_registry(
            reg,
            resource="tool.echo",
            ability="invoke",
            actor=actor,
            raw_chain=_valid_chain(actor=actor, resource="tool.echo"),
            interface_cid=IFACE_A,
            semantic_capability="echo",
        )
        assert decision["allowed"] is True
        assert decision["reason"] == "allowed"
        assert decision["advertisement"]["identity"]["did"] == actor
        assert decision["ucan"]["allowed"] is True
        assert decision["execution_authority"] is False

    def test_expired_matching_advertisement_cannot_dispatch_even_with_ucan(
        self,
    ) -> None:
        clock = _Clock(T0)
        reg = StaticRegistry(clock_ms=clock)
        actor = "did:web:worker"
        reg.publish(_ad(actor, ttl_ms=1_000, published_at_ms=T0))
        clock.advance(10_000)
        decision = authorize_dispatch_from_registry(
            reg,
            resource="tool.echo",
            ability="invoke",
            actor=actor,
            raw_chain=_valid_chain(actor=actor),
            interface_cid=IFACE_A,
        )
        assert decision["allowed"] is False
        assert decision["reason"] == "no_matching_advertisement"

    def test_stats_and_selection_never_claim_execution_authority(self) -> None:
        clock = _Clock()
        reg = StaticRegistry(clock_ms=clock)
        ad = _ad("did:web:alpha", extra={"trust": {"domain": "prod"}})
        reg.publish(ad)
        selected = reg.select(interface_cid=IFACE_A)
        assert selected is not None
        assert is_execution_authority(selected) is False
        assert is_execution_authority(ad) is False
        assert reg.stats()["execution_authority"] is False
        # A client that treats registry membership as authority must fail closed.
        if is_execution_authority(selected):
            raise RegistryAuthorityError("unreachable")  # pragma: no cover
        with pytest.raises(RegistryAuthorityError):
            # Explicit adversarial path: code that asserts registry → execute.
            if not is_execution_authority(selected):
                raise RegistryAuthorityError(
                    "registry presence is not execution authority"
                )


# ---------------------------------------------------------------------------
# Cross-provider abuse matrix (static + libp2p)
# ---------------------------------------------------------------------------


class TestCrossProviderAbuseMatrix:
    @pytest.mark.parametrize("provider", ["static", "libp2p"])
    def test_expiry_withdraw_duplicate_compromise_selection_matrix(
        self, provider: str
    ) -> None:
        clock = _Clock(T0)
        if provider == "static":
            reg: Any = StaticRegistry(require_signed=True, clock_ms=clock)
        else:
            reg = create_libp2p_discovery(require_signed=True, clock_ms=clock)

        try:
            # Compromised unsigned rejected.
            with pytest.raises(RegistryUnsignedError):
                reg.publish(_ad("did:web:unsigned"))

            # Valid signed publish.
            reg.publish(
                _ad(
                    "did:web:alpha",
                    signature=_signature(signer_did="did:web:alpha"),
                    health="healthy",
                    utilization=100_000,
                )
            )
            # Duplicate identity replace.
            reg.publish(
                _ad(
                    "did:web:alpha",
                    signature=_signature(signer_did="did:web:alpha"),
                    health="healthy",
                    utilization=50_000,
                    interfaces=[IFACE_A, IFACE_B],
                )
            )
            reg.publish(
                _ad(
                    "did:web:beta",
                    signature=_signature(signer_did="did:web:beta"),
                    health="degraded",
                    utilization=0,
                )
            )
            chosen = reg.select(interface_cid=IFACE_A)
            assert chosen is not None
            assert chosen["identity"]["did"] == "did:web:alpha"
            assert IFACE_B in chosen["interface_cids"]

            # Withdraw beta; alpha remains selectable.
            assert reg.withdraw("did:web:beta") is True
            assert reg.lookup_by_identity("did:web:beta") is None

            # Expiry hides alpha.
            clock.advance(120_000)
            assert reg.select(interface_cid=IFACE_A) is None

            # Dispatch denied without matching ad, even with valid UCAN.
            decision = authorize_dispatch_from_registry(
                reg,
                resource="tool.echo",
                ability="invoke",
                actor="did:web:alpha",
                raw_chain=_valid_chain(actor="did:web:alpha"),
                interface_cid=IFACE_A,
            )
            assert decision["allowed"] is False
            assert decision["reason"] == "no_matching_advertisement"
            assert reg.stats()["execution_authority"] is False
        finally:
            close = getattr(reg, "close", None)
            if callable(close):
                close()

    def test_all_abuse_case_ids_are_exercised_symbolically(self) -> None:
        """Ensure the RegistryAbuseVector@1 catalog stays wired to live checks.

        Each case maps to a hermetic assertion above; this test guards the
        catalog against silent drift without re-emitting full envelopes.
        """

        # Map case id → a one-line live probe (True means fail-closed as expected).
        clock = _Clock(T0)
        reg = StaticRegistry(require_signed=True, clock_ms=clock)
        results: Dict[str, bool] = {}

        # expired_ttl
        reg_ttl = StaticRegistry(clock_ms=clock)
        reg_ttl.publish(_ad("did:web:ttl", ttl_ms=1_000, published_at_ms=T0))
        clock.advance(2_000)
        results["expired_ttl"] = reg_ttl.lookup_by_identity("did:web:ttl") is None

        # expired_absolute
        clock2 = _Clock(T0)
        reg_abs = StaticRegistry(clock_ms=clock2)
        reg_abs.publish(
            _ad("did:web:abs", expires_at_ms=T0 + 500, ttl_ms=60_000, published_at_ms=T0)
        )
        clock2.advance(501)
        results["expired_absolute"] = (
            reg_abs.lookup_by_identity("did:web:abs") is None
        )

        # withdrawn_identity
        clock3 = _Clock()
        reg_w = StaticRegistry(clock_ms=clock3)
        reg_w.publish(_ad("did:web:w"))
        reg_w.withdraw("did:web:w")
        results["withdrawn_identity"] = reg_w.lookup_by_identity("did:web:w") is None

        # duplicate_identity_replace
        reg_d = StaticRegistry(clock_ms=_Clock())
        reg_d.publish(_ad("did:web:d", interfaces=[IFACE_A]))
        reg_d.publish(_ad("did:web:d", interfaces=[IFACE_B]))
        results["duplicate_identity_replace"] = (
            len(reg_d.list_all()) == 1
            and reg_d.lookup_by_identity("did:web:d")["interface_cids"] == [IFACE_B]
        )

        # unsigned_when_signed_required
        try:
            reg.publish(_ad("did:web:u"))
            results["unsigned_when_signed_required"] = False
        except RegistryUnsignedError:
            results["unsigned_when_signed_required"] = True

        # partial_signature_block
        try:
            reg.publish(
                _ad(
                    "did:web:p",
                    signature={"signer_did": "did:web:p", "signature_alg": "EdDSA"},
                )
            )
            results["partial_signature_block"] = False
        except RegistryUnsignedError:
            results["partial_signature_block"] = True

        # stale_publish_reject
        try:
            StaticRegistry(clock_ms=_Clock(T0)).publish(
                _ad(
                    "did:web:s",
                    published_at_ms=T0 - 10_000,
                    ttl_ms=1_000,
                )
            )
            results["stale_publish_reject"] = False
        except RegistryStaleError:
            results["stale_publish_reject"] = True

        # invalid_identity_shape
        try:
            StaticRegistry(clock_ms=_Clock()).publish(
                {
                    "schema": AGENT_ADVERTISEMENT_SCHEMA,
                    "identity": {"did": "bad"},
                    "ttl_ms": 1000,
                    "interface_cids": [IFACE_A],
                }
            )
            results["invalid_identity_shape"] = False
        except RegistryValidationError:
            results["invalid_identity_shape"] = True

        # selection_equal_health_tiebreak
        reg_sel = StaticRegistry(clock_ms=_Clock())
        reg_sel.publish(
            _ad("did:web:zeta", health="healthy", utilization=100_000, capacity=900_000)
        )
        reg_sel.publish(
            _ad("did:web:alpha", health="healthy", utilization=100_000, capacity=900_000)
        )
        results["selection_equal_health_tiebreak"] = (
            reg_sel.select(interface_cid=IFACE_A)["identity"]["did"] == "did:web:alpha"
        )

        # matching_ad_missing_delegation
        reg_m = StaticRegistry(clock_ms=_Clock())
        reg_m.publish(_ad("did:web:worker"))
        d_missing = authorize_dispatch_from_registry(
            reg_m,
            resource="tool.echo",
            ability="invoke",
            actor="did:web:worker",
            raw_chain=None,
            interface_cid=IFACE_A,
        )
        results["matching_ad_missing_delegation"] = (
            d_missing["allowed"] is False
            and d_missing["reason"] == "missing_delegation_chain"
        )

        # matching_ad_invalid_delegation
        d_bad = authorize_dispatch_from_registry(
            reg_m,
            resource="tool.echo",
            ability="invoke",
            actor="did:web:worker",
            raw_chain=[
                {
                    "issuer": "did:web:root",
                    "audience": "did:web:other",
                    "capabilities": [{"resource": "tool.echo", "ability": "invoke"}],
                }
            ],
            interface_cid=IFACE_A,
        )
        results["matching_ad_invalid_delegation"] = d_bad["allowed"] is False

        # registry_not_execution_authority
        results["registry_not_execution_authority"] = (
            is_execution_authority(_ad("did:web:x")) is False
            and reg_m.stats()["execution_authority"] is False
        )

        missing = [cid for cid in ABUSE_CASE_IDS if not results.get(cid)]
        assert not missing, f"abuse cases not fail-closed: {missing}"
        assert set(results) == set(ABUSE_CASE_IDS)


# Explicit re-export for discovery by other suites (MCPP-063+ may reuse vectors).
__all__ = [
    "ABUSE_CASE_IDS",
    "INTERFACE_LABEL",
    "authorize_dispatch_from_registry",
]
