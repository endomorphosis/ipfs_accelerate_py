from __future__ import annotations

import socket
import time

import pytest
from hypothesis import given, settings, strategies as st

from ipfs_accelerate_py.model_catalog.identity import content_cid
from ipfs_accelerate_py.model_catalog.schema import CatalogSnapshot
from ipfs_accelerate_py.model_catalog.security import (
    AdvertisementVerificationError,
    AdvertisementVerifier,
    AuthorizationPolicyError,
    CapabilityGrant,
    CatalogAuthorizationPolicy,
    CatalogCapability,
    CatalogInputPolicy,
    InputPolicyError,
    ReplayCache,
    SecurityPolicyError,
    URLPolicy,
    URLPolicyError,
)
from ipfs_accelerate_py.mcplusplus_module.service_registry import (
    CATALOG_ENDPOINT_PROTOCOL,
    ServiceRecord,
    ServiceRegistry,
)

NOW = 1_000.0
TRUSTED_KEY = b"offline-trusted-issuer-key"
PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def _forbid_live_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("security tests must not use a live network")

    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _record(
    *,
    peer_id: str = "peer-a",
    key: bytes = TRUSTED_KEY,
    issued_at: float = NOW,
    expires_at: float = NOW + 300.0,
    cid: str | None = None,
) -> ServiceRecord:
    revision = cid or content_cid({"schema_version": "1.0", "records": []})
    record = ServiceRecord(
        service_name="ipfs-accelerate-mcp",
        peer_id=peer_id,
        issuer=peer_id,
        multiaddrs=["/memory/%s" % peer_id],
        catalog_cid=revision,
        catalog_revision=revision,
        operation_summary=["text.generate"],
        interface_cids=["cidv1-ai-catalog"],
        endpoint_protocol=CATALOG_ENDPOINT_PROTOCOL,
        issued_at=issued_at,
        expires_at=expires_at,
        metadata={"server": "offline"},
    )
    record.sign(key)
    return record


def _verifier(**kwargs) -> AdvertisementVerifier:
    return AdvertisementVerifier(
        {"peer-a": TRUSTED_KEY},
        clock=lambda: NOW,
        replay_cache=ReplayCache(),
        **kwargs,
    )


def _public_policy(**kwargs) -> URLPolicy:
    values = {
        "allowed_hosts": ("example.test", "*.example.test"),
        "resolver": lambda _host: (PUBLIC_IP,),
    }
    values.update(kwargs)
    return URLPolicy(**values)


def test_signature_verification_accepts_complete_trusted_advertisement():
    record = _record()

    assert _verifier().verify(record) is record
    payload = record.to_dict()
    assert payload["schema_version"] == "1.0"
    assert payload["nonce"]
    assert payload["signature_algorithm"] == "hmac-sha256"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("issuer", "peer-b"),
        ("service_name", "other-service"),
        ("service_id", "service_" + "0" * 64),
        ("catalog_cid", content_cid({"other": 1})),
        ("catalog_revision", content_cid({"other": 2})),
        ("operation_summary", ["audio.transcribe"]),
        ("interface_cids", ["other-interface"]),
        ("endpoint_protocol", "/mcp+p2p/catalog/2.0.0"),
        ("issued_at", NOW - 1),
        ("expires_at", NOW + 299),
        ("nonce", "A" * 32),
    ],
)
def test_signature_covers_every_security_relevant_field(field_name, replacement):
    record = _record()
    setattr(record, field_name, replacement)

    with pytest.raises(AdvertisementVerificationError):
        _verifier().verify(record)


def test_unsigned_untrusted_and_wrong_key_advertisements_fail_closed():
    unsigned = _record()
    unsigned.signature = None
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier().verify(unsigned)
    assert exc.value.code == "signature_invalid"

    with pytest.raises(AdvertisementVerificationError) as exc:
        AdvertisementVerifier({}, clock=lambda: NOW).verify(_record())
    assert exc.value.code == "issuer_untrusted"

    with pytest.raises(AdvertisementVerificationError) as exc:
        AdvertisementVerifier(
            {"peer-a": b"wrong-key"}, clock=lambda: NOW
        ).verify(_record())
    assert exc.value.code == "signature_invalid"


def test_schema_cid_and_service_identity_are_checked_before_admission():
    wrong_schema = _record()
    wrong_schema.schema_version = "2.0"
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier().verify(wrong_schema)
    assert exc.value.code == "schema_unsupported"

    wrong_cid = _record()
    wrong_cid.catalog_revision = content_cid({"different": True})
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier().verify(wrong_cid)
    assert exc.value.code == "catalog_cid_invalid"

    wrong_identity = _record()
    wrong_identity.service_id = "service_" + "f" * 64
    wrong_identity.sign(TRUSTED_KEY)
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier().verify(wrong_identity)
    assert exc.value.code == "service_identity_invalid"


def test_clock_skew_expiry_lifetime_staleness_and_replay_are_distinct():
    future = _record(issued_at=NOW + 31, expires_at=NOW + 331)
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier().verify(future)
    assert exc.value.code == "issued_in_future"

    expired = _record(issued_at=NOW - 300, expires_at=NOW)
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier().verify(expired)
    assert exc.value.code == "expired"

    overlong = _record(expires_at=NOW + 601)
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier().verify(overlong)
    assert exc.value.code == "lifetime_invalid"

    stale = _record(issued_at=NOW - 200, expires_at=NOW + 50)
    with pytest.raises(AdvertisementVerificationError) as exc:
        _verifier(replay_window=100, max_lifetime=300).verify(stale)
    assert exc.value.code == "stale"

    exact = _record()
    verifier = _verifier()
    verifier.verify(exact)
    with pytest.raises(AdvertisementVerificationError) as exc:
        verifier.verify(ServiceRecord.from_dict(exact.to_dict()))
    assert exc.value.code == "replayed"


def test_registry_rejects_untrusted_and_replayed_records_before_insertion():
    live_now = time.time()
    live_record = _record(issued_at=live_now, expires_at=live_now + 300)
    strict = ServiceRegistry(trusted_issuers={})
    rejected = strict.handle_announce(
        {"record": live_record.to_dict()}, sender_peer_id="peer-a"
    )
    assert rejected == {"status": "rejected", "reason": "issuer_untrusted"}
    assert strict.get_services() == []

    registry = ServiceRegistry(trusted_issuers={"peer-a": TRUSTED_KEY})
    record = _record(issued_at=live_now, expires_at=live_now + 300)
    accepted = registry.handle_announce(
        {"record": record.to_dict()}, sender_peer_id="peer-a"
    )
    replayed = registry.handle_announce(
        {"record": record.to_dict()}, sender_peer_id="peer-a"
    )
    assert accepted["status"] == "accepted"
    assert replayed == {"status": "rejected", "reason": "replayed"}
    assert len(registry.get_services()) == 1


def test_strict_registry_cannot_publish_peer_id_hmac_local_catalogs():
    live_now = time.time()
    empty = CatalogSnapshot()
    record = ServiceRecord(
        service_name="ipfs-accelerate-mcp",
        peer_id="peer-a",
        issuer="peer-a",
        multiaddrs=["/memory/peer-a"],
        catalog_cid=empty.revision,
        catalog_revision=empty.revision,
        operation_summary=["text.generate"],
        interface_cids=["cidv1-ai-catalog"],
        endpoint_protocol=CATALOG_ENDPOINT_PROTOCOL,
        issued_at=live_now,
        expires_at=live_now + 300,
    )
    record.sign(TRUSTED_KEY)

    receive_only = ServiceRegistry(trusted_issuers={"peer-a": TRUSTED_KEY})
    # register_local with a provider immediately refreshes/signs.
    with pytest.raises(RuntimeError, match="local_signing_key"):
        receive_only.register_local(record, catalog_provider=lambda: empty)

    publisher = ServiceRegistry(
        trusted_issuers={"peer-a": TRUSTED_KEY},
        local_signing_key=TRUSTED_KEY,
    )
    publisher.register_local(
        ServiceRecord.from_dict(record.to_dict()),
        catalog_provider=lambda: empty,
    )
    published = publisher.get_local(record.service_name)
    assert published is not None
    assert published.verify_signature(TRUSTED_KEY)
    assert not published.verify_signature()


def test_partial_catalog_advertisements_and_replay_cache_pressure_fail_closed():
    live_now = time.time()
    payload = _record(
        issued_at=live_now, expires_at=live_now + 300
    ).to_dict()
    payload["catalog_cid"] = None

    result = ServiceRegistry().handle_announce(
        {"record": payload}, sender_peer_id="peer-a"
    )
    assert result == {
        "status": "rejected",
        "reason": "invalid_catalog_advertisement",
    }

    cache = ReplayCache(max_entries=1)
    assert cache.consume("issuer", "nonce-a", 20.0, 10.0)
    assert not cache.consume("issuer", "nonce-b", 20.0, 10.0)
    assert cache.consume("issuer", "nonce-b", 30.0, 21.0)


def test_exact_capabilities_do_not_bleed_across_control_plane_actions():
    resource = "catalog:service_" + "a" * 64
    actor = "did:key:reader"
    grants = (
        CapabilityGrant(resource, CatalogCapability.READ.value),
        CapabilityGrant(resource, CatalogCapability.REMOTE_REFRESH.value),
        CapabilityGrant(resource, CatalogCapability.HEALTH_PROBE.value),
        CapabilityGrant(resource, CatalogCapability.invoke("text.generate")),
    )
    policy = CatalogAuthorizationPolicy({actor: grants})

    for grant in grants:
        policy.require(actor, grant.resource, grant.ability)
    assert not policy.is_authorized(
        actor, resource, CatalogCapability.invoke("audio.transcribe")
    )
    assert not policy.is_authorized(actor, resource, "catalog.invoke")
    assert not policy.is_authorized("did:key:other", resource, grants[0].ability)
    with pytest.raises(AuthorizationPolicyError) as exc:
        policy.require(
            actor, resource, CatalogCapability.invoke("embedding.generate")
        )
    assert exc.value.code == "capability_denied"


def test_registry_catalog_read_uses_exact_actor_and_resource_grant():
    snapshot = CatalogSnapshot()
    record = _record(cid=snapshot.revision)
    resource = "catalog:%s" % record.service_id
    policy = CatalogAuthorizationPolicy(
        {
            "reader": (
                CapabilityGrant(resource, CatalogCapability.READ.value),
            )
        }
    )
    registry = ServiceRegistry(authorization_policy=policy)
    registry.register_local(record, catalog_provider=lambda: snapshot)
    params = {
        "service_name": record.service_name,
        "catalog_revision": snapshot.revision,
        "record_type": "providers",
        "cursor": None,
        "limit": 1,
    }

    with pytest.raises(AuthorizationPolicyError):
        registry.handle_catalog_page(params, sender_peer_id="other")
    page = registry.handle_catalog_page(params, sender_peer_id="reader")
    assert page["items"] == []


def test_url_policy_normalizes_only_allowlisted_public_destinations():
    policy = _public_policy()

    assert policy.validate("https://EXAMPLE.test/path") == (
        "https://example.test/path"
    )
    assert policy.validate("https://api.example.test") == (
        "https://api.example.test/"
    )

    denied = (
        "http://example.test/",
        "https://other.test/",
        "https://user:pass@example.test/",
        "https://example.test:444/",
        "https://example.test/#fragment",
        "https://example.test/?token=secret",
        "file:///etc/passwd",
        "https://2130706433/",
        "https://127.1/",
        "https://[::1]/",
    )
    for url in denied:
        with pytest.raises(URLPolicyError):
            policy.validate(url)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
    ],
)
def test_dns_answers_reject_loopback_link_local_and_private_ranges(address):
    policy = URLPolicy(
        allowed_hosts=("example.test",),
        resolver=lambda _host: (address,),
    )

    with pytest.raises(URLPolicyError) as exc:
        policy.validate("https://example.test/")
    assert exc.value.code == "address_denied"


def test_dns_rebinding_and_redirect_hops_are_revalidated():
    rebinding = URLPolicy(
        allowed_hosts=("example.test",),
        resolver=lambda _host: (PUBLIC_IP, "127.0.0.1"),
    )
    with pytest.raises(URLPolicyError):
        rebinding.validate("https://example.test/")

    policy = _public_policy()
    assert policy.validate_redirect(
        "https://example.test/start",
        "/next",
        redirect_count=1,
    ) == "https://example.test/next"
    with pytest.raises(URLPolicyError) as exc:
        policy.validate_redirect(
            "https://example.test/start",
            "https://api.example.test/next",
            redirect_count=1,
        )
    assert exc.value.code == "redirect_denied"
    with pytest.raises(URLPolicyError) as exc:
        policy.validate_redirect(
            "https://example.test/start",
            "/loop",
            redirect_count=policy.max_redirects + 1,
        )
    assert exc.value.code == "redirect_denied"


def test_url_policy_never_performs_ambient_dns():
    policy = URLPolicy(
        allowed_hosts=("example.test",),
        resolver=None,
        require_dns_resolution=True,
    )

    with pytest.raises(URLPolicyError) as exc:
        policy.validate("https://example.test/")
    assert exc.value.code == "dns_required"


def test_input_policy_rejects_recursive_oversized_malformed_secret_and_ssrf_data():
    policy = CatalogInputPolicy()

    recursive = []
    recursive.append(recursive)
    cases = (
        (recursive, "input_recursive"),
        ({"value": "x" * 9_000}, "input_oversized"),
        ({1: "not-a-text-key"}, "input_malformed"),
        ({"authorization": "Bearer very-secret-token-value"}, "secret_input"),
        ({"endpoint_url": "http://169.254.169.254/latest"}, "ssrf_input"),
        ({"value": object()}, "input_malformed"),
    )
    for value, code in cases:
        with pytest.raises(InputPolicyError) as exc:
            policy.validate_record(value)
        assert exc.value.code == code
        assert len(str(exc.value).encode("utf-8")) <= 192


def test_media_page_and_diagnostic_contracts_are_separate_and_bounded():
    policy = CatalogInputPolicy()

    assert policy.validate_media({"content": b"offline media"})["content"]
    with pytest.raises(InputPolicyError) as exc:
        policy.validate_record({"content": b"not allowed in records"})
    assert exc.value.code == "input_oversized"
    with pytest.raises(InputPolicyError) as exc:
        policy.validate_diagnostic({"message": "https://127.0.0.1/private"})
    assert exc.value.code == "ssrf_input"
    assert policy.validate_page({"items": [{"name": "bounded"}]})


_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(1 << 63) + 1, max_value=(1 << 63) - 1),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=128),
)
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=8),
        st.dictionaries(
            st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cs",),
                    blacklist_characters="\x00",
                ),
                min_size=1,
                max_size=32,
            ),
            children,
            max_size=8,
        ),
    ),
    max_leaves=32,
)


@settings(max_examples=100, deadline=None)
@given(_json_values)
def test_arbitrary_structured_inputs_are_accepted_or_fail_with_bounded_errors(value):
    try:
        CatalogInputPolicy().validate_record(value)
    except SecurityPolicyError as exc:
        assert exc.code
        assert len(str(exc).encode("utf-8")) <= 192
