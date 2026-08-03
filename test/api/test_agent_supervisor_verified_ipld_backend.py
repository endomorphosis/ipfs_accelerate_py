"""ASE-038: verified CIDv1 / IPLD / IPFS replication adapter tests.

Coverage:
- only strict CIDv1 objects enter coordination manifests
- fake / truncated / mismatched CIDs fail closed
- unsupported codecs and CAR fail closed
- ipfs_kit_py, Kubo, and cache roles are accurately reported
- degradation is explicit on selection receipts
- put/get rehash verification and codec substitution denial
- runtime-CAS / MCP++ IdentityLink bridges
"""

from __future__ import annotations

import hashlib
from unittest.mock import Mock, patch

import pytest

from ipfs_accelerate_py import ipfs_backend_router
from ipfs_accelerate_py.agent_supervisor.entrypoints.verified_ipld_backend import (
    BACKEND_CAPABILITY_RECEIPT_SCHEMA,
    BackendCapabilityReceipt,
    BackendRoleName,
    InMemoryConformantBackend,
    VerifiedIPLDBackend,
    VerifiedIPLDError,
    admit_cid,
    expected_cid_for_bytes,
    open_verified_ipld_backend,
    verify_bytes_match_cid,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    IdentityKind,
    MultiformatsIdentityError,
    cid_for_bytes,
    cid_for_dag_json,
    link_payload_digest,
    validate_cid,
)

KNOWN_HELLO_WORLD_RAW_CID = (
    "bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e"
)
KNOWN_EMPTY_RAW_CID = (
    "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_backend() -> InMemoryConformantBackend:
    return InMemoryConformantBackend()


@pytest.fixture
def verified(memory_backend: InMemoryConformantBackend) -> VerifiedIPLDBackend:
    return VerifiedIPLDBackend(backend=memory_backend)


@pytest.fixture
def clean_router(monkeypatch):
    """Isolate router selection state between tests."""
    for var in (
        "IPFS_BACKEND",
        "ENABLE_IPFS_KIT",
        "IPFS_KIT_DISABLE",
        "ENABLE_HF_CACHE",
        "KUBO_CMD",
    ):
        monkeypatch.delenv(var, raising=False)
    ipfs_backend_router._get_default_backend_cached.cache_clear()
    ipfs_backend_router._DEFAULT_BACKEND_OVERRIDE = None
    ipfs_backend_router._LAST_SELECTION_RECEIPT = None
    yield
    ipfs_backend_router._get_default_backend_cached.cache_clear()
    ipfs_backend_router._DEFAULT_BACKEND_OVERRIDE = None
    ipfs_backend_router._LAST_SELECTION_RECEIPT = None


class _MismatchBackend:
    """Returns a wrong but structurally valid CID (codec substitution)."""

    backend_name = "mismatch"
    backend_role = "memory"

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        # Deliberately return CID of different bytes (wrong content).
        wrong = cid_for_bytes(data + b"\x00", codec="raw")
        self._store[wrong] = data + b"\x00"
        return wrong

    def cat(self, cid: str) -> bytes:
        return self._store[cid]

    def pin(self, cid: str) -> None:
        return None

    def unpin(self, cid: str) -> None:
        return None

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        # Claim raw but emit dag-json CID for the same JSON-ish bytes when possible,
        # otherwise emit CID of different content.
        _ = codec
        wrong = cid_for_bytes(data + b"x", codec="raw")
        self._store[wrong] = data + b"x"
        return wrong

    def block_get(self, cid: str) -> bytes:
        return self._store[cid]

    def add_path(self, path: str, *, recursive: bool = True, pin: bool = True, chunker=None) -> str:
        raise NotImplementedError

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        raise NotImplementedError

    def ls(self, cid: str) -> list[str]:
        return []

    def dag_export(self, cid: str) -> bytes:
        raise RuntimeError("CAR not available")


class _FakeCidBackend:
    """Mimics HF-style synthetic bafy keys."""

    backend_name = "hf_cache"
    backend_role = ipfs_backend_router.ROLE_CACHE

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def _key(self, data: bytes) -> str:
        return "bafy" + hashlib.sha256(data).hexdigest()[:56]

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        key = self._key(data)
        self._store[key] = data
        return key

    def cat(self, cid: str) -> bytes:
        return self._store[cid]

    def pin(self, cid: str) -> None:
        return None

    def unpin(self, cid: str) -> None:
        return None

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        _ = codec
        return self.add_bytes(data)

    def block_get(self, cid: str) -> bytes:
        return self.cat(cid)

    def add_path(self, path: str, *, recursive: bool = True, pin: bool = True, chunker=None) -> str:
        raise NotImplementedError

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        raise NotImplementedError

    def ls(self, cid: str) -> list[str]:
        return []

    def dag_export(self, cid: str) -> bytes:
        raise RuntimeError("dag_export not available in HF cache backend")


# ---------------------------------------------------------------------------
# Admission / multiformats vectors
# ---------------------------------------------------------------------------


class TestCidAdmission:
    def test_known_raw_vectors_admitted(self) -> None:
        assert admit_cid(KNOWN_HELLO_WORLD_RAW_CID, codecs=("raw",)) == (
            KNOWN_HELLO_WORLD_RAW_CID
        )
        assert admit_cid(KNOWN_EMPTY_RAW_CID, codecs=("raw",)) == KNOWN_EMPTY_RAW_CID
        assert expected_cid_for_bytes(b"hello world") == KNOWN_HELLO_WORLD_RAW_CID

    def test_truncated_and_fake_cids_fail_closed(self) -> None:
        for bad in (
            "",
            "bafy",
            "bafytoo-short",
            "not-a-cid",
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",  # CIDv0
            "BAFKREIFZJUT3TE2NHYEKKLSS27NH3K72YSCO7Y32KOAO5EEI66WOF36N5E",
            "bafy" + "0" * 56,  # synthetic HF-style
        ):
            with pytest.raises(VerifiedIPLDError):
                admit_cid(bad, codecs=("raw",))

    def test_codec_mismatch_fails_closed(self) -> None:
        raw = cid_for_bytes(b'{"a":1}')
        dag = cid_for_dag_json({"a": 1})
        with pytest.raises(VerifiedIPLDError):
            admit_cid(raw, codecs=("dag-json",))
        with pytest.raises(VerifiedIPLDError):
            admit_cid(dag, codecs=("raw",))

    def test_unsupported_codec_fails_closed(self) -> None:
        with pytest.raises(VerifiedIPLDError, match="unsupported codec"):
            expected_cid_for_bytes(b"x", codec="dag-pb")  # type: ignore[arg-type]
        with pytest.raises(VerifiedIPLDError, match="codecs must"):
            admit_cid(KNOWN_HELLO_WORLD_RAW_CID, codecs=("dag-pb",))  # type: ignore[arg-type]

    def test_byte_mismatch_fails_closed(self) -> None:
        cid = cid_for_bytes(b"alpha")
        with pytest.raises(VerifiedIPLDError, match="mismatch"):
            verify_bytes_match_cid(b"beta", cid, codec="raw")


# ---------------------------------------------------------------------------
# Put / get verification
# ---------------------------------------------------------------------------


class TestVerifiedPutGet:
    def test_put_get_raw_round_trip(self, verified: VerifiedIPLDBackend) -> None:
        payload = b"coordination-epoch-bytes\n"
        put = verified.put_raw(payload)
        assert put.cid == cid_for_bytes(payload)
        assert put.codec == "raw"
        assert put.rehashed is True
        assert put.byte_length == len(payload)
        assert put.backend_role == BackendRoleName.MEMORY.value

        data, get = verified.get_raw(put.cid)
        assert data == payload
        assert get.cid == put.cid
        assert get.digest_hex == put.digest_hex

    def test_put_get_dag_json_round_trip(
        self, verified: VerifiedIPLDBackend
    ) -> None:
        obj = {"shard": "s1", "epoch": 3, "nested": {"z": 1, "a": 2}}
        put = verified.put_dag_json(obj)
        assert put.codec == "dag-json"
        assert put.cid == cid_for_dag_json(obj, for_identity=True)

        data, get = verified.get_dag_json(put.cid)
        assert get.cid == put.cid
        # Canonical bytes rehash.
        assert cid_for_bytes(data, codec="dag-json") == put.cid

    def test_codec_substitution_denied(self) -> None:
        backend = _MismatchBackend()
        v = VerifiedIPLDBackend(backend=backend)  # type: ignore[arg-type]
        with pytest.raises(VerifiedIPLDError, match="mismatch|non-admissible"):
            v.put_raw(b"strict-bytes")

    def test_manifest_admission_requires_payload_match(
        self, verified: VerifiedIPLDBackend
    ) -> None:
        payload = b"manifest-row"
        put = verified.put_raw(payload)
        adm = verified.admit_for_manifest(
            put.cid, codec="raw", payload=payload, purpose="epoch-manifest"
        )
        assert adm.cid == put.cid
        assert adm.purpose == "epoch-manifest"
        with pytest.raises(VerifiedIPLDError):
            verified.admit_for_manifest(
                put.cid, codec="raw", payload=b"tampered"
            )

    def test_get_rejects_tampered_store(
        self, memory_backend: InMemoryConformantBackend
    ) -> None:
        v = VerifiedIPLDBackend(backend=memory_backend)
        put = v.put_raw(b"original")
        # Corrupt underlying store while keeping the key.
        memory_backend._blocks[put.cid] = b"tampered"
        with pytest.raises(VerifiedIPLDError, match="mismatch"):
            v.get_raw(put.cid)


# ---------------------------------------------------------------------------
# Cache role / synthetic CID fail-closed
# ---------------------------------------------------------------------------


class TestCacheRoleFailClosed:
    def test_hf_cache_role_reported(self, temp_dir_factory=None) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            hf = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=tmp)
            assert ipfs_backend_router.classify_backend_role(hf) == (
                ipfs_backend_router.ROLE_CACHE
            )
            caps = ipfs_backend_router.describe_backend_capabilities(hf)
            assert caps.conformant_cid is False
            assert caps.supports_car is False
            assert caps.role == ipfs_backend_router.ROLE_CACHE

            # Synthetic key is not a multiformats CID.
            key = hf.add_bytes(b"hello world")
            assert key.startswith("bafy")
            with pytest.raises((MultiformatsIdentityError, VerifiedIPLDError)):
                validate_cid(key, codecs=("raw",))
            with pytest.raises(VerifiedIPLDError):
                admit_cid(key, codecs=("raw",))

    def test_verified_refuses_cache_backend_put(self) -> None:
        fake = _FakeCidBackend()
        v = VerifiedIPLDBackend(backend=fake)  # type: ignore[arg-type]
        cap = v.capabilities()
        assert cap.role == BackendRoleName.CACHE.value
        assert cap.conformant_cid is False
        assert cap.degraded is True
        with pytest.raises(VerifiedIPLDError, match="not conformant|cache"):
            v.put_raw(b"nope")
        with pytest.raises(VerifiedIPLDError, match="CAR export unsupported"):
            v.export_car(KNOWN_HELLO_WORLD_RAW_CID)

    def test_cache_capability_receipt_invariant(self) -> None:
        fake = _FakeCidBackend()
        v = VerifiedIPLDBackend(backend=fake)  # type: ignore[arg-type]
        receipt = v.capabilities()
        assert receipt.schema == BACKEND_CAPABILITY_RECEIPT_SCHEMA
        # Construction itself enforces cache invariants.
        with pytest.raises(VerifiedIPLDError):
            BackendCapabilityReceipt(
                schema=BACKEND_CAPABILITY_RECEIPT_SCHEMA,
                backend_name="hf_cache",
                role=BackendRoleName.CACHE.value,
                preferred_name="ipfs_kit",
                preferred_available=False,
                degraded=True,
                degradation_reasons=("cache",),
                conformant_cid=True,  # illegal for cache
                supports_raw=True,
                supports_dag_json=False,
                supports_car=False,
                supports_pin=True,
                codec_preservation_guaranteed=False,
                notes=(),
                candidate_order=(),
            )


# ---------------------------------------------------------------------------
# Role matrix / degradation
# ---------------------------------------------------------------------------


class TestBackendRolesAndDegradation:
    def test_ipfs_kit_role_matrix(self) -> None:
        with patch.object(
            ipfs_backend_router.IPFSKitBackend, "_init_storage", lambda self: None
        ):
            kit = ipfs_backend_router.IPFSKitBackend()
            kit._storage = Mock()
            assert (
                ipfs_backend_router.classify_backend_role(kit)
                == ipfs_backend_router.ROLE_IPFS_KIT
            )
            caps = ipfs_backend_router.describe_backend_capabilities(kit)
            assert caps.role == ipfs_backend_router.ROLE_IPFS_KIT
            assert caps.conformant_cid is True
            assert caps.supports_car is False
            assert caps.codec_preservation_guaranteed is False
            assert any("codec" in n.lower() for n in caps.notes)

            v = VerifiedIPLDBackend(backend=kit)
            receipt = v.capabilities()
            assert receipt.role == BackendRoleName.IPFS_KIT.value
            assert receipt.supports_car is False
            with pytest.raises(VerifiedIPLDError, match="CAR export unsupported"):
                v.export_car(KNOWN_HELLO_WORLD_RAW_CID)

    def test_kubo_role_matrix(self) -> None:
        kubo = ipfs_backend_router.KuboCLIBackend(cmd="ipfs")
        assert (
            ipfs_backend_router.classify_backend_role(kubo)
            == ipfs_backend_router.ROLE_KUBO
        )
        caps = ipfs_backend_router.describe_backend_capabilities(kubo)
        assert caps.role == ipfs_backend_router.ROLE_KUBO
        assert caps.supports_car is True
        assert caps.conformant_cid is True

        v = VerifiedIPLDBackend(backend=kubo)
        assert v.capabilities().role == BackendRoleName.KUBO.value

    def test_selection_degradation_to_cache_is_explicit(
        self, clean_router, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("IPFS_KIT_DISABLE", "1")
        monkeypatch.setenv("ENABLE_HF_CACHE", "true")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        # Force kit unavailable; HF should win before Kubo.
        backend, receipt = ipfs_backend_router.select_backend()
        assert isinstance(backend, ipfs_backend_router.HuggingFaceCacheBackend)
        assert receipt.degraded is True
        assert receipt.selected_role == ipfs_backend_router.ROLE_CACHE
        assert receipt.preferred_available is False
        assert receipt.degradation_reasons
        assert any(
            "cache" in r.lower() or "ipfs_kit" in r.lower() or "disable" in r.lower()
            for r in receipt.degradation_reasons
        )

        v = VerifiedIPLDBackend(backend=backend, selection=receipt)
        cap = v.capabilities()
        assert cap.degraded is True
        assert cap.role == BackendRoleName.CACHE.value
        assert cap.conformant_cid is False

    def test_memory_role_is_conformant(
        self, memory_backend: InMemoryConformantBackend
    ) -> None:
        assert (
            ipfs_backend_router.classify_backend_role(memory_backend)
            == ipfs_backend_router.ROLE_MEMORY
        )
        caps = ipfs_backend_router.describe_backend_capabilities(memory_backend)
        assert caps.conformant_cid is True
        assert caps.supports_car is True
        assert caps.supports_dag_json is True

    def test_open_factory(self, memory_backend: InMemoryConformantBackend) -> None:
        v = open_verified_ipld_backend(backend=memory_backend)
        put = v.put_raw(b"via-factory")
        assert put.cid == cid_for_bytes(b"via-factory")


# ---------------------------------------------------------------------------
# CAR gating
# ---------------------------------------------------------------------------


class TestCarCapabilityGate:
    def test_memory_car_export_succeeds(
        self, verified: VerifiedIPLDBackend
    ) -> None:
        put = verified.put_raw(b"car-payload")
        car = verified.export_car(put.cid)
        assert isinstance(car, bytes)
        assert put.cid.encode("utf-8") in car
        assert b"car-payload" in car

    def test_car_rejects_bad_cid(self, verified: VerifiedIPLDBackend) -> None:
        with pytest.raises(VerifiedIPLDError):
            verified.export_car("bafy-not-real")

    def test_kit_dag_export_fails_closed(self) -> None:
        with patch.object(
            ipfs_backend_router.IPFSKitBackend, "_init_storage", lambda self: None
        ):
            kit = ipfs_backend_router.IPFSKitBackend()
            kit._storage = Mock()
            with pytest.raises(RuntimeError, match="dag_export|CAR"):
                kit.dag_export(KNOWN_HELLO_WORLD_RAW_CID)


# ---------------------------------------------------------------------------
# Identity bridges
# ---------------------------------------------------------------------------


class TestIdentityBridges:
    def test_runtime_cas_link(self, verified: VerifiedIPLDBackend) -> None:
        payload = b'{"runtime":true}'
        digest = hashlib.sha256(payload).hexdigest()
        artifact_id = f"runtime-artifact:sha256:{digest}"
        link = verified.link_runtime_cas(
            artifact_id, payload_bytes=payload, codec="raw"
        )
        assert link.kind == IdentityKind.RUNTIME_ARTIFACT.value
        assert link.local_id == artifact_id
        assert link.cid == cid_for_bytes(payload)
        assert link.local_id != link.cid
        # Local id is never rewritten to the CID.
        assert "runtime-artifact:" in link.to_dict()["local_id"]

    def test_mcp_payload_digest_link(self, verified: VerifiedIPLDBackend) -> None:
        payload = b"mcp-compaction"
        digest = hashlib.sha256(payload).hexdigest()
        pd = f"sha256:{digest}"
        link = verified.link_mcp_payload_digest(pd)
        assert link.kind == IdentityKind.PAYLOAD_DIGEST.value
        assert link.local_id == pd
        assert link.cid == cid_for_bytes(payload)
        # Same bridge as multiformats helper.
        assert link.cid == link_payload_digest(pd).cid

    def test_bridge_rejects_bad_artifact_id(
        self, verified: VerifiedIPLDBackend
    ) -> None:
        with pytest.raises(VerifiedIPLDError):
            verified.link_runtime_cas("not-an-artifact")


# ---------------------------------------------------------------------------
# Router helpers surface
# ---------------------------------------------------------------------------


class TestRouterSurface:
    def test_get_backend_with_receipt(self, clean_router, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("IPFS_KIT_DISABLE", "1")
        monkeypatch.setenv("ENABLE_HF_CACHE", "true")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        backend, receipt = ipfs_backend_router.get_backend_with_receipt()
        assert backend is not None
        assert isinstance(receipt, ipfs_backend_router.BackendSelectionReceipt)
        assert receipt.degraded is True
        d = receipt.to_dict()
        assert "degradation_reasons" in d
        assert "capabilities" in d
        assert d["capabilities"]["role"] in {
            ipfs_backend_router.ROLE_CACHE,
            ipfs_backend_router.ROLE_KUBO,
            ipfs_backend_router.ROLE_IPFS_KIT,
        }

    def test_describe_roles_are_distinct(self) -> None:
        roles = set()
        with patch.object(
            ipfs_backend_router.IPFSKitBackend, "_init_storage", lambda self: None
        ):
            kit = ipfs_backend_router.IPFSKitBackend()
            kit._storage = Mock()
            roles.add(ipfs_backend_router.classify_backend_role(kit))
        roles.add(
            ipfs_backend_router.classify_backend_role(
                ipfs_backend_router.KuboCLIBackend()
            )
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            roles.add(
                ipfs_backend_router.classify_backend_role(
                    ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=tmp)
                )
            )
        assert roles == {
            ipfs_backend_router.ROLE_IPFS_KIT,
            ipfs_backend_router.ROLE_KUBO,
            ipfs_backend_router.ROLE_CACHE,
        }
