"""ASE-038: strict CIDv1 / IPLD / IPFS verified replication adapter tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import ipfs_backend_router
from ipfs_accelerate_py.agent_supervisor.entrypoints.verified_ipld_backend import (
    BackendCapabilityReceipt,
    BackendRoleName,
    VerifiedIPLDBackend,
    VerifiedIPLDError,
    build_verified_ipld_backend,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    IdentityKind,
    MultiformatsIdentityError,
    cid_for_bytes,
    cid_for_dag_json,
    validate_cid,
)

KNOWN_EMPTY_RAW_CID = (
    "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
)
KNOWN_HELLO_WORLD_RAW_CID = (
    "bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e"
)


class ConformantMemoryBackend:
    """In-memory transport that returns real CIDv1 for raw bytes."""

    BACKEND_NAME = "memory_conformant"
    BACKEND_ROLE = ipfs_backend_router.BackendRole.KUBO

    def __init__(self) -> None:
        self.blocks: dict[str, bytes] = {}
        self.pinned: set[str] = set()
        self.car_enabled = True

    def capability_descriptor(self) -> ipfs_backend_router.BackendCapabilityDescriptor:
        return ipfs_backend_router.BackendCapabilityDescriptor(
            name=self.BACKEND_NAME,
            role=self.BACKEND_ROLE,
            synthetic_identifiers=False,
            admits_strict_cid=True,
            supports_raw=True,
            supports_dag_json=True,
            supports_car=self.car_enabled,
            preserves_requested_codec=True,
            pin_supported=True,
        )

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        cid = cid_for_bytes(data, codec="raw")
        self.blocks[cid] = data
        if pin:
            self.pinned.add(cid)
        return cid

    def cat(self, cid: str) -> bytes:
        if cid not in self.blocks:
            raise RuntimeError(f"CID not found: {cid}")
        return self.blocks[cid]

    def pin(self, cid: str) -> None:
        self.pinned.add(cid)

    def unpin(self, cid: str) -> None:
        self.pinned.discard(cid)

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        if codec not in {"raw", "dag-json"}:
            raise RuntimeError(f"unsupported codec {codec}")
        cid = cid_for_bytes(data, codec=codec)
        self.blocks[cid] = data
        return cid

    def block_get(self, cid: str) -> bytes:
        return self.cat(cid)

    def add_path(self, path: str, **kwargs: Any) -> str:
        return self.add_bytes(Path(path).read_bytes())

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        Path(output_path).write_bytes(self.cat(cid))

    def ls(self, cid: str) -> list[str]:
        return []

    def dag_export(self, cid: str) -> bytes:
        if not self.car_enabled:
            raise RuntimeError("dag_export disabled")
        data = self.cat(cid)
        # Minimal non-empty stand-in; real CAR decoding is out of scope.
        return b"\x0acar-v1\n" + data


class MismatchedCodecBackend(ConformantMemoryBackend):
    """Returns a valid CID but with the wrong codec for the same digest."""

    BACKEND_NAME = "mismatched_codec"

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        # Always mint raw CID even when dag-json was requested.
        cid = cid_for_bytes(data, codec="raw")
        self.blocks[cid] = data
        return cid

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        return self.block_put(data, codec="raw")


class FakeCidBackend:
    """Transport that always returns a truncated / fake identifier."""

    BACKEND_NAME = "fake"
    BACKEND_ROLE = ipfs_backend_router.BackendRole.UNKNOWN

    def __init__(self) -> None:
        self.blocks: dict[str, bytes] = {}

    def capability_descriptor(self) -> ipfs_backend_router.BackendCapabilityDescriptor:
        # Claims to be a conformant Kubo-like transport while returning
        # truncated identifiers — the adapter must fail closed on put.
        return ipfs_backend_router.BackendCapabilityDescriptor(
            name=self.BACKEND_NAME,
            role=ipfs_backend_router.BackendRole.KUBO,
            synthetic_identifiers=False,
            admits_strict_cid=True,
            supports_raw=True,
            supports_dag_json=False,
            supports_car=False,
            preserves_requested_codec=True,
            pin_supported=False,
            degraded=False,
            degradation_reasons=(),
        )

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        handle = "bafkrei"  # truncated
        self.blocks[handle] = data
        return handle

    def cat(self, cid: str) -> bytes:
        return self.blocks[cid]

    def pin(self, cid: str) -> None:
        return None

    def unpin(self, cid: str) -> None:
        return None

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        return self.add_bytes(data)

    def block_get(self, cid: str) -> bytes:
        return self.cat(cid)

    def add_path(self, path: str, **kwargs: Any) -> str:
        return self.add_bytes(Path(path).read_bytes())

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        Path(output_path).write_bytes(self.cat(cid))

    def ls(self, cid: str) -> list[str]:
        return []

    def dag_export(self, cid: str) -> bytes:
        raise RuntimeError("dag_export not available")


# ---------------------------------------------------------------------------
# Multiformats vector + put/get verification
# ---------------------------------------------------------------------------


def test_known_raw_vectors_through_verified_backend() -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    empty = backend.put_raw(b"")
    hello = backend.put_raw(b"hello world")
    assert empty.cid == KNOWN_EMPTY_RAW_CID
    assert hello.cid == KNOWN_HELLO_WORLD_RAW_CID
    assert backend.get_verified(empty.cid) == b""
    assert backend.get_verified(hello.cid) == b"hello world"
    assert empty.transport_matched_cid is True


def test_dag_json_put_get_round_trip_and_codec_separation() -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    obj = {"z": 2, "a": 1, "unicode": "café"}
    put = backend.put_dag_json(obj)
    assert put.codec == "dag-json"
    assert put.cid == cid_for_dag_json(obj, for_identity=True)
    assert put.cid != cid_for_bytes(
        # raw of non-canonical text must not equal dag-json CID
        b'{"z":2,"a":1}',
        codec="raw",
    )
    restored = backend.get_dag_json(put.cid)
    assert restored == {"a": 1, "unicode": "café", "z": 2}
    with pytest.raises(VerifiedIPLDError):
        backend.get_verified(put.cid, codec="raw")


def test_rehash_detects_transport_tamper() -> None:
    transport = ConformantMemoryBackend()
    backend = VerifiedIPLDBackend(transport=transport)
    put = backend.put_raw(b"original-payload")
    # Tamper after verified put.
    transport.blocks[put.cid] = b"tampered-payload"
    backend._local_blocks.clear()
    with pytest.raises(VerifiedIPLDError, match="rehash"):
        backend.get_verified(put.cid)


# ---------------------------------------------------------------------------
# Fail-closed: fake / truncated / mismatched / unsupported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "bafkrei",
        "bafy" + "0" * 56,  # HF-style synthetic cache key shape
        "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",  # CIDv0
        "BAFKREIHDWDCEFGH4DQKJV67UZCMW7OJEE6XEDZDETOJUZJEVTENXQUVYKU",  # upper
        "not-a-cid",
    ],
)
def test_fake_truncated_and_non_cidv1_fail_closed_for_manifest(bad: str) -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    with pytest.raises(VerifiedIPLDError):
        backend.admit_cid_for_manifest(bad)


def test_hf_cache_synthetic_identifier_never_enters_manifest(tmp_path: Path) -> None:
    hf = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=str(tmp_path))
    synthetic = hf.add_bytes(b"cache-payload")
    assert synthetic.startswith("bafy")
    backend = VerifiedIPLDBackend(transport=hf)
    # Synthetic key is not a strict CIDv1.
    with pytest.raises(VerifiedIPLDError):
        backend.admit_cid_for_manifest(synthetic)
    # Verified put returns the local strict CID, not the cache key.
    put = backend.put_raw(b"cache-payload")
    assert put.cid == cid_for_bytes(b"cache-payload")
    assert put.cid != synthetic
    assert put.transport_matched_cid is False
    assert put.transport_handle == synthetic
    admitted = backend.admit_cid_for_manifest(put.cid, codecs=("raw",))
    assert admitted == put.cid
    assert backend.get_verified(put.cid) == b"cache-payload"


def test_mismatched_backend_cid_fails_for_conformant_transport() -> None:
    transport = MismatchedCodecBackend()
    backend = VerifiedIPLDBackend(transport=transport)
    obj = {"manifest": True, "n": 1}
    # dag-json expected CID differs from raw CID of the same encoded bytes.
    with pytest.raises(VerifiedIPLDError, match="mismatched|expected"):
        backend.put_dag_json(obj)


def test_fake_transport_claiming_strict_cid_fails_closed() -> None:
    backend = VerifiedIPLDBackend(transport=FakeCidBackend())
    with pytest.raises(VerifiedIPLDError, match="expected|mismatched|strict"):
        backend.put_raw(b"data")


def test_unsupported_codec_fails_closed() -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    with pytest.raises(VerifiedIPLDError, match="unsupported codec"):
        backend.expected_cid_for_bytes(b"x", codec="dag-pb")
    with pytest.raises(VerifiedIPLDError, match="unsupported codec"):
        backend._put_bytes(b"x", codec="dag-cbor", pin=False)
    with pytest.raises(VerifiedIPLDError):
        backend.admit_cid_for_manifest(KNOWN_EMPTY_RAW_CID, codecs=("dag-pb",))


def test_car_export_capability_gated() -> None:
    transport = ConformantMemoryBackend()
    backend = VerifiedIPLDBackend(transport=transport)
    put = backend.put_dag_json({"car": True})
    car = backend.export_car(put.cid, codec="dag-json")
    assert car.startswith(b"\x0acar-v1\n")

    transport.car_enabled = False
    # Descriptor still claims supports_car=True from cached descriptor method —
    # force a transport that reports no CAR.
    no_car = ConformantMemoryBackend()
    no_car.car_enabled = False

    class _NoCar(ConformantMemoryBackend):
        def capability_descriptor(self):
            base = super().capability_descriptor()
            return ipfs_backend_router.BackendCapabilityDescriptor(
                name=base.name,
                role=base.role,
                synthetic_identifiers=False,
                admits_strict_cid=True,
                supports_raw=True,
                supports_dag_json=True,
                supports_car=False,
                preserves_requested_codec=True,
                pin_supported=True,
            )

    gated = VerifiedIPLDBackend(transport=_NoCar())
    root = gated.put_raw(b"root")
    with pytest.raises(VerifiedIPLDError, match="CAR export is capability-gated"):
        gated.export_car(root.cid, codec="raw")


def test_car_export_fails_closed_on_cache_role(tmp_path: Path) -> None:
    hf = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=str(tmp_path))
    backend = VerifiedIPLDBackend(transport=hf)
    put = backend.put_raw(b"no-car")
    with pytest.raises(VerifiedIPLDError, match="CAR export is capability-gated"):
        backend.export_car(put.cid, codec="raw")


# ---------------------------------------------------------------------------
# Role reporting and degradation
# ---------------------------------------------------------------------------


def test_capability_receipt_reports_ipfs_kit_kubo_and_cache_roles(
    tmp_path: Path,
) -> None:
    hf = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=str(tmp_path))
    cache_backend = VerifiedIPLDBackend(transport=hf)
    cache_receipt = cache_backend.capability_receipt()
    assert cache_receipt.role == BackendRoleName.CACHE.value
    assert cache_receipt.synthetic_identifiers is True
    assert cache_receipt.admits_coordination_manifest is False
    assert cache_receipt.degraded is True
    assert any("cache" in r for r in cache_receipt.degradation_reasons)
    assert cache_receipt.supports_car is False

    kubo_like = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    kubo_receipt = kubo_like.capability_receipt()
    assert kubo_receipt.role == BackendRoleName.KUBO.value
    assert kubo_receipt.admits_coordination_manifest is True
    assert kubo_receipt.supports_car is True

    # Descriptor-level classification for kit without full kit init.
    kit_desc = ipfs_backend_router.BackendCapabilityDescriptor(
        name="ipfs_kit",
        role=ipfs_backend_router.BackendRole.IPFS_KIT,
        synthetic_identifiers=False,
        admits_strict_cid=True,
        supports_raw=True,
        supports_dag_json=False,
        supports_car=False,
        preserves_requested_codec=False,
        pin_supported=True,
        notes=("codec not assumed",),
    )

    class _KitShim(ConformantMemoryBackend):
        BACKEND_NAME = "ipfs_kit"
        BACKEND_ROLE = ipfs_backend_router.BackendRole.IPFS_KIT

        def capability_descriptor(self):
            return kit_desc

    kit_backend = VerifiedIPLDBackend(transport=_KitShim())
    kit_receipt = kit_backend.capability_receipt()
    assert kit_receipt.role == BackendRoleName.IPFS_KIT.value
    assert kit_receipt.preserves_requested_codec is False
    assert kit_receipt.supports_car is False
    assert any("codec" in r for r in kit_receipt.degradation_reasons)


def test_router_describe_backend_roles(tmp_path: Path) -> None:
    hf = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=str(tmp_path))
    desc = ipfs_backend_router.describe_backend(hf)
    assert desc.role is ipfs_backend_router.BackendRole.CACHE
    assert desc.synthetic_identifiers is True
    assert desc.admits_strict_cid is False
    assert desc.degraded is True

    kubo = ipfs_backend_router.KuboCLIBackend(cmd="ipfs")
    kdesc = ipfs_backend_router.describe_backend(kubo)
    assert kdesc.role is ipfs_backend_router.BackendRole.KUBO
    assert kdesc.supports_car is True
    assert kdesc.admits_strict_cid is True


def test_router_selection_receipt_marks_cache_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IPFS_KIT_DISABLE", "1")
    monkeypatch.setenv("ENABLE_HF_CACHE", "true")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("IPFS_BACKEND", raising=False)
    ipfs_backend_router._get_default_backend_cached.cache_clear()
    ipfs_backend_router._DEFAULT_BACKEND_OVERRIDE = None

    backend, receipt = ipfs_backend_router.select_backend_with_receipt()
    assert isinstance(backend, ipfs_backend_router.HuggingFaceCacheBackend)
    assert receipt.selected_role is ipfs_backend_router.BackendRole.CACHE
    assert receipt.degraded is True
    assert receipt.degradation_reasons
    assert any(
        "ipfs_kit" in r or "cache" in r for r in receipt.degradation_reasons
    )


def test_capability_receipt_content_id_stable() -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    left = backend.capability_receipt()
    right = BackendCapabilityReceipt.from_dict(left.to_dict())
    assert left.content_id == right.content_id
    assert left.to_dict()["content_id"] == left.content_id


# ---------------------------------------------------------------------------
# Identity links: runtime-CAS and MCP++ digests
# ---------------------------------------------------------------------------


def test_runtime_cas_and_mcp_links_are_explicit_dual_identity() -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    payload = b'{"runtime":true,"n":7}'
    digest_hex = hashlib.sha256(payload).hexdigest()
    artifact_id = f"runtime-artifact:sha256:{digest_hex}"
    link = backend.link_runtime_cas(
        artifact_id,
        payload_bytes=payload,
        payload_digest=f"sha256:{digest_hex}",
    )
    assert link.kind == IdentityKind.RUNTIME_ARTIFACT.value
    assert link.local_id == artifact_id
    assert link.cid == cid_for_bytes(payload)
    assert link.local_id != link.cid

    mcp_link = backend.link_mcp_compaction_hash(f"sha256:{digest_hex}")
    assert mcp_link.kind == IdentityKind.PAYLOAD_DIGEST.value
    assert mcp_link.local_id == f"sha256:{digest_hex}"
    assert mcp_link.cid == link.cid
    # Bare hex also accepted and normalized.
    bare = backend.link_mcp_compaction_hash(digest_hex)
    assert bare.cid == mcp_link.cid
    assert bare.local_id.startswith("sha256:")


def test_identity_link_not_coordination_authority_without_admission() -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    digest_hex = hashlib.sha256(b"x").hexdigest()
    link = backend.link_mcp_compaction_hash(f"sha256:{digest_hex}")
    # The digest string itself is not a CID.
    with pytest.raises(VerifiedIPLDError):
        backend.admit_cid_for_manifest(link.local_id)
    # The linked CID is admissible once verified as form-valid.
    assert backend.admit_cid_for_manifest(link.cid, codecs=("raw",)) == link.cid


def test_manifest_list_rejects_duplicates_and_bad_entries() -> None:
    backend = VerifiedIPLDBackend(transport=ConformantMemoryBackend())
    a = backend.put_raw(b"a").cid
    b = backend.put_raw(b"b").cid
    assert backend.admit_cid_list_for_manifest([a, b]) == (a, b)
    with pytest.raises(VerifiedIPLDError, match="duplicate"):
        backend.admit_cid_list_for_manifest([a, a])
    with pytest.raises(VerifiedIPLDError):
        backend.admit_cid_list_for_manifest([a, "bafkrei"])


def test_build_verified_ipld_backend_with_explicit_transport() -> None:
    transport = ConformantMemoryBackend()
    backend = build_verified_ipld_backend(transport=transport)
    assert isinstance(backend, VerifiedIPLDBackend)
    assert backend.put_raw(b"z").cid == cid_for_bytes(b"z")


def test_hf_cache_descriptor_matches_router_role(tmp_path: Path) -> None:
    """Regression: router must not claim HF is strict IPFS."""
    hf = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=str(tmp_path))
    assert hf.BACKEND_ROLE is ipfs_backend_router.BackendRole.CACHE
    desc = hf.capability_descriptor()
    assert desc.role is ipfs_backend_router.BackendRole.CACHE
    assert desc.admits_strict_cid is False
    assert desc.supports_car is False
    # Existing transport behaviour still returns a synthetic key.
    key = hf.add_bytes(b"hello")
    assert key.startswith("bafy")
    with pytest.raises((MultiformatsIdentityError, VerifiedIPLDError)):
        validate_cid(key)
