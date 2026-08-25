"""Deterministic tests for EAAEF-011 encrypted raw-export storage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

KIT_ROOT = Path(__file__).resolve().parents[1] / "ipfs_kit_py"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from ipfs_kit_py.external_agent_handoff.storage import (
    ENCRYPTION_ALGORITHM,
    EncryptedExportStore,
    HandoffStorageError,
    content_cid,
    digest_sha256,
    public_receipt_from_reference,
)

try:
    from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
        EncryptedExportReference,
        normalized_stream_identity,
    )
except ImportError:  # pragma: no cover - kit-only checkout
    EncryptedExportReference = None  # type: ignore[misc, assignment]
    normalized_stream_identity = None  # type: ignore[misc, assignment]


PLAINTEXT = b'{"session":"codex-export","messages":[{"role":"user","text":"secret transcript"}]}'
EVENT_A = "sha256:" + ("a" * 64)
EVENT_B = "sha256:" + ("b" * 64)
EVENT_C = "sha256:" + ("c" * 64)


@pytest.fixture
def store(tmp_path: Path) -> EncryptedExportStore:
    return EncryptedExportStore(tmp_path / "handoff-store")


@pytest.fixture
def master_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def test_store_roundtrip_preserves_exact_bytes(store: EncryptedExportStore, master_key: bytes) -> None:
    reference = store.store_raw_export(PLAINTEXT, master_key=master_key, media_type="application/json")
    recovered = store.load_raw_export(reference, master_key=master_key)
    assert recovered == PLAINTEXT
    assert reference["byte_count"] == len(PLAINTEXT)
    assert reference["digest_sha256"] == digest_sha256(PLAINTEXT, prefixed=True)
    assert reference["encryption_algorithm"] == ENCRYPTION_ALGORITHM
    assert reference["disclosure_class"] == "encrypted_raw"
    assert reference["ciphertext_cid"] != reference["key_envelope_cid"]
    assert reference["ciphertext_cid"][7:] != reference["digest_sha256"]


def test_ciphertext_is_not_plaintext(store: EncryptedExportStore, master_key: bytes) -> None:
    reference = store.store_raw_export(PLAINTEXT, master_key=master_key)
    blob = (store.ciphertext_dir / reference["ciphertext_cid"][7:]).read_bytes()
    assert PLAINTEXT not in blob
    assert b"secret transcript" not in blob
    assert content_cid(blob) == reference["ciphertext_cid"]


def test_public_receipt_omits_transcript_bodies(store: EncryptedExportStore, master_key: bytes) -> None:
    reference = store.store_raw_export(PLAINTEXT, master_key=master_key)
    receipt = store.public_receipt(reference, event_content_ids=(EVENT_A, EVENT_B))
    payload = dict(receipt.to_dict())
    encoded = str(payload)
    assert "secret transcript" not in encoded
    assert "transcript" not in encoded.lower() or "transcript" not in payload
    assert "body" not in payload
    assert "raw_bytes" not in payload
    assert payload["ciphertext_cid"] == reference["ciphertext_cid"]
    assert payload["key_envelope_cid"] == reference["key_envelope_cid"]
    assert payload["event_content_ids"] == [EVENT_A, EVENT_B]
    assert payload["disclosure_class"] == "public_projection"


def test_normalized_projection_is_order_sensitive(store: EncryptedExportStore) -> None:
    first = store.emit_normalized_projection((EVENT_A, EVENT_B, EVENT_C))
    second = store.emit_normalized_projection((EVENT_C, EVENT_B, EVENT_A))
    assert first.event_content_ids == (EVENT_A, EVENT_B, EVENT_C)
    assert first.stream_id != second.stream_id
    assert "secret" not in str(first.to_dict())
    stored = (store.projection_dir / first.stream_id[7:]).read_bytes()
    assert b"transcript" not in stored
    if normalized_stream_identity is not None:
        assert first.stream_id == normalized_stream_identity((EVENT_A, EVENT_B, EVENT_C))


def test_wrong_master_key_does_not_reveal_plaintext(
    store: EncryptedExportStore, master_key: bytes
) -> None:
    reference = store.store_raw_export(PLAINTEXT, master_key=master_key)
    with pytest.raises(HandoffStorageError, match="decryption failed"):
        store.load_raw_export(reference, master_key=AESGCM.generate_key(bit_length=256))


def test_empty_and_oversized_exports_are_rejected(
    store: EncryptedExportStore, master_key: bytes
) -> None:
    with pytest.raises(HandoffStorageError, match="nonempty"):
        store.store_raw_export(b"", master_key=master_key)
    with pytest.raises(HandoffStorageError, match="bound"):
        store.store_raw_export(b"x" * (1_048_576 + 1), master_key=master_key)


def test_public_receipt_rejects_transcript_fields() -> None:
    bogus = {
        "ciphertext_cid": EVENT_A,
        "digest_sha256": "1" * 64,
        "byte_count": 4,
        "key_envelope_cid": EVENT_B,
        "transcript_body": "nope",
    }
    with pytest.raises(HandoffStorageError, match="transcript"):
        public_receipt_from_reference(bogus, event_content_ids=(EVENT_C,))


def test_reference_is_encrypted_export_reference_compatible(
    store: EncryptedExportStore, master_key: bytes
) -> None:
    if EncryptedExportReference is None:
        pytest.skip("handoff contracts are not importable in this checkout")
    reference = store.store_raw_export(PLAINTEXT, master_key=master_key)
    decoded = EncryptedExportReference.from_dict(dict(reference))
    assert decoded.ciphertext_cid == reference["ciphertext_cid"]
    assert decoded.key_envelope_cid == reference["key_envelope_cid"]
    assert decoded.byte_count == len(PLAINTEXT)
    assert decoded.digest_sha256 == digest_sha256(PLAINTEXT, prefixed=True)
