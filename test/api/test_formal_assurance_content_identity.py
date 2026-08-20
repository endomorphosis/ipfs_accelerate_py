"""FACP-026: Accelerate pseudo-CID paths → canonical content identity.

Acceptance coverage:
- Raw SHA-256 hex and truncated Qm-like values fail (integrity.unchecked).
- Canonical values decode and recompute against retained bytes (digest_valid).
- One-bit content or identifier mutation fails with the expected integrity state.
"""

from __future__ import annotations

import hashlib
import types
from typing import Any

import pytest

from ipfs_accelerate_py.assurance.content_identity import (
    BUNDLE,
    EVIDENCE_ID,
    GOAL_ID,
    INVENTORIED_PSEUDO_CID_SITES,
    INTERFACE,
    TASK_ID,
    CanonicalIPFSMultiformats,
    ContentIdentityError,
    IdentityErrorCode,
    Integrity,
    classify_pseudo_cid,
    flip_one_bit,
    get_cid,
    install_canonical_multiformats,
    is_qm_like,
    is_raw_sha256_hex,
    legacy_pseudo_cid,
    mint_content_identity,
    mutate_cid_one_bit,
    reject_pseudo_cid,
    validate_canonical_cid,
    verify_content_identity,
    verify_or_raise,
)


def test_module_exports_facp_026_contract() -> None:
    assert TASK_ID == "FACP-026"
    assert GOAL_ID == "FACP-G220"
    assert BUNDLE == "facp/migration/accelerate-cid"
    assert EVIDENCE_ID == "facp/canonical-cid@1"
    assert INTERFACE == "AccelerateContentIdentity@1"
    assert INVENTORIED_PSEUDO_CID_SITES
    assert INVENTORIED_PSEUDO_CID_SITES[0]["symbol"] == "MockIPFSMultiformats.get_cid"


def test_raw_sha256_hex_is_rejected_as_pseudo_cid() -> None:
    payload = "hello"
    hex_digest = legacy_pseudo_cid(payload)
    assert is_raw_sha256_hex(hex_digest)
    assert len(hex_digest) == 64
    assert classify_pseudo_cid(hex_digest) is IdentityErrorCode.PSEUDO_CID_RAW_HEX

    with pytest.raises(ContentIdentityError) as exc_info:
        reject_pseudo_cid(hex_digest)
    assert exc_info.value.code is IdentityErrorCode.PSEUDO_CID_RAW_HEX
    assert exc_info.value.integrity is Integrity.UNCHECKED

    with pytest.raises(ContentIdentityError) as exc_info:
        validate_canonical_cid(hex_digest)
    assert exc_info.value.integrity is Integrity.UNCHECKED

    result = verify_content_identity(hex_digest, payload)
    assert result.ok is False
    assert result.integrity == Integrity.UNCHECKED.value
    assert result.code == IdentityErrorCode.PSEUDO_CID_RAW_HEX.value


@pytest.mark.parametrize(
    "value,code",
    [
        ("sha256:" + "ab" * 32, IdentityErrorCode.PSEUDO_CID_LABELED),
        ("CID:" + "cd" * 32, IdentityErrorCode.PSEUDO_CID_LABELED),
        ("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG", IdentityErrorCode.PSEUDO_CID_QM_FORM),
        ("QmYwAP", IdentityErrorCode.PSEUDO_CID_TRUNCATED),
        ("qm truncated", IdentityErrorCode.PSEUDO_CID_TRUNCATED),
        ("bafkreih", IdentityErrorCode.PSEUDO_CID_TRUNCATED),
    ],
)
def test_labeled_qm_like_and_truncated_forms_fail(
    value: str, code: IdentityErrorCode
) -> None:
    if value.startswith(("Qm", "qm")):
        assert is_qm_like(value.split()[0] if " " in value else value) or value.startswith(
            ("Qm", "qm")
        )
    assert classify_pseudo_cid(value) is code
    with pytest.raises(ContentIdentityError) as exc_info:
        reject_pseudo_cid(value)
    assert exc_info.value.code is code
    assert exc_info.value.integrity is Integrity.UNCHECKED

    result = verify_content_identity(value, b"hello")
    assert result.ok is False
    assert result.integrity == Integrity.UNCHECKED.value
    assert result.code == code.value


def test_canonical_cid_decodes_and_recomputes_against_bytes() -> None:
    payload = "hello"
    identity = mint_content_identity(payload)

    assert identity.cid.startswith("b")
    assert identity.cid == identity.cid.lower()
    assert not is_raw_sha256_hex(identity.cid)
    assert not is_qm_like(identity.cid)
    assert identity.integrity == Integrity.DIGEST_VALID.value
    assert identity.digest_hex == hashlib.sha256(b"hello").hexdigest()
    assert identity.canonical_bytes == b"hello"
    assert identity.codec == "raw"

    validated = validate_canonical_cid(identity.cid, codecs=("raw",))
    assert validated == identity.cid

    verified = verify_content_identity(identity.cid, payload)
    assert verified.ok is True
    assert verified.integrity == Integrity.DIGEST_VALID.value
    assert verified.cid == identity.cid
    assert verified.recomputed_cid == identity.cid
    assert verified.digest_hex == identity.digest_hex

    rebuilt = verify_or_raise(identity.cid, b"hello")
    assert rebuilt.cid == identity.cid
    assert rebuilt.to_dict()["integrity"] == Integrity.DIGEST_VALID.value


def test_get_cid_never_returns_legacy_hex() -> None:
    payload = {"batch": "item-1", "n": 2}
    cid = get_cid(payload)
    legacy = legacy_pseudo_cid(payload)

    assert cid.startswith("b")
    assert cid != legacy
    assert len(legacy) == 64
    assert is_raw_sha256_hex(legacy)
    assert not is_raw_sha256_hex(cid)

    # Structured payloads use dag-json under the closed profile.
    identity = mint_content_identity(payload)
    assert identity.codec == "dag-json"
    assert verify_content_identity(cid, payload).ok is True


def test_one_bit_content_mutation_fails_with_unchecked_integrity() -> None:
    payload = b"accelerate-identity-vector"
    identity = mint_content_identity(payload)
    mutated = flip_one_bit(identity.canonical_bytes, bit_index=0)
    assert mutated != identity.canonical_bytes
    assert hashlib.sha256(mutated).hexdigest() != identity.digest_hex

    result = verify_content_identity(identity.cid, mutated)
    assert result.ok is False
    assert result.integrity == Integrity.UNCHECKED.value
    assert result.code == IdentityErrorCode.DIGEST_MISMATCH.value

    with pytest.raises(ContentIdentityError) as exc_info:
        verify_or_raise(identity.cid, mutated)
    assert exc_info.value.integrity is Integrity.UNCHECKED
    assert exc_info.value.code is IdentityErrorCode.DIGEST_MISMATCH


def test_one_bit_identifier_mutation_fails_with_unchecked_integrity() -> None:
    payload = b"identifier-mutation"
    identity = mint_content_identity(payload)
    mutated_cid = mutate_cid_one_bit(identity.cid)
    assert mutated_cid != identity.cid

    result = verify_content_identity(mutated_cid, payload)
    assert result.ok is False
    assert result.integrity == Integrity.UNCHECKED.value
    assert result.code in {
        IdentityErrorCode.CID_DECODE_FAILED.value,
        IdentityErrorCode.DIGEST_MISMATCH.value,
        IdentityErrorCode.PSEUDO_CID_TRUNCATED.value,
    }

    with pytest.raises(ContentIdentityError) as exc_info:
        verify_or_raise(mutated_cid, payload)
    assert exc_info.value.integrity is Integrity.UNCHECKED


def test_canonical_multiformats_replaces_mock_get_cid() -> None:
    mock_hex = legacy_pseudo_cid("batch-item")
    adapter = CanonicalIPFSMultiformats()
    cid = adapter.get_cid("batch-item")

    assert cid.startswith("b")
    assert cid != mock_hex
    assert verify_content_identity(cid, "batch-item").ok is True
    assert verify_content_identity(mock_hex, "batch-item").ok is False
    assert verify_content_identity(mock_hex, "batch-item").integrity == (
        Integrity.UNCHECKED.value
    )


def test_install_canonical_multiformats_migrates_owner_binding() -> None:
    owner = types.SimpleNamespace(
        ipfs_multiformats=types.SimpleNamespace(
            get_cid=lambda data: legacy_pseudo_cid(data)
        ),
        resources={},
    )
    # Precondition: inventoried mock shape yields raw hex.
    assert is_raw_sha256_hex(owner.ipfs_multiformats.get_cid("hello"))

    installed = install_canonical_multiformats(owner)
    assert isinstance(installed, CanonicalIPFSMultiformats)
    assert owner.ipfs_multiformats is installed
    assert owner.resources["ipfs_multiformats"] is installed

    cid = owner.ipfs_multiformats.get_cid("hello")
    assert not is_raw_sha256_hex(cid)
    assert verify_content_identity(cid, "hello").integrity == Integrity.DIGEST_VALID.value


def test_dag_json_order_independence_and_byte_recompute() -> None:
    left = mint_content_identity({"z": 1, "a": {"y": 2, "x": 3}})
    right = mint_content_identity({"a": {"x": 3, "y": 2}, "z": 1})
    assert left.cid == right.cid
    assert left.canonical_bytes == right.canonical_bytes
    assert left.digest_hex == right.digest_hex
    assert verify_content_identity(left.cid, right.canonical_bytes, codec="dag-json").ok


def test_empty_and_non_string_identifiers_fail_closed() -> None:
    assert classify_pseudo_cid("") is IdentityErrorCode.EMPTY_IDENTIFIER
    assert classify_pseudo_cid(None) is IdentityErrorCode.EMPTY_IDENTIFIER
    with pytest.raises(ContentIdentityError) as exc_info:
        reject_pseudo_cid("   ")
    assert exc_info.value.code is IdentityErrorCode.EMPTY_IDENTIFIER
    assert exc_info.value.integrity is Integrity.UNCHECKED

    result = verify_content_identity("", b"x")
    assert result.ok is False
    assert result.integrity == Integrity.UNCHECKED.value


def test_verification_result_dict_carries_evidence_metadata() -> None:
    identity = mint_content_identity(b"meta")
    payload = verify_content_identity(identity.cid, b"meta").to_dict()
    assert payload["ok"] is True
    assert payload["integrity"] == Integrity.DIGEST_VALID.value
    assert payload["evidence_id"] == EVIDENCE_ID
    assert payload["task_id"] == TASK_ID
    assert payload["cid"] == identity.cid
