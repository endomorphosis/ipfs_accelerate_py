"""PTR-012: independent cross-package ContentIdentity@1 vectors.

These vectors deliberately retain literal canonical bytes, SHA-256 digests,
and CIDs.  Each provider must independently reach those literals; agreement
obtained by round-tripping a value from one provider is not sufficient.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import pytest
from ipfs_datasets_py.utils.cid_utils import (
    canonical_dag_json_bytes as datasets_canonical_dag_json_bytes,
)
from ipfs_datasets_py.utils.cid_utils import (
    cid_for_bytes as datasets_cid_for_bytes,
)
from ipfs_datasets_py.utils.cid_utils import (
    cid_for_dag_json as datasets_cid_for_dag_json,
)
from ipfs_datasets_py.utils.cid_utils import (
    validate_cid as datasets_validate_cid,
)
from multiformats import CID, multihash

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    CID_BASE,
    CID_CODEC,
    CID_VERSION,
    CONTENT_IDENTITY_INTERFACE,
    DIGEST_SIZE,
    MH_TYPE,
    ContentIdentity,
    TestExecutionIdentityError,
    mint_content_identity,
    reject_pseudo_cid,
)


@dataclass(frozen=True)
class IdentityVector:
    """A fixed external expectation, not output captured from an implementation."""

    name: str
    payload: Mapping[str, Any]
    canonical_bytes: bytes
    digest_hex: str
    cid: str


KNOWN_VECTORS = (
    IdentityVector(
        name="nested-ascii",
        # Deliberately not insertion-sorted.
        payload={
            "values": [True, None, 3],
            "nested": {"z": 2, "a": 1},
            "name": "ascii",
            "interface": "ContentIdentity@1",
        },
        canonical_bytes=(
            b'{"interface":"ContentIdentity@1","name":"ascii",'
            b'"nested":{"a":1,"z":2},"values":[true,null,3]}'
        ),
        digest_hex=(
            "ac163a0ec8c38368846bf4491ce35ab4"
            "3e2f17c88d14a3a56a409ed77b5c4d3c"
        ),
        cid="baguqeeravqldudwiyobwrbdl6rerzy22wq7c6f6irukkhjlkicpno624ju6a",
    ),
    IdentityVector(
        name="utf8-unicode",
        payload={
            "text": "café λ",
            "name": "unicode",
            "interface": "ContentIdentity@1",
            "values": [False, 0],
        },
        canonical_bytes=(
            b'{"interface":"ContentIdentity@1","name":"unicode",'
            b'"text":"caf\xc3\xa9 \xce\xbb","values":[false,0]}'
        ),
        digest_hex=(
            "fd5d4485ed3fb993501b229f71603689"
            "9108540d4e5a407bdefe74184df08d65"
        ),
        cid="baguqeera7voujbpnh64zgua3ekpxcybwrgiqqvanjznea6667z2bqtpqrvsq",
    ),
)


class ContradictionKind(str, Enum):
    """Typed reason that a claimed CID contradicts retained identity bytes."""

    VERSION = "version"
    BASE = "base"
    CODEC = "codec"
    MULTIHASH = "multihash"
    DIGEST = "digest"
    MALFORMED_MULTIHASH = "malformed_multihash"


class CIDIdentityContradiction(AssertionError):
    """Independent oracle failure with a stable, inspectable contradiction kind."""

    def __init__(self, kind: ContradictionKind, detail: str) -> None:
        self.kind = kind
        super().__init__(f"{kind.value} contradiction: {detail}")


def _require_frozen_claim(cid: str, retained_bytes: bytes) -> CID:
    """Decode and compare a claim without calling supervisor or dataset helpers."""

    try:
        decoded = CID.decode(cid)
    except (KeyError, TypeError, ValueError) as exc:
        raise CIDIdentityContradiction(
            ContradictionKind.MALFORMED_MULTIHASH,
            "CID or its multihash cannot be decoded",
        ) from exc

    if decoded.version != CID_VERSION:
        raise CIDIdentityContradiction(
            ContradictionKind.VERSION,
            f"expected {CID_VERSION}, got {decoded.version}",
        )
    if decoded.base.name != CID_BASE:
        raise CIDIdentityContradiction(
            ContradictionKind.BASE,
            f"expected {CID_BASE}, got {decoded.base.name}",
        )
    if decoded.codec.name != CID_CODEC:
        raise CIDIdentityContradiction(
            ContradictionKind.CODEC,
            f"expected {CID_CODEC}, got {decoded.codec.name}",
        )
    if decoded.hashfun.name != MH_TYPE or len(decoded.raw_digest) != DIGEST_SIZE:
        raise CIDIdentityContradiction(
            ContradictionKind.MULTIHASH,
            "expected a 32-byte sha2-256 multihash",
        )

    expected_digest = hashlib.sha256(retained_bytes).digest()
    if bytes(decoded.raw_digest) != expected_digest:
        raise CIDIdentityContradiction(
            ContradictionKind.DIGEST,
            "decoded digest does not hash the retained bytes",
        )
    return decoded


@pytest.mark.parametrize("vector", KNOWN_VECTORS, ids=lambda vector: vector.name)
def test_known_vectors_match_three_independent_providers(
    vector: IdentityVector,
) -> None:
    """Supervisor, datasets, and direct multiformats must match fixed literals."""

    supervisor_identity = mint_content_identity(vector.payload)
    datasets_bytes = datasets_canonical_dag_json_bytes(vector.payload)
    datasets_cid_from_bytes = datasets_cid_for_bytes(
        datasets_bytes,
        base=CID_BASE,
        codec=CID_CODEC,
        mh_type=MH_TYPE,
        version=CID_VERSION,
    )
    datasets_cid_from_value = datasets_cid_for_dag_json(
        vector.payload,
        base=CID_BASE,
        mh_type=MH_TYPE,
        version=CID_VERSION,
    )
    direct_digest = hashlib.sha256(vector.canonical_bytes).digest()
    direct_cid = str(
        CID(
            CID_BASE,
            CID_VERSION,
            CID_CODEC,
            multihash.digest(vector.canonical_bytes, MH_TYPE),
        )
    )

    assert datasets_bytes == vector.canonical_bytes
    assert supervisor_identity.canonical_bytes == vector.canonical_bytes
    assert direct_digest.hex() == vector.digest_hex
    assert supervisor_identity.digest_hex == vector.digest_hex
    assert datasets_cid_from_bytes == vector.cid
    assert datasets_cid_from_value == vector.cid
    assert direct_cid == vector.cid
    assert supervisor_identity.cid == vector.cid

    decoded = _require_frozen_claim(vector.cid, vector.canonical_bytes)
    assert decoded.version == 1
    assert decoded.base.name == "base32"
    assert decoded.codec.name == "dag-json"
    assert decoded.hashfun.name == "sha2-256"
    assert bytes(multihash.unwrap(decoded.digest)).hex() == vector.digest_hex
    assert (
        datasets_validate_cid(
            vector.cid,
            codecs=("dag-json",),
            mh_type="sha2-256",
            version=1,
            base="base32",
        )
        == vector.cid
    )
    assert supervisor_identity.interface == CONTENT_IDENTITY_INTERFACE
    assert supervisor_identity.verify() is supervisor_identity


def _contradictory_cids(vector: IdentityVector) -> tuple[tuple[str, ContradictionKind], ...]:
    retained_multihash = multihash.digest(vector.canonical_bytes, MH_TYPE)
    return (
        (
            str(CID("base58btc", 0, "dag-pb", retained_multihash)),
            ContradictionKind.VERSION,
        ),
        (
            str(CID("base58btc", CID_VERSION, CID_CODEC, retained_multihash)),
            ContradictionKind.BASE,
        ),
        (
            str(CID(CID_BASE, CID_VERSION, "raw", retained_multihash)),
            ContradictionKind.CODEC,
        ),
        (
            str(
                CID(
                    CID_BASE,
                    CID_VERSION,
                    CID_CODEC,
                    multihash.digest(vector.canonical_bytes + b"\n", MH_TYPE),
                )
            ),
            ContradictionKind.DIGEST,
        ),
    )


@pytest.mark.parametrize(
    ("claimed_cid", "expected_kind"),
    _contradictory_cids(KNOWN_VECTORS[0]),
    ids=("version", "base", "codec", "digest"),
)
def test_profile_differences_are_typed_contradictions_and_rejected(
    claimed_cid: str,
    expected_kind: ContradictionKind,
) -> None:
    vector = KNOWN_VECTORS[0]

    with pytest.raises(CIDIdentityContradiction) as contradiction:
        _require_frozen_claim(claimed_cid, vector.canonical_bytes)
    assert contradiction.value.kind is expected_kind

    # The datasets boundary validates the frozen CID profile.  A digest
    # contradiction is structurally valid and is rejected only when the CID is
    # bound to retained bytes; the other profile contradictions fail here.
    if expected_kind is ContradictionKind.DIGEST:
        assert (
            datasets_validate_cid(
                claimed_cid,
                codecs=(CID_CODEC,),
                mh_type=MH_TYPE,
                version=CID_VERSION,
                base=CID_BASE,
            )
            == claimed_cid
        )
        assert bytes(CID.decode(claimed_cid).raw_digest) != hashlib.sha256(
            vector.canonical_bytes
        ).digest()
    else:
        with pytest.raises(ValueError):
            datasets_validate_cid(
                claimed_cid,
                codecs=(CID_CODEC,),
                mh_type=MH_TYPE,
                version=CID_VERSION,
                base=CID_BASE,
            )

    # ContentIdentity binds the profile-valid CID to the retained bytes and
    # therefore rejects every contradiction, including the digest-only case.
    with pytest.raises(ValueError):
        ContentIdentity(
            cid=claimed_cid,
            digest_hex=vector.digest_hex,
            canonical_bytes=vector.canonical_bytes,
        ).verify()


LEGACY_PSEUDO_CIDS = (
    "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
    "cid:not-a-cid",
    "sha256:deadbeef",
    "tree:sha256:111",
    "repo:fixture",
    "BAGAAQEERA",
    "",
)


@pytest.mark.parametrize("legacy_value", LEGACY_PSEUDO_CIDS)
def test_legacy_kit_hashes_are_never_admitted_as_cids(legacy_value: str) -> None:
    with pytest.raises(TestExecutionIdentityError):
        reject_pseudo_cid(legacy_value)
    with pytest.raises(ValueError):
        datasets_validate_cid(legacy_value, codecs=(CID_CODEC,))


def _base32_cid_from_binary(raw: bytes) -> str:
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


@pytest.mark.parametrize("declared_size, actual_digest_size", ((32, 31), (31, 32)))
def test_malformed_multihash_lengths_are_rejected(
    declared_size: int,
    actual_digest_size: int,
) -> None:
    vector = KNOWN_VECTORS[0]
    digest = bytes.fromhex(vector.digest_hex)[:actual_digest_size]
    # CIDv1 | dag-json varint(0x0129) | sha2-256 | declared size | digest.
    malformed = _base32_cid_from_binary(
        b"\x01\xa9\x02\x12" + bytes((declared_size,)) + digest
    )

    with pytest.raises(CIDIdentityContradiction) as contradiction:
        _require_frozen_claim(malformed, vector.canonical_bytes)
    assert contradiction.value.kind is ContradictionKind.MALFORMED_MULTIHASH

    with pytest.raises(ValueError):
        CID.decode(malformed)
    with pytest.raises(ValueError):
        datasets_validate_cid(malformed, codecs=(CID_CODEC,))
    with pytest.raises(ValueError):
        ContentIdentity(
            cid=malformed,
            digest_hex=vector.digest_hex,
            canonical_bytes=vector.canonical_bytes,
        ).verify()


@pytest.mark.parametrize(
    "invalid_payload",
    (
        {"value": float("nan")},
        {1: "non-string key"},
        {"value": b"bytes must not be coerced"},
    ),
    ids=("non-finite", "non-string-key", "unsupported-type"),
)
def test_strict_dag_json_vectors_reject_coercion(invalid_payload: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        datasets_canonical_dag_json_bytes(invalid_payload)
    with pytest.raises(TestExecutionIdentityError):
        mint_content_identity(invalid_payload)
