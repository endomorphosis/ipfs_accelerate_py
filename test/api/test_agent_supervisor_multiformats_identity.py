"""Strict DAG-JSON / CIDv1 / multihash identity bridge tests (VFS-010 / VFS-060).

Also covers VFS-060 objective validation repair for VFS-G030: exact-text
discovery of ``objective validation repair``, separation of domain
``vfs/cid-profile@1`` evidence from the synthetic gate, and the refinement
that immutable object identity stays separate from mutable current-tree
projections used by the dependency-aware program-analysis cache.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest
from ipfs_datasets_py.utils.cid_utils import (
    canonical_dag_json_bytes as package_canonical_dag_json_bytes,
)
from ipfs_datasets_py.utils.cid_utils import (
    cid_for_bytes as package_cid_for_bytes,
)
from ipfs_datasets_py.utils.cid_utils import (
    cid_for_dag_json as package_cid_for_dag_json,
)
from ipfs_datasets_py.utils.cid_utils import (
    validate_cid as package_validate_cid,
)

from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    ALLOWED_CODECS,
    CID_BASE,
    CID_PROFILE_EVIDENCE,
    CID_PROFILE_GOAL_ID,
    CID_PROFILE_SCHEMA,
    CID_PROFILE_TASK_ID,
    CID_VERSION,
    DIGEST_SIZE,
    IDENTITY_LINK_SCHEMA,
    MH_TYPE,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    OBJECTIVE_VALIDATION_REPAIR_TASK_ID,
    CIDProfile,
    IdentityKind,
    IdentityLink,
    MultiformatsIdentityError,
    all_covered_evidence_terms,
    canonical_dag_json_bytes,
    cid_for_bytes,
    cid_for_dag_json,
    cid_from_sha256_digest,
    cid_profile,
    cid_profile_evidence_terms,
    covered_evidence_terms,
    digest_hex_from_cid,
    immutable_object_identity_separate_from_tree_projections,
    independent_round_trip_cid,
    independent_round_trip_dag_json,
    link_content_identity,
    link_dag_json,
    link_payload_digest,
    link_raw_bytes,
    link_runtime_artifact,
    objective_validation_repair_evidence_terms,
    parse_payload_digest,
    parse_runtime_artifact_id,
    reject_double_hashed_multihash,
    require_canonical_dag_json_bytes,
    validate_cid,
)
from ipfs_accelerate_py.agent_supervisor.program_analysis_cache import (
    DEPENDENCY_CACHE_EVIDENCE,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_PARENT_REPAIR_TASK_ID,
    ProgramAnalysisCacheKey,
    all_covered_evidence_terms as cache_all_covered_evidence_terms,
    objective_validation_repair_evidence_terms as cache_repair_terms,
    program_analysis_cache_evidence_terms,
    tree_projection_is_not_object_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

# Well-known IPFS / multiformats raw sha2-256 CIDv1 vectors (base32).
KNOWN_EMPTY_RAW_CID = (
    "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
)
KNOWN_HELLO_WORLD_RAW_CID = (
    "bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e"
)


def test_known_raw_vectors_match_frozen_profile() -> None:
    assert cid_for_bytes(b"") == KNOWN_EMPTY_RAW_CID
    assert cid_for_bytes(b"hello world") == KNOWN_HELLO_WORLD_RAW_CID
    assert validate_cid(KNOWN_EMPTY_RAW_CID, codecs=("raw",)) == KNOWN_EMPTY_RAW_CID
    assert (
        validate_cid(KNOWN_HELLO_WORLD_RAW_CID, codecs=("raw",))
        == KNOWN_HELLO_WORLD_RAW_CID
    )


def test_dag_json_cid_stable_across_mapping_order() -> None:
    left = {"unicode": "café", "nested": {"z": 2, "a": 1}}
    right = {"nested": {"a": 1, "z": 2}, "unicode": "café"}

    assert canonical_dag_json_bytes(left) == canonical_dag_json_bytes(right)
    assert cid_for_dag_json(left) == cid_for_dag_json(right)
    assert cid_for_dag_json(left).startswith("b")
    assert validate_cid(cid_for_dag_json(left), codecs=("dag-json",))


def test_raw_and_dag_json_cids_use_declared_codecs() -> None:
    raw_cid = cid_for_bytes(b'{"a":1}')
    dag_cid = cid_for_dag_json({"a": 1})

    assert raw_cid != dag_cid
    assert validate_cid(raw_cid, codecs=("raw",)) == raw_cid
    assert validate_cid(dag_cid, codecs=("dag-json",)) == dag_cid
    with pytest.raises(MultiformatsIdentityError):
        validate_cid(raw_cid, codecs=("dag-json",))
    with pytest.raises(MultiformatsIdentityError):
        validate_cid(dag_cid, codecs=("raw",))


def test_cross_package_bytes_match_cid_utils() -> None:
    payload = {"program": "vfs-010", "n": 1, "nested": {"b": 2, "a": 1}}
    raw = b"exact source bytes\n"

    assert canonical_dag_json_bytes(payload) == package_canonical_dag_json_bytes(
        payload
    )
    assert cid_for_dag_json(payload) == package_cid_for_dag_json(payload)
    assert cid_for_bytes(raw) == package_cid_for_bytes(raw)
    assert package_validate_cid(
        cid_for_dag_json(payload), codecs=("dag-json",)
    ) == cid_for_dag_json(payload)


def test_independent_round_trips() -> None:
    raw = b"\x00round-trip\n"
    obj = {"unicode": "λ", "items": [1, True, None], "nested": {"z": 0, "a": 1}}

    assert independent_round_trip_cid(raw, codec="raw") == cid_for_bytes(raw)
    assert independent_round_trip_dag_json(obj) == cid_for_dag_json(obj)

    encoded = canonical_dag_json_bytes(obj)
    assert independent_round_trip_cid(encoded, codec="dag-json") == cid_for_dag_json(
        obj
    )


def test_content_identity_link_preserves_local_id() -> None:
    value = {"assurance": "vfs-010", "part": "content"}
    formal_id = content_identity(value)
    multiformats_cid = cid_for_dag_json(value, for_identity=True)

    assert formal_id == multiformats_cid

    link = link_content_identity(formal_id, value=value)
    assert link.kind == IdentityKind.CONTENT_IDENTITY.value
    assert link.local_id == formal_id
    assert link.cid == multiformats_cid
    assert link.codec == "dag-json"
    assert link.version == CID_VERSION
    assert link.base == CID_BASE
    assert link.multihash_type == MH_TYPE
    assert link.digest_size == DIGEST_SIZE
    assert len(link.digest_hex) == 64
    # Dual identity: both fields present; neither is dropped.
    assert "local_id" in link.to_dict() and "cid" in link.to_dict()
    assert link.to_dict()["schema"] == IDENTITY_LINK_SCHEMA


def test_content_identity_link_rejects_silent_replacement() -> None:
    value = {"a": 1}
    formal_id = content_identity(value)
    other = content_identity({"a": 2})

    with pytest.raises(MultiformatsIdentityError, match="does not match"):
        link_content_identity(formal_id, value={"a": 2})

    with pytest.raises(MultiformatsIdentityError, match="expected_cid"):
        link_content_identity(formal_id, value=value, expected_cid=other)


def test_runtime_artifact_link_wraps_digest_without_replacing_id() -> None:
    payload = b'{"runtime":true,"n":3}'
    digest_hex = hashlib.sha256(payload).hexdigest()
    artifact_id = f"runtime-artifact:sha256:{digest_hex}"
    payload_digest = f"sha256:{digest_hex}"

    link = link_runtime_artifact(
        artifact_id,
        payload_bytes=payload,
        payload_digest=payload_digest,
        codec="raw",
    )

    assert link.kind == IdentityKind.RUNTIME_ARTIFACT.value
    assert link.local_id == artifact_id
    assert link.cid == cid_for_bytes(payload)
    assert link.cid == cid_from_sha256_digest(digest_hex, codec="raw")
    assert link.local_id != link.cid
    assert link.digest_hex == digest_hex
    # Persisted runtime identity remains the artifact id, not the CID.
    restored = IdentityLink.from_dict(link.to_dict())
    assert restored.local_id == artifact_id
    assert restored.cid == link.cid


def test_payload_digest_link_is_wrap_not_rehash() -> None:
    payload = b"payload-for-digest"
    digest_hex = hashlib.sha256(payload).hexdigest()
    payload_digest = f"sha256:{digest_hex}"

    link = link_payload_digest(payload_digest)
    assert link.local_id == payload_digest
    assert link.cid == cid_for_bytes(payload)
    assert link.local_id != link.cid

    # Double-hash path would CID-address the digest bytes themselves.
    double = cid_for_bytes(bytes.fromhex(digest_hex))
    assert link.cid != double


def test_reject_double_hashing() -> None:
    payload = b"do-not-double-hash"
    direct = cid_for_bytes(payload)
    reject_double_hashed_multihash(payload, direct)

    inner = hashlib.sha256(payload).digest()
    double_cid = cid_for_bytes(inner)
    with pytest.raises(MultiformatsIdentityError, match="double hashing"):
        reject_double_hashed_multihash(payload, double_cid)

    # Wrapping must not equal hashing the digest as payload.
    digest_hex = hashlib.sha256(payload).hexdigest()
    assert cid_from_sha256_digest(digest_hex) == direct
    assert cid_from_sha256_digest(digest_hex) != cid_for_bytes(
        bytes.fromhex(digest_hex)
    )


def test_cid_from_sha256_digest_refuses_ambiguous_already_hashed_false() -> None:
    with pytest.raises(MultiformatsIdentityError, match="ambiguous"):
        cid_from_sha256_digest(b"\x00" * 32, already_hashed=False)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_numbers(bad: float) -> None:
    with pytest.raises(MultiformatsIdentityError, match="non-finite"):
        canonical_dag_json_bytes({"value": bad})
    with pytest.raises(MultiformatsIdentityError):
        cid_for_dag_json({"value": bad})


def test_rejects_default_repr_python_objects() -> None:
    marker = object()
    with pytest.raises(MultiformatsIdentityError, match="default=repr|not JSON"):
        canonical_dag_json_bytes({"value": marker})
    with pytest.raises(MultiformatsIdentityError):
        cid_for_dag_json({"value": marker})


def test_rejects_timestamps_in_identity() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(MultiformatsIdentityError, match="timestamp"):
        canonical_dag_json_bytes({"when": now})
    with pytest.raises(MultiformatsIdentityError, match="timestamp"):
        cid_for_dag_json({"created_at": "2020-01-01"}, for_identity=True)
    with pytest.raises(MultiformatsIdentityError, match="timestamp"):
        link_dag_json({"expires_at_ms": 1}, for_identity=True)


def test_rejects_unsorted_json_bytes() -> None:
    unsorted = b'{"b":1,"a":2}'
    with pytest.raises(MultiformatsIdentityError, match="canonical|unsorted"):
        require_canonical_dag_json_bytes(unsorted)

    canonical = canonical_dag_json_bytes({"b": 1, "a": 2})
    assert canonical == b'{"a":2,"b":1}'
    assert require_canonical_dag_json_bytes(canonical) == canonical


def test_rejects_truncated_and_pseudo_cids() -> None:
    samples = [
        "",
        "not-a-cid",
        "bafkrei",  # truncated
        "BAGAAQEERA",  # wrong case / truncated
        "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",  # CIDv0
        "sha256:deadbeef",
        "cid:not-a-cid",
        "BAFKREIF2PALL7DYBZ7VECQKA3ZO24IRDWABWDI4WC55JZNAQ75Q7EAAVVU",
    ]
    for sample in samples:
        with pytest.raises(MultiformatsIdentityError):
            validate_cid(sample)


def test_rejects_truncated_sha2_256_multihash() -> None:
    from multiformats import CID, multihash

    truncated = str(
        CID(
            "base32",
            1,
            "raw",
            multihash.wrap(bytes(16), "sha2-256"),
        )
    )
    with pytest.raises(MultiformatsIdentityError):
        validate_cid(truncated, codecs=("raw",))


def test_rejects_malformed_multihashes_and_wrong_digest_size() -> None:
    with pytest.raises(MultiformatsIdentityError, match="64"):
        cid_from_sha256_digest("abcd")
    with pytest.raises(MultiformatsIdentityError, match="32"):
        cid_from_sha256_digest(b"\x00" * 16)
    with pytest.raises(MultiformatsIdentityError):
        parse_runtime_artifact_id("runtime-artifact:sha256:deadbeef")
    with pytest.raises(MultiformatsIdentityError):
        parse_payload_digest("sha256:not-hex")


def test_rejects_ambiguous_raw_string_file_input(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"file-bytes")

    with pytest.raises(MultiformatsIdentityError, match="ambiguous"):
        cid_for_bytes("hello")  # type: ignore[arg-type]
    with pytest.raises(MultiformatsIdentityError, match="ambiguous"):
        cid_for_bytes(path)  # type: ignore[arg-type]
    with pytest.raises(MultiformatsIdentityError, match="ambiguous"):
        cid_for_bytes(bytearray(b"x"))  # type: ignore[arg-type]
    with pytest.raises(MultiformatsIdentityError, match="ambiguous"):
        link_raw_bytes(str(path))  # type: ignore[arg-type]

    # Explicit bytes remain valid.
    assert cid_for_bytes(path.read_bytes()) == cid_for_bytes(b"file-bytes")


def test_validate_cid_enforces_lowercase_version_base_digest() -> None:
    cid = cid_for_bytes(b"profile")
    assert validate_cid(cid, codecs=("raw",)) == cid

    with pytest.raises(MultiformatsIdentityError, match="lowercase"):
        validate_cid(cid.upper(), codecs=("raw",))

    digest = digest_hex_from_cid(cid, codecs=("raw",))
    assert len(digest) == DIGEST_SIZE * 2
    assert digest == hashlib.sha256(b"profile").hexdigest()


def test_link_raw_and_dag_json_helpers() -> None:
    raw_link = link_raw_bytes(b"abc", local_id="label:raw:abc")
    assert raw_link.local_id == "label:raw:abc"
    assert raw_link.cid == cid_for_bytes(b"abc")
    assert raw_link.codec == "raw"

    dag_link = link_dag_json({"k": "v"}, local_id="label:dag:kv")
    assert dag_link.local_id == "label:dag:kv"
    assert dag_link.cid == cid_for_dag_json({"k": "v"})
    assert dag_link.codec == "dag-json"


def test_identity_link_round_trip_dict() -> None:
    link = link_dag_json({"round": "trip"})
    restored = IdentityLink.from_dict(link.to_dict())
    assert restored == link
    with pytest.raises(MultiformatsIdentityError, match="unknown fields"):
        IdentityLink.from_dict({**link.to_dict(), "extra": 1})


def test_profile_constants() -> None:
    assert CID_VERSION == 1
    assert CID_BASE == "base32"
    assert MH_TYPE == "sha2-256"
    assert DIGEST_SIZE == 32
    assert ALLOWED_CODECS == frozenset({"raw", "dag-json"})


def test_vfs_g141_cid_profile_is_executable_packet_evidence() -> None:
    """VFS-G141: expose the frozen profile without changing identity bytes."""

    profile = cid_profile()
    assert profile is cid_profile()
    assert profile == CIDProfile()
    assert profile.to_dict() == {
        "schema": CID_PROFILE_SCHEMA,
        "evidence": CID_PROFILE_EVIDENCE,
        "version": 1,
        "base": "base32",
        "codecs": ("dag-json", "raw"),
        "multihash_type": "sha2-256",
        "digest_size": 32,
    }
    assert CID_PROFILE_EVIDENCE == "vfs/cid-profile@1"
    assert CID_PROFILE_GOAL_ID == "VFS-G141"
    assert CID_PROFILE_TASK_ID == "VFS-057"
    assert cid_profile_evidence_terms() == ("vfs/cid-profile@1",)
    assert covered_evidence_terms() == cid_profile_evidence_terms()

    # The descriptor is closed: callers cannot negotiate a weaker profile.
    with pytest.raises(MultiformatsIdentityError, match="profile is frozen"):
        CIDProfile(base="base58btc")
    with pytest.raises(MultiformatsIdentityError, match="profile is frozen"):
        CIDProfile(digest_size=16)

    # Evidence metadata is not an identity input.  Existing direct and
    # compatibility identities continue to address only canonical payload.
    value = {"packet": "VFS-057", "goal": "VFS-G141"}
    cid = cid_for_dag_json(value)
    assert cid == package_cid_for_dag_json(value)
    link = link_content_identity(content_identity(value), value=value)
    assert link.cid == cid
    assert CID_PROFILE_EVIDENCE not in canonical_dag_json_bytes(value).decode()


def test_objective_validation_repair_evidence_term_discoverable() -> None:
    """VFS-G030 objective validation repair: exact-text discovery key present.

    Anchors the synthetic phrase ``objective validation repair`` so objective
    scans re-find the validation gate.  Domain evidence stays separate
    (``vfs/cid-profile@1``).  The repair term never enters CID input bytes,
    IdentityLink payloads, or mutable current-tree projection dimensions.
    Owned by VFS-G030 via repair task VFS-060.
    """

    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
    assert OBJECTIVE_GOAL_ID == "VFS-G030"
    assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-060"
    assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G030"
    assert OBJECTIVE_PARENT_REPAIR_TASK_ID == "VFS-060"
    assert objective_validation_repair_evidence_terms() == (
        "objective validation repair",
    )
    assert cache_repair_terms() == ("objective validation repair",)

    # Domain envelope evidence remains cid-profile only on this bridge.
    assert cid_profile_evidence_terms() == ("vfs/cid-profile@1",)
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == ("vfs/cid-profile@1",)
    assert covered_evidence_terms() == ("vfs/cid-profile@1",)
    assert "objective validation repair" not in covered_evidence_terms()
    assert "objective validation repair" not in cid_profile_evidence_terms()

    # Full discovery set includes the validation-gate meta term last.
    assert all_covered_evidence_terms() == (
        "vfs/cid-profile@1",
        "objective validation repair",
    )
    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE in all_covered_evidence_terms()
    # Cache surface covers dependency-cache + invalidation + profile + repair.
    assert DEPENDENCY_CACHE_EVIDENCE in program_analysis_cache_evidence_terms()
    assert "objective validation repair" not in program_analysis_cache_evidence_terms()
    assert "objective validation repair" in cache_all_covered_evidence_terms()

    # Profile / link identity envelopes never absorb the synthetic repair term.
    profile_payload = cid_profile().to_dict()
    assert profile_payload["evidence"] == CID_PROFILE_EVIDENCE
    assert "objective validation repair" not in profile_payload["evidence"]
    assert profile_payload.get("evidence_objective_validation_repair") is None

    value = {"goal": "VFS-G030", "task": "VFS-060"}
    cid = cid_for_dag_json(value)
    link = link_content_identity(content_identity(value), value=value)
    link_payload = link.to_dict()
    encoded_value = canonical_dag_json_bytes(value).decode()
    encoded_link = json.dumps(link_payload, sort_keys=True)
    assert cid == package_cid_for_dag_json(value)
    assert "objective validation repair" not in encoded_value
    assert "objective validation repair" not in encoded_link
    assert "objective validation repair" not in link.cid
    assert link.cid == cid


def test_vfs_g030_acceptance_identity_tree_separation_and_fail_closed() -> None:
    """VFS-G030 acceptance: cross-package CIDs, mappings, tree separation.

    Proves the acceptance subset for objective validation repair on the
    multiformats + dependency-cache surface: CIDv1/base32/dag-json/sha2-256
    bytes are cross-package reproducible; existing supervisor IDs retain
    compatibility mappings; immutable object identity stays separate from
    mutable current-tree projections; semantic dependencies participate in
    cache keys; corruption-shaped inputs fail closed.
    """

    assert OBJECTIVE_GOAL_ID == "VFS-G030"
    assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-060"
    assert "objective validation repair" in all_covered_evidence_terms()
    assert immutable_object_identity_separate_from_tree_projections() is True

    # Cross-package reproducibility for raw and DAG-JSON.
    payload = {"unicode": "café", "nested": {"z": 2, "a": 1}}
    reordered = {"nested": {"a": 1, "z": 2}, "unicode": "café"}
    assert cid_for_dag_json(payload) == package_cid_for_dag_json(reordered)
    assert cid_for_bytes(b"hello world") == package_cid_for_bytes(b"hello world")
    assert validate_cid(cid_for_dag_json(payload), codecs=("dag-json",))
    assert independent_round_trip_dag_json(payload) == cid_for_dag_json(payload)

    # Existing supervisor IDs retain compatibility mappings.
    formal = content_identity(payload)
    bridge = cid_for_dag_json(payload)
    assert formal == bridge
    link = link_content_identity(formal, value=payload)
    assert link.local_id == formal
    assert link.cid == bridge

    # Mutable current-tree projections bind cache keys only; object CIDs ignore them.
    assert tree_projection_is_not_object_identity(
        forest_identity="forest:sha256:tree-A",
        payload=payload,
    )
    key_tree_a = ProgramAnalysisCacheKey(
        forest_identity="forest:sha256:tree-A",
        objective_revision="obj-1",
        policy_revision="pol-1",
        analyzer_version="analyzer@1",
        schema_version="schema@1",
        configuration_digest="sha256:cfg-1",
        query_digest="sha256:q-1",
        capability_revision="cap@1",
        assumption_digest="sha256:ass-1",
        toolchain_version="tc@1",
    )
    key_tree_b = ProgramAnalysisCacheKey(
        forest_identity="forest:sha256:tree-B",
        objective_revision="obj-1",
        policy_revision="pol-1",
        analyzer_version="analyzer@1",
        schema_version="schema@1",
        configuration_digest="sha256:cfg-1",
        query_digest="sha256:q-1",
        capability_revision="cap@1",
        assumption_digest="sha256:ass-1",
        toolchain_version="tc@1",
    )
    assert key_tree_a.digest != key_tree_b.digest
    # All semantic / policy dimensions participate: policy change re-keys.
    key_policy = ProgramAnalysisCacheKey(
        forest_identity="forest:sha256:tree-A",
        objective_revision="obj-1",
        policy_revision="pol-2",
        analyzer_version="analyzer@1",
        schema_version="schema@1",
        configuration_digest="sha256:cfg-1",
        query_digest="sha256:q-1",
        capability_revision="cap@1",
        assumption_digest="sha256:ass-1",
        toolchain_version="tc@1",
    )
    assert key_policy.digest != key_tree_a.digest
    # Object CID for the same payload is stable across tree projection changes.
    assert cid_for_dag_json(payload) == bridge
    assert "forest:sha256:tree-A" not in bridge
    assert "forest:sha256:tree-B" not in bridge

    # Fail-closed on corrupted / non-identity inputs.
    with pytest.raises(MultiformatsIdentityError):
        validate_cid("not-a-cid")
    with pytest.raises(MultiformatsIdentityError, match="timestamp"):
        cid_for_dag_json({"when": datetime.now(timezone.utc)})
    payload_bytes = b"do-not-double-hash"
    double_cid = cid_for_bytes(hashlib.sha256(payload_bytes).digest())
    with pytest.raises(MultiformatsIdentityError, match="double hashing"):
        reject_double_hashed_multihash(payload_bytes, double_cid)


def test_runtime_artifact_mismatched_payload_rejected() -> None:
    payload = b"left"
    digest_hex = hashlib.sha256(payload).hexdigest()
    artifact_id = f"runtime-artifact:sha256:{digest_hex}"

    with pytest.raises(MultiformatsIdentityError, match="payload_bytes"):
        link_runtime_artifact(artifact_id, payload_bytes=b"right")
    with pytest.raises(MultiformatsIdentityError, match="payload_digest"):
        link_runtime_artifact(
            artifact_id,
            payload_digest="sha256:" + ("ab" * 32),
        )


def test_content_identity_and_package_agree_on_known_object() -> None:
    value = {"hello": "world"}
    formal = content_identity(value)
    bridge = cid_for_dag_json(value)
    package = package_cid_for_dag_json(value)
    assert formal == bridge == package
    assert independent_round_trip_dag_json(value) == formal
