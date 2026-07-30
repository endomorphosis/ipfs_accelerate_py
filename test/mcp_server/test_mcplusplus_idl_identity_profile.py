"""SCA-608 / SCAEV181MCPRUNTIME: MCP-IDL identities are decodable profile-tagged CIDs."""

from __future__ import annotations

import copy
import hashlib

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    IDL_CID_MULTIBASE,
    IDL_CID_MULTICODEC,
    IDL_CID_MULTIHASH,
    IDL_CID_VERSION,
    IDL_IDENTITY_PROFILE,
    IDL_IDENTITY_PROFILE_ID,
    IDLIdentityError,
    InterfaceDescriptorRegistry,
    build_ai_catalog_v1_descriptor,
    build_descriptor,
    canonicalize_descriptor,
    compute_interface_cid,
    identify_interface_descriptor,
    idl_identity_profile,
    is_pseudo_interface_cid,
    validate_interface_cid,
    validate_interface_identity,
)


def _sample_descriptor() -> dict:
    return build_descriptor(
        name="test.iface.v1",
        namespace="test.namespace",
        version="1.0.0",
        methods=[
            {
                "name": "test.iface.v1/ping",
                "operation": "ping",
                "mcp_tool": "ping",
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "output_schema": {"type": "object"},
                "required_authority": "test/read",
            }
        ],
        requires=["mcp++/profile-a-idl"],
    )


def test_idl_identity_profile_is_explicit_and_not_cross_package() -> None:
    profile = idl_identity_profile()
    assert profile["profile"] == IDL_IDENTITY_PROFILE
    assert profile["profile_id"] == IDL_IDENTITY_PROFILE_ID
    assert profile["cid_version"] == IDL_CID_VERSION
    assert profile["multibase"] == IDL_CID_MULTIBASE
    assert profile["multicodec"] == IDL_CID_MULTICODEC
    assert profile["multihash"] == IDL_CID_MULTIHASH
    assert profile["policies"]["pseudo_cid_allowed"] is False
    assert profile["policies"]["digest_labeled_as_cid_allowed"] is False
    assert profile["policies"]["cross_package_profile_equality_allowed"] is False
    # Must not silently claim datasets/accelerator artifact profiles.
    assert profile["profile"] != "strict-dag-json-v1"
    assert profile["profile"] != "ir-canonical-identity-v1"


def test_compute_interface_cid_is_decodable_cidv1_not_pseudo() -> None:
    descriptor = _sample_descriptor()
    cid = compute_interface_cid(descriptor)

    assert not is_pseudo_interface_cid(cid)
    assert not cid.startswith("cidv1-sha256-")
    assert cid == cid.lower()
    assert cid.startswith("b")

    from multiformats import CID

    parsed = CID.decode(cid)
    assert int(parsed.version) == 1
    assert parsed.base.name == "base32"
    assert parsed.codec.name == "raw"
    assert parsed.hashfun.name == "sha2-256"
    canonical = canonicalize_descriptor(descriptor)
    assert bytes(parsed.raw_digest) == hashlib.sha256(canonical).digest()


def test_identify_interface_descriptor_binds_profile_and_verifies() -> None:
    descriptor = build_ai_catalog_v1_descriptor()
    identity = identify_interface_descriptor(descriptor)

    assert identity.profile == IDL_IDENTITY_PROFILE
    assert identity.profile_id == IDL_IDENTITY_PROFILE_ID
    assert identity.validated is True
    assert identity.cid == compute_interface_cid(descriptor)
    assert identity.digest == f"sha256:{hashlib.sha256(identity.canonical_bytes).hexdigest()}"
    assert identity.byte_length == len(identity.canonical_bytes)
    assert identity.multicodec == "raw"
    assert identity.multihash == "sha2-256"
    assert identity.multibase == "base32"
    assert identity.cid_version == 1

    verified = validate_interface_cid(
        identity.cid,
        identity.canonical_bytes,
        expected_profile=IDL_IDENTITY_PROFILE,
    )
    assert verified["validated"] is True
    assert verified["raw_digest"] == hashlib.sha256(
        identity.canonical_bytes
    ).hexdigest()
    assert verified["profile"] == IDL_IDENTITY_PROFILE

    round_trip = validate_interface_identity(identity)
    assert round_trip.validated is True
    assert round_trip.cid == identity.cid


def test_canonical_bytes_stable_across_mapping_order() -> None:
    left = {"name": "a", "namespace": "n", "version": "1", "methods": [], "errors": []}
    right = {"errors": [], "methods": [], "version": "1", "namespace": "n", "name": "a"}
    assert canonicalize_descriptor(left) == canonicalize_descriptor(right)
    assert compute_interface_cid(left) == compute_interface_cid(right)


def test_schema_edit_changes_interface_cid() -> None:
    base = _sample_descriptor()
    changed = copy.deepcopy(base)
    changed["methods"][0]["required_authority"] = "test/write"
    assert compute_interface_cid(changed) != compute_interface_cid(base)


def test_pseudo_and_digest_shaped_cids_are_rejected() -> None:
    descriptor = _sample_descriptor()
    canonical = canonicalize_descriptor(descriptor)
    digest_hex = hashlib.sha256(canonical).hexdigest()

    for bad in (
        f"cidv1-sha256-{digest_hex}",
        f"sha256:{digest_hex}",
        digest_hex,
        "not-a-cid",
        "",
    ):
        assert is_pseudo_interface_cid(bad) or bad in {"not-a-cid", ""}
        with pytest.raises(IDLIdentityError) as exc:
            validate_interface_cid(bad, canonical)
        assert exc.value.reason_code in {
            "pseudo_cid_rejected",
            "cid_not_lowercase",
            "cid_not_decodable",
        }


def test_multihash_digest_mismatch_fails_closed() -> None:
    descriptor = _sample_descriptor()
    identity = identify_interface_descriptor(descriptor)
    altered = identity.canonical_bytes + b"\x00"
    with pytest.raises(IDLIdentityError) as exc:
        validate_interface_cid(identity.cid, altered)
    assert exc.value.reason_code == "multihash_digest_mismatch"


def test_registry_registers_decodable_profile_tagged_cid() -> None:
    registry = InterfaceDescriptorRegistry(
        supported_capabilities=["mcp++/profile-a-idl"]
    )
    cid = registry.register_ai_catalog_v1()
    assert not is_pseudo_interface_cid(cid)
    assert cid in registry.list_interfaces()
    payload = registry.get_descriptor(cid)
    assert payload is not None
    assert payload["interface_cid"] == cid
    # Stored CID re-derives from the descriptor body without the cid field.
    body = {k: v for k, v in payload.items() if k != "interface_cid"}
    assert compute_interface_cid(body) == cid
    identity = identify_interface_descriptor(body)
    assert identity.profile == IDL_IDENTITY_PROFILE


def test_identity_to_dict_exposes_profile_metadata() -> None:
    identity = identify_interface_descriptor(_sample_descriptor())
    payload = identity.to_dict(include_canonical_bytes=True)
    assert payload["profile"] == IDL_IDENTITY_PROFILE
    assert payload["profile_id"] == IDL_IDENTITY_PROFILE_ID
    assert payload["validated"] is True
    assert "canonical_bytes_hex" in payload
    assert not is_pseudo_interface_cid(payload["cid"])
