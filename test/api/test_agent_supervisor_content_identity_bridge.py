"""Contract tests for the SCA ContentIdentity@1 multiformats bridge."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import math
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis import content_identity_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import (
    CONTENT_IDENTITY_INTERFACE,
    CONTENT_IDENTITY_SCHEMA,
    LOGIC_IR_PROFILE,
    MULTICODEC_DAG_JSON,
    MULTICODEC_RAW,
    MULTIHASH_SHA2_256,
    PROVIDER_CID_UTILS,
    PROVIDER_IPLD_CID,
    PROVIDER_IR_CORE_IDENTITY,
    PROVIDER_PROFILE_G,
    STRICT_ARTIFACT_PROFILE,
    CidValidationError,
    ContentIdentityError,
    MultiformatsUnavailableError,
    ProfileContradictionKind,
    compare_provider_identities,
    content_identity_probe,
    decode_and_verify_cid,
    identify_for_profile,
    identify_logic_ir,
    identify_strict_artifact,
    identify_strict_artifact_bytes,
    is_digest_shaped,
    multiformats_available,
    profiles_are_interchangeable,
    require_multiformats,
    reset_provider_import_cache,
    sha256_digest_label,
)


@pytest.fixture(autouse=True)
def _clear_import_cache() -> Any:
    reset_provider_import_cache()
    yield
    reset_provider_import_cache()


def _payload() -> dict[str, Any]:
    return {"unicode": "café", "nested": {"z": 2, "a": 1}, "flag": True, "n": None}


def test_strict_artifact_profile_is_lowercase_base32_cidv1_dag_json_sha2_256() -> None:
    identity = identify_strict_artifact(_payload())

    assert identity.profile == STRICT_ARTIFACT_PROFILE
    assert identity.cid_version == 1
    assert identity.multibase == "base32"
    assert identity.multicodec == MULTICODEC_DAG_JSON
    assert identity.multihash == MULTIHASH_SHA2_256
    assert identity.cid == identity.cid.lower()
    assert identity.cid.startswith("b")
    assert identity.validated is True
    assert identity.byte_length == len(identity.canonical_bytes)
    assert identity.digest == sha256_digest_label(identity.canonical_bytes)
    assert not is_digest_shaped(identity.cid)
    assert identity.provider == PROVIDER_CID_UTILS

    from multiformats import CID

    parsed = CID.decode(identity.cid)
    assert parsed.version == 1
    assert parsed.codec.name == "dag-json"
    assert parsed.hashfun.name == "sha2-256"
    assert parsed.base.name == "base32"
    assert bytes(parsed.raw_digest) == hashlib.sha256(identity.canonical_bytes).digest()


def test_strict_artifact_is_stable_across_mapping_order() -> None:
    left = identify_strict_artifact({"nested": {"z": 2, "a": 1}, "unicode": "café"})
    right = identify_strict_artifact({"unicode": "café", "nested": {"a": 1, "z": 2}})
    assert left.canonical_bytes == right.canonical_bytes
    assert left.cid == right.cid
    assert left.digest == right.digest


def test_strict_artifact_bytes_path_matches_object_path() -> None:
    from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

    payload = _payload()
    from_obj = identify_strict_artifact(payload)
    from_bytes = identify_strict_artifact_bytes(canonical_dag_json_bytes(payload))
    assert from_obj.cid == from_bytes.cid
    assert from_obj.canonical_bytes == from_bytes.canonical_bytes


def test_logic_ir_profile_is_domain_separated_raw_codec() -> None:
    identity = identify_logic_ir(
        {"title": "proof-goal", "steps": ["a", "b"]},
        domain="intent",
        schema_version="1.0.0",
    )

    assert identity.profile == LOGIC_IR_PROFILE
    assert identity.multicodec == MULTICODEC_RAW
    assert identity.multihash == MULTIHASH_SHA2_256
    assert identity.multibase == "base32"
    assert identity.cid_version == 1
    assert identity.domain == "intent"
    assert identity.schema_version == "1.0.0"
    assert identity.cid.startswith("b")
    assert b'"identity_profile":"ir-canonical-identity-v1"' in identity.canonical_bytes
    assert b'"domain":"intent"' in identity.canonical_bytes
    assert identity.digest == sha256_digest_label(identity.canonical_bytes)
    assert identity.validated is True

    from multiformats import CID

    parsed = CID.decode(identity.cid)
    assert parsed.codec.name == "raw"
    assert bytes(parsed.raw_digest) == hashlib.sha256(identity.canonical_bytes).digest()


def test_logic_ir_domain_separation_changes_identity() -> None:
    payload = {"x": 1}
    left = identify_logic_ir(payload, domain="intent", schema_version="1.0.0")
    right = identify_logic_ir(payload, domain="policy", schema_version="1.0.0")
    assert left.canonical_bytes != right.canonical_bytes
    assert left.cid != right.cid
    assert left.digest != right.digest


def test_every_cid_raw_digest_equals_sha256_of_retained_bytes() -> None:
    artifact = identify_strict_artifact(_payload())
    ir = identify_logic_ir({"k": "v"}, domain="test", schema_version="v1")
    for identity in (artifact, ir):
        verified = decode_and_verify_cid(
            identity.cid,
            identity.canonical_bytes,
            expected_codec=identity.multicodec,
            expected_profile=identity.profile,
        )
        assert verified["validated"] is True
        assert verified["raw_digest"] == hashlib.sha256(
            identity.canonical_bytes
        ).hexdigest()
        assert verified["digest"] == identity.digest


def test_decode_rejects_wrong_preimage_bytes() -> None:
    identity = identify_strict_artifact({"a": 1})
    with pytest.raises(CidValidationError, match="multihash_digest_mismatch"):
        decode_and_verify_cid(
            identity.cid,
            b'{"a":2}',
            expected_codec=MULTICODEC_DAG_JSON,
        )


def test_decode_rejects_digest_shaped_string_as_cid() -> None:
    digest = sha256_digest_label(b"not-a-cid")
    with pytest.raises(CidValidationError, match="digest-shaped"):
        decode_and_verify_cid(
            digest,
            b"not-a-cid",
            expected_codec=MULTICODEC_RAW,
        )
    with pytest.raises(CidValidationError, match="digest-shaped"):
        decode_and_verify_cid(
            digest.removeprefix("sha256:"),
            b"not-a-cid",
            expected_codec=MULTICODEC_RAW,
        )


def test_digest_field_is_never_copied_into_cid() -> None:
    identity = identify_strict_artifact({"a": 1})
    payload = identity.to_dict()
    assert payload["digest"].startswith("sha256:")
    assert payload["cid"].startswith("b")
    assert payload["cid"] != payload["digest"]
    assert payload["cid"] != payload["digest"].removeprefix("sha256:")
    assert is_digest_shaped(payload["digest"]) is True
    assert is_digest_shaped(payload["cid"]) is False


def test_identify_for_profile_dispatches_declared_profiles() -> None:
    artifact = identify_for_profile({"a": 1}, profile=STRICT_ARTIFACT_PROFILE)
    ir = identify_for_profile(
        {"a": 1},
        profile=LOGIC_IR_PROFILE,
        domain="intent",
        schema_version="1.0.0",
    )
    assert artifact.profile == STRICT_ARTIFACT_PROFILE
    assert ir.profile == LOGIC_IR_PROFILE
    with pytest.raises(ContentIdentityError, match="unknown content-identity profile"):
        identify_for_profile({"a": 1}, profile="not-a-profile")
    with pytest.raises(ContentIdentityError, match="domain and schema_version"):
        identify_for_profile({"a": 1}, profile=LOGIC_IR_PROFILE)


def test_cross_module_unicode_canonicalization_is_typed_contradiction() -> None:
    """cid_utils (ensure_ascii=False) vs ipld_cid (ensure_ascii=True)."""

    contradictions = compare_provider_identities(
        {"unicode": "café", "nested": {"z": 2, "a": 1}},
        providers=(PROVIDER_CID_UTILS, PROVIDER_IPLD_CID),
    )
    kinds = {item.kind for item in contradictions}
    assert ProfileContradictionKind.CANONICAL_BYTES_MISMATCH in kinds
    assert ProfileContradictionKind.CID_MISMATCH in kinds
    assert ProfileContradictionKind.DIGEST_MISMATCH in kinds
    assert not any(
        item.kind == ProfileContradictionKind.CODEC_MISMATCH for item in contradictions
    )


def test_profile_g_matches_cid_utils_for_unicode_dag_json() -> None:
    contradictions = compare_provider_identities(
        {"unicode": "café", "nested": {"z": 2, "a": 1}},
        providers=(PROVIDER_CID_UTILS, PROVIDER_PROFILE_G),
    )
    assert contradictions == ()


def test_logic_ir_vs_artifact_is_codec_and_profile_contradiction() -> None:
    contradictions = compare_provider_identities(
        {"a": 1},
        domain="intent",
        schema_version="1.0.0",
        providers=(PROVIDER_CID_UTILS, PROVIDER_IR_CORE_IDENTITY),
    )
    kinds = {item.kind for item in contradictions}
    assert ProfileContradictionKind.CODEC_MISMATCH in kinds
    assert ProfileContradictionKind.PROFILE_MISMATCH in kinds
    assert ProfileContradictionKind.CANONICAL_BYTES_MISMATCH in kinds
    assert ProfileContradictionKind.CID_MISMATCH in kinds


def test_profiles_are_not_interchangeable_across_profiles() -> None:
    artifact = identify_strict_artifact({"a": 1})
    ir = identify_logic_ir({"a": 1}, domain="intent", schema_version="1.0.0")
    assert profiles_are_interchangeable(artifact, artifact) is True
    assert profiles_are_interchangeable(artifact, ir) is False
    assert profiles_are_interchangeable(ir, ir) is True


def test_provider_import_is_lazy_at_module_load() -> None:
    source = open(bridge.__file__, encoding="utf-8").read()
    # No top-level multiformats / datasets imports in module body.
    for forbidden in (
        "from multiformats",
        "import multiformats",
        "from ipfs_datasets_py",
        "import ipfs_datasets_py",
    ):
        # Allow mentions only inside functions (lazy).  Reject module-level.
        lines = [
            line
            for line in source.splitlines()
            if forbidden in line and not line.lstrip().startswith("#")
        ]
        for line in lines:
            # Lazy imports live inside functions (indented).
            assert line.startswith(" ") or line.startswith("\t"), line


def _uninstall_multiformats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove multiformats from the import system so fail-closed paths run."""

    import sys

    real_import = builtins.__import__

    def reject_multiformats(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "multiformats" or name.startswith("multiformats."):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    for key in list(sys.modules):
        if key == "multiformats" or key.startswith("multiformats."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    reset_provider_import_cache()
    monkeypatch.setattr(builtins, "__import__", reject_multiformats)


def test_missing_multiformats_fails_closed_for_cid_required_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _uninstall_multiformats(monkeypatch)

    # importlib.import_module goes through __import__
    with pytest.raises(MultiformatsUnavailableError):
        require_multiformats()
    with pytest.raises(MultiformatsUnavailableError):
        identify_strict_artifact({"a": 1})
    with pytest.raises(MultiformatsUnavailableError):
        identify_logic_ir({"a": 1}, domain="d", schema_version="1")
    with pytest.raises(MultiformatsUnavailableError):
        decode_and_verify_cid(
            "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku",
            b"x",
            expected_codec=MULTICODEC_RAW,
        )

    # Digests remain available without multiformats and are not CIDs.
    digest = sha256_digest_label(b"payload")
    assert digest.startswith("sha256:")
    assert is_digest_shaped(digest)
    assert not digest.startswith("baf")


def test_missing_multiformats_does_not_emit_digest_labeled_cid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _uninstall_multiformats(monkeypatch)

    with pytest.raises(MultiformatsUnavailableError) as exc_info:
        identify_strict_artifact({"a": 1})
    # No fallback CID on the exception path.
    assert "cid" not in exc_info.value.details or not is_digest_shaped(
        str(exc_info.value.details.get("cid", ""))
    )


def test_strict_artifact_rejects_nonfinite_numbers() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ContentIdentityError):
            identify_strict_artifact({"value": bad})


def test_strict_artifact_rejects_non_string_map_keys() -> None:
    with pytest.raises(ContentIdentityError):
        identify_strict_artifact({1: "integer-key"})  # type: ignore[dict-item]


def test_content_identity_probe_reports_providers_and_fail_closed_policy() -> None:
    probe = content_identity_probe()
    assert probe["schema"] == CONTENT_IDENTITY_SCHEMA
    assert probe["interface"] == CONTENT_IDENTITY_INTERFACE
    assert probe["artifact_profile"] == STRICT_ARTIFACT_PROFILE
    assert probe["logic_ir_profile"] == LOGIC_IR_PROFILE
    assert probe["cross_profile_equality_allowed"] is False
    assert probe["digest_labeled_as_cid_allowed"] is False
    assert multiformats_available() is True
    assert probe["providers"]["multiformats"] is True
    assert probe["cid_required_operations_ready"] is True


def test_to_dict_schema_contract() -> None:
    identity = identify_strict_artifact({"tool": "scan"})
    payload = identity.to_dict(include_canonical_bytes=True)
    assert payload["schema"] == CONTENT_IDENTITY_SCHEMA
    assert payload["schema_version"] == 1
    assert payload["interface"] == CONTENT_IDENTITY_INTERFACE
    assert payload["canonical_bytes_hex"] == identity.canonical_bytes.hex()
    assert "canonical_bytes" not in payload


def test_compare_provider_identities_default_includes_all_four_providers() -> None:
    contradictions = compare_provider_identities(
        {"unicode": "λ", "nested": {"z": 2, "a": [1, True, None]}},
        domain="intent",
        schema_version="1.0.0",
    )
    providers_seen = {
        (item.left_provider, item.right_provider) for item in contradictions
    }
    # At least the known unicode gap (cid_utils vs ipld_cid) and IR divergence.
    assert any(PROVIDER_CID_UTILS in pair and PROVIDER_IPLD_CID in pair for pair in providers_seen)
    assert any(
        PROVIDER_IR_CORE_IDENTITY in pair for pair in providers_seen
    )


def test_identity_record_hexdigest_property() -> None:
    identity = identify_strict_artifact({"a": 1})
    assert identity.hexdigest == identity.digest.removeprefix("sha256:")
    assert len(identity.hexdigest) == 64


def test_decode_rejects_uppercase_cid() -> None:
    identity = identify_strict_artifact({"a": 1})
    with pytest.raises(CidValidationError, match="lowercase"):
        decode_and_verify_cid(
            identity.cid.upper(),
            identity.canonical_bytes,
            expected_codec=MULTICODEC_DAG_JSON,
        )


def test_decode_rejects_codec_mismatch() -> None:
    identity = identify_strict_artifact({"a": 1})
    with pytest.raises(CidValidationError) as exc_info:
        decode_and_verify_cid(
            identity.cid,
            identity.canonical_bytes,
            expected_codec=MULTICODEC_RAW,
        )
    assert exc_info.value.reason_code in {
        "multicodec_mismatch",
        "cid_validation_failed",
    }


def test_module_exports_content_identity_interface() -> None:
    assert bridge.CONTENT_IDENTITY_INTERFACE == "ContentIdentity@1"
    assert "ContentIdentity" in bridge.__all__
    assert "identify_strict_artifact" in bridge.__all__
    assert "identify_logic_ir" in bridge.__all__
    assert "compare_provider_identities" in bridge.__all__


def test_lazy_import_cache_survives_repeated_calls() -> None:
    first = identify_strict_artifact({"cache": 1})
    second = identify_strict_artifact({"cache": 1})
    assert first.cid == second.cid
    # Modules remain cached after success.
    assert bridge.provider_available(PROVIDER_CID_UTILS) is True


def test_profile_contradiction_to_dict_is_json_ready() -> None:
    contradictions = compare_provider_identities(
        {"unicode": "café"},
        providers=(PROVIDER_CID_UTILS, PROVIDER_IPLD_CID),
    )
    assert contradictions
    payload = contradictions[0].to_dict()
    assert isinstance(payload["kind"], str)
    assert "left_provider" in payload
    assert "reason_code" in payload


def test_importlib_reload_keeps_public_api() -> None:
    reloaded = importlib.reload(bridge)
    assert reloaded.STRICT_ARTIFACT_PROFILE == STRICT_ARTIFACT_PROFILE
    identity = reloaded.identify_strict_artifact({"reload": True})
    assert identity.profile == STRICT_ARTIFACT_PROFILE
    reset_provider_import_cache()
