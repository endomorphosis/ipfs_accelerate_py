"""SCA-220: exact CID, multiformats, and multihash conformance.

Real datasets CID helpers and ``multiformats.CID`` / ``multiformats.multihash``
are invoked when available.  Compatible providers must agree on canonical
bytes and the decoded multihash digest; altered bytes, codecs, and profiles
must fail; missing or incompatible providers emit typed blockers.  No model
call occurs.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis import content_identity_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import (
    AUTHORITY_ROOT_KINDS,
    CONFORMANCE_AGREEMENT_PAYLOAD,
    CONTENT_IDENTITY_BRIDGE_INTERFACE,
    DATASETS_CONTENT_IDENTITY_CAPABILITY_ID,
    DATASETS_CONTENT_IDENTITY_SCHEMA,
    DEFAULT_CAPABILITY_RELATIVE_PATH,
    LOGIC_IR_PROFILE,
    MULTICODEC_DAG_JSON,
    MULTICODEC_RAW,
    MULTIHASH_SHA2_256,
    PROVIDER_CID_UTILS,
    PROVIDER_IPLD_CID,
    PROVIDER_IR_CORE_IDENTITY,
    PROVIDER_MULTIFORMATS,
    PROVIDER_MULTIFORMATS_CID,
    PROVIDER_MULTIFORMATS_MULTIHASH,
    PROVIDER_PROFILE_G,
    STRICT_ARTIFACT_PROFILE,
    CidValidationError,
    ContentIdentityError,
    MultiformatsUnavailableError,
    ProviderIncompatibleError,
    ProviderUnavailableError,
    TypedBlockerKind,
    bind_authority_root,
    build_datasets_content_identity_capability,
    decode_and_verify_cid,
    identify_logic_ir,
    identify_strict_artifact,
    inspect_provider_binding,
    invoke_multiformats_cid_and_multihash,
    is_digest_shaped,
    missing_provider_blockers,
    prove_content_identity_conformance,
    require_provider,
    reset_provider_import_cache,
    sha256_digest_label,
    write_datasets_content_identity_capability,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CAPABILITY_PATH = REPO_ROOT / DEFAULT_CAPABILITY_RELATIVE_PATH


@pytest.fixture(autouse=True)
def _clear_import_cache() -> Any:
    reset_provider_import_cache()
    yield
    reset_provider_import_cache()


def _uninstall_multiformats(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_real_datasets_cid_helpers_and_multiformats_are_invoked() -> None:
    receipt = prove_content_identity_conformance()

    assert receipt.model_calls == 0
    assert receipt.interface == CONTENT_IDENTITY_BRIDGE_INTERFACE
    assert receipt.multiformats_invoked[PROVIDER_MULTIFORMATS_CID] is True
    assert receipt.multiformats_invoked[PROVIDER_MULTIFORMATS_MULTIHASH] is True
    assert "CID.decode" in receipt.multiformats_invoked["symbols"]
    assert "multihash.digest" in receipt.multiformats_invoked["symbols"]

    provider_map = {item.module: item for item in receipt.providers}
    for name in (
        PROVIDER_CID_UTILS,
        PROVIDER_IPLD_CID,
        PROVIDER_PROFILE_G,
        PROVIDER_IR_CORE_IDENTITY,
        PROVIDER_MULTIFORMATS,
        PROVIDER_MULTIFORMATS_CID,
        PROVIDER_MULTIFORMATS_MULTIHASH,
    ):
        assert name in provider_map
        assert provider_map[name].available is True
        assert provider_map[name].compatible is True

    # Direct real-entrypoint invocation with multiformats.CID / multihash.
    identity = identify_strict_artifact(CONFORMANCE_AGREEMENT_PAYLOAD)
    mf = invoke_multiformats_cid_and_multihash(
        identity.canonical_bytes,
        codec=MULTICODEC_DAG_JSON,
    )
    assert mf["cid"] == identity.cid
    assert mf["raw_digest"] == hashlib.sha256(identity.canonical_bytes).hexdigest()
    assert mf["multicodec"] == MULTICODEC_DAG_JSON
    assert mf["multihash"] == MULTIHASH_SHA2_256

    from multiformats import CID, multihash

    parsed = CID.decode(identity.cid)
    assert bytes(parsed.raw_digest) == hashlib.sha256(identity.canonical_bytes).digest()
    assert multihash.digest(identity.canonical_bytes, "sha2-256") is not None


def test_all_dag_json_providers_agree_on_canonical_bytes_and_decoded_digest() -> None:
    receipt = prove_content_identity_conformance()
    assert receipt.agreement.get("agreed") is True
    assert set(receipt.agreement.get("providers_compared") or []) == {
        PROVIDER_CID_UTILS,
        PROVIDER_IPLD_CID,
        PROVIDER_PROFILE_G,
    }

    agreement_vector = next(
        item
        for item in receipt.positive_vectors
        if item.vector_id == "positive.providers_agree_canonical_bytes_and_digest"
    )
    assert agreement_vector.passed is True
    assert agreement_vector.raw_digest == hashlib.sha256(
        # Re-derive from cid_utils so the digest is over exact retained bytes.
        __import__(
            "ipfs_datasets_py.utils.cid_utils", fromlist=["canonical_dag_json_bytes"]
        ).canonical_dag_json_bytes(CONFORMANCE_AGREEMENT_PAYLOAD)
    ).hexdigest()

    # Independent snapshots must still match.
    snapshots = [
        bridge._provider_snapshot(
            name,
            CONFORMANCE_AGREEMENT_PAYLOAD,
            domain="sca-conformance",
            schema_version="1.0.0",
        )
        for name in (PROVIDER_CID_UTILS, PROVIDER_IPLD_CID, PROVIDER_PROFILE_G)
    ]
    first = snapshots[0]
    assert all(item["canonical_bytes"] == first["canonical_bytes"] for item in snapshots)
    assert all(item["cid"] == first["cid"] for item in snapshots)
    assert all(item["digest"] == first["digest"] for item in snapshots)
    verified = decode_and_verify_cid(
        first["cid"],
        first["canonical_bytes"],
        expected_codec=MULTICODEC_DAG_JSON,
    )
    assert verified["raw_digest"] == hashlib.sha256(first["canonical_bytes"]).hexdigest()


def test_positive_vectors_cover_strict_artifact_and_logic_ir() -> None:
    receipt = prove_content_identity_conformance()
    by_id = {item.vector_id: item for item in receipt.positive_vectors}

    strict = by_id["positive.strict_dag_json_cidv1_base32_sha2_256"]
    assert strict.passed is True
    assert strict.codec == MULTICODEC_DAG_JSON
    assert strict.multibase == "base32"
    assert strict.multihash == MULTIHASH_SHA2_256
    assert strict.cid.startswith("b")
    assert not is_digest_shaped(strict.cid)

    ir = by_id["positive.logic_ir_raw_codec_domain_separated"]
    assert ir.passed is True
    assert ir.profile == LOGIC_IR_PROFILE
    assert ir.codec == MULTICODEC_RAW


def test_altered_bytes_codec_and_profile_vectors_fail() -> None:
    receipt = prove_content_identity_conformance()
    by_id = {item.vector_id: item for item in receipt.negative_vectors}

    assert by_id["negative.altered_bytes_must_fail"].passed is True
    assert by_id["negative.codec_mismatch_must_fail"].passed is True
    assert by_id["negative.digest_labeled_as_cid_must_fail"].passed is True
    assert by_id["negative.cross_profile_equality_forbidden"].passed is True

    identity = identify_strict_artifact(CONFORMANCE_AGREEMENT_PAYLOAD)
    with pytest.raises(CidValidationError) as altered:
        decode_and_verify_cid(
            identity.cid,
            b'{"not":"the-preimage"}',
            expected_codec=MULTICODEC_DAG_JSON,
        )
    assert "multihash_digest_mismatch" in altered.value.reason_code or (
        "multihash_digest_mismatch" in str(altered.value)
    )

    with pytest.raises(CidValidationError) as codec:
        decode_and_verify_cid(
            identity.cid,
            identity.canonical_bytes,
            expected_codec=MULTICODEC_RAW,
        )
    assert codec.value.reason_code in {
        "multicodec_mismatch",
        "cid_validation_failed",
    }

    ir = identify_logic_ir(
        CONFORMANCE_AGREEMENT_PAYLOAD,
        domain="sca-conformance",
        schema_version="1.0.0",
    )
    assert identity.profile != ir.profile
    assert identity.multicodec != ir.multicodec
    assert bridge.profiles_are_interchangeable(identity, ir) is False


def test_missing_multiformats_emits_typed_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _uninstall_multiformats(monkeypatch)

    binding = inspect_provider_binding(PROVIDER_MULTIFORMATS)
    assert binding.available is False
    assert binding.blocker is not None
    assert binding.blocker.kind is TypedBlockerKind.MULTIFORMATS_UNAVAILABLE

    blockers = missing_provider_blockers(
        (
            PROVIDER_MULTIFORMATS,
            PROVIDER_MULTIFORMATS_CID,
            PROVIDER_MULTIFORMATS_MULTIHASH,
        )
    )
    assert blockers
    assert all(
        item.kind is TypedBlockerKind.MULTIFORMATS_UNAVAILABLE for item in blockers
    )

    with pytest.raises(MultiformatsUnavailableError):
        require_provider(PROVIDER_MULTIFORMATS)
    with pytest.raises(MultiformatsUnavailableError):
        invoke_multiformats_cid_and_multihash(b"{}", codec=MULTICODEC_DAG_JSON)

    receipt = prove_content_identity_conformance()
    assert receipt.passed is False
    assert receipt.model_calls == 0
    assert any(
        item.kind is TypedBlockerKind.MULTIFORMATS_UNAVAILABLE
        for item in receipt.blockers
    )


def test_incompatible_provider_emits_typed_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Import the real module first, then strip a required symbol.
    module = require_provider(PROVIDER_CID_UTILS)
    monkeypatch.delattr(module, "canonical_dag_json_bytes", raising=False)
    reset_provider_import_cache()
    # Re-seed the cache with the mutilated module.
    bridge._MODULE_CACHE[PROVIDER_CID_UTILS] = module

    binding = inspect_provider_binding(PROVIDER_CID_UTILS)
    assert binding.available is True
    assert binding.compatible is False
    assert binding.blocker is not None
    assert binding.blocker.kind is TypedBlockerKind.INCOMPATIBLE_PROVIDER
    assert "canonical_dag_json_bytes" in binding.missing_symbols

    with pytest.raises(ProviderIncompatibleError) as exc_info:
        require_provider(PROVIDER_CID_UTILS)
    assert exc_info.value.reason_code == "provider_incompatible"


def test_missing_datasets_provider_is_typed_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seed the lazy import cache with a failed lookup so the fail-closed
    # path is exercised without fighting site-packages already on sys.path.
    cause = ModuleNotFoundError(PROVIDER_CID_UTILS)
    with bridge._IMPORT_LOCK:
        bridge._MODULE_CACHE[PROVIDER_CID_UTILS] = None
        bridge._MODULE_ERRORS[PROVIDER_CID_UTILS] = cause

    binding = inspect_provider_binding(PROVIDER_CID_UTILS)
    assert binding.available is False
    assert binding.blocker is not None
    assert binding.blocker.kind is TypedBlockerKind.MISSING_PROVIDER

    with pytest.raises(ProviderUnavailableError):
        require_provider(PROVIDER_CID_UTILS)


def test_authority_roots_bind_to_canonical_bytes_and_distinct_cids() -> None:
    receipt = prove_content_identity_conformance()
    roots = receipt.root_bindings
    assert roots.get("distinct_cids") is True
    assert set(roots.get("kinds") or []) == set(AUTHORITY_ROOT_KINDS)

    seen: set[str] = set()
    for kind in AUTHORITY_ROOT_KINDS:
        bound = bind_authority_root(kind, {"example": kind, "n": 1})
        assert bound.profile == STRICT_ARTIFACT_PROFILE
        assert bound.validated is True
        assert bound.cid.startswith("b")
        assert bound.digest == sha256_digest_label(bound.canonical_bytes)
        verified = decode_and_verify_cid(
            bound.cid,
            bound.canonical_bytes,
            expected_codec=MULTICODEC_DAG_JSON,
        )
        assert verified["raw_digest"] == bound.hexdigest
        assert bound.cid not in seen
        seen.add(bound.cid)

    with pytest.raises(ContentIdentityError, match="unknown authority root"):
        bind_authority_root("not-a-root", {"x": 1})


def test_conformance_receipt_passes_without_model_calls() -> None:
    receipt = prove_content_identity_conformance()
    assert receipt.passed is True
    assert receipt.model_calls == 0
    assert receipt.capability_id == DATASETS_CONTENT_IDENTITY_CAPABILITY_ID
    assert receipt.schema == DATASETS_CONTENT_IDENTITY_SCHEMA
    assert receipt.artifact_profile == STRICT_ARTIFACT_PROFILE
    assert receipt.logic_ir_profile == LOGIC_IR_PROFILE
    assert receipt.blockers == ()
    payload = receipt.to_dict()
    assert payload["model_calls"] == 0
    assert payload["policies"]["decoded_multihash_must_match_canonical_bytes"] is True
    assert payload["policies"]["missing_or_incompatible_provider"] == "typed_blocker"
    assert payload["policies"]["cross_profile_equality_allowed"] is False


def test_capability_document_is_written_and_matches_live_conformance() -> None:
    receipt = prove_content_identity_conformance()
    assert receipt.passed is True

    written = write_datasets_content_identity_capability(
        CAPABILITY_PATH,
        receipt=receipt,
    )
    assert written == CAPABILITY_PATH
    assert CAPABILITY_PATH.is_file()

    on_disk = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    built = build_datasets_content_identity_capability(receipt=receipt)

    assert on_disk["schema"] == DATASETS_CONTENT_IDENTITY_SCHEMA
    assert on_disk["capability_id"] == DATASETS_CONTENT_IDENTITY_CAPABILITY_ID
    assert on_disk["interface"] == CONTENT_IDENTITY_BRIDGE_INTERFACE
    assert on_disk["task_id"] == "SCA-220"
    assert on_disk["passed"] is True
    assert on_disk["model_calls"] == 0
    assert on_disk["artifact_profile"]["multicodec"] == MULTICODEC_DAG_JSON
    assert on_disk["artifact_profile"]["multihash"] == MULTIHASH_SHA2_256
    assert on_disk["logic_ir_profile"]["multicodec"] == MULTICODEC_RAW
    assert on_disk["conformance"]["passed"] is True
    assert on_disk["conformance"]["agreement"]["agreed"] is True
    assert on_disk["policies"]["missing_or_incompatible_provider"] == "typed_blocker"
    assert on_disk["root_bindings"]["distinct_cids"] is True

    # Portable document: no absolute host paths.
    for provider in on_disk["providers"]:
        module_file = provider.get("module_file", "")
        assert not module_file.startswith("/")
        assert "\\" not in module_file

    # Live rebuild must match the durable artifact for identity fields.
    assert on_disk["capability_id"] == built["capability_id"]
    assert on_disk["conformance"]["agreement"]["cid"] == built["conformance"]["agreement"]["cid"]
    assert (
        on_disk["conformance"]["agreement"]["raw_digest"]
        == built["conformance"]["agreement"]["raw_digest"]
    )


def test_module_exports_bridge_conformance_api() -> None:
    assert bridge.CONTENT_IDENTITY_BRIDGE_INTERFACE == "ContentIdentityBridge@1"
    assert "prove_content_identity_conformance" in bridge.__all__
    assert "write_datasets_content_identity_capability" in bridge.__all__
    assert "bind_authority_root" in bridge.__all__
    assert "ProviderIncompatibleError" in bridge.__all__
    assert "TypedBlocker" in bridge.__all__


def test_no_model_call_surface_in_conformance_module_source() -> None:
    source = Path(bridge.__file__).read_text(encoding="utf-8")
    # Conformance must remain deterministic; reject LLM/model client wiring.
    forbidden = (
        "openai",
        "anthropic",
        "ChatCompletion",
        "litellm",
        "grok_client",
    )
    lowered = source.lower()
    for token in forbidden:
        assert token.lower() not in lowered
    assert "model_calls" in source
