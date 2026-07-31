"""PTR-010: core locator and execution identity (ContentIdentity@1)."""

from __future__ import annotations

import hashlib
import json

import pytest
from multiformats import CID, multihash

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    CID_BASE,
    CID_CODEC,
    CID_VERSION,
    CONTENT_IDENTITY_INTERFACE,
    DIGEST_SIZE,
    MH_TYPE,
    REASON_CID_PROVIDER_UNAVAILABLE,
    REASON_NON_REUSABLE,
    TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE,
    CidSupportStatus,
    CompiledTestExecutionKey,
    CompiledTestLocator,
    ContentIdentity,
    TestExecutionIdentityCompiler,
    TestExecutionIdentityError,
    cid_support_available,
    compile_test_execution_key,
    compile_test_locator,
    content_identity_from_retained_bytes,
    mint_content_identity,
    normalize_pytest_node_id,
    probe_cid_support,
    reject_pseudo_cid,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    EligibilityClass,
    TestExecutionKey,
    TestLocatorKey,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _locator_fields(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:demo",
        "package_identity": "ipfs_accelerate_py",
        "node_id": "test/api/test_demo.py::test_alpha",
        "collection_schema_version": "1",
        "root_identity": "root:repo",
        "selection_semantics": "exact_node",
    }
    values.update(changes)
    return values


def _execution_fields(locator_cid: str, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "locator_cid": locator_cid,
        "repository_forest_cid": "forest:baguqeeratestforestidentity0001",
        "git_commit_id": "deadbeefcafebabe",
        "git_tree_id": "tree:abc123",
        "test_module_cid": "module:baguqeeramoduleidentity0001",
        "test_function_cid": "function:baguqeerafnidentity0001",
        "test_ast_cid": "ast:baguqeeraastidentity0001",
        "static_trace_root_cid": "static:baguqeerastaticidentity0001",
        "runtime_trace_root_cid": "runtime:baguqeeraruntimeidentity0001",
        "pytest_version": "8.2.0",
        "python_version": "3.11.9",
        "policy_cid": "policy:baguqeerapolicyidentity0001",
        "eligibility_class": EligibilityClass.REPOSITORY_FOREST_BOUND,
        "markers": ("unit", "slow"),
        "fixture_cids": ("fixture:b", "fixture:a"),
    }
    values.update(changes)
    return values


def _assert_profile_cid(cid: str, *, payload_bytes: bytes) -> None:
    """Assert CIDv1/base32/dag-json/sha2-256 and multihash == sha256(bytes).

    Uses the independent ``multiformats`` package only (no multiformats_identity
    / ipfs_datasets_py import path, which can fail under shadowing PYTHONPATH
    orderings during supervisor validation).
    """

    assert cid == cid.lower()
    assert cid.startswith("b")
    parsed = CID.decode(cid)
    assert parsed.version == CID_VERSION
    assert parsed.codec.name == CID_CODEC
    assert parsed.hashfun.name == MH_TYPE
    assert parsed.base.name == CID_BASE
    assert len(parsed.raw_digest) == DIGEST_SIZE
    assert bytes(parsed.raw_digest) == hashlib.sha256(payload_bytes).digest()
    independent = str(
        CID(CID_BASE, CID_VERSION, CID_CODEC, multihash.digest(payload_bytes, MH_TYPE))
    )
    assert independent == cid


# ---------------------------------------------------------------------------
# ContentIdentity mint / retain / rehash
# ---------------------------------------------------------------------------


def test_mint_content_identity_stable_cidv1_profile() -> None:
    payload = {"alpha": 1, "beta": ["x", "y"], "schema": "probe"}
    first = mint_content_identity(payload)
    second = mint_content_identity({"beta": ["x", "y"], "alpha": 1, "schema": "probe"})

    assert first.interface == CONTENT_IDENTITY_INTERFACE
    assert first.cid == second.cid
    assert first.digest_hex == second.digest_hex
    assert first.canonical_bytes == second.canonical_bytes
    _assert_profile_cid(first.cid, payload_bytes=first.canonical_bytes)
    assert first.verify() is first
    assert first.rehash_cid() == first.cid


def test_retained_canonical_bytes_decode_and_rehash() -> None:
    payload = {"interface": "TestLocatorKey@1", "node": "n1", "v": 1}
    identity = mint_content_identity(payload)

    # Decode retained bytes and re-encode must match.
    parsed = json.loads(identity.canonical_bytes.decode("utf-8"))
    rebuilt = content_identity_from_retained_bytes(
        identity.canonical_bytes, claimed_cid=identity.cid
    )
    assert rebuilt.cid == identity.cid
    assert rebuilt.digest_hex == identity.digest_hex
    assert rebuilt.canonical_bytes == identity.canonical_bytes
    assert (
        bytes(CID.decode(rebuilt.cid).raw_digest).hex()
        == hashlib.sha256(rebuilt.canonical_bytes).hexdigest()
    )
    assert parsed["node"] == "n1"


def test_retained_bytes_claimed_cid_mismatch_fails() -> None:
    identity = mint_content_identity({"k": "v"})
    other = mint_content_identity({"k": "other"})
    with pytest.raises(TestExecutionIdentityError, match="claimed CID"):
        content_identity_from_retained_bytes(
            identity.canonical_bytes, claimed_cid=other.cid
        )


def test_content_identity_rejects_digest_mismatch() -> None:
    identity = mint_content_identity({"z": 1})
    with pytest.raises(TestExecutionIdentityError, match="digest_hex"):
        ContentIdentity(
            cid=identity.cid,
            digest_hex="0" * 64,
            canonical_bytes=identity.canonical_bytes,
        )


def test_reject_pseudo_cid_forms() -> None:
    with pytest.raises(TestExecutionIdentityError, match="pseudo-hash"):
        reject_pseudo_cid("cid:forest:v1")
    with pytest.raises(TestExecutionIdentityError, match="pseudo-hash"):
        reject_pseudo_cid("sha256:" + "ab" * 32)
    with pytest.raises(TestExecutionIdentityError, match="pseudo-hash"):
        reject_pseudo_cid("runtime-artifact:sha256:" + "cd" * 32)
    with pytest.raises(TestExecutionIdentityError, match="CIDv0"):
        reject_pseudo_cid("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG")
    with pytest.raises(TestExecutionIdentityError, match="truncated"):
        reject_pseudo_cid("bafy")

    real = mint_content_identity({"ok": True}).cid
    assert reject_pseudo_cid(real) == real


# ---------------------------------------------------------------------------
# Locator compilation
# ---------------------------------------------------------------------------


def test_compile_test_locator_stable_and_profile() -> None:
    a = compile_test_locator(**_locator_fields())
    b = compile_test_locator(**_locator_fields())

    assert a.reusable is True
    assert b.reusable is True
    assert a.locator_cid == b.locator_cid
    assert a.content_identity is not None
    _assert_profile_cid(
        a.locator_cid, payload_bytes=a.content_identity.canonical_bytes
    )
    assert a.locator is not None
    assert a.locator.content_id == a.locator_cid
    assert a.content_identity.verify().cid == a.locator_cid
    # Multiformats independent agreement on the retained canonical bytes.
    independent = str(
        CID(
            CID_BASE,
            CID_VERSION,
            CID_CODEC,
            multihash.digest(a.content_identity.canonical_bytes, MH_TYPE),
        )
    )
    assert independent == a.locator_cid
    assert a.locator.canonical_bytes() == a.content_identity.canonical_bytes


def test_compile_test_locator_from_contract_instance() -> None:
    key = TestLocatorKey(**_locator_fields())  # type: ignore[arg-type]
    compiled = compile_test_locator(key)
    assert compiled.reusable is True
    assert compiled.locator_cid == key.content_id


def test_compile_test_locator_normalizes_node_id() -> None:
    compiled = compile_test_locator(
        **_locator_fields(node_id=r"test\\api//test_demo.py::test_alpha")
    )
    assert compiled.reusable is True
    assert compiled.locator is not None
    assert compiled.locator.node_id == normalize_pytest_node_id(
        r"test\\api//test_demo.py::test_alpha"
    )
    assert "\\" not in compiled.locator.node_id
    assert "//" not in compiled.locator.node_id


def test_compile_test_locator_parameter_non_reusable_retains_cid() -> None:
    compiled = compile_test_locator(
        **_locator_fields(
            parameter_id="p0",
            non_reusable_reason="unsupported_parameter_type",
        )
    )
    assert compiled.reusable is False
    assert compiled.reason_code == REASON_NON_REUSABLE
    assert compiled.non_reusable_reason == "unsupported_parameter_type"
    assert compiled.locator_cid
    assert compiled.content_identity is not None
    _assert_profile_cid(
        compiled.locator_cid,
        payload_bytes=compiled.content_identity.canonical_bytes,
    )


def test_compile_test_locator_malformed_is_non_reusable() -> None:
    compiled = compile_test_locator(
        repository_id="repo",
        package_identity="pkg",
        # missing node_id
    )
    assert compiled.reusable is False
    assert compiled.locator_cid == ""
    assert compiled.content_identity is None
    assert "malformed" in compiled.non_reusable_reason or compiled.reason_code


def test_compile_test_locator_missing_cid_support_non_reusable() -> None:
    compiler = TestExecutionIdentityCompiler(
        cid_probe=lambda: CidSupportStatus.MISSING,
    )
    compiled = compiler.compile_locator(**_locator_fields())
    assert compiled.reusable is False
    assert compiled.reason_code == REASON_CID_PROVIDER_UNAVAILABLE
    assert compiled.locator_cid == ""
    assert compiled.content_identity is None
    assert compiled.cid_support is CidSupportStatus.MISSING
    # Must not invent a pseudo-CID.
    assert not compiled.locator_cid.startswith("sha256:")
    assert not compiled.locator_cid.startswith("cid:")


def test_compile_test_locator_incompatible_cid_support_non_reusable() -> None:
    compiler = TestExecutionIdentityCompiler(
        cid_probe=lambda: CidSupportStatus.INCOMPATIBLE,
    )
    compiled = compiler.compile_locator(**_locator_fields())
    assert compiled.reusable is False
    assert compiled.reason_code == REASON_CID_PROVIDER_UNAVAILABLE
    assert compiled.locator_cid == ""


# ---------------------------------------------------------------------------
# Execution key compilation
# ---------------------------------------------------------------------------


def test_compile_test_execution_key_stable_and_profile() -> None:
    locator = compile_test_locator(**_locator_fields())
    assert locator.reusable and locator.locator_cid

    first = compile_test_execution_key(
        **_execution_fields(locator.locator_cid)
    )
    second = compile_test_execution_key(
        **_execution_fields(locator.locator_cid)
    )

    assert first.reusable is True
    assert first.execution_cid == second.execution_cid
    assert first.locator_cid == locator.locator_cid
    assert first.content_identity is not None
    _assert_profile_cid(
        first.execution_cid,
        payload_bytes=first.content_identity.canonical_bytes,
    )
    assert first.execution_key is not None
    assert first.execution_key.content_id == first.execution_cid
    first.content_identity.verify()
    assert first.content_identity.rehash_cid() == first.execution_cid


def test_any_bound_change_changes_execution_cid() -> None:
    locator = compile_test_locator(**_locator_fields())
    base = compile_test_execution_key(**_execution_fields(locator.locator_cid))
    assert base.reusable is True

    mutations: list[dict[str, object]] = [
        {"repository_forest_cid": "forest:changed"},
        {"git_commit_id": "c0ffee"},
        {"test_module_cid": "module:changed"},
        {"test_ast_cid": "ast:changed"},
        {"policy_cid": "policy:changed"},
        {"pytest_version": "8.3.0"},
        {"python_version": "3.12.0"},
        {"markers": ("unit",)},
        {"fixture_cids": ("fixture:z",)},
        {"environment_cid": "env:changed"},
        {"static_trace_root_cid": "static:changed"},
        {"runtime_trace_root_cid": "runtime:changed"},
        {"eligibility_class": EligibilityClass.PURE},
    ]
    # Locator change (different locator CID) also changes execution identity.
    other_locator = compile_test_locator(
        **_locator_fields(node_id="test/api/test_demo.py::test_beta")
    )
    mutations.append({"locator_cid": other_locator.locator_cid})

    seen = {base.execution_cid}
    for change in mutations:
        fields = _execution_fields(locator.locator_cid)
        fields.update(change)
        compiled = compile_test_execution_key(**fields)
        assert compiled.reusable is True, change
        assert compiled.execution_cid not in seen, change
        seen.add(compiled.execution_cid)


def test_compile_execution_key_from_contract_instance() -> None:
    locator = compile_test_locator(**_locator_fields())
    key = TestExecutionKey(
        **_execution_fields(locator.locator_cid)  # type: ignore[arg-type]
    )
    compiled = compile_test_execution_key(key)
    assert compiled.reusable is True
    assert compiled.execution_cid == key.content_id


def test_execution_key_eligibility_non_reusable_still_has_cid() -> None:
    locator = compile_test_locator(**_locator_fields())
    compiled = compile_test_execution_key(
        **_execution_fields(
            locator.locator_cid,
            eligibility_class=EligibilityClass.NON_REUSABLE,
        )
    )
    assert compiled.reusable is False
    assert compiled.reason_code == REASON_NON_REUSABLE
    assert "non_reusable" in compiled.non_reusable_reason
    assert compiled.execution_cid
    assert compiled.content_identity is not None
    _assert_profile_cid(
        compiled.execution_cid,
        payload_bytes=compiled.content_identity.canonical_bytes,
    )


def test_compile_execution_key_missing_cid_support_non_reusable() -> None:
    compiler = TestExecutionIdentityCompiler(
        cid_probe=lambda: CidSupportStatus.MISSING,
    )
    compiled = compiler.compile_execution_key(
        **_execution_fields("baguqeeraplacesholdercidvalue01")
    )
    assert compiled.reusable is False
    assert compiled.reason_code == REASON_CID_PROVIDER_UNAVAILABLE
    assert compiled.execution_cid == ""
    assert compiled.content_identity is None


def test_compile_execution_key_malformed_non_reusable() -> None:
    # Missing required locator_cid / forest.
    compiled = compile_test_execution_key(git_commit_id="x")
    assert compiled.reusable is False
    assert compiled.execution_cid == ""
    assert compiled.content_identity is None


# ---------------------------------------------------------------------------
# Compiler surface / probes
# ---------------------------------------------------------------------------


def test_compiler_interface_and_probe_available() -> None:
    compiler = TestExecutionIdentityCompiler()
    assert compiler.interface == TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE
    status = compiler.probe()
    assert status is CidSupportStatus.AVAILABLE
    assert probe_cid_support() is CidSupportStatus.AVAILABLE
    assert cid_support_available() is True


def test_compiled_artifacts_to_dict_shape() -> None:
    locator = compile_test_locator(**_locator_fields())
    execution = compile_test_execution_key(
        **_execution_fields(locator.locator_cid)
    )
    loc_dict = locator.to_dict()
    exec_dict = execution.to_dict()
    assert loc_dict["reusable"] is True
    assert loc_dict["locator_cid"] == locator.locator_cid
    assert loc_dict["content_identity"]["interface"] == CONTENT_IDENTITY_INTERFACE
    assert exec_dict["execution_cid"] == execution.execution_cid
    assert exec_dict["locator_cid"] == locator.locator_cid


def test_normalize_pytest_node_id() -> None:
    assert (
        normalize_pytest_node_id("  path\\to//test.py::test_x  ")
        == "path/to/test.py::test_x"
    )
    with pytest.raises(TestExecutionIdentityError):
        normalize_pytest_node_id("   ")


def test_module_entry_points_match_compiler() -> None:
    fields = _locator_fields()
    via_fn = compile_test_locator(**fields)
    via_cls = TestExecutionIdentityCompiler().compile_locator(**fields)
    assert via_fn.locator_cid == via_cls.locator_cid

    exec_fields = _execution_fields(via_fn.locator_cid)
    e1 = compile_test_execution_key(**exec_fields)
    e2 = TestExecutionIdentityCompiler().compile_execution_key(**exec_fields)
    assert e1.execution_cid == e2.execution_cid


def test_key_order_does_not_affect_execution_cid() -> None:
    locator = compile_test_locator(**_locator_fields())
    a = compile_test_execution_key(
        **_execution_fields(
            locator.locator_cid,
            metadata={"b": 2, "a": 1},
            components={"z": "1", "a": "2"},
        )
    )
    b = compile_test_execution_key(
        **_execution_fields(
            locator.locator_cid,
            metadata={"a": 1, "b": 2},
            components={"a": "2", "z": "1"},
        )
    )
    assert a.execution_cid == b.execution_cid
    assert a.content_identity is not None
    assert b.content_identity is not None
    assert a.content_identity.canonical_bytes == b.content_identity.canonical_bytes


def test_compiled_types_are_not_collected_as_tests() -> None:
    assert getattr(ContentIdentity, "__test__", True) is False
    assert getattr(CompiledTestLocator, "__test__", True) is False
    assert getattr(CompiledTestExecutionKey, "__test__", True) is False
    assert getattr(TestExecutionIdentityCompiler, "__test__", True) is False
    assert getattr(TestExecutionIdentityError, "__test__", True) is False
