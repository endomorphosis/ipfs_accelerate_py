"""DCR-020: canonical MCP contract, schema, method, and CID identity tests.

All conformance vectors remain inline.  This suite must not write undeclared
vector artifacts under data/agent_supervisor/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_identity import (
    CANONICAL_CONTRACT_IDENTITY_INTERFACE,
    CANONICAL_CONTRACT_IDENTITY_SCHEMA,
    CONTRACT_IDENTITY_EVIDENCE_TERM,
    SEMANTIC_CONTRACT_KEY_INTERFACE,
    SEMANTIC_CONTRACT_KEY_SCHEMA,
    CanonicalContractIdentity,
    ClaimedCidMismatchError,
    ContractDirection,
    ContractIdentityDisposition,
    McpContractIdentityError,
    PseudoCidError,
    SemanticContractKey,
    canonical_json_cid,
    classify_alias_collision,
    compare_contract_identities,
    digest_for_canonical_bytes,
    identify_contract_declaration,
    identities_converge,
    is_digest_shaped,
    is_pseudo_cid,
    semantic_contract_key,
    validate_multiformat_cid,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)

# test/api -> test -> ipfs_accelerate package root -> external -> workspace root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DCR_DATA = _REPO_ROOT / "data" / "agent_supervisor" / "deterministic_contract_repair"
_UNDECLARED_VECTOR = _DCR_DATA / "mcp_contract_identity_vectors.json"


def _declaration(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "method": "tools/call",
        "tool": "echo",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    }
    payload.update(overrides)
    return payload


def _identify(**overrides: object) -> CanonicalContractIdentity:
    values: dict[str, object] = {
        "package": "ipfs_accelerate_py",
        "operation": "tools.call.echo",
        "direction": ContractDirection.REQUEST,
        "schema_root": "schemas/echo.request.json",
        "profile": "mcp++/profile-h",
        "transport": "stdio",
        "declaration": _declaration(),
        "source_roots": ("external/ipfs_accelerate", "Mcp-Plus-Plus"),
        "aliases": ("echo",),
    }
    values.update(overrides)
    return identify_contract_declaration(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Canonical JSON / CID
# ---------------------------------------------------------------------------


def test_canonical_json_cid_matches_supervisor_content_identity() -> None:
    payload = {"b": 2, "a": 1, "nested": {"z": True, "a": None}}
    assert canonical_json_cid(payload) == content_identity(payload)
    assert canonical_json_cid(payload).startswith("b")
    # Key order must not matter.
    assert canonical_json_cid({"a": 1, "b": 2}) == canonical_json_cid({"b": 2, "a": 1})


def test_canonical_json_cid_rejects_non_json_values() -> None:
    with pytest.raises(McpContractIdentityError, match="canonical-JSON"):
        canonical_json_cid({"bad": object()})


def test_digest_is_never_accepted_as_cid() -> None:
    digest = digest_for_canonical_bytes(b'{"a":1}')
    assert is_digest_shaped(digest)
    assert is_pseudo_cid(digest)
    with pytest.raises(PseudoCidError, match="digest"):
        validate_multiformat_cid(digest)
    with pytest.raises(PseudoCidError):
        validate_multiformat_cid(digest.removeprefix("sha256:"))


def test_validate_multiformat_cid_accepts_content_identity() -> None:
    cid = content_identity({"fixture": "dcr-020"})
    assert validate_multiformat_cid(cid) == cid


# ---------------------------------------------------------------------------
# Semantic keys
# ---------------------------------------------------------------------------


def test_semantic_contract_key_is_stable_and_content_addressed() -> None:
    first = semantic_contract_key(
        package="swissknife",
        operation="desktop.tools.list",
        direction="request",
        schema_root="idl/tools.list.json",
        profile="desktop",
        transport="orb",
        runtime_instance="virtual-desktop",
    )
    second = SemanticContractKey.from_dict(first.to_dict())
    assert first.key_id == second.key_id
    assert first.key_id == canonical_json_cid(first._identity_payload())
    assert first.interface == SEMANTIC_CONTRACT_KEY_INTERFACE
    assert first.schema == SEMANTIC_CONTRACT_KEY_SCHEMA


def test_direction_and_profile_changes_remain_distinct() -> None:
    base = dict(
        package="ipfs_datasets_py",
        operation="tools.call.search",
        schema_root="schemas/search.json",
        profile="mcp++/default",
        transport="http",
    )
    request = semantic_contract_key(direction=ContractDirection.REQUEST, **base)
    response = semantic_contract_key(direction=ContractDirection.RESPONSE, **base)
    other_profile = semantic_contract_key(
        direction=ContractDirection.REQUEST,
        **{**base, "profile": "mcp++/profile-h"},
    )
    assert request.key_id != response.key_id
    assert request.key_id != other_profile.key_id


def test_absolute_schema_root_is_rejected() -> None:
    with pytest.raises(McpContractIdentityError, match="relocation-stable"):
        semantic_contract_key(
            package="pkg",
            operation="op",
            direction="request",
            schema_root="/tmp/host/schemas/echo.json",
            profile="default",
            transport="stdio",
        )


def test_forged_semantic_key_id_is_rejected() -> None:
    key = semantic_contract_key(
        package="pkg",
        operation="op",
        direction="method",
        schema_root="schemas/op.json",
        profile="default",
        transport="stdio",
    )
    payload = key.to_dict()
    payload["key_id"] = content_identity({"forged": True})
    with pytest.raises(McpContractIdentityError, match="key_id"):
        SemanticContractKey.from_dict(payload)


# ---------------------------------------------------------------------------
# Full identity records
# ---------------------------------------------------------------------------


def test_equivalent_declarations_converge() -> None:
    left = _identify()
    right = _identify(
        declaration=_declaration(),  # equivalent body, different dict order later
        aliases=("echo",),
    )
    # Key order inside declaration must not change identity.
    reordered = _identify(
        declaration={
            "output_schema": left.declaration["output_schema"],
            "method": "tools/call",
            "tool": "echo",
            "input_schema": left.declaration["input_schema"],
        }
    )
    assert identities_converge(left, right)
    assert identities_converge(left, reordered)
    assert left.local_cid == reordered.local_cid
    assert compare_contract_identities(left, reordered) is (
        ContractIdentityDisposition.CONVERGENT
    )
    assert left.interface == CANONICAL_CONTRACT_IDENTITY_INTERFACE
    assert left.schema == CANONICAL_CONTRACT_IDENTITY_SCHEMA
    assert left.evidence_term == CONTRACT_IDENTITY_EVIDENCE_TERM


def test_altered_bytes_remain_distinct() -> None:
    base = _identify()
    altered = _identify(
        declaration=_declaration(tool="echo2"),
        aliases=("echo2",),
    )
    assert not identities_converge(base, altered)
    assert base.local_cid != altered.local_cid
    assert compare_contract_identities(base, altered) is (
        ContractIdentityDisposition.DISTINCT
    )


def test_claimed_cid_mismatch_is_typed() -> None:
    local = _identify()
    other = content_identity({"different": "body"})
    mismatched = _identify(claimed_cid=other)
    assert mismatched.disposition is ContractIdentityDisposition.CLAIMED_CID_MISMATCH
    assert "claimed_cid_mismatch" in mismatched.reason_codes
    assert mismatched.claimed_cid == other
    assert mismatched.local_cid != other
    with pytest.raises(ClaimedCidMismatchError):
        _identify(claimed_cid=other, require_claimed_match=True)


def test_matching_claimed_cid_is_retained() -> None:
    first = _identify()
    second = _identify(claimed_cid=first.local_cid)
    assert second.disposition is ContractIdentityDisposition.CONVERGENT
    assert second.claimed_matches_local
    round_trip = CanonicalContractIdentity.from_dict(second.to_dict())
    assert round_trip.content_id == second.content_id
    assert round_trip.local_cid == second.local_cid


def test_pseudo_cid_claims_are_typed() -> None:
    digest_claim = _identify(
        claimed_cid="sha256:" + ("ab" * 32),
    )
    assert digest_claim.disposition is ContractIdentityDisposition.PSEUDO_CID
    bare = _identify(claimed_cid="a" * 64)
    assert bare.disposition is ContractIdentityDisposition.PSEUDO_CID
    garbage = _identify(claimed_cid="not-a-cid")
    assert garbage.disposition is ContractIdentityDisposition.PSEUDO_CID
    with pytest.raises(PseudoCidError):
        _identify(claimed_cid="sha256:" + ("cd" * 32), require_claimed_match=True)


def test_duplicate_aliases_remain_typed() -> None:
    left = _identify(aliases=("echo", "legacy.echo"))
    right = _identify(
        package="ipfs_kit_py",
        operation="tools.call.echo",
        aliases=("echo",),
        declaration=_declaration(package="kit"),
    )
    disposition, collisions = classify_alias_collision((left, right))
    assert disposition is ContractIdentityDisposition.DUPLICATE_ALIAS
    assert "echo" in collisions
    # Same local body under different alias metadata still converges on CID
    # but compare reports duplicate-alias when alias sets differ.
    same_body = _identify(aliases=("echo", "echo.v1"))
    base = _identify(aliases=("echo",))
    assert identities_converge(base, same_body)
    assert compare_contract_identities(base, same_body) is (
        ContractIdentityDisposition.DUPLICATE_ALIAS
    )


def test_direction_change_is_not_an_alias() -> None:
    request = _identify(direction=ContractDirection.REQUEST)
    response = _identify(direction=ContractDirection.RESPONSE)
    assert request.semantic_key.key_id != response.semantic_key.key_id
    assert compare_contract_identities(request, response) is (
        ContractIdentityDisposition.DISTINCT
    )


def test_forged_local_cid_rejected_on_decode() -> None:
    identity = _identify()
    payload = identity.to_dict()
    payload["local_cid"] = content_identity({"tampered": True})
    with pytest.raises(McpContractIdentityError, match="local_cid"):
        CanonicalContractIdentity.from_dict(payload)


def test_no_undeclared_vector_artifact_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity helpers must not emit a standalone vector artifact."""

    # Run identity work from a throwaway CWD; undeclared DCR vector path must
    # stay absent in the real repository data root.
    monkeypatch.chdir(tmp_path)
    identity = _identify()
    assert identity.local_cid
    _ = identity.to_dict()
    assert not _UNDECLARED_VECTOR.exists()
    # Taskboard declares Generated artifacts: none for DCR-020.
    assert not (tmp_path / "data").exists() or not any(
        (tmp_path / "data").rglob("*identity*vector*")
    )


def test_runtime_instance_is_part_of_semantic_identity() -> None:
    a = _identify(runtime_instance="")
    b = _identify(runtime_instance="svc-1")
    assert a.semantic_key.key_id != b.semantic_key.key_id
    assert a.local_cid != b.local_cid


def test_source_roots_reject_parent_traversal() -> None:
    with pytest.raises(McpContractIdentityError, match="relocation-stable"):
        _identify(source_roots=("external/../secrets",))


def test_policies_document_fail_closed_invariants() -> None:
    policies = _identify().to_dict()["policies"]
    assert policies["trust_claimed_cid"] is False
    assert policies["digest_labeled_as_cid_allowed"] is False
    assert policies["undeclared_vector_artifact_allowed"] is False
    assert policies["absolute_paths_allowed"] is False
