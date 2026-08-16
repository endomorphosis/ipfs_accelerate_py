"""DCR-020 canonical MCP contract identity tests."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_identity import (
    ClaimedMcpContractCidMismatch,
    McpContractIdentityError,
    canonical_mcp_contract_identity,
)


def _declaration(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "package": "@swissknife/desktop",
        "operation": "desktop.open",
        "direction": "request",
        "schema_root": "bafy-schema-root",
        "profile": "mcp++/profile-a@1.0",
        "transport": "http",
        "runtime_instance": "desktop-main",
        "authority_roots": {"descriptor": "bafy-descriptor", "policy": "bafy-policy"},
        "schema": {"required": ["path"], "type": "object"},
    }
    value.update(overrides)
    return value


def test_equivalent_relocated_declarations_converge_but_altered_bytes_bind_separately() -> None:
    first = canonical_mcp_contract_identity(_declaration(), source_bytes=b"descriptor at /old")
    second = canonical_mcp_contract_identity(
        _declaration(authority_roots={"policy": "bafy-policy", "descriptor": "bafy-descriptor"}),
        source_bytes=b"descriptor at /new",
    )

    assert first.semantic_cid == second.semantic_cid
    assert first.declaration_cid != second.declaration_cid
    assert first.semantic_key["runtime_instance"] == "desktop-main"
    assert set(first.semantic_key["authority_roots"]) == {"descriptor", "policy"}


def test_claimed_pseudo_cid_and_direction_or_profile_changes_fail_or_stay_distinct() -> None:
    identity = canonical_mcp_contract_identity(_declaration())
    assert (
        canonical_mcp_contract_identity(
            _declaration(), claimed_cid=identity.declaration_cid
        ).declaration_cid
        == identity.declaration_cid
    )
    with pytest.raises(ClaimedMcpContractCidMismatch):
        canonical_mcp_contract_identity(_declaration(), claimed_cid="sha256:not-a-cid")
    with pytest.raises(ClaimedMcpContractCidMismatch):
        canonical_mcp_contract_identity(_declaration(contract_cid="bafk-pseudo"))
    assert (
        canonical_mcp_contract_identity(_declaration(direction="result")).semantic_cid
        != identity.semantic_cid
    )
    assert (
        canonical_mcp_contract_identity(_declaration(profile="mcp++/profile-e@1.0")).semantic_cid
        != identity.semantic_cid
    )
    with pytest.raises(McpContractIdentityError, match="closed MCP contract direction"):
        canonical_mcp_contract_identity(_declaration(direction="bidirectional"))


def test_duplicate_aliases_are_not_collapsed_or_authoritative() -> None:
    identity = canonical_mcp_contract_identity(
        _declaration(aliases=["open", "open", "desktop-open"])
    )

    assert [item["alias"] for item in identity.alias_bindings] == ["open", "open", "desktop-open"]
    assert len({item["alias_cid"] for item in identity.alias_bindings}) == 3
    assert identity.alias_issues == (
        {"kind": "duplicate_alias", "alias": "open", "occurrence": "1"},
    )
    assert identity.to_dict()["claimed_cid_trusted"] is False
