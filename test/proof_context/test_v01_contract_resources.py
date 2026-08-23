"""PCCE-010: MCP++ contract resources are installed-package only."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.proof_context.compatibility import CompatibilityError
from ipfs_accelerate_py.proof_context.contract_resources import (
    ContractResourceUnavailable,
    admit_cid,
    admit_schema_mapping,
    load_schema_text,
)


def test_missing_installed_contracts_are_unavailable() -> None:
    with pytest.raises(ContractResourceUnavailable):
        load_schema_text("proof-context/v0.1/task-specification.schema.json")


def test_does_not_consult_sibling_source_tree() -> None:
    sibling = (
        Path(__file__).resolve().parents[4] / "Mcp-Plus-Plus" / "schemas"
    )
    # Loader must fail closed even if a sibling checkout exists nearby.
    with pytest.raises((ContractResourceUnavailable, CompatibilityError)):
        load_schema_text(str(sibling / "proof-context/v0.1/task-specification.schema.json"))
    with pytest.raises(CompatibilityError):
        load_schema_text("/etc/passwd")
    with pytest.raises(CompatibilityError):
        load_schema_text("../Mcp-Plus-Plus/schemas/x.json")


def test_explicit_mapping_and_cid_admission() -> None:
    schema = admit_schema_mapping(
        {
            "schema": "pcce/proof-context/v0.1/task-specification",
            "$id": "https://mcp-plus-plus.dev/schemas/proof-context/v0.1/task-specification.schema.json",
        }
    )
    assert "task-specification" in schema["schema"]
    admit_cid("bafkreiapj52u5hi7pco5ebplvecv72olbnqglg2e7emwnmme4gguzsnpu4")
    with pytest.raises(CompatibilityError):
        admit_cid("sha256:abc")
