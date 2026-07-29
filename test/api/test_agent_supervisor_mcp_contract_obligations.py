"""SCA-060 tests for canonical MCP++ logic obligations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractParityClaim,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    DEFAULT_MCP_CONTRACT_CATALOG,
    McpClaimFamily,
    ReviewState,
    admit_source,
    build_contract_from_sources,
    make_source_record,
    register_contract,
    ContractSourceKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_claim_contracts import (
    ClaimStatus,
    CodeClaimRecord,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    MCP_CONTRACT_OBLIGATIONS_INTERFACE,
    LogicFragment,
    McpContractObligation,
    McpContractObligationError,
    McpLogicView,
    compile_contract_claim,
)


def _catalog(family: McpClaimFamily):
    source = make_source_record(
        kind=ContractSourceKind.JSON_SCHEMA,
        subject="repo.inspect",
        source_version="1.2.0",
        schema_version="2020-12",
        path="schemas/repo-inspect.json",
        payload_fingerprint="sha256:repo-inspect-v1",
    )
    catalog = admit_source(DEFAULT_MCP_CONTRACT_CATALOG, source)
    contract, contradictions = build_contract_from_sources(
        claim_family=family,
        subject="repo.inspect",
        sources=(source,),
        tool_name="repo.inspect",
    )
    assert contract.review_state is ReviewState.REVIEWED
    return (
        register_contract(catalog, contract, contradictions=contradictions),
        contract,
    )


def _claim(
    family: McpClaimFamily = McpClaimFamily.ARGUMENTS_PRESERVED,
    *,
    state: ParityState = ParityState.SATISFIED,
    premises=("premise:schema", "premise:route"),
) -> ContractParityClaim:
    return ContractParityClaim(
        family=family,
        state=state,
        operation_id="repo.inspect",
        premise_ids=tuple(premises),
        reason_codes=(
            "parity_satisfied"
            if state is ParityState.SATISFIED
            else "schema_keyword_unsupported"
        ,),
    )


def _compile(
    family: McpClaimFamily = McpClaimFamily.ARGUMENTS_PRESERVED,
    *,
    state: ParityState = ParityState.SATISFIED,
    premises=("premise:schema", "premise:route"),
) -> McpContractObligation:
    catalog, contract = _catalog(family)
    return compile_contract_claim(
        _claim(family, state=state, premises=premises),
        catalog=catalog,
        contract=contract.contract_id,
        repository_id="repository:fixture",
        snapshot_id="tree:fixture",
        scope_ids=("scope:handler", "scope:descriptor"),
        assumption_ids=("assumption:closed-registry",),
        toolchain_id="toolchain:python-3.12",
        policy_id="policy:mcp-v1",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )


def test_compiler_binds_every_authority_dimension_without_minting_proof() -> None:
    result = _compile()

    assert MCP_CONTRACT_OBLIGATIONS_INTERFACE == "McpContractObligations@1"
    assert isinstance(result.code_obligation, CodeProofObligation)
    assert isinstance(result.code_claim, CodeClaimRecord)
    assert result.logic_fragment is LogicFragment.SCHEMA
    assert result.supported is True
    assert result.snapshot_id == "tree:fixture"
    assert result.scope_ids == ("scope:descriptor", "scope:handler")
    assert result.premise_ids == ("premise:route", "premise:schema")
    assert result.assumption_ids == ("assumption:closed-registry",)
    assert result.catalog_id
    assert result.contract_id == result.property_id
    assert result.toolchain_id == "toolchain:python-3.12"
    assert result.policy_id == "policy:mcp-v1"
    assert result.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.invalidators
    assert result.code_claim.status is ClaimStatus.OPEN
    assert result.code_claim.derived_assurance is AssuranceLevel.UNVERIFIED

    metadata = result.code_obligation.metadata
    assert metadata["catalog_id"] == result.catalog_id
    assert metadata["contract_id"] == result.property_id
    assert metadata["snapshot_id"] == result.snapshot_id
    assert metadata["toolchain_id"] == result.toolchain_id
    assert metadata["policy_id"] == result.policy_id
    assert metadata["supported"] is True
    assert metadata["logic_fragment"] == "schema"


def test_logic_view_uses_shared_identity_profile_and_canonical_round_trip() -> None:
    result = _compile()
    view = result.logic_view

    assert view.identity_profile == "ir-canonical-identity-v1"
    assert view.logic_id.startswith("b")
    assert view.identity.multicodec == "raw"
    assert McpLogicView.from_json(view.to_json()).to_json() == view.to_json()

    encoded = result.to_json()
    decoded = McpContractObligation.from_json(encoded)
    assert decoded.to_json() == encoded
    assert decoded.compiled_obligation_id == result.compiled_obligation_id
    assert decoded.code_obligation.obligation_id == result.obligation_id
    assert decoded.shared_ir_claim.obligations[0].obligation_id == view.logic_id


def test_premise_order_and_duplicate_inputs_do_not_change_identity() -> None:
    first = _compile(premises=("premise:z", "premise:a", "premise:z"))
    second = _compile(premises=("premise:a", "premise:z"))

    assert first.premise_ids == second.premise_ids
    assert first.logic_view.logic_id == second.logic_view.logic_id
    assert first.obligation_id == second.obligation_id
    assert first.compiled_obligation_id == second.compiled_obligation_id
    assert first.canonical_bytes() == second.canonical_bytes()


def test_unsupported_analysis_is_an_explicit_unsupported_fragment() -> None:
    result = _compile(state=ParityState.UNSUPPORTED)

    assert result.supported is False
    assert result.logic_fragment is LogicFragment.UNSUPPORTED
    assert result.logic_view.unsupported_reason == "schema_keyword_unsupported"
    assert result.code_claim.status is ClaimStatus.UNSUPPORTED
    assert result.code_obligation.fallback_checks == (
        "mcp-contract:unsupported-fragment",
    )
    assert result.code_obligation.metadata["supported"] is False


@pytest.mark.parametrize(
    "family,fragment",
    [
        (McpClaimFamily.TRANSPORT_PARITY, LogicFragment.RELATION),
        (McpClaimFamily.ARGUMENTS_PRESERVED, LogicFragment.SCHEMA),
        (McpClaimFamily.POLICY_BEFORE_EFFECT, LogicFragment.DEONTIC),
    ],
)
def test_closed_families_select_reviewed_compact_fragments(
    family: McpClaimFamily,
    fragment: LogicFragment,
) -> None:
    result = _compile(family)
    assert result.logic_fragment is fragment
    expression = result.logic_view.expression_dict()
    assert set(expression) == {"schema", "operator", "terms"}
    assert set(expression["terms"]) == {
        "claim_id",
        "operation_id",
        "property_id",
    }


@pytest.mark.parametrize(
    "bad_premise",
    [
        {"node": "premise:x", "source": "def mutate(): pass"},
        "def mutate():\n    pass",
        '{"nodes": ["entire", "graph"]}',
    ],
)
def test_source_and_graph_dumps_are_rejected_as_premises(bad_premise) -> None:
    catalog, contract = _catalog(McpClaimFamily.ARGUMENTS_PRESERVED)
    claim = _claim()
    object.__setattr__(claim, "premise_ids", ("premise:ok", bad_premise))

    with pytest.raises(
        McpContractObligationError,
        match="compact identifier|source or graph|must be a string",
    ):
        compile_contract_claim(
            claim,
            catalog=catalog,
            contract=contract,
            repository_id="repository:fixture",
            snapshot_id="tree:fixture",
            scope_ids=("scope:x",),
            toolchain_id="toolchain:x",
            policy_id="policy:x",
        )


def test_no_freeform_theorem_or_detached_catalog_contract_is_admitted() -> None:
    catalog, contract = _catalog(McpClaimFamily.ARGUMENTS_PRESERVED)
    payload = _claim().to_dict()
    payload["theorem"] = "Everything is safe."

    with pytest.raises(McpContractObligationError, match="unsupported fields"):
        compile_contract_claim(
            payload,
            catalog=catalog,
            contract=contract,
            repository_id="repository:fixture",
            snapshot_id="tree:fixture",
            scope_ids=("scope:x",),
            toolchain_id="toolchain:x",
            policy_id="policy:x",
        )

    detached = replace(contract, subject="other.operation", contract_id="")
    with pytest.raises(McpContractObligationError, match="not bound"):
        compile_contract_claim(
            _claim(),
            catalog=catalog,
            contract=detached,
            repository_id="repository:fixture",
            snapshot_id="tree:fixture",
            scope_ids=("scope:x",),
            toolchain_id="toolchain:x",
            policy_id="policy:x",
        )


def test_tampering_with_canonical_binding_fails_round_trip_validation() -> None:
    result = _compile()
    payload = result.to_dict()
    payload["code_obligation"]["metadata"]["policy_id"] = "policy:other"

    with pytest.raises(
        McpContractObligationError,
        match="mandatory binding|bindings disagree",
    ):
        McpContractObligation.from_dict(payload)
