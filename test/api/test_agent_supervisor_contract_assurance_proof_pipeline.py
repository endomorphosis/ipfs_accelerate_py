"""SCA-218 / SCA-G071 end-to-end proof/cache orchestration (SCAEV071PROOFCACHE).

Wires reviewed parity claims through:

* ``compile_contract_claim`` / ``McpContractObligation``
* ``McpContractProver`` (local checkers + kernel policy for non-local routes)
* sole ``TrustAwareProofCache`` with exact root revalidation
* mismatch classification and vulnerability refinement in the baseline

Zero model calls. Missing or stale evidence withholds downstream authority.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
    PROOF_PIPELINE_EVIDENCE,
    PROOF_PIPELINE_INTERFACE,
    BaselineStageName,
    StageCompleteness,
    TerminalContractStatus,
    materialize_contract_assurance_baseline,
    run_contract_proof_pipeline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractParityClaim,
    McpContractAnalysis,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    DEFAULT_MCP_CONTRACT_CATALOG,
    ContractSourceKind,
    McpClaimFamily,
    admit_source,
    build_contract_from_sources,
    make_source_record,
    register_contract,
)
from ipfs_accelerate_py.agent_supervisor.analysis.swissknife_contract_extractor import (
    extract_swissknife_contracts,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_proof_cache import (
    TrustAwareProofCache,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_prover import (
    ContractProofOutcome,
    ContractProofRoute,
    McpContractProver,
)


SNAPSHOT_ID = "sca-repository-snapshot:sha256:fixture-proof-pipeline"
SCOPE_POLICY = "sca-scope-policy:sha256:fixture-proof-policy"
EVIDENCE_ID = "SCAEV071PROOFCACHE"


def _fixture_snapshot() -> dict[str, object]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/sca-repository-snapshot@1",
        "schema_version": 1,
        "snapshot_id": SNAPSHOT_ID,
        "scope_id": "fixture-scope",
        "scope_policy_id": SCOPE_POLICY,
        "head_commit_id": "commit-fixture-proof",
        "head_tree_id": "tree-fixture-proof",
        "index_tree_id": "tree-fixture-proof",
        "is_clean": True,
        "primary_root": "swissknife",
        "dispositions": [
            {
                "path": "src/services/mcp/mcp-plus-plus.ts",
                "disposition_kind": "semantic",
                "declared_kind": "source",
                "reason_code": "tracked_source",
            }
        ],
        "dependency_identities": [],
        "gitlinks": [],
        "stats": {
            "tracked_path_count": 1,
            "disposition_count": 1,
            "semantic_path_count": 1,
            "excluded_path_count": 0,
            "overlay_path_count": 0,
            "dirty_path_count": 0,
            "deleted_path_count": 0,
            "dependency_identity_count": 0,
            "gitlink_count": 0,
            "hashed_bytes": 128,
        },
    }


def _satisfied_claim(
    family: McpClaimFamily = McpClaimFamily.ARGUMENTS_PRESERVED,
    *,
    operation_id: str = "repo.inspect",
) -> ContractParityClaim:
    return ContractParityClaim(
        family=family,
        state=ParityState.SATISFIED,
        operation_id=operation_id,
        premise_ids=("premise:schema", "premise:route"),
        reason_codes=("parity_satisfied",),
    )


def _refuted_claim(
    family: McpClaimFamily = McpClaimFamily.ARGUMENTS_PRESERVED,
    *,
    operation_id: str = "repo.inspect",
) -> ContractParityClaim:
    from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
        ContractCounterexample,
    )

    counterexample = ContractCounterexample(
        reason_code="schema_diverged",
        boundary_id="boundary:input_schema",
        path="input_schema.type",
        expected={"type": "object"},
        actual={"type": "string"},
        source_ids=("source:descriptor", "source:observed"),
    )
    return ContractParityClaim(
        family=family,
        state=ParityState.REFUTED,
        operation_id=operation_id,
        premise_ids=("premise:schema", "premise:route"),
        reason_codes=("schema_diverged",),
        counterexamples=(counterexample,),
    )


def _catalog_with_contract(
    family: McpClaimFamily = McpClaimFamily.ARGUMENTS_PRESERVED,
    *,
    subject: str = "repo.inspect",
):
    source = make_source_record(
        kind=ContractSourceKind.JSON_SCHEMA,
        subject=subject,
        source_version="1.0.0",
        schema_version="2020-12",
        path=f"schemas/{subject.replace('.', '-')}.json",
        payload_fingerprint=f"sha256:{subject}-v1",
    )
    catalog = admit_source(DEFAULT_MCP_CONTRACT_CATALOG, source)
    contract, contradictions = build_contract_from_sources(
        claim_family=family,
        subject=subject,
        sources=(source,),
        tool_name=subject,
    )
    catalog = register_contract(
        catalog,
        contract,
        contradictions=contradictions,
    )
    return catalog, contract


def _analysis_for_claims(
    *claims: ContractParityClaim,
    contract_id: str,
    operation_id: str = "repo.inspect",
) -> McpContractAnalysis:
    # McpContractAnalysis requires the full SCA-051 family set. Build a complete
    # report by filling missing families as unsupported, then overwrite supplied.
    by_family = {claim.family: claim for claim in claims}
    from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
        PARITY_CLAIM_FAMILIES,
    )

    complete: list[ContractParityClaim] = []
    for family in PARITY_CLAIM_FAMILIES:
        if family in by_family:
            complete.append(by_family[family])
            continue
        complete.append(
            ContractParityClaim(
                family=family,
                state=ParityState.UNSUPPORTED,
                operation_id=operation_id,
                premise_ids=("premise:schema",),
                reason_codes=("fixture_family_not_under_test",),
            )
        )
    return McpContractAnalysis(
        operation_id=operation_id,
        expected_contract_id=contract_id,
        observed_contract_id=f"observed:{operation_id}",
        claims=tuple(complete),
        complete=True,
    )


def test_evidence_constant_is_scaev071proofcache() -> None:
    assert PROOF_PIPELINE_EVIDENCE == EVIDENCE_ID
    assert PROOF_PIPELINE_EVIDENCE == "SCAEV071PROOFCACHE"
    assert PROOF_PIPELINE_INTERFACE == "ContractAssuranceProofPipeline@1"


def test_run_contract_proof_pipeline_proves_and_caches(tmp_path: Path) -> None:
    catalog, contract = _catalog_with_contract()
    analysis = _analysis_for_claims(
        _satisfied_claim(),
        contract_id=contract.contract_id,
    )
    cache = TrustAwareProofCache(tmp_path / "proof-cache")

    cold = run_contract_proof_pipeline(
        analyses=(analysis,),
        catalog=catalog,
        snapshot_id=SNAPSHOT_ID,
        proof_cache=cache,
        repository_id="repository:fixture-proof",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )
    warm = run_contract_proof_pipeline(
        analyses=(analysis,),
        catalog=catalog,
        snapshot_id=SNAPSHOT_ID,
        proof_cache=cache,
        repository_id="repository:fixture-proof",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )

    schema_cold = [
        item
        for item in cold
        if item.claim_family == McpClaimFamily.ARGUMENTS_PRESERVED.value
    ]
    schema_warm = [
        item
        for item in warm
        if item.claim_family == McpClaimFamily.ARGUMENTS_PRESERVED.value
    ]
    assert schema_cold
    assert schema_cold[0].outcome == ContractProofOutcome.PROVED.value
    assert schema_cold[0].terminal_status == TerminalContractStatus.PROVED.value
    assert schema_cold[0].route == ContractProofRoute.LOCAL_SCHEMA.value
    assert schema_cold[0].cache_hit is False
    assert schema_warm[0].cache_hit is True
    assert schema_warm[0].outcome == ContractProofOutcome.PROVED.value
    # Exact roots revalidated on the warm hit.
    roots = schema_warm[0].roots
    assert roots["snapshot"] == SNAPSHOT_ID
    assert roots["policy"] == "policy:fixture"
    assert roots["toolchain"] == "toolchain:fixture"
    assert roots["kernel"]
    assert roots["solver"]


def test_run_contract_proof_pipeline_refutes_with_counterexample(
    tmp_path: Path,
) -> None:
    catalog, contract = _catalog_with_contract()
    analysis = _analysis_for_claims(
        _refuted_claim(),
        contract_id=contract.contract_id,
    )
    outcomes = run_contract_proof_pipeline(
        analyses=(analysis,),
        catalog=catalog,
        snapshot_id=SNAPSHOT_ID,
        proof_cache=TrustAwareProofCache(tmp_path / "proof-cache"),
        repository_id="repository:fixture-proof",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )
    refuted = [
        item
        for item in outcomes
        if item.claim_family == McpClaimFamily.ARGUMENTS_PRESERVED.value
    ]
    assert refuted
    assert refuted[0].outcome == ContractProofOutcome.REFUTED.value
    assert refuted[0].terminal_status == TerminalContractStatus.REFUTED.value
    assert refuted[0].counterexample_id
    assert refuted[0].cache_hit is False


def test_unsupported_route_is_terminal_without_forging_authority(
    tmp_path: Path,
) -> None:
    catalog, contract = _catalog_with_contract(
        McpClaimFamily.TRANSPORT_PARITY
    )
    # TRANSPORT_PARITY routes to SMT; without a provider it is unsupported.
    claim = ContractParityClaim(
        family=McpClaimFamily.TRANSPORT_PARITY,
        state=ParityState.SATISFIED,
        operation_id="repo.inspect",
        premise_ids=("premise:transport", "premise:route"),
        reason_codes=("parity_satisfied",),
    )
    analysis = _analysis_for_claims(
        claim,
        contract_id=contract.contract_id,
    )
    outcomes = run_contract_proof_pipeline(
        analyses=(analysis,),
        catalog=catalog,
        snapshot_id=SNAPSHOT_ID,
        proof_cache=TrustAwareProofCache(tmp_path / "proof-cache"),
        prover=McpContractProver(provider_getter=lambda _pid: None),
        repository_id="repository:fixture-proof",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )
    transport = [
        item
        for item in outcomes
        if item.claim_family == McpClaimFamily.TRANSPORT_PARITY.value
    ]
    assert transport
    assert transport[0].terminal_status in {
        TerminalContractStatus.UNSUPPORTED.value,
        TerminalContractStatus.UNKNOWN.value,
    }
    assert transport[0].outcome in {
        ContractProofOutcome.UNSUPPORTED.value,
        ContractProofOutcome.INCONCLUSIVE.value,
    }


def _observed_tool_contract(operation_id: str, tool_name: str) -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "tool_name": tool_name,
        "name": tool_name,
        "package_id": "ipfs_kit_py",
        "complete": True,
        "routes": [
            {
                "route_id": f"route:{operation_id}",
                "path_class": "direct",
                "callable": True,
                "complete": True,
                "transport": "stdio",
                "input_schema": {
                    "type": "object",
                    "properties": {"cid": {"type": "string"}},
                    "required": ["cid"],
                    "additionalProperties": False,
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                "result_envelope": [
                    "content",
                    "error",
                    "provenance",
                    "receipt",
                ],
                "failure_states": [
                    "unsupported",
                    "unavailable",
                    "denied",
                    "timed_out",
                    "malformed",
                    "partial",
                ],
                "failure_mapping": {
                    "unsupported": "unsupported",
                    "unavailable": "unavailable",
                    "denied": "denied",
                    "timed_out": "timed_out",
                    "malformed": "malformed",
                    "partial": "partial",
                },
                "required_policies": ["mcp++/deontic-policy"],
                "argument_map": {},
                "events": [
                    "policy:authorize",
                    "effect:tools_call",
                ],
                "mutation_capable": True,
                "provenance": True,
                "receipt": True,
                "require_provenance": True,
                "require_receipt": True,
                "source_ids": [f"source:{operation_id}"],
            }
        ],
        "discovery": {"listed": True, "tools": [tool_name], "callable": True},
    }


def test_baseline_wires_proof_pipeline_end_to_end(tmp_path: Path) -> None:
    sources = {
        "src/services/mcp/mcp-plus-plus.ts": """
export const IPFS_KIT_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-kit',
  namespace: 'com.ipfs.kit',
  version: '1.0.0',
  interface_cid: 'bafy-kit',
  methods: [{
    name: 'ipfs.add',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
  }],
  errors: [{ name: 'Unavailable', code: 503 }],
  requires: ['mcp++/cid-envelope', 'mcp++/deontic-policy', 'mcp++/p2p-transport'],
  compatibility: { compatible_with: [], supersedes: [] },
};
"""
    }
    extraction = extract_swissknife_contracts(
        sources,
        repository_tree_id="tree-fixture-proof",
        source_version="git:fixture-proof",
    )
    tool_contracts = [
        item for item in extraction.catalog.contracts if item.tool_name
    ]
    assert tool_contracts
    observed = {
        item.tool_name: _observed_tool_contract(
            f"{item.package_id}:{item.tool_name}"
            if item.package_id
            else item.tool_name,
            item.tool_name,
        )
        for item in tool_contracts
    }
    cache_dir = tmp_path / "proof-cache"
    result = materialize_contract_assurance_baseline(
        snapshot_id=SNAPSHOT_ID,
        snapshot=_fixture_snapshot(),
        extraction=extraction,
        catalog=extraction.catalog,
        extract_expected=False,
        project_graph=True,
        observed_contracts=observed,
        proof_cache_dir=cache_dir,
        run_proof_pipeline=True,
        allow_proof_without_healthy_index=True,
        scope_policy_root=SCOPE_POLICY,
        repository_id="repository:fixture-proof",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        output_root=tmp_path / "baseline",
    )

    assert result.llm_call_count == 0
    findings = dict(result.findings)
    assert PROOF_PIPELINE_EVIDENCE in json.dumps(findings, default=str)
    stage_by_name = {stage.name: stage for stage in result.stages}
    proof_stage = stage_by_name[BaselineStageName.PROOF_CACHE]
    assert proof_stage.completeness in {
        StageCompleteness.COMPLETE,
        StageCompleteness.PARTIAL,
    }
    assert proof_stage.details.get("evidence_id") == EVIDENCE_ID
    assert proof_stage.details.get("sole_cache") == "TrustAwareProofCache"
    assert proof_stage.details.get("prover") == "McpContractProver"

    proof_outcomes = findings["proof_outcomes"]
    assert proof_outcomes["evidence_id"] == EVIDENCE_ID
    assert proof_outcomes["attempted"] >= 1
    assert result.proof_pipeline_outcomes
    terminals = {
        row[3] for row in findings["contract_population"]["contracts"]
    }
    assert terminals <= {
        "proved",
        "refuted",
        "unknown",
        "unsupported",
        "stale",
    }
    # Authority withheld without a healthy index even when proofs run.
    assert findings["claims"]["no_drift"] is False
    assert findings["claims"]["authority_promoted_from_optional_provider"] is False
    for outcome in result.proof_pipeline_outcomes:
        assert outcome.terminal_status in {
            "proved",
            "refuted",
            "unknown",
            "unsupported",
            "stale",
        }
        assert outcome.roots.get("snapshot") == SNAPSHOT_ID
        assert outcome.roots.get("policy")
        assert outcome.roots.get("toolchain")

    # The durable sole cache used by the baseline also memoizes proved receipts
    # under exact root revalidation (satisfied local-schema claim).
    catalog, contract = _catalog_with_contract()
    analysis = _analysis_for_claims(
        _satisfied_claim(),
        contract_id=contract.contract_id,
    )
    shared_cache = TrustAwareProofCache(cache_dir)
    cold_shared = run_contract_proof_pipeline(
        analyses=(analysis,),
        catalog=catalog,
        snapshot_id=SNAPSHOT_ID,
        proof_cache=shared_cache,
        repository_id="repository:fixture-proof",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )
    warm_shared = run_contract_proof_pipeline(
        analyses=(analysis,),
        catalog=catalog,
        snapshot_id=SNAPSHOT_ID,
        proof_cache=shared_cache,
        repository_id="repository:fixture-proof",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )
    satisfied = [
        item
        for item in cold_shared
        if item.claim_family == McpClaimFamily.ARGUMENTS_PRESERVED.value
    ]
    warm_satisfied = [
        item
        for item in warm_shared
        if item.claim_family == McpClaimFamily.ARGUMENTS_PRESERVED.value
    ]
    assert satisfied and satisfied[0].outcome == ContractProofOutcome.PROVED.value
    assert warm_satisfied and warm_satisfied[0].cache_hit is True


def test_missing_observed_contracts_withhold_proof_authority() -> None:
    sources = {
        "src/services/mcp/mcp-plus-plus.ts": """
export const IPFS_KIT_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-kit',
  namespace: 'com.ipfs.kit',
  version: '1.0.0',
  interface_cid: 'bafy-kit',
  methods: [{
    name: 'ipfs.add',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
  }],
  errors: [],
  requires: [],
  compatibility: { compatible_with: [], supersedes: [] },
};
"""
    }
    extraction = extract_swissknife_contracts(
        sources,
        repository_tree_id="tree-fixture-proof",
        source_version="git:fixture-proof",
    )
    result = materialize_contract_assurance_baseline(
        snapshot_id=SNAPSHOT_ID,
        snapshot=_fixture_snapshot(),
        extraction=extraction,
        extract_expected=False,
        project_graph=False,
        run_traces=False,
        observed_contracts={},
        allow_proof_without_healthy_index=True,
        run_proof_pipeline=True,
    )
    stage = next(
        item
        for item in result.stages
        if item.name is BaselineStageName.PROOF_CACHE
    )
    # Without usable observed contracts the stage never promotes authority.
    assert stage.completeness in {
        StageCompleteness.WITHHELD,
        StageCompleteness.PARTIAL,
    }
    assert result.findings["proof_outcomes"]["attempted"] == 0
    assert result.findings["claims"]["no_drift"] is False
    assert result.findings["claims"]["exhaustive"] is False
    assert stage.details.get("evidence_id") == EVIDENCE_ID
    assert stage.root_id == EVIDENCE_ID


def test_index_repository_contracts_exposes_proof_cache_flag() -> None:
    from scripts import index_repository_contracts as module

    assert module.PROOF_PIPELINE_EVIDENCE == EVIDENCE_ID
    parser = module.build_arg_parser()
    args = parser.parse_args(
        [
            "--output-root",
            "/tmp/baseline",
            "--proof-cache-dir",
            "/tmp/proof-cache",
        ]
    )
    assert args.proof_cache_dir == "/tmp/proof-cache"
    assert args.skip_proof_pipeline is False
    skipped = parser.parse_args(
        ["--output-root", "/tmp/baseline", "--skip-proof-pipeline"]
    )
    assert skipped.skip_proof_pipeline is True
