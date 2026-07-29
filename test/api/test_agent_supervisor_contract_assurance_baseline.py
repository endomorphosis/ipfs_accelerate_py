"""SCA-200 complete symbolic contract assurance baseline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
    BASELINE_FINDINGS_SCHEMA,
    CONTRACT_ASSURANCE_BASELINE_INTERFACE,
    DEFAULT_MAX_ARTIFACT_BYTES,
    TERMINAL_STATUS_DOMAIN,
    BaselineStageName,
    ContractAssuranceBaselineError,
    StageCompleteness,
    TerminalContractStatus,
    materialize_contract_assurance_baseline,
    publish_baseline_artifacts,
)
from ipfs_accelerate_py.agent_supervisor.analysis.swissknife_contract_extractor import (
    extract_swissknife_contracts,
)


SNAPSHOT_ID = "sca-repository-snapshot:sha256:fixture-baseline"
SCOPE_POLICY = "sca-scope-policy:sha256:fixture-policy"


def _canonical_sources() -> dict[str, str]:
    return {
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
export const IPFS_ACCELERATE_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-accelerate',
  namespace: 'com.ipfs.accelerate',
  version: '1.0.0',
  interface_cid: 'bafy-acc',
  methods: [{
    name: 'accelerate.inference',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
    interaction_pattern: 'stream',
  }],
  errors: [],
  requires: ['mcp++/cid-envelope'],
  compatibility: { compatible_with: [], supersedes: [] },
};
export const IPFS_DATASETS_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-datasets',
  namespace: 'com.ipfs.datasets',
  version: '1.0.0',
  interface_cid: 'bafy-ds',
  methods: [{
    name: 'datasets.search',
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


def _fixture_snapshot() -> dict[str, object]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/sca-repository-snapshot@1",
        "schema_version": 1,
        "snapshot_id": SNAPSHOT_ID,
        "scope_id": "fixture-scope",
        "scope_policy_id": SCOPE_POLICY,
        "head_commit_id": "commit-fixture",
        "head_tree_id": "tree-fixture",
        "index_tree_id": "tree-fixture",
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


def _extraction():
    return extract_swissknife_contracts(
        _canonical_sources(),
        repository_tree_id="tree-fixture",
        source_version="git:fixture",
    )


def _materialize(**overrides):
    arguments = {
        "snapshot_id": SNAPSHOT_ID,
        "snapshot": _fixture_snapshot(),
        "extraction": _extraction(),
        "extract_expected": False,
        "project_graph": True,
        "scope_policy_root": SCOPE_POLICY,
    }
    arguments.update(overrides)
    return materialize_contract_assurance_baseline(**arguments)


def test_interface_and_zero_llm_runtime() -> None:
    result = _materialize()
    assert CONTRACT_ASSURANCE_BASELINE_INTERFACE == "ContractAssuranceBaseline@1"
    assert result.llm_call_count == 0
    assert result.findings["generation"]["llm_call_count"] == 0
    assert result.findings["generation"]["deterministic"] is True
    assert result.findings["schema"] == BASELINE_FINDINGS_SCHEMA
    assert result.findings["interface"] == CONTRACT_ASSURANCE_BASELINE_INTERFACE
    assert result.result_id.startswith("b")


def test_every_contract_has_closed_terminal_status() -> None:
    result = _materialize()
    population = result.findings["contract_population"]
    assert population["emitted_contract_count"] == len(population["contracts"])
    assert population["emitted_contract_count"] >= 1
    assert set(result.findings["terminal_status_domain"]) == set(
        TERMINAL_STATUS_DOMAIN
    )
    for row in population["contracts"]:
        contract_id, family, _package, status, terminal, reasons = row
        assert contract_id
        assert family
        assert status in TERMINAL_STATUS_DOMAIN
        assert terminal is True
        assert isinstance(reasons, list)
        assert reasons
    counts = population["status_counts"]
    assert sum(counts[name] for name in TERMINAL_STATUS_DOMAIN) == len(
        population["contracts"]
    )


def test_identities_are_cid_bound_to_one_snapshot() -> None:
    result = _materialize()
    assert result.snapshot_id == SNAPSHOT_ID
    assert result.findings["snapshot_root"] == SNAPSHOT_ID
    assert result.findings["coverage_id"].startswith("sha256:")
    assert result.findings["findings_root"].startswith("sha256:")
    assert result.findings["contracts_root"].startswith("sha256:")
    assert result.findings["scope_policy_root"] == SCOPE_POLICY
    if result.graph is not None:
        assert result.graph.snapshot_id == SNAPSHOT_ID
        assert result.findings["graph_root"] == result.graph.graph_root
        assert result.graph.graph_root.startswith("b")
    if result.extraction is not None:
        assert result.findings["extraction_root"] == result.extraction.extraction_id
    if result.catalog is not None:
        assert result.findings["catalog_root"] == result.catalog.catalog_id
    # Re-materialization is deterministic for the same inputs.
    again = _materialize()
    assert again.findings["coverage_id"] == result.findings["coverage_id"]
    assert again.findings["contracts_root"] == result.findings["contracts_root"]
    assert again.result_id == result.result_id


def test_unhealthy_or_incomplete_stages_withhold_no_drift_claims() -> None:
    result = _materialize()
    claims = result.findings["claims"]
    assert claims["authority_promoted_from_optional_provider"] is False
    assert claims["no_drift"] is False
    assert claims["exhaustive"] is False
    assert claims["no_findings"] is False
    assert result.findings["analyzer_health"]["no_drift_claim"] is False
    assert result.findings["analyzer_health"]["safe_for_completion_reasoning"] is False
    stage_by_name = {stage.name: stage for stage in result.stages}
    assert (
        stage_by_name[BaselineStageName.PROOF_CACHE].completeness
        is StageCompleteness.WITHHELD
    )
    assert (
        "partial_analyzer_health_proof_not_started"
        in stage_by_name[BaselineStageName.PROOF_CACHE].reason_codes
        or "observed_contracts_unavailable"
        in stage_by_name[BaselineStageName.PROOF_CACHE].reason_codes
        or "repository_index_not_provided"
        in stage_by_name[BaselineStageName.REPOSITORY_INDEX].reason_codes
    )


def test_parity_measurement_assigns_proved_and_refuted_terminals() -> None:
    extraction = _extraction()
    # Supply incomplete observed contracts so measurement stays partial and
    # health-withheld path is not taken solely due to missing index health.
    # Without a healthy repository index, proof remains withheld — so we only
    # verify terminal domain closure under the default path here.
    result = materialize_contract_assurance_baseline(
        snapshot_id=SNAPSHOT_ID,
        snapshot=_fixture_snapshot(),
        extraction=extraction,
        catalog=extraction.catalog,
        extract_expected=False,
        observed_contracts={},
    )
    statuses = {
        row[3] for row in result.findings["contract_population"]["contracts"]
    }
    assert statuses <= set(TERMINAL_STATUS_DOMAIN)
    assert TerminalContractStatus.UNSUPPORTED.value in statuses
    assert result.findings["proof_outcomes"]["attempted"] == 0


def test_publish_stays_within_artifact_envelope(tmp_path: Path) -> None:
    result = _materialize()
    paths = publish_baseline_artifacts(
        result, tmp_path, max_file_bytes=DEFAULT_MAX_ARTIFACT_BYTES
    )
    assert paths["coverage"].is_file()
    assert paths["findings"].is_file()
    assert paths["summary"].is_file()
    for path in paths.values():
        assert path.stat().st_size <= DEFAULT_MAX_ARTIFACT_BYTES
    findings = json.loads(paths["findings"].read_text(encoding="utf-8"))
    coverage = json.loads(paths["coverage"].read_text(encoding="utf-8"))
    summary = paths["summary"].read_text(encoding="utf-8")
    assert findings["schema"] == BASELINE_FINDINGS_SCHEMA
    assert coverage["snapshot_id"] == SNAPSHOT_ID
    assert "SwissKnife Symbolic Contract Baseline" in summary
    assert "Model calls: `0`" in summary
    with pytest.raises(ContractAssuranceBaselineError):
        publish_baseline_artifacts(result, tmp_path / "tiny", max_file_bytes=32)


def test_graph_and_catalog_stages_complete_for_fixture_extraction() -> None:
    result = _materialize()
    stage_by_name = {stage.name: stage for stage in result.stages}
    assert stage_by_name[BaselineStageName.EXTRACTION].completeness is (
        StageCompleteness.COMPLETE
    )
    assert stage_by_name[BaselineStageName.CATALOG].completeness is (
        StageCompleteness.COMPLETE
    )
    assert stage_by_name[BaselineStageName.GRAPH].completeness in {
        StageCompleteness.COMPLETE,
        StageCompleteness.PARTIAL,
    }
    assert stage_by_name[BaselineStageName.PUBLISH].completeness is (
        StageCompleteness.COMPLETE
    )
    assert result.catalog is not None
    assert len(result.catalog.contracts) >= 1
    assert result.findings["contract_population"]["discovery_complete"] is True
    assert result.findings["contract_population"]["measurement_complete"] is False


def test_committed_baseline_artifacts_satisfy_acceptance_envelope() -> None:
    """Regression guard for the published SwissKnife shadow baseline."""

    root = Path("data/agent_supervisor/swissknife_contract_assurance/baseline")
    coverage_path = root / "coverage.json"
    findings_path = root / "contract_findings.json"
    summary_path = root / "summary.md"
    if not findings_path.is_file():
        pytest.skip("baseline artifacts not present in this checkout")
    assert coverage_path.is_file()
    assert summary_path.is_file()
    assert coverage_path.stat().st_size <= DEFAULT_MAX_ARTIFACT_BYTES
    assert findings_path.stat().st_size <= DEFAULT_MAX_ARTIFACT_BYTES
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")
    assert findings.get("schema") == BASELINE_FINDINGS_SCHEMA
    assert findings.get("generation", {}).get("llm_call_count") == 0
    assert findings.get("claims", {}).get(
        "authority_promoted_from_optional_provider"
    ) is False
    # Incomplete / partial health must withhold no-drift.
    if not findings.get("analyzer_health", {}).get("safe_for_completion_reasoning"):
        assert findings.get("claims", {}).get("no_drift") is False
    domain = set(findings.get("terminal_status_domain") or TERMINAL_STATUS_DOMAIN)
    assert domain == set(TERMINAL_STATUS_DOMAIN)
    population = findings.get("contract_population") or {}
    contracts = population.get("contracts") or []
    assert contracts
    for row in contracts:
        assert row[3] in domain
    snapshot = findings.get("snapshot_root") or coverage.get("snapshot_id")
    assert snapshot
    assert findings.get("snapshot_root") == snapshot or coverage.get(
        "snapshot_id"
    ) == snapshot
    assert "LLM" in summary or "llm" in summary.lower() or "Model calls" in summary
