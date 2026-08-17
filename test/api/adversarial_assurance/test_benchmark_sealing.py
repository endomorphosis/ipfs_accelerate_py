"""AAE-062: campaign seal, signed receipt, benchmark economics, SCG calibration.

Acceptance criteria enforced here:

* Benchmark reports actual counts, detector rates, cache reuse, full/incremental
  cost and savings, model economics, and gap/remediation cost.
* Released EdDSA / did:key signer authority signs the content-addressed campaign
  receipt; cryptographic signature verification is exercised.
* Invalid and unverified signatures are rejected before persistence and before
  seal input (nothing is written on failure).
* ``AssuranceCampaignSeal@1`` commits every declared task artifact.
* SCG calibration evidence is non-authoritative (never production policy authority).
* Cold import is side-effect free; no production policy change.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.receipt_contracts import (
    EXISTING_SIGNATURE_ALGORITHM,
    EXISTING_SIGNATURE_AUTHORITY,
    AssuranceCampaignReceipt,
    ReceiptSignatureBinding,
    SignatureVerificationStatus,
    require_verified_signature_before_persistence,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_PATH = REPO_ROOT / "benchmarks/agent_supervisor/adversarial_assurance.py"
TEST_PATH = Path(__file__).resolve()
ARTIFACT_DIR = REPO_ROOT / "artifacts/agent_supervisor/adversarial_assurance"
BENCHMARK_JSON = ARTIFACT_DIR / "benchmark.json"
RECEIPT_JSON = ARTIFACT_DIR / "campaign_receipt.json"
SCG_JSON = ARTIFACT_DIR / "scg_calibration.json"

TASK_ID = "AAE-062"
BENCHMARK_INTERFACE = "AssuranceBenchmarkReport@1"
SEAL_INTERFACE = "AssuranceCampaignSeal@1"


def _load_benchmark_module():
    """Load the AAE-062 benchmark module (benchmarks/ is not always a package)."""

    name = "benchmarks.agent_supervisor.adversarial_assurance"
    if name in sys.modules:
        return sys.modules[name]
    if "benchmarks" not in sys.modules:
        pkg = types.ModuleType("benchmarks")
        pkg.__path__ = [str(REPO_ROOT / "benchmarks")]  # type: ignore[attr-defined]
        sys.modules["benchmarks"] = pkg
    if "benchmarks.agent_supervisor" not in sys.modules:
        sub = types.ModuleType("benchmarks.agent_supervisor")
        sub.__path__ = [str(REPO_ROOT / "benchmarks/agent_supervisor")]  # type: ignore[attr-defined]
        sys.modules["benchmarks.agent_supervisor"] = sub
    spec = importlib.util.spec_from_file_location(name, BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    assert BENCH_PATH.is_file()
    return _load_benchmark_module()


@pytest.fixture(scope="module")
def pipeline(bench):
    return bench.run_benchmark(write_artifacts=False, repo_root_path=REPO_ROOT)


# ---------------------------------------------------------------------------
# Module surface / cold import
# ---------------------------------------------------------------------------


def test_declared_outputs_exist() -> None:
    assert BENCH_PATH.is_file()
    assert TEST_PATH.is_file()


def test_module_exports_interfaces(bench) -> None:
    assert bench.BENCHMARK_INTERFACE == BENCHMARK_INTERFACE
    assert bench.CAMPAIGN_SEAL_INTERFACE == SEAL_INTERFACE
    assert bench.TASK_ID == TASK_ID
    assert bench.BENCHMARK_EVIDENCE == "aae/seal-benchmark@1"
    desc = bench.benchmark_descriptor()
    assert desc["interface"] == BENCHMARK_INTERFACE
    assert desc["seal_interface"] == SEAL_INTERFACE
    assert desc["production_policy_change"] is False
    assert desc["scg_calibration_authoritative"] is False
    assert desc["signature_authority"] == EXISTING_SIGNATURE_AUTHORITY
    assert desc["signature_algorithm"] == EXISTING_SIGNATURE_ALGORITHM
    assert callable(bench.run_benchmark)
    assert callable(bench.benchmark_assurance_campaign)
    assert bench.benchmark_assurance_campaign is bench.run_benchmark


def test_cold_import_is_side_effect_free() -> None:
    tree = ast.parse(BENCH_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {
                "system",
                "Popen",
                "urlopen",
                "urlretrieve",
            }


# ---------------------------------------------------------------------------
# Benchmark report fields
# ---------------------------------------------------------------------------


def test_benchmark_reports_actual_counts_and_rates(pipeline: dict[str, Any]) -> None:
    report = pipeline["benchmark"]
    assert report["interface_id"] == BENCHMARK_INTERFACE
    assert report["schema"].endswith("adversarial-assurance-benchmark@1")
    assert report["task_id"] == TASK_ID
    assert report["production_policy_changed"] is False
    assert report["targets_are_goals_not_results"] is True
    assert report["fabricated_pass"] is False

    counts = report["counts"]
    for key in (
        "generated",
        "admitted",
        "invalid",
        "equivalent",
        "killed",
        "selected_survivors",
        "full_survivors",
        "scoring_denominator",
        "gap_count",
        "critical_gap_count",
        "remediation_candidates",
        "accepted_promotions",
        "rejected_promotions",
        "mutant_cost_records",
    ):
        assert key in counts, key
        assert isinstance(counts[key], int)
        assert counts[key] >= 0

    assert counts["admitted"] > 0
    assert counts["killed"] > 0
    assert counts["mutant_cost_records"] > 0
    assert counts["generated"] >= counts["admitted"]

    rates = report["detector_rates"]
    for key in (
        "kill_rate_bp",
        "risk_weighted_score_bp",
        "predicted_detector_count",
        "selected_detector_count",
        "observed_detector_count",
        "missed_detector_count",
        "selected_test_rate_bp",
        "selected_proof_rate_bp",
        "selected_policy_rate_bp",
    ):
        assert key in rates, key

    assert rates["kill_rate_bp"] is not None
    assert 0 <= int(rates["kill_rate_bp"]) <= 10_000


def test_benchmark_reports_cache_reuse_and_incremental_economics(
    pipeline: dict[str, Any],
) -> None:
    report = pipeline["benchmark"]
    cache = report["cache_reuse"]
    assert cache["proof_cache_hits"] > 0
    assert cache["proof_cache_misses"] >= 0
    assert cache["proof_cache_reuse_rate_bp"] is not None
    assert 0 <= int(cache["proof_cache_reuse_rate_bp"]) <= 10_000

    cost = report["full_versus_incremental_cost"]
    assert cost["full_cpu_ms_total"] > cost["incremental_cpu_ms_total"]
    assert cost["full_wall_ms_total"] > cost["incremental_wall_ms_total"]
    assert cost["compute_saved_cpu_ms"] == (
        cost["full_cpu_ms_total"] - cost["incremental_cpu_ms_total"]
    )
    assert cost["savings_rate_bp"] is not None
    assert 0 < int(cost["savings_rate_bp"]) <= 10_000
    assert cost["avg_full_cost_per_mutant_cpu_ms"] is not None
    assert cost["avg_incremental_cost_per_mutant_cpu_ms"] is not None

    model = report["model_economics"]
    assert isinstance(model["model_calls"], int)
    assert isinstance(model["model_tokens"], int)
    assert model["model_calls"] >= 0
    assert model["model_tokens"] >= 0

    gap_cost = report["gap_remediation_cost"]
    assert gap_cost["total_gap_count"] >= 1
    assert gap_cost["remediation_total_cost_cpu_ms"] > 0
    assert gap_cost["cost_per_critical_gap_cpu_ms"] is not None
    assert gap_cost["cost_per_promotion_cpu_ms"] is not None


def test_benchmark_report_cid_stable(pipeline: dict[str, Any], bench) -> None:
    report = pipeline["benchmark"]
    rebuilt = bench.build_assurance_benchmark_report(
        seed=bench.DEFAULT_SEED,
        campaign_receipt_cid=report.get("campaign_receipt_cid"),
        scg_calibration_cid=report.get("scg_calibration_cid"),
    )
    assert rebuilt["report_cid"] == report["report_cid"]
    assert rebuilt["report_cid"].startswith("b")


# ---------------------------------------------------------------------------
# Signature authority + verification gates
# ---------------------------------------------------------------------------


def test_released_signer_signs_content_addressed_receipt(
    pipeline: dict[str, Any], bench
) -> None:
    receipt_dict = pipeline["campaign_receipt"]
    receipt = AssuranceCampaignReceipt.from_dict(receipt_dict)
    assert receipt.signature.signature_algorithm == EXISTING_SIGNATURE_ALGORITHM
    assert receipt.signature.signature_authority == EXISTING_SIGNATURE_AUTHORITY
    assert receipt.signature.signer_identity.startswith("did:key:z")
    assert receipt.signature.key_identity == receipt.signature.signer_identity
    assert receipt.signature.signature_verification_status == (
        SignatureVerificationStatus.VERIFIED.value
    )
    assert len(receipt.signature.signature) >= 43

    # Cryptographic verification via released authority path.
    receipt_cid = bench.verify_campaign_receipt_signature(receipt)
    assert receipt_cid == receipt.receipt_cid
    assert require_verified_signature_before_persistence(receipt) == receipt_cid

    # Content-addressed body is what was signed.
    content = bench.extract_campaign_receipt_content(receipt)
    content_cid = bench._structured_cid(content)
    assert content_cid.startswith("b")


def test_invalid_signature_rejected_before_persistence(
    pipeline: dict[str, Any], bench, tmp_path: Path
) -> None:
    ok = AssuranceCampaignReceipt.from_dict(pipeline["campaign_receipt"])
    sig = ok.signature.signature
    bad_sig = ("A" if sig[0] != "A" else "B") + sig[1:]
    bad_binding = ReceiptSignatureBinding(
        signer_identity=ok.signature.signer_identity,
        key_identity=ok.signature.key_identity,
        audience=ok.signature.audience,
        action=ok.signature.action,
        signature=bad_sig,
        signature_verification_status=SignatureVerificationStatus.VERIFIED,
        signature_algorithm=ok.signature.signature_algorithm,
        signature_authority=ok.signature.signature_authority,
    )
    bad_receipt = AssuranceCampaignReceipt(
        header=ok.header,
        receipt_id=ok.receipt_id,
        campaign_plan_cid=ok.campaign_plan_cid,
        campaign_policy_cid=ok.campaign_policy_cid,
        campaign_policy_version=ok.campaign_policy_version,
        admitted_set_cid=ok.admitted_set_cid,
        expected_detection_sets_cid=ok.expected_detection_sets_cid,
        outcomes_cid=ok.outcomes_cid,
        survivor_reports_cid=ok.survivor_reports_cid,
        vacuity_findings_cid=ok.vacuity_findings_cid,
        held_out_evaluation_cid=ok.held_out_evaluation_cid,
        held_out_result=ok.held_out_result,
        authorization_cid=ok.authorization_cid,
        expected_old_revision=ok.expected_old_revision,
        seal_scope=ok.seal_scope,
        seal_status=ok.seal_status,
        seal_evidence_cid=ok.seal_evidence_cid,
        gap_reports_cid=ok.gap_reports_cid,
        input_artifact_cids=ok.input_artifact_cids,
        signature=bad_binding,
        notes=ok.notes,
        metadata=ok.metadata,
    )
    target = tmp_path / "must_not_persist.json"
    with pytest.raises(bench.SignatureGateError) as excinfo:
        bench.persist_campaign_receipt(bad_receipt, target)
    assert excinfo.value.reason_code in {
        "invalid_signature",
        "unverified_signature",
        "receipt_identity_rejected",
    }
    assert not target.exists()


def test_unverified_signature_rejected_before_seal_input(
    pipeline: dict[str, Any], bench
) -> None:
    ok = AssuranceCampaignReceipt.from_dict(pipeline["campaign_receipt"])
    unverified = ReceiptSignatureBinding(
        signer_identity=ok.signature.signer_identity,
        key_identity=ok.signature.key_identity,
        audience=ok.signature.audience,
        action=ok.signature.action,
        signature=ok.signature.signature,
        signature_verification_status=SignatureVerificationStatus.UNVERIFIED,
        signature_algorithm=ok.signature.signature_algorithm,
        signature_authority=ok.signature.signature_authority,
    )
    # Complete terminal status forbids unverified signatures at construction.
    with pytest.raises(Exception):
        AssuranceCampaignReceipt(
            header=ok.header,
            receipt_id=ok.receipt_id,
            campaign_plan_cid=ok.campaign_plan_cid,
            campaign_policy_cid=ok.campaign_policy_cid,
            campaign_policy_version=ok.campaign_policy_version,
            admitted_set_cid=ok.admitted_set_cid,
            expected_detection_sets_cid=ok.expected_detection_sets_cid,
            outcomes_cid=ok.outcomes_cid,
            survivor_reports_cid=ok.survivor_reports_cid,
            vacuity_findings_cid=ok.vacuity_findings_cid,
            held_out_evaluation_cid=ok.held_out_evaluation_cid,
            held_out_result=ok.held_out_result,
            authorization_cid=ok.authorization_cid,
            expected_old_revision=ok.expected_old_revision,
            seal_scope=ok.seal_scope,
            seal_status=ok.seal_status,
            seal_evidence_cid=ok.seal_evidence_cid,
            gap_reports_cid=ok.gap_reports_cid,
            input_artifact_cids=ok.input_artifact_cids,
            signature=unverified,
            notes=ok.notes,
            metadata=ok.metadata,
        )

    # Gate also rejects when presented a mapping with unverified status.
    payload = copy.deepcopy(pipeline["campaign_receipt"])
    payload["signature"]["signature_verification_status"] = "unverified"
    with pytest.raises((bench.SignatureGateError, Exception)):
        bench.reject_unverified_signature_before_seal_input(payload)


def test_valid_receipt_may_persist(pipeline: dict[str, Any], bench, tmp_path: Path) -> None:
    target = tmp_path / "campaign_receipt.json"
    cid = bench.persist_campaign_receipt(pipeline["campaign_receipt"], target)
    assert target.is_file()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["receipt_cid"] == cid
    assert bench.verify_campaign_receipt_signature(loaded) == cid


# ---------------------------------------------------------------------------
# Seal commits every declared artifact
# ---------------------------------------------------------------------------


def test_seal_commits_every_declared_artifact(pipeline: dict[str, Any], bench) -> None:
    seal = pipeline["campaign_seal"]
    assert seal["interface_id"] == SEAL_INTERFACE
    assert seal["commits_every_declared_artifact"] is True
    assert seal["declared_result_completeness"] is True
    assert seal["production_policy_changed"] is False
    assert seal["scg_calibration_authoritative"] is False
    assert seal["signature_verified_before_seal_input"] is True

    path_commitments = seal["path_commitments"]
    for path in bench.DECLARED_ARTIFACT_PATHS:
        assert path in path_commitments, path
        assert isinstance(path_commitments[path], str)
        assert path_commitments[path].startswith("b")

    declared = seal["declared_artifacts"]
    for key in (
        "benchmark",
        "campaign_receipt",
        "scg_calibration",
        "metrics",
        "seal_evidence",
        "benchmark_source",
        "benchmark_test",
    ):
        assert key in declared, key
        assert declared[key]["cid"].startswith("b")

    assert declared["campaign_receipt"]["cid"] == pipeline["campaign_receipt"][
        "receipt_cid"
    ]
    assert declared["benchmark"]["cid"] == pipeline["benchmark"]["report_cid"]
    assert declared["scg_calibration"]["cid"] == pipeline["scg_calibration"][
        "calibration_bundle_cid"
    ]

    for nonclaim in (
        "repository_correctness",
        "mutation_set_completeness",
        "specification_completeness",
    ):
        assert nonclaim in seal["nonclaims"]


def test_seal_rejects_incomplete_declared_set(bench) -> None:
    with pytest.raises(bench.BenchmarkError) as excinfo:
        bench.build_campaign_seal(
            declared_artifacts={"benchmark": {"cid": "bafyincomplete"}},
            seal_evidence={"seal_evidence_cid": "bafyseal"},
            campaign_receipt_cid="bafyreceipt",
            metrics_cid="bafymetrics",
            scg_calibration_cid="bafyscg",
            benchmark_report_cid="bafyreport",
        )
    assert excinfo.value.reason_code == "incomplete_declared_artifacts"


# ---------------------------------------------------------------------------
# SCG calibration non-authoritative
# ---------------------------------------------------------------------------


def test_scg_calibration_is_non_authoritative(pipeline: dict[str, Any]) -> None:
    scg = pipeline["scg_calibration"]
    assert scg["authoritative_for_production_policy"] is False
    assert scg["scg_calibration_authoritative"] is False
    assert scg["production_policy_changed"] is False
    assert scg["production_policy_change_allowed"] is False
    assert scg["consumer"] == "SemanticCompressionGovernor"
    assert scg["record_count"] == 8
    assert len(scg["records"]) == 8
    for record in scg["records"]:
        assert record["authoritative_for_production_policy"] is False
        assert record["production_policy_changed"] is False
        assert record["killed"] is True
        assert record["evidence_cid"].startswith("b")


# ---------------------------------------------------------------------------
# End-to-end write + artifact regeneration
# ---------------------------------------------------------------------------


def test_run_benchmark_writes_artifacts(bench, tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    # Default write surface is only the three declared task artifacts.
    result = bench.run_benchmark(
        write_artifacts=True,
        output_path=out_dir / "benchmark.json",
        receipt_path=out_dir / "campaign_receipt.json",
        scg_path=out_dir / "scg_calibration.json",
        repo_root_path=REPO_ROOT,
    )
    assert (out_dir / "benchmark.json").is_file()
    assert (out_dir / "campaign_receipt.json").is_file()
    assert (out_dir / "scg_calibration.json").is_file()
    assert not (out_dir / "campaign_seal.json").exists()
    assert "campaign_seal" not in (result.get("written") or {})
    assert result["campaign_seal"]["commits_every_declared_artifact"] is True
    assert result["benchmark"]["campaign_seal_cid"] == result["campaign_seal"][
        "seal_cid"
    ]

    receipt = json.loads((out_dir / "campaign_receipt.json").read_text(encoding="utf-8"))
    assert bench.verify_campaign_receipt_signature(receipt) == receipt["receipt_cid"]

    report = json.loads((out_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert report["interface_id"] == BENCHMARK_INTERFACE
    assert report["counts"]["killed"] == result["benchmark"]["counts"]["killed"]

    scg = json.loads((out_dir / "scg_calibration.json").read_text(encoding="utf-8"))
    assert scg["authoritative_for_production_policy"] is False

    # Explicit seal_path remains available for offline inspection.
    seal_target = out_dir / "campaign_seal.json"
    result_with_seal = bench.run_benchmark(
        write_artifacts=True,
        output_path=out_dir / "benchmark2.json",
        receipt_path=out_dir / "campaign_receipt2.json",
        scg_path=out_dir / "scg_calibration2.json",
        seal_path=seal_target,
        repo_root_path=REPO_ROOT,
    )
    assert seal_target.is_file()
    seal = json.loads(seal_target.read_text(encoding="utf-8"))
    assert seal["commits_every_declared_artifact"] is True
    assert result_with_seal["written"]["campaign_seal"].endswith("campaign_seal.json")


def test_pipeline_does_not_change_production_policy(pipeline: dict[str, Any]) -> None:
    assert pipeline["production_policy_changed"] is False
    assert pipeline["scg_calibration_authoritative"] is False
    assert pipeline["fabricated_pass"] is False
    assert pipeline["signature_verified"] is True
    assert pipeline["metrics_available"] is True
    assert pipeline["economics_available"] is True
