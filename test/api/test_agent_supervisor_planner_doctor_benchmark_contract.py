from __future__ import annotations

import copy
import hashlib
import hmac
import json
import subprocess
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config/agent_supervisor_planner_doctor_benchmark.json"
SEAL_PATH = ROOT / "config/agent_supervisor_planner_doctor_benchmark.seal.json"
MANIFEST_PATH = (
    ROOT / "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json"
)
DOC_PATH = ROOT / "docs/architecture/agent_supervisor_planner_doctor_benchmark.md"
SCHEDULER_PATH = (
    ROOT / "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def _without(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for key in keys:
        result.pop(key)
    return result


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_benchmark_manifest_has_reconstructable_content_identities() -> None:
    manifest = _load(MANIFEST_PATH)
    assert manifest["schema"].endswith("planner-doctor-benchmark-manifest@1")
    assert manifest["manifest_cid"] == cid_for_dag_json(
        _without(manifest, "manifest_cid"), for_identity=True
    )
    assert validate_cid(manifest["manifest_cid"]) == manifest["manifest_cid"]

    normalization = manifest["normalization_profile"]
    normalization_payload = _without(
        normalization, "profile_cid", "profile_sha256", "identity_rule"
    )
    assert normalization["profile_cid"] == cid_for_dag_json(
        normalization_payload, for_identity=True
    )
    assert normalization["profile_sha256"] == (
        "sha256:"
        + hashlib.sha256(
            canonical_dag_json_bytes(normalization_payload, for_identity=True)
        ).hexdigest()
    )

    partition_policy = manifest["partition_policy"]
    partition_payload = _without(
        partition_policy, "policy_cid", "policy_sha256", "identity_rule"
    )
    assert partition_policy["policy_cid"] == cid_for_dag_json(
        partition_payload, for_identity=True
    )
    assert partition_policy["policy_sha256"] == (
        "sha256:"
        + hashlib.sha256(
            canonical_dag_json_bytes(partition_payload, for_identity=True)
        ).hexdigest()
    )

    cases = manifest["cases"]
    assert len(cases) == 12
    assert sum(case["partition"] == "development" for case in cases) == 6
    assert sum(case["partition"] == "heldout" for case in cases) == 6
    assert {case["pair_family"] for case in cases} == set(
        manifest["case_population"]["paired_families"]
    )
    for case in cases:
        input_contract = _without(case["input_contract"], "input_commitment_cid")
        input_payload = {
            "source_selector": case["source_selector"],
            "input_contract": input_contract,
        }
        assert case["input_contract"]["input_commitment_cid"] == cid_for_dag_json(
            input_payload, for_identity=True
        )
        assert case["case_cid"] == cid_for_dag_json(
            _without(case, "case_cid"), for_identity=True
        )
        assert validate_cid(case["case_cid"]) == case["case_cid"]
        assert case["promotion_eligible"] is False

    forbidden_payload_keys = {
        "expected",
        "expected_outcome",
        "gold",
        "golden",
        "oracle_body",
        "patch_body",
        "prompt_body",
        "mutation_body",
        "source_body",
    }
    for mapping in _walk(manifest):
        assert forbidden_payload_keys.isdisjoint(mapping)


def test_repository_roots_denominators_and_overlap_firewall_are_frozen() -> None:
    manifest = _load(MANIFEST_PATH)
    source = manifest["source_repository"]
    commit = source["audited_commit"]
    assert (
        subprocess.check_output(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=ROOT)
        .decode()
        .strip()
        == source["audited_tree"]
    )
    records = subprocess.check_output(
        ["git", "ls-tree", "-rz", "--full-tree", "-r", commit], cwd=ROOT
    )
    assert len(records.rstrip(b"\0").split(b"\0")) == source["tracked_entry_count"]

    partitions = manifest["partitions"]
    invariants = manifest["partition_invariants"]
    assert partitions["development"]["index_schema"].endswith(
        "repository-corpus-index@1"
    )
    assert partitions["heldout"]["index_schema"].endswith("repository-corpus-index@1")
    assert sum(item["member_count"] for item in partitions.values()) == 12_253
    assert partitions["development"]["member_count"] == 9_899
    assert partitions["heldout"]["member_count"] == 2_354
    assert all(
        invariants[name] == 0
        for name in (
            "exact_object_overlap_count",
            "normalized_ast_overlap_count",
            "normalized_contract_overlap_count",
            "provenance_family_overlap_count",
        )
    )
    assert invariants["any_cross_partition_family_overlap_forces_corpus_rejection"]
    assert manifest["exposure_contract"]["cids_alone_prove_non_exposure"] is False
    assert manifest["exposure_contract"]["promotion_authority"] is False
    for partition in partitions.values():
        assert validate_cid(partition["index_cid"]) == partition["index_cid"]


def test_policy_preregisters_real_paired_benchmark_and_hard_gates() -> None:
    policy = _load(POLICY_PATH)
    assert policy["policy_cid"] == cid_for_dag_json(
        _without(policy, "policy_cid"), for_identity=True
    )
    assert policy["preregistration"]["self_sealing_forbidden"]
    assert policy["preregistration"]["embedded_seal_forbidden"]
    assert policy["preregistration"]["automatic_promotion_enabled"] is False
    assert policy["population"]["corpus_authority"] == "public-conformance-only"
    assert policy["population"]["promotion_authority"] is False

    assert [arm["arm_id"] for arm in policy["primary_arms"]] == [
        "current-mainline-baseline",
        "deterministic-symbolic",
        "hybrid-residual-only",
    ]
    assert all(arm["real_service_execution_required"] for arm in policy["primary_arms"])
    assert len(policy["diagnostic_ablations"]["arms"]) >= 7
    assert policy["diagnostic_ablations"]["promotion_authority"] is False
    assert [item["stratum_id"] for item in policy["cache_strata"]] == [
        "cold",
        "exact-warm",
        "delta",
        "restart",
    ]
    assert policy["concurrency_sweep"]["requested_workers"] == [1, 2, 4, 6]
    assert policy["concurrency_sweep"]["configured_maximum_workers"] == 6

    metric_groups = policy["metric_registry"]
    assert set(metric_groups) == {
        "parallelism_and_clock",
        "token_efficiency",
        "process_tree_resources",
        "gpu_resources",
        "planner_quality",
        "doctor_quality",
        "solution_quality",
    }
    telemetry = policy["telemetry_contract"]
    assert telemetry["unavailable_must_never_be_encoded_as_numeric_zero"]
    assert telemetry["required_sensor_loss_blocks_promotion"]
    assert telemetry["process_scope"].startswith("entire-descendant-process-tree")

    floors = policy["non_compensable_safety_floors"]
    assert floors["must_pass_before_quality_or_pareto"]
    assert floors["statistical_margin_allowed"] is False
    assert floors["metrics"] and set(floors["metrics"].values()) == {0}
    assert policy["quality_non_inferiority"]["required_before_pareto"]
    assert policy["quality_non_inferiority"]["absolute_margin"] == 0
    assert policy["pareto_promotion"]["evaluated_only_after_safety_and_quality_gates"]
    assert policy["pareto_promotion"]["automatic_promotion"] is False

    evidence = policy["evidence_admission"]
    assert evidence["real_service_execution_required"]
    assert evidence["synthetic_fixture_or_mock_result_may_promote"] is False
    assert evidence["skipped_xfailed_dry_run_or_conformance_only_result_may_promote"] is False
    assert policy["denominator_protection"]["post_result_exclusion_allowed"] is False
    assert policy["budgets"]["qualifying_run"]["scored_cell_executions"] == 1728
    assert policy["stop_conditions"]["early_success_stop_allowed"] is False
    assert policy["non_exposure_enforcement"]["cid_commitments_alone_are_sufficient"] is False


def test_external_operator_seal_is_closed_activation_only_and_tamper_evident() -> None:
    seal = _load(SEAL_PATH)
    assert set(seal) == {
        "artifacts",
        "board_namespace",
        "decision",
        "grant",
        "interface",
        "operator",
        "policy_revision",
        "receipt_id",
        "receipt_version",
        "reviewed_base",
        "schema",
        "task_id",
    }
    assert seal["operator"]["authority_basis"] == "interactive_user_delegation"
    assert seal["operator"]["identity"] == "interactive_user"
    assert all(
        seal["operator"][name] is False
        for name in ("candidate", "model", "automatic_controller")
    )
    assert seal["decision"] == "sealed"
    grant = seal["grant"]
    assert grant["benchmark_activation"] is True
    assert grant["allowed_actions"] == ["activate_benchmark_policy"]
    assert grant["delegable"] is False
    assert all(
        grant[name] is False
        for name in (
            "automatic_promotion",
            "completion_authority",
            "mutation_authority",
            "promotion_authority",
            "protected_anchor_write_authority",
            "task_status_authority",
        )
    )
    assert seal["receipt_id"] == (
        "sha256:"
        + hashlib.sha256(
            canonical_dag_json_bytes(_without(seal, "receipt_id"), for_identity=True)
        ).hexdigest()
    )
    pinned_identity = _load(SCHEDULER_PATH)["manual_completion_seals"]["PDR-003"][
        "expected_receipt_id"
    ]
    assert hmac.compare_digest(seal["receipt_id"], pinned_identity)

    rehashed_tamper = copy.deepcopy(seal)
    rehashed_tamper["artifacts"][0]["sha256"] = "sha256:" + ("0" * 64)
    rehashed_tamper["receipt_id"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_dag_json_bytes(
                _without(rehashed_tamper, "receipt_id"),
                for_identity=True,
            )
        ).hexdigest()
    )
    assert not hmac.compare_digest(rehashed_tamper["receipt_id"], pinned_identity)

    expected_paths = {
        str(POLICY_PATH.relative_to(ROOT)),
        str(MANIFEST_PATH.relative_to(ROOT)),
        str(DOC_PATH.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    }
    assert {artifact["path"] for artifact in seal["artifacts"]} == expected_paths
    for artifact in seal["artifacts"]:
        path = ROOT / artifact["path"]
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["sha256"] == _sha256(path)

    policy = _load(POLICY_PATH)
    manifest = _load(MANIFEST_PATH)
    reviewed = seal["reviewed_base"]
    assert reviewed["benchmark_policy_cid"] == policy["policy_cid"]
    assert reviewed["benchmark_manifest_cid"] == manifest["manifest_cid"]
    assert reviewed["commit"] == manifest["source_repository"]["audited_commit"]
    assert reviewed["tree"] == manifest["source_repository"]["audited_tree"]
    assert reviewed["partition_algorithm"] == manifest["partition_policy"]["algorithm"]
    assert reviewed["partition_algorithm_version"] == 2
    assert reviewed["partition_seed"] == manifest["partition_policy"]["seed"]
    assert reviewed["development_index_cid"] == manifest["partitions"]["development"]["index_cid"]
    assert reviewed["heldout_index_cid"] == manifest["partitions"]["heldout"]["index_cid"]
    assert reviewed["oracle_handle"] == manifest["oracle_contract"]["oracle_handle"]


def test_benchmark_document_states_public_holdout_and_gate_order() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").lower()
    for phrase in (
        "public-conformance-only",
        "cids alone",
        "after the candidate process tree has terminated",
        "unavailable is never encoded as numeric zero",
        "non-compensable",
        "quality non-inferiority",
        "pareto",
        "automatic promotion remains false",
    ):
        assert phrase in text
