#!/usr/bin/env python3
"""Independently verify the immutable PGIR-110 R1-R6 campaign.

This verifier uses only the Python standard library.  It does not import the
builder, start a daemon, open hidden tests, or treat a verified no-go as
training authority.  A successful exit means every arm/seed checkpoint,
metric, receipt, heldout/seed identity, and comparison replayed and stayed
fail-closed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPERIMENTS_DIR = Path(__file__).resolve().parent
FREEZE_DIR = EXPERIMENTS_DIR.parent / "freeze"

TASK_ID = "PGIR-110"
ARM_IDS = ("R1", "R2", "R3", "R4", "R5", "R6")
LEARNED_ARMS = ("R2", "R3", "R4", "R5", "R6")
LEARNED_SEEDS = (32, 33, 34)
N_MEASURES = (
    "token_cross_entropy",
    "latent_separation",
    "retrieval_recall",
    "structural_equivalence",
    "semantic_equivalence",
    "proof_replay_rate",
    "readability_score",
    "calibration_error",
    "ood_acceptance",
)
CAMPAIGN_MEASURES = ("latency", "resource_cost", "target_attainment")
SURFACES = ("compiler", "decompiler")
REASON_CODES = (
    "corpus_not_materialized",
    "historical_semantic_baseline_not_currently_qualified",
    "no_rights_admitted_training_rows",
    "required_holdouts_insufficient",
)
FORBIDDEN_STATUSES = {"measured", "partial", "attained", "promoted"}
SEAL_FIELDS = {
    "arm_cid": ("arm_cid", "arm_sha256"),
    "campaign_cid": ("campaign_cid", "campaign_sha256"),
    "catalog_cid": ("catalog_cid", "catalog_sha256"),
    "checkpoint_cid": ("checkpoint_cid", "checkpoint_sha256"),
    "comparison_cid": ("comparison_cid", "comparison_sha256"),
    "evaluation_cid": ("evaluation_cid", "evaluation_sha256"),
    "heldout_cid": ("heldout_cid", "heldout_sha256"),
    "manifest_cid": ("manifest_cid", "manifest_sha256"),
    "receipt_cid": ("receipt_cid", "receipt_sha256"),
    "recipe_cid": ("recipe_cid", "recipe_sha256"),
    "result_cid": ("result_cid", "result_sha256"),
    "seed_policy_cid": ("seed_policy_cid", "seed_policy_sha256"),
}


class CampaignVerificationError(ValueError):
    """Raised when any identity, metric, or fail-closed gate drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignVerificationError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise CampaignVerificationError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise CampaignVerificationError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def dag_json_cid(value: Any) -> str:
    digest = hashlib.sha256(canonical_bytes(value)).digest()
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + digest).decode(
        "ascii"
    ).rstrip("=").lower()


def raw_cid(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode(
        "ascii"
    ).rstrip("=").lower()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CampaignVerificationError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                CampaignVerificationError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                CampaignVerificationError(f"non-finite number {raw!r} in {path}")
            ),
        )
    if not isinstance(value, dict):
        raise CampaignVerificationError(f"{path} must contain a JSON object")
    validate_value(value, str(path))
    return value


def projection(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    excluded = SEAL_FIELDS[field]
    return {key: value for key, value in payload.items() if key not in excluded}


def require_cid(payload: Mapping[str, Any], field: str, path: Path) -> None:
    require(field in payload, f"{path} missing {field}")
    expected = dag_json_cid(projection(payload, field))
    require(payload[field] == expected, f"{path} {field} does not replay")


def load_named(name: str) -> dict[str, Any]:
    path = EXPERIMENTS_DIR / name
    require(path.is_file(), f"missing {name}")
    return strict_json(path)


def checkpoint_name(arm_id: str, seed: int) -> str:
    if arm_id == "R1":
        return "r1-deterministic.json"
    return f"{arm_id.lower()}-seed-{seed}.json"


def check_metric(metric: Mapping[str, Any], *, path: str) -> None:
    require(metric.get("missing_as_zero") is False, f"{path} treats a missing metric as zero")
    require(metric.get("value") is None, f"{path} fabricated a metric value")
    require(metric.get("numerator") is None, f"{path} fabricated a numerator")
    require(metric.get("confidence_interval") is None, f"{path} fabricated a confidence interval")
    require(metric.get("denominator") == 0, f"{path} invented a denominator")
    require(metric.get("sample_count") == 0, f"{path} invented a sample count")
    status = metric.get("status")
    require(status not in FORBIDDEN_STATUSES, f"{path} uses forbidden status {status!r}")
    require(isinstance(metric.get("reason"), str) and metric["reason"], f"{path} lacks a reason")
    require("fabricat" not in str(metric.get("reason", "")).lower() or "no_fabricated" in metric["reason"], f"{path} reason is unsafe")


def verify() -> dict[str, Any]:
    freeze_root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    freeze_result = strict_json(FREEZE_DIR / "result.v3.json")
    require(freeze_root["qualification"]["decision"] == "no_go", "freeze is not no_go")
    require(freeze_result["decision"] == "no_go", "freeze result is not no_go")
    campaign = load_named("campaign.json")
    require(
        freeze_root["root_cid"] == campaign["campaign_input_root_cid"],
        "campaign is not bound to the freeze root",
    )

    recipe = load_named("recipe.json")
    heldouts = load_named("heldouts.json")
    seeds = load_named("seeds.json")
    catalog = load_named("metric_catalog.json")
    comparison = load_named("comparison.json")
    manifest = load_named("manifest.json")
    result = load_named("result.json")
    for payload, field, path in (
        (recipe, "recipe_cid", "recipe.json"),
        (heldouts, "heldout_cid", "heldouts.json"),
        (seeds, "seed_policy_cid", "seeds.json"),
        (catalog, "catalog_cid", "metric_catalog.json"),
        (campaign, "campaign_cid", "campaign.json"),
        (comparison, "comparison_cid", "comparison.json"),
        (manifest, "manifest_cid", "manifest.json"),
        (result, "result_cid", "result.json"),
    ):
        require_cid(payload, field, EXPERIMENTS_DIR / path)

    require(recipe["hidden_test_selection"] is False, "recipe selects hidden tests")
    require(recipe["learned_inference_authorized"] is False, "recipe authorizes learned inference")
    require(recipe["missing_metric_as_zero"] is False, "recipe treats missing metrics as zero")
    require(heldouts["hidden_labels_opened"] is False, "heldouts opened hidden labels")
    require(heldouts["identical_across_arms"] is True, "heldouts are not identical across arms")
    require(heldouts["schema_binding"] == "RESULT(PGIR-012)", "heldouts are not RESULT(PGIR-012)")
    require(seeds["best_test_selection"] is False, "seed policy allows best-test selection")
    require(seeds["hidden_test_tuning"] is False, "seed policy tunes on hidden tests")
    require(seeds["learned_arm_seeds"] == list(LEARNED_SEEDS), "learned seeds drifted")
    require(seeds["deterministic_arm_seeds"] == [0], "deterministic seed drifted")

    require(campaign["decision"] == "no_go", "campaign decision is not no_go")
    require(campaign["lease_eligible"] is False, "campaign is lease-eligible")
    require(campaign["training_started"] is False, "campaign started training")
    require(campaign["training_admitted_rows"] == 0, "campaign invented training rows")
    require(campaign["data_split_identity"] == "RESULT(PGIR-012)", "campaign split drifted")
    require(set(result["reason_codes"]) == set(REASON_CODES), "reason codes drifted")

    arm_records = []
    checkpoints = []
    evaluations = []
    for arm_id in ARM_IDS:
        arm = load_named(f"arms/{arm_id}.json")
        require_cid(arm, "arm_cid", EXPERIMENTS_DIR / f"arms/{arm_id}.json")
        require(arm["task_id"] == TASK_ID, f"{arm_id} task_id drifted")
        require(arm["decision"] == "no_go", f"{arm_id} is not no_go")
        require(arm["training_started"] is False, f"{arm_id} started training")
        require(arm["proof_in_gradient_path"] is False, f"{arm_id} put proof in the gradient path")
        require(heldouts["heldout_cid"] == campaign["heldout_cid"], "heldout cid mismatch")
        expected_seeds = [0] if arm_id == "R1" else list(LEARNED_SEEDS)
        require(arm["seeds"] == expected_seeds, f"{arm_id} seeds drifted")
        arm_records.append(arm)
        for seed in expected_seeds:
            name = checkpoint_name(arm_id, seed)
            checkpoint = load_named(f"checkpoints/{name}")
            evaluation = load_named(f"evaluations/{name}")
            require_cid(checkpoint, "checkpoint_cid", EXPERIMENTS_DIR / f"checkpoints/{name}")
            require_cid(evaluation, "evaluation_cid", EXPERIMENTS_DIR / f"evaluations/{name}")
            require(checkpoint["arm_id"] == arm_id, f"{name} checkpoint arm drifted")
            require(checkpoint["seed"] == seed, f"{name} checkpoint seed drifted")
            require(checkpoint["shared_checkpoint_write"] is False, f"{name} wrote a shared checkpoint")
            require(checkpoint["weights"]["digest"] is None, f"{name} invented a weights digest")
            require(checkpoint["weights"]["status"] == "not_created", f"{name} claims weights exist")
            require(checkpoint["data_split_identity"] == "RESULT(PGIR-012)", f"{name} split drifted")
            require(evaluation["hidden_test_opened"] is False, f"{name} opened hidden tests")
            require(evaluation["best_test_selection"] is False, f"{name} selected on the test set")
            require(evaluation["fabricated_target_attainment"] is False, f"{name} fabricated attainment")
            require(evaluation["heldout_cid"] == heldouts["heldout_cid"], f"{name} evaluation heldout drifted")
            require(evaluation["checkpoint_cid"] == checkpoint["checkpoint_cid"], f"{name} evaluation/checkpoint CID mismatch")
            reported = {item["metric_id"] for item in evaluation["metrics"]}
            for metric_id in N_MEASURES:
                require(metric_id in reported, f"{name} missing N metric {metric_id}")
            for metric_id in CAMPAIGN_MEASURES:
                require(metric_id in reported, f"{name} missing campaign metric {metric_id}")
            surfaces_seen = {item["surface"] for item in evaluation["metrics"] if item["metric_id"] in N_MEASURES}
            require(surfaces_seen == set(SURFACES), f"{name} missing a compiler/decompiler surface")
            for index, metric in enumerate(evaluation["metrics"]):
                check_metric(metric, path=f"{name}.metrics[{index}]")
            if arm_id == "R1":
                require(
                    evaluation["historical_r1_report_cid"],
                    "R1 evaluation dropped the historical baseline CID",
                )
                e1 = [item for item in evaluation["metrics"] if str(item["metric_id"]).startswith("e1_")]
                require(e1, "R1 evaluation dropped E1 historical references")
                require(
                    all(item["status"] == "historical_not_currently_qualified" for item in e1),
                    "R1 treated a historical fixture as currently qualified",
                )
            checkpoints.append(checkpoint)
            evaluations.append(evaluation)

    require(len(checkpoints) == 16, f"expected 16 checkpoints, found {len(checkpoints)}")
    require(len(evaluations) == 16, f"expected 16 evaluations, found {len(evaluations)}")
    require(len({item["checkpoint_cid"] for item in checkpoints}) == 16, "checkpoint CIDs are not unique")
    require(len({item["evaluation_cid"] for item in evaluations}) == 16, "evaluation CIDs are not unique")

    receipts = {}
    for name in (
        "admission",
        "leases",
        "resources",
        "proof",
        "training",
        "evaluation",
        "reducer",
    ):
        payload = load_named(f"receipts/{name}.json")
        require_cid(payload, "receipt_cid", EXPERIMENTS_DIR / f"receipts/{name}.json")
        receipts[name] = payload
    require(receipts["admission"]["admitted"] is False, "admission receipt admitted the campaign")
    require(receipts["admission"]["authorizes_execution"] is False, "admission authorizes execution")
    require(receipts["leases"]["granted_count"] == 0, "a lease was granted")
    require(receipts["leases"]["production_pointer_mutated"] is False, "production pointer mutated")
    require(receipts["resources"]["training_started"] is False, "resource receipt claims training")
    require(receipts["resources"]["training_gpu_ms"] == 0, "resource receipt invented GPU time")
    require(receipts["resources"]["bounded_exhaustion"]["typed"] is True, "exhaustion is not typed")
    require(receipts["resources"]["bounded_exhaustion"]["kind"] == "admission_closed", "wrong exhaustion kind")
    require(receipts["proof"]["authority"] is False, "proof receipt claimed authority")
    require(receipts["proof"]["timeout_as_falsehood"] is False, "timeout treated as falsehood")
    require(receipts["training"]["failed_experiments_deleted"] is False, "failed experiments deleted")
    require(receipts["training"]["threshold_weakened"] is False, "thresholds weakened")
    require(receipts["training"]["shared_checkpoint_writes"] == 0, "shared checkpoint writes occurred")
    require(receipts["evaluation"]["hidden_test_opened"] is False, "evaluation receipt opened hidden tests")
    require(receipts["evaluation"]["measured_campaign_holdout_metrics"] == 0, "evaluation invented holdout metrics")
    require(receipts["reducer"]["cas_applied"] is False, "reducer CAS applied")
    require(receipts["reducer"]["promotion_pointer_mutated"] is False, "promotion pointer mutated")

    require(comparison["decision"] == "no_go", "comparison decision is not no_go")
    require(comparison["winner"] is None, "comparison invented a winner")
    require(comparison["admitted_candidate"] is None, "comparison admitted a candidate")
    require(comparison["promotion_authorized"] is False, "comparison authorized promotion")
    require(comparison["hidden_test_opened"] is False, "comparison opened hidden tests")
    require(comparison["best_test_selection"] is False, "comparison used best-test selection")
    require(comparison["same_heldouts"] is True, "comparison used different heldouts")
    require(comparison["same_seed_policy"] is True, "comparison used different seeds")
    require(comparison["fabricated_target_attainment"] is False, "comparison fabricated attainment")
    require(len(comparison["pairs"]) == 15, "comparison pair count drifted")
    require(all(pair["winner"] is None for pair in comparison["pairs"]), "a comparison pair has a winner")
    require(all(pair["status"] == "unavailable" for pair in comparison["pairs"]), "a comparison pair is measured")

    require(result["decision"] == "no_go", "result decision is not no_go")
    require(result["completion_authoritative"] is False, "result is completion-authoritative")
    require(result["descendant_execution_authorized"] is False, "result authorizes descendants")
    require(result["training_task_eligible_count"] == 0, "result invented eligible training tasks")
    require(result["result_identity"] == "RESULT(PGIR-110)", "result identity drifted")
    require(result["campaign_cid"] == campaign["campaign_cid"], "result campaign CID drifted")
    require(result["comparison_cid"] == comparison["comparison_cid"], "result comparison CID drifted")
    require(result["manifest_cid"] == manifest["manifest_cid"], "result manifest CID drifted")
    require(set(result["reason_codes"]) == set(REASON_CODES), "result reason codes drifted")

    file_entries = manifest["files"]
    expected_files = {
        "recipe.json",
        "heldouts.json",
        "seeds.json",
        "metric_catalog.json",
        "campaign.json",
        "comparison.json",
    }
    expected_files.update(f"arms/{arm_id}.json" for arm_id in ARM_IDS)
    expected_files.update(
        f"checkpoints/{checkpoint_name(arm_id, seed)}"
        for arm_id in ARM_IDS
        for seed in ((0,) if arm_id == "R1" else LEARNED_SEEDS)
    )
    expected_files.update(
        f"evaluations/{checkpoint_name(arm_id, seed)}"
        for arm_id in ARM_IDS
        for seed in ((0,) if arm_id == "R1" else LEARNED_SEEDS)
    )
    expected_files.update(
        f"receipts/{name}.json"
        for name in (
            "admission",
            "leases",
            "resources",
            "proof",
            "training",
            "evaluation",
            "reducer",
        )
    )
    require(set(file_entries) == expected_files, "manifest file set drifted")
    for name, entry in file_entries.items():
        path = EXPERIMENTS_DIR / name
        data = path.read_bytes()
        require(entry["raw_cid"] == raw_cid(data), f"manifest raw CID drifted for {name}")
        require(
            entry["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(),
            f"manifest sha256 drifted for {name}",
        )
        require(entry["size_bytes"] == len(data), f"manifest size drifted for {name}")

    require(
        campaign["heldout_cid"] == heldouts["heldout_cid"],
        "campaign/heldout CID mismatch",
    )
    require(campaign["seed_policy_cid"] == seeds["seed_policy_cid"], "campaign/seed CID mismatch")
    require(campaign["recipe_cid"] == recipe["recipe_cid"], "campaign/recipe CID mismatch")
    require(campaign["comparison_cid"] == comparison["comparison_cid"], "campaign/comparison CID mismatch")
    require(
        campaign["checkpoint_cids"] == [item["checkpoint_cid"] for item in checkpoints],
        "campaign checkpoint list drifted",
    )
    require(
        campaign["evaluation_cids"] == [item["evaluation_cid"] for item in evaluations],
        "campaign evaluation list drifted",
    )
    return {
        "arm_count": len(arm_records),
        "campaign_cid": campaign["campaign_cid"],
        "checkpoint_count": len(checkpoints),
        "comparison_cid": comparison["comparison_cid"],
        "decision": result["decision"],
        "evaluation_count": len(evaluations),
        "result_cid": result["result_cid"],
        "task_id": TASK_ID,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = verify()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignVerificationError as exc:
        print(f"campaign verification failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
