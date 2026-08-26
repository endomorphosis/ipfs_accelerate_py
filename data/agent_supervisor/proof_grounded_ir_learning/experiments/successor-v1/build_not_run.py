#!/usr/bin/env python3
"""Build the deterministic PGIR-206 typed ``not_run`` evidence package.

The superseding PGIR-205 input freeze is ``no_go`` and explicitly denies
descendant execution.  This builder materializes only documentary JSON.  It
does not import an accelerator stack, inspect devices, request a lease, train,
evaluate, invoke a prover, open hidden labels, or contact a network.

Default mode creates absent artifacts and refuses to replace different bytes.
``--check`` is strictly read-only and requires every expected byte to exist.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).absolute().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[4]
FREEZE_DIR = PACKAGE_DIR.parents[1] / "freeze" / "successor-v1"

TASK_ID = "PGIR-206"
TASK_TITLE = "Re-run R1-R6 on the superseding freeze"
TASK_CID = "baguqeerafze3wxxiomo4rhuxaguuk35d4tj4lrp2jtuj6jcinxnokczuctca"
TASK_KEY = "task/v1/2e49bb5ee8731dc89e9701a9456fa3e4d3c5c5fa4ce89f24486ddae50b3414c4"
OBJECTIVE_ID = "PGIR-G110"
PARENT_GOAL = "PGIR-G110"
SUBGOAL = "controlled-comparisons-v2"

PGIR205_ROOT_CID = "baguqeerajvu2dvjjxe4l6dibujguiedwhdahseziqpz7xhhp724jlxhlxz4q"
PGIR205_ROOT_SHA256 = "sha256:4d69a1d529b938bf0d01a24d44107638c079132883f3fb9ceffeb895dcebbe79"
PGIR205_RESULT_CID = "baguqeerarcuvejfqyjsfbqdtel67wu2lhjec3oh27enlqbxsvgyfbytsgd2q"
PGIR205_RESULT_SHA256 = "sha256:88a95224b0c26450c07322fdfb534b3a482db8faf91ab806f2a9b050e27230f5"
PGIR205_MANIFEST_CID = "baguqeeraw4k4u62ddeg2cvpn6qjqn3nwd2mdzrczc373zgtt67hbkgkrlbaq"
PGIR205_REVISION_SET_CID = "baguqeeraukisqjimylkyns2gr7af5imn4pmkv45gnja52c5vug4shc7lj7sq"
PGIR205_PLAN_RECEIPT_CID = "baguqeeranh57nf23krpy5qwgrivpnp2z7o2a5gcdzulvm73ekbb6upk6rwda"
PGIR205_VERIFICATION_CID = "baguqeeraxjwl7huyq2d5py3itpji5epx5om5a65f53vathq3jcmkfvxrr5sa"
PGIR205_PORTABILITY_CID = "baguqeera2ua47f7yht3oijt5vg4nzjsgnd7d75w5tdfw52fg6hj6v4tbmqfa"
PGIR211_ACCEPTANCE_CID = "baguqeeram562re6snweb5nuinwprb4ehccvkin7kpylihktizlirsss7pllq"
PGIR211_VERIFIER_RUN_CID = "baguqeerajzm5jdegqw6ihv3kbyggz4vxxeo6yqi6j4oxwdkeob6vqf2ve3aa"

PGIR205_FOREST = (
    {
        "role": "implementation",
        "commit": "011b7fd2bf15b380089944d2487989220a343338",
        "tree": "947e573d8d5ad299c8348e5a5bd507439bed4427",
        "parents": ["20ef9e48d59b505b04e6236a9a31aaba287c36fd"],
        "subject": "PGIR-205: Issue a superseding campaign input freeze",
    },
    {
        "role": "merge",
        "commit": "b4cd376393002c5a7daccdd8cac5c744ac6bf8aa",
        "tree": "947e573d8d5ad299c8348e5a5bd507439bed4427",
        "parents": [
            "20ef9e48d59b505b04e6236a9a31aaba287c36fd",
            "011b7fd2bf15b380089944d2487989220a343338",
        ],
        "subject": "Merge commit '011b7fd2bf15b380089944d2487989220a343338' into agent/pgir-successor-current-supervisor-20260825",
    },
    {
        "role": "completion",
        "commit": "f2e8d607de2396f15681ec2b6edb546b68c9952a",
        "tree": "24ddf9e4b9a02811c2505c6a47311add63f76d60",
        "parents": ["b4cd376393002c5a7daccdd8cac5c744ac6bf8aa"],
        "subject": "PGIR-205: mark todo completed",
    },
)

REASON_CODES = (
    "no_rights_admitted_training_rows",
    "corpus_not_materialized",
    "required_holdouts_insufficient",
    "tokenizer_not_admitted",
    "historical_semantic_baseline_not_currently_qualified",
    "integrated_evidence_does_not_authorize_execution",
    "portability_no_go",
)
FAILED_HOLDOUTS = (
    "compiler",
    "cross_reference",
    "domain",
    "exception",
    "length",
    "lineage",
    "notation",
    "premise",
    "proof_library",
    "publication",
    "rare_operator",
    "time",
    "type",
)
ARM_DEFINITIONS = (
    ("R1", "deterministic_compiler_decompiler", False, (0,), ()),
    ("R2", "token_cross_entropy_only", True, (32, 33, 34), ("token_cross_entropy",)),
    ("R3", "token_cross_entropy_plus_cosine", True, (32, 33, 34), ("token_cross_entropy", "normalized_cosine")),
    ("R4", "supervised_contrastive", True, (32, 33, 34), ("supervised_contrastive",)),
    (
        "R5",
        "full_multi_task",
        True,
        (32, 33, 34),
        (
            "token_cross_entropy",
            "normalized_cosine",
            "supervised_contrastive",
            "cycle",
            "structural",
            "relation",
            "semantic",
            "source_span",
            "calibration",
            "regularization",
        ),
    ),
    (
        "R6",
        "proof_grounded_curriculum",
        True,
        (32, 33, 34),
        ("full_multi_task", "nondifferentiable_proof_curriculum"),
    ),
)
METRIC_DEFINITIONS = (
    ("token_cross_entropy", "nats_per_canonical_token", "minimize"),
    ("latent_separation", "score", "maximize"),
    ("retrieval_recall", "rate", "maximize"),
    ("structural_equivalence", "rate", "maximize"),
    ("semantic_equivalence", "rate", "maximize"),
    ("proof_replay_rate", "rate", "maximize"),
    ("readability_score", "score", "maximize"),
    ("calibration_error", "error", "minimize"),
    ("ood_acceptance", "rate", "maximize"),
    ("latency", "milliseconds", "minimize"),
    ("resource_cost", "milli_resource_units", "minimize"),
    ("target_attainment", "boolean", "maximize"),
)

EXPECTED_FILES = (
    "README.md",
    "arms.json",
    "build_not_run.py",
    "campaign.json",
    "comparison.json",
    "heldouts.json",
    "manifest.json",
    "metrics.json",
    "receipts/admission.json",
    "receipts/checkpoint.json",
    "receipts/evaluation.json",
    "receipts/proof.json",
    "receipts/reducer_cas.json",
    "receipts/resource.json",
    "receipts/training.json",
    "result.json",
    "seeds.json",
    "verify_not_run.py",
)
MANIFEST_INPUTS = tuple(
    name for name in EXPECTED_FILES if name not in {"manifest.json", "result.json"}
)


class BuildError(ValueError):
    """Raised when an input or existing artifact is not exact."""


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise BuildError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise BuildError(f"{path} contains a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise BuildError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def render_json(value: Any) -> bytes:
    validate_value(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def digest_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def unsigned_varint(value: int) -> bytes:
    require(value >= 0, "varint value must be non-negative")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def cid(codec: int, data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    prefix = (
        unsigned_varint(1)
        + unsigned_varint(codec)
        + unsigned_varint(0x12)
        + unsigned_varint(len(digest))
    )
    return "b" + base64.b32encode(prefix + digest).decode("ascii").rstrip("=").lower()


def dag_json_cid(value: Any) -> str:
    return cid(0x0129, canonical_bytes(value))


def raw_cid(data: bytes) -> str:
    return cid(0x55, data)


def seal(payload: dict[str, Any], cid_field: str, sha_field: str) -> dict[str, Any]:
    if cid_field in payload or sha_field in payload:
        raise BuildError(f"seal fields already present: {cid_field}, {sha_field}")
    projection = dict(payload)
    payload[cid_field] = dag_json_cid(projection)
    payload[sha_field] = digest_sha256(canonical_bytes(projection))
    return payload


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BuildError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                BuildError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                BuildError(f"non-finite value {raw!r} in {path}")
            ),
        )
    except OSError as exc:
        raise BuildError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"{path} must contain an object")
    validate_value(value, str(path))
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def task_binding() -> dict[str, Any]:
    return {
        "current_task_cid": TASK_CID,
        "current_task_key": TASK_KEY,
        "objective_id": OBJECTIVE_ID,
        "parent_goal": PARENT_GOAL,
        "subgoal": SUBGOAL,
        "task_id": TASK_ID,
        "title": TASK_TITLE,
    }


def input_binding() -> dict[str, Any]:
    return {
        "campaign_input_root_cid": PGIR205_ROOT_CID,
        "campaign_input_root_sha256": PGIR205_ROOT_SHA256,
        "completion_authoritative": False,
        "descendant_execution_authorized": False,
        "manifest_cid": PGIR205_MANIFEST_CID,
        "pgir_205_forest": [dict(item) for item in PGIR205_FOREST],
        "plan_admission_receipt_cid": PGIR205_PLAN_RECEIPT_CID,
        "portability_receipt_cid": PGIR205_PORTABILITY_CID,
        "result_cid": PGIR205_RESULT_CID,
        "result_identity": "RESULT(PGIR-205)",
        "result_sha256": PGIR205_RESULT_SHA256,
        "revision_set_cid": PGIR205_REVISION_SET_CID,
        "verification_receipt_cid": PGIR205_VERIFICATION_CID,
    }


def zero_effects() -> dict[str, Any]:
    return {
        "checkpoint_created": False,
        "evaluation_invoked": False,
        "gpu_probe_performed": False,
        "hidden_labels_opened": False,
        "network_accessed": False,
        "optimizer_steps": 0,
        "proof_invoked": False,
        "promotion_attempted": False,
        "reducer_cas_attempted": False,
        "resource_lease_acquired": False,
        "resource_lease_requested": False,
        "training_started": False,
        "weights_created": False,
    }


def experiment_keys() -> list[dict[str, Any]]:
    return [
        {
            "arm_id": arm_id,
            "experiment_key": f"{arm_id}/seed-{seed}",
            "seed": seed,
        }
        for arm_id, _kind, _learned, seeds, _losses in ARM_DEFINITIONS
        for seed in seeds
    ]


def build_payloads() -> dict[str, dict[str, Any]]:
    freeze_root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    freeze_result = strict_json(FREEZE_DIR / "result.json")
    freeze_manifest = strict_json(FREEZE_DIR / "manifest.json")
    freeze_revisions = strict_json(FREEZE_DIR / "descendant_task_revisions.json")
    freeze_admission = strict_json(FREEZE_DIR / "plan_admission_receipt.json")
    freeze_verification = strict_json(FREEZE_DIR / "verification_receipt.json")
    freeze_portability = strict_json(FREEZE_DIR / "portability_receipt.json")
    integrated_acceptance = strict_json(
        FREEZE_DIR / "integrated-acceptance" / "integrated_acceptance.json"
    )
    require(freeze_root.get("root_cid") == PGIR205_ROOT_CID, "PGIR-205 root CID drifted")
    require(freeze_root.get("root_sha256") == PGIR205_ROOT_SHA256, "PGIR-205 root SHA drifted")
    freeze_root_projection = {
        key: value for key, value in freeze_root.items() if key not in {"root_cid", "root_sha256"}
    }
    require(dag_json_cid(freeze_root_projection) == PGIR205_ROOT_CID, "PGIR-205 root CID does not replay")
    require(
        digest_sha256(canonical_bytes(freeze_root_projection)) == PGIR205_ROOT_SHA256,
        "PGIR-205 root SHA does not replay",
    )
    require(freeze_root.get("qualification", {}).get("decision") == "no_go", "PGIR-205 root is not no_go")
    require(freeze_root.get("qualification", {}).get("descendant_execution_authorized") is False, "PGIR-205 authorizes execution")
    require(freeze_root.get("qualification", {}).get("lease_barrier") == "closed", "PGIR-205 lease barrier is not closed")
    require(freeze_result.get("result_cid") == PGIR205_RESULT_CID, "PGIR-205 result CID drifted")
    require(freeze_result.get("result_sha256") == PGIR205_RESULT_SHA256, "PGIR-205 result SHA drifted")
    freeze_result_projection = {
        key: value for key, value in freeze_result.items() if key not in {"result_cid", "result_sha256"}
    }
    require(dag_json_cid(freeze_result_projection) == PGIR205_RESULT_CID, "PGIR-205 result CID does not replay")
    require(
        digest_sha256(canonical_bytes(freeze_result_projection)) == PGIR205_RESULT_SHA256,
        "PGIR-205 result SHA does not replay",
    )
    require(freeze_result.get("decision") == "no_go", "PGIR-205 result is not no_go")
    require(freeze_result.get("descendant_execution_authorized") is False, "PGIR-205 result authorizes execution")
    require(freeze_result.get("completion_authoritative") is False, "PGIR-205 result became completion-authoritative")
    require(freeze_result.get("training_task_eligible_count") == 0, "PGIR-205 has an eligible training task")
    require(freeze_result.get("reason_codes") == list(REASON_CODES), "PGIR-205 reason-code population drifted")
    require(freeze_manifest.get("manifest_cid") == PGIR205_MANIFEST_CID, "PGIR-205 manifest CID drifted")
    require(freeze_revisions.get("revision_set_cid") == PGIR205_REVISION_SET_CID, "PGIR-205 revision set drifted")
    require(freeze_admission.get("receipt_id") == PGIR205_PLAN_RECEIPT_CID, "PGIR-205 admission receipt drifted")
    require(freeze_admission.get("admitted") is False, "PGIR-205 admission unexpectedly passed")
    require(freeze_admission.get("authorizes_execution") is False, "PGIR-205 admission authorizes execution")
    require(freeze_verification.get("receipt_cid") == PGIR205_VERIFICATION_CID, "PGIR-205 verification receipt drifted")
    require(freeze_verification.get("authorizes_execution") is False, "PGIR-205 verification authorizes execution")
    require(freeze_portability.get("receipt_cid") == PGIR205_PORTABILITY_CID, "PGIR-205 portability receipt drifted")
    require(freeze_portability.get("status") == "portability_no_go", "PGIR-205 portability is not no_go")
    require(integrated_acceptance.get("acceptance_cid") == PGIR211_ACCEPTANCE_CID, "PGIR-211 acceptance CID drifted")
    require(integrated_acceptance.get("completion_authoritative") is False, "PGIR-211 became completion-authoritative")
    require(
        freeze_root.get("integrated_evidence", {}).get("fresh_verifier_run", {}).get("run_cid")
        == PGIR211_VERIFIER_RUN_CID,
        "PGIR-211 verifier-run CID drifted",
    )
    revision_rows = freeze_revisions.get("revisions")
    require(isinstance(revision_rows, list), "PGIR-205 revision population is absent")
    pgir206_rows = [row for row in revision_rows if isinstance(row, dict) and row.get("task_id") == TASK_ID]
    require(len(pgir206_rows) == 1, "PGIR-205 must bind exactly one PGIR-206 revision")
    require(pgir206_rows[0].get("current_task_cid") == TASK_CID, "active PGIR-206 task CID drifted")
    require(pgir206_rows[0].get("current_task_key") == TASK_KEY, "active PGIR-206 task key drifted")

    keys = experiment_keys()
    heldouts = seal(
        {
            "campaign_input_root_cid": PGIR205_ROOT_CID,
            "failed_holdout_count": len(FAILED_HOLDOUTS),
            "hidden_labels_opened": False,
            "hidden_test_commitment": "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded",
            "holdouts": [
                {"count": 0, "holdout_id": holdout_id, "status": "permanent_no_go"}
                for holdout_id in FAILED_HOLDOUTS
            ],
            "identical_across_arms": True,
            "leakage_passed": True,
            "result_identity": "RESULT(PGIR-202)",
            "schema": "PGIRExperimentHeldouts@2",
            "split_binding_cid": "baguqeera456gtyfkktybzrjbujmp2mqy6i2bins372xgdsgvaahdcvpxy7aq",
            "status": "not_run",
            "task_binding": task_binding(),
        },
        "heldout_cid",
        "heldout_sha256",
    )
    seeds = seal(
        {
            "best_test_selection": False,
            "deterministic_arm_seeds": [0],
            "experiment_key_count": len(keys),
            "experiment_keys": keys,
            "hidden_test_tuning": False,
            "learned_arm_seeds": [32, 33, 34],
            "schema": "PGIRExperimentSeedPolicy@2",
            "status": "not_run",
            "task_binding": task_binding(),
        },
        "seed_cid",
        "seed_sha256",
    )
    arms = seal(
        {
            "arm_count": len(ARM_DEFINITIONS),
            "arms": [
                {
                    "arm_id": arm_id,
                    "disposition": "not_run",
                    "execution_authorized": False,
                    "kind": kind,
                    "learned": learned,
                    "loss_components": list(losses),
                    "proof_in_gradient_path": False,
                    "reason_code": "admission_closed",
                    "seeds": list(arm_seeds),
                    "training_started": False,
                }
                for arm_id, kind, learned, arm_seeds, losses in ARM_DEFINITIONS
            ],
            "campaign_input_root_cid": PGIR205_ROOT_CID,
            "disposition_count": len(keys),
            "dispositions": [
                {
                    "arm_id": key["arm_id"],
                    "disposition": "not_run",
                    "execution_authorized": False,
                    "experiment_key": key["experiment_key"],
                    "reason_code": "admission_closed",
                    "seed": key["seed"],
                }
                for key in keys
            ],
            "experiment_key_count": len(keys),
            "schema": "PGIRExperimentArmSet@2",
            "status": "not_run",
            "task_binding": task_binding(),
        },
        "arm_set_cid",
        "arm_set_sha256",
    )
    campaign = seal(
        {
            "arm_set_cid": arms["arm_set_cid"],
            "authorizes_execution": False,
            "decision": "no_go",
            "disposition": "typed_not_run",
            "execution_status": "not_run",
            "experiment_key_count": len(keys),
            "heldout_cid": heldouts["heldout_cid"],
            "input_binding": input_binding(),
            "lease_eligible": False,
            "observed_effects": zero_effects(),
            "reason_codes": list(REASON_CODES),
            "schema": "PGIRControlledCampaign@2",
            "seed_cid": seeds["seed_cid"],
            "task_binding": task_binding(),
            "training_admitted_rows": 0,
        },
        "campaign_cid",
        "campaign_sha256",
    )
    metric_definitions = [
        {"direction": direction, "metric_id": metric_id, "unit": unit}
        for metric_id, unit, direction in METRIC_DEFINITIONS
    ]
    metric_cells = [
        {
            "arm_id": key["arm_id"],
            "confidence_interval": None,
            "denominator": 0,
            "experiment_key": key["experiment_key"],
            "metric_id": metric_id,
            "missing_as_zero": False,
            "numerator": None,
            "reason_code": "admission_closed",
            "sample_count": 0,
            "seed": key["seed"],
            "status": "not_run",
            "target_attained": None,
            "unit": unit,
            "value": None,
        }
        for key in keys
        for metric_id, unit, _direction in METRIC_DEFINITIONS
    ]
    metrics = seal(
        {
            "campaign_cid": campaign["campaign_cid"],
            "cell_count": len(metric_cells),
            "cells": metric_cells,
            "metric_count": len(metric_definitions),
            "metric_definitions": metric_definitions,
            "schema": "PGIRExperimentMetricMatrix@2",
            "status": "not_run",
            "task_binding": task_binding(),
        },
        "metrics_cid",
        "metrics_sha256",
    )
    arm_ids = [item[0] for item in ARM_DEFINITIONS]
    pair_rows = [
        {
            "effect": None,
            "left_arm_id": left,
            "paired_sample_count": 0,
            "pair_id": f"{left}__{right}",
            "reason_code": "admission_closed",
            "right_arm_id": right,
            "status": "not_run",
            "uncertainty": None,
            "winner": None,
        }
        for left_index, left in enumerate(arm_ids)
        for right in arm_ids[left_index + 1 :]
    ]
    comparison = seal(
        {
            "campaign_cid": campaign["campaign_cid"],
            "metric_ids": [item[0] for item in METRIC_DEFINITIONS],
            "metrics_cid": metrics["metrics_cid"],
            "no_winner": True,
            "pair_count": len(pair_rows),
            "pairs": pair_rows,
            "schema": "PGIRExperimentComparison@2",
            "status": "not_run",
            "task_binding": task_binding(),
        },
        "comparison_cid",
        "comparison_sha256",
    )

    gates = [
        {"gate_id": gate_id, "passed": False}
        for gate_id in (
            "rights",
            "corpus",
            "holdouts",
            "tokenizer",
            "current_baseline",
            "integrated_evidence",
            "portability",
        )
    ]
    admission = seal(
        {
            "admitted": False,
            "authorizes_execution": False,
            "campaign_cid": campaign["campaign_cid"],
            "checked_gates": gates,
            "decision": "rejected",
            "execution_status": "not_run",
            "input_plan_admission_receipt_cid": PGIR205_PLAN_RECEIPT_CID,
            "reason_codes": list(REASON_CODES),
            "schema": "PGIRExperimentAdmissionReceipt@2",
            "task_binding": task_binding(),
        },
        "receipt_cid",
        "receipt_sha256",
    )
    training = seal(
        {
            "admission_receipt_cid": admission["receipt_cid"],
            "batch_count": 0,
            "campaign_cid": campaign["campaign_cid"],
            "data_rows_read": 0,
            "experiment_key_count": len(keys),
            "experiment_keys": keys,
            "gpu_probe_performed": False,
            "optimizer_steps": 0,
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentTrainingReceipt@2",
            "status": "not_run",
            "task_binding": task_binding(),
            "training_started": False,
        },
        "receipt_cid",
        "receipt_sha256",
    )
    checkpoint = seal(
        {
            "campaign_cid": campaign["campaign_cid"],
            "checkpoint_count": 0,
            "checkpoint_paths": [],
            "experiment_key_count": len(keys),
            "experiment_keys": keys,
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentCheckpointReceipt@2",
            "shared_checkpoint_write": False,
            "status": "not_run",
            "task_binding": task_binding(),
            "training_receipt_cid": training["receipt_cid"],
            "weight_artifacts": [],
            "weights_created": False,
        },
        "receipt_cid",
        "receipt_sha256",
    )
    proof = seal(
        {
            "attempt_count": 0,
            "authority_granted": False,
            "campaign_cid": campaign["campaign_cid"],
            "checked_proof_count": 0,
            "experiment_key_count": len(keys),
            "experiment_keys": keys,
            "hidden_labels_opened": False,
            "kernel_check_count": 0,
            "nondifferentiable": True,
            "proof_invoked": False,
            "proof_results": [],
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentProofReceipt@2",
            "status": "not_run",
            "task_binding": task_binding(),
            "timeout_as_falsehood": False,
            "training_receipt_cid": training["receipt_cid"],
        },
        "receipt_cid",
        "receipt_sha256",
    )
    evaluation = seal(
        {
            "best_test_selection": False,
            "campaign_cid": campaign["campaign_cid"],
            "evaluation_invoked": False,
            "experiment_key_count": len(keys),
            "experiment_keys": keys,
            "hidden_test_access": False,
            "hidden_labels_opened": False,
            "hidden_test_selection": False,
            "hidden_test_tuning": False,
            "metric_cell_count": len(metric_cells),
            "measured_cell_count": 0,
            "metrics_cid": metrics["metrics_cid"],
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentEvaluationReceipt@2",
            "status": "not_run",
            "task_binding": task_binding(),
            "target_attainment_claim_count": 0,
        },
        "receipt_cid",
        "receipt_sha256",
    )
    resource = seal(
        {
            "bounded_exhaustion": {
                "kind": "admission_closed",
                "reason_codes": list(REASON_CODES),
            },
            "campaign_cid": campaign["campaign_cid"],
            "experiment_key_count": len(keys),
            "experiment_keys": keys,
            "gpu_probe_performed": False,
            "gpu_seconds_used": 0,
            "lease_acquired": False,
            "lease_id": None,
            "lease_requested": False,
            "measured_cost": None,
            "network_accessed": False,
            "proof_seconds_used": 0,
            "provider_call_count": 0,
            "reason_code": "admission_closed",
            "resource_class": "gpu-large",
            "schema": "PGIRExperimentResourceReceipt@2",
            "status": "not_run",
            "task_binding": task_binding(),
            "token_count": 0,
        },
        "receipt_cid",
        "receipt_sha256",
    )
    reducer_cas = seal(
        {
            "campaign_cid": campaign["campaign_cid"],
            "candidate_pointer": None,
            "compare_and_swap_attempted": False,
            "expected_pointer": None,
            "observed_pointer_after": None,
            "observed_pointer_before": None,
            "pointer_unchanged": True,
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentReducerCASReceipt@2",
            "status": "not_run",
            "task_binding": task_binding(),
            "winner": None,
        },
        "receipt_cid",
        "receipt_sha256",
    )

    payloads: dict[str, dict[str, Any]] = {
        "arms.json": arms,
        "campaign.json": campaign,
        "comparison.json": comparison,
        "heldouts.json": heldouts,
        "metrics.json": metrics,
        "receipts/admission.json": admission,
        "receipts/checkpoint.json": checkpoint,
        "receipts/evaluation.json": evaluation,
        "receipts/proof.json": proof,
        "receipts/reducer_cas.json": reducer_cas,
        "receipts/resource.json": resource,
        "receipts/training.json": training,
        "seeds.json": seeds,
    }
    generated_bytes = {name: render_json(payload) for name, payload in payloads.items()}
    source_bytes = {
        name: (PACKAGE_DIR / name).read_bytes()
        for name in ("README.md", "build_not_run.py", "verify_not_run.py")
    }
    object_fields = {
        "arms.json": ("arm_set_cid", "arm_set_sha256"),
        "campaign.json": ("campaign_cid", "campaign_sha256"),
        "comparison.json": ("comparison_cid", "comparison_sha256"),
        "heldouts.json": ("heldout_cid", "heldout_sha256"),
        "metrics.json": ("metrics_cid", "metrics_sha256"),
        "seeds.json": ("seed_cid", "seed_sha256"),
    }
    manifest_artifacts: dict[str, Any] = {}
    for name in MANIFEST_INPUTS:
        data = generated_bytes[name] if name in generated_bytes else source_bytes[name]
        cid_field, sha_field = object_fields.get(name, ("receipt_cid", "receipt_sha256"))
        payload = payloads.get(name)
        manifest_artifacts[name] = {
            "object_cid": payload[cid_field] if payload is not None else None,
            "object_sha256": payload[sha_field] if payload is not None else None,
            "raw_cid": raw_cid(data),
            "sha256": digest_sha256(data),
            "size_bytes": len(data),
        }
    manifest = seal(
        {
            "artifact_count": len(manifest_artifacts),
            "artifacts": manifest_artifacts,
            "campaign_cid": campaign["campaign_cid"],
            "decision": "no_go",
            "execution_status": "not_run",
            "immutability": "supersede_never_overwrite",
            "json_artifact_count": len(payloads),
            "schema": "PGIRExperimentNotRunManifest@2",
            "task_binding": task_binding(),
        },
        "manifest_cid",
        "manifest_sha256",
    )
    payloads["manifest.json"] = manifest
    receipt_cids = {
        name.removeprefix("receipts/").removesuffix(".json"): payloads[name]["receipt_cid"]
        for name in payloads
        if name.startswith("receipts/")
    }
    result = seal(
        {
            "arm_count": len(ARM_DEFINITIONS),
            "arm_set_cid": arms["arm_set_cid"],
            "campaign_cid": campaign["campaign_cid"],
            "checkpoint_count": 0,
            "comparison_cid": comparison["comparison_cid"],
            "completion_authoritative": False,
            "decision": "no_go",
            "descendant_execution_authorized": False,
            "disposition": "typed_not_run",
            "execution_authorized": False,
            "execution_status": "not_run",
            "experiment_key_count": len(keys),
            "heldout_cid": heldouts["heldout_cid"],
            "input_binding": input_binding(),
            "manifest_cid": manifest["manifest_cid"],
            "measured_cell_count": 0,
            "metric_cell_count": len(metric_cells),
            "metric_count": len(METRIC_DEFINITIONS),
            "metrics_cid": metrics["metrics_cid"],
            "observed_effects": zero_effects(),
            "pair_count": len(pair_rows),
            "reason_codes": list(REASON_CODES),
            "receipt_cids": receipt_cids,
            "result_identity": "RESULT(PGIR-206)",
            "schema": "pgir-task-result@1",
            "seed_cid": seeds["seed_cid"],
            "task_binding": task_binding(),
            "training_admitted_rows": 0,
        },
        "result_cid",
        "result_sha256",
    )
    payloads["result.json"] = result
    return payloads


def expected_bytes() -> dict[str, bytes]:
    return {name: render_json(payload) for name, payload in build_payloads().items()}


def materialize(*, check: bool) -> dict[str, Any]:
    require_real_package_directory()
    for source_name in ("README.md", "build_not_run.py", "verify_not_run.py"):
        source_stat = os.lstat(PACKAGE_DIR / source_name)
        require(stat.S_ISREG(source_stat.st_mode), f"{source_name} is not a regular file")
        require(not stat.S_ISLNK(source_stat.st_mode), f"{source_name} must not be a symlink")
    expected = expected_bytes()
    expected_generated = set(EXPECTED_FILES) - {"README.md", "build_not_run.py", "verify_not_run.py"}
    require(set(expected) == expected_generated, "internal generated inventory drift")
    created: list[str] = []
    exact: list[str] = []
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    package_fd = os.open(PACKAGE_DIR, directory_flags)
    receipts_fd: int | None = None
    try:
        try:
            receipts_stat = os.stat("receipts", dir_fd=package_fd, follow_symlinks=False)
        except FileNotFoundError:
            require(not check, "missing receipts directory")
            try:
                os.mkdir("receipts", 0o755, dir_fd=package_fd)
                os.fsync(package_fd)
            except FileExistsError:
                pass
            receipts_stat = os.stat("receipts", dir_fd=package_fd, follow_symlinks=False)
        require(stat.S_ISDIR(receipts_stat.st_mode), "receipts is not a directory")
        require(not stat.S_ISLNK(receipts_stat.st_mode), "receipts must not be a symlink")
        receipts_fd = os.open("receipts", directory_flags, dir_fd=package_fd)
        for name in sorted(expected):
            if name.startswith("receipts/"):
                directory_fd = receipts_fd
                leaf_name = name.removeprefix("receipts/")
            else:
                directory_fd = package_fd
                leaf_name = name
            try:
                file_stat = os.stat(leaf_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                require(not check, f"missing {name}")
                create_at(directory_fd, leaf_name, expected[name])
                created.append(name)
                continue
            require(stat.S_ISREG(file_stat.st_mode), f"{name} is not a regular file")
            require(not stat.S_ISLNK(file_stat.st_mode), f"{name} must not be a symlink")
            require(
                read_at(directory_fd, leaf_name) == expected[name],
                f"refusing to replace different bytes in {name}",
            )
            exact.append(name)
    finally:
        if receipts_fd is not None:
            os.close(receipts_fd)
        os.close(package_fd)
    return {
        "artifact_count": len(EXPECTED_FILES),
        "check_only": check,
        "created": created,
        "decision": "no_go",
        "exact": exact,
        "execution_status": "not_run",
        "result_cid": expected_bytes_result_cid(expected["result.json"]),
        "task_id": TASK_ID,
    }


def expected_bytes_result_cid(data: bytes) -> str:
    value = json.loads(data)
    result_cid = value.get("result_cid")
    require(isinstance(result_cid, str), "result CID absent")
    return result_cid


def require_real_package_directory() -> None:
    """Reject redirected package paths before any artifact write."""

    package_stat = os.lstat(PACKAGE_DIR)
    require(stat.S_ISDIR(package_stat.st_mode), "package path is not a directory")
    require(not stat.S_ISLNK(package_stat.st_mode), "package path must not be a symlink")
    require(PACKAGE_DIR.resolve() == PACKAGE_DIR, "package path traverses a symlink")
    try:
        PACKAGE_DIR.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise BuildError("package directory escapes the repository") from exc


def read_at(directory_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        file_stat = os.fstat(fd)
        require(stat.S_ISREG(file_stat.st_mode), f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def create_at(directory_fd: int, name: str, data: bytes) -> None:
    """Create exactly one artifact without following or replacing a name."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(name, flags, 0o644, dir_fd=directory_fd)
    created_stat = os.fstat(fd)
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise OSError(f"short write while creating {name}")
            offset += written
        os.fsync(fd)
    except BaseException:
        try:
            try:
                named_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (named_stat.st_dev, named_stat.st_ino) == (
                    created_stat.st_dev,
                    created_stat.st_ino,
                ):
                    os.unlink(name, dir_fd=directory_fd)
        finally:
            os.close(fd)
        raise
    os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="read-only replay; require every generated artifact byte to be exact",
    )
    args = parser.parse_args(argv)
    try:
        summary = materialize(check=args.check)
    except (BuildError, OSError, ValueError) as exc:
        print(f"PGIR-206 build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
