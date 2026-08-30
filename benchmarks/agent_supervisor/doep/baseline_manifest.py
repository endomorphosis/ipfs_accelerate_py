"""Sealed Codex-primed baseline for DOEP paired qualification.

This module freezes the exact DOEP-PLAN-V5 source forest, policy, and sealed
validation-environment identities that represent the current Codex-campaign-
primed operating mode.  It does not implement a second priming subsystem or
measurement harness; DOEP-111 builds the harness on top of this seal.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

SCHEMA = "ipfs_accelerate_py.agent_supervisor.doep.codex-primed-baseline-manifest@1"
PROGRAM_ID = "agent-supervisor-direct-objective-and-event-driven-planning-v1"
PLAN_REVISION = "DOEP-PLAN-V5"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
PLAN_EPOCH = 1
TASK_ID = "DOEP-001"
TASK_CID = "sha256:940f797121528fe86fe8f86fb808d3cb8c60e6bdd20b20e3439b45bc11a94b74"
GOAL_ID = "DOEP-G010.S2"
PARENT_GOAL_ID = "DOEP-G010"
ROOT_GOAL_ID = "DOEP-G000"
BOARD_NAMESPACE = PROGRAM_ID
VALIDATION_PROFILE = "doep-validation/DOEP-PLAN-V5/DOEP-001@1"

# Immutable source forest bound by DOEP-PLAN-V5. Branch names are navigation
# only; receipts and measurements must bind these exact commits and trees.
BASE_REPOSITORIES: dict[str, dict[str, str]] = {
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
}

MODE = "codex_primed"
MODE_DESCRIPTION = (
    "Historical Codex-campaign-primed operating mode: a custom campaign prompt "
    "and primed context are required before the first task. The direct "
    "supervisor candidate replaces this priming path after qualification."
)

SEALED_VALIDATION_ENVIRONMENT: dict[str, str] = {
    "path": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
    "python_interpreter": "/usr/bin/python3.12",
    "formal_toolchain_deployment_identity": (
        "fa9916ef2e4a927ae633309de7ef09b9d85e798e9a6711ae13bc502c6015e0c8"
    ),
    "private_home_prefix": "ipfs-accelerate-validation-home-",
}

POLICY_BINDINGS: dict[str, str] = {
    "plan_markdown_sha256": (
        "sha256:4cf5bbd5ea983c27e73da581095e523a1e3c253cce29ea426d296f1135a1252f"
    ),
    "objectives_markdown_sha256": (
        "sha256:258f2b5da0417d2125d3c08c1f50f76f8eec443ce4f015fc2a9f7295fbdcee77"
    ),
    "todo_markdown_sha256": (
        "sha256:3b8a94cf297c0636dbf8278d002147367527ed0d58e40c87e5f61add4dac6222"
    ),
    "board_json_sha256": (
        "sha256:c4b0516b645f2e575e863115552bd642f575aca8fea4efcd7cb0b375d5504dfd"
    ),
    "validation_profiles_json_sha256": (
        "sha256:bbb8289468109b87cb25f8a8a94c67fd27c08e16d3c10954d7d003b1393bd7af"
    ),
    "scheduler_json_sha256": (
        "sha256:ffa6a94a752f20543c113ac7f11298c76ca3d6c1978fae0215d588a434091e2f"
    ),
}

AUTHORITY_POLICY: dict[str, Any] = {
    "operational_state": (
        "DuckDB transactional authority through an exclusive Quack state owner"
    ),
    "history_analytics": (
        "optional non-authoritative DuckLake projection; never scheduling, "
        "completion, policy, or proof authority"
    ),
    "completion_authoritative_for_worker": False,
    "model_claim_is_completion_authority": False,
    "simulated_is_never_live": True,
    "estimates_are_never_measurements": True,
    "missing_measurements_remain_missing": True,
}

UNAVAILABLE: dict[str, str] = {
    "paired_measurements": (
        "not_run; DOEP-111..125 execute harnesses and paired cohorts later"
    ),
    "promotion_decision": (
        "held; promotion requires closed DOEP-G130 evidence with zero hard-gate "
        "violations"
    ),
    "custom_codex_campaign_prompt_bytes": (
        "not sealed as authoritative input; priming is an operating mode "
        "identity, not a substitutable policy blob"
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def build_baseline_manifest() -> dict[str, Any]:
    """Return the sealed Codex-primed baseline identity (deterministic)."""

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "program_id": PROGRAM_ID,
        "board_namespace": BOARD_NAMESPACE,
        "plan_revision": PLAN_REVISION,
        "plan_cid": PLAN_CID,
        "plan_epoch": PLAN_EPOCH,
        "task_id": TASK_ID,
        "task_cid": TASK_CID,
        "goal_id": GOAL_ID,
        "parent_goal_id": PARENT_GOAL_ID,
        "root_goal_id": ROOT_GOAL_ID,
        "validation_profile": VALIDATION_PROFILE,
        "mode": MODE,
        "mode_description": MODE_DESCRIPTION,
        "baseline": {
            "label": "codex_primed",
            "base_repositories": BASE_REPOSITORIES,
            "provider_fixture": "hermetic-no-paid-provider-v1",
            "model_configuration": "no-live-model-quality-claim",
            "policy": "doep-closed-codex-primed-baseline-v1",
            "token_accounting": "existing-provider-usage-and-token-ledgers-v1",
        },
        "candidate": {
            "label": "direct_supervisor",
            "repository_commit": "unsealed",
            "repository_tree": "unsealed",
            "status": "not_yet_qualified",
        },
        "sealed_validation_environment": SEALED_VALIDATION_ENVIRONMENT,
        "policy_bindings": POLICY_BINDINGS,
        "authority_policy": AUTHORITY_POLICY,
        "unavailable": UNAVAILABLE,
        "measurements": {
            "status": "not_run",
            "baseline_receipt_id": None,
            "candidate_receipt_id": None,
            "token_and_model_call_change": None,
            "human_intervention_change": None,
            "safety_gate_result": None,
        },
        "promotion_eligible": False,
        "non_promotion_reason": (
            "paired baseline and candidate execution have not run; "
            "measurements remain explicitly not_run"
        ),
        "competing_subsystem_created": False,
        "authority": False,
        "generation": 1,
    }
    body = {key: value for key, value in manifest.items()}
    manifest["manifest_cid"] = sha256_digest(canonical_json_bytes(body))
    return manifest


def assert_sealed_identity(manifest: Mapping[str, Any]) -> None:
    """Fail closed when a loaded manifest drifts from the sealed constants."""

    assert manifest.get("schema") == SCHEMA
    assert manifest.get("program_id") == PROGRAM_ID
    assert manifest.get("plan_revision") == PLAN_REVISION
    assert manifest.get("plan_cid") == PLAN_CID
    assert manifest.get("task_id") == TASK_ID
    assert manifest.get("task_cid") == TASK_CID
    assert manifest.get("mode") == MODE
    assert manifest.get("promotion_eligible") is False
    measurements = manifest.get("measurements")
    assert isinstance(measurements, dict)
    assert measurements.get("status") == "not_run"
    baseline = manifest.get("baseline")
    assert isinstance(baseline, dict)
    assert baseline.get("base_repositories") == BASE_REPOSITORIES
    assert manifest.get("competing_subsystem_created") is False
    assert manifest.get("authority") is False
    env = manifest.get("sealed_validation_environment")
    assert isinstance(env, dict)
    assert env.get("path") == SEALED_VALIDATION_ENVIRONMENT["path"]
    assert env.get("python_interpreter") == SEALED_VALIDATION_ENVIRONMENT[
        "python_interpreter"
    ]
    assert env.get("formal_toolchain_deployment_identity") == (
        SEALED_VALIDATION_ENVIRONMENT["formal_toolchain_deployment_identity"]
    )


def load_or_build_manifest() -> dict[str, Any]:
    """Build the sealed manifest. Pure; does not mutate the tree."""

    return build_baseline_manifest()
