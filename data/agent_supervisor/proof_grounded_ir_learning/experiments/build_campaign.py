#!/usr/bin/env python3
"""Build the immutable PGIR-110 R1-R6 controlled-campaign evidence tree.

The builder is deterministic and write-once.  It binds every arm and seed to
the PGIR-014 ``IRCampaignInputRoot@1`` freeze, attempts admission, and records
the fail-closed no-go without inventing training rows, hidden-test scores, or
model weights.  Existing experiment bytes are never replaced: a changed input
requires a new superseding campaign location and task revision.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parents[3]
FREEZE_DIR = EXPERIMENTS_DIR.parent / "freeze"

TASK_ID = "PGIR-110"
TASK_TITLE = "Run R1-R6 controlled campaign"
OBJECTIVE_ID = "PGIR-G110"
PARENT_GOAL = "PGIR-G110"
SUBGOAL = "controlled-comparisons"
CAMPAIGN_SCHEMA = "IRControlledCampaign@1"
ARM_SCHEMA = "IRExperimentArm@1"
CHECKPOINT_SCHEMA = "IRExperimentCheckpointManifest@1"
EVALUATION_SCHEMA = "IRExperimentEvaluation@1"
COMPARISON_SCHEMA = "IRExperimentComparison@1"
RESULT_SCHEMA = "pgir-task-result@1"
RECIPE_SCHEMA = "IRControlledCampaignRecipe@1"
SEED_POLICY_SCHEMA = "IRExperimentSeedPolicy@1"
HELDOUT_SCHEMA = "IRExperimentHeldoutBinding@1"
METRIC_CATALOG_SCHEMA = "IRExperimentMetricCatalog@1"
MANIFEST_SCHEMA = "IRControlledCampaignManifest@1"
RECEIPT_SCHEMA = "IRExperimentReceipt@1"

CURRENT_TASK_CID = "baguqeeragjtn4knjvdexk4ya373ljydixx5r6c4moxr3sj6apuf2slfutexq"
REVISED_TASK_CID = "baguqeeranjhprx27smrkacgi7t5wuwu5kogvlishd77wwddi3dgzsxcijeba"
REVISED_TASK_KEY = "task/v1/6a4ef8df5f9322a008c8fcfb6a5a9d538d55a2471fff6b0c68d8cd995c484902"
SEMANTIC_FINGERPRINT = "6a4ef8df5f9322a008c8fcfb6a5a9d538d55a2471fff6b0c68d8cd995c484902"
OBJECTIVE_REVISION = "baguqeeragjtn4knjvdexk4ya373ljydixx5r6c4moxr3sj6apuf2slfutexq"

CAMPAIGN_INPUT_ROOT_CID = "baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q"
FREEZE_RESULT_CID = "baguqeerai2ipwhyywztjob62ju5pokmm4o6unqqee3poyrabj37aby6fuoca"
FREEZE_RESULT_IDENTITY = "RESULT(PGIR-014)"
PLAN_ADMISSION_RECEIPT_ID = "baguqeera2nvxj5chnrbyu2yzpbyny6gmccqegjg6ydpfzaulav65lbcelcgq"
REVISION_SET_CID = "baguqeerakogtsnyz26ycza6qhzh5uysuyg6xffsfskvsumnwpeqvtzs57hga"
REPOSITORY_ID = (
    "repository:sha256:4d87e009c221f83df2c5846e6085d4917204de75df8dc438b045c3bbff059dbc"
)
SOURCE_TREE_ID = "04fbb09b4a8b34e77d11bd8da6642e0978baa02c"
SOURCE_SET_ID = "SRCSET-1"
DATASETS_COMMIT = "df93e91e6338c84a17c3208ef68b88de8566f78c"
ACCELERATE_COMMIT = "8d46a6d25dd006c8cab3c9d9612707d2a014e79c"
DATASETS_SELECTED_COMMIT = "b20bd9e3cfae79e8888929daf64f52b2f8a5689a"

SPLIT_BINDING = "RESULT(PGIR-012)"
CORPUS_BINDING = "RESULT(PGIR-011)"
COMPILER_BINDING = "RESULT(PGIR-021)"
DECOMPILER_BINDING = "RESULT(PGIR-022)"
COMPILER_ALIAS = "COMPILER-CURRENT-1"
DECOMPILER_ALIAS = "DECOMPILER-CURRENT-1"
SPLIT_MANIFEST_DIGEST = "sha256:047b263b85067aa3dad6760f623c2855fbaf776d565ec9c273c49425fcc14eb4"
SPLIT_MANIFEST_SHA256 = "sha256:9e552a46d1f850fd0455d2c5b1d87810077fd35eb88ea849e64de24090bc167f"
SPLIT_ROOT_SHA256 = "sha256:b522f15f2597ed4902f1af9b7f3aac5b855193d289369df70ccfda5ce8798f9d"
HIDDEN_TEST_COMMITMENT = "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
SPLIT_BINDING_CID = "baguqeerajllfwoyk3ztsigcstzdnslee6qg22rndwprupylolfn6iv5lifya"
CORPUS_BINDING_CID = "baguqeera4ezrwsq2kupnh4l52z2jfglbzalr6flzjw3xsyefnu5luv6v2hta"
COMPILER_BINDING_CID = "baguqeerarxul7wfpbcka6lzwbe6afembjteohwfdps6mjh7pntg7xlz7ywta"
DECOMPILER_BINDING_CID = "baguqeeracirjnyp5plk7uobfwoj2bol3hgqhnr6a4k7geipnex3drntqrnxq"
TOKENIZER_POLICY_CID = "baguqeerahoedy5eyabjpcixpxwcjlbh54femnpb3krvirv3scto4nqfkplua"
TOKENIZER_POLICY_STATUS = "no_learned_tokenizer_admitted"
R1_HISTORICAL_REPORT_CID = "baguqeerau73uowpiy22d7rohi7gvbtfkwivyldlivxnxqw4zzc7zeyzavasq"
R1_HISTORICAL_RECIPE_CID = "baguqeerazuhonzzynznbhtlfgmsbmlrl4fzs73ogo3ogek4e5xbb577unuea"
R1_HISTORICAL_MANIFEST_CID = "baguqeeraf3mevd4zrpkcy6hmsamfyszkq5zeisq2ipu6bvupquprtfqi53ta"
LOSS_CONFIGURATION_IDENTITY = "IRLossConfiguration@1"
DEFAULT_SAMPLER_SEED = 32
RESOURCE_PROFILE = "RP-MIXED"
LEASE_POLICY = "LEASE-DEFAULT"

REASON_CODES = (
    "corpus_not_materialized",
    "historical_semantic_baseline_not_currently_qualified",
    "no_rights_admitted_training_rows",
    "required_holdouts_insufficient",
)
INSUFFICIENT_HOLDOUTS = (
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
POPULATED_HOLDOUTS = (
    ("family", "statute_family", 4999),
    ("jurisdiction", "jurisdiction", 74),
)
EXCLUDED_POPULATIONS = (
    "blind_holdout",
    "holdout_cases",
    "hidden_test",
    "canary",
    "statute_family",
    "jurisdiction",
)
HISTORICAL_R1_POPULATIONS = ("pilot", "repair_development")

DEPENDS_ON = (
    "PGIR-033",
    "PGIR-041",
    "PGIR-053",
    "PGIR-062",
    "PGIR-071",
    "PGIR-081",
    "PGIR-100",
)
DEPENDENCY_TASK_CIDS = {
    "PGIR-014": "baguqeeralbl2yjo6l5gazcmslpzqtu67un4txk3wwpjr45thh5sckwq67yhq",
    "PGIR-033": "baguqeeraec4hazq72xwydpro254qhwmkbku4ltt4zztufxisilmypnfy3j4q",
    "PGIR-041": "baguqeeraaqbbaijc5io2yhzzg3yalmk4omyeihzz6k232a5tv4agtmwjzfaq",
    "PGIR-053": "baguqeerawj24izvekc2cafc7a4uxdxwvij6apmowrozz4usshrml44jymnba",
    "PGIR-062": "baguqeerads3akcfu5xrj756eye3f4kltzdonl73adn7tkfpmcj6nqfvap34a",
    "PGIR-071": "baguqeerafzx2mkqlntkaw7ublebojigyh2ibjzcm2dgqvzhopxtqx7624rjq",
    "PGIR-081": "baguqeeran4usi4fm4b5tzu72kzcb5cpe5aljp6iwdlpg33ph2kzftyfnkdoa",
    "PGIR-100": "baguqeerainaqx5w72m2epc7wsolk3elqihpcganr6d3lrsuzseldzkqngija",
}

SURFACES = ("compiler", "decompiler")
DECODE_MODES = ("teacher_forcing", "free_run")
ARCHITECTURES = ("shared_latent", "shared_encoder_typed_head")
LEARNED_SEEDS = (32, 33, 34)
DETERMINISTIC_SEED = 0

N_MEASURES = (
    ("N1", "token_cross_entropy", "nats_per_canonical_token", False),
    ("N2", "latent_separation", "score", True),
    ("N3", "retrieval_recall", "rate", True),
    ("N4", "structural_equivalence", "rate", True),
    ("N5", "semantic_equivalence", "rate", True),
    ("N6", "proof_replay_rate", "rate", True),
    ("N7", "readability_score", "score", True),
    ("N8", "calibration_error", "error", False),
    ("N8", "ood_acceptance", "rate", True),
)
CAMPAIGN_MEASURES = (
    ("latency", "milliseconds", False),
    ("resource_cost", "milli_resource_units", False),
)
E1_MEASURES = (
    "parser_acceptance",
    "type_acceptance",
    "exact",
    "canonical",
    "ast",
    "graph",
    "source_span",
    "semantic",
    "proof",
    "unsupported",
    "latency",
)
TOKEN_CLASSES = (
    "padding",
    "binder",
    "operator",
    "type",
    "source",
    "family",
    "proof",
    "tactic",
)

ARMS: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "R1",
        "kind": "deterministic",
        "title": "deterministic compiler/decompiler baseline",
        "learned": False,
        "seeds": (DETERMINISTIC_SEED,),
        "loss_components": (),
        "proof_in_gradient_path": False,
        "uses_model": False,
        "historical_baseline": True,
    },
    {
        "arm_id": "R2",
        "kind": "token_ce",
        "title": "token-class cross-entropy only",
        "learned": True,
        "seeds": LEARNED_SEEDS,
        "loss_components": ("token_class_ce",),
        "proof_in_gradient_path": False,
        "uses_model": True,
        "historical_baseline": False,
    },
    {
        "arm_id": "R3",
        "kind": "ce_cosine",
        "title": "token-class CE plus normalized cosine",
        "learned": True,
        "seeds": LEARNED_SEEDS,
        "loss_components": ("token_class_ce", "normalized_cosine"),
        "proof_in_gradient_path": False,
        "uses_model": True,
        "historical_baseline": False,
    },
    {
        "arm_id": "R4",
        "kind": "supervised_contrastive",
        "title": "supervised contrastive with false-negative filter",
        "learned": True,
        "seeds": LEARNED_SEEDS,
        "loss_components": ("supervised_contrastive",),
        "proof_in_gradient_path": False,
        "uses_model": True,
        "historical_baseline": False,
    },
    {
        "arm_id": "R5",
        "kind": "full_multitask",
        "title": "full multi-task IRLossConfiguration@1 mix",
        "learned": True,
        "seeds": LEARNED_SEEDS,
        "loss_components": (
            "token_class_ce",
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
        "proof_in_gradient_path": False,
        "uses_model": True,
        "historical_baseline": False,
    },
    {
        "arm_id": "R6",
        "kind": "proof_grounded_curriculum",
        "title": "proof-grounded curriculum with nondifferentiable proof labels",
        "learned": True,
        "seeds": LEARNED_SEEDS,
        "loss_components": (
            "token_class_ce",
            "normalized_cosine",
            "supervised_contrastive",
            "cycle",
            "structural",
            "relation",
            "semantic",
            "proof",
            "source_span",
            "calibration",
            "regularization",
        ),
        "proof_in_gradient_path": False,
        "uses_model": True,
        "historical_baseline": False,
    },
)


class CampaignBuildError(RuntimeError):
    """Raised when the controlled campaign cannot be sealed without invention."""


def _validate_canonical_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise CampaignBuildError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CampaignBuildError(f"{path} contains a non-string key")
        for key, item in value.items():
            _validate_canonical_value(item, f"{path}.{key}")
        return
    raise CampaignBuildError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    _validate_canonical_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rendered_bytes(value: Any) -> bytes:
    _validate_canonical_value(value)
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
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


def add_projection_identity(
    payload: Mapping[str, Any], *, cid_field: str, sha_field: str | None = None
) -> dict[str, Any]:
    result = dict(payload)
    projection = dict(result)
    if sha_field:
        result[sha_field] = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    result[cid_field] = dag_json_cid(projection)
    return result


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CampaignBuildError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                CampaignBuildError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                CampaignBuildError(f"non-finite number {raw!r} in {path}")
            ),
        )
    if not isinstance(value, dict):
        raise CampaignBuildError(f"{path} must contain a JSON object")
    return value


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise CampaignBuildError(
                f"refusing to replace different bytes at {path.relative_to(REPOSITORY_ROOT)}"
            )
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    data = rendered_bytes(payload)
    write_once(path, data)
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def seed_label(seed: int, *, learned: bool) -> str:
    if not learned:
        return "deterministic"
    return f"seed-{seed}"


def checkpoint_stem(arm_id: str, seed: int, *, learned: bool) -> str:
    return f"{arm_id.lower()}-{seed_label(seed, learned=learned)}"


def holdout_map() -> dict[str, Any]:
    holdouts: dict[str, Any] = {}
    populated = {name: (split, count) for name, split, count in POPULATED_HOLDOUTS}
    required = set(INSUFFICIENT_HOLDOUTS) | set(populated)
    extra = ("family", "jurisdiction")
    for name in (*INSUFFICIENT_HOLDOUTS, *extra):
        if name in populated:
            split, count = populated[name]
            holdouts[name] = {"count": count, "split": split, "status": "populated"}
        else:
            holdouts[name] = {
                "count": 0,
                "split": "temporal" if name == "time" else name,
                "status": "insufficient",
            }
    if set(holdouts) != required:
        raise CampaignBuildError("holdout map drifted from freeze")
    return holdouts


def freeze_gates() -> None:
    root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    result = strict_json(FREEZE_DIR / "result.v3.json")
    revisions = strict_json(FREEZE_DIR / "descendant_task_revisions.json")
    if root.get("root_cid") != CAMPAIGN_INPUT_ROOT_CID:
        raise CampaignBuildError("freeze campaign root CID drifted")
    if root.get("qualification", {}).get("decision") != "no_go":
        raise CampaignBuildError("freeze is not a no_go; refusing to invent a new decision")
    if root.get("qualification", {}).get("descendant_execution_authorized") is not False:
        raise CampaignBuildError("freeze authorizes descendant execution")
    if root.get("qualification", {}).get("training_admitted_rows") != 0:
        raise CampaignBuildError("freeze reports admitted training rows")
    if result.get("result_cid") != FREEZE_RESULT_CID:
        raise CampaignBuildError("freeze result CID drifted")
    pgir_110 = next(
        item
        for item in revisions.get("revisions", [])
        if item.get("task_id") == TASK_ID
    )
    if pgir_110.get("lease_eligible") is not False:
        raise CampaignBuildError("PGIR-110 is unexpectedly lease-eligible")
    if pgir_110.get("revised_task_cid") != REVISED_TASK_CID:
        raise CampaignBuildError("PGIR-110 revised task CID drifted")
    if pgir_110.get("current_task_cid") != CURRENT_TASK_CID:
        raise CampaignBuildError("PGIR-110 current task CID drifted")
    if tuple(root.get("qualification", {}).get("reason_codes") or ()) != REASON_CODES:
        raise CampaignBuildError("freeze reason codes drifted")
    if tuple(root.get("qualification", {}).get("insufficient_holdouts") or ()) != INSUFFICIENT_HOLDOUTS:
        raise CampaignBuildError("freeze insufficient holdouts drifted")


def metric_entry(
    *,
    metric_id: str,
    n_metric: str | None,
    unit: str,
    higher_is_better: bool | None,
    surface: str,
    status: str,
    reason: str,
    decode_mode: str | None = None,
    token_class: str | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "architecture": architecture,
        "confidence_interval": None,
        "decode_mode": decode_mode,
        "denominator": 0,
        "higher_is_better": higher_is_better,
        "metric_id": metric_id,
        "missing_as_zero": False,
        "n_metric": n_metric,
        "numerator": None,
        "reason": reason,
        "sample_count": 0,
        "status": status,
        "surface": surface,
        "token_class": token_class,
        "unit": unit,
        "value": None,
    }
    return payload


def evaluation_metrics(arm: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    holdout_reason = "required_holdouts_insufficient"
    rights_reason = "no_rights_admitted_training_rows"
    tokenizer_reason = "no_learned_tokenizer_admitted"
    if arm["learned"]:
        common_reason = (
            "admission_denied:"
            + ",".join(REASON_CODES)
            + ";training_not_started"
        )
    else:
        common_reason = (
            "historical_semantic_baseline_not_currently_qualified;"
            + holdout_reason
        )
    for surface in SURFACES:
        for n_metric, metric_id, unit, higher in N_MEASURES:
            if metric_id == "token_cross_entropy":
                if not arm["learned"]:
                    reason = f"unsupported:{tokenizer_reason};deterministic_arm_has_no_token_ce"
                    status = "unsupported"
                    metrics.append(
                        metric_entry(
                            metric_id=metric_id,
                            n_metric=n_metric,
                            unit=unit,
                            higher_is_better=higher,
                            surface=surface,
                            status=status,
                            reason=reason,
                        )
                    )
                    continue
                for decode_mode in DECODE_MODES:
                    for token_class in TOKEN_CLASSES:
                        metrics.append(
                            metric_entry(
                                metric_id=metric_id,
                                n_metric=n_metric,
                                unit=unit,
                                higher_is_better=higher,
                                surface=surface,
                                status="unavailable",
                                reason=f"{common_reason};{tokenizer_reason}",
                                decode_mode=decode_mode,
                                token_class=token_class,
                            )
                        )
                continue
            metrics.append(
                metric_entry(
                    metric_id=metric_id,
                    n_metric=n_metric,
                    unit=unit,
                    higher_is_better=higher,
                    surface=surface,
                    status="unavailable",
                    reason=common_reason,
                )
            )
        for metric_id, unit, higher in CAMPAIGN_MEASURES:
            metrics.append(
                metric_entry(
                    metric_id=metric_id,
                    n_metric=None,
                    unit=unit,
                    higher_is_better=higher,
                    surface=surface,
                    status="unavailable",
                    reason=common_reason + ";no_measured_campaign_calls",
                )
            )
        if arm["historical_baseline"]:
            for metric_id in E1_MEASURES:
                metrics.append(
                    metric_entry(
                        metric_id=f"e1_{metric_id}",
                        n_metric=None,
                        unit="historical_reference_only",
                        higher_is_better=None,
                        surface=surface,
                        status="historical_not_currently_qualified",
                        reason=(
                            "historical_fixture=RESULT(PGIR-023);"
                            f"report_cid={R1_HISTORICAL_REPORT_CID};"
                            "populations=pilot,repair_development;"
                            "hidden_test_selection=false;"
                            f"{holdout_reason};"
                            "not_a_campaign_holdout_measurement"
                        ),
                    )
                )
    if arm["learned"]:
        metrics.append(
            metric_entry(
                metric_id="target_attainment",
                n_metric=None,
                unit="claim",
                higher_is_better=None,
                surface="campaign",
                status="not_attained",
                reason="no_fabricated_target_attainment;" + rights_reason,
            )
        )
    else:
        metrics.append(
            metric_entry(
                metric_id="target_attainment",
                n_metric=None,
                unit="claim",
                higher_is_better=None,
                surface="campaign",
                status="not_attained",
                reason="no_fabricated_target_attainment;deterministic_fixture_not_currently_qualified",
            )
        )
    return metrics


def build_heldouts() -> dict[str, Any]:
    return add_projection_identity(
        {
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "excluded_populations": list(EXCLUDED_POPULATIONS),
            "hidden_labels_opened": False,
            "hidden_test_commitment": HIDDEN_TEST_COMMITMENT,
            "hidden_test_selection": False,
            "historical_r1_populations": list(HISTORICAL_R1_POPULATIONS),
            "holdouts": holdout_map(),
            "identical_across_arms": True,
            "interface": "proof-grounded-ir-learning/experiment-heldouts/v1",
            "kind": "ir-sealed-split-root/v1",
            "leakage_passed": True,
            "schema": HELDOUT_SCHEMA,
            "schema_binding": SPLIT_BINDING,
            "split_binding_cid": SPLIT_BINDING_CID,
            "split_manifest_digest": SPLIT_MANIFEST_DIGEST,
            "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
            "split_root_sha256": SPLIT_ROOT_SHA256,
            "task_id": TASK_ID,
        },
        cid_field="heldout_cid",
        sha_field="heldout_sha256",
    )


def build_seeds() -> dict[str, Any]:
    return add_projection_identity(
        {
            "best_test_selection": False,
            "deterministic_arm_seeds": [DETERMINISTIC_SEED],
            "hidden_test_tuning": False,
            "interface": "proof-grounded-ir-learning/experiment-seed-policy/v1",
            "learned_arm_seeds": list(LEARNED_SEEDS),
            "loss_configuration_identity": LOSS_CONFIGURATION_IDENTITY,
            "policy": "same heldouts and seeds for every R1-R6 arm; no post-hoc seed drop",
            "sampler_seed_origin": "IRLossConfiguration@1 DEFAULT_SAMPLER_SEED plus two successors",
            "schema": SEED_POLICY_SCHEMA,
            "source_seed": DEFAULT_SAMPLER_SEED,
            "task_id": TASK_ID,
        },
        cid_field="seed_policy_cid",
        sha_field="seed_policy_sha256",
    )


def build_metric_catalog() -> dict[str, Any]:
    metrics = []
    for n_metric, metric_id, unit, higher in N_MEASURES:
        metrics.append(
            {
                "higher_is_better": higher,
                "metric_id": metric_id,
                "missing_as_zero": False,
                "n_metric": n_metric,
                "required": True,
                "unit": unit,
            }
        )
    for metric_id, unit, higher in CAMPAIGN_MEASURES:
        metrics.append(
            {
                "higher_is_better": higher,
                "metric_id": metric_id,
                "missing_as_zero": False,
                "n_metric": None,
                "required": True,
                "unit": unit,
            }
        )
    for metric_id in E1_MEASURES:
        metrics.append(
            {
                "higher_is_better": None,
                "metric_id": f"e1_{metric_id}",
                "missing_as_zero": False,
                "n_metric": None,
                "required": False,
                "unit": "historical_reference_only",
            }
        )
    metrics.append(
        {
            "higher_is_better": None,
            "metric_id": "target_attainment",
            "missing_as_zero": False,
            "n_metric": None,
            "required": True,
            "unit": "claim",
        }
    )
    return add_projection_identity(
        {
            "decode_modes": list(DECODE_MODES),
            "interface": "proof-grounded-ir-learning/experiment-metric-catalog/v1",
            "metrics": metrics,
            "missing_as_zero": False,
            "schema": METRIC_CATALOG_SCHEMA,
            "surfaces": list(SURFACES),
            "task_id": TASK_ID,
            "token_classes": list(TOKEN_CLASSES),
        },
        cid_field="catalog_cid",
        sha_field="catalog_sha256",
    )


def build_recipe(heldouts: Mapping[str, Any], seeds: Mapping[str, Any]) -> dict[str, Any]:
    return add_projection_identity(
        {
            "architectures": list(ARCHITECTURES),
            "arms": [
                {
                    "arm_id": arm["arm_id"],
                    "historical_baseline": arm["historical_baseline"],
                    "kind": arm["kind"],
                    "learned": arm["learned"],
                    "loss_components": list(arm["loss_components"]),
                    "proof_in_gradient_path": arm["proof_in_gradient_path"],
                    "seeds": list(arm["seeds"]),
                    "title": arm["title"],
                    "uses_model": arm["uses_model"],
                }
                for arm in ARMS
            ],
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "compiler_identity": COMPILER_BINDING,
            "data_split_identity": SPLIT_BINDING,
            "decompiler_identity": DECOMPILER_BINDING,
            "depends_on": list(DEPENDS_ON),
            "excluded_populations": list(EXCLUDED_POPULATIONS),
            "heldout_cid": heldouts["heldout_cid"],
            "hidden_test_commitment": HIDDEN_TEST_COMMITMENT,
            "hidden_test_selection": False,
            "interface": "proof-grounded-ir-learning/controlled-campaign-recipe/v1",
            "lease_policy": LEASE_POLICY,
            "learned_inference_authorized": False,
            "loss_configuration_identity": LOSS_CONFIGURATION_IDENTITY,
            "missing_metric_as_zero": False,
            "resource_profile": RESOURCE_PROFILE,
            "schema": RECIPE_SCHEMA,
            "seed_policy_cid": seeds["seed_policy_cid"],
            "source_dataset_revisions": CORPUS_BINDING,
            "task_id": TASK_ID,
            "tokenizer_policy_status": TOKENIZER_POLICY_STATUS,
        },
        cid_field="recipe_cid",
        sha_field="recipe_sha256",
    )


def build_checkpoint(
    arm: Mapping[str, Any],
    seed: int,
    *,
    heldouts: Mapping[str, Any],
    seeds: Mapping[str, Any],
) -> dict[str, Any]:
    learned = bool(arm["learned"])
    if learned:
        identity = f"none/not-created/{arm['arm_id'].lower()}/{seed_label(seed, learned=True)}"
        status = "not_created"
        reason = "admission_denied;no_shared_checkpoint_write;training_not_started"
    else:
        identity = "none/deterministic"
        status = "none_deterministic"
        reason = "deterministic_arm_has_no_model_weights;historical_baseline_not_currently_qualified"
    return add_projection_identity(
        {
            "architectures_instantiated": [],
            "architectures_intended": list(ARCHITECTURES) if learned else [],
            "arm_id": arm["arm_id"],
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "compiler_identity": COMPILER_BINDING,
            "data_split_identity": SPLIT_BINDING,
            "decompiler_identity": DECOMPILER_BINDING,
            "heldout_cid": heldouts["heldout_cid"],
            "interface": "proof-grounded-ir-learning/experiment-checkpoint/v1",
            "learned": learned,
            "loss_components": list(arm["loss_components"]),
            "loss_configuration_identity": LOSS_CONFIGURATION_IDENTITY,
            "model_checkpoint_identity": identity,
            "proof_in_gradient_path": False,
            "reason": reason,
            "schema": CHECKPOINT_SCHEMA,
            "seed": seed,
            "seed_label": seed_label(seed, learned=learned),
            "seed_policy_cid": seeds["seed_policy_cid"],
            "shared_checkpoint_write": False,
            "status": status,
            "task_id": TASK_ID,
            "tokenizer_policy_status": TOKENIZER_POLICY_STATUS,
            "uses_model": bool(arm["uses_model"]),
            "weights": {
                "digest": None,
                "path": None,
                "reason": reason,
                "status": "not_created",
            },
        },
        cid_field="checkpoint_cid",
        sha_field="checkpoint_sha256",
    )


def build_evaluation(
    arm: Mapping[str, Any],
    seed: int,
    checkpoint: Mapping[str, Any],
    *,
    heldouts: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = evaluation_metrics(arm)
    statuses = sorted({item["status"] for item in metrics})
    return add_projection_identity(
        {
            "arm_id": arm["arm_id"],
            "best_test_selection": False,
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "catalog_cid": catalog["catalog_cid"],
            "checkpoint_cid": checkpoint["checkpoint_cid"],
            "fabricated_target_attainment": False,
            "heldout_cid": heldouts["heldout_cid"],
            "hidden_test_opened": False,
            "hidden_test_selection": False,
            "historical_r1_manifest_cid": (
                R1_HISTORICAL_MANIFEST_CID if arm["historical_baseline"] else None
            ),
            "historical_r1_recipe_cid": (
                R1_HISTORICAL_RECIPE_CID if arm["historical_baseline"] else None
            ),
            "historical_r1_report_cid": (
                R1_HISTORICAL_REPORT_CID if arm["historical_baseline"] else None
            ),
            "interface": "proof-grounded-ir-learning/experiment-evaluation/v1",
            "metric_count": len(metrics),
            "metrics": metrics,
            "missing_metric_as_zero": False,
            "model_checkpoint_identity": checkpoint["model_checkpoint_identity"],
            "paired_uncertainty": "unavailable",
            "reason_codes": list(REASON_CODES),
            "schema": EVALUATION_SCHEMA,
            "seed": seed,
            "seed_label": checkpoint["seed_label"],
            "statuses": statuses,
            "surfaces": list(SURFACES),
            "task_id": TASK_ID,
        },
        cid_field="evaluation_cid",
        sha_field="evaluation_sha256",
    )


def lease_record(key: str, *, scope: str, arm_id: str | None, seed: int | None) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "attempt": 1,
        "disposition": "never_granted",
        "fence": 1,
        "key": key,
        "policy": LEASE_POLICY,
        "reason": "lease_ineligible;freeze_descendant_execution_unauthorized",
        "renewable": True,
        "scope": scope,
        "seed": seed,
    }


def build_receipts(
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    leases = [
        lease_record("campaign:reducer", scope="reducer", arm_id=None, seed=None),
        lease_record("campaign:comparison", scope="comparison", arm_id=None, seed=None),
    ]
    for arm in ARMS:
        leases.append(
            lease_record(
                f"arm:{arm['arm_id']}",
                scope="arm",
                arm_id=arm["arm_id"],
                seed=None,
            )
        )
        for seed in arm["seeds"]:
            label = seed_label(seed, learned=bool(arm["learned"]))
            prefix = f"arm:{arm['arm_id']}/{label}"
            leases.append(lease_record(prefix, scope="arm_seed", arm_id=arm["arm_id"], seed=seed))
            leases.append(
                lease_record(
                    f"{prefix}/checkpoint",
                    scope="checkpoint",
                    arm_id=arm["arm_id"],
                    seed=seed,
                )
            )
            leases.append(
                lease_record(
                    f"{prefix}/proof",
                    scope="proof",
                    arm_id=arm["arm_id"],
                    seed=seed,
                )
            )
            leases.append(
                lease_record(
                    f"{prefix}/evaluation",
                    scope="evaluation",
                    arm_id=arm["arm_id"],
                    seed=seed,
                )
            )
    admission = add_projection_identity(
        {
            "admitted": False,
            "authorizes_execution": False,
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "freeze_plan_admission_receipt_id": PLAN_ADMISSION_RECEIPT_ID,
            "interface": "proof-grounded-ir-learning/experiment-admission/v1",
            "lease_eligible": False,
            "reason_codes": list(REASON_CODES),
            "revised_task_cid": REVISED_TASK_CID,
            "schema": RECEIPT_SCHEMA,
            "task_id": TASK_ID,
            "training_admitted_rows": 0,
            "verdict": "rejected",
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    lease_receipt = add_projection_identity(
        {
            "granted_count": 0,
            "interface": "proof-grounded-ir-learning/experiment-leases/v1",
            "lease_count": len(leases),
            "leases": leases,
            "policy": LEASE_POLICY,
            "production_pointer_mutated": False,
            "reducer_cas": "reserved_not_applied",
            "schema": RECEIPT_SCHEMA,
            "task_id": TASK_ID,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    resources = add_projection_identity(
        {
            "bounded_exhaustion": {
                "gpu_ms": "not_started",
                "kind": "admission_closed",
                "proof_ms": "not_started",
                "tokens": "not_started",
                "typed": True,
            },
            "interface": "proof-grounded-ir-learning/experiment-resources/v1",
            "profile": RESOURCE_PROFILE,
            "proof_ms": 0,
            "provider_tokens": 0,
            "schema": RECEIPT_SCHEMA,
            "task_id": TASK_ID,
            "training_gpu_ms": 0,
            "training_steps": 0,
            "training_started": False,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    proof = add_projection_identity(
        {
            "attempts": 0,
            "authority": False,
            "checked_counterexamples": 0,
            "interface": "proof-grounded-ir-learning/experiment-proof/v1",
            "kernel_verified": 0,
            "reason": "proof_loop_not_started;admission_denied",
            "schema": RECEIPT_SCHEMA,
            "task_id": TASK_ID,
            "timeout_as_falsehood": False,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    training = add_projection_identity(
        {
            "arms_started": 0,
            "failed_experiments_deleted": False,
            "interface": "proof-grounded-ir-learning/experiment-training/v1",
            "reason": "training_not_started;admission_denied",
            "schema": RECEIPT_SCHEMA,
            "shared_checkpoint_writes": 0,
            "task_id": TASK_ID,
            "threshold_weakened": False,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    evaluation = add_projection_identity(
        {
            "evaluation_cids": [item["evaluation_cid"] for item in evaluations],
            "hidden_test_opened": False,
            "interface": "proof-grounded-ir-learning/experiment-evaluation-receipt/v1",
            "measured_campaign_holdout_metrics": 0,
            "schema": RECEIPT_SCHEMA,
            "task_id": TASK_ID,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    reducer = add_projection_identity(
        {
            "cas_applied": False,
            "checkpoint_cids": [item["checkpoint_cid"] for item in checkpoints],
            "interface": "proof-grounded-ir-learning/experiment-reducer/v1",
            "promotion_pointer_mutated": False,
            "reason": "no_admitted_candidate",
            "schema": RECEIPT_SCHEMA,
            "task_id": TASK_ID,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )
    return {
        "admission": admission,
        "evaluation": evaluation,
        "leases": lease_receipt,
        "proof": proof,
        "reducer": reducer,
        "resources": resources,
        "training": training,
    }


def build_arm_record(
    arm: Mapping[str, Any],
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    arm_checkpoints = [item for item in checkpoints if item["arm_id"] == arm["arm_id"]]
    arm_evaluations = [item for item in evaluations if item["arm_id"] == arm["arm_id"]]
    return add_projection_identity(
        {
            "arm_id": arm["arm_id"],
            "checkpoint_cids": [item["checkpoint_cid"] for item in arm_checkpoints],
            "decision": "no_go",
            "evaluation_cids": [item["evaluation_cid"] for item in arm_evaluations],
            "historical_baseline": arm["historical_baseline"],
            "interface": "proof-grounded-ir-learning/experiment-arm/v1",
            "kind": arm["kind"],
            "learned": arm["learned"],
            "loss_components": list(arm["loss_components"]),
            "proof_in_gradient_path": False,
            "reason_codes": list(REASON_CODES),
            "schema": ARM_SCHEMA,
            "seeds": list(arm["seeds"]),
            "status": "admission_denied",
            "task_id": TASK_ID,
            "title": arm["title"],
            "training_started": False,
            "uses_model": arm["uses_model"],
        },
        cid_field="arm_cid",
        sha_field="arm_sha256",
    )


def build_comparison(
    *,
    arm_records: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    heldouts: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = []
    arm_ids = [arm["arm_id"] for arm in ARMS]
    for index, left in enumerate(arm_ids):
        for right in arm_ids[index + 1 :]:
            pairs.append(
                {
                    "baseline": left,
                    "candidate": right,
                    "reason": "no_measured_paired_cases;admission_denied",
                    "status": "unavailable",
                    "winner": None,
                }
            )
    return add_projection_identity(
        {
            "admitted_candidate": None,
            "arm_cids": [item["arm_cid"] for item in arm_records],
            "best_test_selection": False,
            "catalog_cid": catalog["catalog_cid"],
            "decision": "no_go",
            "evaluation_cids": [item["evaluation_cid"] for item in evaluations],
            "fabricated_target_attainment": False,
            "heldout_cid": heldouts["heldout_cid"],
            "hidden_test_opened": False,
            "interface": "proof-grounded-ir-learning/experiment-comparison/v1",
            "pairs": pairs,
            "promotion_authorized": False,
            "reason_codes": list(REASON_CODES),
            "schema": COMPARISON_SCHEMA,
            "same_heldouts": True,
            "same_seed_policy": True,
            "task_id": TASK_ID,
            "winner": None,
        },
        cid_field="comparison_cid",
        sha_field="comparison_sha256",
    )


def build_campaign(
    *,
    recipe: Mapping[str, Any],
    heldouts: Mapping[str, Any],
    seeds: Mapping[str, Any],
    catalog: Mapping[str, Any],
    arm_records: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    return add_projection_identity(
        {
            "architectures": list(ARCHITECTURES),
            "arm_cids": [item["arm_cid"] for item in arm_records],
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "catalog_cid": catalog["catalog_cid"],
            "checkpoint_cids": [item["checkpoint_cid"] for item in checkpoints],
            "comparison_cid": comparison["comparison_cid"],
            "compiler_identity": COMPILER_BINDING,
            "current_task_cid": CURRENT_TASK_CID,
            "data_split_identity": SPLIT_BINDING,
            "decision": "no_go",
            "decompiler_identity": DECOMPILER_BINDING,
            "depends_on": [
                {
                    "campaign_lease_eligible": False,
                    "current_task_cid": DEPENDENCY_TASK_CIDS[task_id],
                    "task_id": task_id,
                }
                for task_id in DEPENDS_ON
            ],
            "evaluation_cids": [item["evaluation_cid"] for item in evaluations],
            "freeze_result_cid": FREEZE_RESULT_CID,
            "freeze_result_identity": FREEZE_RESULT_IDENTITY,
            "heldout_cid": heldouts["heldout_cid"],
            "interface": "proof-grounded-ir-learning/controlled-campaign/v1",
            "lease_eligible": False,
            "objective_id": OBJECTIVE_ID,
            "parent_goal": PARENT_GOAL,
            "receipt_cids": {
                name: item["receipt_cid"] for name, item in sorted(receipts.items())
            },
            "recipe_cid": recipe["recipe_cid"],
            "resource_profile": RESOURCE_PROFILE,
            "revised_task_cid": REVISED_TASK_CID,
            "schema": CAMPAIGN_SCHEMA,
            "seed_policy_cid": seeds["seed_policy_cid"],
            "source_dataset_revisions": CORPUS_BINDING,
            "subgoal": SUBGOAL,
            "task_id": TASK_ID,
            "title": TASK_TITLE,
            "training_admitted_rows": 0,
            "training_started": False,
        },
        cid_field="campaign_cid",
        sha_field="campaign_sha256",
    )


def build_result(
    *,
    campaign: Mapping[str, Any],
    comparison: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    manifest_cid: str,
) -> dict[str, Any]:
    return add_projection_identity(
        {
            "campaign_cid": campaign["campaign_cid"],
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "comparison_cid": comparison["comparison_cid"],
            "completion_authoritative": False,
            "compiler_identity": COMPILER_BINDING,
            "data_split_identity": SPLIT_BINDING,
            "decision": "no_go",
            "decompiler_identity": DECOMPILER_BINDING,
            "descendant_execution_authorized": False,
            "disposition": "frozen_no_go",
            "executed_task_revision": REVISED_TASK_CID,
            "freeze_result_cid": FREEZE_RESULT_CID,
            "lease": {
                "attempt": 1,
                "disposition": "terminal-no-grant",
                "fence": 1,
                "policy": LEASE_POLICY,
            },
            "manifest_cid": manifest_cid,
            "model_checkpoint_identity": "per-arm-seed/not-created",
            "objective_revision": OBJECTIVE_REVISION,
            "plan_admission_receipt_id": PLAN_ADMISSION_RECEIPT_ID,
            "reason_codes": list(REASON_CODES),
            "receipt_cids": {
                name: item["receipt_cid"] for name, item in sorted(receipts.items())
            },
            "repository_id": REPOSITORY_ID,
            "result_identity": "RESULT(PGIR-110)",
            "revision_set_cid": REVISION_SET_CID,
            "rollback": "retain this immutable campaign and create a separately admitted superseding freeze before any R1-R6 training lease",
            "schema": RESULT_SCHEMA,
            "source_dataset_revisions": CORPUS_BINDING,
            "source_tree_id": SOURCE_TREE_ID,
            "task_id": TASK_ID,
            "training_task_eligible_count": 0,
            "unresolved_identities": [],
        },
        cid_field="result_cid",
        sha_field="result_sha256",
    )


def build_manifest(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return add_projection_identity(
        {
            "files": {
                name: {
                    "path": item["path"],
                    "raw_cid": item["raw_cid"],
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                }
                for name, item in sorted(files.items())
            },
            "interface": "proof-grounded-ir-learning/controlled-campaign-manifest/v1",
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
        },
        cid_field="manifest_cid",
        sha_field="manifest_sha256",
    )


def relative_payload(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    data = rendered_bytes(payload)
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def assemble() -> dict[str, Any]:
    freeze_gates()
    heldouts = build_heldouts()
    seeds = build_seeds()
    catalog = build_metric_catalog()
    recipe = build_recipe(heldouts, seeds)
    checkpoints: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in arm["seeds"]:
            checkpoint = build_checkpoint(arm, seed, heldouts=heldouts, seeds=seeds)
            evaluation = build_evaluation(
                arm, seed, checkpoint, heldouts=heldouts, catalog=catalog
            )
            checkpoints.append(checkpoint)
            evaluations.append(evaluation)
    receipts = build_receipts(checkpoints=checkpoints, evaluations=evaluations)
    arm_records = [
        build_arm_record(arm, checkpoints=checkpoints, evaluations=evaluations)
        for arm in ARMS
    ]
    comparison = build_comparison(
        arm_records=arm_records,
        evaluations=evaluations,
        heldouts=heldouts,
        catalog=catalog,
    )
    campaign = build_campaign(
        recipe=recipe,
        heldouts=heldouts,
        seeds=seeds,
        catalog=catalog,
        arm_records=arm_records,
        checkpoints=checkpoints,
        evaluations=evaluations,
        receipts=receipts,
        comparison=comparison,
    )
    files: dict[str, dict[str, Any]] = {}
    planned = {
        "recipe.json": recipe,
        "heldouts.json": heldouts,
        "seeds.json": seeds,
        "metric_catalog.json": catalog,
        "campaign.json": campaign,
        "comparison.json": comparison,
    }
    for name, payload in planned.items():
        files[name] = relative_payload(EXPERIMENTS_DIR / name, payload)
    for record in arm_records:
        name = f"arms/{record['arm_id']}.json"
        files[name] = relative_payload(EXPERIMENTS_DIR / name, record)
    for checkpoint in checkpoints:
        name = f"checkpoints/{checkpoint_stem(checkpoint['arm_id'], checkpoint['seed'], learned=checkpoint['learned'])}.json"
        files[name] = relative_payload(EXPERIMENTS_DIR / name, checkpoint)
    for evaluation in evaluations:
        name = f"evaluations/{checkpoint_stem(evaluation['arm_id'], evaluation['seed'], learned=evaluation['arm_id'] != 'R1')}.json"
        files[name] = relative_payload(EXPERIMENTS_DIR / name, evaluation)
    for name, payload in receipts.items():
        rel = f"receipts/{name}.json"
        files[rel] = relative_payload(EXPERIMENTS_DIR / rel, payload)
    manifest = build_manifest(files)
    result = build_result(
        campaign=campaign,
        comparison=comparison,
        receipts=receipts,
        manifest_cid=manifest["manifest_cid"],
    )
    files["manifest.json"] = relative_payload(EXPERIMENTS_DIR / "manifest.json", manifest)
    files["result.json"] = relative_payload(EXPERIMENTS_DIR / "result.json", result)
    return {
        "arm_records": arm_records,
        "campaign": campaign,
        "catalog": catalog,
        "checkpoints": checkpoints,
        "comparison": comparison,
        "evaluations": evaluations,
        "files": files,
        "heldouts": heldouts,
        "manifest": manifest,
        "receipts": receipts,
        "recipe": recipe,
        "result": result,
        "seeds": seeds,
    }


def persist(bundle: Mapping[str, Any]) -> None:
    mapping = {
        EXPERIMENTS_DIR / "recipe.json": bundle["recipe"],
        EXPERIMENTS_DIR / "heldouts.json": bundle["heldouts"],
        EXPERIMENTS_DIR / "seeds.json": bundle["seeds"],
        EXPERIMENTS_DIR / "metric_catalog.json": bundle["catalog"],
        EXPERIMENTS_DIR / "campaign.json": bundle["campaign"],
        EXPERIMENTS_DIR / "comparison.json": bundle["comparison"],
        EXPERIMENTS_DIR / "manifest.json": bundle["manifest"],
        EXPERIMENTS_DIR / "result.json": bundle["result"],
    }
    for record in bundle["arm_records"]:
        mapping[EXPERIMENTS_DIR / "arms" / f"{record['arm_id']}.json"] = record
    for checkpoint in bundle["checkpoints"]:
        mapping[
            EXPERIMENTS_DIR
            / "checkpoints"
            / f"{checkpoint_stem(checkpoint['arm_id'], checkpoint['seed'], learned=checkpoint['learned'])}.json"
        ] = checkpoint
    for evaluation in bundle["evaluations"]:
        mapping[
            EXPERIMENTS_DIR
            / "evaluations"
            / f"{checkpoint_stem(evaluation['arm_id'], evaluation['seed'], learned=evaluation['arm_id'] != 'R1')}.json"
        ] = evaluation
    for name, payload in bundle["receipts"].items():
        mapping[EXPERIMENTS_DIR / "receipts" / f"{name}.json"] = payload
    for path, payload in mapping.items():
        write_json(path, payload)


def write_runtime_checkpoint(bundle: Mapping[str, Any]) -> None:
    directory = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR")
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign_cid": bundle["campaign"]["campaign_cid"],
        "comparison_cid": bundle["comparison"]["comparison_cid"],
        "decision": "no_go",
        "result_cid": bundle["result"]["result_cid"],
        "schema": "pgir-110-runtime-checkpoint@1",
        "task_id": TASK_ID,
    }
    sealed = add_projection_identity(payload, cid_field="checkpoint_cid", sha_field="checkpoint_sha256")
    path = root / "pgir-110.result.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(rendered_bytes(sealed))
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="assemble the campaign and refuse to write files",
    )
    args = parser.parse_args(argv)
    bundle = assemble()
    if not args.check:
        persist(bundle)
        write_runtime_checkpoint(bundle)
    print(
        json.dumps(
            {
                "campaign_cid": bundle["campaign"]["campaign_cid"],
                "comparison_cid": bundle["comparison"]["comparison_cid"],
                "decision": bundle["result"]["decision"],
                "result_cid": bundle["result"]["result_cid"],
                "task_id": TASK_ID,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignBuildError as exc:
        print(f"campaign build failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
