#!/usr/bin/env python3
"""Build the immutable PGIR-111 qualification, decision, and next board.

The builder is deterministic and write-once.  It applies the closed 16 final
acceptance criteria and 32 report sections to already-sealed campaign
evidence.  It never invents metrics, never opens hidden tests, never promotes
a model, and never uploads.  Existing qualification bytes are never replaced:
a changed input requires a new superseding qualification location and task
revision.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


QUALIFICATION_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = QUALIFICATION_DIR.parents[3]
FREEZE_DIR = QUALIFICATION_DIR.parent / "freeze"
EXPERIMENTS_DIR = QUALIFICATION_DIR.parent / "experiments"
DOCS_DIR = REPOSITORY_ROOT / "docs" / "architecture" / "proof_grounded_ir_learning"
DATASETS_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"

TASK_ID = "PGIR-111"
TASK_TITLE = "Qualify, publish or reject, and issue the next board"
OBJECTIVE_ID = "PGIR-G110"
PARENT_GOAL = "PGIR-G110"
SUBGOAL = "final-decision-report"
RECIPE_SCHEMA = "IRQualificationRecipe@1"
ACCEPTANCE_SCHEMA = "IRFinalAcceptanceCatalog@1"
SECTIONS_SCHEMA = "IRFinalReportSections@1"
DECISION_SCHEMA = "IRQualificationDecision@1"
PROMOTION_SCHEMA = "IRQualificationPromotionReceipt@1"
PUBLICATION_SCHEMA = "IRQualificationPublicationReceipt@1"
MANIFEST_SCHEMA = "IRQualificationManifest@1"
RESULT_SCHEMA = "pgir-task-result@1"
VERIFICATION_SCHEMA = "IRQualificationVerificationReceipt@1"

TASK_CID = "baguqeerad2idzhwzfqlyxjh7u34bczkyutsvq7oxcmuaglupr436gqfo6kga"
OBJECTIVE_REVISION = TASK_CID
REPOSITORY_ID = (
    "repository:sha256:4d87e009c221f83df2c5846e6085d4917204de75df8dc438b045c3bbff059dbc"
)
SOURCE_TREE_ID = "04fbb09b4a8b34e77d11bd8da6642e0978baa02c"
SOURCE_SET_ID = "SRCSET-1"
DATASETS_AUTHORITY = "df93e91e6338c84a17c3208ef68b88de8566f78c"
ACCELERATE_AUTHORITY = "8d46a6d25dd006c8cab3c9d9612707d2a014e79c"
DATASETS_CHECKOUT = "d144be65ffe4c6423e4e1c30cd692812607343eb"
ACCELERATE_CHECKOUT = "0cc04ebb640c4c981cf4650016e096a73ab0e8c0"
DATASETS_TREE = "37b9cb40644831c85c6fdf07d0228e45061e239a"
ACCELERATE_TREE = "697ee660025fbf14a1cbe6c24fd8da5365df84d5"
DATASETS_REVIEWED = "7f0fe2bbad3c70928234c6e2312ee3182fd7681f"
ACCELERATE_REVIEWED = "c821d0b43877591bbb0fa3f328fbccff187b56e7"
DATASETS_LIVE_DELTA = "2f93a232612d1b8d1da6b52abfa1639621a86ac82eef2180f163eaa9d6b547f4"
ACCELERATE_LIVE_DELTA = "0d13706bbdd5f50118999dc928172c8f0df29aea8f86613b0f5664e60435c87c"
DATASETS_RANGE_LOG = "aaeff6d8976787159e8ec747fc60a5d27b6515773068c06e968cfb3a107dd21e"
ACCELERATE_RANGE_LOG = "0a70de8c18be990e59660a0a4cbaf00cf81cf31b3321ad9b03bab0a666eaf61e"
DATASETS_SELECTED = "b20bd9e3cfae79e8888929daf64f52b2f8a5689a"
PINSET_SHA256 = "8e3a4b1bd81639393ddda35e5dfb3b95f9e7320afa898bde0b3eb9a0317a6b76"
MODEL_LEGACY = (
    "justicedao/legal-ir-autoencoder-checkpoints@94ca549d102e3e31781370aec1247f91365440eb"
)
MODEL_LEGACY_STATE = "7236de26bd3d7f8414ffa04805f1b6e8a8849f9e0103cec6edb4985b911658be"

FREEZE_RESULT_CID = "baguqeerai2ipwhyywztjob62ju5pokmm4o6unqqee3poyrabj37aby6fuoca"
FREEZE_ROOT_CID = "baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q"
TOKENIZER_POLICY_CID = "baguqeerahoedy5eyabjpcixpxwcjlbh54femnpb3krvirv3scto4nqfkplua"
CORPUS_MANIFEST_CID = "bafkreiha35x7mcukzzb5x67hmykwsny5wipf5jb4do5gpsl24mxvix55n4"
SPLIT_MANIFEST_DIGEST = "sha256:047b263b85067aa3dad6760f623c2855fbaf776d565ec9c273c49425fcc14eb4"
SPLIT_MANIFEST_SHA256 = "sha256:9e552a46d1f850fd0455d2c5b1d87810077fd35eb88ea849e64de24090bc167f"
SPLIT_ROOT_SHA256 = "sha256:b522f15f2597ed4902f1af9b7f3aac5b855193d289369df70ccfda5ce8798f9d"
HIDDEN_TEST_COMMITMENT = "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
R1_REPORT_CID = "baguqeerau73uowpiy22d7rohi7gvbtfkwivyldlivxnxqw4zzc7zeyzavasq"
R1_RECIPE_CID = "baguqeerazuhonzzynznbhtlfgmsbmlrl4fzs73ogo3ogek4e5xbb577unuea"
R1_MANIFEST_CID = "baguqeeraf3mevd4zrpkcy6hmsamfyszkq5zeisq2ipu6bvupquprtfqi53ta"
SOURCE_MANIFEST_CID = "baguqeerasownoxqyrppw3ft3us3yvd26ghvqnjl74nr2rw5o7sm3sjehip7a"
SOURCE_RESULT_IDENTITY = "sha256:86f7c07cf7b62847b81315b6529942da7647acf9922f32cf5ccef9d8bb221e9c"
MODULES_IDENTITY = "sha256:532e1c8b60fcf77515e772368f80475d2f62ba1545490826c3baeea1b402920b"
SUPERVISOR_INVENTORY_CID = "baguqeerablvf72zunpjvbievbspxqnc4eqgxneqjwrg5v6imr7edavovmwca"
RELEASE_INVENTORY_IDENTITY = (
    "sha256:f58457d29289d5140bfc63596f28a4648e2bad1ea6222e7b62d22e8cf9b95bb6"
)
GAP_MATRIX_CID = "baguqeeraspldjlypaoamdclucjsbktrmkramzhe3kra7tip5h3e5s5zkfnia"
BASELINE_SUMMARY_CID = "baguqeerasr4hxpxwkhe64btsxu56moh7v7ww2a6covaavjd7qnzb4xgl5cpq"
POSITIVE_PAIR_RECIPE_CID = "bafkreigxfl76kqkwaydsbbgagirxw6i2fnoymzxcd3sernzkeey3qwa76u"
NEGATIVE_PAIR_RECIPE_CID = "bafkreifzdty5u3uf34z7id3e2mjdidy3fkoucloe2qnudywspggygv4cvy"
LOCAL_RELEASE_CID = "bafkreigwdei25h3eg2k2l6gp6ak5tbkbcfabi6vsoqzrzv6k6mpm73lhge"
LOCAL_RELEASE_ID = "sha256-b8b062360926fa1fb09c22f44740982f9401f435b524cc41db08e093a206c425"
P4_EVIDENCE_CID = "bafkreia35e5pexkbnq7x2lqtoomcwx34hceroyzsltkb4rqirjjvlqkdle"
PROOF_ROOT = "bafkreiedk7zooeftd4qnhysbuazs6ulntis3ixn5vye6q7bgtxgrdlrfna"
PGIR_110_TASK_CID = "baguqeeragjtn4knjvdexk4ya373ljydixx5r6c4moxr3sj6apuf2slfutexq"
PGIR_110_REVISED_TASK_CID = "baguqeeranjhprx27smrkacgi7t5wuwu5kogvlishd77wwddi3dgzsxcijeba"
PGIR_072_TASK_CID = "baguqeeraarlt745ftpwax4tdovajsxrgp5r72fkudlbs5kmg3q2cbqow6mpq"
PGIR_090_TASK_CID = "baguqeeragt5xhqov2e5vna6pp2zmghhrqhq2v4qzl36a3aucytrcom2erkta"
PGIR_100_TASK_CID = "baguqeerainaqx5w72m2epc7wsolk3elqihpcganr6d3lrsuzseldzkqngija"

COMPILER_BINDING = "RESULT(PGIR-021)"
DECOMPILER_BINDING = "RESULT(PGIR-022)"
CORPUS_BINDING = "RESULT(PGIR-011)"
SPLIT_BINDING = "RESULT(PGIR-012)"
LOSS_CONFIGURATION = "IRLossConfiguration@1"
RESOURCE_PROFILE = "RP-CPU-M"
LEASE_POLICY = "LEASE-DEFAULT"

QUALIFIED_CLAIM = (
    "The qualified compiler/decompiler checkpoint was trained from "
    "content-addressed, lineage-safe JusticeDAO source and proof artifacts under "
    "the declared split, compiler, tokenizer, loss, curriculum, and supervisor "
    "configuration. It achieved the reported held-out token, latent-retrieval, "
    "structural, semantic, and proof metrics without exceeding the admitted "
    "regression thresholds. Learned outputs remain candidates until validated by "
    "the canonical parser, type system, translation contracts, and required proof "
    "or counterexample authorities."
)

REASON_CODES = (
    "corpus_not_materialized",
    "historical_semantic_baseline_not_currently_qualified",
    "no_candidate_checkpoint",
    "no_learned_tokenizer_admitted",
    "no_rights_admitted_training_rows",
    "publication_not_authorized",
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
M2_GATES = (
    "lineage",
    "syntax",
    "type",
    "semantic",
    "proof",
    "calibration",
    "family",
    "jurisdiction",
    "source_span",
    "latency",
    "resource",
)
METRIC_FAMILIES = (
    "token",
    "latent_retrieval",
    "structural",
    "semantic",
    "proof",
    "calibration_ood",
    "latency_resource",
)
SOURCE_RELEASES = (
    (
        "justicedao/patent-legal-ir-graphrag",
        "845669408081f1334c54519d2bb7df6bf780ccd5",
        2174,
        "quarantined",
    ),
    (
        "justicedao/wetwijzer_netherlands_legal_corpus",
        "827e9412f55cbe332f18824ff669bdbbae39005d",
        4999,
        "quarantined",
    ),
)
DEPENDS_ON = ("PGIR-072", "PGIR-090", "PGIR-100", "PGIR-110")
DEPENDENCY_TASK_CIDS = {
    "PGIR-014": "baguqeeralbl2yjo6l5gazcmslpzqtu67un4txk3wwpjr45thh5sckwq67yhq",
    "PGIR-072": PGIR_072_TASK_CID,
    "PGIR-090": PGIR_090_TASK_CID,
    "PGIR-100": PGIR_100_TASK_CID,
    "PGIR-110": PGIR_110_TASK_CID,
}

REPORT_SECTIONS = (
    (1, "exact_source_revisions", "Exact source revisions"),
    (2, "justicedao_revisions", "Exact JusticeDAO repository revisions and configurations"),
    (3, "current_state_inventory", "Current-state inventory"),
    (4, "source_versus_derived", "Source versus derived record counts"),
    (5, "lineage_and_split", "Lineage and split design"),
    (6, "leakage_audit", "Leakage-audit results"),
    (7, "canonical_bridge", "Canonical bridge-IR design"),
    (8, "compiler_architecture", "Compiler architecture"),
    (9, "decompiler_architecture", "Decompiler architecture"),
    (10, "deterministic_baseline", "Deterministic baseline"),
    (11, "learned_model_architecture", "Learned-model architecture"),
    (12, "tokenizer_and_vocabulary", "Tokenizer and vocabulary"),
    (13, "loss_configuration", "Loss configuration"),
    (14, "training_curriculum", "Training curriculum"),
    (15, "hard_negative_generation", "Hard-negative generation"),
    (16, "lean_capable_results", "Lean-capable model results"),
    (17, "tactician_results", "Tactician results"),
    (18, "hammer_results", "Hammer results"),
    (19, "kernel_verification", "Kernel-verification results"),
    (20, "cross_entropy_metrics", "Cross-entropy metrics"),
    (21, "cosine_contrastive_metrics", "Cosine and contrastive metrics"),
    (22, "retrieval_metrics", "Retrieval metrics"),
    (23, "structural_metrics", "Structural metrics"),
    (24, "semantic_metrics", "Semantic metrics"),
    (25, "proof_metrics", "Proof metrics"),
    (26, "calibration_ood_metrics", "Calibration and OOD metrics"),
    (27, "resource_utilization", "Resource utilization"),
    (28, "multi_supervisor_scheduling", "Multi-supervisor scheduling results"),
    (29, "promotion_or_rejection", "Checkpoint promotion or rejection decision"),
    (30, "published_artifacts", "Published artifacts"),
    (31, "known_limitations", "Known limitations"),
    (32, "next_board_recommendation", "Exact recommendation for the next training and data-improvement board"),
)

ACCEPTANCE_CRITERIA = (
    (
        "F01",
        "current_input_child_evidence",
        "All PGIR-G010 through PGIR-G110 children have fresh current-input evidence.",
        "G000",
    ),
    (
        "F02",
        "no_source_lineage_leakage",
        "No source-lineage leakage across related derivatives.",
        "G000",
    ),
    (
        "F03",
        "one_canonical_bridge",
        "One canonical typed bridge is bound and used.",
        "G000",
    ),
    (
        "F04",
        "deterministic_baseline_measured",
        "A current-input deterministic baseline is measured and qualified.",
        "G000",
    ),
    (
        "F05",
        "proof_aware_contracts",
        "Proof-aware pair, loss, and evaluation contracts are sealed.",
        "G000",
    ),
    (
        "F06",
        "resumable_resource_aware_campaign",
        "A resumable resource-aware campaign exists and may execute only from sealed inputs.",
        "G000",
    ),
    (
        "F07",
        "deterministic_promotion_or_nogo",
        "Promotion is deterministic policy admission or a documented no-go.",
        "G000",
    ),
    (
        "F08",
        "authorized_append_only_publication",
        "Append-only qualified publication occurs only when independently authorized.",
        "G000",
    ),
    (
        "F09",
        "token_metrics_with_uncertainty",
        "Actual token metrics are reported with paired uncertainty.",
        "G110",
    ),
    (
        "F10",
        "latent_retrieval_metrics_with_uncertainty",
        "Actual latent and retrieval metrics are reported with paired uncertainty.",
        "G110",
    ),
    (
        "F11",
        "structural_metrics_with_uncertainty",
        "Actual structural metrics are reported with paired uncertainty.",
        "G110",
    ),
    (
        "F12",
        "semantic_metrics_with_uncertainty",
        "Actual semantic metrics are reported with paired uncertainty.",
        "G110",
    ),
    (
        "F13",
        "proof_metrics_with_uncertainty",
        "Actual proof metrics are reported with paired uncertainty.",
        "G110",
    ),
    (
        "F14",
        "calibration_ood_metrics_with_uncertainty",
        "Actual calibration and OOD metrics are reported with paired uncertainty.",
        "G110",
    ),
    (
        "F15",
        "latency_resource_metrics_with_uncertainty",
        "Actual latency and resource results are reported with paired uncertainty.",
        "G110",
    ),
    (
        "F16",
        "no_hidden_test_tune_admitted_publish_next_board",
        "Hidden tests are never used for tuning; only an admitted candidate may be published; the next content-addressed board is produced.",
        "G110",
    ),
)


class QualificationBuildError(RuntimeError):
    """Raised when qualification cannot be sealed without invention."""


def _validate_canonical_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise QualificationBuildError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise QualificationBuildError(f"{path} contains a non-string key")
        for key, item in value.items():
            _validate_canonical_value(item, f"{path}.{key}")
        return
    raise QualificationBuildError(f"{path} contains unsupported {type(value).__name__}")


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


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise QualificationBuildError(
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


def write_text(path: Path, text: str) -> dict[str, Any]:
    data = text.encode("utf-8")
    write_once(path, data)
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualificationBuildError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                QualificationBuildError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                QualificationBuildError(f"non-finite number {raw!r} in {path}")
            ),
        )
    if not isinstance(value, dict):
        raise QualificationBuildError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationBuildError(message)


def load_freeze() -> dict[str, Any]:
    root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    result = strict_json(FREEZE_DIR / "result.v3.json")
    tokenizer = strict_json(FREEZE_DIR / "tokenizer_policy.json")
    require(root["root_cid"] == FREEZE_ROOT_CID, "freeze root CID drifted")
    require(result["result_cid"] == FREEZE_RESULT_CID, "freeze result CID drifted")
    require(result["decision"] == "no_go", "freeze decision is not no_go")
    require(tokenizer["policy_cid"] == TOKENIZER_POLICY_CID, "tokenizer policy CID drifted")
    require(tokenizer["status"] == "no_learned_tokenizer_admitted", "tokenizer unexpectedly admitted")
    return {"root": root, "result": result, "tokenizer": tokenizer}


def load_supporting_evidence() -> dict[str, Any]:
    corpus = strict_json(DATASETS_ROOT / "data/ir_learning/corpora/corpus_root.json")
    split = strict_json(DATASETS_ROOT / "data/ir_learning/splits/split_root.json")
    leakage = strict_json(DATASETS_ROOT / "data/ir_learning/splits/leakage_report.json")
    holdouts = strict_json(DATASETS_ROOT / "data/ir_learning/splits/holdout_report.json")
    publication_policy = strict_json(
        DATASETS_ROOT / "data/ir_learning/releases/publication_policy.json"
    )
    local_release = strict_json(DATASETS_ROOT / "data/ir_learning/releases/sealed/release_root.json")
    require(corpus["training_admitted_rows"] == 0, "training rows unexpectedly admitted")
    require(corpus["materialized"] is False, "corpus unexpectedly materialized")
    require(split["leakage_passed"] is True, "leakage report no longer passes")
    require(leakage["passed"] is True, "leakage audit no longer passes")
    require(holdouts["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT, "hidden-test commitment drifted")
    require(publication_policy["require_qualification"] is True, "publication policy dropped qualification")
    require(publication_policy["default_publication_admission"] == "deny", "publication default is not deny")
    require(local_release["release_cid"] == LOCAL_RELEASE_CID, "local release CID drifted")
    experiment_readme = (EXPERIMENTS_DIR / "README.md").read_text(encoding="utf-8")
    require("decision is deliberately `no_go`" in experiment_readme, "PGIR-110 README lost no-go")
    require((EXPERIMENTS_DIR / "result.json").exists() is False, "unexpected PGIR-110 result.json")
    return {
        "corpus": corpus,
        "split": split,
        "leakage": leakage,
        "holdouts": holdouts,
        "publication_policy": publication_policy,
        "local_release": local_release,
    }


def metric_unavailable(family: str) -> dict[str, Any]:
    return {
        "confidence_interval": "unavailable",
        "denominator": 0,
        "family": family,
        "hidden_test_used": False,
        "missing_as_zero": False,
        "numerator": 0,
        "paired_uncertainty": "unavailable",
        "reason": "admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized",
        "status": "not_run",
        "value": None,
    }


def evaluate_acceptance() -> list[dict[str, Any]]:
    resolutions = {
        "F01": (
            "resolved_with_evidence",
            "Inventory, freeze, contract, campaign, and security children have sealed current-input receipts. RESULT(PGIR-110) is a documented execution no-go, not a missing child.",
        ),
        "F02": (
            "satisfied",
            "IRSplitManifest@1 leakage audit passed with zero violations. Related derivatives remain in one leakage group.",
        ),
        "F03": (
            "satisfied_with_limitations",
            "COMPILER-CURRENT-1 and DECOMPILER-CURRENT-1 are bound. The gap matrix records remaining unsupported directions; no second canonical bridge was admitted.",
        ),
        "F04": (
            "no_go",
            "Historical RESULT(PGIR-023) fixture metrics exist by CID only and remain not currently qualified for campaign heldouts.",
        ),
        "F05": (
            "satisfied",
            "Closed pair, loss, and evaluation contracts are sealed. Model output cannot silently become canonical or proof-grounded.",
        ),
        "F06": (
            "satisfied_execution_denied",
            "IRLearningCampaign@1 and R1-R6 leases exist. Descendant execution remains unauthorized under RESULT(PGIR-014).",
        ),
        "F07": (
            "satisfied",
            "Independent promotion comparison produced a documented no-go. No candidate was compared and no pointer mutated.",
        ),
        "F08": (
            "satisfied",
            "Remote publication is denied. The PGIR-090 local package is not a qualified upload.",
        ),
        "F09": (
            "no_go",
            "Token metrics were reported as not_run with zero denominators and unavailable paired uncertainty.",
        ),
        "F10": (
            "no_go",
            "Latent and retrieval metrics were reported as not_run with zero denominators and unavailable paired uncertainty.",
        ),
        "F11": (
            "no_go",
            "Structural metrics were reported as not_run with zero denominators and unavailable paired uncertainty.",
        ),
        "F12": (
            "no_go",
            "Current-input semantic campaign metrics were reported as not_run. Historical R1 scores stay unqualified.",
        ),
        "F13": (
            "no_go",
            "Proof metrics were reported as not_run. No independently checked campaign proof entered curriculum.",
        ),
        "F14": (
            "no_go",
            "Calibration and OOD metrics were reported as not_run with unavailable paired uncertainty.",
        ),
        "F15": (
            "no_go",
            "Latency and resource campaign measures were reported as not_run. No GPU or prover lease was granted.",
        ),
        "F16": (
            "satisfied",
            "Hidden tests stayed sealed, no candidate was published, and the next content-addressed board is issued.",
        ),
    }
    records = []
    for criterion_id, slug, text, source in ACCEPTANCE_CRITERIA:
        status, evidence = resolutions[criterion_id]
        records.append(
            {
                "criterion_id": criterion_id,
                "evidence": evidence,
                "gate_pass": status.startswith("satisfied") or status == "resolved_with_evidence",
                "qualification_pass": status.startswith("satisfied")
                and criterion_id in {"F02", "F07", "F08", "F16"},
                "schema_source": source,
                "slug": slug,
                "status": status,
                "text": text,
            }
        )
    require(len(records) == 16, "acceptance catalog must contain 16 criteria")
    return records


def section_payloads() -> list[dict[str, Any]]:
    facts = {
        1: {
            "status": "resolved",
            "summary": "Campaign authority remains SRCSET-1. Reviewed revisions are comparison inputs only.",
            "bindings": {
                "datasets_authority_commit": DATASETS_AUTHORITY,
                "datasets_authority_tree": DATASETS_TREE,
                "datasets_checkout": DATASETS_CHECKOUT,
                "datasets_live_delta_sha256": DATASETS_LIVE_DELTA,
                "datasets_range_log_sha256": DATASETS_RANGE_LOG,
                "datasets_reviewed": DATASETS_REVIEWED,
                "datasets_reviewed_commits_ahead": 1717,
                "accelerate_authority_commit": ACCELERATE_AUTHORITY,
                "accelerate_authority_tree": ACCELERATE_TREE,
                "accelerate_checkout": ACCELERATE_CHECKOUT,
                "accelerate_live_delta_sha256": ACCELERATE_LIVE_DELTA,
                "accelerate_range_log_sha256": ACCELERATE_RANGE_LOG,
                "accelerate_reviewed": ACCELERATE_REVIEWED,
                "accelerate_reviewed_commits_ahead": 3616,
                "selected_datasets_commit": DATASETS_SELECTED,
                "source_manifest_cid": SOURCE_MANIFEST_CID,
                "source_result_identity": SOURCE_RESULT_IDENTITY,
                "source_set_id": SOURCE_SET_ID,
                "source_tree_id": SOURCE_TREE_ID,
            },
        },
        2: {
            "status": "resolved",
            "summary": "JDAO-PINSET-1 pins 21 Hub revisions and admits zero repositories for proof-grounded training.",
            "bindings": {
                "pinset_id": "JDAO-PINSET-1",
                "pinset_sha256": PINSET_SHA256,
                "public_dataset_repository_count": 21,
                "source_releases": [
                    {
                        "disposition": disposition,
                        "repository_id": repository_id,
                        "revision": revision,
                        "source_records": count,
                    }
                    for repository_id, revision, count, disposition in SOURCE_RELEASES
                ],
                "training_repositories_admitted": 0,
            },
        },
        3: {
            "status": "resolved",
            "summary": "Datasets, supervisor, release, baseline, and gap inventories are sealed. Baseline tests remain mixed and non-qualifying.",
            "bindings": {
                "accelerator_baseline": "360 passed",
                "baseline_summary_cid": BASELINE_SUMMARY_CID,
                "datasets_baseline": "801 passed, 2 failed, 2 skipped, 13 errors",
                "gap_matrix_cid": GAP_MATRIX_CID,
                "modules_identity": MODULES_IDENTITY,
                "release_inventory_identity": RELEASE_INVENTORY_IDENTITY,
                "supervisor_inventory_cid": SUPERVISOR_INVENTORY_CID,
            },
        },
        4: {
            "status": "resolved",
            "summary": "Source and derived counts stay distinct. Derivatives do not inflate source rows.",
            "bindings": {
                "derived_count": 38690,
                "patent_source_groups": 2174,
                "source_count": 7173,
                "training_admitted_rows": 0,
            },
        },
        5: {
            "status": "resolved_with_no_go",
            "summary": "Lineage-safe multidimensional splits exist. Thirteen required holdouts remain insufficient.",
            "bindings": {
                "hidden_test_commitment": HIDDEN_TEST_COMMITMENT,
                "insufficient_holdouts": list(INSUFFICIENT_HOLDOUTS),
                "populated_holdouts": [
                    {"count": count, "name": name, "split": split}
                    for name, split, count in POPULATED_HOLDOUTS
                ],
                "split_manifest_digest": SPLIT_MANIFEST_DIGEST,
                "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                "split_root_sha256": SPLIT_ROOT_SHA256,
            },
        },
        6: {
            "status": "satisfied",
            "summary": "The leakage audit passed with an empty violation set.",
            "bindings": {"leakage_passed": True, "violations": 0},
        },
        7: {
            "status": "satisfied_with_limitations",
            "summary": "One canonical bridge is bound. Unsupported constructs remain explicit in the gap matrix.",
            "bindings": {
                "compiler_identity": COMPILER_BINDING,
                "decompiler_identity": DECOMPILER_BINDING,
                "gap_matrix_cid": GAP_MATRIX_CID,
            },
        },
        8: {
            "status": "resolved",
            "summary": "TypedDeonticCanonicalCompiler is COMPILER-CURRENT-1. No learned compiler stage is admitted.",
            "bindings": {
                "entrypoint": "ipfs_datasets_py.logic.legal_ir.canonical_compiler.TypedDeonticCanonicalCompiler",
                "interface": "CanonicalStructuredTextCompiler@1",
                "learned_stages": [],
                "symbolic_alias": "COMPILER-CURRENT-1",
            },
        },
        9: {
            "status": "resolved",
            "summary": "SourceWithheldCanonicalDecompiler is DECOMPILER-CURRENT-1. It does not use a model.",
            "bindings": {
                "entrypoint": "ipfs_datasets_py.logic.legal_ir.canonical_decompiler.SourceWithheldCanonicalDecompiler",
                "interface": "SourceWithheldCanonicalParaphraser@1",
                "symbolic_alias": "DECOMPILER-CURRENT-1",
                "uses_model": False,
            },
        },
        10: {
            "status": "no_go",
            "summary": "Historical R1 fixture metrics are referenced by CID and are not currently qualified.",
            "bindings": {
                "hidden_test_selection": False,
                "historical_manifest_cid": R1_MANIFEST_CID,
                "historical_recipe_cid": R1_RECIPE_CID,
                "historical_report_cid": R1_REPORT_CID,
                "measured_populations": ["pilot", "repair_development"],
                "qualification": "not_currently_qualified",
            },
        },
        11: {
            "status": "not_run",
            "summary": "Shared-latent and shared-encoder/typed-head arms were declared. No weights were written.",
            "bindings": {
                "architectures_intended": ["shared_latent", "shared_encoder_typed_head"],
                "architectures_instantiated": [],
                "candidate_checkpoint": None,
            },
        },
        12: {
            "status": "no_go",
            "summary": "No learned tokenizer or vocabulary is admitted. Unknown tokens fail closed.",
            "bindings": {
                "learned_vocabulary_identity": "none",
                "policy_cid": TOKENIZER_POLICY_CID,
                "status": "no_learned_tokenizer_admitted",
            },
        },
        13: {
            "status": "resolved_unused",
            "summary": "IRLossConfiguration@1 is the fixed-point identity. No training step consumed it.",
            "bindings": {
                "identity": LOSS_CONFIGURATION,
                "proof_in_gradient_path": False,
            },
        },
        14: {
            "status": "not_run",
            "summary": "R1-R6 arms and seeds were bound. Every arm/seed lease remained ungranted.",
            "bindings": {
                "arms": ["R1", "R2", "R3", "R4", "R5", "R6"],
                "learned_seeds": [32, 33, 34],
                "pgir_110_revised_task_cid": PGIR_110_REVISED_TASK_CID,
                "pgir_110_task_cid": PGIR_110_TASK_CID,
            },
        },
        15: {
            "status": "resolved_unused",
            "summary": "Hard-negative recipes exist. Timeout, unknown, and model-only labels cannot become negatives.",
            "bindings": {
                "negative_recipe_cid": NEGATIVE_PAIR_RECIPE_CID,
                "positive_recipe_cid": POSITIVE_PAIR_RECIPE_CID,
            },
        },
        16: {
            "status": "not_run",
            "summary": "Lean-capable providers remain candidate producers. No campaign proof authority was conferred.",
            "bindings": {"attempts_admitted_to_curriculum": 0, "role": "proposal_only"},
        },
        17: {
            "status": "not_run",
            "summary": "Tactician surfaces exist in inventory. No campaign tactician lease was granted.",
            "bindings": {"attempts_admitted_to_curriculum": 0, "role": "proposal_only"},
        },
        18: {
            "status": "not_run",
            "summary": "Hammer/ATP/SMT routing exists. No campaign hammer result became proof authority.",
            "bindings": {"attempts_admitted_to_curriculum": 0, "role": "proposal_only"},
        },
        19: {
            "status": "not_run",
            "summary": "Independent kernel verification remains the only proof authority. No campaign kernel receipt exists.",
            "bindings": {"proof_root": PROOF_ROOT, "verified_campaign_proofs": 0},
        },
        20: {
            "status": "not_run",
            "summary": "Cross-entropy was not measured on admitted heldouts.",
            "bindings": metric_unavailable("token"),
        },
        21: {
            "status": "not_run",
            "summary": "Cosine and contrastive metrics were not measured on admitted heldouts.",
            "bindings": metric_unavailable("latent_retrieval"),
        },
        22: {
            "status": "not_run",
            "summary": "Retrieval metrics were not measured on admitted heldouts.",
            "bindings": metric_unavailable("latent_retrieval"),
        },
        23: {
            "status": "not_run",
            "summary": "Structural metrics were not measured on admitted heldouts.",
            "bindings": metric_unavailable("structural"),
        },
        24: {
            "status": "not_run",
            "summary": "Current-input semantic campaign metrics were not measured. Historical R1 scores stay unqualified.",
            "bindings": metric_unavailable("semantic"),
        },
        25: {
            "status": "not_run",
            "summary": "Proof replay was not measured on admitted heldouts.",
            "bindings": metric_unavailable("proof"),
        },
        26: {
            "status": "not_run",
            "summary": "Calibration and OOD metrics were not measured on admitted heldouts.",
            "bindings": metric_unavailable("calibration_ood"),
        },
        27: {
            "status": "not_run",
            "summary": "No GPU, prover, or training resource lease was granted.",
            "bindings": metric_unavailable("latency_resource"),
        },
        28: {
            "status": "resolved",
            "summary": "Multi-supervisor inventory and campaign control exist. Learning stages did not overlap because no training lease issued.",
            "bindings": {
                "descendant_execution_authorized": False,
                "resource_profile": RESOURCE_PROFILE,
                "supervisor_inventory_cid": SUPERVISOR_INVENTORY_CID,
            },
        },
        29: {
            "status": "no_go",
            "summary": "Every non-compensable M2 gate is represented. No candidate existed, so promotion is no_go.",
            "bindings": {
                "candidate": None,
                "decision": "no_go",
                "human_approval": False,
                "m2_gates": list(M2_GATES),
                "pointer_mutated": False,
            },
        },
        30: {
            "status": "denied",
            "summary": "No remote artifact was published. The local PGIR-090 package remains unqualified packaging evidence.",
            "bindings": {
                "local_release_cid": LOCAL_RELEASE_CID,
                "local_release_id": LOCAL_RELEASE_ID,
                "p4_evidence_cid": P4_EVIDENCE_CID,
                "remote_revision": None,
                "upload_authorized": False,
            },
        },
        31: {
            "status": "resolved",
            "summary": "Zero rights-admitted rows, unmaterialized corpus, incomplete holdouts, unqualified historical baseline, and no learned tokenizer block qualification.",
            "bindings": {"reason_codes": list(REASON_CODES)},
        },
        32: {
            "status": "resolved",
            "summary": "The next board is docs/architecture/proof_grounded_ir_learning/next.todo.md. It starts with rights, corpus materialization, holdouts, tokenizer, baseline requalification, a superseding freeze, then R1-R6 and requalification.",
            "bindings": {
                "next_board_path": "docs/architecture/proof_grounded_ir_learning/next.todo.md",
                "next_task_ids": [
                    "PGIR-200",
                    "PGIR-201",
                    "PGIR-202",
                    "PGIR-203",
                    "PGIR-204",
                    "PGIR-205",
                    "PGIR-206",
                    "PGIR-207",
                ],
            },
        },
    }
    records = []
    for number, slug, title in REPORT_SECTIONS:
        payload = facts[number]
        records.append(
            {
                "bindings": payload["bindings"],
                "number": number,
                "slug": slug,
                "status": payload["status"],
                "summary": payload["summary"],
                "title": title,
            }
        )
    require(len(records) == 32, "report catalog must contain 32 sections")
    return records


def build_decision(
    acceptance: Sequence[Mapping[str, Any]], sections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    promoting_gates = [item for item in acceptance if item["criterion_id"] in {
        "F04",
        "F09",
        "F10",
        "F11",
        "F12",
        "F13",
        "F14",
        "F15",
    }]
    require(all(item["status"] == "no_go" for item in promoting_gates), "unexpected promoting gate pass")
    require(all(item["number"] == index + 1 for index, item in enumerate(sections)), "section order drifted")
    return add_projection_identity(
        {
            "candidate_checkpoint": None,
            "decision": "no_go",
            "descendant_execution_authorized": False,
            "evaluator_or_model_held_authority": False,
            "hidden_tests_opened": False,
            "human_approval": False,
            "interface": "proof-grounded-ir-learning/qualification-decision/v1",
            "publication_authorized": False,
            "qualified_claim_emitted": False,
            "reason_codes": list(REASON_CODES),
            "schema": DECISION_SCHEMA,
            "task_id": TASK_ID,
            "training_admitted_rows": 0,
        },
        cid_field="decision_cid",
        sha_field="decision_sha256",
    )


def build_promotion_receipt(decision: Mapping[str, Any]) -> dict[str, Any]:
    return add_projection_identity(
        {
            "admitted_gates": [],
            "candidate_checkpoint": None,
            "decision": "no_go",
            "decision_cid": decision["decision_cid"],
            "human_approval": False,
            "interface": "proof-grounded-ir-learning/qualification-promotion/v1",
            "lease_key": "promotion-pointer",
            "m2_gates": list(M2_GATES),
            "pointer_mutated": False,
            "reason": "no_candidate_checkpoint;freeze_descendant_execution_unauthorized",
            "required_gates": list(M2_GATES),
            "schema": PROMOTION_SCHEMA,
            "self_promotion": False,
            "task_id": TASK_ID,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )


def build_publication_receipt(decision: Mapping[str, Any]) -> dict[str, Any]:
    return add_projection_identity(
        {
            "append_only": True,
            "decision": "denied",
            "decision_cid": decision["decision_cid"],
            "human_approval": False,
            "interface": "proof-grounded-ir-learning/qualification-publication/v1",
            "lease_key": "hf-publication:Publicus/proof-grounded-ir-learning",
            "local_package_cid": LOCAL_RELEASE_CID,
            "local_package_qualified": False,
            "reason": "qualification_gates_failed;require_qualification;default_publication_admission_deny",
            "remote_revision": None,
            "schema": PUBLICATION_SCHEMA,
            "task_id": TASK_ID,
            "trust_remote_code": False,
            "upload_attempted": False,
        },
        cid_field="receipt_cid",
        sha_field="receipt_sha256",
    )


def render_final_report(
    *,
    acceptance: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    promotion: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> str:
    lines = [
        "# ProofGroundedIRLearningFabric final evidence report",
        "",
        "This report is `RESULT(PGIR-111)`. Every metric cites a content-addressed",
        "receipt or an explicit `not_run` / `no_go` status. Missing evidence is never",
        "inferred as a pass. The exact qualified-claim text is withheld because the",
        "admitted gates did not pass.",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Decision CID: `{decision['decision_cid']}`",
        f"- Freeze root: `{FREEZE_ROOT_CID}`",
        f"- Freeze result: `{FREEZE_RESULT_CID}`",
        f"- Promotion receipt: `{promotion['receipt_cid']}`",
        f"- Publication receipt: `{publication['receipt_cid']}`",
        f"- Qualified claim emitted: `{str(decision['qualified_claim_emitted']).lower()}`",
        "",
        "## Final acceptance criteria",
        "",
    ]
    for item in acceptance:
        lines.extend(
            [
                f"### {item['criterion_id']} {item['text']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Evidence: {item['evidence']}",
                "",
            ]
        )
    for item in sections:
        lines.extend(
            [
                f"## {item['number']}. {item['title']}",
                "",
                item["summary"],
                "",
                f"- Section status: `{item['status']}`",
            ]
        )
        for key in sorted(item["bindings"]):
            value = item["bindings"][key]
            rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
            lines.append(f"- `{key}`: `{rendered}`")
        lines.append("")
    lines.extend(
        [
            "## Authorized closing claim",
            "",
            "The qualification decision is `no_go`. The exact required qualified-claim",
            "text is withheld because the admitted gates did not pass. No candidate",
            "checkpoint exists. Promotion is `no_go`. Remote publication is `denied`.",
            "This is an integrity success and an execution denial, not a fabricated",
            "training result.",
            "",
            "Reason codes:",
            "",
        ]
    )
    for code in REASON_CODES:
        lines.append(f"- `{code}`")
    lines.extend(
        [
            "",
            "Never claim universal legal-semantic understanding.",
            "",
        ]
    )
    text = "\n".join(lines)
    require(QUALIFIED_CLAIM not in text, "qualified claim leaked into no-go report")
    require(
        text.count("\n## ") == 34,
        "final report must contain acceptance, 32 numbered sections, and closing claim",
    )
    require(sum(f"\n## {number}. " in text for number in range(1, 33)) == 32, "numbered sections missing")
    return text


def next_task_block(
    *,
    task_id: str,
    title: str,
    track: str,
    parent_goal: str,
    subgoal: str,
    owning_repository: str,
    owned_paths: str,
    objective: str,
    depends_on: str,
    resource_profile: str,
    expected_inputs: str,
    expected_outputs: str,
    allowed_effects: str,
    prohibited_effects: str,
    acceptance: str,
    evidence: str,
    lease_key: str,
    outputs: str,
    validation: str,
    bundle: str,
    lane: str,
    conflict: str,
) -> str:
    return "\n".join(
        [
            f"## {task_id} {title}",
            "",
            "- Status: todo",
            "- Completion: supervisor-evidence",
            "- Is schedulable: true",
            "- Priority: P0",
            f"- Track: {track}",
            f"- Parent goal: {parent_goal}",
            f"- Subgoal: {subgoal}",
            f"- Owning repository: {owning_repository}",
            f"- Owned paths: {owned_paths}",
            "- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`",
            "- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists",
            "- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists",
            "- Compiler identity: `RESULT(PGIR-021)`",
            "- Decompiler identity: `RESULT(PGIR-022)`",
            "- Model checkpoint identity: none until a later admitted candidate exists",
            f"- Objective: {objective}",
            f"- Depends on: {depends_on}",
            f"- Resource profile: {resource_profile}",
            f"- Expected inputs: {expected_inputs}",
            f"- Expected outputs: {expected_outputs}",
            f"- Allowed effects: {allowed_effects}",
            f"- Prohibited effects: {prohibited_effects}",
            f"- Acceptance criteria: {acceptance}",
            f"- Required proof or evaluation evidence: {evidence}",
            f"- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `{lease_key}`",
            "- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence",
            f"- Result identity: `RESULT({task_id})`",
            f"- Outputs: {outputs}",
            f"- Validation: {validation}",
            f"- Bundle: {bundle}",
            f"- Parallel lane: {lane}",
            f"- Predicted files: {outputs}",
            f"- Conflict policy: {conflict}",
            "",
        ]
    )


def render_next_board(*, decision: Mapping[str, Any]) -> str:
    body = [
        "# Proof-Grounded IR Learning Fabric Next Improvement Board",
        "",
        "This board is issued by `RESULT(PGIR-111)`. It is the exact next",
        "training and data-improvement board after the first campaign's",
        f"`no_go` decision `{decision['decision_cid']}`.",
        "",
        "It does not replace the protected original board. Workers must keep",
        "`docs/architecture/proof_grounded_ir_learning.todo.md`,",
        "`docs/architecture/proof_grounded_ir_learning.objectives.md`, and",
        "`data/agent_supervisor/proof_grounded_ir_learning/justice_dao_pinset.yaml`",
        "unchanged unless a later admitted plan revision says otherwise.",
        "",
        "The freeze chain remains binding. No learned-model, pair-mining,",
        "proof-curriculum, training, promotion, or publication task may be",
        "leased before a superseding `IRCampaignInputRoot@1` admits rights,",
        "materialized corpus rows, complete required holdouts, and a current",
        "tokenizer or an explicit deterministic-only restriction.",
        "",
        "## Frozen identities inherited from RESULT(PGIR-111)",
        "",
        f"- `SRCSET-1` remains (`{DATASETS_AUTHORITY}`, `{ACCELERATE_AUTHORITY}`).",
        f"- `JDAO-PINSET-1` remains SHA-256 `{PINSET_SHA256}` and still admits zero training repositories.",
        f"- `RESULT(PGIR-014)` `{FREEZE_RESULT_CID}` remains the current no-go freeze.",
        f"- Hidden-test commitment `{HIDDEN_TEST_COMMITMENT}` stays sealed.",
        "- `MODEL-LEGACY-1` remains artifact-only and never promotion authority.",
        "",
        "## Why the previous campaign cannot be promoted",
        "",
    ]
    for code in REASON_CODES:
        body.append(f"- `{code}`")
    body.extend(
        [
            "",
            "No worker may treat historical `RESULT(PGIR-023)` fixture scores, the",
            "PGIR-090 local package, or `MODEL-LEGACY-1` as current qualification.",
            "",
        ]
    )
    body.append(
        next_task_block(
            task_id="PGIR-200",
            title="Admit or permanently quarantine JusticeDAO source rights",
            track="source-curation",
            parent_goal="PGIR-G020",
            subgoal="rights-admission",
            owning_repository="ipfs_datasets_py",
            owned_paths="ipfs_datasets_py/data/ir_learning/corpora/**",
            objective="Re-evaluate source and transformation rights for the 7,173 quarantined source rows and emit either a rights-admitted row set or a permanent quarantine with explicit residual gaps.",
            depends_on="PGIR-111",
            resource_profile="RP-IO-PINNED",
            expected_inputs="JDAO-PINSET-1, RESULT(PGIR-004), RESULT(PGIR-011), RESULT(PGIR-111)",
            expected_outputs="updated rights/quarantine manifests and an admitted-row count that is either positive or explicitly permanently zero",
            allowed_effects="owned corpus rights/quarantine artifacts only",
            prohibited_effects="silent un-quarantine, trust-remote-code, training, publication",
            acceptance="every quarantined row has a fresh rights decision; training_admitted_rows is either >0 with cited licenses or remains 0 with a permanent no-go reason",
            evidence="rights receipts, license/cutoff/jurisdiction bindings, and a replayable admitted-row count",
            lease_key="corpus-rights",
            outputs="ipfs_datasets_py/data/ir_learning/corpora/",
            validation="python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py",
            bundle="pgir/next/rights",
            lane="rights-admission",
            conflict="exclusive corpus-rights writer",
        )
    )
    body.append(
        next_task_block(
            task_id="PGIR-201",
            title="Materialize the sealed corpus after rights admission",
            track="source-curation",
            parent_goal="PGIR-G020",
            subgoal="corpus-materialization",
            owning_repository="ipfs_datasets_py",
            owned_paths="ipfs_datasets_py/data/ir_learning/corpora/**",
            objective="Materialize only rights-admitted source rows into the sealed corpus root. Keep source and derived counts distinct and leave derivatives linked to their source CID groups.",
            depends_on="PGIR-200",
            resource_profile="RP-IO-PINNED",
            expected_inputs="RESULT(PGIR-200) rights decision and RESULT(PGIR-011) manifests",
            expected_outputs="a corpus root with materialized=true only when admitted rows exist; otherwise a documented still-unmaterialized no-go",
            allowed_effects="owned corpus materialization artifacts",
            prohibited_effects="inflating source counts with derivatives, hidden-test access, training",
            acceptance="materialized flag matches the admitted-row set; source_count remains 7173 or a cited superseding count; derived_count stays separate",
            evidence="corpus_root, corpus_manifest, and load receipts",
            lease_key="corpus-materialize",
            outputs="ipfs_datasets_py/data/ir_learning/corpora/",
            validation="python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py",
            bundle="pgir/next/corpus",
            lane="corpus-materialize",
            conflict="exclusive corpus-root writer",
        )
    )
    body.append(
        next_task_block(
            task_id="PGIR-202",
            title="Populate the thirteen insufficient holdouts",
            track="source-curation",
            parent_goal="PGIR-G020",
            subgoal="holdout-completion",
            owning_repository="ipfs_datasets_py",
            owned_paths="ipfs_datasets_py/data/ir_learning/splits/**",
            objective="Populate compiler, cross_reference, domain, exception, length, lineage, notation, premise, proof_library, publication, rare_operator, time, and type holdouts, or document why a named holdout remains impossible.",
            depends_on="PGIR-201",
            resource_profile="RP-CPU-M",
            expected_inputs="RESULT(PGIR-012), RESULT(PGIR-201), hidden-test commitment",
            expected_outputs="a superseding split root whose required holdouts are populated or explicitly permanently insufficient",
            allowed_effects="owned split/holdout artifacts",
            prohibited_effects="opening hidden tests, random-row splitting as the principal method, leakage-group splits",
            acceptance="leakage audit still passes; hidden-test commitment unchanged; every previously insufficient holdout is populated or has a permanent no-go reason",
            evidence="holdout_report, leakage_report, and split_root identities",
            lease_key="split-holdouts",
            outputs="ipfs_datasets_py/data/ir_learning/splits/",
            validation="python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py",
            bundle="pgir/next/holdouts",
            lane="holdout-completion",
            conflict="exclusive split-root writer",
        )
    )
    body.append(
        next_task_block(
            task_id="PGIR-203",
            title="Admit a learned tokenizer or restrict the campaign to deterministic-only",
            track="model",
            parent_goal="PGIR-G050",
            subgoal="tokenizer-admission",
            owning_repository="ipfs_accelerate_py",
            owned_paths="data/agent_supervisor/proof_grounded_ir_learning/freeze/**",
            objective="Either freeze a compatible learned tokenizer/vocabulary CID or issue an explicit deterministic-only campaign restriction that keeps R2-R6 ineligible.",
            depends_on="PGIR-202",
            resource_profile="RP-CPU-M",
            expected_inputs="RESULT(PGIR-030) architecture surfaces and the current tokenizer freeze policy",
            expected_outputs="a superseding IRTokenizerFreezePolicy@1 that is either admitted or permanently deterministic-only",
            allowed_effects="owned tokenizer-policy artifacts under a new freeze location",
            prohibited_effects="mutating the current freeze in place, promoting MODEL-LEGACY-1, unfrozen vocabulary mutation",
            acceptance="unknown tokens still fail closed; learned training remains unauthorized until a tokenizer is admitted",
            evidence="tokenizer policy CID and golden token-class receipts",
            lease_key="tokenizer",
            outputs="data/agent_supervisor/proof_grounded_ir_learning/freeze/",
            validation="python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_modal_autoencoder.py",
            bundle="pgir/next/tokenizer",
            lane="tokenizer-admission",
            conflict="serial tokenizer freeze",
        )
    )
    body.append(
        next_task_block(
            task_id="PGIR-204",
            title="Requalify or replace the historical R1 semantic baseline",
            track="evaluation",
            parent_goal="PGIR-G040",
            subgoal="current-input-baseline",
            owning_repository="ipfs_datasets_py",
            owned_paths="ipfs_datasets_py/data/ir_learning/evaluations/deterministic/**",
            objective="Re-run or replace RESULT(PGIR-023) on current-input admitted rows and declared non-hidden partitions so the deterministic baseline is either currently qualified or explicitly retired.",
            depends_on="PGIR-202",
            resource_profile="RP-PROVER",
            expected_inputs="RESULT(PGIR-021), RESULT(PGIR-022), RESULT(PGIR-202)",
            expected_outputs="a current-input R1 report with E1 metrics, denominators, and tool versions, or a retirement receipt",
            allowed_effects="owned deterministic evaluation artifacts",
            prohibited_effects="hidden-test selection, missing metric as zero, treating historical fixture scores as current",
            acceptance="either a current-input qualified R1 CID exists or the historical baseline is retired by CID",
            evidence="recipe, identities, strata, tool versions, and independent replay",
            lease_key="evaluation:deterministic",
            outputs="ipfs_datasets_py/data/ir_learning/evaluations/deterministic/",
            validation="python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py",
            bundle="pgir/next/r1",
            lane="deterministic-requalify",
            conflict="one report reducer lease",
        )
    )
    body.append(
        next_task_block(
            task_id="PGIR-205",
            title="Issue a superseding campaign input freeze",
            track="qualification",
            parent_goal="PGIR-G030",
            subgoal="superseding-freeze",
            owning_repository="ipfs_accelerate_py",
            owned_paths="data/agent_supervisor/proof_grounded_ir_learning/freeze/**",
            objective="Bind the new rights, corpus, split, tokenizer, and baseline identities into a superseding IRCampaignInputRoot@1 whose previous_root_cid is the current no-go freeze.",
            depends_on="PGIR-200, PGIR-201, PGIR-202, PGIR-203, PGIR-204",
            resource_profile="RP-CPU-S",
            expected_inputs="RESULT(PGIR-014) and all PGIR-200..204 outputs",
            expected_outputs="a new freeze root, descendant task revisions, and a go or documented no-go",
            allowed_effects="a separately located freeze; never overwrite the current root",
            prohibited_effects="in-place freeze mutation, hidden-test access, promotion",
            acceptance="previous_root_cid equals the current freeze; training tasks are eligible only when rights, corpus, holdouts, and tokenizer gates pass",
            evidence="independent freeze verifier and plan-admission receipt",
            lease_key="campaign-freeze-root",
            outputs="data/agent_supervisor/proof_grounded_ir_learning/freeze/",
            validation="python -m pytest -q test/api/test_agent_supervisor_formal_plan_validator.py test/api/test_agent_supervisor_task_identity.py",
            bundle="pgir/next/freeze",
            lane="freeze-root",
            conflict="global serial freeze barrier",
        )
    )
    body.append(
        next_task_block(
            task_id="PGIR-206",
            title="Re-run R1-R6 on the superseding freeze",
            track="experiments",
            parent_goal="PGIR-G110",
            subgoal="controlled-comparisons-v2",
            owning_repository="ipfs_accelerate_py",
            owned_paths="data/agent_supervisor/proof_grounded_ir_learning/experiments/**",
            objective="Execute deterministic, CE-only, CE+cosine, contrastive, full multi-task, and proof-grounded arms on identical frozen heldouts only after the superseding freeze authorizes descendant execution.",
            depends_on="PGIR-205",
            resource_profile="RP-MIXED",
            expected_inputs="superseding freeze, architectures, losses, pairs, proof loop, evaluator, security",
            expected_outputs="arm checkpoints, actual metrics/CIs/costs/failures, and a comparison report",
            allowed_effects="isolated training/proof/evaluation artifacts under a new campaign location",
            prohibited_effects="hidden-test tuning, best-test selection, fabricated target attainment, shared checkpoint writes",
            acceptance="same heldouts/seeds; every R metric reported; bounded exhaustion typed; no invented scores",
            evidence="training/checkpoint/proof/evaluation/resource receipts and paired statistical report",
            lease_key="campaign:reducer",
            outputs="data/agent_supervisor/proof_grounded_ir_learning/experiments/",
            validation="python -m pytest -q test/api/test_agent_supervisor_proof_workflow_e2e.py test/api/test_agent_supervisor_scheduler.py",
            bundle="pgir/next/r1-r6",
            lane="experiment-orchestrator",
            conflict="promotion/test reducer independent of trainers",
        )
    )
    body.append(
        next_task_block(
            task_id="PGIR-207",
            title="Re-qualify, publish or reject, and issue the following board",
            track="qualification",
            parent_goal="PGIR-G110",
            subgoal="final-decision-report-v2",
            owning_repository="ipfs_accelerate_py",
            owned_paths="data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md",
            objective="Apply the same 16 final criteria and 32 report sections to RESULT(PGIR-206). Emit promote, reject, no-go, or resource-exhausted. Publish only if independently authorized. Issue the next board.",
            depends_on="PGIR-072, PGIR-090, PGIR-100, PGIR-206",
            resource_profile="RP-CPU-M",
            expected_inputs="every accepted successor result, experiment comparisons, current promotion/publication authorities",
            expected_outputs="successor final report, decision, publication receipt, and next board",
            allowed_effects="qualification artifacts; promotion/publication only under current independent authority",
            prohibited_effects="universal understanding claim, missing-failure suppression, model self-promotion, unauthorized upload",
            acceptance="all 16 criteria and 32 sections resolved with evidence or explicit no-go; exact qualified-claim text used only if gates pass",
            evidence="manifest/evaluation/proof/promotion/publication verifiers and complete result graph",
            lease_key="final-decision",
            outputs="data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md",
            validation="python -m pytest -q test/api/test_agent_supervisor_goal_completion.py test/api/test_agent_supervisor_proof_goal_completion.py",
            bundle="pgir/next/qualification",
            lane="final-qualifier",
            conflict="one independent qualification/promotion authority; evaluator/model cannot hold it",
        )
    )
    text = "\n".join(body)
    require("PGIR-200" in text and "PGIR-207" in text, "next board missing required tasks")
    require(QUALIFIED_CLAIM not in text, "qualified claim leaked into next board")
    return text


def build_recipe(
    *,
    acceptance: Mapping[str, Any],
    sections: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    return add_projection_identity(
        {
            "acceptance_cid": acceptance["catalog_cid"],
            "campaign_input_root_cid": FREEZE_ROOT_CID,
            "compiler_identity": COMPILER_BINDING,
            "data_split_identity": SPLIT_BINDING,
            "decompiler_identity": DECOMPILER_BINDING,
            "decision_cid": decision["decision_cid"],
            "depends_on": list(DEPENDS_ON),
            "hidden_test_commitment": HIDDEN_TEST_COMMITMENT,
            "hidden_test_selection": False,
            "interface": "proof-grounded-ir-learning/qualification-recipe/v1",
            "lease_policy": LEASE_POLICY,
            "metric_families": list(METRIC_FAMILIES),
            "missing_metric_as_zero": False,
            "reason_codes": list(REASON_CODES),
            "resource_profile": RESOURCE_PROFILE,
            "schema": RECIPE_SCHEMA,
            "sections_cid": sections["catalog_cid"],
            "source_dataset_revisions": CORPUS_BINDING,
            "task_id": TASK_ID,
            "tokenizer_policy_status": "no_learned_tokenizer_admitted",
        },
        cid_field="recipe_cid",
        sha_field="recipe_sha256",
    )


def build_all() -> dict[str, Any]:
    freeze = load_freeze()
    evidence = load_supporting_evidence()
    acceptance_rows = evaluate_acceptance()
    section_rows = section_payloads()
    acceptance = add_projection_identity(
        {
            "criteria": acceptance_rows,
            "criterion_count": 16,
            "interface": "proof-grounded-ir-learning/final-acceptance/v1",
            "schema": ACCEPTANCE_SCHEMA,
            "task_id": TASK_ID,
        },
        cid_field="catalog_cid",
        sha_field="catalog_sha256",
    )
    sections = add_projection_identity(
        {
            "interface": "proof-grounded-ir-learning/final-report-sections/v1",
            "schema": SECTIONS_SCHEMA,
            "section_count": 32,
            "sections": section_rows,
            "task_id": TASK_ID,
        },
        cid_field="catalog_cid",
        sha_field="catalog_sha256",
    )
    decision = build_decision(acceptance_rows, section_rows)
    promotion = build_promotion_receipt(decision)
    publication = build_publication_receipt(decision)
    recipe = build_recipe(acceptance=acceptance, sections=sections, decision=decision)
    report_text = render_final_report(
        acceptance=acceptance_rows,
        sections=section_rows,
        decision=decision,
        promotion=promotion,
        publication=publication,
    )
    next_board_text = render_next_board(decision=decision)
    files = {
        "acceptance": write_json(QUALIFICATION_DIR / "acceptance.json", acceptance),
        "decision": write_json(QUALIFICATION_DIR / "decision.json", decision),
        "final_report": write_text(DOCS_DIR / "final_report.md", report_text),
        "next_board": write_text(DOCS_DIR / "next.todo.md", next_board_text),
        "promotion": write_json(QUALIFICATION_DIR / "promotion_receipt.json", promotion),
        "publication": write_json(QUALIFICATION_DIR / "publication_receipt.json", publication),
        "recipe": write_json(QUALIFICATION_DIR / "recipe.json", recipe),
        "sections": write_json(QUALIFICATION_DIR / "report_sections.json", sections),
    }
    result = add_projection_identity(
        {
            "campaign_input_root_cid": FREEZE_ROOT_CID,
            "completion_authoritative": False,
            "decision": "no_go",
            "decision_cid": decision["decision_cid"],
            "dependency_task_cids": dict(DEPENDENCY_TASK_CIDS),
            "disposition": "qualified_no_go",
            "freeze_decision": freeze["result"]["decision"],
            "freeze_result_cid": FREEZE_RESULT_CID,
            "hidden_tests_opened": False,
            "objective_id": OBJECTIVE_ID,
            "objective_revision": OBJECTIVE_REVISION,
            "parent_goal": PARENT_GOAL,
            "publication_authorized": False,
            "qualified_claim_emitted": False,
            "reason_codes": list(REASON_CODES),
            "repository_id": REPOSITORY_ID,
            "result_identity": "RESULT(PGIR-111)",
            "rollback": "retain this immutable qualification and create a separately admitted superseding root",
            "schema": RESULT_SCHEMA,
            "source_tree_id": SOURCE_TREE_ID,
            "subgoal": SUBGOAL,
            "task_cid": TASK_CID,
            "task_id": TASK_ID,
            "training_admitted_rows": evidence["corpus"]["training_admitted_rows"],
        },
        cid_field="result_cid",
        sha_field="result_sha256",
    )
    files["result"] = write_json(QUALIFICATION_DIR / "result.json", result)
    manifest = add_projection_identity(
        {
            "bundle": "pgir/qualification/final",
            "decision": "no_go",
            "files": files,
            "interface": "proof-grounded-ir-learning/qualification-manifest/v1",
            "recipe_cid": recipe["recipe_cid"],
            "result_cid": result["result_cid"],
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
        },
        cid_field="manifest_cid",
        sha_field="manifest_sha256",
    )
    files["manifest"] = write_json(QUALIFICATION_DIR / "manifest.json", manifest)
    return {
        "acceptance": acceptance,
        "decision": decision,
        "files": files,
        "manifest": manifest,
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Create missing artifacts only. Refuses to replace different bytes.",
    )
    args = parser.parse_args()
    del args
    built = build_all()
    print(
        json.dumps(
            {
                "decision": built["decision"]["decision"],
                "decision_cid": built["decision"]["decision_cid"],
                "manifest_cid": built["manifest"]["manifest_cid"],
                "result_cid": built["result"]["result_cid"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
