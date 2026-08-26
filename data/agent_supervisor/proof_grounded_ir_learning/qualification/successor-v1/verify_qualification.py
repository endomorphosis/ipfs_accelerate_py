#!/usr/bin/env python3
"""Independently replay the immutable PGIR-207 terminal qualification.

This verifier is intentionally standard-library-only.  It verifies the exact
PGIR-206 typed-not-run input, every successor qualification identity and byte
seal, the closed 16-criterion/32-section decision, the complete result graph,
and the blocked terminal board.  ``--fresh-recursive`` additionally proves
that the committed package replays from a new ``git clone --no-local`` with an
independently initialized local ``ipfs_datasets_py`` submodule.

A successful replay confirms a truthful ``no_go``.  It never grants training,
promotion, publication, scheduling, or supervisor-completion authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[4]
DATA_ROOT = PACKAGE_DIR.parents[1]
FREEZE_DIR = DATA_ROOT / "freeze" / "successor-v1"
EXPERIMENT_DIR = DATA_ROOT / "experiments" / "successor-v1"
DOCS_DIR = (
    REPOSITORY_ROOT
    / "docs"
    / "architecture"
    / "proof_grounded_ir_learning"
    / "successor-v1"
)
NESTED_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"

TASK_ID = "PGIR-207"
TASK_TITLE = "Re-qualify, publish or reject, and issue the following board"
TASK_KEY = "task/v1/efe4a7c323386342928737eaf8f18cad23f597f16cc27cd4acf32939066a3279"
TASK_CID = "baguqeera57skpqzdhbrufeuhg7vpr4mmvur7lf7rntbhzvfm6mutsbtkgj4q"
RESULT_IDENTITY = "RESULT(PGIR-207)"
PGIR_206_RESULT_CID = "baguqeera7pwtswk2472r2yi2ijs5veamgx4hmy2nice2bnd3fvf72o3kgswq"
PGIR_206_MANIFEST_CID = "baguqeerao2izb4i6bv7ffg3flharhjlph35gpwl4rbekzrrwm3n25mibqxsq"
PGIR_206_IMPLEMENTATION = "0373c2968b9b46b43ee9f564051182df137c08e4"
PGIR_206_IMPLEMENTATION_PARENT = "c03180fbbd64b225a121fcc7be046bd0beab8918"
PGIR_205_RESULT_CID = "baguqeerarcuvejfqyjsfbqdtel67wu2lhjec3oh27enlqbxsvgyfbytsgd2q"
CAMPAIGN_INPUT_ROOT_CID = "baguqeerajvu2dvjjxe4l6dibujguiedwhdahseziqpz7xhhp724jlxhlxz4q"
PGIR_211_ACCEPTANCE_CID = "baguqeeram562re6snweb5nuinwprb4ehccvkin7kpylihktizlirsss7pllq"
NESTED_CURRENT = "2a06dfe8546cdde78ff6d101a94708be0e6bf6e6"
NESTED_CURRENT_TREE = "7169c2a67929044a02350bc26d0a51c853a4981b"

PGIR_206_RECEIPTS = {
    "admission": "baguqeeraqyqjwwkt3jmr2b2hnwpwafhwndre4fkd7gcck5rvwzcyfh22shia",
    "checkpoint": "baguqeeraa2bj4uvhwqlcqyo6iastvodab6ulvnycmu2emvyohhwdrm3dcmka",
    "evaluation": "baguqeerashtuscaphw2hpicra6klxt6a2dae64gqbuanvpmdzqzj5ljotrda",
    "proof": "baguqeeraa55wxtbh2s63j22qgytbnsfjpatnstg4vun4nqg5qlvu3fgcx36a",
    "reducer_cas": "baguqeera5x32cyuywisqviyfi7nbzfwnjdmtf5k4zzbbdkdywzikzswswrna",
    "resource": "baguqeerani6ro4d7naxn4mrb5rgzuiw77zq3hckyttbhgihqvtcj6oquziiq",
    "training": "baguqeeral5izctrs4pkjsiwmodfufnvgcsvmvz3fucarh5atlduzta7aaiaa",
}
HISTORICAL_TASK_CIDS = {
    "PGIR-072": "baguqeeraarlt745ftpwax4tdovajsxrgp5r72fkudlbs5kmg3q2cbqow6mpq",
    "PGIR-090": "baguqeeragt5xhqov2e5vna6pp2zmghhrqhq2v4qzl36a3aucytrcom2erkta",
    "PGIR-100": "baguqeerainaqx5w72m2epc7wsolk3elqihpcganr6d3lrsuzseldzkqngija",
    "PGIR-111": "baguqeerad2idzhwzfqlyxjh7u34bczkyutsvq7oxcmuaglupr436gqfo6kga",
}
HISTORICAL_EVIDENCE_CIDS = {
    "PGIR-072": "baguqeerazlsqaghk6m2c2ru3qbrsfprv7b5nbc5oa5lvfxz6d4b6qqgxpvra",
    "PGIR-090": "bafkreigwdei25h3eg2k2l6gp6ak5tbkbcfabi6vsoqzrzv6k6mpm73lhge",
    "PGIR-100": HISTORICAL_TASK_CIDS["PGIR-100"],
    "PGIR-111": "baguqeeraejs56hwzs3bqtgzoayrc2fxwgfnhcsxjthi4dh7gh64wptlkfhwa",
}

REASON_CODES = (
    "no_rights_admitted_training_rows",
    "corpus_not_materialized",
    "required_holdouts_insufficient",
    "tokenizer_not_admitted",
    "historical_semantic_baseline_not_currently_qualified",
    "integrated_evidence_does_not_authorize_execution",
    "portability_no_go",
    "pgir_206_execution_not_authorized",
    "pgir_206_typed_not_run",
    "no_candidate_checkpoint",
    "publication_not_authorized",
)

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

CRITERION_STATUS = {
    "F01": "resolved_with_evidence",
    "F02": "satisfied",
    "F03": "satisfied_with_limitations",
    "F04": "no_go",
    "F05": "satisfied",
    "F06": "satisfied_execution_denied",
    "F07": "satisfied",
    "F08": "satisfied",
    "F09": "no_go",
    "F10": "no_go",
    "F11": "no_go",
    "F12": "no_go",
    "F13": "no_go",
    "F14": "no_go",
    "F15": "no_go",
    "F16": "satisfied",
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

SECTION_STATUS = (
    "resolved", "resolved", "resolved", "resolved", "resolved_with_no_go",
    "satisfied", "satisfied_with_limitations", "resolved", "resolved", "no_go",
    "not_run", "no_go", "resolved_unused", "not_run", "resolved_unused",
    "not_run", "not_run", "not_run", "not_run", "not_run", "not_run",
    "not_run", "not_run", "not_run", "not_run", "not_run", "not_run",
    "resolved", "no_go", "denied", "resolved", "resolved",
)

GRAPH_DEPENDENCIES = {
    "PGIR-072": (),
    "PGIR-090": (),
    "PGIR-100": (),
    "PGIR-111": (),
    "PGIR-200": ("PGIR-111",),
    "PGIR-201": ("PGIR-200",),
    "PGIR-202": ("PGIR-201",),
    "PGIR-203": ("PGIR-202",),
    "PGIR-204": ("PGIR-202",),
    "PGIR-208": ("PGIR-200", "PGIR-201", "PGIR-202"),
    "PGIR-209": ("PGIR-202", "PGIR-204"),
    "PGIR-210": ("PGIR-204", "PGIR-208"),
    "PGIR-211": ("PGIR-209", "PGIR-210"),
    "PGIR-205": (
        "PGIR-200", "PGIR-201", "PGIR-202", "PGIR-203", "PGIR-204",
        "PGIR-208", "PGIR-209", "PGIR-210", "PGIR-211",
    ),
    "PGIR-206": ("PGIR-205",),
    "PGIR-207": ("PGIR-072", "PGIR-090", "PGIR-100", "PGIR-206"),
}
DEPENDENCY_FIRST_ORDER = (
    "PGIR-072", "PGIR-090", "PGIR-100", "PGIR-111", "PGIR-200",
    "PGIR-201", "PGIR-202", "PGIR-203", "PGIR-204", "PGIR-208",
    "PGIR-209", "PGIR-210", "PGIR-211", "PGIR-205", "PGIR-206",
    "PGIR-207",
)
EXPECTED_EDGES = tuple(
    (task_id, dependency)
    for task_id in DEPENDENCY_FIRST_ORDER
    for dependency in GRAPH_DEPENDENCIES[task_id]
)
GRAPH_DISPOSITIONS = {
    "PGIR-072": "historical_completed",
    "PGIR-090": "historical_completed",
    "PGIR-100": "historical_completed",
    "PGIR-111": "historical_qualified_no_go",
    "PGIR-200": "permanent_no_go",
    "PGIR-201": "not_materialized",
    "PGIR-202": "permanent_no_go",
    "PGIR-203": "permanently_deterministic_only",
    "PGIR-204": "retired",
    "PGIR-208": "qualified_no_go",
    "PGIR-209": "qualified_no_go",
    "PGIR-210": "qualified_no_go",
    "PGIR-211": "qualified_no_go",
    "PGIR-205": "frozen_no_go",
    "PGIR-206": "typed_not_run",
    "PGIR-207": "qualified_no_go",
}

SEALED_PATHS = (
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/README.md",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/acceptance.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/build_qualification.py",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/decision.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/evaluation_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/proof_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/promotion_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/publication_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/recipe.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/report_sections.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/result_graph.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verify_qualification.py",
    "docs/architecture/proof_grounded_ir_learning/successor-v1/final_report.md",
    "docs/architecture/proof_grounded_ir_learning/successor-v1/next.todo.md",
)
FINAL_PATHS = (
    *SEALED_PATHS,
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/manifest.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verification_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/result.json",
)

GIT_ENVIRONMENT = {
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
}
PYTHON_ENVIRONMENT = {
    **GIT_ENVIRONMENT,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}


class QualificationVerificationError(ValueError):
    """Raised when any terminal evidence or fail-closed gate drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationVerificationError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise QualificationVerificationError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise QualificationVerificationError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rendered_bytes(value: Any) -> bytes:
    validate_value(value)
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def dag_json_cid(value: Any) -> str:
    digest = hashlib.sha256(canonical_bytes(value)).digest()
    prefix = b"\x01\xa9\x02\x12\x20"
    return "b" + base64.b32encode(prefix + digest).decode("ascii").rstrip("=").lower()


def raw_cid(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    prefix = b"\x01\x55\x12\x20"
    return "b" + base64.b32encode(prefix + digest).decode("ascii").rstrip("=").lower()


def identity(data: bytes) -> dict[str, Any]:
    return {
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON artifact: {path}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        data = path.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                QualificationVerificationError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                QualificationVerificationError(f"constant {raw!r} in {path}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationVerificationError(f"invalid JSON in {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    validate_value(value, str(path))
    require(data == rendered_bytes(value), f"non-canonical rendered JSON bytes: {path}")
    return value


def verify_projection_identity(payload: Mapping[str, Any], name: str) -> None:
    sha_field = f"{name}_sha256"
    cid_field = f"{name}_cid"
    require(sha_field in payload and cid_field in payload, f"missing {name} projection identity")
    projection = {
        key: value for key, value in payload.items() if key not in {sha_field, cid_field}
    }
    expected_sha = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    require(payload[sha_field] == expected_sha, f"{sha_field} drifted")
    require(payload[cid_field] == dag_json_cid(projection), f"{cid_field} drifted")


def require_task_binding(payload: Mapping[str, Any], label: str) -> None:
    binding = payload.get("task_binding")
    require(isinstance(binding, dict), f"{label} task binding missing")
    expected = {
        "current_task_key": TASK_KEY,
        "current_task_cid": TASK_CID,
        "task_id": TASK_ID,
        "title": TASK_TITLE,
        "objective_id": "PGIR-G110",
        "parent_goal": "PGIR-G110",
        "subgoal": "final-decision-report-v2",
    }
    require(binding == expected, f"{label} task binding drifted")


def deep_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        result: list[Any] = []
        for child in value.values():
            result.extend(deep_values(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(deep_values(child))
        return result
    return [value]


def verify_pgir_206() -> dict[str, dict[str, Any]]:
    result = strict_json(EXPERIMENT_DIR / "result.json")
    manifest = strict_json(EXPERIMENT_DIR / "manifest.json")
    evaluation = strict_json(EXPERIMENT_DIR / "receipts" / "evaluation.json")
    proof = strict_json(EXPERIMENT_DIR / "receipts" / "proof.json")
    verify_projection_identity(result, "result")
    verify_projection_identity(manifest, "manifest")
    verify_projection_identity(evaluation, "receipt")
    verify_projection_identity(proof, "receipt")
    require(result["result_cid"] == PGIR_206_RESULT_CID, "PGIR-206 result CID drifted")
    require(manifest["manifest_cid"] == PGIR_206_MANIFEST_CID, "PGIR-206 manifest CID drifted")
    require(
        manifest.get("artifact_count") == 16
        and manifest.get("json_artifact_count") == 13
        and isinstance(manifest.get("artifacts"), dict)
        and len(manifest["artifacts"]) == 16,
        "PGIR-206 manifest population drifted",
    )
    for relative, record in manifest["artifacts"].items():
        require(isinstance(record, dict), f"PGIR-206 manifest record invalid: {relative}")
        path = EXPERIMENT_DIR / relative
        require(path.is_file() and not path.is_symlink(), f"PGIR-206 artifact missing: {relative}")
        observed = identity(path.read_bytes())
        require(
            all(record.get(key) == value for key, value in observed.items()),
            f"PGIR-206 artifact bytes drifted: {relative}",
        )
    require(result.get("result_identity") == "RESULT(PGIR-206)", "PGIR-206 identity drifted")
    require(result.get("decision") == "no_go", "PGIR-206 is not no_go")
    require(result.get("disposition") == "typed_not_run", "PGIR-206 disposition drifted")
    require(result.get("execution_status") == "not_run", "PGIR-206 execution status drifted")
    require(result.get("execution_authorized") is False, "PGIR-206 authorized execution")
    require(result.get("descendant_execution_authorized") is False, "PGIR-206 authorized descendants")
    require(result.get("completion_authoritative") is False, "PGIR-206 claimed completion authority")
    require(result.get("checkpoint_count") == 0, "PGIR-206 recorded a checkpoint")
    require(result.get("metric_cell_count") == 192, "PGIR-206 metric-cell population drifted")
    require(result.get("measured_cell_count") == 0, "PGIR-206 recorded measured cells")
    require(result.get("training_admitted_rows") == 0, "PGIR-206 admitted training rows")
    require(result.get("receipt_cids") == PGIR_206_RECEIPTS, "PGIR-206 receipt CID map drifted")
    effects = result.get("observed_effects")
    require(isinstance(effects, dict) and effects, "PGIR-206 effect map missing")
    for key, value in effects.items():
        expected = 0 if key == "optimizer_steps" else False
        require(value == expected, f"PGIR-206 observed prohibited effect: {key}")
    for receipt, label in ((evaluation, "evaluation"), (proof, "proof")):
        require(receipt.get("receipt_cid") == PGIR_206_RECEIPTS[label], f"PGIR-206 {label} receipt drifted")
        require(receipt.get("status") == "not_run", f"PGIR-206 {label} unexpectedly ran")
    return {"result": result, "manifest": manifest, "evaluation": evaluation, "proof": proof}


def verify_acceptance(acceptance: Mapping[str, Any]) -> None:
    verify_projection_identity(acceptance, "acceptance")
    require_task_binding(acceptance, "acceptance")
    require(
        (acceptance.get("schema"), acceptance.get("interface"))
        == ("PGIRFinalAcceptanceCatalog@2", "proof-grounded-ir-learning/final-acceptance/v2"),
        "acceptance schema/interface drifted",
    )
    rows = acceptance.get("criteria")
    require(acceptance.get("criterion_count") == 16 and isinstance(rows, list) and len(rows) == 16, "acceptance population drifted")
    require([row.get("criterion_id") for row in rows] == list(CRITERION_STATUS), "criterion order drifted")
    for row in rows:
        criterion_id = row["criterion_id"]
        status = CRITERION_STATUS[criterion_id]
        require(row.get("status") == status, f"{criterion_id} status drifted")
        require(isinstance(row.get("evidence"), str) and row["evidence"].strip(), f"{criterion_id} evidence missing")
        evidence_cids = row.get("evidence_cids")
        require(isinstance(evidence_cids, list) and evidence_cids, f"{criterion_id} evidence CID closure missing")
        require(all(isinstance(cid, str) and cid.startswith("b") for cid in evidence_cids), f"{criterion_id} evidence CID invalid")
        resolved = status not in {"unresolved", "missing"}
        require(row.get("gate_resolved") is resolved, f"{criterion_id} resolved gate drifted")
        expected_promotion = False
        require(row.get("promotion_gate_pass") is expected_promotion, f"{criterion_id} promotion gate drifted")
    require(acceptance.get("promotion_gate_pass_count") == 0, "promotion-gate pass count drifted")


def verify_sections(sections: Mapping[str, Any]) -> None:
    verify_projection_identity(sections, "sections")
    require_task_binding(sections, "sections")
    require(
        (sections.get("schema"), sections.get("interface"))
        == ("PGIRFinalReportSectionCatalog@2", "proof-grounded-ir-learning/final-report-sections/v2"),
        "report-section schema/interface drifted",
    )
    rows = sections.get("sections")
    require(sections.get("section_count") == 32 and isinstance(rows, list) and len(rows) == 32, "report-section population drifted")
    for row, (number, slug, title), status in zip(rows, REPORT_SECTIONS, SECTION_STATUS, strict=True):
        require((row.get("number"), row.get("slug"), row.get("title")) == (number, slug, title), f"report section {number} identity drifted")
        require(row.get("status") == status, f"report section {number} status drifted")
        require(isinstance(row.get("summary"), str) and row["summary"].strip(), f"report section {number} summary missing")
        require(isinstance(row.get("bindings"), dict) and row["bindings"], f"report section {number} evidence bindings missing")
        evidence_cids = row.get("evidence_cids")
        require(isinstance(evidence_cids, list) and evidence_cids, f"report section {number} evidence CID closure missing")
        require(all(isinstance(cid, str) and cid.startswith("b") for cid in evidence_cids), f"report section {number} evidence CID invalid")


def verify_receipts(evaluation: Mapping[str, Any], proof: Mapping[str, Any]) -> None:
    verify_projection_identity(evaluation, "evaluation")
    verify_projection_identity(proof, "proof")
    require_task_binding(evaluation, "evaluation receipt")
    require_task_binding(proof, "proof receipt")
    require(evaluation.get("schema") == "PGIRQualificationEvaluationReceipt@1", "evaluation receipt schema drifted")
    require(proof.get("schema") == "PGIRQualificationProofReceipt@1", "proof receipt schema drifted")
    require(evaluation.get("verifier_interface") == "pgir-terminal-qualification-evaluation/v1", "evaluation verifier interface drifted")
    require(proof.get("verifier_interface") == "pgir-terminal-qualification-proof/v1", "proof verifier interface drifted")
    require(evaluation.get("status") == "verified_typed_not_run", "evaluation receipt status drifted")
    require(evaluation.get("metric_cell_count") == 192, "evaluation declared metric-cell population drifted")
    require(evaluation.get("checked_metric_cell_count") == 192, "evaluation metric-cell population drifted")
    require(evaluation.get("measured_cell_count") == 0, "evaluation recorded measured cells")
    require(evaluation.get("authorizes_promotion") is False, "evaluation authorized promotion")
    for key in (
        "best_test_selection", "evaluation_invoked", "hidden_labels_opened",
        "hidden_test_access", "hidden_test_selection", "hidden_test_tuning",
        "missing_metric_as_zero", "uncertainty_substituted",
    ):
        require(evaluation.get(key) is False, f"evaluation receipt effect drifted: {key}")
    require(evaluation.get("target_attainment_claim_count") == 0, "evaluation claimed target attainment")
    require(proof.get("status") == "verified_typed_not_run", "proof receipt status drifted")
    for key in ("proof_attempt_count", "checked_proof_count", "proof_invocation_count", "kernel_check_count"):
        require(proof.get(key) == 0, f"proof receipt {key} drifted")
    require(proof.get("proof_invoked") is False, "proof receipt recorded a proof invocation")
    require(proof.get("authorizes_promotion") is False, "proof receipt authorized promotion")
    require(proof.get("authority_granted") is False, "proof receipt granted authority")
    require(proof.get("hidden_labels_opened") is False, "proof receipt opened hidden labels")
    require(proof.get("proof_results") == [], "proof receipt recorded proof results")
    for payload, source_cid, label in (
        (evaluation, PGIR_206_RECEIPTS["evaluation"], "evaluation"),
        (proof, PGIR_206_RECEIPTS["proof"], "proof"),
    ):
        values = deep_values(payload)
        require(PGIR_206_RESULT_CID in values, f"{label} receipt lost PGIR-206 result binding")
        require(source_cid in values, f"{label} receipt lost source receipt binding")


def verify_decision(
    decision: Mapping[str, Any],
    promotion: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> None:
    for payload, name, label in (
        (decision, "decision", "decision"),
        (promotion, "promotion", "promotion"),
        (publication, "publication", "publication"),
    ):
        verify_projection_identity(payload, name)
        require_task_binding(payload, label)
    require(decision.get("schema") == "PGIRQualificationDecision@2", "decision schema drifted")
    require(promotion.get("schema") == "PGIRQualificationPromotionReceipt@2", "promotion schema drifted")
    require(publication.get("schema") == "PGIRQualificationPublicationReceipt@2", "publication schema drifted")
    require(decision.get("interface") == "proof-grounded-ir-learning/qualification-decision/v2", "decision interface drifted")
    require(promotion.get("interface") == "proof-grounded-ir-learning/qualification-promotion/v2", "promotion interface drifted")
    require(publication.get("interface") == "proof-grounded-ir-learning/qualification-publication/v2", "publication interface drifted")
    require(decision.get("decision") == "no_go", "qualification decision is not no_go")
    require(decision.get("disposition") == "qualified_no_go", "qualification disposition drifted")
    require(decision.get("candidate_checkpoint") is None, "spurious candidate checkpoint")
    require(decision.get("qualified_claim_emitted") is False, "qualified claim was emitted")
    require(decision.get("publication_authorized") is False, "publication was authorized")
    require(decision.get("hidden_tests_opened") is False, "hidden tests were opened")
    require(decision.get("criterion_count") == 16 and decision.get("section_count") == 32, "decision population counts drifted")
    require(decision.get("all_criteria_resolved") is True and decision.get("all_sections_resolved") is True, "decision left evidence unresolved")
    require(decision.get("no_go_criterion_ids") == ["F04", "F09", "F10", "F11", "F12", "F13", "F14", "F15"], "decision no-go criterion set drifted")
    require(tuple(decision.get("reason_codes", ())) == REASON_CODES, "decision reason codes drifted")
    require(promotion.get("decision") == "no_go", "promotion decision drifted")
    require(promotion.get("candidate_checkpoint") is None, "promotion names a candidate")
    for key in ("promotion_authorized", "pointer_mutated", "cas_attempted", "self_promotion"):
        require(promotion.get(key) is False, f"promotion effect drifted: {key}")
    require(promotion.get("pointer_unchanged") is True, "promotion pointer is not explicitly unchanged")
    require(publication.get("decision") == "denied", "publication decision drifted")
    for key in ("publication_authorized", "upload_attempted", "network_accessed"):
        require(publication.get(key) is False, f"publication effect drifted: {key}")
    require(publication.get("upload_authorized") is False, "publication upload was authorized")
    require(publication.get("remote_revision") is None, "publication recorded a remote revision")


def edge_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    for source_key, target_key in (
        ("task_id", "dependency_task_id"),
        ("from", "to"),
        ("dependent", "dependency"),
        ("source", "target"),
    ):
        if source_key in row and target_key in row:
            return str(row[source_key]), str(row[target_key])
    raise QualificationVerificationError("result graph edge has no recognized endpoint fields")


def verify_graph(graph: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    verify_projection_identity(graph, "graph")
    require_task_binding(graph, "result graph")
    require(
        (graph.get("schema"), graph.get("interface"))
        == ("PGIRCompleteResultGraph@1", "proof-grounded-ir-learning/complete-result-graph/v1"),
        "result graph schema/interface drifted",
    )
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    require(graph.get("node_count") == 16 and isinstance(nodes, list) and len(nodes) == 16, "result graph node population drifted")
    require(graph.get("edge_count") == 28 and isinstance(edges, list) and len(edges) == 28, "result graph edge population drifted")
    require(tuple(graph.get("dependency_first_order", ())) == DEPENDENCY_FIRST_ORDER, "dependency-first order drifted")
    node_ids = tuple(row.get("task_id") for row in nodes)
    require(node_ids == DEPENDENCY_FIRST_ORDER, "result graph node order drifted")
    for row in nodes:
        task_id = row["task_id"]
        require(row.get("result_identity") == f"RESULT({task_id})", f"graph result identity drifted: {task_id}")
        require(tuple(row.get("depends_on", ())) == GRAPH_DEPENDENCIES[task_id], f"graph dependencies drifted: {task_id}")
        require(isinstance(row.get("evidence_cid"), str) and row["evidence_cid"].startswith("b"), f"graph evidence CID missing: {task_id}")
    observed_edges = tuple(edge_pair(row) for row in edges)
    require(observed_edges == EXPECTED_EDGES, "result graph exact edge order drifted")
    positions = {task_id: index for index, task_id in enumerate(DEPENDENCY_FIRST_ORDER)}
    require(all(positions[dependency] < positions[task_id] for task_id, dependency in observed_edges), "result graph is not dependency-first")
    leaves = tuple(row["task_id"] for row in nodes if row.get("historical_leaf"))
    require(leaves == ("PGIR-072", "PGIR-090", "PGIR-100", "PGIR-111"), "historical graph leaves drifted")
    require(graph.get("historical_leaf_count") == 4, "historical graph leaf count drifted")
    require(graph.get("root_node_id") == "result:PGIR-207", "result graph root drifted")

    root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    integrated = strict_json(FREEZE_DIR / "integrated-acceptance" / "integrated_acceptance.json")
    require(root.get("root_cid") == CAMPAIGN_INPUT_ROOT_CID, "graph source campaign root drifted")
    require(integrated.get("acceptance_cid") == PGIR_211_ACCEPTANCE_CID, "graph source PGIR-211 acceptance drifted")
    evidence = {
        **HISTORICAL_EVIDENCE_CIDS,
        "PGIR-200": root["bindings"]["rights"]["binding_cid"],
        "PGIR-201": root["bindings"]["corpus"]["binding_cid"],
        "PGIR-202": root["bindings"]["split"]["binding_cid"],
        "PGIR-203": root["bindings"]["tokenizer_policy"]["result_cid"],
        "PGIR-204": root["gates"]["current_baseline"]["retirement_cid"],
        "PGIR-208": integrated["predecessor_acceptance_cids"]["PGIR-208"],
        "PGIR-209": integrated["predecessor_acceptance_cids"]["PGIR-209"],
        "PGIR-210": integrated["predecessor_acceptance_cids"]["PGIR-210"],
        "PGIR-211": integrated["acceptance_cid"],
        "PGIR-205": PGIR_205_RESULT_CID,
        "PGIR-206": PGIR_206_RESULT_CID,
        "PGIR-207": decision["decision_cid"],
    }
    expected_nodes = [
        {
            "depends_on": list(GRAPH_DEPENDENCIES[task_id]),
            "dependency_first_position": position,
            "disposition": GRAPH_DISPOSITIONS[task_id],
            "evidence_cid": evidence[task_id],
            "historical_leaf": task_id in HISTORICAL_TASK_CIDS,
            "node_id": f"result:{task_id}",
            "result_identity": f"RESULT({task_id})",
            "task_cid": HISTORICAL_TASK_CIDS.get(
                task_id, TASK_CID if task_id == TASK_ID else None
            ),
            "task_id": task_id,
        }
        for position, task_id in enumerate(DEPENDENCY_FIRST_ORDER)
    ]
    require(nodes == expected_nodes, "result graph exact node records drifted")
    expected_edge_rows = [
        {"dependency": dependency, "dependent": task_id, "edge_id": f"{task_id}->{dependency}"}
        for task_id, dependency in EXPECTED_EDGES
    ]
    require(edges == expected_edge_rows, "result graph exact edge records drifted")


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    verify_projection_identity(manifest, "manifest")
    require_task_binding(manifest, "manifest")
    require(
        (manifest.get("schema"), manifest.get("interface"))
        == ("PGIRTerminalQualificationManifest@1", "proof-grounded-ir-learning/qualification-manifest/v2"),
        "manifest schema/interface drifted",
    )
    records = manifest.get("artifacts", manifest.get("files"))
    require(isinstance(records, dict), "manifest artifact map missing")
    require(manifest.get("artifact_count") == 14 and len(records) == 14, "manifest artifact count drifted")
    require(set(records) == set(SEALED_PATHS), "manifest sealed path population drifted")
    require(tuple(manifest.get("sealed_path_order", ())) == SEALED_PATHS, "manifest explicit sealed path order drifted")
    for relative, record in records.items():
        require(isinstance(record, dict), f"manifest record is not an object: {relative}")
        path = REPOSITORY_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing regular sealed file: {relative}")
        require(record == identity(path.read_bytes()), f"manifest byte identity drifted: {relative}")
    excluded = manifest.get("excluded_paths")
    if excluded is not None:
        require(tuple(excluded) == FINAL_PATHS[-3:], "manifest exclusion population drifted")


def verify_docs(report: str, board: str) -> None:
    require(QUALIFIED_CLAIM not in report and QUALIFIED_CLAIM not in board, "qualified claim leaked")
    require("Never claim universal legal-semantic understanding." in report, "universal-claim prohibition missing")
    require("qualification decision is `no_go`" in report.lower(), "report no-go closing claim missing")
    for number, _slug, title in REPORT_SECTIONS:
        require(f"## {number}. {title}" in report, f"missing report section {number}")
    for criterion_id in CRITERION_STATUS:
        require(f"### {criterion_id} " in report, f"missing report criterion {criterion_id}")
    headings = re.findall(r"^## (PGIR-\d{3})\b", board, flags=re.MULTILINE)
    require(headings == ["PGIR-212"], "following board is not exactly one PGIR-212 card")
    required_lines = (
        "- Status: blocked",
        "- Completion: manual-external-evidence",
        "- Is schedulable: false",
        "- Depends on: PGIR-207",
    )
    for line in required_lines:
        require(line in board, f"following board missing {line}")
    lowered = board.lower()
    require("no descendants" in lowered or "descendants: none" in lowered, "PGIR-212 descendant denial missing")
    require("validation: none" in lowered or "no runnable validation" in lowered, "PGIR-212 runnable validation was not denied")
    require("pytest" not in lowered and "python " not in lowered, "PGIR-212 contains a runnable validation command")


def commit_record(commit: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, f"invalid commit ID: {commit}")
    return {
        "commit": commit,
        "parents": git(REPOSITORY_ROOT, "show", "-s", "--format=%P", commit).split(),
        "subject": git(REPOSITORY_ROOT, "show", "-s", "--format=%s", commit),
        "tree": git(REPOSITORY_ROOT, "show", "-s", "--format=%T", commit),
    }


def verify_pgir_206_forest(forest: Any) -> dict[str, Any]:
    require(isinstance(forest, list) and len(forest) == 3, "PGIR-206 forest population drifted")
    require(all(isinstance(row, dict) for row in forest), "PGIR-206 forest row is not an object")
    require([row.get("role") for row in forest] == ["implementation", "merge", "completion"], "PGIR-206 forest roles drifted")
    for row in forest:
        observed = {**commit_record(str(row.get("commit"))), "role": row.get("role")}
        require(row == observed, f"PGIR-206 {row.get('role')} Git object record drifted")
    implementation, merge, completion = forest
    require(implementation["commit"] == PGIR_206_IMPLEMENTATION, "PGIR-206 implementation commit drifted")
    require(implementation["parents"] == [PGIR_206_IMPLEMENTATION_PARENT], "PGIR-206 implementation parent drifted")
    require(implementation["subject"] == "PGIR-206: Re-run R1-R6 on the superseding freeze", "PGIR-206 implementation subject drifted")
    require(len(merge["parents"]) == 2 and merge["parents"][1] == implementation["commit"], "PGIR-206 merge parents drifted")
    require(
        merge["subject"]
        == f"Merge commit '{PGIR_206_IMPLEMENTATION}' into agent/pgir-successor-current-supervisor-20260825",
        "PGIR-206 merge subject drifted",
    )
    require(completion["parents"] == [merge["commit"]], "PGIR-206 completion parent drifted")
    require(completion["subject"] == "PGIR-206: mark todo completed", "PGIR-206 completion subject drifted")
    require(
        git(REPOSITORY_ROOT, "merge-base", PGIR_206_IMPLEMENTATION_PARENT, merge["parents"][0])
        == PGIR_206_IMPLEMENTATION_PARENT,
        "PGIR-206 merge first parent does not descend from the implementation base",
    )
    for ancestor, descendant in (
        (implementation["commit"], merge["commit"]),
        (merge["commit"], completion["commit"]),
        (completion["commit"], git(REPOSITORY_ROOT, "rev-parse", "HEAD")),
    ):
        require(git(REPOSITORY_ROOT, "merge-base", ancestor, descendant) == ancestor, f"PGIR-206 ancestry drifted: {ancestor} -> {descendant}")
    require(
        git(REPOSITORY_ROOT, "rev-parse", f"{completion['commit']}:ipfs_datasets_py")
        == NESTED_CURRENT,
        "PGIR-206 completion gitlink drifted",
    )
    return {
        "implementation": implementation["commit"],
        "merge": merge["commit"],
        "completion": completion["commit"],
        "completion_tree": completion["tree"],
    }


def verify_static() -> dict[str, Any]:
    pgir_206 = verify_pgir_206()
    acceptance = strict_json(PACKAGE_DIR / "acceptance.json")
    sections = strict_json(PACKAGE_DIR / "report_sections.json")
    evaluation = strict_json(PACKAGE_DIR / "evaluation_receipt.json")
    proof = strict_json(PACKAGE_DIR / "proof_receipt.json")
    decision = strict_json(PACKAGE_DIR / "decision.json")
    promotion = strict_json(PACKAGE_DIR / "promotion_receipt.json")
    publication = strict_json(PACKAGE_DIR / "publication_receipt.json")
    graph = strict_json(PACKAGE_DIR / "result_graph.json")
    recipe = strict_json(PACKAGE_DIR / "recipe.json")
    manifest = strict_json(PACKAGE_DIR / "manifest.json")
    verification = strict_json(PACKAGE_DIR / "verification_receipt.json")
    result = strict_json(PACKAGE_DIR / "result.json")

    verify_acceptance(acceptance)
    verify_sections(sections)
    verify_receipts(evaluation, proof)
    verify_decision(decision, promotion, publication)
    verify_graph(graph, decision)
    verify_projection_identity(recipe, "recipe")
    require_task_binding(recipe, "recipe")
    require(
        (recipe.get("schema"), recipe.get("interface"))
        == ("PGIRQualificationRecipe@2", "proof-grounded-ir-learning/qualification-recipe/v2"),
        "recipe schema/interface drifted",
    )
    verify_manifest(manifest)
    verify_projection_identity(verification, "verification")
    require_task_binding(verification, "verification receipt")
    require(
        (verification.get("schema"), verification.get("verifier_interface"))
        == (
            "PGIRQualificationVerificationReceipt@1",
            "pgir-terminal-qualification-independent-verifier/v1",
        ),
        "verification receipt schema/interface drifted",
    )
    require(verification.get("verification_verdict") == "verified_terminal_no_go", "verification verdict drifted")
    require(verification.get("all_integrity_checks_passed") is True, "verification checks did not all pass")
    for key in ("authorizes_execution", "authorizes_promotion", "authorizes_publication"):
        require(verification.get(key) is False, f"verification receipt unexpectedly {key}")
    verify_projection_identity(result, "result")
    require(result.get("schema") == "pgir-task-result@1", "result schema drifted")
    require_task_binding(result, "result")

    forest = recipe.get("pgir_206_forest")
    forest_summary = verify_pgir_206_forest(forest)
    for payload, label in ((manifest, "manifest"), (result, "result")):
        require(payload.get("pgir_206_forest") == forest, f"{label} PGIR-206 forest drifted")
    require(sections["sections"][0]["bindings"].get("pgir_206_forest") == forest, "report section 1 forest drifted")
    require(sections["sections"][27]["bindings"].get("pgir_206_forest") == forest, "report section 28 forest drifted")
    expected_recursive = {
        "nested_commit": NESTED_CURRENT,
        "nested_tree": NESTED_CURRENT_TREE,
        "outer_commit": forest_summary["completion"],
        "outer_tree": forest_summary["completion_tree"],
    }
    require(result.get("recursive_repository") == expected_recursive, "result recursive repository binding drifted")
    require(result.get("source_tree_id") == forest_summary["completion_tree"], "result source tree drifted")
    expected_checks = [
        ("recursive-repository-forest", forest_summary["completion"]),
        ("pgir-206-implementation-merge-completion", forest_summary["completion"]),
        ("pgir-206-result", PGIR_206_RESULT_CID),
        ("pgir-206-manifest", PGIR_206_MANIFEST_CID),
        ("acceptance-16-resolved", acceptance["acceptance_cid"]),
        ("report-sections-32-resolved", sections["sections_cid"]),
        ("evaluation-typed-not-run", evaluation["evaluation_cid"]),
        ("proof-typed-not-run", proof["proof_cid"]),
        ("terminal-no-go-decision", decision["decision_cid"]),
        ("promotion-pointer-unchanged", promotion["promotion_cid"]),
        ("publication-not-attempted", publication["publication_cid"]),
        ("complete-result-graph-16x28", graph["graph_cid"]),
        ("qualification-recipe", recipe["recipe_cid"]),
        ("sealed-artifact-manifest-14", manifest["manifest_cid"]),
        ("following-board-pgir-212-blocked", "PGIR-212"),
    ]
    require(
        verification.get("checks")
        == [
            {"check_id": check_id, "evidence": evidence, "status": "passed"}
            for check_id, evidence in expected_checks
        ],
        "verification check population or evidence drifted",
    )

    report_path = DOCS_DIR / "final_report.md"
    board_path = DOCS_DIR / "next.todo.md"
    readme_path = PACKAGE_DIR / "README.md"
    require(readme_path.is_file() and not readme_path.is_symlink(), "qualification README missing")
    require(report_path.is_file() and not report_path.is_symlink(), "successor final report missing")
    require(board_path.is_file() and not board_path.is_symlink(), "successor next board missing")
    readme = readme_path.read_text(encoding="utf-8")
    require(
        "/usr/bin/python3.12 -S data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verify_qualification.py --fresh-recursive"
        in readme,
        "README lost canonical fresh-recursive verification command",
    )
    verify_docs(report_path.read_text(encoding="utf-8"), board_path.read_text(encoding="utf-8"))

    expected_files = {str((REPOSITORY_ROOT / path).resolve()) for path in FINAL_PATHS}
    observed_files = {
        str(path.resolve())
        for directory in (PACKAGE_DIR, DOCS_DIR)
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    require(observed_files == expected_files, "terminal qualification exact file population drifted")
    for relative in FINAL_PATHS:
        mode = (REPOSITORY_ROOT / relative).lstat().st_mode
        permission = stat.S_IMODE(mode)
        require(
            stat.S_ISREG(mode) and permission & 0o111 == 0 and permission & 0o002 == 0,
            f"unsafe terminal artifact mode: {relative}",
        )

    links = {
        "acceptance_cid": acceptance["acceptance_cid"],
        "sections_cid": sections["sections_cid"],
        "evaluation_cid": evaluation["evaluation_cid"],
        "proof_cid": proof["proof_cid"],
        "decision_cid": decision["decision_cid"],
        "promotion_cid": promotion["promotion_cid"],
        "publication_cid": publication["publication_cid"],
        "result_graph_cid": graph["graph_cid"],
    }
    recipe_values = deep_values(recipe)
    for label, cid in links.items():
        require(cid in recipe_values, f"recipe lost {label}")
    manifest_values = deep_values(manifest)
    for cid in (recipe["recipe_cid"], graph["graph_cid"], decision["decision_cid"]):
        require(cid in manifest_values, f"manifest lost qualification link {cid}")
    verification_values = deep_values(verification)
    for cid in (manifest["manifest_cid"], decision["decision_cid"], graph["graph_cid"]):
        require(cid in verification_values, f"verification receipt lost qualification link {cid}")

    require(result.get("task_id") == TASK_ID, "result task drifted")
    require(result.get("result_identity") == RESULT_IDENTITY, "result identity drifted")
    require(result.get("decision") == "no_go", "result decision drifted")
    require(result.get("disposition") == "qualified_no_go", "result disposition drifted")
    require(result.get("completion_authoritative") is False, "result claimed completion authority")
    require(result.get("execution_status") == "not_run", "result execution status drifted")
    require(result.get("execution_authorized") is False, "result authorized execution")
    require(result.get("promotion_authorized") is False, "result authorized promotion")
    require(result.get("publication_authorized") is False, "result authorized publication")
    require(result.get("automated_descendant_count") == 0, "result created automated descendants")
    require(result.get("next_task_ids") == ["PGIR-212"], "result next-task population drifted")
    require(result.get("candidate_checkpoint") is None, "result names a candidate checkpoint")
    require(result.get("hidden_tests_opened") is False, "result opened hidden tests")
    require(result.get("qualified_claim_emitted") is False, "result emitted the qualified claim")
    result_values = deep_values(result)
    for cid in (manifest["manifest_cid"], verification["verification_cid"], graph["graph_cid"], decision["decision_cid"]):
        require(cid in result_values, f"result lost qualification link {cid}")
    require(PGIR_206_RESULT_CID in result_values, "result lost PGIR-206 binding")
    require(tuple(result.get("reason_codes", ())) == REASON_CODES, "result reason codes drifted")

    del pgir_206
    return {
        "verified": True,
        "fresh_recursive_verified": False,
        "decision": "no_go",
        "completion_authoritative": False,
        "execution_authorized": False,
        "promotion_authorized": False,
        "publication_authorized": False,
        "manifest_cid": manifest["manifest_cid"],
        "result_cid": result["result_cid"],
        "verification_cid": verification["verification_cid"],
    }


def run_git(repository: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ("/usr/bin/git", "-C", str(repository), *args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=GIT_ENVIRONMENT,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationVerificationError(f"git {' '.join(args)} failed: {exc}") from exc


def git(repository: Path, *args: str) -> str:
    process = run_git(repository, *args)
    detail = (process.stderr or process.stdout).decode("utf-8", "replace").strip()
    require(process.returncode == 0, f"git {' '.join(args)} failed: {detail}")
    return process.stdout.decode("utf-8").strip()


def fresh_recursive_replay(static_result: Mapping[str, Any]) -> dict[str, Any]:
    source_head = git(REPOSITORY_ROOT, "rev-parse", "HEAD")
    source_tree = git(REPOSITORY_ROOT, "rev-parse", "HEAD^{tree}")
    source_nested = git(REPOSITORY_ROOT, "rev-parse", "HEAD:ipfs_datasets_py")
    require(source_nested == NESTED_CURRENT, "source gitlink drifted")
    require(NESTED_ROOT.joinpath(".git").exists(), "source ipfs_datasets_py checkout is not initialized")
    require(git(NESTED_ROOT, "rev-parse", "HEAD") == NESTED_CURRENT, "source nested checkout drifted")
    require(git(NESTED_ROOT, "rev-parse", "HEAD^{tree}") == NESTED_CURRENT_TREE, "source nested tree drifted")
    require(git(REPOSITORY_ROOT, "status", "--porcelain=v1", "--untracked-files=all") == "", "source outer checkout is not clean")
    require(git(NESTED_ROOT, "status", "--porcelain=v1", "--untracked-files=all") == "", "source nested checkout is not clean")

    with tempfile.TemporaryDirectory(prefix="pgir207-fresh-recursive-") as temporary:
        checkout = Path(temporary) / "checkout"
        clone = subprocess.run(
            (
                "/usr/bin/git", "clone", "--no-local", "--no-checkout", "--",
                str(REPOSITORY_ROOT), str(checkout),
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=GIT_ENVIRONMENT,
            timeout=180,
        )
        detail = (clone.stderr or clone.stdout).decode("utf-8", "replace").strip()
        require(clone.returncode == 0, f"fresh outer clone failed: {detail}")
        git(checkout, "checkout", "--detach", source_head)
        update = subprocess.run(
            (
                "/usr/bin/git", "-C", str(checkout),
                "-c", "protocol.file.allow=always",
                "-c", f"submodule.ipfs_datasets_py.url={NESTED_ROOT}",
                "submodule", "update", "--init", "--", "ipfs_datasets_py",
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=GIT_ENVIRONMENT,
            timeout=180,
        )
        detail = (update.stderr or update.stdout).decode("utf-8", "replace").strip()
        require(update.returncode == 0, f"fresh nested clone failed: {detail}")
        clone_nested = checkout / "ipfs_datasets_py"
        require(git(checkout, "rev-parse", "HEAD") == source_head, "fresh outer HEAD drifted")
        require(git(checkout, "rev-parse", "HEAD^{tree}") == source_tree, "fresh outer tree drifted")
        require(git(clone_nested, "rev-parse", "HEAD") == NESTED_CURRENT, "fresh nested HEAD drifted")
        require(git(clone_nested, "rev-parse", "HEAD^{tree}") == NESTED_CURRENT_TREE, "fresh nested tree drifted")
        require(git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == "", "fresh outer checkout is not clean")
        require(git(clone_nested, "status", "--porcelain=v1", "--untracked-files=all") == "", "fresh nested checkout is not clean")

        verifier = checkout / Path(__file__).resolve().relative_to(REPOSITORY_ROOT)
        replay = subprocess.run(
            ("/usr/bin/python3.12", "-S", str(verifier), "--static-only"),
            cwd=checkout,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=PYTHON_ENVIRONMENT,
            timeout=180,
        )
        detail = (replay.stderr or replay.stdout).decode("utf-8", "replace").strip()
        require(replay.returncode == 0, f"fresh verifier replay failed: {detail}")
        try:
            replay_result = json.loads(replay.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QualificationVerificationError(f"fresh verifier output was not JSON: {exc}") from exc
        require(replay_result == dict(static_result), "fresh verifier result differs from source replay")

    result = dict(static_result)
    result.update(
        {
            "fresh_recursive_verified": True,
            "outer_commit": source_head,
            "outer_tree": source_tree,
            "nested_commit": NESTED_CURRENT,
            "nested_tree": NESTED_CURRENT_TREE,
            "clone_mode": "git clone --no-local plus local exact submodule override",
        }
    )
    return result


def emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fresh-recursive", action="store_true", help="replay from a new outer and nested checkout")
    modes.add_argument("--static-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(sys.flags.no_site == 1, "verification requires Python -S (no site initialization)")
    result = verify_static()
    if args.fresh_recursive:
        result = fresh_recursive_replay(result)
    emit(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        QualificationVerificationError,
        AttributeError,
        KeyError,
        IndexError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        emit(
            {
                "verified": False,
                "fresh_recursive_verified": False,
                "decision": "no_go",
                "completion_authoritative": False,
                "execution_authorized": False,
                "promotion_authorized": False,
                "publication_authorized": False,
                "error": str(exc),
            }
        )
        raise SystemExit(1)
