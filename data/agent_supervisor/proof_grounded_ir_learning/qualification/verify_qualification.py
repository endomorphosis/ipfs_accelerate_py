#!/usr/bin/env python3
"""Independently verify the immutable PGIR-111 qualification package.

This verifier uses only the Python standard library.  It does not import the
builder, start a daemon, open hidden tests, or treat a verified no-go as
promotion or publication authority.  A successful exit means the 16 criteria,
32 report sections, no-go decision, withheld qualified claim, denied
publication, and next board all replayed and stayed fail-closed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


QUALIFICATION_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = QUALIFICATION_DIR.parents[3]
FREEZE_DIR = QUALIFICATION_DIR.parent / "freeze"
EXPERIMENTS_DIR = QUALIFICATION_DIR.parent / "experiments"
DOCS_DIR = REPOSITORY_ROOT / "docs" / "architecture" / "proof_grounded_ir_learning"
DATASETS_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"

TASK_ID = "PGIR-111"
FREEZE_RESULT_CID = "baguqeerai2ipwhyywztjob62ju5pokmm4o6unqqee3poyrabj37aby6fuoca"
FREEZE_ROOT_CID = "baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q"
HIDDEN_TEST_COMMITMENT = "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
QUALIFIED_CLAIM = (
    "The qualified compiler/decompiler checkpoint was trained from "
    "content-addressed, lineage-safe JusticeDAO source and proof artifacts under "
    "the declared split, compiler, tokenizer, loss, curriculum, and supervisor "
    "configuration."
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
REQUIRED_CRITERIA = tuple(f"F{index:02d}" for index in range(1, 17))
REQUIRED_SECTION_TITLES = (
    "Exact source revisions",
    "Exact JusticeDAO repository revisions and configurations",
    "Current-state inventory",
    "Source versus derived record counts",
    "Lineage and split design",
    "Leakage-audit results",
    "Canonical bridge-IR design",
    "Compiler architecture",
    "Decompiler architecture",
    "Deterministic baseline",
    "Learned-model architecture",
    "Tokenizer and vocabulary",
    "Loss configuration",
    "Training curriculum",
    "Hard-negative generation",
    "Lean-capable model results",
    "Tactician results",
    "Hammer results",
    "Kernel-verification results",
    "Cross-entropy metrics",
    "Cosine and contrastive metrics",
    "Retrieval metrics",
    "Structural metrics",
    "Semantic metrics",
    "Proof metrics",
    "Calibration and OOD metrics",
    "Resource utilization",
    "Multi-supervisor scheduling results",
    "Checkpoint promotion or rejection decision",
    "Published artifacts",
    "Known limitations",
    "Exact recommendation for the next training and data-improvement board",
)
NEXT_TASKS = tuple(f"PGIR-{index}" for index in range(200, 208))
SEAL_FIELDS = {
    "catalog_cid": ("catalog_cid", "catalog_sha256"),
    "decision_cid": ("decision_cid", "decision_sha256"),
    "manifest_cid": ("manifest_cid", "manifest_sha256"),
    "receipt_cid": ("receipt_cid", "receipt_sha256"),
    "recipe_cid": ("recipe_cid", "recipe_sha256"),
    "result_cid": ("result_cid", "result_sha256"),
}


class QualificationVerificationError(ValueError):
    """Raised when any identity, criterion, or fail-closed gate drifts."""


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
                raise QualificationVerificationError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                QualificationVerificationError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                QualificationVerificationError(f"non-finite number {raw!r} in {path}")
            ),
        )
    if not isinstance(value, dict):
        raise QualificationVerificationError(f"{path} must contain a JSON object")
    validate_value(value, str(path))
    return value


def verify_identity(payload: Mapping[str, Any], cid_field: str, sha_field: str) -> None:
    require(cid_field in payload, f"missing {cid_field}")
    require(sha_field in payload, f"missing {sha_field}")
    projection = {key: value for key, value in payload.items() if key not in {cid_field, sha_field}}
    expected_sha = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    require(payload[sha_field] == expected_sha, f"{sha_field} drifted")
    require(payload[cid_field] == dag_json_cid(projection), f"{cid_field} drifted")


def verify_file_record(record: Mapping[str, Any]) -> bytes:
    path = REPOSITORY_ROOT / record["path"]
    require(path.is_file(), f"missing {record['path']}")
    data = path.read_bytes()
    require(raw_cid(data) == record["raw_cid"], f"raw CID drifted for {record['path']}")
    require(
        "sha256:" + hashlib.sha256(data).hexdigest() == record["sha256"],
        f"sha256 drifted for {record['path']}",
    )
    require(len(data) == record["size_bytes"], f"size drifted for {record['path']}")
    return data


def verify() -> dict[str, Any]:
    freeze_result = strict_json(FREEZE_DIR / "result.v3.json")
    freeze_root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    tokenizer = strict_json(FREEZE_DIR / "tokenizer_policy.json")
    corpus = strict_json(DATASETS_ROOT / "data/ir_learning/corpora/corpus_root.json")
    leakage = strict_json(DATASETS_ROOT / "data/ir_learning/splits/leakage_report.json")
    holdouts = strict_json(DATASETS_ROOT / "data/ir_learning/splits/holdout_report.json")
    publication_policy = strict_json(
        DATASETS_ROOT / "data/ir_learning/releases/publication_policy.json"
    )
    require(freeze_result["result_cid"] == FREEZE_RESULT_CID, "freeze result CID drifted")
    require(freeze_result["decision"] == "no_go", "freeze is not no_go")
    require(freeze_root["root_cid"] == FREEZE_ROOT_CID, "freeze root CID drifted")
    require(tokenizer["status"] == "no_learned_tokenizer_admitted", "tokenizer unexpectedly admitted")
    require(corpus["training_admitted_rows"] == 0, "training rows unexpectedly admitted")
    require(corpus["materialized"] is False, "corpus unexpectedly materialized")
    require(leakage["passed"] is True, "leakage audit no longer passes")
    require(holdouts["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT, "hidden-test commitment drifted")
    require(publication_policy["require_qualification"] is True, "publication policy dropped qualification")
    experiment_readme = (EXPERIMENTS_DIR / "README.md").read_text(encoding="utf-8")
    require("decision is deliberately `no_go`" in experiment_readme, "PGIR-110 README lost no-go")
    require(not (EXPERIMENTS_DIR / "result.json").exists(), "unexpected PGIR-110 result.json")

    acceptance = strict_json(QUALIFICATION_DIR / "acceptance.json")
    sections = strict_json(QUALIFICATION_DIR / "report_sections.json")
    decision = strict_json(QUALIFICATION_DIR / "decision.json")
    promotion = strict_json(QUALIFICATION_DIR / "promotion_receipt.json")
    publication = strict_json(QUALIFICATION_DIR / "publication_receipt.json")
    recipe = strict_json(QUALIFICATION_DIR / "recipe.json")
    result = strict_json(QUALIFICATION_DIR / "result.json")
    manifest = strict_json(QUALIFICATION_DIR / "manifest.json")
    report = (DOCS_DIR / "final_report.md").read_text(encoding="utf-8")
    next_board = (DOCS_DIR / "next.todo.md").read_text(encoding="utf-8")

    for payload, cid_field, sha_field in (
        (acceptance, "catalog_cid", "catalog_sha256"),
        (sections, "catalog_cid", "catalog_sha256"),
        (decision, "decision_cid", "decision_sha256"),
        (promotion, "receipt_cid", "receipt_sha256"),
        (publication, "receipt_cid", "receipt_sha256"),
        (recipe, "recipe_cid", "recipe_sha256"),
        (result, "result_cid", "result_sha256"),
        (manifest, "manifest_cid", "manifest_sha256"),
    ):
        verify_identity(payload, cid_field, sha_field)

    require(acceptance["criterion_count"] == 16, "acceptance catalog count drifted")
    require(len(acceptance["criteria"]) == 16, "acceptance rows drifted")
    require(
        tuple(item["criterion_id"] for item in acceptance["criteria"]) == REQUIRED_CRITERIA,
        "acceptance IDs drifted",
    )
    require(sections["section_count"] == 32, "section catalog count drifted")
    require(len(sections["sections"]) == 32, "section rows drifted")
    require(
        tuple(item["title"] for item in sections["sections"]) == REQUIRED_SECTION_TITLES,
        "section titles drifted",
    )
    require(
        tuple(item["number"] for item in sections["sections"]) == tuple(range(1, 33)),
        "section numbers drifted",
    )

    require(decision["decision"] == "no_go", "qualification decision is not no_go")
    require(decision["qualified_claim_emitted"] is False, "qualified claim was emitted")
    require(decision["publication_authorized"] is False, "publication unexpectedly authorized")
    require(decision["candidate_checkpoint"] is None, "spurious candidate checkpoint")
    require(decision["hidden_tests_opened"] is False, "hidden tests were opened")
    require(tuple(decision["reason_codes"]) == REASON_CODES, "reason codes drifted")
    require(promotion["decision"] == "no_go", "promotion is not no_go")
    require(promotion["pointer_mutated"] is False, "promotion pointer mutated")
    require(promotion["self_promotion"] is False, "self-promotion recorded")
    require(publication["decision"] == "denied", "publication is not denied")
    require(publication["upload_attempted"] is False, "upload was attempted")
    require(publication["remote_revision"] is None, "remote revision recorded")
    require(result["decision"] == "no_go", "result decision drifted")
    require(result["completion_authoritative"] is False, "result claimed completion authority")
    require(result["task_id"] == TASK_ID, "result task drifted")

    promoting = {
        "F04",
        "F09",
        "F10",
        "F11",
        "F12",
        "F13",
        "F14",
        "F15",
    }
    for item in acceptance["criteria"]:
        if item["criterion_id"] in promoting:
            require(item["status"] == "no_go", f"{item['criterion_id']} unexpectedly passed")
            require(item["gate_pass"] is False, f"{item['criterion_id']} gate_pass drifted")
        if item["criterion_id"] == "F16":
            require(item["status"] == "satisfied", "F16 must be satisfied by issuing the next board")

    require(QUALIFIED_CLAIM not in report, "qualified claim leaked into final report")
    require(QUALIFIED_CLAIM not in next_board, "qualified claim leaked into next board")
    require("Never claim universal legal-semantic understanding." in report, "closing prohibition missing")
    require("The qualification decision is `no_go`." in report, "closing no-go missing")
    for index, title in enumerate(REQUIRED_SECTION_TITLES, start=1):
        require(f"## {index}. {title}" in report, f"missing report section {index}")
    require("## Authorized closing claim" in report, "missing authorized closing claim")
    require("## Final acceptance criteria" in report, "missing acceptance section")
    for criterion_id in REQUIRED_CRITERIA:
        require(f"### {criterion_id} " in report, f"missing {criterion_id} in report")
    for task_id in NEXT_TASKS:
        require(f"## {task_id} " in next_board, f"missing {task_id} in next board")
    require("Status: todo" in next_board, "next board has no todo tasks")
    require("justice_dao_pinset.yaml" in next_board, "next board dropped pinset protection")

    for key, record in manifest["files"].items():
        verify_file_record(record)
        del key

    require(recipe["decision_cid"] == decision["decision_cid"], "recipe/decision CID mismatch")
    require(recipe["acceptance_cid"] == acceptance["catalog_cid"], "recipe/acceptance CID mismatch")
    require(recipe["sections_cid"] == sections["catalog_cid"], "recipe/sections CID mismatch")
    require(manifest["result_cid"] == result["result_cid"], "manifest/result CID mismatch")
    require(manifest["decision"] == "no_go", "manifest decision drifted")

    receipt = {
        "campaign_input_root_cid": FREEZE_ROOT_CID,
        "criterion_count": 16,
        "decision": "no_go",
        "decision_cid": decision["decision_cid"],
        "freeze_result_cid": FREEZE_RESULT_CID,
        "interface": "proof-grounded-ir-learning/qualification-verification/v1",
        "manifest_cid": manifest["manifest_cid"],
        "publication_authorized": False,
        "qualified_claim_emitted": False,
        "reason_codes": list(REASON_CODES),
        "result_cid": result["result_cid"],
        "schema": "IRQualificationVerificationReceipt@1",
        "section_count": 32,
        "task_id": TASK_ID,
    }
    projection = dict(receipt)
    receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    receipt["receipt_cid"] = dag_json_cid(projection)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-receipt",
        action="store_true",
        help="Persist verification_receipt.json after a successful replay.",
    )
    args = parser.parse_args()
    receipt = verify()
    if args.write_receipt:
        path = QUALIFICATION_DIR / "verification_receipt.json"
        data = (
            json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if path.exists() and path.read_bytes() != data:
            raise QualificationVerificationError("refusing to replace different verification receipt")
        path.write_bytes(data)
    print(json.dumps({"decision": receipt["decision"], "receipt_cid": receipt["receipt_cid"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationVerificationError as exc:
        print(f"qualification verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
