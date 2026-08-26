#!/usr/bin/env python3
"""Independently replay the append-only PGIR-204 retirement adjudication.

The seven PGIR-204 payloads are immutable nested evidence.  In particular,
this verifier deliberately does not repair their bytes: it records and
corrects the legacy commit/tree terminology only in the outer PGIR-209 seal.
It is a fail-closed evidence gate and never authorizes PGIR-205 execution.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "ipfs_datasets_py"
ACCEPTANCE_PATH = (
    ROOT
    / "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1"
    / "baseline-acceptance/baseline_acceptance.json"
)
PAYLOAD_ROOT = DATASETS / "data/ir_learning/evaluations/deterministic/successor-v1"
NESTED_RESULT_COMMIT = "2a06dfe8546cdde78ff6d101a94708be0e6bf6e6"
PREDECESSOR_DATASETS_COMMIT = "8736a0023d5d3afe4d0e5b044a3e4480966a8bf7"
MISLABELLED_OUTER_COMMIT = "c4cf42fccb58d73b9f48c7f70799165b29cfe3a9"
MISLABELLED_OUTER_TREE = "df137b9691df22a4d928d998062e81a24491705a"
HIDDEN_TEST_COMMITMENT = (
    "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
)
PAYLOAD_NAMES = (
    "identities.json", "manifest.json", "recipe.json", "replay_receipt.json",
    "retirement_receipt.json", "strata.json", "tool_versions.json",
)
HOLDOUT_KEYS = (
    "compiler", "cross_reference", "domain", "exception", "family", "jurisdiction",
    "length", "lineage", "notation", "premise", "proof_library", "publication",
    "rare_operator", "time", "type",
)
HOLDOUT_REPORT_KEYS = (
    "compiler", "cross_reference", "domain", "exception", "length", "lineage",
    "notation", "premise", "proof_library", "publication", "rare_operator", "time", "type",
)
PARTITION_NAMES = (
    "train", "validation", "canary", "holdout", "statute_family", "jurisdiction",
    "temporal", "external_test", "lineage", "publication", "domain", "notation",
    "type", "compiler", "proof_library", "premise", "length", "rare_operator",
    "exception", "cross_reference",
)
HOLDOUT_PARTITIONS = {
    "compiler": "compiler", "cross_reference": "cross_reference", "domain": "domain",
    "exception": "exception", "family": "statute_family", "jurisdiction": "jurisdiction",
    "length": "length", "lineage": "lineage", "notation": "notation",
    "premise": "premise", "proof_library": "proof_library", "publication": "publication",
    "rare_operator": "rare_operator", "time": "temporal", "type": "type",
}


class BaselineAcceptanceError(ValueError):
    """An immutable identity or safety invariant did not replay."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineAcceptanceError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise BaselineAcceptanceError(f"{path} contains a float")
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_value(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, child in value.items():
            validate_value(child, f"{path}.{key}")
        return
    raise BaselineAcceptanceError(f"{path} has unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def raw_cid(data: bytes) -> str:
    multihash = b"\x01\x55\x12\x20" + hashlib.sha256(data).digest()
    return "b" + base64.b32encode(multihash).decode("ascii").rstrip("=").lower()


def dag_json_cid(value: Any) -> str:
    multihash = b"\x01\xa9\x02\x12\x20" + hashlib.sha256(canonical_bytes(value)).digest()
    return "b" + base64.b32encode(multihash).decode("ascii").rstrip("=").lower()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BaselineAcceptanceError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle, object_pairs_hook=pairs,
                parse_float=lambda raw: (_ for _ in ()).throw(
                    BaselineAcceptanceError(f"float {raw!r} in {path}")
                ),
                parse_constant=lambda raw: (_ for _ in ()).throw(
                    BaselineAcceptanceError(f"non-finite number {raw!r} in {path}")
                ),
            )
    except OSError as exc:
        raise BaselineAcceptanceError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineAcceptanceError(f"invalid JSON in {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    validate_value(value, str(path))
    return value


def run_git(repository: Path, *args: str) -> str:
    try:
        process = subprocess.run(
            ("git", "-C", str(repository), *args), text=True, encoding="utf-8",
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BaselineAcceptanceError(f"git {' '.join(args)} unavailable: {exc}") from exc
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip() or "no diagnostic"
        raise BaselineAcceptanceError(f"git {' '.join(args)} failed: {detail}")
    return process.stdout.strip()


def git_bytes(repository: Path, revision_path: str) -> bytes:
    try:
        process = subprocess.run(
            ("git", "-C", str(repository), "show", revision_path), capture_output=True,
            check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BaselineAcceptanceError(f"cannot read sealed blob {revision_path}: {exc}") from exc
    if process.returncode:
        raise BaselineAcceptanceError(f"sealed blob unavailable: {revision_path}")
    return process.stdout


def verify_record(record: Mapping[str, Any], *, revision: str) -> None:
    path = ROOT / str(record["path"])
    require(path.is_file(), f"missing sealed input {record['path']}")
    data = path.read_bytes()
    require(len(data) == record["size_bytes"], f"size drifted for {record['path']}")
    require("sha256:" + hashlib.sha256(data).hexdigest() == record["sha256"],
            f"sha256 drifted for {record['path']}")
    require(raw_cid(data) == record["raw_cid"], f"raw CID drifted for {record['path']}")
    prefix = "ipfs_datasets_py/"
    require(str(record["path"]).startswith(prefix), "record escapes datasets gitlink")
    require(data == git_bytes(DATASETS, f"{revision}:{str(record['path'])[len(prefix):]}"),
            f"working bytes differ from sealed Git blob for {record['path']}")


def verify_acceptance_identity(acceptance: Mapping[str, Any]) -> None:
    projection = {key: value for key, value in acceptance.items()
                  if key not in {"acceptance_sha256", "acceptance_cid"}}
    require(acceptance.get("acceptance_sha256") ==
            "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest(),
            "baseline acceptance SHA-256 drifted")
    require(acceptance.get("acceptance_cid") == dag_json_cid(projection),
            "baseline acceptance CID drifted")
    verifier = acceptance["verifier"]
    source = Path(__file__).read_bytes()
    require(verifier == {
        "path": "scripts/verify_proof_grounded_ir_learning_successor_baseline.py",
        "sha256": "sha256:" + hashlib.sha256(source).hexdigest(),
        "raw_cid": raw_cid(source),
    }, "tracked verifier source identity drifted")


def verify_payloads(acceptance: Mapping[str, Any]) -> None:
    records = acceptance["payloads"]
    require(len(records) == len(PAYLOAD_NAMES), "exactly seven PGIR-204 payloads are required")
    require(tuple(Path(str(item["path"])).name for item in records) == PAYLOAD_NAMES,
            "payload order or population drifted")
    for record in records:
        require(record["task_id"] == "PGIR-204" and record["result_identity"] == "RESULT(PGIR-204)",
                "payload result identity drifted")
        verify_record(record, revision=NESTED_RESULT_COMMIT)
    by_name = {Path(str(record["path"])).name: record for record in records}
    manifest = strict_json(PAYLOAD_ROOT / "manifest.json")
    require(manifest["outcome"] == "historical_r1_retired" and
            manifest["report_cid"] == "baguqeeraw4nh2c7xxamku4juzt5257krzlzuaxe64vl5cuz4h4c4iwm6xdjq",
            "valid PGIR-204 retirement drifted")
    for name, artifact in manifest["artifacts"].items():
        require(name in by_name and artifact["content_cid"] == by_name[name]["raw_cid"],
                f"manifest CID does not close {name}")
    manifest_projection = dict(manifest)
    manifest_cid = manifest_projection.pop("manifest_cid")
    require(manifest_cid == dag_json_cid(manifest_projection), "PGIR-204 manifest CID drifted")
    retirement = strict_json(PAYLOAD_ROOT / "retirement_receipt.json")
    retirement_projection = dict(retirement)
    retirement_cid = retirement_projection.pop("retirement_cid")
    require(retirement_cid == dag_json_cid(retirement_projection) and
            manifest["report_cid"] == retirement_cid, "retirement CID drifted")


def verify_input_closure(acceptance: Mapping[str, Any]) -> None:
    records = acceptance["input_closure"]
    expected_paths = (
        "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json",
        "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/rights_manifest.json",
        "ipfs_datasets_py/data/ir_learning/splits/successor-v1/split_root.json",
        "ipfs_datasets_py/data/ir_learning/splits/successor-v1/holdout_report.json",
        "ipfs_datasets_py/data/ir_learning/splits/successor-v1/leakage_report.json",
        "ipfs_datasets_py/data/ir_learning/splits/successor-v1/ir_split_manifest.json",
        "ipfs_datasets_py/data/ir_learning/evaluations/deterministic/manifest.json",
        "ipfs_datasets_py/data/ir_learning/evaluations/deterministic/r1_baseline.json",
    )
    require(tuple(item["path"] for item in records) == expected_paths,
            "input CID closure population drifted")
    for record in records:
        verify_record(record, revision=(
            PREDECESSOR_DATASETS_COMMIT if "/successor-v1/" in record["path"] else NESTED_RESULT_COMMIT
        ))
    by_path = {str(record["path"]): str(record["raw_cid"]) for record in records}
    replay = strict_json(PAYLOAD_ROOT / "replay_receipt.json")
    expected_replay_cids = {
        "successor_corpus_root": by_path[expected_paths[0]],
        "successor_rights_manifest": by_path[expected_paths[1]],
        "successor_split_root": by_path[expected_paths[2]],
        "successor_holdout_report": by_path[expected_paths[3]],
        "successor_leakage_report": by_path[expected_paths[4]],
        "historical_r1_manifest": by_path[expected_paths[6]],
        "historical_r1_report": by_path[expected_paths[7]],
    }
    require(replay["input_content_cids"] == expected_replay_cids,
            "PGIR-204 replay input CID closure drifted")


def verify_partitions(acceptance: Mapping[str, Any]) -> None:
    corpus = strict_json(DATASETS / "data/ir_learning/corpora/successor-v1/corpus_root.json")
    rights = strict_json(DATASETS / "data/ir_learning/corpora/successor-v1/rights_manifest.json")
    split = strict_json(DATASETS / "data/ir_learning/splits/successor-v1/ir_split_manifest.json")
    holdout_report = strict_json(DATASETS / "data/ir_learning/splits/successor-v1/holdout_report.json")
    leakage = strict_json(DATASETS / "data/ir_learning/splits/successor-v1/leakage_report.json")
    require(rights["training_admitted_rows"] == 0 and rights["admitted_source_record_ids"] == [],
            "admitted corpus is not empty")
    counts = corpus["counts"]
    require(corpus["materialized"] is False and counts["admitted_source_rows"] == 0 and
            counts["materialized_source_rows"] == 0 and counts["materialized_derived_artifacts"] == 0 and
            counts["candidate_source_rows_excluded_by_rights"] == 7173,
            "corpus is not the sealed empty corpus")
    require(leakage["passed"] is True and leakage["violations"] == [] and
            leakage["audit_scope"]["assignment_count"] == 0, "leakage result drifted")
    require(tuple(split["partition_names"]) == PARTITION_NAMES, "twenty partition names drifted")
    require(split["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT and
            split["hidden_test_commitment_status"] == "unchanged_inherited",
            "hidden-test commitment drifted")
    table = acceptance["partition_protection_table"]
    require(len(table) == 20 and tuple(row["partition"] for row in table) == PARTITION_NAMES,
            "twenty-partition table drifted")
    protected_axes = [row for row in table if row["protection_state"] == "protected_holdout_axis"]
    require(len(protected_axes) == 15, "exactly fifteen holdout axes must remain protected")
    require(all(row["non_hidden"] is False for row in protected_axes),
            "a protected holdout axis was called non-hidden")
    require(table[0] == {"partition": "train", "protection_state": "unprotected_training_partition",
                         "non_hidden": True, "assigned_rows": 0, "status": "empty_not_run"},
            "train partition protection state drifted")
    special = {"validation", "canary", "holdout", "external_test"}
    require(all(row["protection_state"] == "protected_evaluation_partition" and
                row["non_hidden"] is False and row["assigned_rows"] == 0
                for row in table if row["partition"] in special),
            "protected evaluation partition state drifted")
    inverse_axes = {partition: key for key, partition in HOLDOUT_PARTITIONS.items()}
    for row in protected_axes:
        key = inverse_axes[row["partition"]]
        expected = split["holdouts"][key]
        reported = holdout_report["holdouts"].get(key)
        require(row == {
            "partition": row["partition"], "protection_state": "protected_holdout_axis",
            "non_hidden": False, "assigned_rows": 0, "status": "permanent_no_go",
            "holdout_key": key, "reason_code": "no_rights_admitted_materialized_rows",
        }, f"partition table row drifted for {key}")
        require(expected["count"] == 0 and expected["status"] == "permanent_no_go" and
                expected["permanent_no_go_reason"] == "no_rights_admitted_materialized_rows",
                f"split holdout replay drifted for {key}")
        require(reported is not None and reported["count"] == 0 and
                reported["status"] == "permanent_no_go" and
                reported["permanent_no_go_reason"] == "no_rights_admitted_materialized_rows",
                f"holdout report replay drifted for {key}")
    require(tuple(holdout_report["in_scope_holdouts"]) == HOLDOUT_REPORT_KEYS,
            "holdout catalog drifted")


def verify_forest(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    forest = acceptance["forest"]
    outer = forest["outer_commits"]
    require(len(outer) == 4, "outer forest must contain predecessor, implementation, merge, completion")
    require(tuple(record["role"] for record in outer) ==
            ("predecessor", "implementation", "merge", "completion"),
            "outer forest role order drifted")
    for record in outer:
        commit = record["commit"]
        require(run_git(ROOT, "cat-file", "-t", commit) == "commit", f"{commit} is not a commit")
        require(run_git(ROOT, "rev-parse", f"{commit}^{{tree}}") == record["tree"],
                f"outer tree drifted for {commit}")
        require(run_git(ROOT, "show", "-s", "--format=%P", commit).split() == record["parents"],
                f"outer parents drifted for {commit}")
        link = run_git(ROOT, "ls-tree", commit, "ipfs_datasets_py").split()
        require(len(link) == 4 and link[:3] == ["160000", "commit", record["datasets_gitlink"]],
                f"outer gitlink drifted for {commit}")
    predecessor = outer[0]
    require(predecessor == {
        "role": "predecessor", "commit": MISLABELLED_OUTER_COMMIT,
        "tree": MISLABELLED_OUTER_TREE, "parents": ["249e7fcac0d8e6e6baa0034ee4bb5b24034c74f5"],
        "datasets_gitlink": PREDECESSOR_DATASETS_COMMIT,
    }, "legacy object correction drifted")
    require(outer[1]["parents"] == [MISLABELLED_OUTER_COMMIT] and
            outer[2]["parents"] == ["b87bc6d28fbfefd836696f972f51ee9a677b5071", outer[1]["commit"]] and
            outer[3]["parents"] == [outer[2]["commit"]],
            "implementation, merge, and completion topology drifted")
    nested = forest["datasets_commits"]
    require(len(nested) == 2, "nested forest must bind predecessor and PGIR-204 commit")
    require(tuple(record["role"] for record in nested) ==
            ("pgir_202_predecessor", "pgir_204_nested_result"), "nested forest role order drifted")
    for record in nested:
        commit = record["commit"]
        require(run_git(DATASETS, "cat-file", "-t", commit) == "commit", f"nested {commit} is not a commit")
        require(run_git(DATASETS, "rev-parse", f"{commit}^{{tree}}") == record["tree"],
                f"nested tree drifted for {commit}")
        require(run_git(DATASETS, "show", "-s", "--format=%P", commit).split() == record["parents"],
                f"nested parents drifted for {commit}")
    require(nested[-1]["commit"] == NESTED_RESULT_COMMIT and
            nested[-1]["parents"] == [PREDECESSOR_DATASETS_COMMIT],
            "PGIR-204 nested parent chain drifted")
    require(all(record["datasets_gitlink"] == NESTED_RESULT_COMMIT for record in outer[1:]),
            "implementation/merge/completion gitlinks do not bind PGIR-204")
    current_link = run_git(ROOT, "ls-tree", "HEAD", "ipfs_datasets_py").split()
    require(len(current_link) == 4 and current_link[:3] == ["160000", "commit", NESTED_RESULT_COMMIT],
            "current outer checkout does not retain the PGIR-204 gitlink")
    require(run_git(DATASETS, "rev-parse", "HEAD") == NESTED_RESULT_COMMIT,
            "nested checkout differs from the sealed PGIR-204 gitlink")
    return {"outer_commit_count": 4, "nested_commit_count": 2,
            "corrected_legacy_commit": MISLABELLED_OUTER_COMMIT,
            "corrected_legacy_tree": MISLABELLED_OUTER_TREE}


def portability_outcome() -> dict[str, Any]:
    refs = run_git(DATASETS, "for-each-ref", "--contains", NESTED_RESULT_COMMIT,
                   "--format=%(refname)", "refs/remotes").splitlines()
    remote_refs = [ref for ref in refs if ref and not ref.endswith("/HEAD")]
    if remote_refs:
        return {"status": "replayed", "datasets_commit": NESTED_RESULT_COMMIT,
                "remote_refs": remote_refs, "pgir_205_execution_authorized": False}
    return {"status": "portability_no_go", "blocker_type": "unpublished_ref_portability_no_go",
            "datasets_commit": NESTED_RESULT_COMMIT, "remote_refs": [],
            "pgir_205_execution_authorized": False,
            "detail": "No remote-tracking ref contains the sealed nested commit; fresh recursive portability is not asserted."}


def verify() -> dict[str, Any]:
    acceptance = strict_json(ACCEPTANCE_PATH)
    require(acceptance["schema"] == "proof-grounded-ir-learning/successor-baseline-acceptance/v1",
            "wrong acceptance schema")
    require(acceptance["task_id"] == "PGIR-209" and
            acceptance["result_identity"] == "RESULT(PGIR-209)", "wrong task identity")
    require(acceptance["decision"] == "baseline_retired_fail_closed" and
            acceptance["completion_authoritative"] is False and
            acceptance["pgir_205_execution_authorized"] is False, "unsafe acceptance decision")
    verify_acceptance_identity(acceptance)
    verify_payloads(acceptance)
    verify_input_closure(acceptance)
    verify_partitions(acceptance)
    forest = verify_forest(acceptance)
    portability = portability_outcome()
    return {"schema": "proof-grounded-ir-learning/successor-baseline-verification@1",
            "verified": True, "task_id": "PGIR-209", "decision": "baseline_retired_fail_closed",
            "payload_count": 7, "partition_count": 20, "protected_holdout_axis_count": 15,
            "forest": forest, "portability": portability,
            "pgir_205_execution_authorized": False}


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    try:
        outcome = verify()
    except (OSError, BaselineAcceptanceError, ValueError, KeyError, TypeError) as exc:
        outcome = {"schema": "proof-grounded-ir-learning/successor-baseline-verification@1",
                   "verified": False, "task_id": "PGIR-209",
                   "error_type": "baseline_acceptance_verification_error", "error": str(exc),
                   "pgir_205_execution_authorized": False}
        print(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
