#!/usr/bin/env python3
"""Fail-closed, portable re-adjudication of the PGIR-208 source-chain seal.

PGIR-210 accepts the PGIR-202 nested commit as an immutable ancestor of the
PGIR-204 descendant.  It does not admit data, materialize a corpus, open
hidden tests, or authorize PGIR-205.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "ipfs_datasets_py"
V1 = ROOT / "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance/source_chain_acceptance.json"
V2_DIR = ROOT / "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance-v2"
RECEIPT = V2_DIR / "source_chain_acceptance_v2.json"
NETWORK_RECEIPT = V2_DIR / "network_replay_receipt.json"
VERIFICATION_RECEIPT = V2_DIR / "verification_receipt.json"
SEALED = "8736a0023d5d3afe4d0e5b044a3e4480966a8bf7"
CURRENT = "2a06dfe8546cdde78ff6d101a94708be0e6bf6e6"
HIDDEN = "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
HOLDOUTS = ("compiler", "cross_reference", "domain", "exception", "length", "lineage", "notation", "premise", "proof_library", "publication", "rare_operator", "time", "type")
TASKS = ("PGIR-200",) * 4 + ("PGIR-201",) * 5 + ("PGIR-202",) * 5
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class SourceChainV2Error(ValueError):
    """A mandatory immutable identity or no-go invariant did not replay."""


def require(value: bool, message: str) -> None:
    if not value:
        raise SourceChainV2Error(message)


def validate(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise SourceChainV2Error(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has non-string key")
        for key, item in value.items():
            validate(item, f"{path}.{key}")
        return
    raise SourceChainV2Error(f"{path} has unsupported {type(value).__name__}")


def canonical(value: Any) -> bytes:
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def raw_cid(data: bytes) -> str:
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + hashlib.sha256(data).digest()).decode().rstrip("=").lower()


def dag_cid(value: Any) -> str:
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + hashlib.sha256(canonical(value)).digest()).decode().rstrip("=").lower()


def read_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SourceChainV2Error(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=pairs,
                              parse_float=lambda value: (_ for _ in ()).throw(SourceChainV2Error(f"float {value} in {path}")),
                              parse_constant=lambda value: (_ for _ in ()).throw(SourceChainV2Error(f"non-finite number {value} in {path}")))
    except OSError as exc:
        raise SourceChainV2Error(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SourceChainV2Error(f"invalid JSON in {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain an object")
    validate(value, str(path))
    return value


def git(repository: Path, *args: str) -> str:
    try:
        process = subprocess.run(("git", "-C", str(repository), *args), text=True,
                                 encoding="utf-8", capture_output=True, timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceChainV2Error(f"git {' '.join(args)} unavailable: {exc}") from exc
    if process.returncode:
        raise SourceChainV2Error(f"git {' '.join(args)} failed: {process.stderr.strip() or process.stdout.strip() or 'no diagnostic'}")
    return process.stdout.strip()


def git_blob(repository: Path, revision_path: str) -> bytes:
    process = subprocess.run(("git", "-C", str(repository), "show", revision_path), capture_output=True,
                             timeout=45, check=False)
    if process.returncode:
        raise SourceChainV2Error(f"historical blob unavailable: {revision_path}")
    return process.stdout


def verify_record(record: Mapping[str, Any], revision: str) -> None:
    path = ROOT / str(record["path"])
    require(path.is_file(), f"missing bound input {record['path']}")
    data = path.read_bytes()
    require(len(data) == record["size_bytes"], f"size drifted for {record['path']}")
    require("sha256:" + hashlib.sha256(data).hexdigest() == record["sha256"], f"sha256 drifted for {record['path']}")
    require(raw_cid(data) == record["raw_cid"], f"raw CID drifted for {record['path']}")
    if str(record["path"]).startswith("ipfs_datasets_py/"):
        repository, relative = DATASETS, str(record["path"])[len("ipfs_datasets_py/"):]
    else:
        repository, relative = ROOT, str(record["path"])
    require(data == git_blob(repository, f"{revision}:{relative}"), f"Git blob differs for {record['path']}")


def tree_gitlink(commit: str) -> str:
    fields = git(ROOT, "ls-tree", commit, "ipfs_datasets_py").split()
    require(len(fields) == 4 and fields[0] == "160000", f"missing nested gitlink in {commit}")
    return fields[2]


def verify_v1_and_records(receipt: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = receipt["superseded_pg_208"]
    raw = V1.read_bytes()
    require("sha256:" + hashlib.sha256(raw).hexdigest() == predecessor["source_sha256"], "PGIR-208 receipt bytes drifted")
    require(raw_cid(raw) == predecessor["source_raw_cid"], "PGIR-208 receipt CID drifted")
    v1 = read_json(V1)
    projection = {key: value for key, value in v1.items() if key not in {"acceptance_sha256", "acceptance_cid"}}
    require(v1["acceptance_sha256"] == "sha256:" + hashlib.sha256(canonical(projection)).hexdigest(), "PGIR-208 canonical digest drifted")
    require(v1["acceptance_cid"] == dag_cid(projection), "PGIR-208 canonical CID drifted")
    require((v1["task_id"], v1["result_identity"], len(v1["payloads"]), len(v1["historical_inputs"])) == ("PGIR-208", "RESULT(PGIR-208)", 14, 10), "PGIR-208 closure shape drifted")
    require(tuple(item["task_id"] for item in v1["payloads"]) == TASKS, "sealed task topology drifted")
    for record in v1["payloads"]:
        verify_record(record, SEALED)
    for record in v1["historical_inputs"]:
        verify_record(record, SEALED if str(record["path"]).startswith("ipfs_datasets_py/") else "b87bc6d28fbfefd836696f972f51ee9a677b5071")
    return v1


def verify_population_and_results() -> None:
    corpus_dir = DATASETS / "data/ir_learning/corpora/successor-v1"
    split_dir = DATASETS / "data/ir_learning/splits/successor-v1"
    rights, quarantine, releases = (read_json(corpus_dir / name) for name in ("rights_manifest.json", "quarantine_manifest.json", "source_releases.json"))
    manifest, counts, lineage, load, root = (read_json(corpus_dir / name) for name in ("corpus_manifest.json", "count_receipt.json", "lineage_graph.json", "load_receipt.json", "corpus_root.json"))
    holdout, leakage, split_manifest, split_replay, split_root = (read_json(split_dir / name) for name in ("holdout_report.json", "leakage_report.json", "ir_split_manifest.json", "replay_receipt.json", "split_root.json"))
    ranges = quarantine["row_dispositions"]
    require(len(ranges) == 2, "quarantine requires two population ranges")
    seen: set[str] = set()
    for row in ranges:
        first, last = row["record_id_range"]["first"], row["record_id_range"]["last"]
        expanded = {row["record_id_format"] % index for index in range(first, last + 1)}
        require(len(expanded) == row["row_count"] and not (seen & expanded), "quarantine range is not unique and exhaustive")
        seen |= expanded
    require(len(seen) == 7173 and sorted(row["row_count"] for row in ranges) == [2174, 4999], "7173 is not independently reconstructed")
    require(rights["training_admitted_rows"] == 0 and rights["admitted_source_record_ids"] == [] and rights["quarantined_source_record_count"] == len(seen), "rights admission drifted")
    for value in (manifest, counts, root):
        nested_counts = value["counts"] if "counts" in value else value
        require(nested_counts["admitted_source_rows"] == 0 and nested_counts["materialized_source_rows"] == 0 and nested_counts["materialized_derived_artifacts"] == 0, "corpus result is not empty")
    require(manifest["result_identity"] == root["result_identity"] == "RESULT(PGIR-201)", "corpus result identity drifted")
    require(load["result_identity"] == "RESULT(PGIR-201)" and load["no_go"]["result_identity"] == "RESULT(PGIR-200)", "corpus replay result link drifted")
    for name, identity in root["artifacts"].items():
        data = (corpus_dir / name).read_bytes()
        require(identity["path"] == name and identity["size_bytes"] == len(data), f"corpus artifact size/link drifted: {name}")
        require(identity["sha256"] == hashlib.sha256(data).hexdigest() and identity["content_cid"] == raw_cid(data), f"corpus artifact identity drifted: {name}")
    require(root["manifest_cid"] == root["artifacts"]["corpus_manifest.json"]["content_cid"] and root["lineage_graph_cid"] == root["artifacts"]["lineage_graph.json"]["content_cid"], "corpus root CID links drifted")
    require(lineage["admitted_lineage_groups"] == lineage["edges"] == lineage["materialized_row_lineage"] == [], "lineage is not empty")
    require(root["materialized"] is False and root["materialized_source_record_ids"] == [], "corpus materialization drifted")
    require(leakage["passed"] is True and leakage["violations"] == [] and split_manifest["assignments"] == {}, "leakage or assignment drifted")
    require(split_root["result_identity"] == "RESULT(PGIR-202)" and split_root["split_manifest_digest"] == split_manifest["split_manifest_digest"], "split result link/digest drifted")
    require(split_root["split_manifest_sha256"] == hashlib.sha256((split_dir / "ir_split_manifest.json").read_bytes()).hexdigest(), "split manifest hash drifted")
    require(split_replay["result_identity"] == "RESULT(PGIR-202)" and split_replay["input_corpus"]["result_identity"] == "RESULT(PGIR-201)", "split replay result link drifted")
    require(split_manifest["samples_by_split"] == {name: [] for name in split_manifest["partition_names"]}, "split assignments are not exactly empty")
    for document in (holdout, split_manifest, split_root):
        require(document["hidden_test_commitment"] == HIDDEN and document.get("hidden_test_commitment_status") == "unchanged_inherited", "hidden-test commitment drifted")
    require(tuple(holdout["in_scope_holdouts"]) == HOLDOUTS, "thirteen named holdouts drifted")
    for name in HOLDOUTS:
        for document in (holdout, split_manifest, split_root):
            row = document["holdouts"][name]
            require(row["count"] == 0 and row["status"] == "permanent_no_go" and row["permanent_no_go_reason"] == "no_rights_admitted_materialized_rows", f"holdout no-go drifted: {name}")
    require(len(releases["releases"]) == 21 and all(row["training_admitted_rows"] == 0 for row in releases["releases"]), "source release closure drifted")


def verify_campaign_root_recursive_inputs() -> None:
    """Replay every file record recursively bound by the historical campaign root."""
    inventory = read_json(DATASETS / "data/ir_learning/source_inventory/release_inventory.json")
    require(inventory["counts"]["repository_count"] == 21 and inventory["counts"]["inventory_candidate_source_rows"] == 7173 and inventory["counts"]["training_admitted_source_rows"] == 0, "historical inventory/reconciliation drifted")
    campaign = read_json(ROOT / "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json")
    require(campaign["schema"] == "IRCampaignInputRoot@1" and campaign["bindings"]["corpus"]["source_count"] == 7173 and campaign["bindings"]["rights"]["training_admitted_rows"] == 0, "campaign root lineage/reconciliation drifted")
    checked = 0
    for binding in campaign["bindings"].values():
        records = ([binding["file"]] if "file" in binding else []) + binding.get("files", [])
        for record in records:
            if "git_blob" in record:
                repository = DATASETS if record["repository"] == "ipfs_datasets_py" else ROOT
                blob = record["git_blob"]
                require(git(repository, "cat-file", "-t", blob) == "blob", f"campaign binding is not a blob: {record['path']}")
                data = git_blob(repository, blob)
            else:
                require(record["repository"] == "pgir-freeze", f"untyped campaign binding: {record['path']}")
                data = (ROOT / record["path"]).read_bytes()
            require(len(data) == record["size_bytes"] and "sha256:" + hashlib.sha256(data).hexdigest() == record["sha256"] and raw_cid(data) == record["raw_cid"], f"campaign binding identity drifted: {record['path']}")
            checked += 1
    require(checked > 0, "campaign root contained no recursively bound inputs")


def verify_forest(receipt: Mapping[str, Any]) -> None:
    forest = receipt["forest"]
    cas = forest["compare_and_swap"]
    require(cas["decision"] == "adjudicated_serial_three_task_chain" and len(cas["tasks"]) == 3, "CAS topology drifted")
    expected_tasks = ("PGIR-200", "PGIR-201", "PGIR-202")
    for expected, row in zip(expected_tasks, cas["tasks"], strict=True):
        require(row["task_id"] == expected, "CAS task role order drifted")
        for key in ("implementation", "merge", "completion"):
            commit = row[key]["commit"]
            require(COMMIT.fullmatch(commit) is not None, f"invalid {key} commit")
            require(git(ROOT, "rev-parse", f"{commit}^{{tree}}") == row[key]["tree"], f"{key} tree drifted")
            require(git(ROOT, "show", "-s", "--format=%P", commit).split() == row[key]["parents"], f"{key} parent chain drifted")
        require(tree_gitlink(row["implementation"]["parents"][0]) == row["old_gitlink"], f"CAS old gitlink drifted: {expected}")
        require(tree_gitlink(row["implementation"]["commit"]) == row["new_gitlink"], f"CAS new gitlink drifted: {expected}")
        require(row["merge"]["tree"] == row["implementation"]["tree"], f"merge tree inequality: {expected}")
        require(row["merge"]["parents"] == [row["implementation"]["parents"][0], row["implementation"]["commit"]], f"merge topology drifted: {expected}")
        require(row["completion"]["parents"] == [row["merge"]["commit"]], f"completion parent drifted: {expected}")
    nested = forest["nested"]
    require(git(DATASETS, "merge-base", "--is-ancestor", SEALED, CURRENT) == "", "PGIR-202 is not ancestor of PGIR-204 gitlink")
    require(git(DATASETS, "show", "-s", "--format=%P", CURRENT).split() == [SEALED], "intervening nested parent chain drifted")
    require(git(DATASETS, "rev-parse", f"{SEALED}^{{tree}}") == nested["sealed_tree"], "sealed nested tree drifted")
    require(git(DATASETS, "rev-parse", f"{CURRENT}^{{tree}}") == nested["current_tree"], "current nested tree drifted")
    for record in nested["sealed_paths"]:
        relative = record["path"]
        require(git_blob(DATASETS, f"{SEALED}:{relative}") == git_blob(DATASETS, f"{CURRENT}:{relative}"), f"PGIR-204 changed sealed path {relative}")
    require(tree_gitlink("HEAD") == CURRENT and git(DATASETS, "rev-parse", "HEAD") == CURRENT, "current recursive checkout gitlink drifted")
    observed = forest["outer_observation"]
    require(git(ROOT, "merge-base", "--is-ancestor", observed["commit"], "HEAD") == "", "current outer checkout no longer descends from observation")
    require(git(ROOT, "rev-parse", f"{observed['commit']}^{{tree}}") == observed["tree"], "current outer tree identity drifted")


def verify_citations(v1: Mapping[str, Any], network: bool) -> dict[str, Any]:
    releases = read_json(DATASETS / "data/ir_learning/corpora/successor-v1/source_releases.json")["releases"]
    closure = {entry["release_id"]: entry for entry in v1["citation_closure"]}
    recorded = read_json(NETWORK_RECEIPT)
    require(recorded["network_execution_required"] is True and recorded["offline_replay_permitted"] is False, "network receipt permits offline replay")
    require(len(closure) == len(releases) == recorded["response_count"] == 21, "citation count drifted")
    require(recorded["response_hashes"] == {release_id: entry["response_sha256"] for release_id, entry in closure.items()}, "network receipt response hashes drifted")
    for release in releases:
        expected = closure.get(release["id"])
        require(expected is not None and release["revision"] == expected["revision"] == release["citation"]["observed_revision"], f"citation revision drifted: {release['id']}")
        require(expected["response_sha256"] == "sha256:" + release["citation"]["response_sha256"], f"citation source hash drifted: {release['id']}")
    if not network:
        return {"requested": False, "verified_count": 0, "sealed_count": 21}
    for release in releases:
        expected = closure[release["id"]]
        request = urllib.request.Request(release["citation"]["url"], headers={"Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise SourceChainV2Error(f"citation network replay failed for {release['id']}: {exc}") from exc
        require(len(body) == expected["response_size_bytes"], f"citation size drifted: {release['id']}")
        require("sha256:" + hashlib.sha256(body).hexdigest() == expected["response_sha256"], f"citation response hash drifted: {release['id']}")
        require(raw_cid(body) == expected["response_raw_cid"], f"citation response CID drifted: {release['id']}")
        require(json.loads(body).get("sha") == release["revision"], f"citation response revision drifted: {release['id']}")
    return {"requested": True, "verified_count": 21, "sealed_count": 21}


def verify_receipt(receipt: Mapping[str, Any]) -> None:
    require(receipt["schema"] == "proof-grounded-ir-learning/successor-source-chain-acceptance/v2", "wrong v2 schema")
    require((receipt["task_id"], receipt["result_identity"], receipt["decision"]) == ("PGIR-210", "RESULT(PGIR-210)", "superseding_permanent_no_go"), "unsafe v2 identity")
    require(receipt["completion_authoritative"] is False and receipt["pgir_205_execution_authorized"] is False, "execution authorization claimed")
    require(receipt["source_population"] == {"candidate_source_rows": 7173, "training_admitted_rows": 0, "materialized_source_rows": 0, "materialized_derived_artifacts": 0, "historical_derived_artifacts_observational_only": 38690}, "source population drifted")
    portability = receipt["portability_no_go"]
    require(portability["status"] == "portability_no_go" and portability["pgir_205_execution_authorized"] is False and len(portability["missing_outer_commits"]) > 0 and len(portability["missing_nested_commits"]) > 0, "portable checkout blocker is not typed and fail-closed")
    projection = {key: value for key, value in receipt.items() if key not in {"acceptance_sha256", "acceptance_cid"}}
    require(receipt["acceptance_sha256"] == "sha256:" + hashlib.sha256(canonical(projection)).hexdigest(), "v2 acceptance digest drifted")
    require(receipt["acceptance_cid"] == dag_cid(projection), "v2 acceptance CID drifted")
    source = ROOT / receipt["verifier_source_identity"]["path"]
    data = source.read_bytes()
    require("sha256:" + hashlib.sha256(data).hexdigest() == receipt["verifier_source_identity"]["sha256"], "verifier source identity drifted")
    require(raw_cid(data) == receipt["verifier_source_identity"]["raw_cid"], "verifier source CID drifted")
    verification = read_json(VERIFICATION_RECEIPT)
    require(verification["focused_test_count"] == 34 and verification["post_merge_verification"] is True and verification["pgir_205_execution_authorized"] is False, "test receipt is incomplete or unsafe")


def portability(receipt: Mapping[str, Any]) -> dict[str, Any]:
    blocker = receipt["portability_no_go"]
    outer = [commit for commit in blocker["missing_outer_commits"] if not git(ROOT, "for-each-ref", "--contains", commit, "--format=%(refname)", "refs/remotes")]
    nested = [commit for commit in blocker["missing_nested_commits"] if not git(DATASETS, "for-each-ref", "--contains", commit, "--format=%(refname)", "refs/remotes")]
    return {"status": "portability_no_go" if outer or nested else "publication_state_changed", "observation_method": blocker["observation_method"], "missing_outer_commits": outer, "missing_nested_commits": nested, "pgir_205_execution_authorized": False}


def verify(network: bool) -> dict[str, Any]:
    receipt = read_json(RECEIPT)
    verify_receipt(receipt)
    v1 = verify_v1_and_records(receipt)
    verify_population_and_results()
    verify_campaign_root_recursive_inputs()
    verify_forest(receipt)
    citations = verify_citations(v1, network)
    return {"schema": "proof-grounded-ir-learning/successor-source-chain-verification@2", "verified": True, "task_id": "PGIR-210", "decision": "superseding_permanent_no_go", "payload_count": 14, "historical_input_count": 10, "candidate_source_rows": 7173, "training_admitted_rows": 0, "materialized_source_rows": 0, "holdout_permanent_no_go_count": 13, "citations": citations, "portability": portability(receipt), "pgir_205_execution_authorized": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="store_true", help="perform all 21 exact-revision HTTPS replays; no offline replay is accepted")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.network), sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, SourceChainV2Error, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema": "proof-grounded-ir-learning/successor-source-chain-verification@2", "verified": False, "task_id": "PGIR-210", "error_type": "source_chain_v2_verification_error", "error": str(exc), "pgir_205_execution_authorized": False}, sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
