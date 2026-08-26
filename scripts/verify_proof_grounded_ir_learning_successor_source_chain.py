#!/usr/bin/env python3
"""Fail-closed replay of the PGIR-200..202 successor source chain.

This verifier intentionally has no dependency on the corpus builder.  It binds
the sealed JSON bytes, their historical inputs, the exact-revision metadata
citations, and the nested Git history independently.  It is an evidence gate,
not an admission, materialization, training, or PGIR-205 execution authority.
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
ACCEPTANCE_PATH = (
    ROOT
    / "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1"
    / "source-chain-acceptance/source_chain_acceptance.json"
)
SOURCE_RELEASES_PATH = (
    ROOT
    / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/source_releases.json"
)
HIDDEN_TEST_COMMITMENT = (
    "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
)
HOLDOUTS = (
    "compiler", "cross_reference", "domain", "exception", "length", "lineage",
    "notation", "premise", "proof_library", "publication", "rare_operator", "time",
    "type",
)
EXPECTED_PAYLOAD_TASKS = (
    "PGIR-200", "PGIR-200", "PGIR-200", "PGIR-200", "PGIR-201", "PGIR-201",
    "PGIR-201", "PGIR-201", "PGIR-201", "PGIR-202", "PGIR-202", "PGIR-202",
    "PGIR-202", "PGIR-202",
)
COMMIT_HEX = re.compile(r"^[0-9a-f]{40}$")


class SourceChainError(ValueError):
    """A sealed identity or no-go invariant drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceChainError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise SourceChainError(f"{path} contains a float")
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_value(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has non-string key")
        for key, child in value.items():
            validate_value(child, f"{path}.{key}")
        return
    raise SourceChainError(f"{path} contains unsupported {type(value).__name__}")


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
                raise SourceChainError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=pairs,
                parse_float=lambda raw: (_ for _ in ()).throw(
                    SourceChainError(f"float {raw!r} in {path}")
                ),
                parse_constant=lambda raw: (_ for _ in ()).throw(
                    SourceChainError(f"non-finite number {raw!r} in {path}")
                ),
            )
    except OSError as exc:
        raise SourceChainError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SourceChainError(f"invalid JSON in {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain an object")
    validate_value(value, str(path))
    return value


def run_git(repository: Path, *args: str) -> str:
    try:
        process = subprocess.run(
            ("git", "-C", str(repository), *args), text=True, encoding="utf-8",
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceChainError(f"git {' '.join(args)} unavailable: {exc}") from exc
    if process.returncode:
        diagnostic = process.stderr.strip() or process.stdout.strip() or "no diagnostic"
        raise SourceChainError(f"git {' '.join(args)} failed: {diagnostic}")
    return process.stdout.strip()


def git_bytes(repository: Path, revision_path: str) -> bytes:
    try:
        process = subprocess.run(
            ("git", "-C", str(repository), "show", revision_path), capture_output=True,
            check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceChainError(f"cannot read historical blob {revision_path}: {exc}") from exc
    if process.returncode:
        raise SourceChainError(f"historical blob unavailable: {revision_path}")
    return process.stdout


def verify_identity_record(record: Mapping[str, Any], *, repository: Path, revision: str,
                           relative_to_repository: str) -> None:
    path = ROOT / str(record["path"])
    require(path.is_file(), f"missing sealed input {record['path']}")
    data = path.read_bytes()
    require(len(data) == record["size_bytes"], f"size drifted for {record['path']}")
    require("sha256:" + hashlib.sha256(data).hexdigest() == record["sha256"],
            f"sha256 drifted for {record['path']}")
    require(raw_cid(data) == record["raw_cid"], f"raw CID drifted for {record['path']}")
    require(data == git_bytes(repository, f"{revision}:{relative_to_repository}"),
            f"working bytes differ from sealed Git blob for {record['path']}")


def verify_acceptance_identity(acceptance: Mapping[str, Any]) -> None:
    projection = {key: value for key, value in acceptance.items()
                  if key not in {"acceptance_sha256", "acceptance_cid"}}
    expected_sha = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    require(acceptance.get("acceptance_sha256") == expected_sha,
            "acceptance_sha256 drifted")
    require(acceptance.get("acceptance_cid") == dag_json_cid(projection),
            "acceptance_cid drifted")


def verify_forest(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    forest = acceptance["forest"]
    for record in forest["outer_commits"]:
        commit = record["commit"]
        require(COMMIT_HEX.fullmatch(commit) is not None, "invalid outer commit identity")
        require(run_git(ROOT, "rev-parse", f"{commit}^{{tree}}") == record["tree"],
                f"outer tree drifted for {commit}")
        actual_parents = run_git(ROOT, "show", "-s", "--format=%P", commit).split()
        require(actual_parents == record["parents"], f"outer parents drifted for {commit}")
    for record in forest["datasets_commits"]:
        commit = record["commit"]
        require(COMMIT_HEX.fullmatch(commit) is not None, "invalid datasets commit identity")
        require(run_git(DATASETS, "rev-parse", f"{commit}^{{tree}}") == record["tree"],
                f"datasets tree drifted for {commit}")
        actual_parents = run_git(DATASETS, "show", "-s", "--format=%P", commit).split()
        require(actual_parents == record["parents"], f"datasets parents drifted for {commit}")

    cas = forest["submodule_compare_and_swap_adjudication"]
    require(cas["decision"] == "adjudicated_serial_chain", "CAS race not adjudicated")
    required_links = cas["required_outer_gitlinks"]
    require(len(required_links) == 3, "CAS adjudication must bind three outer gitlinks")
    for link in required_links:
        output = run_git(ROOT, "ls-tree", link["outer_commit"], "ipfs_datasets_py")
        fields = output.split()
        require(len(fields) == 4 and fields[0] == "160000" and fields[2] == link["datasets_commit"],
                f"outer gitlink mismatch for {link['outer_commit']}")
    require([link["datasets_commit"] for link in required_links] ==
            [item["commit"] for item in forest["datasets_commits"]],
            "CAS chain does not match nested history")
    require(run_git(ROOT, "merge-base", "--is-ancestor", "249e7fcac0d8e6e6baa0034ee4bb5b24034c74f5", "HEAD") == "",
            "PGIR-202 completion is not an ancestor of checkout")
    require(run_git(ROOT, "merge-base", "--is-ancestor", "b87bc6d28fbfefd836696f972f51ee9a677b5071", "HEAD") == "",
            "source-chain acceptance does not descend from the sealed successor base")
    current_gitlink = run_git(ROOT, "ls-tree", "HEAD", "ipfs_datasets_py").split()
    expected_head = forest["datasets_commits"][-1]["commit"]
    require(len(current_gitlink) == 4 and current_gitlink[2] == expected_head,
            "current outer gitlink is not the sealed PGIR-202 nested commit")
    require(run_git(DATASETS, "rev-parse", "HEAD") == expected_head,
            "checked-out datasets HEAD differs from outer gitlink")
    return {"outer_commit_count": len(forest["outer_commits"]),
            "datasets_commit_count": len(forest["datasets_commits"]),
            "cas_decision": cas["decision"]}


def verify_records(acceptance: Mapping[str, Any]) -> None:
    payloads = acceptance["payloads"]
    require(len(payloads) == 14, "exactly fourteen sealed payloads are required")
    require(tuple(item.get("task_id") for item in payloads) == EXPECTED_PAYLOAD_TASKS,
            "payload task order drifted")
    for record in payloads:
        path = str(record["path"])
        require(path.startswith("ipfs_datasets_py/"), "payload escapes datasets gitlink")
        verify_identity_record(record, repository=DATASETS,
                               revision="8736a0023d5d3afe4d0e5b044a3e4480966a8bf7",
                               relative_to_repository=path.removeprefix("ipfs_datasets_py/"))
    historical = acceptance["historical_inputs"]
    require(len(historical) == 10, "historical input closure drifted")
    for record in historical:
        path = str(record["path"])
        if path.startswith("ipfs_datasets_py/"):
            verify_identity_record(record, repository=DATASETS,
                                   revision="8736a0023d5d3afe4d0e5b044a3e4480966a8bf7",
                                   relative_to_repository=path.removeprefix("ipfs_datasets_py/"))
        else:
            verify_identity_record(record, repository=ROOT,
                                   revision="b87bc6d28fbfefd836696f972f51ee9a677b5071",
                                   relative_to_repository=path)


def verify_semantics(acceptance: Mapping[str, Any]) -> None:
    require(acceptance["schema"] == "proof-grounded-ir-learning/successor-source-chain-acceptance/v1",
            "wrong acceptance schema")
    require(acceptance["task_id"] == "PGIR-208", "wrong task identity")
    require(acceptance["result_identity"] == "RESULT(PGIR-208)", "wrong result identity")
    require(acceptance["decision"] == "sealed_permanent_no_go", "unsafe acceptance decision")
    require(acceptance["completion_authoritative"] is False, "completion authority claimed")
    require(acceptance["pgir_205_execution_authorized"] is False, "PGIR-205 was authorized")
    population = acceptance["source_population"]
    require(population == {"candidate_source_rows": 7173, "training_admitted_rows": 0,
                           "materialized_source_rows": 0, "materialized_derived_artifacts": 0,
                           "historical_derived_artifacts_observational_only": 38690},
            "source population drifted")
    holdout_policy = acceptance["holdout_policy"]
    require(holdout_policy["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT,
            "hidden-test commitment drifted")
    require(tuple(holdout_policy["required_holdouts"]) == HOLDOUTS, "holdout catalog drifted")
    require(holdout_policy["leakage_passed"] is True and holdout_policy["hidden_tests_opened"] is False,
            "unsafe leakage or hidden-test status")

    rights = strict_json(ROOT / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/rights_manifest.json")
    corpus = strict_json(ROOT / "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json")
    holdouts = strict_json(ROOT / "ipfs_datasets_py/data/ir_learning/splits/successor-v1/holdout_report.json")
    split = strict_json(ROOT / "ipfs_datasets_py/data/ir_learning/splits/successor-v1/split_root.json")
    leakage = strict_json(ROOT / "ipfs_datasets_py/data/ir_learning/splits/successor-v1/leakage_report.json")
    require(rights["training_admitted_rows"] == 0 and rights["admitted_source_record_ids"] == [],
            "rights admission drifted")
    require(corpus["materialized"] is False and corpus["counts"]["admitted_source_rows"] == 0 and
            corpus["counts"]["materialized_source_rows"] == 0, "corpus materialization drifted")
    require(leakage["passed"] is True and leakage["violations"] == [], "leakage audit drifted")
    require(split["hidden_test_commitment"] == HIDDEN_TEST_COMMITMENT and
            split["hidden_test_commitment_status"] == "unchanged_inherited", "hidden-test replay drifted")
    require(tuple(holdouts["in_scope_holdouts"]) == HOLDOUTS, "holdout report order drifted")
    for name in HOLDOUTS:
        row = holdouts["holdouts"][name]
        require(row["count"] == 0 and row["status"] == "permanent_no_go" and
                row["permanent_no_go_reason"] == "no_rights_admitted_materialized_rows",
                f"holdout {name} is not a sealed permanent no-go")


def verify_citations(acceptance: Mapping[str, Any], *, network: bool) -> dict[str, Any]:
    releases = strict_json(SOURCE_RELEASES_PATH)["releases"]
    closure = acceptance["citation_closure"]
    require(len(releases) == 21 and len(closure) == 21, "exactly 21 citations are required")
    by_id = {item["release_id"]: item for item in closure}
    require(len(by_id) == 21, "duplicate citation closure release id")
    for release in releases:
        citation = release["citation"]
        expected = by_id.get(release["id"])
        require(expected is not None, f"missing citation closure for {release['id']}")
        require(release["revision"] == citation["observed_revision"] == expected["revision"],
                f"citation revision drifted for {release['id']}")
        require(citation["response_sha256"] == expected["response_sha256"].removeprefix("sha256:"),
                f"citation hash drifted for {release['id']}")
        require(citation["url"].endswith("/revision/" + release["revision"]),
                f"citation URL is not exact revision for {release['id']}")
    if not network:
        return {"requested": False, "verified_count": 0, "sealed_count": 21}
    for release in releases:
        expected = by_id[release["id"]]
        request = urllib.request.Request(release["citation"]["url"], headers={"Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise SourceChainError(f"citation network replay failed for {release['id']}: {exc}") from exc
        require(len(data) == expected["response_size_bytes"], f"citation size drifted for {release['id']}")
        require("sha256:" + hashlib.sha256(data).hexdigest() == expected["response_sha256"],
                f"citation bytes drifted for {release['id']}")
        require(raw_cid(data) == expected["response_raw_cid"], f"citation CID drifted for {release['id']}")
        try:
            response_json = json.loads(data)
        except json.JSONDecodeError as exc:
            raise SourceChainError(f"citation response is not JSON for {release['id']}") from exc
        require(response_json.get("sha") == release["revision"],
                f"citation response revision drifted for {release['id']}")
    return {"requested": True, "verified_count": 21, "sealed_count": 21}


def portability_outcome(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    """Report the only permitted non-terminal outcome for unavailable gitlinks.

    A local object proves replay in this checkout, but not that a fresh clone
    can fetch it.  Remote-tracking refs are merely evidence of publication; if
    none contains the sealed datasets commit we emit the explicit blocker.
    """
    datasets_commit = acceptance["forest"]["datasets_commits"][-1]["commit"]
    refs = run_git(DATASETS, "for-each-ref", "--contains", datasets_commit,
                   "--format=%(refname)", "refs/remotes").splitlines()
    remote_refs = tuple(ref for ref in refs if ref and not ref.endswith("/HEAD"))
    if remote_refs:
        return {"status": "replayed", "datasets_commit": datasets_commit,
                "remote_refs": list(remote_refs), "pgir_205_execution_authorized": False}
    return {"status": "blocked", "blocker_type": "unpublished_ref_portability_blocker",
            "datasets_commit": datasets_commit, "remote_refs": [],
            "pgir_205_execution_authorized": False,
            "detail": "No remote-tracking ref contains the sealed datasets gitlink; a clean recursive clone is not asserted portable."}


def verify(*, network: bool) -> dict[str, Any]:
    acceptance = strict_json(ACCEPTANCE_PATH)
    verify_acceptance_identity(acceptance)
    verify_semantics(acceptance)
    verify_records(acceptance)
    forest = verify_forest(acceptance)
    citations = verify_citations(acceptance, network=network)
    portability = portability_outcome(acceptance)
    return {"schema": "proof-grounded-ir-learning/successor-source-chain-verification@1",
            "verified": True, "task_id": "PGIR-208", "decision": "sealed_permanent_no_go",
            "payload_count": 14, "historical_input_count": 10, "candidate_source_rows": 7173,
            "training_admitted_rows": 0, "materialized_source_rows": 0,
            "holdout_permanent_no_go_count": 13, "hidden_test_commitment": HIDDEN_TEST_COMMITMENT,
            "citations": citations, "forest": forest, "portability": portability,
            "pgir_205_execution_authorized": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="store_true", help="replay all 21 exact-revision metadata citations")
    args = parser.parse_args(argv)
    try:
        outcome = verify(network=args.network)
    except (OSError, SourceChainError, ValueError, KeyError, TypeError) as exc:
        outcome = {"schema": "proof-grounded-ir-learning/successor-source-chain-verification@1",
                   "verified": False, "task_id": "PGIR-208", "error_type": "source_chain_verification_error",
                   "error": str(exc), "pgir_205_execution_authorized": False}
        print(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
