"""Fresh local finite cache qualification with native Python and Lean.

Use released accelerate/datasets packages on PYTHONPATH. --output must name a
new directory. Original authored source and all exact finite requests are saved.
--replay reopens the durable owners in a genuinely fresh process and rechecks.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
from ipfs_accelerate_py.agent_supervisor.proof import finite_checked_cache as checked
from ipfs_accelerate_py.agent_supervisor.proof import finite_cache_correspondence as keys
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseHead
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig

INPUTS = [-2, -1, 0, 1, 2]
SOURCE = b"def increment(n: int) -> int:\n    return n + 1\n"
LEAN = Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean")


def write(root, name, value):
    (root / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def owners(root):
    cx = duckdb.connect(str(root / "current.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    store = DuckDBASTStore(connection=cx)
    cas = ImmutableCAS(root / "cas")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=cas,
        catalog=CodebaseCatalog(store, cas))
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=root / ("admission-" + str(os.getpid()) + ".json"),
        proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192),
        lane_reservations={}, auto_renew_leases=False))
    cache = checked.FiniteCheckedCache(checked.FormalVerificationCache(root / "proof-cache"), cas)
    return cx, index, scheduler, cache


def arguments(root, request, index, scheduler, offset):
    return dict(index=index, repository=root / "repository",
        expected_head=CodebaseHead.from_dict(request["head"]),
        contract=IntegerOffsetContract("calc.py", "increment", "n", offset),
        inputs=request["inputs"], tool_policy=request["tool_policy"], scheduler=scheduler)


def replay(root):
    request = json.loads((root / "request.json").read_text())
    cx, index, scheduler, cache = owners(root)
    try:
        result = {}
        for name, offset in (("positive", 1), ("refuted", 2)):
            lookup = cache.lookup(owner_inputs=arguments(root, request, index, scheduler, offset))
            result[name] = {key: lookup[key] for key in ("status", "record_cid", "positive_reuse_eligible", "fresh_native_observation")}
        return {"process_id": os.getpid(), "results": result, "reopened_durable_DuckDB_and_CAS": True}
    finally:
        assert scheduler.snapshot()["active_lease_count"] == 0
        cx.close()


def export(root, name, result, cache):
    write(root, name + ".json", result)
    target = root / (name + "-artifacts")
    target.mkdir()
    bundle = cache.artifacts.get(result["evidence"]["bundle_cid"])
    write(target, "bundle.json", bundle)
    for key, cid in bundle["body_cids"].items():
        (target / checked.NAMES[key]).write_bytes(cache.artifacts.get_bytes(cid))


def qualify(root):
    root.mkdir(parents=True, exist_ok=False)
    repository = root / "repository"
    repository.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Authored finite cache fixture")
    git("config", "user.email", "fixture@example.invalid")
    (repository / "calc.py").write_bytes(SOURCE)
    git("add", "calc.py")
    git("commit", "-qm", "Authored bounded source")
    cx, index, scheduler, cache = owners(root)
    try:
        head = index.prepare_current(repository, repository_id="repository:rpi007-authored-finite",
            operation_id="initial", expected_head=None, scheduler=scheduler).head
        request = {"head": head.to_dict(), "inputs": INPUTS,
            "tool_policy": seal_finite_integer_tools(python_executable=Path(sys.executable), lean_executable=LEAN),
            "source_sha256": hashlib.sha256(SOURCE).hexdigest(), "source_provenance": "authored qualification fixture",
            "contracts": [IntegerOffsetContract("calc.py", "increment", "n", value).to_dict() for value in (1, 2)]}
        write(root, "request.json", request)
        results = {}
        for name, offset in (("positive", 1), ("refuted", 2)):
            result = cache.check_and_store(owner_inputs=arguments(root, request, index, scheduler, offset))
            assert result["status"] == name and result["positive_reuse_eligible"] == (offset == 1)
            results[name] = result
            export(root, name, result, cache)
        positive_args = arguments(root, request, index, scheduler, 1)
        duplicate = cache.check_and_store(owner_inputs=positive_args)
        assert duplicate["duplicate_publication"] and duplicate["record_cid"] == results["positive"]["record_cid"]
        write(root, "duplicate.json", duplicate)
        refs = cache.dependents("source", results["positive"]["evidence"]["correspondence"]["materials"]["source"]["source_cid"])
        assert len(refs) == 2 and all(not row["positive_reuse_eligible"] for row in refs)
        write(root, "reverse-dependencies.json", refs)
    finally:
        assert scheduler.snapshot()["active_lease_count"] == 0
        cx.close()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay", str(root)],
                           check=True, capture_output=True, text=True, timeout=90)
    replayed = json.loads(child.stdout)
    assert replayed["process_id"] != os.getpid()
    for name in results:
        assert replayed["results"][name]["record_cid"] == results[name]["record_cid"]
        assert replayed["results"][name]["positive_reuse_eligible"] == (name == "positive")
    assert replayed["results"]["positive"]["fresh_native_observation"]
    assert not replayed["results"]["refuted"]["fresh_native_observation"]
    write(root, "restart.json", replayed)
    sources = []
    for name in sorted(set(keys.PRODUCERS + checked.EXECUTION_PRODUCERS)):
        path = Path(importlib.import_module(name).__file__).resolve()
        repo = "datasets" if "/ipfs_datasets_py/" in str(path) else "accelerate"
        prefix = "/ipfs_datasets_py/" if repo == "datasets" else "/ipfs_accelerate_py/"
        relative = prefix.lstrip("/") + str(path).split(prefix)[1]
        sources.append({"repository": repo, "module": name, "path": relative,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write(root, "producer-sources.json", {"sources": sources,
        "reproducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    summary = {"schema": "finite-checked-cache-qualification/v1", "status": "qualified_selected_profile",
        "profile": keys.PROFILE, "native_dimensions": 16, "finite_inputs_per_contract": len(INPUTS),
        "positive_records": 1, "refuted_records": 1, "positive_native_artifacts": len(checked.NAMES),
        "fresh_process_native_positive_recheck": True, "duplicate_publication_same_record": True,
        "reverse_dependencies": len(refs), "scope": checked.SCOPE, "model_mode": "off",
        "source_provenance": "authored fixture", "checker_call_savings_claimed": False,
        "source_runtime_semantics_verified": False, "execution_authority": False, "completion_authority": False}
    write(root, "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--replay", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.replay.resolve()) if args.replay else qualify(args.output.resolve()), sort_keys=True))
