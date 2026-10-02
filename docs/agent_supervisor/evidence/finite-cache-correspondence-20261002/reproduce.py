"""New local key/finite-observation qualification; no saved proof admission.

Run with the released accelerate/datasets packages on PYTHONPATH. --output must
be a new directory. The child replay opens the same durable DuckDB/CAS afresh.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
from ipfs_accelerate_py.agent_supervisor.proof import finite_cache_correspondence as keys
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
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
INSTRUCTION = ("Under python-integer-offset-finite@1, calc.py::increment(n) must return an exact int for inputs [-2,-1,0,1,2].\n"
               "Under python-integer-offset-finite@1, calc.py::increment(n) must return n + 2 for inputs [-2,-1,0,1,2].")
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
    return cx, index, scheduler


def replay(root):
    request = json.loads((root / "request.json").read_text())
    report = json.loads((root / "correspondence.json").read_text())
    cx, index, scheduler = owners(root)
    try:
        native = keys.verify_finite_cache_correspondence(report, index=index,
            repository=root / "repository", expected_head=CodebaseHead.from_dict(request["head"]),
            contract=IntegerOffsetContract.from_dict(request["contract"]), inputs=request["inputs"],
            tool_policy=request["tool_policy"], scheduler=scheduler)
        return {"process_id": os.getpid(), "key_ids": [key.key_id for key in native],
                "reopened_durable_DuckDB_and_CAS": True, "positive_proof_admission": False}
    finally:
        assert scheduler.snapshot()["active_lease_count"] == 0
        cx.close()


def qualify(root):
    root.mkdir(parents=True, exist_ok=False)
    repository = root / "repository"
    repository.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Authored finite correspondence fixture")
    git("config", "user.email", "fixture@example.invalid")
    (repository / "calc.py").write_bytes(SOURCE)
    git("add", "calc.py")
    git("commit", "-qm", "Authored bounded source")
    cx, index, scheduler = owners(root)
    try:
        head = index.prepare_current(repository, repository_id="repository:rpi004-authored-finite",
            operation_id="initial", expected_head=None, scheduler=scheduler).head
        tools = seal_finite_integer_tools(python_executable=Path(sys.executable), lean_executable=LEAN)
        contract = IntegerOffsetContract("calc.py", "increment", "n", 2)
        report = keys.prepare_finite_cache_correspondence(index=index, repository=repository,
            expected_head=head, contract=contract, inputs=INPUTS, tool_policy=tools, scheduler=scheduler)
        write(root, "request.json", {"head": head.to_dict(), "contract": contract.to_dict(),
            "inputs": INPUTS, "tool_policy": tools, "source_sha256": hashlib.sha256(SOURCE).hexdigest(),
            "instruction": INSTRUCTION, "source_provenance": "authored qualification fixture"})
        write(root, "correspondence.json", report)
        matched = matcher.match_finite_integer_intent(index=index, repository=repository,
            repository_id=head.repository_id, expected_head=head, source_text=INSTRUCTION,
            intent_document=matcher.build_finite_integer_intent(INSTRUCTION),
            output=root / "observation", tool_policy=tools, scheduler=scheduler)
        write(root, "matcher.json", matched)
        observation = matched["observation"]
        material = report["materials"]
        joins = {
            "head": observation["head"] == material["source"]["head"],
            "source_cid": observation["source_cid"] == material["source"]["source_cid"],
            "source_sha256": observation["source_sha256"] == material["source"]["source_sha256"],
            "contract": observation["contract"] == material["obligation"]["contract"],
            "inputs": observation["domain_inputs"] == material["bounds"]["inputs"],
            "compilation": observation["compiled_cid"] == material["translation"]["compiled_cid"],
            "tool_policy": observation["tool_policy"] == material["policy"]["native_tools"],
        }
        assert all(joins.values())
        assert matched["status"] == "finite_counterexample" and len(matched["current_facts"]) == 1
        assert len(matched["finite_counterexamples"]) == len(INPUTS)
        assert observation["kernel_checked_model_table"] is True
    finally:
        assert scheduler.snapshot()["active_lease_count"] == 0
        cx.close()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay", str(root)],
                           check=True, capture_output=True, text=True, timeout=60)
    replayed = json.loads(child.stdout)
    assert replayed["process_id"] != os.getpid()
    assert replayed["key_ids"] == [report[name] for name in ("canonical_key_id", "execution_key_id", "bridged_key_id")]
    write(root, "restart.json", replayed)
    summary = {"schema": "finite-cache-correspondence-qualification/v1", "status": "qualified_selected_profile",
        "profile": keys.PROFILE, "native_dimensions": 16, "source_and_receipt_joins": joins,
        "finite_inputs": len(INPUTS), "finite_counterexamples": len(matched["finite_counterexamples"]),
        "bounded_type_facts": len(matched["current_facts"]), "fresh_Python_and_Lean_observation": True,
        "fresh_process_key_replay": True, "model_mode": "off", "source_provenance": "authored fixture",
        "scope": "key identity and explicit finite observation join; no positive proof admission",
        "key_preparation_scope": dict(keys.FALSE), "proof_authority": False, "execution_authority": False,
        "completion_authority": False}
    write(root, "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--replay", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.replay.resolve()) if args.replay else qualify(args.output.resolve()), sort_keys=True))
