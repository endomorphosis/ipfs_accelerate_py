"""Real captured-source key replay; no checked-proof or execution promotion."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import finite_cache_correspondence as owner
from ipfs_accelerate_py.agent_supervisor.proof.canonical_cache_key_bridge import bridge_canonical_proof_cache_key
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    CacheRejectionReason, FormalVerificationCache, ProofCacheKey,
)
from ipfs_datasets_py.logic.common.canonical_cache_key import (
    REQUIRED_IDENTITY_FIELDS, CanonicalProofCacheKey, content_digest,
)
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as finite
from ipfs_datasets_py.logic.software_contracts import codebase_integer_profile as scalar
from ipfs_datasets_py.logic.software_contracts.cache import CacheIntegrityError
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError,
)
from test.api.test_agent_supervisor_formal_verification_cache import _receipt
from test.api.test_finite_integer_codebase import (  # native DuckDB/CAS owners and ELF tools
    finite_prepared, finite_tools, finite_source, finite_git, finite_match,
)


def arguments(prepared, tools, **changes):
    values = {name: prepared[name] for name in ("index", "repository", "expected_head", "scheduler")}
    values.update(contract=scalar.IntegerOffsetContract("calc.py", "increment", "n", 2),
                  inputs=[-2, -1, 0, 1, 2], tool_policy=deepcopy(tools))
    values.update(changes)
    return values


def resign(report):
    report["report_sha256"] = content_digest({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def test_native_owners_derive_all_sixteen_dimensions_and_replay(finite_prepared, finite_tools):
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    semantic, execution, bridged = owner.verify_finite_cache_correspondence(report, **args)
    assert type(semantic) is CanonicalProofCacheKey
    assert type(execution) is type(bridged) is ProofCacheKey
    assert semantic.to_dict() == report["canonical_key"]
    assert execution.to_dict() == report["execution_key"]
    assert bridged.to_dict() == report["bridged_key"]
    assert len(report["field_correspondence"]) == 16
    assert {row["canonical_dimension"] for row in report["field_correspondence"]} == set(REQUIRED_IDENTITY_FIELDS)
    assert report["report_sha256"] == content_digest({k: v for k, v in report.items() if k != "report_sha256"})
    for row in report["field_correspondence"]:
        material = report["materials"][row["material"]]
        for path in row["execution_fields"]:
            selected = execution.to_dict()
            for component in path.split("."):
                selected = selected[component]
            assert selected == material, (row, selected, material)
        stored = report["canonical_key"][row["canonical_dimension"]]
        assert stored == (material if row["encoding"] == "literal" else content_digest(material))
    assert report["model_mode"] == "off"
    assert report["materials"]["source"]["model"]["model_dependencies"] == []
    assert report["materials"]["source"]["source_sha256"] == hashlib.sha256(finite_source()).hexdigest()
    assert report["materials"]["formalization"]["compiled"]["body_offset"] == 1
    assert report["materials"]["obligation"]["contract"]["offset"] == 2
    assert report["source_observed_before_and_after"] is True
    assert all(report[name] is False for name in owner.FALSE)
    assert "obligation_id" not in execution.obligation
    assert "obligation_id" not in bridged.obligation


def test_semantic_premises_bounds_and_operational_limits_stay_distinct(finite_prepared, finite_tools):
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    materials = report["materials"]
    assert [row["kind"] for row in materials["assumptions"]].count("native_source_model_equation") == 1
    assert {row["text"] for row in materials["assumptions"] if row["kind"] == "declared_runtime_assumption"} == set(scalar.ASSUMPTIONS)
    assert materials["bounds"] == finite.build_finite_integer_domain(args["inputs"])
    altered_domain = owner.prepare_finite_cache_correspondence(**dict(args, inputs=[0, 1]))
    altered_budget = owner.prepare_finite_cache_correspondence(**dict(args, observation_limits={"timeout_seconds": 17, "memory_mb": 2048}))
    for key in ("canonical_key_id", "execution_key_id", "bridged_key_id"):
        assert len({report[key], altered_domain[key], altered_budget[key]}) == 3
    assert report["canonical_key"]["bounds"] == altered_budget["canonical_key"]["bounds"]
    assert report["canonical_key"]["policy"] != altered_budget["canonical_key"]["policy"]
    assert report["execution_key"]["resource_budget"] == altered_domain["execution_key"]["resource_budget"]
    assert report["canonical_key"]["assumptions"] == altered_domain["canonical_key"]["assumptions"]


@pytest.mark.parametrize("field", REQUIRED_IDENTITY_FIELDS)
def test_each_full_key_material_is_owner_derived_not_a_caller_assertion(finite_prepared, finite_tools, field):
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    # Even coherently rehashing all keys and the outer report cannot replace
    # source-derived native materials with caller-defined claims.
    report["materials"][field] = {"forged": field} if field not in {
        "provider", "checker", "evidence_kind", "authority_ceiling"
    } else ("different:tool" if field in {"provider", "checker"} else report["materials"][field])
    if field == "evidence_kind":
        report["materials"][field] = "llm_output"
    elif field == "authority_ceiling":
        report["materials"][field] = "advisory"
    if field == "assumptions":
        report["materials"][field] = [{"forged": field}]
    execution = deepcopy(report["execution_key"])
    for mapping in report["field_correspondence"]:
        for path in mapping["execution_fields"]:
            target = execution
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = deepcopy(report["materials"][mapping["material"]])
    semantic = CanonicalProofCacheKey.build(**report["materials"], source_cid=report["canonical_key"]["source_cid"])
    execution = ProofCacheKey.from_dict(execution)
    bridged = bridge_canonical_proof_cache_key(semantic, execution_key=execution)
    for label, key in (("canonical", semantic), ("execution", execution), ("bridged", bridged)):
        report[label + "_key"] = key.to_dict()
        report[label + "_key_id"] = key.key_id
    resign(report)
    with pytest.raises(owner.FiniteCacheCorrespondenceError, match="does not replay"):
        owner.verify_finite_cache_correspondence(report, **args)


@pytest.mark.parametrize("damage", ["missing_dimension", "unknown_field", "missing_mapping", "wrong_mapping",
                                    "model_dependency", "authority", "derived_key", "scope", "report_hash"])
def test_closed_replay_refuses_omissions_aliases_and_promotion(finite_prepared, finite_tools, damage):
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    if damage == "missing_dimension":
        del report["canonical_key"]["environment"]
    elif damage == "unknown_field":
        report["live_kernel_handle"] = "forged"
    elif damage == "missing_mapping":
        report["field_correspondence"].pop()
    elif damage == "wrong_mapping":
        report["field_correspondence"][5]["execution_fields"] = ["resource_budget"]
    elif damage == "model_dependency":
        report["materials"]["source"]["model"] = {"mode": "learned", "model_dependencies": ["unbound-checkpoint"]}
    elif damage == "authority":
        report["proof_authority"] = True
    elif damage == "derived_key":
        report["execution_key"]["obligation"]["obligation_id"] = "pretend-checked"
    elif damage == "scope":
        report["contract_satisfaction_checked"] = True
    else:
        report["report_sha256"] = "0" * 64
    if damage != "report_hash":
        resign(report)
    with pytest.raises(owner.FiniteCacheCorrespondenceError, match="does not replay"):
        owner.verify_finite_cache_correspondence(report, **args)


def test_no_source_solver_or_checker_execution_and_no_positive_cache_admission(finite_prepared, finite_tools, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("target source / solver / checker execution is not preparation")
    monkeypatch.setattr(finite.BoundedToolRunner, "run", forbidden)
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    _, execution, bridged = owner.verify_finite_cache_correspondence(report, **args)
    cache = FormalVerificationCache(tmp_path / "proof-cache")
    for key in (execution, bridged):
        result = cache.put(key, _receipt())
        assert not result.stored
        assert CacheRejectionReason.BINDING_MISMATCH.value in result.reason_codes
        assert not cache.lookup(key).hit
    reopened = FormalVerificationCache(tmp_path / "proof-cache")
    assert not reopened.lookup(bridged).hit


def test_changed_source_or_current_head_refuses_old_key_then_derives_new_key(finite_prepared, finite_tools):
    args = arguments(finite_prepared, finite_tools)
    original = owner.prepare_finite_cache_correspondence(**args)
    (args["repository"] / "calc.py").write_bytes(finite_source(2))
    with pytest.raises(StaleCodebaseError):
        owner.verify_finite_cache_correspondence(original, **args)
    finite_git(args["repository"], "add", "calc.py")
    finite_git(args["repository"], "commit", "-qm", "exact native successor source")
    head = args["index"].prepare_current(args["repository"], repository_id=finite_prepared["repository_id"],
        operation_id="successor", expected_head=args["expected_head"], scheduler=args["scheduler"]).head
    successor_args = dict(args, expected_head=head)
    successor = owner.prepare_finite_cache_correspondence(**successor_args)
    assert successor["materials"]["formalization"]["compiled"]["body_offset"] == 2
    assert original["canonical_key_id"] != successor["canonical_key_id"]
    assert original["execution_key_id"] != successor["execution_key_id"]
    with pytest.raises(owner.FiniteCacheCorrespondenceError, match="does not replay"):
        owner.verify_finite_cache_correspondence(original, **successor_args)


@pytest.mark.parametrize("bad_source", [
    b"import os\ndef increment(n: int) -> int:\n    return n + 1\n",
    b"def increment(n: int) -> int:\n    return n + global_offset\n",
    b"def increment(n: int) -> int:\n    return external(n)\n",
])
def test_open_source_dependencies_cannot_claim_closed_profile(finite_prepared, finite_tools, bad_source):
    args = arguments(finite_prepared, finite_tools)
    (args["repository"] / "calc.py").write_bytes(bad_source)
    finite_git(args["repository"], "add", "calc.py")
    finite_git(args["repository"], "commit", "-qm", "unsupported source fixture")
    args["expected_head"] = args["index"].prepare_current(args["repository"],
        repository_id=finite_prepared["repository_id"], operation_id="unsupported",
        expected_head=args["expected_head"], scheduler=args["scheduler"]).head
    with pytest.raises(scalar.UnsupportedIntegerProfile):
        owner.prepare_finite_cache_correspondence(**args)


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_replay_requires_exact_cas_source_not_warm_owner_state(finite_prepared, finite_tools, damage):
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    cas = args["index"].artifacts
    source_cid = report["materials"]["source"]["source_cid"]
    path = cas.path_for(source_cid, source=True)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(finite_source(99))
    with pytest.raises((CacheIntegrityError, FileNotFoundError)):
        owner.verify_finite_cache_correspondence(report, **args)


def test_changed_native_tool_or_cross_environment_is_not_accepted(finite_prepared, finite_tools, tmp_path):
    args = arguments(finite_prepared, finite_tools)
    python_copy = tmp_path / "python-elf"
    shutil.copyfile(finite_tools["python"]["path"], python_copy)
    args["tool_policy"] = finite.seal_finite_integer_tools(
        python_executable=python_copy, lean_executable=Path(finite_tools["lean"]["path"]))
    report = owner.prepare_finite_cache_correspondence(**args)
    python_copy.write_bytes(python_copy.read_bytes() + b"identity-change")
    with pytest.raises(finite.FiniteIntegerObservationError, match="pinned native tool changed"):
        owner.verify_finite_cache_correspondence(report, **args)
    args["tool_policy"] = finite.seal_finite_integer_tools(
        python_executable=python_copy, lean_executable=Path(finite_tools["lean"]["path"]))
    changed = owner.prepare_finite_cache_correspondence(**args)
    assert changed["canonical_key"]["environment"] != report["canonical_key"]["environment"]
    with pytest.raises(owner.FiniteCacheCorrespondenceError, match="does not replay"):
        owner.verify_finite_cache_correspondence(report, **args)
    policy = deepcopy(finite_tools)
    policy["environment"]["PYTHONPATH"] = "/foreign/overrides"
    policy["policy_cid"] = cid_for_structured({k: v for k, v in policy.items() if k != "policy_cid"})
    with pytest.raises(finite.FiniteIntegerObservationError, match="invalid finite native tool policy"):
        owner.prepare_finite_cache_correspondence(**dict(args, tool_policy=policy))


def test_producer_inventory_binds_native_owners_and_rejects_inflight_drift(finite_prepared, finite_tools, monkeypatch):
    args = arguments(finite_prepared, finite_tools)
    expected = owner._pins()
    for name in (scalar.__name__, finite.__name__,
                 "ipfs_datasets_py.logic.software_contracts.duckdb_ast_store",
                 "ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot",
                 "ipfs_datasets_py.logic.software_contracts.content"):
        assert expected[name] == hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
    calls = []
    def drift():
        calls.append(1)
        return expected if len(calls) == 1 else dict(expected, **{scalar.__name__: "0" * 64})
    monkeypatch.setattr(owner, "_pins", drift)
    with pytest.raises(owner.FiniteCacheCorrespondenceError, match="producing code changed"):
        owner.prepare_finite_cache_correspondence(**args)


def test_current_lowering_interpreter_and_inflight_source_are_bound(finite_prepared, finite_tools, monkeypatch):
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    current = report["materials"]["environment"]["compiler_python"]["binary"]
    assert current["path"] == str(Path(sys.executable).resolve())
    assert current["sha256"] == hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    compile_source = scalar.compile_integer_offset
    def changed_after_lowering(*values, **options):
        compiled = compile_source(*values, **options)
        (args["repository"] / "calc.py").write_bytes(finite_source(2))
        return compiled
    monkeypatch.setattr(scalar, "compile_integer_offset", changed_after_lowering)
    with pytest.raises(StaleCodebaseError):
        owner.prepare_finite_cache_correspondence(**args)


@pytest.mark.parametrize("bad_inputs", [[], [True], [1, 0], [0, 0], list(range(33)), [2**31 + 1]])
def test_native_domain_bounds_are_closed(finite_prepared, finite_tools, bad_inputs):
    with pytest.raises(finite.FiniteIntegerObservationError):
        owner.prepare_finite_cache_correspondence(**arguments(finite_prepared, finite_tools, inputs=bad_inputs))


def test_cancellation_and_expired_shared_admission_leave_no_key(finite_prepared, finite_tools):
    event = threading.Event()
    event.set()
    with pytest.raises(LeaseCancelledError):
        owner.prepare_finite_cache_correspondence(**arguments(finite_prepared, finite_tools, cancel_event=event))
    with pytest.raises(LeaseTimeoutError):
        owner.prepare_finite_cache_correspondence(**arguments(finite_prepared, finite_tools, timeout_seconds=1e-12))


def test_fresh_process_reopens_native_duckdb_and_cas_and_rederives_exact_keys(finite_prepared, finite_tools, tmp_path):
    args = arguments(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**args)
    request = tmp_path / "replay.json"
    request.write_text(json.dumps({"report": report, "head": args["expected_head"].to_dict(),
        "repository": str(args["repository"]), "contract": args["contract"].to_dict(),
        "inputs": args["inputs"], "tool_policy": args["tool_policy"]}))
    finite_prepared["connection"].close()
    child = r'''
import json, sys
from pathlib import Path
import duckdb
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog,CodebaseHead
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler,ResourceSchedulerConfig
from ipfs_accelerate_py.agent_supervisor.proof.finite_cache_correspondence import verify_finite_cache_correspondence
path=Path(sys.argv[1]); row=json.loads(path.read_text())
cx=duckdb.connect(str(path.parent/'current.duckdb'),config={'threads':1,'memory_limit':'64MB'})
store=DuckDBASTStore(connection=cx); cas=ImmutableCAS(path.parent/'artifacts')
index=RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),artifacts=cas,catalog=CodebaseCatalog(store,cas))
scheduler=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=path.parent/'restart-admission.json',proof_resource_sampler=lambda:ProofHostResources(8,8192,8192),lane_reservations={},auto_renew_leases=False))
keys=verify_finite_cache_correspondence(row['report'],index=index,repository=Path(row['repository']),expected_head=CodebaseHead.from_dict(row['head']),contract=IntegerOffsetContract.from_dict(row['contract']),inputs=row['inputs'],tool_policy=row['tool_policy'],scheduler=scheduler)
assert scheduler.snapshot()['active_lease_count']==0
print(json.dumps({'ids':[key.key_id for key in keys],'pid':__import__('os').getpid()}))
cx.close()
'''
    completed = subprocess.run([sys.executable, "-c", child, str(request)], text=True,
                               capture_output=True, timeout=60, check=True)
    actual = json.loads(completed.stdout)
    assert actual["pid"] != os.getpid()
    assert actual["ids"] == [report[key] for key in ("canonical_key_id", "execution_key_id", "bridged_key_id")]


def test_fresh_matcher_fact_and_counterexample_join_exact_key_materials(finite_prepared, finite_tools):
    # The matcher performs actual Python/Lean runs. Key preparation itself only
    # reobserves/recompiles, and cannot adopt that receipt's positive authority.
    match = finite_match(finite_prepared, finite_tools)
    report = owner.prepare_finite_cache_correspondence(**arguments(finite_prepared, finite_tools))
    observation = match["observation"]
    materials = report["materials"]
    assert match["status"] == "finite_counterexample"
    assert observation["kernel_checked_model_table"] is True
    assert observation["runtime_observation_coverage_complete"] is True
    assert observation["head"] == materials["source"]["head"]
    assert observation["source_cid"] == materials["source"]["source_cid"]
    assert observation["source_sha256"] == materials["source"]["source_sha256"]
    assert observation["contract"] == materials["obligation"]["contract"]
    assert observation["domain_inputs"] == materials["bounds"]["inputs"]
    assert observation["domain_cid"] == cid_for_structured(materials["bounds"])
    assert observation["compiled_cid"] == materials["translation"]["compiled_cid"]
    assert observation["tool_policy"] == materials["policy"]["native_tools"]
    assert len(match["current_facts"]) == 1
    assert len(match["finite_counterexamples"]) == len(materials["bounds"]["inputs"])
    provenance = match["current_facts"][0]["provenance_refs"]
    for value in (observation["source_cid"], observation["domain_cid"], observation["compiled_cid"],
                  observation["tool_policy_cid"], observation["result_cid"],
                  cid_for_structured(observation["lean_certificate"])):
        assert value in provenance
    assert materials["authority_ceiling"] == "none"
    assert all(report[name] is False for name in owner.FALSE)
    assert owner.verify_finite_cache_correspondence(report, **arguments(finite_prepared, finite_tools))[0].key_id == report["canonical_key_id"]
