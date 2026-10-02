"""Durable finite evidence; real native Python/Lean, no warmed proof authority."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import finite_checked_cache as owner
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache, ProofCacheKey, SingleFlightExecutionError
from ipfs_datasets_py.logic.software_contracts.cache import CacheIntegrityError
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from test.api.test_finite_cache_correspondence import arguments, scalar
from test.api.test_finite_integer_codebase import finite_prepared, finite_tools, finite_source


@pytest.fixture
def setup(finite_prepared, finite_tools, tmp_path):
    args = arguments(finite_prepared, finite_tools,
        contract=scalar.IntegerOffsetContract("calc.py", "increment", "n", 1))
    store = owner.FiniteCheckedCache(FormalVerificationCache(tmp_path / "proof"), args["index"].artifacts)
    return store, args


def edit_sql(store, statement, params=()):
    with store._transaction() as cx:
        cx.execute(statement, params)


def replace_record(store, result, record):
    cid = store.artifacts.put(record)
    edit_sql(store, "UPDATE finite_checked_cache_records SET record_cid=? WHERE request_key=?",
             [cid, result["request_key"]])


def test_actual_positive_exact_theorem_scope_and_complete_bodies(setup, monkeypatch):
    store, args = setup
    assert store.lookup(owner_inputs=args)["status"] == "miss"
    saved = store.check_and_store(owner_inputs=args)
    assert saved["status"] == "positive" and saved["positive_reuse_eligible"]
    assert saved["fresh_native_observation"] and not saved["duplicate_publication"]
    assert saved["source_runtime_semantics_verified"] is saved["execution_authority"] is saved["completion_authority"] is False
    record = saved["evidence"]
    assert len(record["correspondence"]["field_correspondence"]) == 16
    assert set(record["execution_producers"]) == set(owner.EXECUTION_PRODUCERS)
    assert record["scope"] == record["receipt"]["metadata"]["scope"] == owner.SCOPE
    assert record["receipt"]["authoritative_assurance"] == "kernel_verified"
    bundle = store.artifacts.get(record["bundle_cid"])
    assert set(bundle["body_cids"]) == set(owner.NAMES)
    assert len(bundle["observation"]["observations"]) == 5
    assert not Path(bundle["observation"]["output"]).exists()
    calls = []
    real = owner.native.observe_finite_integer_source
    def count(**kwargs):
        calls.append(True)
        return real(**kwargs)
    monkeypatch.setattr(owner.native, "observe_finite_integer_source", count)
    looked = store.lookup(owner_inputs=args)
    duplicate = store.check_and_store(owner_inputs=args)
    assert len(calls) == 2
    assert looked["record_cid"] == duplicate["record_cid"] == saved["record_cid"]
    assert duplicate["duplicate_publication"]
    refs = store.dependents("source", record["correspondence"]["materials"]["source"]["source_cid"])
    assert refs == [{"request_key": saved["request_key"], "record_cid": saved["record_cid"],
                     "historical_only": True, "positive_reuse_eligible": False}]


def test_refuted_contract_is_durable_but_never_positive(setup, monkeypatch):
    store, args = setup
    args["contract"] = scalar.IntegerOffsetContract("calc.py", "increment", "n", 2)
    result = store.check_and_store(owner_inputs=args)
    assert result["status"] == "refuted" and not result["positive_reuse_eligible"]
    assert result["evidence"]["receipt"] is result["evidence"]["formal_key"] is None
    with store._transaction() as cx:
        assert cx.execute("SELECT count(*) FROM proof_cache_entries").fetchone()[0] == 0
    def forbidden(**kwargs):
        raise AssertionError("historical negative lookup must not invoke a checker")
    monkeypatch.setattr(owner.native, "observe_finite_integer_source", forbidden)
    looked = store.lookup(owner_inputs=args)
    assert looked["record_cid"] == result["record_cid"]
    assert looked["fresh_native_observation"] is looked["positive_reuse_eligible"] is False


@pytest.mark.parametrize("failed", ["python", "lean"])
def test_real_failed_native_process_is_explicit_negative_history(setup, failed):
    store, args = setup
    selected = {name: Path(args["tool_policy"][name]["path"]) for name in ("python", "lean")}
    selected[failed] = Path("/usr/bin/false")
    args["tool_policy"] = owner.native.seal_finite_integer_tools(
        python_executable=selected["python"], lean_executable=selected["lean"])
    saved = store.check_and_store(owner_inputs=args)
    looked = store.lookup(owner_inputs=args)
    assert saved["status"] == looked["status"] == failed + "_failed"
    assert saved["evidence"]["formal_key"] is saved["evidence"]["receipt"] is None
    assert not saved["positive_reuse_eligible"] and not looked["positive_reuse_eligible"]
    assert not looked["fresh_native_observation"]


@pytest.mark.parametrize("name", sorted(owner.NAMES))
def test_missing_each_native_artifact_refuses_positive_reuse(setup, name):
    store, args = setup
    result = store.check_and_store(owner_inputs=args)
    bundle = store.artifacts.get(result["evidence"]["bundle_cid"])
    store.artifacts.path_for(bundle["body_cids"][name], source=True).unlink()
    with pytest.raises((FileNotFoundError, CacheIntegrityError)):
        store.lookup(owner_inputs=args)


@pytest.mark.parametrize("target", ["record", "bundle", "source", "lean_source", "lean_olean"])
def test_corrupted_cas_bytes_refuse_reuse(setup, target):
    store, args = setup
    result = store.check_and_store(owner_inputs=args)
    if target == "record":
        path = store.artifacts.path_for(result["record_cid"])
    elif target == "bundle":
        path = store.artifacts.path_for(result["evidence"]["bundle_cid"])
    else:
        bundle = store.artifacts.get(result["evidence"]["bundle_cid"])
        path = store.artifacts.path_for(bundle["body_cids"][target], source=True)
    path.write_bytes(b"corrupt")
    with pytest.raises((ValueError, CacheIntegrityError)):
        store.lookup(owner_inputs=args)


def test_rehashed_native_lean_body_cannot_change_the_checked_statement(setup):
    store, args = setup
    result = store.check_and_store(owner_inputs=args)
    record = deepcopy(result["evidence"])
    bundle = store.artifacts.get(record["bundle_cid"])
    observation = bundle["observation"]
    raw = b"theorem forged : False := by decide\n"
    cid = store.artifacts.put_bytes(raw)
    bundle["body_cids"]["lean_source"] = cid
    observation["artifacts"]["lean_source"].update(cid=cid, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    observation["result_cid"] = cid_for_structured({k:v for k,v in observation.items() if k != "result_cid"})
    record["bundle_cid"] = store.artifacts.put(bundle)
    replace_record(store, result, record)
    with pytest.raises(owner.native.FiniteIntegerObservationError):
        store.lookup(owner_inputs=args)


@pytest.mark.parametrize("damage", ["dependency_missing", "dependency_extra", "status", "formal_receipt_missing", "formal_receipt_malformed", "record_extra", "marker_assurance", "record_producer", "metadata", "duplicate_schema"])
def test_sql_and_rehashed_record_corruption_refuses_reuse(setup, damage):
    store, args = setup
    result = store.check_and_store(owner_inputs=args)
    if damage == "dependency_missing":
        edit_sql(store, "DELETE FROM finite_checked_cache_dependencies WHERE kind='source'")
    elif damage == "dependency_extra":
        edit_sql(store, "INSERT INTO finite_checked_cache_dependencies VALUES (?, 'extra', 'value')", [result["request_key"]])
    elif damage == "status":
        edit_sql(store, "UPDATE finite_checked_cache_records SET status='refuted'")
    elif damage == "formal_receipt_missing":
        edit_sql(store, "DELETE FROM proof_cache_entries")
    elif damage == "formal_receipt_malformed":
        edit_sql(store, "UPDATE proof_cache_entries SET entry_json='{}'")
    elif damage == "metadata":
        edit_sql(store, "UPDATE finite_checked_cache_meta SET artifact_root='/different'")
    elif damage == "duplicate_schema":
        edit_sql(store, "CREATE TABLE finite_checked_cache_rogue (value INTEGER)")
    else:
        record = deepcopy(result["evidence"])
        if damage == "record_extra":
            record["unknown"] = True
        elif damage == "marker_assurance":
            record["receipt"] = {"verdict": "proved", "assurance": "kernel_verified"}
        else:
            record["producer"] = "0" * 64
        replace_record(store, result, record)
    with pytest.raises(owner.FiniteCheckedCacheError):
        store.lookup(owner_inputs=args)


def test_dependency_lookup_bounds_and_exact_record_join(setup):
    store, args = setup
    result = store.check_and_store(owner_inputs=args)
    source = result["evidence"]["correspondence"]["materials"]["source"]["source_cid"]
    for limit in (0, 65, True):
        with pytest.raises(owner.FiniteCheckedCacheError):
            store.dependents("source", source, limit=limit)
    store.check_and_store(owner_inputs=dict(args, inputs=[0, 1]))
    with pytest.raises(owner.FiniteCheckedCacheError, match="overflows"):
        store.dependents("source", source, limit=1)
    edit_sql(store, "INSERT INTO finite_checked_cache_dependencies VALUES (?, 'forged', 'query')", [result["request_key"]])
    with pytest.raises(owner.FiniteCheckedCacheError, match="does not join"):
        store.dependents("forged", "query")
    edit_sql(store, "INSERT INTO finite_checked_cache_dependencies VALUES ('orphan', 'source', ?)", [source])
    with pytest.raises(owner.FiniteCheckedCacheError):
        store.dependents("source", source)


def test_capacity_and_sql_primary_keys_never_silently_truncate(setup):
    store, args = setup
    store.max_records = 1
    result = store.check_and_store(owner_inputs=args)
    with pytest.raises(Exception):
        edit_sql(store, "INSERT INTO finite_checked_cache_records VALUES (?, ?, 'positive', '')",
                 [result["request_key"], result["record_cid"]])
    with pytest.raises(owner.FiniteCheckedCacheError, match="capacity exhausted"):
        store.check_and_store(owner_inputs=dict(args, inputs=[0, 1]))
    assert store.lookup(owner_inputs=args)["positive_reuse_eligible"]


@pytest.mark.parametrize("damage", ["unique_removed", "primary_removed", "column_default", "nullable_column"])
def test_rehashed_sql_schema_still_requires_normative_columns_and_constraints(setup, damage):
    store, args = setup
    result = store.check_and_store(owner_inputs=args)
    columns = owner._TABLES["finite_checked_cache_records"]
    if damage == "unique_removed":
        columns = columns.replace("record_cid VARCHAR UNIQUE", "record_cid VARCHAR")
    elif damage == "primary_removed":
        columns = columns.replace("request_key VARCHAR PRIMARY KEY", "request_key VARCHAR NOT NULL")
    elif damage == "column_default":
        columns = columns.replace("status VARCHAR NOT NULL", "status VARCHAR NOT NULL DEFAULT 'positive'")
    else:
        columns = columns.replace("status VARCHAR NOT NULL", "status VARCHAR")
    with store._transaction() as cx:
        cx.execute("ALTER TABLE finite_checked_cache_records RENAME TO prior_records")
        cx.execute("CREATE TABLE finite_checked_cache_records (" + columns + ")")
        cx.execute("INSERT INTO finite_checked_cache_records SELECT * FROM prior_records")
        cx.execute("DROP TABLE prior_records")
        cx.execute("UPDATE finite_checked_cache_meta SET catalog_hash=?", [store._catalog_hash(cx)])
    with pytest.raises(owner.FiniteCheckedCacheError, match="normative"):
        store.lookup(owner_inputs=args)
    with pytest.raises(owner.FiniteCheckedCacheError, match="normative"):
        owner.FiniteCheckedCache(store.cache, store.artifacts)


def test_current_source_domain_producer_and_tool_fences(setup, monkeypatch):
    store, args = setup
    store.check_and_store(owner_inputs=args)
    assert store.lookup(owner_inputs=dict(args, inputs=[0, 1]))["status"] == "miss"
    changed = deepcopy(args["tool_policy"])
    changed["lean"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        store.lookup(owner_inputs=dict(args, tool_policy=changed))
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_producer", lambda: "0" * 64)
        with pytest.raises(owner.FiniteCheckedCacheError, match="producer"):
            store.lookup(owner_inputs=args)
    with monkeypatch.context() as patch:
        pins = owner._execution_pins()
        pins["ipfs_datasets_py.logic.backends.codebase_process"] = "0" * 64
        patch.setattr(owner, "_execution_pins", lambda: pins)
        with pytest.raises(owner.FiniteCheckedCacheError, match="producer"):
            store.lookup(owner_inputs=args)
    (args["repository"] / "calc.py").write_bytes(finite_source(2))
    with pytest.raises(StaleCodebaseError):
        store.lookup(owner_inputs=args)


def test_fresh_checker_failure_cannot_be_replaced_by_saved_positive(setup, monkeypatch):
    store, args = setup
    store.check_and_store(owner_inputs=args)
    def failed(**kwargs):
        raise RuntimeError("actual checker unavailable")
    monkeypatch.setattr(owner.native, "observe_finite_integer_source", failed)
    with pytest.raises(RuntimeError, match="checker unavailable"):
        store.lookup(owner_inputs=args)


def test_interrupted_publication_never_leaves_an_eligible_partial_record(setup, monkeypatch):
    store, args = setup
    real = store.cache.put
    def interrupted(*a, **kw):
        value = real(*a, **kw)
        assert value.stored
        raise RuntimeError("interrupted after native proof row")
    with monkeypatch.context() as patch:
        patch.setattr(store.cache, "put", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            store.check_and_store(owner_inputs=args)
    assert store.lookup(owner_inputs=args)["status"] == "miss"
    with store._transaction() as cx:
        assert cx.execute("SELECT count(*) FROM finite_checked_cache_records").fetchone()[0] == 0
        assert cx.execute("SELECT count(*) FROM finite_checked_cache_dependencies").fetchone()[0] == 0
    # The existing owner intentionally retains a bounded failed-flight result.
    # It must not be bypassed; retry only after its native TTL has expired.
    with pytest.raises(SingleFlightExecutionError):
        store.check_and_store(owner_inputs=args)
    clock = store.cache._clock
    monkeypatch.setattr(store.cache, "_clock", lambda: clock() + 61)
    assert store.check_and_store(owner_inputs=args)["positive_reuse_eligible"]


def test_concurrent_real_publication_has_one_immutable_receipt(setup, monkeypatch):
    store, args = setup
    # The source catalog has one native connection. Serialize only its calls,
    # leaving the independent proof owner publication genuinely concurrent.
    lock, barrier = threading.RLock(), threading.Barrier(2)
    for name in ("prepare_finite_cache_correspondence", "verify_finite_cache_correspondence"):
        original = getattr(owner.correspondence, name)
        def serial(*a, _original=original, **kw):
            with lock:
                return _original(*a, **kw)
        monkeypatch.setattr(owner.correspondence, name, serial)
    fresh = store._fresh
    def staged(*a):
        with lock:
            value = fresh(*a)
        barrier.wait(timeout=60)
        return value
    monkeypatch.setattr(store, "_fresh", staged)
    with ThreadPoolExecutor(max_workers=2) as executor:
        runs = list(executor.map(lambda _: store.check_and_store(owner_inputs=args), range(2)))
    assert len({r["record_cid"] for r in runs}) == 1
    assert sorted(r["duplicate_publication"] for r in runs) == [False, True]
    assert all(r["positive_reuse_eligible"] for r in runs)
    monkeypatch.setattr(store, "_fresh", fresh)
    assert store.lookup(owner_inputs=args)["record_cid"] == runs[0]["record_cid"]


def test_fresh_process_reopens_every_owner_and_rechecks_native_evidence(setup, finite_prepared, tmp_path):
    store, args = setup
    result = store.check_and_store(owner_inputs=args)
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({"head": args["expected_head"].to_dict(), "repository": str(args["repository"]),
        "contract": args["contract"].to_dict(), "inputs": args["inputs"], "tool_policy": args["tool_policy"]}))
    finite_prepared["connection"].close()
    code = r'''
import json, os, sys
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
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache,FormalVerificationCache
path=Path(sys.argv[1]);row=json.loads(path.read_text())
cx=duckdb.connect(str(path.parent/'current.duckdb'),config={'threads':1,'memory_limit':'64MB'})
ast=DuckDBASTStore(connection=cx);cas=ImmutableCAS(path.parent/'artifacts')
index=RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=ast),artifacts=cas,catalog=CodebaseCatalog(ast,cas))
scheduler=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=path.parent/'restart-admission.json',proof_resource_sampler=lambda:ProofHostResources(8,8192,8192),lane_reservations={},auto_renew_leases=False))
cache=FiniteCheckedCache(FormalVerificationCache(path.parent/'proof'),cas)
result=cache.lookup(owner_inputs=dict(index=index,repository=Path(row['repository']),expected_head=CodebaseHead.from_dict(row['head']),contract=IntegerOffsetContract.from_dict(row['contract']),inputs=row['inputs'],tool_policy=row['tool_policy'],scheduler=scheduler))
assert scheduler.snapshot()['active_lease_count']==0
print(json.dumps({'pid':os.getpid(),'record_cid':result['record_cid'],'positive_reuse_eligible':result['positive_reuse_eligible'],'fresh_native_observation':result['fresh_native_observation']}))
cx.close()
'''
    run = subprocess.run([sys.executable, "-c", code, str(path)], check=True, capture_output=True, text=True, timeout=60)
    actual = json.loads(run.stdout)
    assert actual == {"pid": actual["pid"], "record_cid": result["record_cid"],
                      "positive_reuse_eligible": True, "fresh_native_observation": True}
    assert actual["pid"] != os.getpid()
