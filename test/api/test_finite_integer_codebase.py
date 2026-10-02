"""Independent native qualification of finite, source-rooted IntentIR facts."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority, ObservedFact, TypedIntent,
)
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as native
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS, CacheIntegrityError
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig, LeaseCancelledError, LeaseTimeoutError,
)

INPUTS = [-2, -1, 0, 1, 2]
VIEW = "repository:finite-intent-independent"
PYTHON = Path("/home/barberb/.local/bin/python").resolve()
LEAN = Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean")
FALSE_FIELDS = ("source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
                "proof_authority", "execution_authority", "completion_authority", "mutation_authority")


def finite_text(inputs=INPUTS, offset=2):
    domain = json.dumps(inputs, separators=(",", ":"))
    sign = "+" if offset >= 0 else "-"
    prefix = "Under python-integer-offset-finite@1, calc.py::increment(n) must return "
    suffix = f" for inputs {domain}."
    return prefix + "an exact int" + suffix + "\n" + prefix + f"n {sign} {abs(offset)}" + suffix


def finite_source(offset=1):
    return f"def increment(n: int) -> int:\n    return n + {offset}\n".encode()


def finite_git(repository, *arguments):
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()


@pytest.fixture(scope="module")
def finite_tools():
    assert PYTHON.is_file() and LEAN.is_file(), "native tools are required for qualification"
    return native.seal_finite_integer_tools(python_executable=PYTHON, lean_executable=LEAN)


@pytest.fixture
def finite_prepared(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    finite_git(repository, "init", "-q")
    finite_git(repository, "config", "user.name", "Finite Intent Fixture")
    finite_git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / "calc.py").write_bytes(finite_source())
    finite_git(repository, "add", "calc.py")
    finite_git(repository, "commit", "-qm", "authored exact finite input")
    cx = duckdb.connect(str(tmp_path / "current.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    store = DuckDBASTStore(connection=cx)
    cas = ImmutableCAS(tmp_path / "artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=cas,
                                   catalog=CodebaseCatalog(store, cas))
    owner = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "admission.json", proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192),
        lane_reservations={}, auto_renew_leases=False, proof_backoff_seconds=.02, poll_interval_seconds=.005,
    ))
    head = index.prepare_current(repository, repository_id=VIEW, operation_id="initial",
                                 expected_head=None, scheduler=owner).head
    yield {"index": index, "repository": repository, "repository_id": VIEW,
           "expected_head": head, "scheduler": owner, "output": tmp_path / "observation", "connection": cx}
    assert owner.snapshot()["active_lease_count"] == owner.snapshot()["waiting_request_count"] == 0
    cx.close()


def finite_match(prepared, tools, **changes):
    arguments = {key: value for key, value in prepared.items() if key != "connection"}
    text = changes.pop("source_text", finite_text())
    document = changes.pop("intent_document", matcher.build_finite_integer_intent(text))
    arguments.update(intent_document=document, source_text=text, tool_policy=deepcopy(tools), timeout_seconds=30)
    arguments.update(changes)
    return matcher.match_finite_integer_intent(**arguments)


def assert_finite_scope(result):
    assert all(result[field] is False for field in FALSE_FIELDS)
    assert result["scope"] == "explicit_finite_domain_only"
    assert result["unbounded_behavior_status"] == "unresolved"
    assert result["removed_task_ids"] == []


def test_exact_cnl_native_ast_and_complete_clause_spans_are_source_aligned():
    text = finite_text()
    document = matcher.build_finite_integer_intent(text)
    query = matcher.prepare_finite_integer_query(intent_document=document, source_text=text)
    assert query["supported"] is query["semantic_alignment_verified"] is True
    assert query["domain_inputs"] == INPUTS
    assert query["requirement_ids"] == sorted([matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID])
    assert len(document.statements) == len(document.sources) == 2
    source_rows = sorted((source.to_dict() for source in document.sources), key=lambda row: row["ref_id"])
    by_ref = {source["ref_id"]: source for source in source_rows}
    for statement in document.statements:
        source = by_ref[statement.source_ref_ids[0]]
        span = source["span"]
        assert text[span["start_char"]:span["end_char"]] == statement.normalized_text
        assert source["content_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert source["source_revision"] == "authored:1"
        assert source["review_status"] == "unreviewed"
    assert matcher.prepare_finite_integer_query(intent_document=document.to_dict(), source_text=text,
                                                source_identity=source_rows) == query
    assert query["query_cid"] == cid_for_structured({key: value for key, value in query.items() if key != "query_cid"})
    assert all(query[field] is False for field in FALSE_FIELDS)


@pytest.mark.parametrize("change", ["duplicate", "bool", "unsorted", "empty", "different_domains", "spaces", "trailing", "universal"])
def test_cnl_rejects_nonclosed_or_changed_request_domains(change):
    text = finite_text()
    if change == "duplicate":
        text = finite_text([-2, -1, 0, 1, 1])
    elif change == "bool":
        text = finite_text([False, 1])
    elif change == "unsorted":
        text = finite_text([1, 0])
    elif change == "empty":
        text = finite_text([])
    elif change == "different_domains":
        first, second = text.split("\n")
        text = first + "\n" + second.replace("[-2,-1,0,1,2]", "[0,1,2]")
    elif change == "spaces":
        text = text.replace("[-2,-1,0,1,2]", "[-2, -1, 0, 1, 2]")
    elif change == "trailing":
        text += "\n"
    else:
        text = text.replace("for inputs [-2,-1,0,1,2]", "for every Python integer")
    with pytest.raises(matcher.FiniteIntegerIntentError):
        matcher.build_finite_integer_intent(text)


@pytest.mark.parametrize("change", ["hash", "span_outside", "identity", "statement", "span_shift", "missing_clause"])
def test_native_intent_tampering_cannot_create_supported_clause(change):
    text = finite_text()
    document = matcher.build_finite_integer_intent(text).to_dict()
    kwargs = {}
    if change == "hash":
        document["sources"][0]["content_sha256"] = "0" * 64
    elif change == "span_outside":
        document["sources"][0]["span"]["end_char"] = len(text) + 1
    elif change == "identity":
        kwargs["source_identity"] = deepcopy(document["sources"])
        kwargs["source_identity"][0]["source_revision"] = "foreign:2"
    elif change == "statement":
        document["statements"][0]["arguments"][-1] = "1"
    elif change == "span_shift":
        document["sources"][0]["span"]["start_char"] += 1
    else:
        document["statements"] = document["statements"][:1]
    if change in {"hash", "span_outside", "identity"}:
        with pytest.raises(matcher.FiniteIntegerIntentError):
            matcher.prepare_finite_integer_query(intent_document=document, source_text=text, **kwargs)
    else:
        query = matcher.prepare_finite_integer_query(intent_document=document, source_text=text, **kwargs)
        assert query["supported"] is False
        assert query["semantic_alignment_verified"] is False
        assert query["contract"] is query["domain_cid"] is None


def test_real_native_observation_yields_one_finite_fact_and_one_residual(finite_prepared, finite_tools):
    result = finite_match(finite_prepared, finite_tools)
    assert result["status"] == "finite_counterexample", result
    assert result["eligible_clause_ids"] == [matcher.TYPE_STATEMENT_ID]
    assert result["residual_clause_ids"] == [matcher.OFFSET_STATEMENT_ID]
    assert result["eligible_requirements"] == result["eligible_clause_ids"]
    assert len(result["finite_counterexamples"]) == len(INPUTS)
    assert len(result["typed_intent"]["desired_predicates"]) == 2
    typed = TypedIntent.from_dict(result["typed_intent"])
    fact = ObservedFact.from_dict(result["current_facts"][0])
    assert fact.authority is FactAuthority.BOUNDED_OBSERVATION
    assert fact.current_root_id == typed.current_root_id == result["current_root_id"] == finite_prepared["expected_head"].snapshot_cid
    assert fact.predicate.property_id == matcher.TYPE_PREDICATE
    assert result["observation"]["status"] == "observed"
    assert result["observation_cid"] == result["observation"]["result_cid"]
    assert result["observation"]["trace_cid"] in fact.provenance_refs
    assert result["observation"]["domain_cid"] in fact.provenance_refs
    assert result["observation"]["source_cid"] in fact.provenance_refs
    assert result["observation"]["result_cid"] in fact.provenance_refs
    assert_finite_scope(result)


def test_canonical_predicate_order_cannot_discharge_the_unsatisfied_offset_clause(finite_prepared, finite_tools):
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
        ObligationGraphCompiler, ObligationStatus, obligation_id_for_predicate,
    )

    head = finite_prepared["expected_head"]
    manifest = finite_prepared["index"].load(head.manifest_cid)
    source_cid = next(entry.source_cid for entry in manifest.snapshot.entries if entry.path == "calc.py")
    # Authored contracts have independent content-derived predicate IDs. Pick a
    # genuine valid request whose native canonical order differs from clause
    # order, so the regression cannot pass because a temporary path hashes luckily.
    for offset in range(2, 66):
        text = finite_text(offset=offset)
        document = matcher.build_finite_integer_intent(text)
        query = matcher.prepare_finite_integer_query(intent_document=document, source_text=text)
        typed = matcher._typed(query, head, source_cid)
        explicit = typed.metadata["requirement_predicate_ids"]
        clause_order = [explicit[row["statement_id"]] for row in query["statements"]]
        if [row.predicate_id for row in typed.desired_predicates] != clause_order:
            break
    else:
        pytest.fail("no valid reordered native predicate inventory in bounded authored controls")

    result = finite_match(finite_prepared, finite_tools, source_text=text, intent_document=document)
    assert result["eligible_clause_ids"] == [matcher.TYPE_STATEMENT_ID]
    assert result["residual_clause_ids"] == [matcher.OFFSET_STATEMENT_ID]
    typed = TypedIntent.from_dict(result["typed_intent"])
    binding = typed.metadata["requirement_predicate_ids"]
    clauses = {row["statement_id"]: row for row in result["clause_results"]}
    assert {requirement: row["predicate_id"] for requirement, row in clauses.items()} == dict(binding)
    fact = ObservedFact.from_dict(result["current_facts"][0])
    assert fact.predicate.predicate_id == binding[matcher.TYPE_STATEMENT_ID]
    assert fact.predicate.property_id == matcher.TYPE_PREDICATE
    assert fact.authority is FactAuthority.BOUNDED_OBSERVATION
    graph = ObligationGraphCompiler().compile(typed_intent=typed, current_facts=(fact,),
        current_root_id=head.snapshot_cid)
    assert graph.node(obligation_id_for_predicate(binding[matcher.TYPE_STATEMENT_ID])).status is ObligationStatus.DISCHARGED
    assert graph.node(obligation_id_for_predicate(binding[matcher.OFFSET_STATEMENT_ID])).status is not ObligationStatus.DISCHARGED
    assert not graph.complete
    assert_finite_scope(result)


def test_authorized_successor_requires_new_capture_and_satisfies_both_finite_clauses(finite_prepared, finite_tools):
    first = finite_match(finite_prepared, finite_tools)
    old = finite_prepared["expected_head"]
    commit = finite_git(finite_prepared["repository"], "rev-parse", "HEAD")
    (finite_prepared["repository"] / "calc.py").write_bytes(finite_source(2))
    assert finite_git(finite_prepared["repository"], "rev-parse", "HEAD") == commit
    with pytest.raises(StaleCodebaseError):
        finite_match(finite_prepared, finite_tools, output=finite_prepared["output"].parent / "stale")
    successor = finite_prepared["index"].prepare_current(finite_prepared["repository"], repository_id=VIEW,
        operation_id="authorized-successor", expected_head=old, scheduler=finite_prepared["scheduler"]).head
    finite_prepared["expected_head"] = successor
    second = finite_match(finite_prepared, finite_tools, output=finite_prepared["output"].parent / "successor")
    assert second["eligible_clause_ids"] == sorted([matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID])
    assert second["residual_clause_ids"] == second["finite_counterexamples"] == []
    assert second["status"] == "bounded_observed_satisfied"
    assert second["source_cid"] != first["source_cid"]
    assert second["current_root_id"] != first["current_root_id"]
    assert len(second["typed_intent"]["desired_predicates"]) == len(second["current_facts"]) == 2
    assert {fact["fact_id"] for fact in first["current_facts"]}.isdisjoint(fact["fact_id"] for fact in second["current_facts"])
    assert_finite_scope(second)


def test_new_requested_domain_gets_a_distinct_trace_and_no_reused_fact(finite_prepared, finite_tools):
    first = finite_match(finite_prepared, finite_tools)
    second = finite_match(finite_prepared, finite_tools, source_text=finite_text([0, 1, 2]),
                          output=finite_prepared["output"].parent / "different-domain")
    assert second["domain_inputs"] == [0, 1, 2]
    assert second["domain_cid"] != first["domain_cid"]
    assert second["observation"]["trace_cid"] != first["observation"]["trace_cid"]
    assert second["current_facts"][0]["fact_id"] != first["current_facts"][0]["fact_id"]
    assert_finite_scope(second)


def test_valid_but_unsupported_native_intent_retains_every_goal_without_execution(finite_prepared, finite_tools, monkeypatch):
    document = matcher.build_finite_integer_intent(finite_text()).to_dict()
    document["statements"][0]["arguments"][-1] = "99"
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("unsupported intent executed"))
    result = finite_match(finite_prepared, finite_tools, intent_document=document)
    assert result["query"]["supported"] is False
    assert result["observation"] is None and result["current_facts"] == []
    assert result["eligible_clause_ids"] == []
    assert result["residual_clause_ids"] == sorted([matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID])
    assert len(result["typed_intent"]["desired_predicates"]) == 2
    assert_finite_scope(result)


@pytest.mark.parametrize("control", ["wrong_domain", "authority", "trace", "wrong_output", "type_alias"])
def test_forged_observer_return_cannot_create_current_facts(finite_prepared, finite_tools, monkeypatch, control):
    original = native.observe_finite_integer_source
    def corrupt(**kwargs):
        result = original(**kwargs)
        assert result["status"] == "observed", result
        if control == "wrong_domain":
            result["domain_inputs"] = [0, 1, 2]
        elif control == "authority":
            result["behavior_authority"] = True
        elif control == "trace":
            result["observations"] = result["observations"][:-1]
        elif control == "wrong_output":
            result["output"] = str(Path(kwargs["output"]).parent)
        else:
            result["observations"][2]["input"] = False
        result["result_cid"] = cid_for_structured({key: value for key, value in result.items() if key != "result_cid"})
        (Path(kwargs["output"]) / "result.json").write_bytes(canonical_dag_json_bytes(result))
        return result
    monkeypatch.setattr(native, "observe_finite_integer_source", corrupt)
    with pytest.raises(ValueError):
        finite_match(finite_prepared, finite_tools)


@pytest.mark.parametrize("control", ["cancel", "deadline"])
def test_cancelled_or_expired_match_has_no_native_execution_or_lease(finite_prepared, finite_tools, monkeypatch, control):
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("cancelled match executed"))
    arguments = {}
    if control == "cancel":
        event = threading.Event()
        event.set()
        arguments["cancel_event"] = event
        exception = LeaseCancelledError
    else:
        arguments["timeout_seconds"] = 1e-9
        exception = LeaseTimeoutError
    with pytest.raises(exception):
        finite_match(finite_prepared, finite_tools, **arguments)
    assert not finite_prepared["output"].exists()


def test_structural_exit_fence_rejects_edit_after_valid_observation(finite_prepared, finite_tools, monkeypatch):
    original = native.observe_finite_integer_source
    def edit_after(**kwargs):
        result = original(**kwargs)
        assert result["status"] == "observed", result
        (finite_prepared["repository"] / "calc.py").write_bytes(finite_source(2))
        return result
    monkeypatch.setattr(native, "observe_finite_integer_source", edit_after)
    with pytest.raises(StaleCodebaseError):
        finite_match(finite_prepared, finite_tools)
    assert finite_prepared["index"].current(VIEW) == finite_prepared["expected_head"]


def test_public_api_has_no_caller_evidence_or_receipt_bypass(finite_prepared, finite_tools):
    with pytest.raises(TypeError, match="unexpected keyword"):
        finite_match(finite_prepared, finite_tools, observation={"status": "observed", "type_clause_satisfied": True})
