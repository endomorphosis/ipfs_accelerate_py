"""Closed native intent inputs and actual owner verification at the planner seam."""
from dataclasses import replace
import hashlib
import json
import shutil
import threading

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.planning import checked_integer_codebase as subject
from ipfs_datasets_py.logic.intent_ir.schema import IntentModality, StatementKind

TEXT = "Under python-integer-offset@1, counter.py::increment(n) must return n + 1."


def _query(document=None, text=TEXT, **kwargs):
    return subject.prepare_integer_offset_query(
        intent_document=document or subject.build_integer_offset_intent(text), source_text=text, **kwargs)


def _statement(document=None, **changes):
    document = document or subject.build_integer_offset_intent(TEXT)
    return replace(document, statements=(replace(document.statements[0], **changes),))


@pytest.mark.parametrize("offset,sign,magnitude", [(0, "+", "0"), (1, "+", "1"), (-3, "-", "3"),
                                                   (2**64 - 1, "+", str(2**64 - 1))])
def test_closed_sentence_builds_native_exact_ordered_arguments(offset, sign, magnitude):
    text = f"Under python-integer-offset@1, counter.py::increment(n) must return n {sign} {magnitude}."
    native = subject.build_integer_offset_intent(text)
    query = _query(native, text)
    assert native.statements[0].arguments == ("counter.py", "increment", "n", str(offset), "python-integer-offset@1")
    assert query["supported"] and query["semantic_alignment_verified"]
    assert query["alignment_scope"] == subject.CNL_PROFILE
    assert query["contract"]["offset"] == offset
    assert query["intent_source"]["identity"]["content_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert query["intent_source"]["span"] == {"start_char": 0, "end_char": len(text)}
    assert all(query[key] is False for key in (
        "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
        "execution_authority", "completion_authority", "behavioral_satisfaction"))
    assert _query(native.to_dict(), text) == query


@pytest.mark.parametrize("text", [
    TEXT + "\n", TEXT + " Also repair everything.", TEXT.replace("must", "may"),
    TEXT.replace("n + 1", "m + 1"), TEXT.replace("+ 1", "+ 01"),
    TEXT.replace("+ 1", "- 0"), TEXT.replace("+ 1", "+ 18446744073709551616"),
    TEXT.replace("counter.py", "../counter.py"), TEXT.replace("counter.py", "/counter.py"),
    TEXT.replace("counter.py", "counter.js"), TEXT.replace("increment(n)", "increment(result)"),
    "repair counter.py", "x" * 4097,
], ids=lambda value: hashlib.sha256(value.encode()).hexdigest()[:8])
def test_closed_sentence_builder_abstains_outside_exact_grammar(text):
    with pytest.raises(subject.CheckedIntegerIntentError):
        subject.build_integer_offset_intent(text)


@pytest.mark.parametrize("changes", [
    {"predicate": "repair", "arguments": ("agent", "counter.py")},
    {"arguments": ("increment", "counter.py", "n", "1", "python-integer-offset@1")},
    {"arguments": ("counter.py", "increment", "n", "2", "python-integer-offset@1")},
    {"arguments": ("counter.py", "increment", "n", "01", "python-integer-offset@1")},
    {"arguments": ("counter.py", "increment", "n", "1", "python-integer-offset@2")},
    {"modality": IntentModality.PROHIBITED},
    {"normalized_text": "The function should not increment."},
])
def test_valid_native_atom_mismatch_remains_unknown(changes):
    query = _query(_statement(**changes))
    assert not query["supported"]
    assert query["reasons"]
    assert query["semantic_alignment_verified"] is False


def test_typed_authored_atom_is_explicitly_unqualified_for_free_text():
    text = "This explicitly authored typed atom requests an integer offset contract."
    native = subject.build_integer_offset_intent(TEXT)
    source = replace(native.sources[0], content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                     span=replace(native.sources[0].span, end_char=len(text)))
    native = replace(native, sources=(source,), statements=(replace(native.statements[0], normalized_text=text),))
    query = _query(native, text)
    assert query["supported"]
    assert query["semantic_alignment_verified"] is False
    assert query["alignment_scope"] == "explicit_authored_native_atom_only"


def test_source_provenance_and_decoder_are_rechecked():
    native = subject.build_integer_offset_intent(TEXT)
    bad_span = replace(native.sources[0], span=replace(native.sources[0].span, end_char=len(TEXT) - 1))
    for document in (replace(native, sources=(bad_span,)),
                     replace(native, sources=(replace(native.sources[0], content_sha256="0" * 64),))):
        with pytest.raises(subject.CheckedIntegerIntentError):
            _query(document)
    with pytest.raises(subject.CheckedIntegerIntentError):
        _query(source_identity={"ref_id": native.sources[0].ref_id})
    raw = native.to_dict()
    raw["statements"][0]["unexpected_authority"] = True
    with pytest.raises(subject.CheckedIntegerIntentError):
        _query(raw)
    with pytest.raises(subject.CheckedIntegerIntentError):
        _query(_statement(kind=StatementKind.ASSUMPTION))  # A native document needs a goal.


def test_query_is_detached_and_retains_all_residual_ids():
    native = subject.build_integer_offset_intent(TEXT)
    native = replace(native, statements=(*native.statements,
        replace(native.statements[0], statement_id="another-goal")))
    first = _query(native)
    assert first["supported"] is False
    assert first["requirement_ids"] == ["another-goal", subject.STATEMENT_ID]
    first["statement"]["arguments"][0] = "changed.py"
    assert _query(native)["statement"]["arguments"][0] == "counter.py"


@pytest.mark.parametrize("field,value", [("title", "x" * (1024 * 1024 + 1)),
                                         ("tags", ["x"] * 20_001)], ids=["oversized-string", "oversized-list"])
def test_query_rejects_oversized_inputs_before_native_decoding(field, value, monkeypatch):
    raw = subject.build_integer_offset_intent(TEXT).to_dict()
    raw[field] = value
    monkeypatch.setattr(subject, "decode_intent_ir", lambda _: pytest.fail("oversized native decoding started"))
    with pytest.raises(subject.CheckedIntegerIntentError):
        _query(raw)


@pytest.fixture
def native(tmp_path):
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits, RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig

    repository = tmp_path / "private-host-locator"
    repository.mkdir()
    source = repository / "counter.py"
    source.write_text("def increment(n: int) -> int:\n    return n + 1\n")
    host = [ProofHostResources(8, 8192, 8192)]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "scheduler.json", proof_resource_sampler=lambda: host[0],
        lane_reservations={}, auto_renew_leases=False, proof_backoff_seconds=0,
        poll_interval_seconds=0.005,
    ))
    connection = duckdb.connect(str(tmp_path / "codebase.duckdb"), config={"threads": 1, "memory_limit": "128MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(tmp_path / "artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                                    catalog=CodebaseCatalog(store, artifacts))
    limits = CodebaseScanLimits(max_entries=16, max_file_bytes=1024)
    published = index.prepare_current(repository, repository_id="worktree:integer", operation_id="initial",
                                     expected_head=None, scheduler=scheduler, limits=limits)
    try:
        yield {"index": index, "repository": repository, "source": source, "scheduler": scheduler,
               "head": published.head, "host": host, "limits": limits}
    finally:
        connection.close()


def _match(native, **kwargs):
    return subject.match_checked_integer_intent(
        index=native["index"], repository=native["repository"], repository_id="worktree:integer",
        intent_document=subject.build_integer_offset_intent(TEXT), source_text=TEXT,
        scheduler=native["scheduler"], timeout_seconds=20, **kwargs)


NATIVE_SOLVERS = pytest.mark.skipif(not shutil.which("z3") or not shutil.which("cvc5"),
                                    reason="native Z3 and CVC5 required")


@NATIVE_SOLVERS
def test_native_fixed_then_buggy_source_answers_exact_conditional_contract(native):
    matched = _match(native)
    assert matched["status"] == "conditional_contract_matched"
    assert matched["checked_result"]["solver_replayed"] is True
    assert matched["checked_result"]["head"] == native["head"].to_dict()
    assert str(native["repository"]) not in json.dumps(matched)
    assert matched["residual_requirements"] == [{"statement_id": subject.STATEMENT_ID,
                                                 "status": "runtime_behavior_unresolved"}]
    for key in ("source_semantics_verified", "runtime_behavior_verified", "proof_authority",
                "execution_authority", "completion_authority", "behavioral_satisfaction"):
        assert matched[key] is False
    for key in ("current_behavioral_facts", "behavioral_satisfied_requirements", "runtime_refutations", "removed_task_ids"):
        assert matched[key] == []
    native["source"].write_text("def increment(n: int) -> int:\n    return n\n")
    successor = native["index"].prepare_current(
        native["repository"], repository_id="worktree:integer", operation_id="buggy", expected_head=native["head"],
        scheduler=native["scheduler"], limits=native["limits"],
    )
    refuted = _match(native)
    assert refuted["status"] == "conditional_contract_refuted"
    assert refuted["checked_result"]["head"] == successor.head.to_dict()
    assert refuted["match_cid"] != matched["match_cid"]
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_generic_repair_does_not_invoke_property_verifier(native, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

    def forbidden(*args, **kwargs):
        pytest.fail("generic repair was interpreted as an integer property")
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify", forbidden)
    result = subject.match_checked_integer_intent(
        index=native["index"], repository=native["repository"], repository_id="worktree:integer",
        intent_document=_statement(predicate="repair", arguments=("agent", "counter.py")), source_text=TEXT,
        scheduler=native["scheduler"],
    )
    assert result["status"] == "unknown" and result["checked_result"] is None
    assert result["current_behavioral_facts"] == []


def test_unsupported_source_is_unknown_with_fresh_owner_binding(native):
    native["source"].write_text("def increment(n: int) -> int:\n    return n * n\n")
    native["index"].prepare_current(
        native["repository"], repository_id="worktree:integer", operation_id="unsupported",
        expected_head=native["head"], scheduler=native["scheduler"], limits=native["limits"],
    )
    result = _match(native)
    assert result["status"] == "unknown"
    assert result["checked_result"]["status"] == "unsupported"
    assert result["checked_result"]["solver_replayed"] is False
    assert result["reasons"] == ["checked_contract_unsupported"]
    assert result["current_behavioral_facts"] == result["removed_task_ids"] == []


@NATIVE_SOLVERS
def test_exit_source_drift_withholds_native_match(native, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

    original = CodebaseIntegerVerifier.verify
    def verify_then_drift(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        native["source"].write_text("def increment(n: int) -> int:\n    return n + 99\n")
        return result
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify", verify_then_drift)
    result = "withheld"
    with pytest.raises(ValueError):
        result = _match(native)
    assert result == "withheld"
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


@NATIVE_SOLVERS
def test_external_cancellation_after_owner_result_withholds_match(native, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError

    cancel = threading.Event()
    original = CodebaseIntegerVerifier.verify
    def verify_then_cancel(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        cancel.set()
        return result
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify", verify_then_cancel)
    with pytest.raises(LeaseCancelledError):
        _match(native, cancel_event=cancel)
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_caller_cannot_supply_a_positive_result(native):
    with pytest.raises(TypeError, match="checked_result"):
        _match(native, checked_result={"status": "proved"})


@pytest.mark.parametrize("defect", ["head", "contract", "authority", "replay"])
@NATIVE_SOLVERS
def test_owner_api_binding_regression_rejects_even_a_positive_outcome(native, monkeypatch, defect):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

    original = CodebaseIntegerVerifier.verify
    def verify_then_corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if defect == "head":
            result["head"]["generation"] += 1
        elif defect == "contract":
            result["contract"]["offset"] += 1
        elif defect == "authority":
            result["behavior_authority"] = True
        else:
            result["solver_replayed"] = False
        return result
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify", verify_then_corrupt)
    with pytest.raises(subject.CheckedIntegerIntentError):
        _match(native)


def test_external_pressure_before_entry_cleans_up(native):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError

    native["host"][0] = replace(native["host"][0], memory_stall_percent=8)
    with pytest.raises(LeaseTimeoutError):
        _match(native, admission_timeout_seconds=0.01)
    snapshot = native["scheduler"].snapshot()
    assert snapshot["active_lease_count"] == snapshot["waiting_request_count"] == 0
