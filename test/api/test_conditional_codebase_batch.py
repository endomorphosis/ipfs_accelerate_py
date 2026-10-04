"""Independent consumer checks for ordered, bounded historical intent lookups."""
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import conditional_codebase_batch as batch
from ipfs_accelerate_py.agent_supervisor.planning import conditional_codebase_evidence as single
from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
from ipfs_datasets_py.duckdb_control.codebase_verification_queries import (
    CodebaseVerificationQueryPage, CodebaseVerificationQueryRequest,
)
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_verification.applicability import RequestedInputDomain
from ipfs_datasets_py.logic.intent_ir.schema import ReviewStatus
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError

from .test_conditional_codebase_evidence import (
    native_current, request, TEXT, PATH, STATEMENT, VIEW, FALSE_FIELDS, assert_runtime_residuals,
)


def envelope(identity="requirement:0", *, unsupported=False, predicates=("n >= 0",)):
    native, contract, domain = request(domain=RequestedInputDomain("increment", predicates=predicates))
    if unsupported:
        native = replace(native, statements=(replace(native.statements[0], review_status=ReviewStatus.UNREVIEWED),
                                              native.statements[1]))
    return {"requirement_id": identity, "intent_document": native, "source_text": TEXT,
            "path": PATH, "contract": contract, "domain": domain, "statement_id": STATEMENT}


def match(native, requirements=None, **controls):
    return batch.match_conditional_codebase_requirements(**native,
        requirements=(envelope(),) if requirements is None else requirements, **controls)


def assert_batch(result, identities):
    assert result["schema"] == batch.SCHEMA
    assert result["requested_requirement_ids"] == identities
    assert [row["requirement_id"] for row in result["matches"]] == identities
    assert result["complete_input_ledger"] is True
    assert all(result[field] is False for field in FALSE_FIELDS)
    assert result["recorded_execution_attested"] is False
    assert result["current_facts"] == result["current_behavioral_facts"] == []
    assert result["eligible_requirements"] == result["behavioral_satisfied_requirements"] == []
    assert result["runtime_refutations"] == result["removed_task_ids"] == []
    assert result["batch_cid"] == cid_for_structured({k: v for k, v in result.items() if k != "batch_cid"})
    for row in result["matches"]:
        assert_runtime_residuals(row["match"])
    assert [(row["requirement_id"], row["statement_id"]) for row in result["residual_requirements"]] == [
        (identity, statement) for identity in identities for statement in (STATEMENT, "runtime-goal")]
    assert all(row["status"] == "runtime_behavior_unresolved" for row in result["residual_requirements"])


def test_native_batch_keeps_duplicate_requests_and_single_match_bytes(native_current, monkeypatch):
    requirements = (envelope("first"), envelope("unsupported", unsupported=True), envelope("duplicate"))
    expected = []
    for row in requirements:
        expected.append(single.match_conditional_codebase_intent(**native_current,
            **{k: v for k, v in row.items() if k != "requirement_id"}))
    original = CodebaseVerificationCatalog.query_many_current
    seen = []

    def query(owner, *args, **kwargs):
        seen.append(kwargs)
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_many_current", query)
    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current",
        lambda *args, **kwargs: pytest.fail("batch repeated single selector lookups"))
    result = match(native_current, requirements)
    assert_batch(result, ["first", "unsupported", "duplicate"])
    assert [row["match"] for row in result["matches"]] == expected
    assert len(seen) == 1
    assert type(seen[0]["requests"]) is tuple and len(seen[0]["requests"]) == 2
    assert all(type(row) is CodebaseVerificationQueryRequest and row.page_size == 2 and row.cursor is None
               for row in seen[0]["requests"])
    assert result["inventory"] == {"inventory_cid": expected[0]["indexed_query_page"]["inventory_cid"],
                                   "epoch": expected[0]["indexed_query_page"]["epoch"]}
    assert result["matches"][1]["match"]["indexed_query_page"] is None
    assert result["producer"]["matcher"] == expected[0]["producer"]
    assert result["producer"]["module"] == batch.__name__


def test_unsupported_batch_observes_source_without_querying_catalog(native_current, monkeypatch):
    monkeypatch.setattr(CodebaseVerificationCatalog, "query_many_current",
        lambda *args, **kwargs: pytest.fail("unsupported request reached catalog"))
    result = match(native_current, (envelope(unsupported=True),))
    assert_batch(result, ["requirement:0"])
    assert result["inventory"] is None
    assert result["matches"][0]["match"]["status"] == "unknown"
    (native_current["repository"] / PATH).write_text("def increment(n: int) -> int:\n    return n + 1\n")
    with pytest.raises(StaleCodebaseError):
        match(native_current, (envelope(unsupported=True),))


@pytest.mark.parametrize("kind", ["list", "empty", "too_many", "tuple_subclass", "iterator"])
def test_exact_bounded_input_container_rejects_before_observation(native_current, monkeypatch, kind):
    class TupleSubclass(tuple):
        pass
    class Iterator:
        def __iter__(self):
            pytest.fail("untrusted iterable executed")
    values = {"list": [envelope()], "empty": (), "too_many": tuple(envelope(str(i)) for i in range(33)),
              "tuple_subclass": TupleSubclass((envelope(),)), "iterator": Iterator()}
    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current", lambda *args, **kwargs: pytest.fail("invalid batch observed"))
    with pytest.raises(batch.ConditionalCodebaseBatchError):
        match(native_current, values[kind])


@pytest.mark.parametrize("change", [
    {"extra": True}, {"requirement_id": "bad id"}, {"requirement_id": ""},
    {"requirement_id": True}, {"contract": {}}, {"domain": {}},
    {"verification_cid": "invalid"}, {"expected_key_id": "invalid"}, {"path": "../escape.py"},
])
def test_later_invalid_row_aborts_before_any_owner_observation(native_current, monkeypatch, change):
    invalid = envelope("second", unsupported=True)
    invalid.update(change)
    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current", lambda *args, **kwargs: pytest.fail("invalid row observed owner"))
    with pytest.raises(ValueError):
        match(native_current, (envelope(), invalid))


def test_duplicate_identity_and_missing_field_reject_before_observation(native_current, monkeypatch):
    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current", lambda *args, **kwargs: pytest.fail("invalid identity observed owner"))
    with pytest.raises(batch.ConditionalCodebaseBatchError):
        match(native_current, (envelope(), envelope()))
    missing = envelope()
    del missing["domain"]
    with pytest.raises(batch.ConditionalCodebaseBatchError):
        match(native_current, (missing,))


@pytest.mark.parametrize("controls", [
    {"timeout_seconds": True}, {"timeout_seconds": 0}, {"timeout_seconds": 601},
    {"timeout_seconds": float("nan")}, {"admission_timeout_seconds": True},
    {"admission_timeout_seconds": -1}, {"admission_timeout_seconds": float("inf")},
    {"memory_mb": True}, {"memory_mb": 63}, {"memory_mb": 4097}, {"cancel_event": object()},
])
def test_invalid_resource_controls_reject_before_observation(native_current, monkeypatch, controls):
    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current", lambda *args, **kwargs: pytest.fail("invalid controls observed owner"))
    with pytest.raises(batch.ConditionalCodebaseBatchError):
        match(native_current, **controls)


def test_aggregate_request_budget_applies_before_owner_access(native_current, monkeypatch):
    monkeypatch.setattr(batch, "MAX_REQUEST_BYTES", 32)
    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current", lambda *args, **kwargs: pytest.fail("oversized batch observed owner"))
    with pytest.raises(batch.ConditionalCodebaseBatchError, match="byte limit"):
        match(native_current)


@pytest.mark.parametrize("kind", ["list", "missing", "extra", "reversed", "inventory", "epoch"])
def test_returned_page_inventory_is_exact_ordered_and_shared(native_current, monkeypatch, kind):
    original = CodebaseVerificationCatalog.query_many_current

    def query(owner, *args, **kwargs):
        pages = original(owner, *args, **kwargs)
        if kind == "list":
            return list(pages)
        if kind == "missing":
            return pages[:1]
        if kind == "extra":
            return pages + (pages[0],)
        if kind == "reversed":
            return tuple(reversed(pages))
        if kind == "epoch":
            return pages[:1] + (replace(pages[1], epoch=pages[1].epoch + 1),)
        return pages[:1] + (replace(pages[1], inventory_cid=cid_for_structured({"wrong": "inventory"})),)

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_many_current", query)
    with pytest.raises(batch.ConditionalCodebaseBatchError):
        match(native_current, (envelope(), envelope("other", predicates=("n < 0",))))


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong@1"), ("head_cid", "bad-head"), ("selector_cid", "bad-selector"),
    ("page_cid", "bad-page"), ("start_cursor", {"wrong": True}),
    ("entries", [{"fabricated": True}]), ("authority", {"historical_conditional_evidence": True}),
])
def test_independent_page_receipt_binding_checks(native_current, monkeypatch, field, value):
    original = CodebaseVerificationQueryPage.to_dict
    monkeypatch.setattr(CodebaseVerificationQueryPage, "to_dict", lambda page: {**original(page), field: value})
    with pytest.raises(batch.ConditionalCodebaseBatchError, match="receipt lost"):
        match(native_current)


def test_source_edit_after_owner_batch_withholds_every_match(native_current, monkeypatch):
    original = CodebaseVerificationCatalog.query_many_current

    def query(owner, *args, **kwargs):
        pages = original(owner, *args, **kwargs)
        (native_current["repository"] / PATH).write_text("def increment(n: int) -> int:\n    return n + 1\n")
        return pages

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_many_current", query)
    with pytest.raises(StaleCodebaseError):
        match(native_current, (envelope(), envelope("second")))
    assert native_current["index"].current(VIEW) == native_current["expected_head"]


def test_cancellation_before_and_after_lookup_never_returns_partial_results(native_current, monkeypatch):
    event = threading.Event()
    original = CodebaseVerificationCatalog.query_many_current
    calls = []

    def query(owner, *args, **kwargs):
        calls.append(True)
        pages = original(owner, *args, **kwargs)
        event.set()
        return pages

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_many_current", query)
    event.set()
    with pytest.raises(LeaseCancelledError):
        match(native_current, cancel_event=event)
    assert not calls
    event.clear()
    with pytest.raises(LeaseCancelledError):
        match(native_current, cancel_event=event)
    assert calls == [True]


def test_owner_lookup_time_counts_toward_overall_deadline(native_current, monkeypatch):
    now = [batch.time.monotonic()]
    monkeypatch.setattr(batch, "time", SimpleNamespace(monotonic=lambda: now[0]))
    original = CodebaseVerificationCatalog.query_many_current

    def query(owner, *args, **kwargs):
        pages = original(owner, *args, **kwargs)
        now[0] += 4
        return pages

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_many_current", query)
    with pytest.raises(LeaseTimeoutError, match="batch deadline"):
        match(native_current, timeout_seconds=3)


def test_caller_envelope_mutation_cannot_change_owned_query(native_current, monkeypatch):
    requirements = (envelope(),)
    original = CodebaseVerificationCatalog.query_many_current

    def query(owner, *args, **kwargs):
        requirements[0]["path"] = "other.py"
        requirements[0]["requirement_id"] = "changed"
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_many_current", query)
    result = match(native_current, requirements)
    assert_batch(result, ["requirement:0"])
    assert result["matches"][0]["match"]["query"]["path"] == PATH


def test_output_size_limit_withholds_complete_batch(native_current, monkeypatch):
    monkeypatch.setattr(batch, "MAX_RESULT_BYTES", 64)
    with pytest.raises(batch.ConditionalCodebaseBatchError, match="byte limit"):
        match(native_current)


def test_lookup_does_not_launch_solver_or_version_processes(native_current, monkeypatch):
    import subprocess
    from pathlib import Path
    original = subprocess.run

    def git_only(command, *args, **kwargs):
        assert Path(str(command[0])).name == "git", "lookup launched a solver or version process"
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", git_only)
    assert_batch(match(native_current, (envelope(), envelope("second"))), ["requirement:0", "second"])


@pytest.mark.parametrize("cause", ["cancelled", "deadline"])
def test_preparation_checks_deadline_and_cancellation_between_native_rows(native_current, monkeypatch, cause):
    from ipfs_datasets_py.logic.intent_ir import decoder
    event = threading.Event()
    now = [batch.time.monotonic()]
    monkeypatch.setattr(batch, "time", SimpleNamespace(monotonic=lambda: now[0]))
    original = decoder.decode_intent_ir
    calls = []

    def decode(*args, **kwargs):
        calls.append(True)
        result = original(*args, **kwargs)
        if cause == "cancelled":
            event.set()
        else:
            now[0] += 4
        return result

    monkeypatch.setattr(decoder, "decode_intent_ir", decode)
    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current",
        lambda *args, **kwargs: pytest.fail("expired preparation observed owner"))
    with pytest.raises(LeaseCancelledError if cause == "cancelled" else LeaseTimeoutError):
        match(native_current, (envelope(), envelope("later")), timeout_seconds=3, cancel_event=event)
    assert calls == [True]


def test_aggregate_raw_input_budget_precedes_native_intent_decoding(native_current, monkeypatch):
    from ipfs_datasets_py.logic.intent_ir import decoder
    requirements = (envelope(), envelope("later"))
    sizes = []
    for item in requirements:
        inert = {**item, "intent_document": item["intent_document"].to_dict(),
            "contract": item["contract"].to_dict(), "domain": item["domain"].to_dict(),
            "source_identity": None, "verification_cid": None, "expected_key_id": None}
        sizes.append(len(json.dumps(inert, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("utf-8")))
    cap = max(sizes) + 1
    assert all(size < cap for size in sizes) and sum(sizes) > cap
    monkeypatch.setattr(batch, "MAX_REQUEST_BYTES", cap)
    monkeypatch.setattr(decoder, "decode_intent_ir", lambda *args, **kwargs: pytest.fail("oversized raw batch decoded"))
    with pytest.raises(batch.ConditionalCodebaseBatchError, match="byte limit"):
        match(native_current, requirements)


def test_all_input_rows_are_owned_before_first_native_decode(native_current, monkeypatch):
    from ipfs_datasets_py.logic.intent_ir import decoder
    requirements = (envelope(), envelope("later"))
    original = decoder.decode_intent_ir

    def decode(*args, **kwargs):
        requirements[1]["path"] = "other.py"
        requirements[1]["requirement_id"] = "changed"
        return original(*args, **kwargs)

    monkeypatch.setattr(decoder, "decode_intent_ir", decode)
    result = match(native_current, requirements)
    assert_batch(result, ["requirement:0", "later"])
    assert all(row["match"]["query"]["path"] == PATH for row in result["matches"])


@pytest.mark.parametrize("cause", ["cancelled", "deadline"])
def test_final_producer_check_is_inside_cancellation_deadline(native_current, monkeypatch, cause):
    event = threading.Event()
    now = [batch.time.monotonic()]
    monkeypatch.setattr(batch, "time", SimpleNamespace(monotonic=lambda: now[0]))
    original = batch._producer_pin
    calls = []

    def pin():
        calls.append(True)
        result = original()
        if len(calls) == 3:
            if cause == "cancelled":
                event.set()
            else:
                now[0] += 4
        return result

    monkeypatch.setattr(batch, "_producer_pin", pin)
    with pytest.raises(LeaseCancelledError if cause == "cancelled" else LeaseTimeoutError):
        match(native_current, timeout_seconds=3, cancel_event=event)
    assert len(calls) == 3
