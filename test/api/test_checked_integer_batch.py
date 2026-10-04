"""Complete requirement ledgers over native bounded owner batches."""
from dataclasses import replace
import json
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import checked_integer_batch as subject
from ipfs_accelerate_py.agent_supervisor.planning.checked_integer_codebase import STATEMENT_ID
from test.api.test_checked_integer_codebase import NATIVE_SOLVERS, TEXT, native

SECOND = "Under python-integer-offset@1, delta.py::delta(n) must return n + 3."
UNSUPPORTED = "Under python-integer-offset@1, square.py::square(n) must return n + 1."


def _requirements():
    rows = subject.build_integer_offset_requirements([TEXT, SECOND, UNSUPPORTED, TEXT, TEXT])
    rows[-1]["intent_document"]["statements"][0].update(predicate="repair", arguments=["agent", "counter.py"])
    return rows


def test_complete_pure_ledger_preserves_order_duplicates_and_unsupported_atoms():
    requirements = _requirements()
    ledger = subject.prepare_integer_requirement_ledger(requirements)
    assert ledger["input_scope"] == "explicit_requirement_envelopes_only"
    assert ledger["requirement_count"] == 5 and ledger["complete_input_ledger"] is True
    assert [row["requirement_id"] for row in ledger["requirements"]] == [row["requirement_id"] for row in requirements]
    assert ledger["requirements"][0]["query"]["contract_cid"] == ledger["requirements"][3]["query"]["contract_cid"]
    assert ledger["requirements"][-1]["query"]["supported"] is False
    assert all(row["native_requirement_ids"] == [STATEMENT_ID] for row in ledger["requirements"])
    assert all(ledger[name] is False for name in ("proof_authority", "execution_authority", "behavior_authority"))
    requirements[0]["intent_document"]["statements"][0]["arguments"][0] = "changed.py"
    assert ledger["requirements"][0]["query"]["contract"]["path"] == "counter.py"


def test_additional_native_requirements_are_all_kept_residual():
    rows = subject.build_integer_offset_requirements([TEXT])
    extra = dict(rows[0]["intent_document"]["statements"][0], statement_id="extra-goal")
    rows[0]["intent_document"]["statements"].append(extra)
    ledger = subject.prepare_integer_requirement_ledger(rows)
    assert ledger["requirements"][0]["query"]["supported"] is False
    assert ledger["requirements"][0]["native_requirement_ids"] == ["extra-goal", STATEMENT_ID]


@pytest.mark.parametrize("defect", ["empty", "oversized", "generator", "duplicate-id", "extra-field", "missing", "locator-id"])
def test_malformed_or_unbounded_inputs_reject_before_owner_access(defect):
    rows = subject.build_integer_offset_requirements([TEXT, TEXT])
    if defect == "empty":
        rows = []
    elif defect == "oversized":
        rows *= 17
    elif defect == "generator":
        rows = iter(rows)
    elif defect == "duplicate-id":
        rows[1]["requirement_id"] = rows[0]["requirement_id"]
    elif defect == "extra-field":
        rows[0]["checked_result"] = {"status": "proved"}
    elif defect == "missing":
        del rows[0]["statement_id"]
    else:
        rows[0]["requirement_id"] = "/private/repository"
    with pytest.raises(subject.CheckedIntegerBatchError):
        subject.prepare_integer_requirement_ledger(rows)


@pytest.fixture
def batch_native(native):
    root = native["repository"]
    (root / "delta.py").write_text("def delta(n: int) -> int:\n    return n + 2\n")
    (root / "square.py").write_text("def square(n: int) -> int:\n    return n * n\n")
    publication = native["index"].prepare_current(
        root, repository_id="worktree:integer", operation_id="batch-sources", expected_head=native["head"],
        scheduler=native["scheduler"], limits=native["limits"],
    )
    native["head"] = publication.head
    return native


def _match(native, **kwargs):
    controls = {"requirements": _requirements(), "timeout_seconds": 30, "max_workers": 2, "memory_mb": 1536}
    controls.update(kwargs)
    return subject.match_checked_integer_requirements(
        index=native["index"], repository=native["repository"], repository_id="worktree:integer",
        scheduler=native["scheduler"], **controls)


@NATIVE_SOLVERS
def test_native_batch_yields_one_complete_ledger_and_one_owner_call(batch_native, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
        PlanCreateMaterials, freeze_plan_create_input_snapshot,
    )
    from test.api.test_plan_create_semantic_input_identity import _request

    original = CodebaseIntegerVerifier.verify_many
    calls = []
    def record(self, *args, **kwargs):
        calls.append([item.to_dict() for item in kwargs["contracts"]])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify_many", record)
    result = _match(batch_native)
    assert len(calls) == 1 and len(calls[0]) == 4
    assert calls[0][0] == calls[0][3]
    assert [row["status"] for row in result["requirement_results"]] == [
        "conditional_contract_matched", "conditional_contract_refuted", "unknown",
        "conditional_contract_matched", "unknown",
    ]
    assert result["conditional_summary"] == {"total": 5, "matched": 2, "refuted": 1, "unknown": 2}
    assert len(result["residual_requirements"]) == 5
    assert result["conditional_residual_requirement_ids"] == ["requirement:0001", "requirement:0002", "requirement:0004"]
    assert all(row["runtime_behavior_status"] == "unresolved" for row in result["requirement_results"])
    for name in ("current_facts", "current_behavioral_facts", "runtime_refutations", "removed_task_ids"):
        assert result[name] == []
    for name in ("proof_authority", "execution_authority", "completion_authority", "behavior_authority"):
        assert result[name] is False
    assert str(batch_native["repository"]) not in json.dumps(result)
    materials = PlanCreateMaterials(extra={"checked_integer_requirements": result})
    assert materials.current_facts == () and materials.current_roots is None
    frozen = freeze_plan_create_input_snapshot(_request(), materials=materials)
    assert frozen.material_binding["reuse_supported"] is True
    assert frozen == freeze_plan_create_input_snapshot(_request(), materials=materials)
    state = batch_native["scheduler"].snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0


@NATIVE_SOLVERS
def test_native_durable_index_records_are_historical_not_current_facts(batch_native):
    from ipfs_datasets_py.duckdb_control.codebase_evidence_index import CodebaseEvidenceIndex

    result = _match(batch_native)
    batch = result["verification_batch"]
    assert batch["evidence_receipt_cids"]
    evidence = CodebaseEvidenceIndex(batch_native["index"].catalog)
    for identity in batch["evidence_receipt_cids"]:
        record = evidence.get(identity, expected_head=batch_native["head"])
        assert record is not None
        value = record.to_dict()
        assert value["authority"] == "historical_conditional"
        assert value["requires_fresh_native_checks"] is True
        assert value["requires_current_source_observation"] is True
    assert result["current_facts"] == []


def test_all_unsupported_requirements_skip_checking_without_dropping_native_residuals(batch_native, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

    rows = subject.build_integer_offset_requirements([TEXT])
    extra = dict(rows[0]["intent_document"]["statements"][0], statement_id="extra-goal")
    rows[0]["intent_document"]["statements"].append(extra)
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify_many", lambda *args, **kwargs: pytest.fail("unsupported batch reached prover"))
    result = _match(batch_native, requirements=rows)
    assert result["status"] == "unknown" and result["verification_batch"] is None
    assert result["residual_requirements"] == [
        {"requirement_id": "requirement:0000", "statement_id": "extra-goal", "status": "runtime_behavior_unresolved"},
        {"requirement_id": "requirement:0000", "statement_id": STATEMENT_ID, "status": "runtime_behavior_unresolved"},
    ]


@NATIVE_SOLVERS
@pytest.mark.parametrize("defect", ["missing", "reordered", "head", "requested-contract", "authority", "positive-without-replay"])
def test_batch_coverage_and_bindings_fail_closed(batch_native, monkeypatch, defect):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

    original = CodebaseIntegerVerifier.verify_many
    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if defect == "missing":
            result["results"].pop()
        elif defect == "reordered":
            result["results"][0], result["results"][1] = result["results"][1], result["results"][0]
        elif defect == "head":
            result["head"]["generation"] += 1
        elif defect == "requested-contract":
            result["requested_contracts"][0]["offset"] += 1
        elif defect == "authority":
            result["behavior_authority"] = True
        else:
            result["results"][0]["solver_replayed"] = False
        return result
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify_many", corrupt)
    with pytest.raises(subject.CheckedIntegerBatchError):
        _match(batch_native)


@NATIVE_SOLVERS
def test_source_edit_after_native_batch_withholds_whole_match(batch_native, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

    original = CodebaseIntegerVerifier.verify_many
    checked = []
    def drift(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        checked.append(result["results"][0]["status"])
        batch_native["source"].write_text("def increment(n: int) -> int:\n    return n + 9\n")
        return result
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify_many", drift)
    returned = "withheld"
    with pytest.raises(ValueError):
        returned = _match(batch_native)
    assert checked == ["proved"]
    assert returned == "withheld"
    assert batch_native["scheduler"].snapshot()["active_lease_count"] == 0


@NATIVE_SOLVERS
@pytest.mark.parametrize("interrupt", ["cancel", "deadline"])
def test_native_result_does_not_escape_cancellation_or_overall_deadline(batch_native, monkeypatch, interrupt):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError

    cancel = threading.Event()
    now = [subject.time.monotonic()]
    class Clock:
        @staticmethod
        def monotonic():
            return now[0]
    monkeypatch.setattr(subject, "time", Clock)
    original = CodebaseIntegerVerifier.verify_many
    def interrupt_after(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if interrupt == "cancel":
            cancel.set()
        else:
            now[0] += 31
        return result
    monkeypatch.setattr(CodebaseIntegerVerifier, "verify_many", interrupt_after)
    with pytest.raises(LeaseCancelledError if interrupt == "cancel" else LeaseTimeoutError):
        _match(batch_native, cancel_event=cancel)
    assert batch_native["scheduler"].snapshot()["active_lease_count"] == 0


def test_external_pressure_suppresses_batch_before_admission(batch_native):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError

    batch_native["host"][0] = replace(batch_native["host"][0], memory_stall_percent=8)
    with pytest.raises(LeaseTimeoutError):
        _match(batch_native, admission_timeout_seconds=0.01)
    state = batch_native["scheduler"].snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0
