"""Supervisor materializes current-bound SPAR clause records. Not completion."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.semantic_state.spar_accepted_root import (
    CLAUSE_EVIDENCE_SCHEMA,
    REQUIRED_CLAUSES,
    _producer_rejected_clause_record,
    admit_current_bound_clause_records,
    materialize_clause_records,
)


def _subject() -> dict:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/spar-accepted-root-subject@1",
        "profile_cid": "profile:sealed",
        "source_forest_root": "sha256:" + "a" * 64,
        "task_receipt_cids": [f"receipt:{i}" for i in range(51)],
        "goal_contract_cids": [f"contract:{i}" for i in range(32)],
        "semantic_acceptance_authority": False,
        "completion_authority": False,
    }


def _current(**overrides) -> dict:
    current = {
        "current_rollout_mode": "bootstrap",
        "required_gate_task": "SPAR-043",
        "required_gate_receipt_cid": "baguqeera-spar043",
        "runtime_settled": True,
        "merge_queue_empty": True,
        "reports": [
            {
                "path": "benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json",
                "available": True,
                "content_digest": "sha256:" + "c" * 64,
                "can_authorize_completion": False,
                "writes_repository": False,
                "zero_safety_floors": {"false_task_completions": 0, "rollback_failure": 0},
            }
        ],
    }
    current.update(overrides)
    return current


def test_materialize_four_current_bound_records():
    records = materialize_clause_records(_subject(), _current())
    assert set(records) == set(REQUIRED_CLAUSES)
    for name, row in records.items():
        assert row["schema"] == CLAUSE_EVIDENCE_SCHEMA
        assert row["clause"] == name
        assert row["nomination_only"] is False
        assert row["source_forest_root"] == "sha256:" + "a" * 64
        assert row["payload_cid"].startswith("sha256:")
        assert row["subject_digest"].startswith("sha256:")


def test_cwd_benchmark_report_mints_safety_floors(tmp_path, monkeypatch):
    report = tmp_path / "benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        '{"can_authorize_completion": false, "writes_repository": false, '
        '"zero_safety_floors": {"false_task_completions": 0}}'
    )
    monkeypatch.chdir(tmp_path)
    records = materialize_clause_records(
        _subject(),
        _current(reports=[], required_gate_receipt_cid="", runtime_settled=False, merge_queue_empty=False),
    )
    assert "safety_floors_noncompensable_accepted" in records


def test_completion_boolean_on_the_report_cannot_mint_safety_floors():
    current = _current()
    current["reports"][0]["can_authorize_completion"] = True
    records = materialize_clause_records(_subject(), current)
    assert "safety_floors_noncompensable_accepted" not in records


def test_missing_runtime_settlement_cannot_mint_fixed_point():
    records = materialize_clause_records(
        _subject(), _current(runtime_settled=False, merge_queue_empty=False)
    )
    assert "fixed_point_accepted" not in records
    assert "required_mode_roots_accepted" in records


def test_supervisor_admits_materialized_current_bound_records():
    subject = _subject()
    records = materialize_clause_records(subject, _current())
    outcomes = admit_current_bound_clause_records(subject, records)
    assert outcomes is not None
    assert all(row["accepted"] is True for row in outcomes.values())
    assert all(row["evidence_cid"].startswith("sha256:") for row in outcomes.values())


def test_forest_mismatch_record_cannot_admit():
    subject = _subject()
    records = materialize_clause_records(subject, _current())
    records["fixed_point_accepted"]["source_forest_root"] = "sha256:" + "b" * 64
    assert admit_current_bound_clause_records(subject, records) is None


def test_producer_record_rejection_is_honored():
    raw = {
        "clause_outcomes": {
            "fixed_point_accepted": {
                "accepted": False,
                "reason": "clause_record_is_nomination_only",
            }
        }
    }
    assert _producer_rejected_clause_record(raw) is True
    stub = {
        "clause_outcomes": {
            name: {"accepted": False, "reason": "current_source_clause_evidence_unavailable"}
            for name in REQUIRED_CLAUSES
        }
    }
    assert _producer_rejected_clause_record(stub) is False


def test_missing_spar043_receipt_cannot_mint_required_mode_roots():
    records = materialize_clause_records(
        _subject(), _current(required_gate_receipt_cid="")
    )
    assert "required_mode_roots_accepted" not in records
    assert "self_hosted_capstone_accepted" not in records
