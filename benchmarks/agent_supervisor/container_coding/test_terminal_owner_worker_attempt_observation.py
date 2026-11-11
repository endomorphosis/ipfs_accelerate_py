"""Authored child-event/receipt joins; no providers, native launches or proofs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_owner_worker_attempt_observation as collector
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as supervisor


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _chain(attempt="a" * 32, *, spawned=True, returncode=7):
    # These complete authored events test the collector, not native PID custody.
    common = dict(schema=collector.EVENT_SCHEMA, attempt_id=attempt, purpose="coding",
        native_prompt_sha256=_sha(b"private original task"),
        workspace_identity_sha256=_sha(b"/private/allocated-checkout"),
        boundary_manifest_sha256="b" * 64, absolute_deadline_ns=90000000000,
        owner_pid=123, owner_start_ticks=456, observation_only=True,
        execution_authority=False, completion_authority=False, retry_authority=False)
    prepared = dict(common, event="prepared")
    first = dict(common, event="spawned", prepared_sha256=_sha(_raw(prepared)),
        child_pid=789, child_start_ticks=101112) if spawned else None
    retired = dict(common, event="retired", prepared_sha256=_sha(_raw(prepared)),
        spawned_sha256=None if first is None else _sha(_raw(first)),
        child_pid=None if first is None else first["child_pid"],
        child_start_ticks=None if first is None else first["child_start_ticks"],
        waited=spawned, child_returncode=returncode if spawned else None,
        cleanup_succeeded=True, brokers_closed=True)
    events = dict(prepared=prepared, retired=retired)
    if first is not None:
        events["spawned"] = first
    receipt = dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        invocation_id="private-invocation-" + attempt, owner_worker_attempt_id=attempt,
        native_prompt_sha256=common["native_prompt_sha256"], workspace="/private/allocated-checkout",
        status="failed" if returncode else "provider_returned", router_calls=1,
        native_rollout_usage=dict(usage=dict(input_tokens=100, output_tokens=5, total_tokens=105)))
    return events, receipt


@pytest.fixture
def evidence():
    # Match existing protected-reader fixtures: /tmp has writable ancestors.
    with tempfile.TemporaryDirectory(prefix="owner-worker-census-", dir=Path.home()) as name:
        parent = Path(name); parent.chmod(0o750)
        events, receipt = _chain()
        report = dict(owner_worker_attempt_census_expected=True, provider_invocations=[receipt])

        def write(values=events):
            paths = {}
            for event, row in values.items():
                path = parent / ("worker-attempt-" + row["attempt_id"] + "-" + event + ".json")
                if path.exists() and not path.is_symlink():
                    path.chmod(0o600)
                path.write_bytes(_raw(row)); path.chmod(0o400)
                paths[event] = path
            return paths

        paths = write()
        yield dict(parent=parent, events=events, receipt=receipt, report=report, write=write, paths=paths)


def _collect(evidence):
    return collector.collect(parent=evidence["parent"], report=evidence["report"])


@pytest.mark.parametrize("returncode", [0, 7, -15])
def test_waited_success_and_failed_worker_receipts_have_complete_census_without_usage_claims(evidence, returncode):
    events, receipt = _chain(returncode=returncode)
    evidence["write"](events); evidence["report"]["provider_invocations"] = [receipt]
    result = _collect(evidence)
    assert result["status"] == "observed" and result["all_attempts_receipted"] is True
    assert result["counts"] == dict(observed_attempts=1, prepared_attempts=1, spawned_attempts=1,
        retired_attempts=1, explicit_no_spawn_attempts=0, matched_router_receipts=1, unresolved_attempts=0)
    assert result["attempts"][0]["prepared_sha256"] == _sha(_raw(events["prepared"]))
    assert result["provider_tokens"] is None and result["provider_usage_complete_established"] is False
    assert result["total_token_savings_measured"] is False
    assert all(result[key] is False for key in ("execution_authority", "completion_authority",
        "retry_authority", "settlement_authority", "proof_authority"))
    assert evidence["report"]["provider_invocations"] == [receipt]
    text = _raw(result)
    assert b"private" not in text and b"allocated-checkout" not in text
    assert b"private-invocation" not in text and b'"owner_pid"' not in text


def test_second_failed_child_missing_router_receipt_cannot_hide_behind_first_charge(evidence):
    second, _receipt = _chain("c" * 32)
    evidence["write"](second)
    original = deepcopy(evidence["report"]["provider_invocations"])
    supervisor._collect_owner_worker_attempt_census(evidence["report"], parent=evidence["parent"])
    census = evidence["report"]["owner_worker_attempt_census"]
    assert census["all_attempts_receipted"] is False
    assert census["counts"]["spawned_attempts"] == 2
    assert census["counts"]["matched_router_receipts"] == 1
    assert "missing_router_receipt" in census["issues"]
    assert evidence["report"]["unreceipted_provider_attempt"]["usage"] is None
    assert evidence["report"]["provider_invocations"] == original
    assert original[0]["native_rollout_usage"]["usage"]["total_tokens"] == 105


def test_two_distinct_attempts_retain_all_failed_receipts(evidence):
    events, receipt = _chain("c" * 32)
    evidence["write"](events)
    evidence["report"]["provider_invocations"].append(receipt)
    result = _collect(evidence)
    assert result["all_attempts_receipted"] is True
    assert result["counts"]["spawned_attempts"] == result["counts"]["matched_router_receipts"] == 2
    assert len(evidence["report"]["provider_invocations"]) == 2


def test_explicit_no_spawn_retirement_does_not_invent_a_provider_call(evidence):
    events, _receipt = _chain(spawned=False)
    evidence["paths"]["spawned"].unlink(); evidence["write"](events)
    evidence["report"]["provider_invocations"] = []
    result = _collect(evidence)
    assert result["all_attempts_receipted"] is True
    assert result["counts"]["spawned_attempts"] == result["counts"]["matched_router_receipts"] == 0
    assert result["counts"]["explicit_no_spawn_attempts"] == 1
    assert "provider_calls" not in result
    assert evidence["report"]["provider_invocations"] == []


def test_missing_expected_census_is_unknown_even_when_one_router_receipt_survives(evidence):
    for path in evidence["paths"].values(): path.unlink()
    result = _collect(evidence)
    assert result["all_attempts_receipted"] is False
    assert "expected_census_missing" in result["issues"]
    assert "router_without_owner_events" in result["issues"]


@pytest.mark.parametrize("expected", [None, True, 1, "true", [], {}])
def test_missing_or_malformed_selected_census_never_becomes_known_zero(evidence, expected):
    evidence["report"]["owner_worker_attempt_census_expected"] = expected
    for path in evidence["paths"].values(): path.unlink()
    supervisor._collect_owner_worker_attempt_census(evidence["report"], parent=evidence["parent"])
    assert evidence["report"]["owner_worker_attempt_census"]["all_attempts_receipted"] is False
    assert evidence["report"]["unreceipted_provider_attempt"]["usage"] is None


@pytest.mark.parametrize("selection", ["absent", False])
def test_legacy_and_unselected_routes_do_not_read_files_or_add_report_fields(evidence, monkeypatch, selection):
    if selection == "absent": evidence["report"].pop("owner_worker_attempt_census_expected")
    else: evidence["report"]["owner_worker_attempt_census_expected"] = False
    monkeypatch.setattr(collector, "_events", lambda *args: pytest.fail("unselected census read"))
    before = deepcopy(evidence["report"])
    assert _collect(evidence)["status"] == "not_selected"
    supervisor._collect_owner_worker_attempt_census(evidence["report"], parent=evidence["parent"])
    assert evidence["report"] == before


@pytest.mark.parametrize("marker,complete", [(None, True), (False, True), ({}, False), ({"usage": None}, False)])
def test_prior_unknown_coverage_survives_but_false_sentinel_is_not_unknown(evidence, marker, complete):
    evidence["report"]["unreceipted_provider_attempt"] = marker
    assert _collect(evidence)["all_attempts_receipted"] is complete


def test_identical_receipt_reprints_are_benign(evidence):
    evidence["report"]["provider_invocations"].append(deepcopy(evidence["receipt"]))
    assert _collect(evidence)["all_attempts_receipted"] is True
    assert _collect(evidence)["counts"]["matched_router_receipts"] == 1


@pytest.mark.parametrize("change", ["usage", "bool_int"])
def test_conflicting_receipt_reprints_remain_visible_and_invalidate_usage(evidence, change):
    duplicate = deepcopy(evidence["receipt"])
    if change == "usage": duplicate["native_rollout_usage"]["usage"]["total_tokens"] = 205
    else: duplicate["router_calls"] = True  # Python dict equality would hide this contradiction.
    supervisor._collect_provider_usage_rows(evidence["report"], [duplicate, deepcopy(duplicate)])
    assert evidence["report"]["provider_invocations"] == [evidence["receipt"]]
    assert evidence["report"]["provider_invocation_conflicts"] == [duplicate]
    assert evidence["report"]["unreceipted_provider_attempt"]["usage"] is None
    result = _collect(evidence)
    assert result["all_attempts_receipted"] is False
    assert "conflicting_invocation_receipts" in result["issues"]


@pytest.mark.parametrize("change", ["prompt", "workspace", "attempt", "purpose", "status", "missing_id", "two_invocations"])
def test_foreign_or_ambiguous_receipt_cannot_cover_a_child(evidence, change):
    row = evidence["receipt"]
    if change == "prompt": row["native_prompt_sha256"] = "e" * 64
    elif change == "workspace": row["workspace"] += "-other"
    elif change == "attempt": row["owner_worker_attempt_id"] = "e" * 32
    elif change == "purpose": row["purpose"] = "planning"
    elif change == "status": row["status"] = "unknown"
    elif change == "missing_id": row.pop("owner_worker_attempt_id")
    elif change == "two_invocations":
        other = deepcopy(row); other["invocation_id"] += "-other"
        evidence["report"]["provider_invocations"].append(other)
    assert _collect(evidence)["all_attempts_receipted"] is False


@pytest.mark.parametrize("change", [
    "missing_prepared", "missing_spawned", "missing_retired", "extra_field", "wrong_purpose",
    "boolean_deadline", "boolean_owner", "authority_claim", "wrong_prepared_pin", "wrong_spawned_pin",
    "wrong_child", "unwaited", "boolean_returncode", "unknown_returncode", "cleanup_failed", "brokers_open",
    "foreign_workspace", "foreign_prompt", "foreign_owner", "foreign_deadline", "no_spawn_contradiction",
])
def test_event_binding_and_retirement_failures_stay_unknown(evidence, change):
    events = evidence["events"]
    if change.startswith("missing_"):
        evidence["paths"][change.removeprefix("missing_")].unlink()
    else:
        if change == "extra_field": events["prepared"]["extra"] = False
        elif change == "wrong_purpose": events["prepared"]["purpose"] = "planning"
        elif change == "boolean_deadline": events["prepared"]["absolute_deadline_ns"] = True
        elif change == "boolean_owner": events["prepared"]["owner_pid"] = True
        elif change == "authority_claim": events["prepared"]["execution_authority"] = 1
        elif change == "wrong_prepared_pin": events["retired"]["prepared_sha256"] = "e" * 64
        elif change == "wrong_spawned_pin": events["retired"]["spawned_sha256"] = "e" * 64
        elif change == "wrong_child": events["retired"]["child_pid"] += 1
        elif change == "unwaited":
            events["retired"].update(waited=False, child_returncode=None)
        elif change == "boolean_returncode": events["retired"]["child_returncode"] = True
        elif change == "unknown_returncode": events["retired"]["child_returncode"] = None
        elif change == "cleanup_failed": events["retired"]["cleanup_succeeded"] = False
        elif change == "brokers_open": events["retired"]["brokers_closed"] = False
        elif change == "foreign_workspace": events["retired"]["workspace_identity_sha256"] = "e" * 64
        elif change == "foreign_prompt": events["retired"]["native_prompt_sha256"] = "e" * 64
        elif change == "foreign_owner": events["retired"]["owner_start_ticks"] += 1
        elif change == "foreign_deadline": events["retired"]["absolute_deadline_ns"] += 1
        elif change == "no_spawn_contradiction":
            events["retired"].update(spawned_sha256=None, child_pid=None,
                child_start_ticks=None, child_returncode=None, waited=False)
        evidence["write"]()
    assert _collect(evidence)["all_attempts_receipted"] is False


@pytest.mark.parametrize("change", ["symlink", "hardlink", "mode", "parent_mode", "oversized",
    "duplicate_key", "noncanonical", "bad_name", "too_many_attempts", "too_many_entries", "too_many_receipts"])
def test_unsafe_malformed_and_excessive_evidence_is_unknown(evidence, monkeypatch, change):
    path = evidence["paths"]["prepared"]
    if change == "symlink":
        original = path.with_name("original.json"); path.rename(original); path.symlink_to(original.name)
    elif change == "hardlink": os.link(path, path.with_name("other.json"))
    elif change == "mode": path.chmod(0o440)
    elif change == "parent_mode": evidence["parent"].chmod(0o770)
    elif change in {"oversized", "duplicate_key", "noncanonical"}:
        raw = {"oversized": b"x" * (collector.MAX_EVENT_BYTES + 1),
            "duplicate_key": b'{"schema":"one","schema":"two"}',
            "noncanonical": path.read_bytes() + b"\n"}[change]
        path.chmod(0o600); path.write_bytes(raw); path.chmod(0o400)
    elif change == "bad_name": (evidence["parent"] / "worker-attempt-invalid.json").write_text("{}")
    elif change == "too_many_attempts":
        monkeypatch.setattr(collector, "MAX_ATTEMPTS", 1)
        events, _receipt = _chain("e" * 32); evidence["write"](events)
    elif change == "too_many_entries":
        monkeypatch.setattr(collector, "MAX_DIRECTORY_ENTRIES", 3)
        (evidence["parent"] / "other.json").write_text("{}")
    elif change == "too_many_receipts":
        monkeypatch.setattr(collector, "MAX_RECEIPTS", 1)
        evidence["report"]["provider_invocations"].append(deepcopy(evidence["receipt"]))
    assert _collect(evidence)["all_attempts_receipted"] is False


def test_explicit_no_spawn_cannot_cover_a_router_receipt(evidence):
    events, _receipt = _chain(spawned=False)
    evidence["paths"]["spawned"].unlink(); evidence["write"](events)
    result = _collect(evidence)
    assert result["all_attempts_receipted"] is False
    assert "receipt_without_child_birth" in result["issues"]


def test_events_changing_after_reads_invalidate_the_census(evidence, monkeypatch):
    original = collector._read
    changed = [False]
    def read(fd, name, **kwargs):
        raw = original(fd, name, **kwargs)
        if name.endswith("-retired.json") and not changed[0]:
            changed[0] = True
            path = evidence["paths"]["prepared"]
            path.chmod(0o600); path.write_bytes(path.read_bytes() + b" "); path.chmod(0o400)
        return raw
    monkeypatch.setattr(collector, "_read", read)
    assert _collect(evidence)["all_attempts_receipted"] is False


def test_planning_receipts_do_not_require_worker_events(evidence):
    planning = dict(schema="router-implementation-invocation@1", phase="planning", purpose="planning",
        invocation_id="separate-planning", status="provider_returned")
    evidence["report"]["provider_invocations"].append(planning)
    assert _collect(evidence)["all_attempts_receipted"] is True


def test_raw_planning_and_coding_log_reprints_preserve_existing_phases_and_complete_census(evidence):
    planning = dict(schema="router-implementation-invocation@1", phase="planning", purpose="planning",
        invocation_id="separate-planning", status="provider_returned")
    evidence["report"]["provider_invocations"].insert(0, planning)
    original = deepcopy(evidence["report"]["provider_invocations"])
    raw_planning, raw_coding = (deepcopy(row) for row in original)
    raw_planning.pop("phase"); raw_coding.pop("phase")
    supervisor._collect_provider_usage_rows(evidence["report"], [raw_planning, raw_coding])
    assert evidence["report"]["provider_invocations"] == original
    assert len(evidence["report"]["provider_invocations"]) == 2
    assert "provider_invocation_conflicts" not in evidence["report"]
    assert "unreceipted_provider_attempt" not in evidence["report"]
    assert _collect(evidence)["all_attempts_receipted"] is True


def test_explicit_contradictory_planning_log_phase_is_retained_as_conflict(evidence):
    planning = dict(schema="router-implementation-invocation@1", phase="planning", purpose="planning",
        invocation_id="separate-planning", status="provider_returned")
    evidence["report"]["provider_invocations"].insert(0, planning)
    contradictory = dict(planning, phase="coding")
    supervisor._collect_provider_usage_rows(evidence["report"], [contradictory])
    assert evidence["report"]["provider_invocation_conflicts"] == [contradictory]
    assert evidence["report"]["unreceipted_provider_attempt"]["usage"] is None
    assert _collect(evidence)["all_attempts_receipted"] is False


def test_ordinary_receipt_collection_keeps_distinct_rounds_and_ignores_exact_reprints(evidence):
    second, row = _chain("c" * 32)
    evidence["write"](second)
    supervisor._collect_provider_usage_rows(evidence["report"],
        [deepcopy(evidence["receipt"]), row, deepcopy(row)])
    assert len(evidence["report"]["provider_invocations"]) == 2
    assert "provider_invocation_conflicts" not in evidence["report"]
    assert _collect(evidence)["all_attempts_receipted"] is True


@pytest.mark.parametrize("error", [RuntimeError("private failure details"), KeyboardInterrupt()])
def test_census_failure_preserves_outcome_charges_and_marks_unknown_before_interrupt(evidence, monkeypatch, error):
    report = evidence["report"]
    report.update(task_completed=False, error={"type": "original_provider_failure"})
    original = deepcopy(report)
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(collector, "collect", fail)
    if isinstance(error, Exception):
        supervisor._collect_owner_worker_attempt_census(report, parent=evidence["parent"])
    else:
        with pytest.raises(KeyboardInterrupt):
            supervisor._collect_owner_worker_attempt_census(report, parent=evidence["parent"])
    assert report["task_completed"] is False and report["error"] == original["error"]
    assert report["provider_invocations"] == original["provider_invocations"]
    assert report["owner_worker_attempt_census"]["all_attempts_receipted"] is False
    assert report["unreceipted_provider_attempt"]["usage"] is None
    assert b"private failure" not in _raw(report["owner_worker_attempt_census"])
