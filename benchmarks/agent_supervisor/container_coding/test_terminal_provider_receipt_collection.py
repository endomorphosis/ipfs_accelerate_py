"""Local receipt scan gaps survive good rows without becoming an attempt census."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver


def native(identity="coding", **fields):
    row = dict(schema="router-implementation-invocation@1", invocation_id=identity,
        provider="codex_cli", model="authored-model", purpose="coding", status="provider_returned",
        external_container_boundary=True, router_calls=1, native_rollout_usage=None)
    row.update(fields)
    return row


def report(**fields):
    value = dict(provider_invocations=[], coding_dispatch_possible=True, implementation_route="model_router")
    value.update(fields)
    return value


def log(path, *rows):
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return raw


def scan(value, state):
    result = driver._collect_provider_usage_logs(value, state=state)
    assert result is value["provider_receipt_collection"]
    assert set(result["counts"]) == set(driver.PROVIDER_RECEIPT_COLLECTION_COUNTERS)
    assert all(type(count) is int and count >= 0 for count in result["counts"].values())
    assert result["attempt_population_complete"] is result["provider_call_count"] is None
    assert result["campaign_call_population_complete"] is None
    assert result["observation_only"] is True
    assert all(result[field] is False for field in driver.PROVIDER_RECEIPT_COLLECTION_FALSE)
    return result


def test_clean_scan_deduplicates_copies_and_keeps_diagnostics_separate(tmp_path):
    first = native(prompt="PRIVATE_SOURCE_TEXT", workspace="/private/source")
    planning = native("planning", purpose="planning")
    log(tmp_path / "first.log", first, planning)
    raw = log(tmp_path / "second.log", first)
    with (tmp_path / "second.log").open("ab") as stream:
        stream.write(b"ordinary diagnostic output\n")
    value = report()
    result = scan(value, tmp_path)
    assert result["status"] == "observed" and result["complete"] is True
    assert result["reason_codes"] == []
    assert result["counts"]["log_files_seen"] == result["counts"]["log_files_read"] == 2
    assert result["counts"]["provider_observations"] == 3
    assert result["counts"]["bytes_read"] > len(raw)
    assert [row["phase"] for row in value["provider_invocations"]] == ["coding", "planning"]
    assert len(value["provider_invocations"]) == 2
    assert "unreceipted_provider_attempt" not in value
    for private in ("PRIVATE_SOURCE_TEXT", "/private/source", str(tmp_path)):
        assert private not in json.dumps(result)


def test_oversized_retry_log_marks_gap_even_after_good_receipt_and_false_marker(tmp_path):
    log(tmp_path / "good.log", native())
    with (tmp_path / "retry.log").open("wb") as stream:
        stream.truncate(driver.MAX_PROVIDER_USAGE_LOG_BYTES + 1)
    value = report(unreceipted_provider_attempt=False)
    result = scan(value, tmp_path)
    assert result["status"] == "incomplete" and result["complete"] is False
    assert result["counts"]["log_files_seen"] == 2 and result["counts"]["log_files_read"] == 1
    assert result["counts"]["oversized_logs"] == 1
    assert result["reason_codes"] == ["oversized_log"]
    assert len(value["provider_invocations"]) == 1
    assert value["unreceipted_provider_attempt"] == dict(phase="coding", usage=None,
        reason="provider_receipt_log_scan_incomplete")


def test_unreadable_log_preserves_good_receipt(tmp_path, monkeypatch):
    log(tmp_path / "good.log", native())
    denied = tmp_path / "denied.log"
    log(denied, native("unavailable"))
    actual = driver.os.open
    def open_file(path, flags, *args, **kwargs):
        if Path(path) == denied:
            assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
            raise PermissionError("PRIVATE_ERROR_TEXT")
        return actual(path, flags, *args, **kwargs)
    monkeypatch.setattr(driver.os, "open", open_file)
    value = report()
    result = scan(value, tmp_path)
    assert result["counts"]["unreadable_logs"] == 1 and result["complete"] is False
    assert [row["invocation_id"] for row in value["provider_invocations"]] == ["coding"]
    assert "PRIVATE_ERROR_TEXT" not in json.dumps(result)


@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink", "broken_link", "hardlink", "cyclic_link"])
def test_nonregular_logs_are_refused_without_reading_or_blocking(tmp_path, kind):
    log(tmp_path / "good.log", native())
    candidate = tmp_path / "nonregular.log"
    if kind == "fifo":
        os.mkfifo(candidate)
    elif kind == "directory":
        candidate.mkdir()
    elif kind == "hardlink":
        source = tmp_path / "hardlink-source.txt"
        source.write_text("private copied bytes")
        os.link(source, candidate)
    elif kind == "cyclic_link":
        candidate.symlink_to(candidate)
    else:
        target = tmp_path / "target.txt"
        if kind == "symlink":
            target.write_text("private linked bytes")
        candidate.symlink_to(target)
    value = report()
    result = scan(value, tmp_path)
    assert result["counts"]["nonregular_logs"] == 1 and result["complete"] is False
    assert result["counts"]["log_files_read"] == 1
    assert len(value["provider_invocations"]) == 1


def test_symlink_subtree_cannot_hide_an_unread_retry_log(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    log(state / "good.log", native())
    outside = tmp_path / "outside"
    outside.mkdir()
    log(outside / "hidden.log", native("hidden-retry"))
    (state / "linked-directory").symlink_to(outside, target_is_directory=True)
    value = report()
    result = scan(value, state)
    assert result["counts"]["directory_errors"] == 1 and result["complete"] is False
    assert result["counts"]["log_files_seen"] == 1
    assert [row["invocation_id"] for row in value["provider_invocations"]] == ["coding"]
    assert isinstance(value["unreceipted_provider_attempt"], dict)


def test_changed_log_snapshot_is_not_parsed(tmp_path, monkeypatch):
    log(tmp_path / "good.log", native())
    changing = tmp_path / "changing.log"
    log(changing, native("changing-retry"))
    actual = Path.lstat
    reads = 0
    def lstat(path, *args, **kwargs):
        nonlocal reads
        value = actual(path, *args, **kwargs)
        if path == changing:
            reads += 1
            if reads == 2:
                names = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
                changed = {key: getattr(value, key) for key in names}
                changed["st_mtime_ns"] += 1
                return SimpleNamespace(**changed)
        return value
    monkeypatch.setattr(Path, "lstat", lstat)
    value = report()
    result = scan(value, tmp_path)
    assert result["counts"]["unstable_logs"] == 1 and result["complete"] is False
    assert [row["invocation_id"] for row in value["provider_invocations"]] == ["coding"]
    assert result["counts"]["bytes_read"] > 0


@pytest.mark.parametrize("constant,limit,reason", [
    ("MAX_PROVIDER_USAGE_ENTRIES", 1, "entry_limit"),
    ("MAX_PROVIDER_USAGE_LOGS", 1, "file_limit"),
    ("MAX_PROVIDER_USAGE_LINES", 1, "line_limit"),
    ("MAX_PROVIDER_USAGE_OBSERVATIONS", 1, "observation_limit"),
])
def test_bounded_scan_preserves_admitted_rows_and_marks_partial_population(tmp_path, monkeypatch, constant, limit, reason):
    log(tmp_path / "a.log", native("first"))
    log(tmp_path / "z.log", native("second"))
    monkeypatch.setattr(driver, constant, limit)
    value = report()
    result = scan(value, tmp_path)
    assert result["complete"] is False and reason in result["reason_codes"]
    assert result["counts"]["bound_exceeded"] == 1
    assert len(value["provider_invocations"]) == 1
    assert isinstance(value["unreceipted_provider_attempt"], dict)


def test_total_byte_limit_retains_receipts_from_earlier_snapshot(tmp_path, monkeypatch):
    raw = log(tmp_path / "a.log", native("first"))
    log(tmp_path / "z.log", native("second"))
    monkeypatch.setattr(driver, "MAX_PROVIDER_USAGE_TOTAL_BYTES", len(raw) + 1)
    value = report()
    result = scan(value, tmp_path)
    assert result["reason_codes"] == ["byte_limit"] and result["complete"] is False
    assert result["counts"]["bytes_read"] == len(raw)
    assert [row["invocation_id"] for row in value["provider_invocations"]] == ["first"]


def test_oversized_line_keeps_other_receipts_with_incomplete_coverage(tmp_path, monkeypatch):
    raw = log(tmp_path / "a.log", native())
    monkeypatch.setattr(driver, "MAX_PROVIDER_USAGE_LINE_BYTES", len(raw))
    (tmp_path / "z.log").write_bytes(b"diagnostic" * len(raw) + b"\n")
    value = report()
    result = scan(value, tmp_path)
    assert result["counts"]["oversized_lines"] == 1 and result["complete"] is False
    assert len(value["provider_invocations"]) == 1


@pytest.mark.parametrize("malformed", [b'{"schema":', b"[", b"{" + b"x" * 100])
def test_json_looking_truncated_lines_cannot_be_a_clean_scan(tmp_path, malformed):
    raw = log(tmp_path / "readable.log", native())
    (tmp_path / "readable.log").write_bytes(raw + malformed)
    value = report()
    result = scan(value, tmp_path)
    assert result["counts"]["malformed_json_candidates"] == 1 and result["complete"] is False
    assert len(value["provider_invocations"]) == 1


@pytest.mark.parametrize("route,possible", [("doctor_candidate", True), ("model_router", False)])
def test_scan_gap_does_not_invent_provider_dispatch_on_provider_free_paths(tmp_path, route, possible):
    (tmp_path / "invalid.log").write_bytes(b'{"schema":')
    value = report(implementation_route=route, coding_dispatch_possible=possible)
    result = scan(value, tmp_path)
    assert result["complete"] is False
    assert "unreceipted_provider_attempt" not in value


def test_empty_clean_scan_cannot_prove_zero_attempts(tmp_path):
    value = report()
    result = scan(value, tmp_path)
    assert result["complete"] is True and result["counts"]["provider_observations"] == 0
    assert "unreceipted_provider_attempt" not in value


def test_conflicting_same_identity_rows_remain_available_to_existing_refusal(tmp_path):
    log(tmp_path / "calls.log", native(), native(status="failed", error_type="AuthoredFailure"))
    value = report()
    result = scan(value, tmp_path)
    assert result["complete"] is True
    assert len(value["provider_invocations"]) == 2
    with pytest.raises(ValueError, match="conflicting provider invocation"):
        comparison._invocations(value)


def test_semantic_dialogue_and_doctor_rows_keep_their_existing_collectors(tmp_path):
    call = native()
    dialogue = dict(schema="supervisor-semantic-decision-dialogue@1", session_id="authored-dialogue",
                    attempts=[dict(provider_receipt={"native_invocation": call})])
    doctor = dict(schema="native-doctor-candidate-materialization@1", task_cid="authored-task")
    log(tmp_path / "calls.log", call, dialogue, doctor)
    value = report(doctor_dispatch={"task_cid": "authored-task"})
    result = scan(value, tmp_path)
    assert result["complete"] is True
    assert value["semantic_decision_dialogues"] == [dialogue]
    assert len(value["provider_invocations"]) == 1
    assert value["doctor_invocations"] == [doctor]
    assert result["counts"]["provider_observations"] == 2 and result["counts"]["doctor_observations"] == 1


@pytest.mark.parametrize("declaration", [None, [], {"task_cid": []}, {}])
def test_optional_doctor_metadata_cannot_break_collection_or_admit_unbound_rows(tmp_path, declaration):
    doctor = dict(schema="native-doctor-candidate-materialization@1", task_cid=None)
    log(tmp_path / "calls.log", native(), doctor)
    value = report(doctor_dispatch=declaration)
    result = scan(value, tmp_path)
    assert result["complete"] is True and len(value["provider_invocations"]) == 1
    assert result["counts"]["doctor_observations"] == 0
    assert "doctor_invocations" not in value


def test_late_collection_error_keeps_already_retained_receipts(tmp_path, monkeypatch):
    log(tmp_path / "calls.log", native())
    original = driver._collect_provider_usage_rows
    def collect(value, observations):
        original(value, observations)
        raise ValueError("PRIVATE_COLLECTOR_ERROR")
    monkeypatch.setattr(driver, "_collect_provider_usage_rows", collect)
    value = report()
    result = scan(value, tmp_path)
    assert len(value["provider_invocations"]) == 1
    assert result["counts"]["collection_errors"] == 1 and result["complete"] is False
    assert "PRIVATE_COLLECTOR_ERROR" not in json.dumps(result)


def test_shutdown_interruption_is_not_swallowed_by_log_collection(tmp_path, monkeypatch):
    log(tmp_path / "calls.log", native())
    def interrupt(value, observations):
        raise KeyboardInterrupt
    monkeypatch.setattr(driver, "_collect_provider_usage_rows", interrupt)
    with pytest.raises(KeyboardInterrupt):
        driver._collect_provider_usage_logs(report(), state=tmp_path)


@pytest.mark.parametrize("stdout", ["", '{"schema":', "[]", "[" * 2000 + "]" * 2000])
@pytest.mark.parametrize("prior", [False, None])
def test_missing_or_malformed_planner_reply_marks_a_possible_attempt(stdout, prior):
    value = report(unreceipted_provider_attempt=prior)
    with pytest.raises(ValueError):
        driver._retain_planner_reply(value, stdout, missing_reason="missing_or_malformed_final_receipt")
    assert value["provider_invocations"] == []
    assert value["unreceipted_provider_attempt"] == dict(phase="planning", usage=None,
        reason="missing_or_malformed_final_receipt")


@pytest.mark.parametrize("prior", [True, {"phase": "planning", "usage": None, "reason": "earlier_gap"}])
def test_new_scan_gap_preserves_an_existing_failure_reason(tmp_path, prior):
    (tmp_path / "bad.log").write_bytes(b'{"schema":')
    value = report(unreceipted_provider_attempt=deepcopy(prior))
    scan(value, tmp_path)
    assert value["unreceipted_provider_attempt"] == prior


def test_failed_planner_receipt_is_retained_without_becoming_a_missing_attempt():
    value = report()
    receipt = native("planning", purpose="planning", status="failed")
    text, observed = driver._retain_planner_reply(value, json.dumps(receipt) + "\nauthored output\n",
                                                 missing_reason="interrupted_without_receipt")
    assert text == "authored output" and observed == receipt
    assert value["provider_invocations"] == [{"phase": "planning", **receipt}]
    assert "unreceipted_provider_attempt" not in value


def test_planner_decoder_preserves_prior_gap_and_shutdown_signals(monkeypatch):
    earlier = dict(phase="coding", usage=None, reason="earlier_gap")
    value = report(unreceipted_provider_attempt=deepcopy(earlier))
    with pytest.raises(ValueError):
        driver._retain_planner_reply(value, "", missing_reason="interrupted_without_receipt")
    assert value["unreceipted_provider_attempt"] == earlier
    def interrupt(stdout):
        raise KeyboardInterrupt
    monkeypatch.setattr(driver, "_router_reply", interrupt)
    with pytest.raises(KeyboardInterrupt):
        driver._retain_planner_reply(value, "", missing_reason="interrupted_without_receipt")
