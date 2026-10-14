"""Bounded collection of authored owner audits; no provider or lake access."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_numerical_retrieval_observation as collector
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as protocol


def _digest(value):
    return hashlib.sha256(protocol.canonical_json(value)).hexdigest()


def _fixture(session="a" * 32):
    descriptor = dict(schema=protocol.DESCRIPTOR_SCHEMA, session_id=session,
        owner_uid=os.geteuid(), worker_uid=os.geteuid() + 1, owner_pid=99999,
        owner_start_time_ticks=123, workspace="/allocated/private-candidate",
        task_id="sample", task_cid="task:sample", context_bundle_sha256="b" * 64,
        context_receipt_sha256="c" * 64, source_tree_id="sha256:tree",
        boundary_manifest_sha256="d" * 64, socket_path="/owner/closed.sock",
        socket_device=1, socket_inode=2, issued_monotonic_ns=1, expires_monotonic_ns=1_000_000_001,
        max_requests=64, max_total_response_bytes=524288,
        proof_authority=False, execution_authority=False, completion_authority=False)
    digest = _digest(descriptor)
    audit = dict(schema="supervisor-numerical-obligation-session@1", session=descriptor,
        descriptor_sha256=digest, requests=[
            dict(ordinal=1, status="ok", request_sha256="1" * 64, response_sha256="2" * 64,
                 response_bytes=2048, returned_rows=2, delivered=True, elapsed_ns=150),
            dict(ordinal=2, status="retrieval_refused", request_sha256="3" * 64, response_sha256="4" * 64,
                 response_bytes=233, returned_rows=0, delivered=False, elapsed_ns=250),
            dict(ordinal=3, status="invalid_request", request_sha256=None, response_sha256=None,
                 response_bytes=0, returned_rows=0, delivered=False, elapsed_ns=300)],
        accepted_requests=3, inflight_request=None, response_bytes=2281,
        service_stopped=True, accounting_complete=True, retrieval_provider_calls=0,
        source_obligations_discharged=0, proof_authority=False, execution_authority=False,
        completion_authority=False)
    observation = dict(schema="numerical-obligation-dispatch-observation@1",
        descriptor_sha256=digest, session_id=session,
        **{key: descriptor[key] for key in collector._JOIN_FIELDS},
        requests_observed=False, proof_authority=False, execution_authority=False, completion_authority=False)
    row = dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        invocation_id="invocation-" + session, workspace=descriptor["workspace"],
        numerical_obligation_retrieval=observation,
        external_container_boundary=dict(manifest_sha256=descriptor["boundary_manifest_sha256"],
            owner_uid=descriptor["owner_uid"], worker_uid=descriptor["worker_uid"]),
        source_formalization_context={key: descriptor[key] for key in
            ("task_id", "task_cid", "context_bundle_sha256")})
    return audit, row


@pytest.fixture
def evidence():
    with tempfile.TemporaryDirectory(prefix="numerical-collector-", dir=Path.home()) as name:
        parent = Path(name)
        parent.chmod(0o750)
        audit, row = _fixture()
        scope = dict(task_alias="sample", task_cid="task:sample", baseline_commit="0" * 40, outputs=[])
        report = dict(worker_cleanup_returncode=0, provider_invocations=[row])
        def write(value=audit):
            path = parent / ("numerical-" + value["session"]["session_id"] + "-audit.json")
            if path.exists():
                path.chmod(0o600)
            path.write_bytes(protocol.canonical_json(value))
            path.chmod(0o400)
            return path
        path = write()
        yield parent, scope, report, audit, row, write, path


def _collect(evidence):
    parent, scope, report, *_ = evidence
    return collector.collect(parent=parent, scope=scope, report=report)


def test_complete_owner_audit_collects_exact_counts_without_private_text(evidence):
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"] == dict(sessions=1, accepted_requests=3, delivered_responses=1,
        prepared_pages=1, delivered_pages=1, prepared_rows=2, delivered_rows=2,
        response_bytes=2281, delivered_response_bytes=2048,
        request_hash_count=2, response_hash_count=2,
        unique_request_hash_count=2, unique_response_hash_count=2)
    assert result["sessions"][0]["status_counts"] == {"ok": 1, "retrieval_refused": 1, "invalid_request": 1}
    raw = protocol.canonical_json(result).decode()
    assert "/allocated/" not in raw and "/owner/" not in raw and '"requests"' not in raw
    assert result["provider_tokens"] is None and result["token_estimate"] is False
    assert result["proof_authority"] is result["task_correctness_established"] is False


def test_closed_selected_session_with_no_requests_is_a_known_zero(evidence):
    *_, audit, row, write, path = evidence
    audit.update(requests=[], accepted_requests=0, response_bytes=0)
    write()
    result = _collect(evidence)
    assert result["status"] == "observed" and result["counts"]["accepted_requests"] == 0


def test_no_numerical_observation_is_not_selected_without_private_io(monkeypatch):
    monkeypatch.setattr(collector, "_open_directory", lambda *_: pytest.fail("unexpected owner IO"))
    result = collector.collect(parent=Path("/nonexistent"), scope=None,
        report={"provider_invocations": [{"phase": "planning"}]})
    assert result["status"] == "not_selected" and result["counts"] is None


@pytest.mark.parametrize("unreceipted", [False, True])
def test_selected_dispatch_without_router_receipt_is_unknown_not_unselected(monkeypatch, unreceipted):
    monkeypatch.setattr(collector, "_open_directory", lambda *_: pytest.fail("missing receipt reached owner IO"))
    report = dict(provider_invocations=[], numerical_formalization_selection={"enabled": True},
                  coding_dispatch_possible=True)
    if unreceipted:
        report["unreceipted_provider_attempt"] = {"phase": "coding", "usage": None}
    result = collector.collect(parent=Path("/nonexistent"), scope=None, report=report)
    assert result["status"] == "unavailable"
    assert result["reason"] == "selected_session_receipt_unavailable" and result["counts"] is None


@pytest.mark.parametrize("cleanup", [None, 1, False, "0"])
def test_cleanup_must_be_exact_success_before_private_io(evidence, monkeypatch, cleanup):
    evidence[2]["worker_cleanup_returncode"] = cleanup
    monkeypatch.setattr(collector, "_open_directory", lambda *_: pytest.fail("cleanup failure reached owner IO"))
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None


@pytest.mark.parametrize("change", ["missing", "inflight", "not_stopped", "incomplete", "sum", "bool_count",
    "bool_bytes", "bool_ordinal", "bool_rows", "bool_elapsed", "bool_flag", "provider_call", "discharged",
    "ordinal", "extra", "unhashed_response", "zero_hashed_response", "undocumented_delivery", "refusal_rows",
    "oversized_response", "unknown_status", "raw_query"])
def test_missing_incomplete_or_malformed_audit_never_yields_false_zero(evidence, change):
    parent, scope, report, audit, row, write, path = evidence
    if change == "missing": path.unlink()
    else:
        if change == "inflight": audit["inflight_request"] = 1
        elif change == "not_stopped": audit["service_stopped"] = False
        elif change == "incomplete": audit["accounting_complete"] = False
        elif change == "sum": audit["response_bytes"] += 1
        elif change == "bool_count": audit["accepted_requests"] = True
        elif change == "bool_bytes": audit["response_bytes"] = False
        elif change == "bool_ordinal": audit["requests"][0]["ordinal"] = True
        elif change == "bool_rows": audit["requests"][0]["returned_rows"] = False
        elif change == "bool_elapsed": audit["requests"][0]["elapsed_ns"] = False
        elif change == "bool_flag": audit["proof_authority"] = 0
        elif change == "provider_call": audit["retrieval_provider_calls"] = 1
        elif change == "discharged": audit["source_obligations_discharged"] = True
        elif change == "ordinal": audit["requests"][1]["ordinal"] = 1
        elif change == "extra": audit["extra"] = "unexpected"
        elif change == "unhashed_response": audit["requests"][0]["response_sha256"] = None
        elif change == "zero_hashed_response": audit["requests"][0]["response_bytes"] = 0
        elif change == "undocumented_delivery": audit["requests"][2]["delivered"] = True
        elif change == "refusal_rows": audit["requests"][1]["returned_rows"] = 1
        elif change == "oversized_response": audit["requests"][0]["response_bytes"] = 8193
        elif change == "unknown_status": audit["requests"][0]["status"] = "private error text"
        elif change == "raw_query": audit["requests"][0]["query"] = "source body"
        write()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None and result["sessions"] == []


@pytest.mark.parametrize("change", ["task", "alias", "descriptor_hash", "receipt", "bundle", "source_tree",
    "workspace", "boundary", "worker_uid", "formalization", "session", "phase", "duplicate_session", "duplicate_invocation"])
def test_foreign_or_ambiguous_task_router_binding_is_refused(evidence, change):
    parent, scope, report, audit, row, write, path = evidence
    observed = row["numerical_obligation_retrieval"]
    if change == "task": scope["task_cid"] = "another:task"
    elif change == "alias": scope["task_alias"] = "another-task"
    elif change == "descriptor_hash": observed["descriptor_sha256"] = "f" * 64
    elif change == "receipt": observed["context_receipt_sha256"] = "f" * 64
    elif change == "bundle": observed["context_bundle_sha256"] = "f" * 64
    elif change == "source_tree": observed["source_tree_id"] = "another:tree"
    elif change == "workspace": row["workspace"] = "/another/workspace"
    elif change == "boundary": row["external_container_boundary"]["manifest_sha256"] = "f" * 64
    elif change == "worker_uid": row["external_container_boundary"]["worker_uid"] += 1
    elif change == "formalization": row["source_formalization_context"]["task_id"] = "another-task"
    elif change == "session": observed["session_id"] = "../not-a-session"
    elif change == "phase": row["phase"] = "planning"
    elif change == "duplicate_session":
        extra = deepcopy(row); extra["invocation_id"] += "other"
        report["provider_invocations"].append(extra)
    else:
        report["provider_invocations"].append(deepcopy(row))
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None


@pytest.mark.parametrize("change", ["symlink", "hardlink", "writable", "group_readable", "parent_writable",
                                     "oversized", "duplicate_json", "nonfinite_json", "noncanonical_json"])
def test_private_file_boundary_and_strict_parser(evidence, change):
    parent, scope, report, audit, row, write, path = evidence
    if change == "symlink":
        other = parent / "other.json"
        path.rename(other); path.symlink_to(other)
    elif change == "hardlink": os.link(path, parent / "other.json")
    elif change == "writable": path.chmod(0o600)
    elif change == "group_readable": path.chmod(0o440)
    elif change == "parent_writable": parent.chmod(0o770)
    else:
        path.chmod(0o600)
        if change == "oversized": raw = b"x" * 65537
        elif change == "duplicate_json": raw = b'{"schema":"x","schema":"x"}'
        elif change == "nonfinite_json": raw = b'{"extra":NaN}'
        else: raw = json.dumps(audit, indent=2).encode()
        path.write_bytes(raw); path.chmod(0o400)
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None


def test_multisession_aggregation_retains_every_trial_and_refuses_partial_success(evidence):
    parent, scope, report, audit, row, write, path = evidence
    second, extra = _fixture("e" * 32)
    second_path = write(second)
    report["provider_invocations"].append(extra)
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"]["sessions"] == 2 and result["counts"]["accepted_requests"] == 6
    assert result["counts"]["response_bytes"] == 4562
    assert result["counts"]["request_hash_count"] == 4 and result["counts"]["unique_request_hash_count"] == 2
    second_path.unlink()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None and result["sessions"] == []


def test_session_population_cap_refuses_before_opening_audit(evidence, monkeypatch):
    evidence[2]["provider_invocations"] *= 17
    monkeypatch.setattr(collector, "_open_directory", lambda *_: pytest.fail("overlarge session population reached IO"))
    assert _collect(evidence)["status"] == "unavailable"


def test_replacement_during_private_read_is_unavailable(evidence, monkeypatch):
    original = collector._read
    def replaced(fd, relative, **kwargs):
        raw = original(fd, relative, **kwargs)
        path = evidence[0] / relative
        path.unlink()
        path.write_bytes(raw)
        path.chmod(0o400)
        return raw
    monkeypatch.setattr(collector, "_read", replaced)
    assert _collect(evidence)["status"] == "unavailable"
