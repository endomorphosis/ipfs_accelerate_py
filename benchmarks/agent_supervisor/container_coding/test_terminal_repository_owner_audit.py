"""Authored, pure historical records; no native admission or provider claims."""
from copy import deepcopy
import hashlib
import json
import os
import stat

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as bindings
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def authored_owner_evidence(*, session_id="1" * 32, invocation_id="9" * 32):
    """Reusable bytes/row fixture; all history is authored, never live authority."""
    descriptor = dict(schema=protocol.DESCRIPTOR_SCHEMA, session_id=session_id,
        owner_uid=os.getuid(), worker_uid=os.getuid() + 1, owner_pid=123,
        owner_start_time_ticks=1, workspace="/authored/candidate", boundary_manifest_sha256="a" * 64,
        socket_path="/authored/owner/session.sock", socket_device=1, socket_inode=2,
        issued_monotonic_ns=1, expires_monotonic_ns=60_000_000_001,
        max_requests=64, max_total_response_bytes=524288,
        proof_authority=False, execution_authority=False, completion_authority=False,
        representation="compact", config_sha256="b" * 64, helper_sha256="c" * 64, read_only=True)
    identity = dict(resolved_path="/authored/python3", sha256="d" * 64, device=1, inode=2,
        size=4096, mode=stat.S_IFREG | 0o755, uid=os.getuid(), gid=os.getgid(), mtime_ns=1, ctime_ns=1)
    sources = {name: str(index) * 64 for index, name in enumerate(bindings._SOURCES, 5)}
    binding = dict(schema=bindings.SCHEMA, requested_policy=bindings.POLICIES[1],
        workspace=descriptor["workspace"], descriptor_path="/authored/owner/descriptor.json",
        descriptor_sha256=_sha(protocol.canonical_json(descriptor)), descriptor=descriptor,
        session_id=descriptor["session_id"], representation="compact", owner_config_sha256="b" * 64,
        owner_helper_sha256="c" * 64, python_executable="/authored/python3", python_identity=identity,
        helper_script="/authored/runtime/repository_source_owner_stdio.py", helper_sha256=sources[bindings._SOURCES[0]],
        helper_sources=sources, read_only=True, **dict.fromkeys(bindings._FALSE_FIELDS, False))
    binding["config"] = bindings._config(binding)
    binding["config_sha256"] = _sha(protocol.canonical_json(binding["config"]))
    instruction = bindings._instruction(binding).encode()
    binding.update(instruction_sha256=_sha(instruction), instruction_bytes=len(instruction),
                   usage=bindings.unavailable_usage())
    operations = [dict(schema="repository-source-mcp-audit@1", sequence=1,
        event="initialize", tool=None, success=True, returned_bytes=0, compact=False,
        canonical_changed=False, canonical_uncertain=False, refreshed=False),
        dict(schema="repository-source-mcp-audit@1", sequence=2, event="tool",
        tool="repository_read", success=True, returned_bytes=120, compact=True,
        canonical_changed=False, canonical_uncertain=False, refreshed=False)]
    raw = b"".join(protocol.canonical_json(record) + b"\n" for record in operations)
    requests = [dict(ordinal=index, status="ok", request_sha256=str(index + 1) * 64,
        prepared_response_sha256=str(index + 3) * 64, prepared_response_bytes=240,
        response_sha256=str(index + 3) * 64, delivered=True, response_bytes=240,
        elapsed_ns=1_000_000) for index in (1, 2)]
    child = dict(lease_id="2" * 32, parent_lease_id="3" * 32, lane="validation",
        cpu_slots=1, memory_mb=512, gpu_memory_mb=0, unified_memory_mb=0,
        child_process_slots=1, requires_gpu=False, owner_pid=descriptor["owner_pid"],
        acquired_at=1_700_000_000., wait_seconds=0.01, released=True, cancelled=False)
    session = dict(schema="repository-source-owner-session@1", session=deepcopy(descriptor),
        descriptor_sha256=binding["descriptor_sha256"], requests=requests, accepted_requests=2,
        inflight_request=None, response_bytes=480, service_stopped=True, cleanup_complete=True,
        cleanup_pending=False, accounting_complete=True, native_child=child,
        native_child_acquired=True, native_child_acquisition_uncertain=False, native_child_released=True,
        service_error=None, cleanup_error=None, construction_error=None, thread_launch_attempted=True,
        join_error=None, prepared_mcp_usage={**mcp.summarize_repository_audit_bytes(raw),
            "store_cleanup_complete": True, "artifacts_retained_after_invocation": False},
        retained_mcp_audit=dict(name="repository-mcp-" + descriptor["session_id"] + "-audit-operations.jsonl",
            sha256=_sha(raw), bytes=len(raw), content_free=True), canonical_source_read_only=True,
        cross_uid_deployment_qualified=False, automatic_terminal_wiring=False, provider_calls=0,
        token_savings_measured=False, proof_authority=False, execution_authority=False, completion_authority=False)
    boundary = dict(schema="supervisor-container-worker-boundary@1", container_id="e" * 64,
        image_id="sha256:" + "f" * 64, namespaces={name: "authored-" + name for name in ("pid", "mnt", "net")},
        owner_uid=descriptor["owner_uid"], worker_uid=descriptor["worker_uid"], purpose="coding",
        manifest_sha256="a" * 64, completion_authority=False)
    row = dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        provider="codex_cli", invocation_id=invocation_id, status="provider_returned", completion_authority=False,
        workspace=descriptor["workspace"], repository_source_policy=binding,
        external_container_boundary=boundary, usage=dict(codex_repository_source_configured=True,
            codex_repository_source_config_sha256=binding["config_sha256"]))
    return dict(audit_raw=protocol.canonical_json(session), operations_raw=raw, row=row)


def _session(evidence):
    return json.loads(evidence["audit_raw"])


def _set_session(evidence, session):
    evidence["audit_raw"] = protocol.canonical_json(session)


def _set_operations(evidence, rows):
    raw = b"".join(protocol.canonical_json(row) + b"\n" for row in rows)
    session = _session(evidence)
    session["retained_mcp_audit"].update(sha256=_sha(raw), bytes=len(raw))
    session["prepared_mcp_usage"] = {**mcp.summarize_repository_audit_bytes(raw),
        "store_cleanup_complete": True, "artifacts_retained_after_invocation": False}
    evidence["operations_raw"] = raw
    _set_session(evidence, session)


def test_recounts_prepared_operations_without_claiming_consumption():
    evidence = authored_owner_evidence()
    before = deepcopy(evidence)
    result = audit.summarize_owner_session(**evidence)
    assert evidence == before
    assert result["schema"] == audit.SCHEMA and result["status"] == "joined_complete"
    assert result["prepared_mcp_usage"]["successful_reads"] == 1
    assert result["transport"]["completed_frame_sends"] == 2
    assert result["transport"]["reserved_response_bytes"] == 480
    assert result["owner_config_sha256"] != result["worker_config_sha256"]
    assert result["consumption"]["delivered_reads"] is None
    assert result["consumption"]["provider_tokens"] is None
    assert result["cleanup"]["native_parent_released"] is None
    assert result["request_response_payloads_recomputed"] is False
    output = protocol.canonical_json(result)
    assert b"/authored" not in output and evidence["row"]["invocation_id"].encode() not in output
    assert b"owner_uid" not in output and b"request_sha256" not in output


def test_failed_invocation_and_completed_service_error_remain_joinable():
    evidence = authored_owner_evidence()
    evidence["row"].update(status="failed", usage=None)
    session = _session(evidence)
    session["service_error"] = "LeaseCancelledError"
    _set_session(evidence, session)
    result = audit.summarize_owner_session(**evidence)
    assert result["worker_configuration_observed"] is None
    assert result["cleanup"]["service_error_reported"] is True


def test_real_empty_operations_bytes_establish_only_prepared_zero():
    evidence = authored_owner_evidence()
    _set_operations(evidence, [])
    result = audit.summarize_owner_session(**evidence)
    assert result["operations_bytes"] == 0
    assert result["prepared_mcp_usage"]["status"] == "not_observed"
    assert result["prepared_mcp_usage"]["tool_calls"] == 0
    assert result["consumption"]["delivered_reads"] is None


def test_reserved_failed_send_remains_reserved_and_prepared_not_delivered():
    evidence = authored_owner_evidence()
    session = _session(evidence)
    session["requests"][1].update(status="transport_error", delivered=False)
    _set_session(evidence, session)
    result = audit.summarize_owner_session(**evidence)
    assert result["transport"]["reserved_response_bytes"] == 480
    assert result["transport"]["completed_frame_send_bytes"] == 240
    assert result["prepared_mcp_usage"]["successful_reads"] == 1


def test_prepared_operation_can_precede_missing_prepared_transport_record():
    evidence = authored_owner_evidence()
    session = _session(evidence)
    session["requests"][1].update(status="stale_context", delivered=False, prepared_response_sha256=None,
        prepared_response_bytes=None, response_sha256=None, response_bytes=0)
    session["response_bytes"] = 240
    _set_session(evidence, session)
    result = audit.summarize_owner_session(**evidence)
    assert result["transport"]["prepared_responses"] == 1
    assert result["prepared_mcp_usage"]["tool_calls"] == 1


def test_budget_fallback_has_distinct_prepared_and_reserved_digests():
    evidence = authored_owner_evidence()
    session = _session(evidence)
    session["requests"][1].update(status="budget_exhausted", prepared_response_bytes=524289,
                                  prepared_response_sha256="a" * 64)
    _set_session(evidence, session)
    result = audit.summarize_owner_session(**evidence)
    assert result["transport"]["prepared_final_changes"] == 1
    assert result["transport"]["reserved_response_bytes"] == 480


def test_duplicate_request_can_receive_closed_invalid_request_refusal():
    evidence = authored_owner_evidence()
    # Duplicate requests are accounted as attempts but do not invoke the MCP
    # handler a second time. Neither hash uniqueness nor operation order is a join.
    rows = [json.loads(line) for line in evidence["operations_raw"].splitlines()]
    _set_operations(evidence, rows[:1])
    session = _session(evidence)
    session["requests"][1].update(status="invalid_request",
        request_sha256=session["requests"][0]["request_sha256"])
    _set_session(evidence, session)
    result = audit.summarize_owner_session(**evidence)
    assert result["transport"]["completed_frame_sends"] == 2
    assert result["transport"]["status_counts"]["invalid_request"] == 1
    assert result["prepared_mcp_usage"]["tool_calls"] == 0


@pytest.mark.parametrize("kind", ["read", "retrieval"])
def test_native_aggregate_observations_are_recomputed_and_preserved(kind):
    evidence = authored_owner_evidence()
    rows = [json.loads(line) for line in evidence["operations_raw"].splitlines()]
    if kind == "read":
        rows[1].update(schema="repository-source-mcp-audit@3", read_observation=dict(
            schema="repository-source-read-event@1", generation=1, source_identity_sha256="a" * 64,
            scope="file", representation="compact", encoding="utf-8", content_sha256=_sha(b"abc"),
            content_bytes=3))
    else:
        rows[1].update(schema="repository-source-mcp-audit@2", tool="repository_context_query", compact=False,
            retrieval=dict(requested_generation=1, binding_sha256="d" * 64, returned_items=2, error=None))
    _set_operations(evidence, rows)
    result = audit.summarize_owner_session(**evidence)
    observation = result["prepared_mcp_usage"][kind + "_observation"]
    assert observation["complete"] is True
    if kind == "read":
        assert observation["identified_reads"] == 1
        assert observation["content_bytes"] == 3
    else:
        assert observation["successful_queries"] == 1
        assert observation["candidates_returned"] == 2
    assert result["consumption"]["delivered_reads"] is None


def test_historical_join_does_not_use_files_or_live_descriptor(monkeypatch):
    evidence = authored_owner_evidence()
    def forbidden(*args, **kwargs):
        raise AssertionError("historical analysis accessed live state")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(protocol, "load_descriptor", forbidden)
    monkeypatch.setattr(bindings, "_file_identity", forbidden)
    monkeypatch.setattr(mcp, "summarize_repository_audit", forbidden)
    assert audit.summarize_owner_session(**evidence)["status"] == "joined_complete"


@pytest.mark.parametrize("field,value", [("released", False), ("owner_pid", 999),
    ("cpu_slots", True), ("memory_mb", 513), ("requires_gpu", True),
    ("lane", "training"), ("acquired_at", -1), ("wait_seconds", True)])
def test_native_child_evidence_cannot_authorize_or_change_the_fixed_demand(field, value):
    evidence = authored_owner_evidence()
    session = _session(evidence)
    session["native_child"][field] = value
    _set_session(evidence, session)
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("phase", "planning"), ("purpose", "planning"), ("provider", "other"),
    ("invocation_id", ""), ("invocation_id", "9" * 33), ("invocation_id", True),
    ("status", "not_started"), ("workspace", "/foreign"), ("completion_authority", True)])
def test_wrong_invocation_refused(field, value):
    evidence = authored_owner_evidence()
    evidence["row"][field] = value
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("field,value", [
    ("service_stopped", False), ("cleanup_complete", False), ("cleanup_pending", True),
    ("accounting_complete", False), ("native_child_acquired", None), ("native_child_released", None),
    ("native_child_acquisition_uncertain", True), ("cleanup_error", "OSError"), ("join_error", "RuntimeError"),
    ("thread_launch_attempted", False), ("canonical_source_read_only", False),
    ("cross_uid_deployment_qualified", True), ("automatic_terminal_wiring", True),
    ("provider_calls", True), ("provider_calls", 1), ("token_savings_measured", True),
    ("proof_authority", True), ("execution_authority", True), ("completion_authority", True),
    ("service_error", "secret /raw/path"), ("accepted_requests", 1), ("accepted_requests", True),
    ("inflight_request", 2), ("response_bytes", 479), ("native_child", None)])
def test_incomplete_or_inconsistent_session_refused(field, value):
    evidence = authored_owner_evidence()
    session = _session(evidence)
    session[field] = value
    _set_session(evidence, session)
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("field,value", [
    ("ordinal", True), ("ordinal", 2), ("status", "unknown"), ("status", []),
    ("delivered", 1), ("elapsed_ns", -1), ("elapsed_ns", True), ("request_sha256", None),
    ("request_sha256", "z" * 64), ("prepared_response_sha256", None), ("prepared_response_bytes", None),
    ("prepared_response_bytes", 239), ("response_sha256", None), ("response_bytes", 0),
    ("response_bytes", True), ("response_bytes", 524289)])
def test_malformed_transport_record_refused(field, value):
    evidence = authored_owner_evidence()
    session = _session(evidence)
    session["requests"][0][field] = value
    _set_session(evidence, session)
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("location", ["session", "record", "child", "retained", "binding", "boundary"])
def test_unknown_fields_refused(location):
    evidence = authored_owner_evidence()
    session = _session(evidence)
    target = {"session": session, "record": session["requests"][0], "child": session["native_child"],
        "retained": session["retained_mcp_audit"], "binding": evidence["row"]["repository_source_policy"],
        "boundary": evidence["row"]["external_container_boundary"]}[location]
    target["foreign"] = True
    _set_session(evidence, session)
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("field,value", [("name", "../operations.jsonl"), ("sha256", "0" * 64),
    ("bytes", True), ("bytes", 0), ("content_free", False)])
def test_retained_operations_binding_refused(field, value):
    evidence = authored_owner_evidence()
    session = _session(evidence)
    session["retained_mcp_audit"][field] = value
    _set_session(evidence, session)
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("field,value", [("owner_uid", True), ("owner_uid", 999999),
    ("worker_uid", None), ("worker_uid", 999999), ("manifest_sha256", "0" * 64)])
def test_boundary_identity_mismatch_refused(field, value):
    evidence = authored_owner_evidence()
    evidence["row"]["external_container_boundary"][field] = value
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("mutation", ["session_descriptor", "descriptor_hash", "prepared_count",
    "store_cleanup", "edit_tool", "canonical_edit", "operations_overpopulation", "worker_config",
    "owner_config_as_worker", "binding_usage", "truncated_operations", "duplicate_json"])
def test_semantic_and_hash_edges_refused(mutation):
    evidence = authored_owner_evidence()
    session = _session(evidence)
    if mutation == "session_descriptor": session["session"]["workspace"] = "/other/candidate"
    elif mutation == "descriptor_hash": session["descriptor_sha256"] = "0" * 64
    elif mutation == "prepared_count": session["prepared_mcp_usage"]["successful_reads"] += 1
    elif mutation == "store_cleanup": session["prepared_mcp_usage"]["store_cleanup_complete"] = None
    elif mutation in {"edit_tool", "canonical_edit", "operations_overpopulation"}:
        rows = [json.loads(line) for line in evidence["operations_raw"].splitlines()]
        if mutation == "edit_tool": rows[1]["tool"] = "repository_replace_symbol"
        elif mutation == "canonical_edit": rows[1]["canonical_changed"] = True
        else: rows.append({**rows[1], "sequence": 3})
        _set_operations(evidence, rows)
        session = _session(evidence)
    elif mutation in {"worker_config", "owner_config_as_worker"}:
        evidence["row"]["usage"]["codex_repository_source_config_sha256"] = (
            "0" * 64 if mutation == "worker_config" else evidence["row"]["repository_source_policy"]["owner_config_sha256"])
    elif mutation == "binding_usage": evidence["row"]["repository_source_policy"]["usage"]["tool_calls"] = 0
    elif mutation == "truncated_operations":
        evidence["operations_raw"] = evidence["operations_raw"][:-1]
        session["retained_mcp_audit"].update(sha256=_sha(evidence["operations_raw"]), bytes=len(evidence["operations_raw"]))
    _set_session(evidence, session)
    if mutation == "duplicate_json": evidence["audit_raw"] = evidence["audit_raw"][:-1] + b',"schema":"repository-source-owner-session@1"}'
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)


@pytest.mark.parametrize("name,value", [("audit_raw", None), ("audit_raw", b""),
    ("audit_raw", b"x" * (audit.MAX_AUDIT_BYTES + 1)), ("operations_raw", None),
    ("operations_raw", b"x" * (audit.MAX_OPERATIONS_BYTES + 1)), ("row", None)])
def test_input_type_and_byte_limits(name, value):
    evidence = authored_owner_evidence()
    evidence[name] = value
    with pytest.raises(ValueError):
        audit.summarize_owner_session(**evidence)
