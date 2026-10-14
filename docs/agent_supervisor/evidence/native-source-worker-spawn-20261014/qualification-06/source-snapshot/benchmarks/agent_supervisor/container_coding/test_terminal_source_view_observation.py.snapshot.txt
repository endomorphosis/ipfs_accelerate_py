"""Closed source audit joins and exact model-input reconstruction, without models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_view_observation as collector
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as context_audit
from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_retrieval_observation import _fixture as numerical_fixture
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from test.api.test_semantic_router_translation import native  # noqa: F401


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fixture(session="a" * 32):
    audit, row = numerical_fixture(session)
    descriptor = audit["session"]
    descriptor.update(schema=protocol.DESCRIPTOR_SCHEMA,
        native_prompt_sha256=descriptor.pop("context_bundle_sha256"),
        semantic_sha256=descriptor.pop("context_receipt_sha256"),
        scope_cid=descriptor.pop("source_tree_id"), semantic_root_cid="semantic:source",
        source_manifest_sha256="f" * 64, profile="python-comments@1")
    descriptor.pop("task_cid")
    digest = _sha(protocol.canonical_json(descriptor))
    advice = protocol.guidance(descriptor["profile"]).encode()
    audit.update(schema="supervisor-worker-source-session@1", descriptor_sha256=digest)
    for item in audit["requests"]:
        item["returned_source_bytes"] = item.pop("returned_rows") * 20
    row.pop("numerical_obligation_retrieval"); row.pop("source_formalization_context")
    row["source_view_retrieval"] = dict(schema="worker-source-dispatch-observation@1",
        descriptor_sha256=digest, session_id=session,
        **{key: descriptor[key] for key in (*protocol.BINDING_FIELDS, "boundary_manifest_sha256")},
        guidance_sha256=_sha(advice), guidance_bytes=len(advice), requests_observed=False,
        proof_authority=False, execution_authority=False, completion_authority=False)
    row["semantic_translation"] = {key: descriptor[key] for key in (
        "task_id", "scope_cid", "semantic_root_cid", "semantic_sha256", "native_prompt_sha256")}
    return audit, row


@pytest.fixture
def evidence():
    with tempfile.TemporaryDirectory(prefix="source-collector-", dir=Path.home()) as name:
        parent = Path(name); parent.chmod(0o750)
        audit, row = _fixture()
        scope = dict(task_alias="sample", task_cid="native:task")
        report = dict(worker_cleanup_returncode=0, provider_invocations=[row], source_view_profile="python-comments@1")
        def write(value=audit):
            path = parent / ("source-view-" + value["session"]["session_id"] + "-audit.json")
            if path.exists(): path.chmod(0o600)
            path.write_bytes(protocol.canonical_json(value)); path.chmod(0o400)
            return path
        path = write()
        yield parent, scope, report, audit, row, write, path


def _collect(evidence):
    return collector.collect(parent=evidence[0], scope=evidence[1], report=evidence[2])


def _typed_audit(audit, row, *, shared=False):
    """Author the new optional record fields without inventing source reads."""
    audit["schema"] = "supervisor-worker-source-session@3" if shared else "supervisor-worker-source-session@2"
    for item in audit["requests"]:
        item.update(returned_candidate_context_bytes=0, candidate_context_page=False,
                    discovery_page=False, returned_discovery_rows=0)
    if shared:
        descriptor = audit["session"]
        descriptor.update(schema=protocol.DESCRIPTOR_SCHEMA_V2,
                          view_policy_binding_sha256="5" * 64, source_scope_sha256="6" * 64)
        digest = _sha(protocol.canonical_json(descriptor))
        audit["descriptor_sha256"] = digest
        row["source_view_retrieval"].update(schema="worker-source-dispatch-observation@2",
            descriptor_sha256=digest, **{key: descriptor[key] for key in collector.SHARED_PIN_FIELDS})


def _mixed_pages(evidence, *, shared=False):
    audit, row = evidence[3:5]
    _typed_audit(audit, row, shared=shared)
    source = audit["requests"][0]
    candidate = dict(source, ordinal=2, request_sha256="3" * 64, response_sha256="4" * 64,
        response_bytes=3072, returned_source_bytes=0, returned_candidate_context_bytes=60,
        candidate_context_page=True, delivered=False)
    discovery = dict(source, ordinal=3, request_sha256="5" * 64, response_sha256="6" * 64,
        response_bytes=4096, returned_source_bytes=0, discovery_page=True, returned_discovery_rows=2)
    undelivered_discovery = dict(discovery, ordinal=4, request_sha256="7" * 64,
        response_sha256="8" * 64, returned_discovery_rows=3, delivered=False)
    refusal = dict(audit["requests"][1], ordinal=5, request_sha256="9" * 64, response_sha256="a" * 64)
    audit.update(requests=[source, candidate, discovery, undelivered_discovery, refusal],
                 accepted_requests=5, response_bytes=13545)
    evidence[5]()


@pytest.mark.parametrize("shared", [False, True], ids=["legacy-descriptor", "shared-policy"])
def test_mixed_source_candidate_and_discovery_pages_keep_all_wire_bytes(evidence, shared):
    _mixed_pages(evidence, shared=shared)
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"] == dict(sessions=1, accepted_requests=5, delivered_responses=2,
        prepared_pages=4, delivered_pages=2, prepared_source_bytes=40, delivered_source_bytes=40,
        response_bytes=13545, delivered_response_bytes=6144, request_hash_count=5,
        response_hash_count=5, unique_request_hash_count=5, unique_response_hash_count=5)
    assert result["candidate_context_counts"] == dict(prepared_pages=1, delivered_pages=0,
        prepared_bytes=60, delivered_bytes=0)
    assert result["discovery_counts"] == dict(prepared_pages=2, delivered_pages=1,
        prepared_rows=5, delivered_rows=2)
    assert result["sessions"][0]["discovery_counts"] == result["discovery_counts"]
    assert result["provider_tokens"] is None and result["total_token_savings_measured"] is False
    assert not any(result[key] for key in ("proof_authority", "execution_authority", "completion_authority"))
    encoded = protocol.canonical_json(result)
    assert b"/allocated/" not in encoded and b"/owner/" not in encoded and b'"requests"' not in encoded


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_empty_historical_audits_do_not_invent_discovery_measurements(evidence, schema):
    if schema != 1:
        _typed_audit(evidence[3], evidence[4], shared=schema == 3)
    evidence[3].update(requests=[], accepted_requests=0, response_bytes=0)
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == "observed" and result["counts"]["response_bytes"] == 0
    assert result["discovery_counts"] == (dict.fromkeys(collector.DISCOVERY_COUNT_FIELDS, 0)
                                          if schema == 3 else None)


@pytest.mark.parametrize("missing", ["all", "one-record"])
def test_legacy_candidate_audit_discovery_is_unknown_when_records_are_untyped(evidence, missing):
    _typed_audit(evidence[3], evidence[4])
    records = evidence[3]["requests"] if missing == "all" else evidence[3]["requests"][-1:]
    for item in records:
        item.pop("discovery_page"); item.pop("returned_discovery_rows")
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == "observed" and result["discovery_counts"] is None
    assert result["counts"]["response_bytes"] == 2281
    assert result["candidate_context_counts"] == dict(prepared_pages=0, delivered_pages=0,
                                                      prepared_bytes=0, delivered_bytes=0)


def test_multi_session_discovery_is_unknown_if_one_complete_session_is_historical(evidence):
    _mixed_pages(evidence, shared=True)
    historical, row = _fixture("e" * 32)
    evidence[2]["provider_invocations"].append(row)
    evidence[5](historical)
    result = _collect(evidence)
    assert result["status"] == "observed" and result["discovery_counts"] is None
    assert result["sessions"][0]["discovery_counts"]["prepared_rows"] == 5
    assert result["sessions"][1]["discovery_counts"] is None
    assert result["counts"]["response_bytes"] == 15826


def test_typed_sessions_sum_discovery_deliveries_without_deduplicating_rows(evidence):
    _mixed_pages(evidence, shared=True)
    extra, row = _fixture("e" * 32)
    _typed_audit(extra, row, shared=True)
    extra["requests"][0].update(returned_source_bytes=0, discovery_page=True, returned_discovery_rows=64)
    evidence[2]["provider_invocations"].append(row)
    evidence[5](extra)
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["discovery_counts"] == dict(prepared_pages=3, delivered_pages=2,
                                              prepared_rows=69, delivered_rows=66)
    assert result["counts"]["response_bytes"] == 15826
    assert result["counts"]["delivered_source_bytes"] == 40


@pytest.mark.parametrize("change", ["boolean_rows", "negative_rows", "excess_rows", "numeric_flag",
    "false_flag_with_rows", "empty_discovery", "source_overlap", "candidate_overlap", "refusal",
    "missing_flag", "missing_rows", "missing_pair", "unknown_field", "false_total", "boolean_delivery"])
def test_falsified_shared_discovery_records_are_unavailable(evidence, change):
    _mixed_pages(evidence, shared=True)
    record = evidence[3]["requests"][2]
    if change == "boolean_rows": record["returned_discovery_rows"] = True
    elif change == "negative_rows": record["returned_discovery_rows"] = -1
    elif change == "excess_rows": record["returned_discovery_rows"] = 65
    elif change == "numeric_flag": record["discovery_page"] = 1
    elif change == "false_flag_with_rows": record["discovery_page"] = False
    elif change == "empty_discovery": record["returned_discovery_rows"] = 0
    elif change == "source_overlap": record["returned_source_bytes"] = 1
    elif change == "candidate_overlap": record.update(candidate_context_page=True, returned_candidate_context_bytes=1)
    elif change == "refusal": record["status"] = "retrieval_refused"
    elif change == "missing_flag": record.pop("discovery_page")
    elif change == "missing_rows": record.pop("returned_discovery_rows")
    elif change == "missing_pair":
        record.pop("discovery_page"); record.pop("returned_discovery_rows")
    elif change == "unknown_field": record["discovery_source_text"] = "private source"
    elif change == "false_total": evidence[3]["response_bytes"] -= 1
    else: record["delivered"] = 1
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None and result["discovery_counts"] is None


@pytest.mark.parametrize("change", ["audit_version", "observation_version", "policy_pin", "scope_pin",
    "missing_policy_pin", "missing_scope_pin", "malformed_descriptor_pin", "extra_pin", "guidance"])
def test_shared_descriptor_dispatch_and_audit_modes_must_join_exactly(evidence, change):
    _mixed_pages(evidence, shared=True)
    audit, row = evidence[3:5]
    observation = row["source_view_retrieval"]
    if change == "audit_version": audit["schema"] = "supervisor-worker-source-session@2"
    elif change == "observation_version": observation["schema"] = "worker-source-dispatch-observation@1"
    elif change == "policy_pin": observation["view_policy_binding_sha256"] = "7" * 64
    elif change == "scope_pin": observation["source_scope_sha256"] = "8" * 64
    elif change == "missing_policy_pin": observation.pop("view_policy_binding_sha256")
    elif change == "missing_scope_pin": observation.pop("source_scope_sha256")
    elif change == "malformed_descriptor_pin":
        audit["session"]["source_scope_sha256"] = "invalid"
        digest = _sha(protocol.canonical_json(audit["session"]))
        audit["descriptor_sha256"] = observation["descriptor_sha256"] = digest
        observation["source_scope_sha256"] = "invalid"
    elif change == "extra_pin": observation["unclassified_pin"] = "9" * 64
    else: observation["guidance_sha256"] = "0" * 64
    evidence[5]()
    assert _collect(evidence)["status"] == "unavailable"


@pytest.mark.parametrize("change", ["shared_audit", "shared_observation", "new_record_on_old_audit",
                                     "half_pair", "false_discovery_rows"])
def test_legacy_modes_cannot_claim_shared_or_untyped_discovery_authority(evidence, change):
    if change not in {"shared_audit", "shared_observation", "new_record_on_old_audit"}:
        _typed_audit(evidence[3], evidence[4])
    if change == "shared_audit": evidence[3]["schema"] = "supervisor-worker-source-session@3"
    elif change == "shared_observation": evidence[4]["source_view_retrieval"]["schema"] = "worker-source-dispatch-observation@2"
    elif change == "new_record_on_old_audit": evidence[3]["requests"][0].update(discovery_page=False, returned_discovery_rows=0)
    elif change == "half_pair": evidence[3]["requests"][0].pop("returned_discovery_rows")
    else: evidence[3]["requests"][0]["returned_discovery_rows"] = 1
    evidence[5]()
    assert _collect(evidence)["status"] == "unavailable"


def test_source_counts_are_exact_and_contain_no_raw_source_or_token_estimate(evidence):
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"] == dict(sessions=1, accepted_requests=3, delivered_responses=1,
        prepared_pages=1, delivered_pages=1, prepared_source_bytes=40, delivered_source_bytes=40,
        response_bytes=2281, delivered_response_bytes=2048, request_hash_count=2,
        response_hash_count=2, unique_request_hash_count=2, unique_response_hash_count=2)
    assert result["provider_tokens"] is None and result["total_token_savings_measured"] is False
    encoded = protocol.canonical_json(result)
    assert b"/allocated/" not in encoded and b"/owner/" not in encoded
    assert b"returned_rows" not in encoded


def test_zero_requires_a_complete_matching_closed_session(evidence):
    evidence[3].update(requests=[], accepted_requests=0, response_bytes=0)
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == "observed" and result["counts"]["accepted_requests"] == 0
    evidence[6].unlink()
    assert _collect(evidence)["status"] == "unavailable"


@pytest.mark.parametrize("profile,status", [("disabled", "not_selected"), (None, "unavailable"),
    (False, "unavailable"), ("unknown", "unavailable"), ("identity@1", "unavailable"),
    ("python-whitespace@1", "unavailable"), ("python-comments@1", "unavailable")])
def test_missing_selected_router_session_cannot_report_zero(profile, status, monkeypatch):
    monkeypatch.setattr(collector, "_open_directory", lambda *_: pytest.fail("missing session reached private IO"))
    result = collector.collect(parent=Path("/absent"), scope=None,
        report=dict(provider_invocations=[], source_view_profile=profile))
    assert result["status"] == status and result["counts"] is None


def _provider_free_report(route):
    requested = dict(semantic_transport_schema="supervisor-semantic-router-input@1",
        semantic_metadata_view="common-bindings-instruction@1", coding_reply_mode="legacy",
        source_view_profile="python-comments@1", source_edit_mode="mapped-span@2", source_view_policy=None)
    return dict(arm="full", provider_invocations=[], implementation_route=route,
        worker_cleanup_returncode=0, source_view_profile=requested["source_view_profile"],
        source_edit_mode=requested["source_edit_mode"],
        semantic_metadata_view={"requested_mode": requested["semantic_metadata_view"]},
        coding_control_application=dict(schema="terminal-coding-control-application@1",
            status="not_applicable_provider_free", implementation_route=route, requested=requested,
            model_coding_dispatched=False, token_savings_established=False, completion_authority=False),
        doctor_dispatch=dict(route=route, status="candidate_ready", provider_calls=0,
            task_cid="authored-task", candidate_commit="a" * 40),
        doctor_invocations=[dict(schema=("native-doctor-candidate-materialization@1"
            if route == "doctor_candidate" else "native-doctor-contract-candidate-materialization@1"),
            status="candidate_materialized", task_cid="authored-task", candidate_commit="a" * 40,
            provider_calls=0, publication_authority=False, completion_authority=False)])


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
def test_observed_doctor_makes_model_views_inapplicable_without_zero_counts(route, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
    report = _provider_free_report(route)
    monkeypatch.setattr(collector, "_open_directory", lambda *_: pytest.fail("inapplicable view reached private IO"))
    monkeypatch.setattr(audit, "collect_terminal_context_audit", lambda **_: pytest.fail("inapplicable input reached replay"))
    observed = collector.collect(parent=Path("/absent"), scope=None, report=report)
    assert observed["status"] == "not_applicable"
    assert observed["reason"] == "provider_free_doctor_materialization_observed"
    assert observed["counts"] is observed["provider_tokens"] is None
    assert observed["completion_authority"] is observed["total_token_savings_measured"] is False
    driver._final_context_audit(report, state=Path("/absent"), deadline=0)
    assert report["context_input_audit"]["status"] == "not_applicable"
    assert report["context_input_audit"]["any_model_input_verified"] is None


@pytest.mark.parametrize("change", ["no_application", "route_only", "no_receipt", "foreign_task",
    "foreign_commit", "wrong_receipt", "provider_call", "boolean_calls", "coding_receipt",
    "missing_provider_attempt", "false_like_attempt", "failed_cleanup", "requested_drift", "authority"])
def test_doctor_label_cannot_hide_missing_or_conflicting_model_evidence(change):
    report = _provider_free_report("doctor_candidate")
    if change == "no_application": report.pop("coding_control_application")
    elif change == "route_only": report.pop("doctor_dispatch")
    elif change == "no_receipt": report.pop("doctor_invocations")
    elif change == "foreign_task": report["doctor_invocations"][0]["task_cid"] = "foreign-task"
    elif change == "foreign_commit": report["doctor_invocations"][0]["candidate_commit"] = "b" * 40
    elif change == "wrong_receipt": report["doctor_invocations"][0]["schema"] = "router-implementation-invocation@1"
    elif change == "provider_call": report["doctor_invocations"][0]["provider_calls"] = 1
    elif change == "boolean_calls": report["doctor_invocations"][0]["provider_calls"] = False
    elif change == "coding_receipt": report["provider_invocations"] = [{"phase": "coding"}]
    elif change == "missing_provider_attempt": report["unreceipted_provider_attempt"] = {"phase": "coding"}
    elif change == "false_like_attempt": report["unreceipted_provider_attempt"] = 0
    elif change == "failed_cleanup": report["worker_cleanup_returncode"] = 1
    elif change == "requested_drift": report["source_view_profile"] = "identity@1"
    elif change == "authority": report["coding_control_application"]["completion_authority"] = True
    assert collector.provider_free_coding_application_observed(report) is False
    observed = collector.collect(parent=Path("/absent"), scope=None, report=report)
    assert observed["status"] == "unavailable" and observed["counts"] is None


@pytest.mark.parametrize("change", ["alias", "semantic", "scope", "prompt", "manifest", "profile", "selected_profile",
    "disabled", "workspace", "boundary", "owner", "guidance", "guidance_bytes", "schema", "extra",
    "phase", "duplicate", "descriptor", "cleanup", "inflight", "stopped", "incomplete", "sum",
    "bool_count", "numeric_detail", "bool_detail", "excess_detail", "impossible_detail", "refusal_detail", "raw_content"])
def test_falsified_foreign_or_incomplete_selected_evidence_is_unavailable(evidence, change):
    parent, scope, report, audit, row, write, path = evidence
    observation = row["source_view_retrieval"]
    if change == "alias": scope["task_alias"] = "foreign"
    elif change == "semantic": row["semantic_translation"]["semantic_root_cid"] = "foreign:root"
    elif change == "scope": observation["scope_cid"] = "foreign:scope"
    elif change == "prompt": observation["native_prompt_sha256"] = "0" * 64
    elif change == "manifest": observation["source_manifest_sha256"] = "0" * 64
    elif change == "profile": observation["profile"] = "identity@1"
    elif change == "selected_profile": report["source_view_profile"] = "python-whitespace@1"
    elif change == "disabled": report["source_view_profile"] = "disabled"
    elif change == "workspace": row["workspace"] = "/foreign"
    elif change == "boundary": row["external_container_boundary"]["manifest_sha256"] = "0" * 64
    elif change == "owner": row["external_container_boundary"]["owner_uid"] += 1
    elif change == "guidance": observation["guidance_sha256"] = "0" * 64
    elif change == "guidance_bytes": observation["guidance_bytes"] = True
    elif change == "schema": observation["schema"] = "foreign"
    elif change == "extra": observation["private"] = True
    elif change == "phase": row["phase"] = "planning"
    elif change == "duplicate": report["provider_invocations"].append(deepcopy(row))
    elif change == "descriptor": audit["descriptor_sha256"] = "0" * 64
    elif change == "cleanup": report["worker_cleanup_returncode"] = False
    elif change == "inflight": audit["inflight_request"] = 1
    elif change == "stopped": audit["service_stopped"] = False
    elif change == "incomplete": audit["accounting_complete"] = False
    elif change == "sum": audit["response_bytes"] += 1
    elif change == "bool_count": audit["accepted_requests"] = True
    elif change == "numeric_detail": audit["requests"][0]["returned_rows"] = 2
    elif change == "bool_detail": audit["requests"][0]["returned_source_bytes"] = False
    elif change == "excess_detail": audit["requests"][0]["returned_source_bytes"] = 8193
    elif change == "impossible_detail": audit["requests"][0]["returned_source_bytes"] = 2049
    elif change == "refusal_detail": audit["requests"][1]["returned_source_bytes"] = 1
    else: audit["requests"][0]["content"] = "private source"
    write()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None and result["sessions"] == []


def test_multiple_session_missing_one_refuses_partial_aggregation(evidence):
    extra, row = _fixture("e" * 32)
    evidence[2]["provider_invocations"].append(row)
    path = evidence[5](extra)
    assert _collect(evidence)["counts"]["delivered_source_bytes"] == 80
    path.unlink()
    assert _collect(evidence)["status"] == "unavailable"


@pytest.mark.parametrize("change", ["symlink", "writable", "group_readable", "hardlink", "parent", "replacement"])
def test_source_audit_files_remain_private_stable_single_links(evidence, monkeypatch, change):
    parent, _, _, _, _, _, path = evidence
    if change == "symlink":
        other = path.with_name("other"); path.rename(other); path.symlink_to(other)
    elif change == "writable": path.chmod(0o600)
    elif change == "group_readable": path.chmod(0o440)
    elif change == "hardlink":
        import os
        os.link(path, path.with_name("other"))
    elif change == "parent": parent.chmod(0o770)
    else:
        read = collector._read
        def replaced(fd, name, **kwargs):
            raw = read(fd, name, **kwargs); path.unlink()
            path.write_bytes(raw); path.chmod(0o400)
            return raw
        monkeypatch.setattr(collector, "_read", replaced)
    assert _collect(evidence)["status"] == "unavailable"


def test_source_guidance_matches_actual_historical_model_projection(native):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    profile = "python-comments@1"
    advice = protocol.guidance(profile)
    observation = dict(protocol.semantic_binding(encoded, profile),
        guidance_sha256=_sha(advice.encode()), guidance_bytes=len(advice.encode()))
    workspace = root.parent / "removed-worktree"
    router = encoded.provider_prompt + advice
    model, advisory = render_model_prompt(prompt=router, purpose="coding", workspace=workspace, semantic_transport=True)
    receipt = dict(semantic_translation=encoded.receipt, workspace=str(workspace), purpose="coding",
        source_view_retrieval=observation, router_prompt_sha256=_sha(router.encode()), router_prompt_bytes=len(router.encode()),
        workspace_advisory_sha256=_sha(advisory.encode()), workspace_advisory_bytes=len(advisory.encode()),
        model_prompt_sha256=_sha(model.encode()), model_prompt_bytes=len(model.encode()))
    actual, _, checks = context_audit._model_projection(rendered=prompt, receipt=receipt, repository=root, formalization=None)
    assert actual == model and all(checks.values())
    receipt["source_view_retrieval"]["source_manifest_sha256"] = "0" * 64
    _, _, checks = context_audit._model_projection(rendered=prompt, receipt=receipt, repository=root, formalization=None)
    assert checks["source_view_semantic_binding"] is False
