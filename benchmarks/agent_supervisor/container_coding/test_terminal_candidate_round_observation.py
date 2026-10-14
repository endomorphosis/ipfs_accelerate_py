"""Closed candidate-reader accounting, including failures and unknown coverage."""
from pathlib import Path
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_candidate_round_observation as collector
from benchmarks.agent_supervisor.container_coding import terminal_source_view_observation as baseline
from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import _fixture as source_fixture
from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import _typed_audit
from benchmarks.agent_supervisor.container_coding.test_terminal_source_request_observation import _upgrade
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_context as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_request_observation as attribution
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import _candidate_round_guidance


def _fixture(session="a" * 32, prior_session="f" * 32):
    audit, row = source_fixture(session)
    descriptor = audit["session"]
    descriptor.update(schema=protocol.DESCRIPTOR_SCHEMA, prior_session_id=prior_session,
        prior_descriptor_sha256="9" * 64, workspace_generation="0" * 64, round_context_sha256="1" * 64,
        round_number=2, candidate_scope_cid="candidate:scope", candidate_semantic_root_cid="candidate:root",
        source_scope_sha256="2" * 64, view_policy_binding_sha256=None)
    digest = collector._sha(protocol.canonical_json(descriptor))
    audit.update(schema=collector.AUDIT_SCHEMA, descriptor_sha256=digest)
    probe = dict(audit["requests"][0], returned_source_bytes=0,
        returned_candidate_context_bytes=60, candidate_context_page=True,
        discovery_page=False, returned_discovery_rows=0,
        request_operation="candidate_context", request_source_state="working-candidate", request_view="not_applicable")
    source = dict(probe, ordinal=2, request_sha256="3" * 64, response_sha256="4" * 64,
        returned_candidate_context_bytes=0, candidate_context_page=False, returned_source_bytes=40,
        request_operation="read", request_view="compact")
    undelivered = dict(probe, ordinal=3, request_sha256="5" * 64, response_sha256="6" * 64,
        response_bytes=3072, delivered=False)
    refusal = dict(source, ordinal=4, request_sha256="7" * 64, response_sha256="8" * 64,
        status="stale_context", response_bytes=233, returned_source_bytes=0, delivered=False)
    invalid = dict(refusal, ordinal=5, request_sha256=None, response_sha256=None,
        status="invalid_request", response_bytes=0, **dict(zip(attribution.FIELDS, attribution.UNKNOWN)))
    audit.update(requests=[probe, source, undelivered, refusal, invalid], accepted_requests=5, response_bytes=7401)
    observation = dict(schema=dispatch.OBSERVATION_SCHEMA, **{key: descriptor[key] for key in collector.JOIN_FIELDS},
        descriptor_sha256=digest, handoff_sha256="a" * 64, provider_input_sha256="e" * 64,
        provider_input_bytes=100, native_prompt_bytes=70,
        candidate_artifacts={name: dict(sha256=str(index) * 64, bytes=500)
            for index, name in enumerate(("summary", "capsules", "formal"), 3)},
        producer_binding_sha256="6" * 64, source_files=2,
        expires_monotonic_ns=descriptor["expires_monotonic_ns"],
        dispatch_deadline_monotonic_ns=descriptor["expires_monotonic_ns"], remaining_timeout_seconds=1,
        validation_phase="before_provider", owner_validation_inherited=True, producer_validation_inherited=True,
        neural_inference_replayed=False, owner_private_state_replayed_here=False,
        canonical_source_current=True, workspace_source_current=True, **dispatch.FLAGS)
    guidance = _candidate_round_guidance(observation).encode()
    row.pop("source_view_retrieval")
    row.update(status="provider_returned", candidate_round_dispatch=observation, semantic_translation=None,
        native_prompt_sha256=descriptor["native_prompt_sha256"], native_prompt_bytes=70,
        prompt_sha256=observation["provider_input_sha256"], prompt_bytes=100,
        candidate_retrieval_guidance=dict(sha256=collector._sha(guidance), bytes=len(guidance),
            requests_observed=False, proof_authority=False, completion_authority=False),
        candidate_round_preflight=dict(schema="supervisor-candidate-round-preflight@1", requests=1,
            session_id=session, workspace_generation=descriptor["workspace_generation"], artifact="summary",
            artifact_sha256=observation["candidate_artifacts"]["summary"]["sha256"],
            response_sha256=probe["response_sha256"], response_bytes=probe["response_bytes"],
            provider_input_bytes=0, child_custody_checked=True, source_freshness_checked=True,
            producer_freshness_checked=True, proof_authority=False, completion_authority=False))
    return audit, row


@pytest.fixture
def evidence():
    with tempfile.TemporaryDirectory(prefix="candidate-collector-", dir=Path.home()) as name:
        parent = Path(name); parent.chmod(0o750)
        audit, row = _fixture()
        scope = dict(task_alias="sample")
        report = dict(worker_cleanup_returncode=0, provider_invocations=[row], source_view_profile="python-comments@1")
        def write(value=audit, prefix="round-view"):
            path = parent / (prefix + "-" + value["session"]["session_id"] + "-audit.json")
            if path.exists(): path.chmod(0o600)
            path.write_bytes(protocol.canonical_json(value)); path.chmod(0o400)
            return path
        path = write()
        yield parent, scope, report, audit, row, write, path


def _collect(evidence):
    return collector.collect(parent=evidence[0], scope=evidence[1], report=evidence[2])


def test_candidate_source_artifact_and_preflight_bytes_are_distinct(evidence):
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"] == dict(sessions=1, accepted_requests=5, delivered_responses=2,
        prepared_pages=3, delivered_pages=2, prepared_source_bytes=40, delivered_source_bytes=40,
        response_bytes=7401, delivered_response_bytes=4096, request_hash_count=4, response_hash_count=4,
        unique_request_hash_count=4, unique_response_hash_count=4)
    assert result["candidate_context_counts"] == dict(prepared_pages=2, delivered_pages=1,
        prepared_bytes=120, delivered_bytes=60)
    assert result["preflight_counts"] == dict(requests=1, response_bytes=2048,
        candidate_context_bytes=60, provider_input_bytes=0)
    assert result["sessions"][0]["status_counts"] == {"ok": 3, "stale_context": 1, "invalid_request": 1}
    assert result["request_operation_counts"]["unparsed_requests"] == 1
    assert result["sessions"][0]["prior_router_receipt_matched"] is False
    assert result["provider_tokens"] is None and result["total_token_savings_measured"] is False
    assert result["model_input_replayed"] is result["proof_authority"] is result["source_semantics_verified"] is False
    raw = protocol.canonical_json(result)
    assert b"/allocated/" not in raw and b"/owner/" not in raw and b'"requests":[' not in raw


def test_provider_failure_after_preflight_preserves_known_session_cost(evidence):
    evidence[4].update(status="failed", failure_phase="provider_invocation")
    result = _collect(evidence)
    assert result["status"] == "observed" and result["counts"]["response_bytes"] == 7401


@pytest.mark.parametrize("change", ["missing", "empty", "inflight", "running", "incomplete", "wrong_sum",
    "extra_field", "source_artifact_overlap", "candidate_false", "candidate_zero", "candidate_boolean",
    "bad_candidate_bytes", "source_overflow", "baseline_read", "discovery", "edit", "untyped",
    "boolean_delivery", "duplicate_ordinal", "unknown_status", "response_hash", "invalid_unknown"])
def test_missing_or_inconsistent_candidate_audit_is_unknown(evidence, change):
    audit = evidence[3]
    record = audit["requests"][0]
    if change == "missing": evidence[6].unlink()
    elif change == "empty": audit.update(requests=[], accepted_requests=0, response_bytes=0)
    elif change == "inflight": audit["inflight_request"] = 6
    elif change == "running": audit["service_stopped"] = False
    elif change == "incomplete": audit["accounting_complete"] = False
    elif change == "wrong_sum": audit["response_bytes"] -= 1
    elif change == "extra_field": record["source_text"] = "do not export"
    elif change == "source_artifact_overlap": record["returned_source_bytes"] = 1
    elif change == "candidate_false": record["candidate_context_page"] = False
    elif change == "candidate_zero": record["returned_candidate_context_bytes"] = 0
    elif change == "candidate_boolean": record["returned_candidate_context_bytes"] = True
    elif change == "bad_candidate_bytes": record["returned_candidate_context_bytes"] = record["response_bytes"] + 1
    elif change == "source_overflow": audit["requests"][1]["returned_source_bytes"] = 2049
    elif change == "baseline_read": audit["requests"][1]["request_source_state"] = "baseline"
    elif change == "discovery": record.update(discovery_page=True, returned_discovery_rows=1)
    elif change == "edit": record["source_edit_page"] = False
    elif change == "untyped": record.pop("request_source_state")
    elif change == "boolean_delivery": record["delivered"] = 1
    elif change == "duplicate_ordinal": record["ordinal"] = 2
    elif change == "unknown_status": record["status"] = "private exception"
    elif change == "response_hash": record["response_sha256"] = None
    else: audit["requests"][-1]["request_sha256"] = "a" * 64
    if change != "missing": evidence[5]()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None
    assert result["candidate_context_counts"] is result["preflight_counts"] is None


@pytest.mark.parametrize("change", ["no_probe", "wrong_session", "wrong_generation", "wrong_artifact", "wrong_hash",
    "wrong_bytes", "boolean_bytes", "boolean_requests", "no_custody", "proof", "provider_bytes", "undelivered",
    "duplicate_response", "source_response"])
def test_preflight_must_join_exactly_one_delivered_candidate_response(evidence, change):
    probe = evidence[4]["candidate_round_preflight"]
    if change == "no_probe": evidence[4].pop("candidate_round_preflight")
    elif change == "wrong_session": probe["session_id"] = "e" * 32
    elif change == "wrong_generation": probe["workspace_generation"] = "e" * 64
    elif change == "wrong_artifact": probe["artifact"] = "formal"
    elif change == "wrong_hash": probe["response_sha256"] = "e" * 64
    elif change == "wrong_bytes": probe["response_bytes"] += 1
    elif change == "boolean_bytes": probe["response_bytes"] = True
    elif change == "boolean_requests": probe["requests"] = True
    elif change == "no_custody": probe["child_custody_checked"] = False
    elif change == "proof": probe["proof_authority"] = True
    elif change == "provider_bytes": probe["provider_input_bytes"] = 2048
    elif change == "undelivered": evidence[3]["requests"][0]["delivered"] = False
    elif change == "duplicate_response": evidence[3]["requests"][2]["response_sha256"] = probe["response_sha256"]
    else: probe["response_sha256"] = evidence[3]["requests"][1]["response_sha256"]
    evidence[5]()
    assert _collect(evidence)["status"] == "unavailable"


@pytest.mark.parametrize("change", ["schema", "unknown", "task", "native_prompt", "provider_prompt", "descriptor",
    "prior", "generation", "round_boolean", "flag_boolean", "expiry", "deadline", "remaining", "source_files",
    "artifact_pin", "artifact_bool", "guidance", "guidance_type", "source_mixed", "baseline_semantics", "boundary",
    "workspace", "cleanup", "scope", "receipt_phase", "duplicate_session", "duplicate_invocation", "unreceipted"])
def test_candidate_receipt_descriptor_and_task_bindings_are_closed(evidence, change):
    row, audit, report = evidence[4], evidence[3], evidence[2]
    observation = row["candidate_round_dispatch"]
    if change == "schema": observation["schema"] = "unknown"
    elif change == "unknown": observation["source_text"] = "private"
    elif change == "task": observation["task_id"] = "other"
    elif change == "native_prompt": row["native_prompt_sha256"] = "e" * 64
    elif change == "provider_prompt": row["prompt_sha256"] = row["native_prompt_sha256"]
    elif change == "descriptor": observation["descriptor_sha256"] = "e" * 64
    elif change == "prior": observation["prior_session_id"] = observation["session_id"]
    elif change == "generation": observation["workspace_generation"] = "e" * 64
    elif change == "round_boolean": observation["round_number"] = True
    elif change == "flag_boolean": observation["source_obligations_discharged"] = False
    elif change == "expiry": observation["expires_monotonic_ns"] = 0
    elif change == "deadline": observation["dispatch_deadline_monotonic_ns"] = 1
    elif change == "remaining": observation["remaining_timeout_seconds"] = -1
    elif change == "source_files": observation["source_files"] = True
    elif change == "artifact_pin": observation["candidate_artifacts"]["summary"]["sha256"] = "e" * 64
    elif change == "artifact_bool": observation["candidate_artifacts"]["formal"]["bytes"] = True
    elif change == "guidance": row["candidate_retrieval_guidance"]["sha256"] = "e" * 64
    elif change == "guidance_type": row["candidate_retrieval_guidance"]["requests_observed"] = 0
    elif change == "source_mixed": row["source_view_retrieval"] = {}
    elif change == "baseline_semantics": row["semantic_translation"] = {}
    elif change == "boundary": row["external_container_boundary"]["manifest_sha256"] = "e" * 64
    elif change == "workspace": row["workspace"] = "/different"
    elif change == "cleanup": report["worker_cleanup_returncode"] = None
    elif change == "scope": evidence[1]["task_alias"] = "other"
    elif change == "receipt_phase": row["phase"] = "planning"
    elif change == "duplicate_session": report["provider_invocations"].append(dict(row, invocation_id="different"))
    elif change == "duplicate_invocation":
        extra, receipt = _fixture("e" * 32)
        receipt["invocation_id"] = row["invocation_id"]
        report["provider_invocations"].append(receipt); evidence[5](extra)
    else: report["unreceipted_provider_attempt"] = {"usage": None}
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None


@pytest.mark.parametrize("change", ["writable", "public", "symlink", "hardlink", "duplicate_json", "noncanonical", "oversized"])
def test_candidate_audit_requires_an_unchanged_bounded_owner_private_file(evidence, change):
    path = evidence[6]
    if change == "writable": path.chmod(0o600)
    elif change == "public": path.chmod(0o444)
    elif change == "symlink":
        real = path.with_suffix(".body"); path.rename(real); path.symlink_to(real)
    elif change == "hardlink": path.with_suffix(".copy").hardlink_to(path)
    else:
        path.chmod(0o600)
        raw = path.read_bytes()
        if change == "duplicate_json": raw = b'{"schema":"bad",' + raw[1:]
        elif change == "noncanonical": raw += b"\n"
        else: raw += b" " * collector.MAX_AUDIT_BYTES
        path.write_bytes(raw); path.chmod(0o400)
    assert _collect(evidence)["status"] == "unavailable"


def test_multiple_candidate_sessions_are_all_counted_or_all_unknown(evidence):
    extra, row = _fixture("e" * 32)
    evidence[2]["provider_invocations"].append(row)
    path = evidence[5](extra)
    result = _collect(evidence)
    assert result["status"] == "observed" and result["counts"]["sessions"] == 2
    assert result["counts"]["response_bytes"] == 14802 and result["preflight_counts"]["requests"] == 2
    assert result["counts"]["response_hash_count"] == 8 and result["counts"]["unique_response_hash_count"] == 4
    path.unlink()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["sessions"] == [] and result["counts"] is None


@pytest.mark.parametrize("marker", sorted(collector.MARKERS))
def test_partial_candidate_markers_are_unavailable_not_unselected(marker, monkeypatch):
    monkeypatch.setattr(collector, "_open_directory", lambda *args: pytest.fail("partial receipt opened owner directory"))
    result = collector.collect(parent=Path("/missing"), scope=None, report={"provider_invocations": [{marker: {}}]})
    assert result["status"] == "unavailable" and result["counts"] is None


@pytest.mark.parametrize("unreceipted", [False, True])
def test_no_receipts_preserve_unknown_second_attempt_without_private_io(unreceipted, monkeypatch):
    monkeypatch.setattr(collector, "_open_directory", lambda *args: pytest.fail("no receipt opened owner directory"))
    result = collector.collect(parent=Path("/missing"), scope=None, report=dict(provider_invocations=[],
        candidate_rounds=2, unreceipted_provider_attempt={"phase": "coding", "usage": None} if unreceipted else False))
    assert result["status"] == ("unavailable" if unreceipted else "not_selected")
    assert result["counts"] is None


@pytest.mark.parametrize("tamper", [False, True])
def test_prior_router_join_is_explicit_and_mismatches_refuse(evidence, tamper):
    audit, row = source_fixture("f" * 32)
    observation = evidence[4]["candidate_round_dispatch"]
    observation["prior_descriptor_sha256"] = audit["descriptor_sha256"]
    evidence[3]["session"]["prior_descriptor_sha256"] = audit["descriptor_sha256"]
    digest = collector._sha(protocol.canonical_json(evidence[3]["session"]))
    observation["descriptor_sha256"] = evidence[3]["descriptor_sha256"] = digest
    evidence[2]["provider_invocations"].insert(0, row)
    if tamper: row["source_view_retrieval"]["native_prompt_sha256"] = "e" * 64
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == ("unavailable" if tamper else "observed")
    if not tamper: assert result["sessions"][0]["prior_router_receipt_matched"] is True


def test_baseline_coverage_names_excluded_candidate_sessions(evidence):
    audit, row = source_fixture("f" * 32)
    _typed_audit(audit, row); _upgrade(audit)
    evidence[2]["provider_invocations"].insert(0, row)
    evidence[5](audit, prefix="source-view")
    result = baseline.collect(parent=evidence[0], scope=evidence[1], report=evidence[2])
    assert result["status"] == "observed" and result["counts"]["sessions"] == 1
    assert result["session_scope"] == "baseline_source_sessions_only"
    assert result["candidate_sessions_excluded"] == 1 and result["all_coding_sessions_observed"] is False
    assert result["request_operation_coverage"]["all_sessions_observed"] is False
    assert result["request_operation_coverage"]["excluded_candidate_sessions"] == 1
    assert result["candidate_session_observation_schema"] == collector.SCHEMA


@pytest.mark.parametrize("change", [None, "report_binding", "descriptor_binding", "missing_recipe"])
def test_candidate_policy_is_joined_to_frozen_benchmark_selection(evidence, change):
    from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_policy import source_view_observation
    policy = SourceViewPolicy.from_mapping(profile="python-comments@1",
        protected_lines={b"module.py".hex(): [1, 2]})
    selected = source_view_observation(policy.profile, policy.to_dict())
    evidence[2].update(arm="full", **selected)
    descriptor = evidence[3]["session"]
    descriptor["view_policy_binding_sha256"] = selected["source_view_policy_binding_sha256"]
    if change == "report_binding": evidence[2]["source_view_policy_binding_sha256"] = "e" * 64
    elif change == "descriptor_binding": descriptor["view_policy_binding_sha256"] = "e" * 64
    elif change == "missing_recipe": evidence[2].pop("source_view_policy")
    digest = collector._sha(protocol.canonical_json(descriptor))
    evidence[3]["descriptor_sha256"] = evidence[4]["candidate_round_dispatch"]["descriptor_sha256"] = digest
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == ("observed" if change is None else "unavailable")


@pytest.mark.parametrize("unreceipted", [{}, 0, ""])
def test_false_like_unreceipted_attempt_is_unknown(evidence, unreceipted):
    evidence[2].update(provider_invocations=[], candidate_rounds=2, unreceipted_provider_attempt=unreceipted)
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None
