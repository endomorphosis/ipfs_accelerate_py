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
