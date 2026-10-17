"""Actual native producer -> copied sealed evidence -> offline observation.

The existing native fixture authors container/worktree/peer isolation only.
Public instruction and router identities below are authored metadata. Native
resources, source indexing, sockets, audit bytes and cleanup are real. These
cases do not qualify a different-UID deployment or provider/model ingestion.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_observation as observation
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as binding_api
from test.api.test_repository_source_owner_broker import (
    owner, short_service_directory, initialize, tool, message, exchange)
from test.api.test_repository_source_parent_admission import native_parent


def _wait_for_response_accounting(owner, expected):
    # The client can receive a frame before send_frame's final cancellation
    # check. Normal-close fixtures wait for that real accounting boundary;
    # dedicated transport cases exercise cancellation in the intervening gap.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with owner._accounting_lock:
            if owner._inflight is None and len(owner._records) == expected:
                return
        time.sleep(0.005)
    raise AssertionError("owner response accounting did not complete before normal fixture close")


def _row(binding):
    descriptor = binding["descriptor"]
    return dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        provider="codex_cli", invocation_id="a" * 32, workspace=binding["workspace"],
        status="provider_returned", completion_authority=False,
        repository_source_policy=binding,
        external_container_boundary=dict(schema="supervisor-container-worker-boundary@1",
            container_id="e" * 64, image_id="sha256:" + "f" * 64,
            namespaces={key: "authored-" + key for key in ("pid", "mnt", "net")},
            owner_uid=descriptor["owner_uid"], worker_uid=descriptor["worker_uid"],
            manifest_sha256=descriptor["boundary_manifest_sha256"], purpose="coding", completion_authority=False),
        public_instruction=dict(schema="supervisor-public-instruction-inclusion@1", task_id="authored-task",
            task_cid="authored-task-cid", artifact_sha256="a" * 64, source_sha256="b" * 64,
            block_sha256="c" * 64, manifest_signature_verified=True, verbatim_utf8=True,
            artifact="/authored/instruction.json", context_cid="authored-context", manifest_cid="authored-manifest",
            source_path="instruction.md", source_bytes=31, semantic_minification_applied=False,
            source_freshness_verified=True, historical_replay=False,
            completion_authority=False, scope_expansion_authority=False))


@pytest.mark.parametrize("stale_read", [False, True])
def test_native_retained_copy_recounts_without_claiming_tool_delivery(owner, native_parent, monkeypatch, stale_read, request):
    monkeypatch.setattr(binding_api.protocol, "load_descriptor", lambda *args, **kwargs: copy.deepcopy(owner.descriptor))
    binding = binding_api.create_binding(workspace=owner.workspace,
        requested_policy="repository-source-raw@1", descriptor_path=owner.descriptor_path,
        descriptor_sha256=owner.descriptor_sha256)
    binding["usage"] = binding_api.unavailable_usage()
    row = _row(binding)
    report = dict(repository_source_policy="repository-source-raw@1", provider_invocations=[row],
        worker_cleanup_returncode=0, repository_source_dispatch={"helper_adoption_guaranteed": False},
        usage={"total_tokens": None})
    report["public_instruction"] = {key: row["public_instruction"][key] for key in (
        "artifact", "context_cid", "manifest_cid", "task_cid", "source_path", "source_sha256", "source_bytes",
        "completion_authority", "scope_expansion_authority")}
    report["public_instruction"]["sha256"] = row["public_instruction"]["artifact_sha256"]
    scope = dict(task_cid="authored-task-cid", task_alias="authored-task", baseline_commit="d" * 40,
        outputs=[dict(path="calc.py", effect="modify", baseline_sha256="e" * 64)])
    original_report = copy.deepcopy(report)
    initialize(owner)
    generation = tool(owner, "repository_status")["generation"]
    if stale_read:
        response = exchange(owner, message("tools/call", params=dict(name="repository_read",
            arguments=dict(generation=generation + 1, path="calc.py"))))
        assert response["ok"] is True  # Outer success does not mean tool success.
        assert response["message"]["result"]["isError"] is True
    else:
        assert "value + 1" in tool(owner, "repository_read", generation=generation, path="calc.py")["content"]
    assert tool(owner, "repository_context_query", generation=generation, query="add", max_bytes=4096)["candidates"]
    _wait_for_response_accounting(owner, 5)
    original = owner.close()
    assert original["cleanup_complete"] and original["native_child_released"]
    assert not native_parent.released  # Parent lifetime is not the child's audit.
    operations_path = owner.audit_path.parent / original["retained_mcp_audit"]["name"]
    paths = [owner.audit_path, operations_path]
    originals = {path: path.read_bytes() for path in paths}
    # Root-owned /home and owner-private HOME are actual protected ancestors;
    # neither the shared workspace (0775) nor /tmp qualifies as evidence custody.
    with tempfile.TemporaryDirectory(prefix=".owner-observation-", dir=Path.home()) as directory:
        parent = Path(directory) / "worker-observations"
        parent.mkdir(mode=0o750)
        for path, raw in originals.items():
            retained = parent / path.name
            retained.write_bytes(raw)
            retained.chmod(0o400)
        result = observation.collect(parent=parent, scope=scope, report=report)
        assert result["status"] == "observed", result
        assert result["session_count"] == 1
        assert result["transport"]["accepted_requests"] == 5
        assert result["transport"]["completed_frame_sends"] == 5
        assert result["transport"]["status_counts"]["ok"] == 5
        assert result["prepared_mcp_counts"]["tool_calls"] == 3
        assert result["prepared_mcp_counts"]["successful_reads"] == int(not stale_read)
        assert result["prepared_mcp_counts"]["errors"] == int(stale_read)
        correlation = result["operation_correlation"]
        assert correlation["status"] == "observed", correlation
        counts = correlation["counts"]
        assert counts["total_operations"] == counts["completed_socket_send_operations"] == 4
        assert counts["completed_socket_send_tool_results"] == 3
        assert counts["completed_socket_send_successful_tool_results"] == 3 - int(stale_read)
        assert counts["completed_socket_send_successful_reads"] == int(not stale_read)
        assert counts["completed_socket_send_failed_tool_results"] == int(stale_read)
        assert counts["unconfirmed_operations"] == 0
        assert correlation["worker_receipt_verified"] is False
        session = result["sessions"][0]
        assert session["audit_sha256"] == hashlib.sha256(originals[owner.audit_path]).hexdigest()
        assert session["operations_sha256"] == hashlib.sha256(originals[operations_path]).hexdigest()
        assert session["prepared_mcp_usage"]["retrieval_observation"]["successful_queries"] == 1
        if not stale_read:
            assert session["prepared_mcp_usage"]["read_observation"]
        assert result["delivered_tool_calls"] is result["delivered_reads"] is result["model_consumed_tokens"] is None
        assert result["provider_ingestion_verified"] is result["token_savings_measured"] is False
        assert result["native_parent_release_verified"] is result["worker_cleanup_independently_verified"] is False
        assert "value + 1" not in json.dumps(result) and str(owner.workspace) not in json.dumps(result)
        assert report == original_report
        request.node.user_properties.append(("owner_observation", json.dumps(result, sort_keys=True)))
        request.node.user_properties.append(("evidence_boundary",
            "New local native fixture; authored container/peer/router/task identities; copied sealed evidence; no provider ingestion"))
        # Corrupt only the copy, retaining the original counterexample and hashes.
        copied_operations = parent / operations_path.name
        copied_operations.chmod(0o600)
        copied_operations.write_bytes(copied_operations.read_bytes() + b"{}\n")
        copied_operations.chmod(0o400)
        rejected = observation.collect(parent=parent, scope=scope, report=report)
        assert rejected["status"] == "unavailable" and rejected["prepared_mcp_counts"] is None
    assert {path: path.read_bytes() for path in paths} == originals
    assert not native_parent.released
