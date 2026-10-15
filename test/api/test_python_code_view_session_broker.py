"""Task-local session custody and matched framed transport diagnostics."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
from ipfs_accelerate_py.agent_supervisor.runtime import python_code_view_session_broker as module
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_session_broker import (
    CodeViewSessionError,
    PythonCodeViewSessionBroker,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_wire_worktree import (
    build_python_code_view_wire_edit,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.python_code_view import (
    PythonCodeViewError,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import (
    PatchScope,
    WorktreeFenceError,
    create_isolated_worktree,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json

SID = "SESSION-FIXTURE"
SOURCE = (
    "# Audit λ\n"
    "def adjusted_total(base_quantity, adjustment_quantity):\n"
    '    """Keep literal λ."""\n'
    "    intermediate_quantity = base_quantity + adjustment_quantity\n"
    "    return intermediate_quantity * 2\n"
)


def _large():
    lines = ["# Repeated arithmetic stress fixture λ\n"]
    for index in range(8):
        a, b = "base_quantity_total_value", "adjustment_quantity_total_value"
        lines.append(f"def adjusted_total_{index}({a}, {b}):\n")
        previous = None
        for stage in range(6):
            local = f"intermediate_quantity_stage_{stage}"
            expr = " + ".join([a, b] * 6)
            if previous is not None:
                expr += " + " + previous
            lines.append(f"    {local} = {expr}\n")
            previous = local
        lines.append(f"    return {previous} * 2\n\n")
    return "".join(lines)


def _git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def tree(tmp_path, request):
    source = getattr(request, "param", SOURCE)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "session@example.com")
    _git(repo, "config", "user.name", "Session fixture")
    _git(repo, "config", "core.autocrlf", "false")
    for index in range(5):
        (repo / ("calc.py" if index == 0 else f"extra{index}.py")).write_bytes(source.encode())
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    store = WorktreeLifecycleStore(repo_root=repo, store_dir=tmp_path / "lifecycle", lease_seconds=300)
    with create_isolated_worktree(repo_root=repo, worktree_path=tmp_path / "isolated",
            task_id="SESSION-FIXTURE", lifecycle_store=store) as worktree:
        scope = PatchScope(allowed_paths=("calc.py", "extra1.py", "extra2.py", "extra3.py", "extra4.py"),
                           effect_paths=("calc.py", "extra1.py", "extra2.py", "extra3.py", "extra4.py"),
                           task_owned_paths=("calc.py", "extra1.py", "extra2.py", "extra3.py", "extra4.py"))
        yield worktree, scope, repo, source


def _open(tree, **kwargs):
    worktree, scope, _, _ = tree
    return PythonCodeViewSessionBroker(worktree=worktree, scope=scope, session_id=SID,
        lease_id=worktree.lease_id, fence=worktree.fence, **kwargs)


def _read(broker):
    return broker.read_view(path="calc.py", session_id=SID)


def _ack(broker, response):
    legend = response["state"]["legend_id"]
    broker.acknowledge_legend(legend_id=legend, read_id=response["read_id"], session_id=SID)
    broker.acknowledge_read(read_id=response["read_id"], view_id=response["state"]["view_id"], session_id=SID)


def _native(broker, response, acknowledge=True):
    view_id = response["state"]["view_id"]
    delivery = broker.deliver_native_source(view_id=view_id, session_id=SID)
    if acknowledge:
        broker.acknowledge_native_source(delivery_id=delivery["delivery_id"], view_id=view_id, session_id=SID)
    return delivery


def _request(broker, response):
    packet = broker.restore_read(response=response, session_id=SID)
    function = packet["functions"][0]
    slot = next(s for s in packet["slots"] if s["kind"] == "return" and s["function"] == function["name"])
    local = [a for a in function["aliases"] if a["kind"] == "local"][-1]
    request = build_python_code_view_wire_edit(view_id=packet["view_id"], slot_id=slot["slot_id"],
        before_sha256=slot["before_sha256"], expression=f"({local['alias']} * 3) + 1")
    return request, function["name"], local["original"]


def test_explicit_legend_and_snapshot_acknowledgements_control_references(tree):
    broker = _open(tree)
    first = _read(broker)
    assert _read(broker)["legend"] is not None, "emission alone is not acknowledgement"
    with pytest.raises(CodeViewSessionError, match="acknowledged"):
        broker.read_view(path="calc.py", session_id=SID, known_read_id=first["read_id"])
    _ack(broker, first)
    assert _read(broker)["legend"] is None
    ref = broker.read_view(path="calc.py", session_id=SID, known_read_id=first["read_id"])
    assert "state" not in ref
    assert broker.restore_read(response=ref, session_id=SID) == broker.restore_read(response=first, session_id=SID)


def test_same_label_different_broker_has_no_shared_acknowledgements(tree):
    first, second = _open(tree), _open(tree)
    response = _read(first)
    _ack(first, response)
    assert _read(second)["broker_id"] != response["broker_id"]
    with pytest.raises(CodeViewSessionError):
        second.acknowledge_legend(legend_id=response["state"]["legend_id"], read_id=response["read_id"], session_id=SID)
    with pytest.raises(CodeViewSessionError):
        second.restore_read(response=response, session_id=SID)


def test_valid_current_response_not_emitted_by_broker_is_refused(tree):
    broker = _open(tree)
    response = _read(broker)
    _ack(broker, response)
    # The acknowledged legend makes a valid omission possible, but the broker
    # has not yet emitted that distinct body/read identity.
    forged = {**response, "legend": None}
    forged["read_id"] = cid_for_dag_json({k: v for k, v in forged.items() if k != "read_id"})
    assert forged["read_id"] != response["read_id"]
    with pytest.raises(CodeViewSessionError, match="not emitted"):
        broker.restore_read(response=forged, session_id=SID)
    emitted = _read(broker)
    assert emitted == forged
    assert broker.restore_read(response=emitted, session_id=SID)["view_id"] == response["state"]["view_id"]


@pytest.mark.parametrize("fault", ["private_only", "delivery_only", "wrong_bytes"])
def test_native_delivery_and_ack_are_required_for_prepare(tree, fault):
    broker = _open(tree)
    response = _read(broker)
    request, _, _ = _request(broker, response)
    if fault != "private_only":
        _native(broker, response, acknowledge=fault != "delivery_only")
    source = tree[3] if fault != "wrong_bytes" else tree[3] + "# changed\n"
    with pytest.raises(CodeViewSessionError, match="acknowledged"):
        broker.prepare_edit(wire_request=request, visible_sources={"calc.py": source}, session_id=SID)
    assert (tree[0].worktree_path / "calc.py").read_bytes() == tree[3].encode()


@pytest.mark.parametrize("tree", [SOURCE, SOURCE.replace("\n", "\r\n"), SOURCE.removesuffix("\n")], indirect=True)
def test_native_edit_invalidates_preimages_but_reuses_exact_legend(tree):
    broker = _open(tree)
    response = _read(broker)
    _ack(broker, response)
    delivery = _native(broker, response)
    request, _, local = _request(broker, response)
    prepared = broker.prepare_edit(wire_request=json.dumps(request), visible_sources={"calc.py": delivery["source"]}, session_id=SID)
    result = broker.apply_edit(prepared=prepared, visible_sources={"calc.py": delivery["source"]}, session_id=SID)
    assert result.applied, result.to_dict()
    expected = tree[3].replace(f"return {local} * 2", f"return ({local} * 3) + 1")
    assert (tree[0].worktree_path / "calc.py").read_bytes() == expected.encode()
    assert (tree[2] / "calc.py").read_bytes() == tree[3].encode()
    after = _read(broker)
    assert after["legend"] is None and after["state"]["legend_id"] == response["state"]["legend_id"]
    assert after["state"]["view_id"] != response["state"]["view_id"]
    with pytest.raises(CodeViewSessionError):
        broker.prepare_edit(wire_request=request, visible_sources={"calc.py": delivery["source"]}, session_id=SID)
    new_request, _, _ = _request(broker, after)
    with pytest.raises(CodeViewSessionError, match="acknowledged"):
        broker.prepare_edit(wire_request=new_request, visible_sources={"calc.py": expected}, session_id=SID)
    with pytest.raises(CodeViewSessionError):
        broker.deliver_native_source(view_id=after["state"]["view_id"], known_delivery_id=delivery["delivery_id"], session_id=SID)
    execution = subprocess.run([sys.executable, "-B", "-c", "from calc import adjusted_total; assert adjusted_total(4,3)==22; assert adjusted_total(-5,4)==-2"],
        cwd=tree[0].worktree_path, text=True, capture_output=True)
    assert execution.returncode == 0, execution.stderr
    assert prepared.receipt()["lean_admits"] == 0


def test_acknowledged_native_reference_still_checks_current_bytes(tree):
    broker = _open(tree)
    response = _read(broker)
    delivery = _native(broker, response)
    reference = broker.deliver_native_source(view_id=response["state"]["view_id"],
        known_delivery_id=delivery["delivery_id"], session_id=SID)
    assert "source" not in reference and reference["source_sha256"] == delivery["source_sha256"]
    (tree[0].worktree_path / "calc.py").write_bytes((tree[3] + "# changed\n").encode())
    with pytest.raises(PythonCodeViewError, match="preimage"):
        broker.deliver_native_source(view_id=response["state"]["view_id"],
            known_delivery_id=delivery["delivery_id"], session_id=SID)


@pytest.mark.parametrize("fault", ["ttl", "lease", "scope"])
def test_expiry_and_scope_binding_refuse_and_clear_retained_state(tree, monkeypatch, fault):
    broker = _open(tree)
    _read(broker)
    if fault == "ttl":
        monkeypatch.setattr(module, "_monotonic", lambda: broker._deadline + 1)
    elif fault == "lease":
        record = tree[0].lifecycle_store.load_workspace(tree[0].worktree_path)
        monkeypatch.setattr(module, "_wall_time", lambda: record.expires_at + 1)
    else:
        broker._scope = PatchScope(allowed_paths=("extra1.py",), effect_paths=("calc.py",), task_owned_paths=("extra1.py",))
    with pytest.raises(CodeViewSessionError):
        _read(broker)
    assert not broker._views and not broker._legends


def test_close_releases_handles_and_is_idempotent(tree):
    broker = _open(tree)
    response = _read(broker)
    _ack(broker, response)
    assert broker.close_session(session_id=SID)["retained_encoded_bytes"] == 0
    assert broker.close_session(session_id=SID)["lean_admits"] == 0
    with pytest.raises(CodeViewSessionError, match="closed"):
        _read(broker)


def test_real_fence_renewal_refuses_and_clears_all_custody(tree):
    broker = _open(tree)
    response = _read(broker)
    _ack(broker, response)
    _native(broker, response)
    worktree = tree[0]
    renewed = worktree.lifecycle_store.renew_lease(worktree.worktree_path,
        lease_id=worktree.lease_id, expected_fence=worktree.fence)
    assert renewed.fence > worktree.fence
    try:
        with pytest.raises(WorktreeFenceError, match="fence|owner|stale"):
            _read(broker)
        assert broker._closed
        assert not any((broker._views, broker._legends, broker._reads, broker._native,
                        broker._known_legends, broker._known_reads, broker._known_native))
    finally:
        # Fixture controller follows its real renewal so context cleanup retains
        # current ownership; the broker stays closed and cannot follow it.
        worktree.fence = renewed.fence


def test_retention_limit_refuses_without_partial_publication(tree):
    broker = _open(tree)
    for path in ("calc.py", "extra1.py", "extra2.py", "extra3.py"):
        broker.read_view(path=path, session_id=SID)
    before = set(broker._views)
    with pytest.raises(CodeViewSessionError, match="retention"):
        broker.read_view(path="extra4.py", session_id=SID)
    assert set(broker._views) == before and len(broker._views) == module.MAX_VIEWS


def _render(frame):
    return json.dumps(frame, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("tree,reads", [(SOURCE, 1), (_large(), 1), (_large(), 6)], indirect=["tree"], ids=["tiny-one", "large-one", "large-six"])
def test_matched_framed_broker_transport_sessions(tree, reads, record_property):
    # All strategies use the SAME expression broker and initial legend/snapshot.
    # This isolates read transport; it is not a generic native-editor benchmark.
    source = tree[3]
    transcript_stats = {}
    outcomes = []
    for strategy in ("native_full", "native_reference", "session_reference"):
        # Each paired candidate starts from the same clean worktree bytes.
        # Existing native executor forbids dirty apply, so reset only this private
        # fixture between diagnostic candidates, keeping the canonical repo intact.
        _git(tree[0].worktree_path, "reset", "--hard", "HEAD")
        broker = _open(tree)
        frames = []
        def record(role, name, content, frames=frames):
            frames.append({"role": role, "name": name, "content": content})
        record("system", "tool_guidance", {
            "profile": "python-arithmetic-model-packet-session-columns@1",
            "read_view": "Supply session_id and path; use known_read_id only after legend and read acknowledgements.",
            "native_source": "Supply current view_id; acknowledge delivery before editing. References require acknowledged exact preimages.",
            "submit_edit": "Supply session_id and the wire edit with full view ID, slot ID, preimage hash and literal alias expression. The controller prepares and applies with caller native visibility.",
            "close": "Supply session_id after checking the post-edit view. No message grants execution, task acceptance or Lean admission.",
        })
        record("user", "task", "Inspect the first function and change its return to three times the last intermediate quantity plus one.")
        record("assistant", "read_view", {"session_id": SID, "path": "calc.py"})
        response = _read(broker)
        record("tool", "read_view", response)
        record("assistant", "legend_ack", {"session_id": SID, "legend_id": response["state"]["legend_id"], "read_id": response["read_id"]})
        legend_ack = broker.acknowledge_legend(legend_id=response["state"]["legend_id"], read_id=response["read_id"], session_id=SID)
        record("tool", "legend_ack", legend_ack)
        record("assistant", "state_ack", {"session_id": SID, "read_id": response["read_id"], "view_id": response["state"]["view_id"]})
        state_ack = broker.acknowledge_read(read_id=response["read_id"], view_id=response["state"]["view_id"], session_id=SID)
        record("tool", "state_ack", state_ack)
        native = None
        for _ in range(reads):
            if strategy == "session_reference":
                record("assistant", "read_reference", {"session_id": SID, "path": "calc.py", "known_read_id": response["read_id"]})
                reply = broker.read_view(path="calc.py", known_read_id=response["read_id"], session_id=SID)
                assert broker.restore_read(response=reply, session_id=SID) == broker.restore_read(response=response, session_id=SID)
                record("tool", "read_reference", reply)
            else:
                arguments = {"view_id": response["state"]["view_id"], "session_id": SID}
                if native and strategy == "native_reference":
                    arguments["known_delivery_id"] = native["delivery_id"]
                record("assistant", "native_read", arguments)
                reply = broker.deliver_native_source(view_id=response["state"]["view_id"], session_id=SID,
                    known_delivery_id=native["delivery_id"] if native and strategy == "native_reference" else None)
                record("tool", "native_read", reply)
                if "source" in reply:
                    native = reply
                    record("assistant", "native_ack", {"session_id": SID, "delivery_id": reply["delivery_id"], "view_id": response["state"]["view_id"]})
                    record("tool", "native_ack", broker.acknowledge_native_source(delivery_id=reply["delivery_id"],
                        view_id=response["state"]["view_id"], session_id=SID))
        if native is None:
            record("assistant", "required_native_expansion", {"session_id": SID, "view_id": response["state"]["view_id"]})
            native = _native(broker, response, acknowledge=False)
            record("tool", "required_native_expansion", native)
            record("assistant", "native_ack", {"session_id": SID, "delivery_id": native["delivery_id"], "view_id": response["state"]["view_id"]})
            record("tool", "native_ack", broker.acknowledge_native_source(delivery_id=native["delivery_id"],
                view_id=response["state"]["view_id"], session_id=SID))
        request, name, local = _request(broker, response)
        record("assistant", "submit_edit", {"session_id": SID, "wire_request": request})
        prepared = broker.prepare_edit(wire_request=_render(request), visible_sources={"calc.py": native["source"]}, session_id=SID)
        result = broker.apply_edit(prepared=prepared, visible_sources={"calc.py": native["source"]}, session_id=SID)
        assert result.applied, result.to_dict()
        record("tool", "submit_edit", result.to_dict())
        record("assistant", "post_edit_view", {"session_id": SID, "path": "calc.py"})
        fresh = _read(broker)
        record("tool", "post_edit_view", fresh)
        record("assistant", "close", {"session_id": SID})
        record("tool", "close", broker.close_session(session_id=SID))
        expected = source.replace(f"return {local} * 2", f"return ({local} * 3) + 1", 1)
        assert (tree[0].worktree_path / "calc.py").read_bytes() == expected.encode()
        positive, negative = (22, -2) if source == SOURCE else (757, -107)
        diagnostic = f"from calc import {name}; assert {name}(4,3)=={positive}; assert {name}(-5,4)=={negative}; assert {name}(0,0)==1"
        record("assistant", "diagnostic", {"path": "calc.py", "python_assertions": diagnostic})
        execution = subprocess.run([sys.executable, "-B", "-c", diagnostic],
            cwd=tree[0].worktree_path, text=True, capture_output=True)
        assert execution.returncode == 0, execution.stderr
        record("tool", "diagnostic", {"native_fixture_execution": True, "task_acceptance": None, "lean_admits": 0})
        outcomes.append(hashlib.sha256(expected.encode()).hexdigest())
        transcript_stats[strategy] = {"frame_count": len(frames), "framed_transcript_bytes": len(_render(frames).encode()),
            "sum_frame_bytes": sum(len(_render(f).encode()) for f in frames),
            "full_history_prefix_bytes_sum": sum(len(_render(frames[:i]).encode()) for i in range(1, len(frames)+1)),
            "transcript_sha256": hashlib.sha256(_render(frames).encode()).hexdigest(),
            "frames": frames,
            "native_source_body_deliveries": sum(isinstance(f["content"], dict) and "source" in f["content"]
                and f["content"].get("schema") == "supervisor-python-code-view-native-delivery@1" for f in frames),
            "model_input_tokens": None, "provider_billed_usage": None}
    assert len(set(outcomes)) == 1 and (tree[2] / "calc.py").read_bytes() == source.encode()
    record_property("python_code_view_session_probe", _render({"schema":"supervisor-python-code-view-session-probe@1",
        "fixture":"tiny-one" if source == SOURCE else ("large-one" if reads == 1 else "large-six"),
        "source_bytes":len(source.encode()),"requested_reads":reads,"strategies":transcript_stats,
        "matched_final_native_sha256":outcomes[0],"shared_initial_expression_broker_setup":True,
        "public_requests_and_replies_recorded":True,"shared_tool_guidance_recorded":True,
        "native_fixture_execution_passed":True,"tokenizer_compatibility":None,"provider_calls":0,"downloads":0,"lean_admits":0}))
