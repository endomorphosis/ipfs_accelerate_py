"""Provider observations are scoped activity, never completion authority."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_provider as provider


def binding_fixture(tmp_path, monkeypatch):
    config = {"state_root": str(tmp_path), "run_id": "test-run"}
    native_id = "attempt:" + "1" * 32
    path = tmp_path / "provider-activity" / (hashlib.sha256(native_id.encode()).hexdigest() + ".json")
    body = {"schema": "wwl.native-provider-activity-binding@1", "run_id": "test-run", "task_id": "WWL-001",
        "task_cid": "outer-cid", "portal_task_cid": "portal-cid", "native_attempt_id": native_id,
        "portal_attempt_number": 1, "capability_nonce": "2" * 64, "activity_path": str(path), "issued_at": time.time()}
    body["binding_id"] = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    record = tmp_path / "provider-activity-binding.json"
    record.write_text(json.dumps(body));record.chmod(0o600)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_TASK_CID", "portal-cid")
    return config, body, path


def test_activity_binding_requires_exact_native_attempt_and_untampered_owned_file(tmp_path, monkeypatch):
    config, binding, _ = binding_fixture(tmp_path, monkeypatch)
    assert provider._activity_binding(config, "WWL-001", 1) == binding
    assert provider._activity_binding(config, "WWL-001", 2) is None
    assert provider._activity_binding(config, "WWL-002", 1) is None
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_TASK_CID", "prior-portal-cid")
    assert provider._activity_binding(config, "WWL-001", 1) is None
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_TASK_CID", "portal-cid")
    record = tmp_path / "provider-activity-binding.json"
    binding["native_attempt_id"] = "late-prior-owner"
    record.write_text(json.dumps(binding))
    assert provider._activity_binding(config, "WWL-001", 1) is None


def test_only_matched_distinct_completed_tools_advance_activity(tmp_path, monkeypatch):
    _, binding, path = binding_fixture(tmp_path, monkeypatch)
    observer = provider._SemanticActivity(binding)
    for event in ({"type": "thought", "data": "spam"}, {"type": "text", "data": "completed"},
                  {"type": "tool_call_update", "toolCallId": "unknown", "status": "completed"}):
        observer.event(event)
    assert not path.exists()
    start = {"type": "tool_call", "toolCallId": "call-1", "toolName": "read_file", "status": "pending"}
    finish = {"type": "tool_call_update", "toolCallId": "call-1", "status": "completed"}
    observer.feed((json.dumps(start) + "\n" + json.dumps(finish) + "\n").encode())
    first = path.read_bytes()
    for _ in range(100):
        observer.event(finish);observer.event(start);observer.event({"type": "thought", "data": "more spam"})
    assert path.read_bytes() == first
    value = json.loads(first)
    assert value["sequence"] == 1 and value["completion_authority"] is False
    assert value["binding_id"] == binding["binding_id"] and value["native_attempt_id"] == binding["native_attempt_id"]
    assert "read_file" not in first.decode() and "call-1" not in first.decode()
    assert os.stat(path).st_mode & 0o077 == 0


def test_overlong_or_embedded_tool_frames_cannot_fake_progress(tmp_path, monkeypatch):
    _, binding, path = binding_fixture(tmp_path, monkeypatch)
    observer = provider._SemanticActivity(binding)
    observer.feed(b"x" * (1024 * 1024 + 1) + b"\n")
    observer.feed(json.dumps({"type": "text", "data": '{"type":"tool_call_update","status":"completed"}'}).encode() + b"\n")
    assert not path.exists() and not observer.buffer


def test_live_relay_publishes_completed_tool_before_provider_exits(tmp_path, monkeypatch):
    _, binding, path = binding_fixture(tmp_path, monkeypatch)
    observer = provider._SemanticActivity(binding)
    output = io.StringIO()
    monkeypatch.setattr(provider.sys, "stdout", output)
    code = """import json,pathlib,sys,time
print(json.dumps({'type':'tool_call','toolCallId':'call-live','toolName':'read_file','status':'pending'}),flush=True)
print(json.dumps({'type':'tool_call_update','toolCallId':'call-live','status':'completed'}),flush=True)
p=pathlib.Path(sys.argv[1]);deadline=time.monotonic()+3
while not p.exists() and time.monotonic()<deadline:time.sleep(.01)
raise SystemExit(0 if p.exists() else 9)
"""
    assert provider._relay_with_activity([sys.executable, "-u", "-c", code, str(path)], observer) == 0
    assert json.loads(path.read_text())["sequence"] == 1
    assert "call-live" in output.getvalue()
