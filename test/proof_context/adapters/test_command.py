"""PCCE-032 adversarial tests for the argv-only command adapter."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.proof_context.adapters.base import CancellationToken, execute_propose
from ipfs_accelerate_py.proof_context.adapters.command import (
    CommandAdapter, CommandPolicy, decode_structured_output, invoke_command,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CONTEXT_PACK_SCHEMA, MODEL_ROUTE_DECISION_SCHEMA, TASK_SPECIFICATION_SCHEMA,
    ContextPack, ModelRouteDecision, TaskSpecification,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError, MalformedError, ProofCancelledError, ProofTimeoutError,
)

CID = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajru"
CID_B = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrv"
CID_C = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrw"
OWNED = "src/demo.py"


def _policy(tmp_path: Path, code: str, *, timeout: float = 2, extra: tuple[str, ...] = ()) -> CommandPolicy:
    binary = os.path.realpath(sys.executable)
    return CommandPolicy(binary, (binary,), str(tmp_path), (str(tmp_path),), ("-c", code, *extra), timeout_seconds=timeout)


def _records() -> tuple[TaskSpecification, ContextPack, ModelRouteDecision]:
    task = TaskSpecification.from_mapping({"schema": TASK_SPECIFICATION_SCHEMA, "task_id": "PCCE-032", "objective_id": "PCCE-G300", "repository_state_cid": CID, "owned_paths": [OWNED], "declared_files": [OWNED], "route_cid": CID_B, "provenance": "live"})
    pack = ContextPack.from_mapping({"schema": CONTEXT_PACK_SCHEMA, "pack_cid": CID_C, "repository_state_cid": CID, "sufficiency": "sufficient", "task_id": "PCCE-032", "provenance": "live"})
    route = ModelRouteDecision.from_mapping({"schema": MODEL_ROUTE_DECISION_SCHEMA, "decision_cid": CID_B, "task_id": "PCCE-032", "repository_state_cid": CID, "provider": "local", "model": "agent", "revision": "r1", "tier": "medium", "provenance": "live"})
    return task, pack, route


def test_policy_rejects_relative_or_non_allowlisted_executable_and_cwd(tmp_path: Path) -> None:
    binary = os.path.realpath(sys.executable)
    with pytest.raises(BoundaryViolationError): CommandPolicy("python", (binary,), str(tmp_path), (str(tmp_path),))
    with pytest.raises(BoundaryViolationError): CommandPolicy(binary, (binary,), str(tmp_path), ("/tmp",))


def test_argv_is_literal_and_environment_is_hermetic(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    code = "import json,os,sys; print(json.dumps({'argv':sys.argv[1:], 'env':sorted(os.environ), 'cwd':os.getcwd()}))"
    policy = _policy(tmp_path, code, extra=(f"; touch {marker}",))
    result = invoke_command(policy, {"hello": "world"})
    observed = json.loads(result.stdout)
    assert observed["argv"] == [f"; touch {marker}"]
    assert not marker.exists()
    assert observed["cwd"] == str(tmp_path)
    assert "HOME" in observed["env"] and "PATH" in observed["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in observed["env"]


def test_output_is_bounded_redacted_and_nonzero_does_not_decode(tmp_path: Path) -> None:
    execution = invoke_command(_policy(tmp_path, "import sys; sys.stderr.write('token=super-secret\\n'); print('{}')"), {"x": 1})
    assert b"super-secret" not in execution.log_bytes and b"[redacted]" in execution.log_bytes
    oversized = _policy(tmp_path, "import sys; sys.stdout.write('x' * 2600000)")
    with pytest.raises(BoundaryViolationError): invoke_command(oversized, {"x": 1})


@pytest.mark.parametrize("output", [b"", b"[]", b"{} trailing", b"```json\\n{}\\n```", b'{"patch":"x"}'])
def test_structured_decoder_is_closed_and_fail_closed(output: bytes) -> None:
    with pytest.raises(MalformedError): decode_structured_output(output)


def test_timeout_and_cancellation_kill_the_process_group(tmp_path: Path) -> None:
    sleeper = _policy(tmp_path, "import time; time.sleep(5)", timeout=.1)
    with pytest.raises(ProofTimeoutError): invoke_command(sleeper, {"x": 1})
    token = CancellationToken()
    threading.Timer(.05, token.cancel).start()
    with pytest.raises(ProofCancelledError): invoke_command(_policy(tmp_path, "import time; time.sleep(5)"), {"x": 1}, token)


def test_timeout_terminates_descendant_processes(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    code = f"import pathlib,subprocess,sys,time; child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(10)"
    with pytest.raises(ProofTimeoutError): invoke_command(_policy(tmp_path, code, timeout=.1), {"x": 1})
    child_pid = int(pid_file.read_text())
    proc_state = Path(f"/proc/{child_pid}/stat")
    for _ in range(30):
        # A short-lived zombie is already terminated; PID 1 reaps it asynchronously.
        if not proc_state.exists() or proc_state.read_text().split()[2] == "Z":
            break
        time.sleep(.02)
    assert not proc_state.exists() or proc_state.read_text().split()[2] == "Z"


def test_adapter_accepts_only_strict_identity_bound_proposal(tmp_path: Path) -> None:
    code = "import json,sys; r=json.load(sys.stdin); print(json.dumps({'task_id':r['task_id'],'repository_state_cid':r['repository_state_cid'],'pack_cid':r['pack_cid'],'route_cid':r['route_cid'],'declared_files':r['declared_files'],'patch':'diff --git a/src/demo.py b/src/demo.py\\n','model':r['model'],'revision':r['revision'],'token_count':1,'cached_token_count':0,'latency_ms':99,'cost_micros':0}))"
    task, pack, route = _records()
    result = execute_propose(CommandAdapter(_policy(tmp_path, code)), task, pack, route)
    assert result.proposal.provenance == result.invocation.provenance == "live"
    assert result.accepted is result.approved is False


def test_adapter_rejects_patch_outside_declared_scope(tmp_path: Path) -> None:
    code = "import json,sys; r=json.load(sys.stdin); print(json.dumps({'task_id':r['task_id'],'repository_state_cid':r['repository_state_cid'],'pack_cid':r['pack_cid'],'route_cid':r['route_cid'],'declared_files':r['declared_files'],'patch':'diff --git a/src/secret.py b/src/secret.py\\n','model':r['model'],'revision':r['revision'],'token_count':1,'cached_token_count':0,'latency_ms':1,'cost_micros':0}))"
    task, pack, route = _records()
    with pytest.raises(BoundaryViolationError):
        execute_propose(CommandAdapter(_policy(tmp_path, code)), task, pack, route)
