"""Actual registered children consume one host-native candidate dispatch.

The native task, claim, grant, allocation, source generation, run lease, census,
Popen births and Unix peer PID/birth checks are real. This single-UID host
authors only the missing deployment UID separation. No provider or target code
executes, and these tests do not qualify the production container handoff.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import struct
import socket
import subprocess
import sys
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_client as client
from ipfs_accelerate_py.agent_supervisor.runtime.candidate_dispatch_broker import CandidateDispatchBroker
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as read_client
from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_round_broker import WorkerCandidateRoundBroker
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
from ipfs_accelerate_py.agent_supervisor.runtime.native_candidate_round_worker import run_native_candidate_round_worker
from test.api.test_native_candidate_round_admission import (
    candidate_case, capture,
)


POLICY = dict(provider="codex_cli", model="authored-candidate-no-provider", timeout=30,
    max_output_tokens=128, reasoning_effort="high")


CHILD = r'''
import json,os,signal,socket,sys
from pathlib import Path
from types import SimpleNamespace
signal.alarm(25)
sys.path.insert(0,sys.argv[1])
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as transport
worker_uid=int(sys.argv[4])
# Author this host's absent distinct UID; retain descriptor bytes/modes,
# kernel server PID/start ticks, socket inode and owner peer observations.
transport.os=SimpleNamespace(**{name:getattr(os,name) for name in dir(os)
    if name not in {'getuid','geteuid','access'}},getuid=lambda:worker_uid,
    geteuid=lambda:worker_uid,access=lambda *args:False)
transport._protected_parent=lambda parent,owner:parent.stat()
handle=None
for line in sys.stdin:
    row=json.loads(line)
    try:
        if row['operation']=='prepare':
            handle=client.prepare_candidate_router_input(descriptor_path=Path(row.get('descriptor_path',sys.argv[2])),
                descriptor_sha256=row.get('descriptor_sha256',sys.argv[3]),prompt=row['prompt'],workspace=Path.cwd())
            result=handle.validate_for_run(prompt=row['prompt'],workspace=Path.cwd(),
                boundary_sha256=row['boundary_sha256'],worker_attempt_id=row['worker_attempt_id'],
                **row['policy'])
        elif row['operation']=='validate':
            result=handle.validate_for_run(prompt=row['prompt'],workspace=Path.cwd(),
                boundary_sha256=row['boundary_sha256'],worker_attempt_id=row['worker_attempt_id'],
                **row['policy'])
        elif row['operation']=='authorize':
            result=handle.authorize_dispatch(invocation_id=row['invocation_id'])
        elif row['operation']=='drop_ack':
            descriptor=client.load_descriptor(Path(sys.argv[2]),sys.argv[3],workspace=Path.cwd())
            request=client.dispatch_request(descriptor,operation='consume',invocation_id=row['invocation_id'])
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as channel:
                channel.settimeout(5)
                channel.connect(descriptor['socket_path'])
                client.transport.send_frame(channel,request)
                channel.shutdown(socket.SHUT_RDWR)
            result={'ack_received':False}
        elif row['operation']=='stop':
            print(json.dumps({'ok':True,'stopped':True}),flush=True)
            break
        else:
            raise ValueError('unknown authored operation')
        print(json.dumps({'ok':True,'result':result},sort_keys=True),flush=True)
    except Exception as error:
        print(json.dumps({'ok':False,'error_type':type(error).__name__},sort_keys=True),flush=True)
'''


def _row(case, operation="prepare", **changes):
    return dict(operation=operation, prompt=case.candidate_prompt.provider_prompt.decode("utf-8"),
        boundary_sha256=json.loads(case.worker_attempt._common)["boundary_manifest_sha256"],
        worker_attempt_id=case.worker_attempt.attempt_id, policy=dict(POLICY)) | changes


def _reply(child, row):
    child.stdin.write(json.dumps(row) + "\n")
    child.stdin.flush()
    line = child.stdout.readline()
    assert line, "bounded authored client exited before its reply"
    return json.loads(line)


def _uid_seam(broker, monkeypatch):
    """Keep actual peer PID/ancestry and author only host UID distinction."""
    protocol = broker.protocol
    original_identity, original_peer = protocol.process_identity, broker._peer
    owner_pid = os.getpid()
    def identity(pid):
        observed = original_identity(pid)
        return observed if pid == owner_pid else {**observed, "uid": broker.worker_uid}
    monkeypatch.setattr(protocol, "process_identity", identity)
    class ChannelUID:
        def __init__(self, channel):
            self.channel = channel
        def getsockopt(self, *args):
            pid, _uid, group = struct.unpack("3i", self.channel.getsockopt(*args))
            return struct.pack("3i", pid, broker.worker_uid, group)
    monkeypatch.setattr(broker, "_peer", lambda channel: original_peer(ChannelUID(channel)))


@pytest.fixture
def dispatch_case(candidate_case, monkeypatch):
    case = candidate_case
    gate = capture(case)
    broker = CandidateDispatchBroker(admission=gate, **POLICY)
    _uid_seam(broker, monkeypatch)
    children = []
    def spawn(*, registered=True):
        def birth():
            child = subprocess.Popen([sys.executable, "-I", "-c", CHILD,
                str(Path(__file__).resolve().parents[2]), str(broker.descriptor_path),
                broker.descriptor_sha256, str(broker.worker_uid)], cwd=case.native.workspace,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            children.append(child)
            case.children.append(child)
            return child
        return gate.execute_spawn(birth, broker.note_spawned) if registered else birth()
    yield case, gate, broker, spawn
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream is not None:
                stream.close()
    broker.close()


def test_actual_registered_child_checks_and_consumes_once(dispatch_case):
    case, gate, broker, spawn = dispatch_case
    child = spawn()
    prepared = _reply(child, _row(case))
    assert prepared["ok"] is True, prepared
    invocation_id = uuid.uuid4().hex
    allowed = _reply(child, dict(operation="authorize", invocation_id=invocation_id))
    assert allowed["ok"] is True, allowed
    assert type(allowed["result"]["timeout_seconds"]) is int
    assert 1 <= allowed["result"]["timeout_seconds"] <= POLICY["timeout"]
    replay = _reply(child, dict(operation="authorize", invocation_id=uuid.uuid4().hex))
    assert replay["ok"] is False
    assert gate.process is child and child.pid != case.first_process.pid
    assert _reply(child, dict(operation="stop"))["stopped"] is True
    child.wait(timeout=5)
    assert child.returncode == 0
    assert case.native.source.get_task(case.native.task_cid).status == "in_progress"
    assert case.candidate_prompt.dispatch_usable is False


def test_foreign_sibling_cannot_check_the_registered_child_descriptor(dispatch_case):
    case, _gate, _broker, spawn = dispatch_case
    registered = spawn()
    foreign = spawn(registered=False)
    assert _reply(foreign, _row(case))["ok"] is False
    assert _reply(registered, _row(case))["ok"] is True


@pytest.mark.parametrize("change", ["run_lease", "grant", "claim"])
def test_native_authority_drift_refuses_consume_after_successful_check(dispatch_case, change):
    case, _gate, _broker, spawn = dispatch_case
    child = spawn()
    assert _reply(child, _row(case))["ok"] is True
    if change == "run_lease":
        case.run_coordinator.release(case.run_lease, reason="authored dispatch drift")
        case.run_lease = None
    elif change == "grant":
        case.native.server._command_gateway.revoke_grant(case.native.daemon_grant_id)
    else:
        claim = case.native.daemon.coordinator.get_task_claim(case.native.attempt.claim_id)
        lease = case.native.daemon.coordinator.protect_task_claim(claim)
        case.native.daemon.coordinator.release(lease, reason="authored dispatch claim drift")
    assert _reply(child, dict(operation="authorize", invocation_id=uuid.uuid4().hex))["ok"] is False


def test_complete_prompt_policy_and_attempt_bindings_remain_exact(dispatch_case):
    case, _gate, _broker, spawn = dispatch_case
    child = spawn()
    assert _reply(child, _row(case))["ok"] is True
    changes = [dict(prompt=case.candidate_prompt.provider_prompt.decode() + "\nchanged"),
        dict(worker_attempt_id="f" * 32), dict(boundary_sha256="f" * 64)]
    for field, value in ( ("provider", "grok_cli"), ("model", "foreign-model"),
            ("timeout", POLICY["timeout"] + 1), ("max_output_tokens", 256), ("reasoning_effort", "low")):
        changes.append(dict(policy={**POLICY, field:value}))
    for changed in changes:
        assert _reply(child, _row(case, "validate", **changed))["ok"] is False, changed
    assert _reply(child, _row(case, "validate"))["ok"] is True


def test_candidate_source_change_between_check_and_consume_refuses(dispatch_case):
    case, _gate, _broker, spawn = dispatch_case
    child = spawn()
    assert _reply(child, _row(case))["ok"] is True
    path = case.native.workspace / "module.py"
    path.write_bytes(path.read_bytes() + b"\n# source changed after dispatch check\n")
    assert _reply(child, dict(operation="authorize", invocation_id=uuid.uuid4().hex))["ok"] is False


def test_server_close_retires_endpoint_and_refuses_later_authorization(dispatch_case):
    case, _gate, broker, spawn = dispatch_case
    child = spawn()
    assert _reply(child, _row(case))["ok"] is True
    broker.close()
    assert not broker.socket_path.exists() and not broker.descriptor_path.exists()
    assert not broker._thread.is_alive()
    assert _reply(child, dict(operation="authorize", invocation_id=uuid.uuid4().hex))["ok"] is False


def _owner_check(broker):
    request = client.dispatch_request(broker.descriptor, operation="check", invocation_id=uuid.uuid4().hex)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(5)
        channel.connect(str(broker.socket_path))
        client.transport.send_frame(channel, request)
        try:
            response = client.transport.receive_frame(channel)
        except (OSError, ValueError, RuntimeError):
            return False
    return response.get("ok") is True


def test_owner_and_prebirth_requests_cannot_replace_registered_child_custody(dispatch_case):
    case, _gate, broker, spawn = dispatch_case
    assert _owner_check(broker) is False
    child = spawn()
    assert _owner_check(broker) is False
    assert _reply(child, _row(case))["ok"] is True


def test_lost_consume_ack_does_not_restore_dispatch(dispatch_case):
    case, _gate, _broker, spawn = dispatch_case
    child = spawn()
    assert _reply(child, _row(case))["ok"] is True
    dropped = _reply(child, dict(operation="drop_ack", invocation_id=uuid.uuid4().hex))
    assert dropped["ok"] is True and dropped["result"]["ack_received"] is False
    # The one server loop processes this next fresh request after the lost ACK;
    # a new request ID and a new client-side check cannot refund consumption.
    repeated = _reply(child, _row(case))
    if repeated["ok"]:
        repeated = _reply(child, dict(operation="authorize", invocation_id=uuid.uuid4().hex))
    assert repeated["ok"] is False


def test_v1_candidate_read_descriptor_cannot_admit_a_router_call(dispatch_case):
    case, _gate, _broker, spawn = dispatch_case
    old = WorkerCandidateRoundBroker(case.snapshot)
    try:
        assert read_client.validate_descriptor(old.descriptor)["schema"] == read_client.DESCRIPTOR_SCHEMA
        child = spawn()
        assert _reply(child, _row(case, descriptor_path=str(old.descriptor_path),
            descriptor_sha256=old.descriptor_sha256))["ok"] is False
        assert _reply(child, _row(case))["ok"] is True
    finally:
        old.close()


def test_raw_prepared_candidate_json_never_admits_ordinary_router(dispatch_case, monkeypatch):
    case, _gate, _broker, _spawn = dispatch_case
    from types import SimpleNamespace
    def forbidden(*args, **kwargs):
        pytest.fail("raw advisory candidate reached provider discovery")
    monkeypatch.setitem(sys.modules, "ipfs_accelerate_py.llm_allocation.intelligence_index",
        SimpleNamespace(discover_available_providers=forbidden, select_efficient_route=forbidden))
    with pytest.raises((ValueError, RuntimeError)):
        router.run(prompt=case.candidate_prompt.provider_prompt.decode(), provider=POLICY["provider"],
            model=POLICY["model"], timeout=POLICY["timeout"], max_output_tokens=POLICY["max_output_tokens"])


ROUTER_CHILD = r'''
import hashlib,json,os,signal,sys
from pathlib import Path
from types import SimpleNamespace
signal.alarm(25)
# Literal owner-supplied package roots, independent of the candidate descriptor.
sys.path[:0]=json.loads(sys.argv[1])
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as transport
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
from ipfs_accelerate_py.llm_allocation import intelligence_index
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
worker_uid=int(sys.argv[4])
transport.os=SimpleNamespace(**{name:getattr(os,name) for name in dir(os)
    if name not in {'getuid','geteuid','access'}},getuid=lambda:worker_uid,
    geteuid=lambda:worker_uid,access=lambda *args:False)
transport._protected_parent=lambda parent,owner:parent.stat()
artifact,artifact_sha,attempt_id=Path(sys.argv[5]),sys.argv[6],sys.argv[7]
output=Path(sys.argv[8])
exit_code=int(sys.argv[9])
prompt=sys.stdin.buffer.read().decode('utf-8')
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==artifact_sha
def authored_boundary(*,artifact,expected_sha256,workspace,purpose='coding'):
    assert Path(artifact).resolve()==Path(sys.argv[5]).resolve()
    assert hashlib.sha256(Path(artifact).read_bytes()).hexdigest()==expected_sha256==artifact_sha
    assert Path(workspace)==Path.cwd() and purpose=='coding'
    return dict(schema='supervisor-container-worker-boundary@1',manifest_sha256=artifact_sha,
        owner_uid=os.getuid(),worker_uid=worker_uid,purpose=purpose,
        authored_host_deployment_only=True,completion_authority=False)
boundary.verify_container_worker_boundary=authored_boundary
intelligence_index.discover_available_providers=lambda:['codex_cli']
intelligence_index.select_efficient_route=lambda **kwargs:SimpleNamespace(
    provider='codex_cli',model_name=kwargs['model_name'],catalog_revision='authored-no-provider')
llm_router.get_llm_provider=lambda *args,**kwargs:object()
observed=[]
thread='authored-native-candidate-thread-0001'
rollout=Path(os.environ['CODEX_HOME'])/'sessions/2026/10/15'/('rollout-'+thread+'.jsonl')
def generate(model_prompt,**kwargs):
    assert kwargs['allow_local_fallback'] is False and kwargs['allow_cross_provider_fallback'] is False
    assert kwargs['side_effecting'] is True and kwargs['provider']=='codex_cli'
    observed.append(model_prompt)
    assert len(observed)==1
    # These authored categories test real recovery/custody, not measured model cost.
    rows=[dict(type='session_meta',payload=dict(id=thread,cwd=str(Path.cwd()))),
        dict(type='event_msg',payload=dict(type='token_count',info=dict(total_token_usage=dict(
            input_tokens=31,cached_input_tokens=7,output_tokens=4,total_tokens=35))))]
    if exit_code==0:
        rows.append(dict(type='event_msg',payload=dict(type='task_complete')))
    rollout.parent.mkdir(parents=True)
    rollout.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    set_last_cli_observation('codex_cli',dict(exit_code=exit_code,thread_id=thread,
        prompt_tokens=31,cached_tokens=7,completion_tokens=4))
    return 'Authored candidate response; no task completion or target execution.'
llm_router.generate_text=generate
handle=client.prepare_candidate_router_input(descriptor_path=Path(sys.argv[2]),
    descriptor_sha256=sys.argv[3],prompt=prompt,workspace=Path.cwd())
policy=json.loads(sys.argv[10])
counter=router.PromptTokenCounter(provider=policy['provider'],model=policy['model'],
    tokenizer_id='authored-utf8-byte-proxy',tokenizer_revision='fixture-v1',
    special_token_policy='literal-utf8',count_tokens=lambda text:len(text.encode('utf-8')),
    count_kind='estimate',compatibility_basis='authored_fixture_only')
receipts=[]
options=dict(prompt=prompt,container_boundary=artifact,container_boundary_sha256=artifact_sha,
    owner_worker_attempt_id=attempt_id,candidate_round_input=handle,prompt_token_counter=counter,
    receipt_observer=receipts.append,**policy)
error_type=None
try:
    router.run(**options)
except RuntimeError as error:
    if exit_code==0:
        raise
    error_type=type(error).__name__
assert len(receipts)==1 and len(observed)==1
# An actual second run through the same typed handle must not reach the stub.
replay_refused=False
try:
    router.run(**options)
except (ValueError,RuntimeError):
    replay_refused=True
assert replay_refused and len(observed)==1
(output/'joined.json').write_text(json.dumps(dict(receipt=receipts[0],provider_calls=len(observed),
    generated_prompt=observed[0],guidance=client.AdmittedCandidateRouterInput.guidance(handle),
    replay_refused=replay_refused,error_type=error_type,rollout_sha256=hashlib.sha256(rollout.read_bytes()).hexdigest())))
raise SystemExit(exit_code)
'''


@pytest.mark.parametrize("exit_code", [0, 7])
def test_native_launcher_typed_client_and_router_preserve_real_child_accounting(
        dispatch_case, tmp_path, exit_code):
    case, gate, broker, _spawn = dispatch_case
    output = tmp_path / "authored-router-observations"
    output.mkdir()
    # The fixture owner supplies these trusted literal roots. The -I child
    # discovers no package roots through its descriptor or candidate source.
    roots = [str(Path(__file__).resolve().parents[2])]
    for entry in sys.path:
        if entry and Path(entry).is_absolute() and Path(entry).is_dir():
            literal = str(Path(entry).resolve())
            if literal not in roots:
                roots.append(literal)
    environment = dict(PATH=os.defpath, LANG="C.UTF-8", IPFS_ACCEL_SKIP_CORE="1",
        PYTHONDONTWRITEBYTECODE="1", IPFS_KIT_DISABLE="1", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS="0",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", RAYON_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false", CODEX_HOME=str(output / "codex-home"))
    command = [sys.executable, "-I", "-c", ROUTER_CHILD, json.dumps(roots),
        str(broker.descriptor_path), broker.descriptor_sha256, str(broker.worker_uid),
        str(case.boundary), _row(case)["boundary_sha256"], case.worker_attempt.attempt_id,
        str(output), str(exit_code), json.dumps(POLICY)]
    cleaned = []
    def cleanup(process):
        assert process is gate.process and type(process) is subprocess.Popen
        assert process.returncode == exit_code
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
        cleaned.append(process)
        return True
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        completed = run_native_candidate_round_worker(admission=gate, source_broker=broker,
            command=command, environment=environment, cleanup=cleanup, stdout=stdout, stderr=stderr)
    assert completed.returncode == exit_code
    assert cleaned == [gate.process] and gate.process.pid != case.first_process.pid
    joined = json.loads((output / "joined.json").read_text())
    receipt = joined["receipt"]
    native_prompt = case.candidate_prompt.provider_prompt.decode()
    assert joined["provider_calls"] == 1 and joined["replay_refused"] is True
    assert joined["generated_prompt"].count(native_prompt) == 1
    assert joined["guidance"] in joined["generated_prompt"]
    assert receipt["owner_worker_attempt_id"] == case.worker_attempt.attempt_id
    assert receipt["native_prompt_sha256"] == hashlib.sha256(native_prompt.encode()).hexdigest()
    assert receipt["model_prompt_sha256"] == hashlib.sha256(joined["generated_prompt"].encode()).hexdigest()
    assert receipt["router_calls"] == 1 and receipt["completion_authority"] is False
    dispatch = receipt["candidate_round_dispatch"]
    assert dispatch["dispatch_consumed"] is True and dispatch["provider_attempted"] is True
    assert dispatch["host_native_admission_only"] is True
    assert dispatch["production_container_dispatch_qualified"] is False
    assert receipt["usage"]["exit_code"] == exit_code
    assert receipt["usage"]["prompt_tokens"] == 31 and receipt["usage"]["completion_tokens"] == 4
    usage = receipt["native_rollout_usage"]
    assert usage["rollout_sha256"] == joined["rollout_sha256"]
    assert usage["usage"] == dict(input_tokens=31,cached_input_tokens=7,output_tokens=4,total_tokens=35)
    assert usage["observed_token_count_records"] == 1 and usage["billing_total_verified"] is False
    assert usage["task_complete_observed"] is (exit_code == 0)
    assert receipt["status"] == ("provider_returned" if exit_code == 0 else "failed")
    assert joined["error_type"] == (None if exit_code == 0 else "RuntimeError")
    printed = [json.loads(line) for line in (output / "stdout.log").read_text().splitlines()
        if line.startswith('{') and '"router-implementation-invocation@1"' in line]
    assert len(printed) == 1 and printed[0] == receipt
    retired = json.loads(case.worker_attempt._state.retired_event.raw)
    assert retired["child_pid"] == gate.process.pid and retired["child_returncode"] == exit_code
    assert all(retired[key] is True for key in ("waited", "cleanup_succeeded", "brokers_closed"))
    assert retired["absolute_deadline_ns"] == json.loads(case.prior_worker_attempt._common)["absolute_deadline_ns"]
    assert case.native.source.get_task(case.native.task_cid).status == "in_progress"
