"""V2 native dispatch preserves candidate-only reads after consumption.

The native owner/claim/source and actual child/kernel birth remain real. The
single-UID fixture authors only deployment UID separation. No provider or
target module executes, and this does not qualify container deployment.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_client as client
from ipfs_accelerate_py.agent_supervisor.runtime.candidate_dispatch_broker import CandidateDispatchBroker
from test.api.test_candidate_dispatch import POLICY, _uid_seam
from test.api.test_native_candidate_round_admission import candidate_case, capture  # noqa: F401
from test.api.test_owner_candidate_round_context import CHANGED


READ_CHILD = r'''
import json,os,signal,sys,uuid
from pathlib import Path
from types import SimpleNamespace
signal.alarm(25)
sys.path.insert(0,sys.argv[1])
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as old_client
transport=client.transport
worker_uid=int(sys.argv[4])
transport.os=SimpleNamespace(**{name:getattr(os,name) for name in dir(os)
    if name not in {'getuid','geteuid','access'}},getuid=lambda:worker_uid,
    geteuid=lambda:worker_uid,access=lambda *args:False)
transport._protected_parent=lambda parent,owner:parent.stat()
prompt=sys.stdin.buffer.read().decode('utf-8')
descriptor_path,descriptor_sha=Path(sys.argv[2]),sys.argv[3]
descriptor=client.load_descriptor(descriptor_path,descriptor_sha,workspace=Path.cwd())
v1_refused=False
try:
    old_client.validate_descriptor(descriptor)
except (ValueError,TypeError,KeyError):
    v1_refused=True
assert v1_refused
handle=client.prepare_candidate_router_input(descriptor_path=descriptor_path,
    descriptor_sha256=descriptor_sha,prompt=prompt,workspace=Path.cwd())
checked=handle.validate_for_run(prompt=prompt,workspace=Path.cwd(),
    boundary_sha256=descriptor['boundary_manifest_sha256'],
    worker_attempt_id=descriptor['worker_attempt_id'],**json.loads(sys.argv[5]))
consumed=handle.authorize_dispatch(invocation_id=uuid.uuid4().hex)
assert consumed['dispatch_consumed'] is True
# Both exchanges occur after dispatch consumption, through actual public V2
# client framing and unchanged native generation/source/peer checks.
source=client.request_page(descriptor_path,descriptor_sha,path='module.py',view='original',
    expected_workspace_generation=descriptor['workspace_generation'],max_bytes=8192,
    workspace=Path.cwd())
summary=client.request_context_page(descriptor_path,descriptor_sha,artifact='summary',
    expected_workspace_generation=descriptor['workspace_generation'],max_bytes=8192,
    workspace=Path.cwd())
print(json.dumps(dict(v1_refused=v1_refused,checked=checked,consumed=consumed,
    source=source,summary=summary)),flush=True)
'''


def _body(page):
    return (page["content"].encode("utf-8") if page["encoding"] == "utf-8"
        else base64.b64decode(page["content"], validate=True))


def test_registered_native_child_keeps_source_and_summary_reads_after_dispatch(
        candidate_case, monkeypatch):
    case = candidate_case
    gate = capture(case)
    broker = CandidateDispatchBroker(admission=gate, **POLICY)
    _uid_seam(broker, monkeypatch)
    child = None
    try:
        def birth():
            nonlocal child
            child = subprocess.Popen([sys.executable, "-I", "-c", READ_CHILD,
                str(Path(__file__).resolve().parents[2]), str(broker.descriptor_path),
                broker.descriptor_sha256, str(broker.worker_uid), json.dumps(POLICY)],
                cwd=case.native.workspace, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            case.children.append(child)
            return child
        observed = gate.execute_spawn(birth, broker.note_spawned)
        assert observed is child
        stdout, stderr = child.communicate(case.candidate_prompt.provider_prompt, timeout=30)
        assert child.returncode == 0, stderr.decode("utf-8", errors="replace")
        result = json.loads(stdout)
        assert result["v1_refused"] is True
        assert result["checked"]["dispatch_consumed"] is False
        assert result["consumed"]["dispatch_consumed"] is True
        source, summary = result["source"], result["summary"]
        assert source["ok"] is True and summary["ok"] is True
        page = source["page"]
        generation = case.snapshot.metadata["workspace_generation"]
        assert _body(page) == CHANGED == (case.native.workspace / "module.py").read_bytes()
        assert page["source_sha256"] == hashlib.sha256(CHANGED).hexdigest()
        assert page["workspace_generation"] == generation and page["source_state"] == "working-candidate"
        assert page["baseline_evidence_applicable"] is False
        context = summary["page"]
        assert context["artifact"] == "summary" and context["workspace_generation"] == generation
        assert context["artifact_sha256"] == hashlib.sha256(case.snapshot.artifacts["summary"]).hexdigest()
        assert context["baseline_evidence_applicable"] is False
        assert context["proof_authority"] is False and context["completion_authority"] is False
        assert _body(context)
        assert broker._dispatch_state["spent"] is True
        assert case.native.source.get_task(case.native.task_cid).status == "in_progress"
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()
        broker.close()


def test_advertised_candidate_client_runs_standalone_isolated_help():
    result = subprocess.run([sys.executable, "-I", str(Path(client.__file__)), "--help"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=10, check=True, text=True)
    assert "--expected-workspace-generation" in result.stdout
    assert "--candidate-dispatch-descriptor" in result.stdout
    assert "--candidate-dispatch-descriptor-sha256" in result.stdout
    assert "--path" in result.stdout and "--artifact" in result.stdout
