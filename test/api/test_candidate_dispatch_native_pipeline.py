"""Typed owner admission -> real isolated Python child -> native router.

This local fixture authors cross-UID/path isolation and replaces provider calls.
Git edits, sealed stdin, child custody, descriptor/socket transport, protected
handoff validation, native formal artifacts and router response checks are real.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as client
from test.api.test_candidate_round_native_pipeline import _prepared
from test.api.test_owner_candidate_round_context import source_service, native, SOURCE, CHANGED  # noqa: F401
from test.api.test_owner_candidate_round_prompt import public_service, selected, scenario  # noqa: F401


CHILD = r'''
import base64,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
config=json.loads(sys.argv[1]);sys.path[:0]=config['imports']
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_reply_contract as reply
from ipfs_accelerate_py.agent_supervisor.runtime import codex_usage_receipt
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.llm_allocation import intelligence_index
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
prompt=sys.stdin.buffer.read().decode()
wire=json.loads(prompt)
descriptor=Path(config['descriptor']);digest=config['descriptor_sha256']
generation=wire['candidate_binding']['workspace_generation']
client.transport.os=SimpleNamespace(**{name:getattr(os,name) for name in dir(os)
    if name not in {'getuid','geteuid','access'}},getuid=lambda:config['worker_uid'],
    geteuid=lambda:config['worker_uid'],access=lambda *args:False)
client.transport._protected_parent=lambda parent,owner:parent.stat()
guard._protected_directory=lambda *args,**kwargs:None
boundary.verify_container_worker_boundary=lambda **kwargs:dict(owner_uid=config['owner_uid'],worker_uid=config['worker_uid'])
codex_usage_receipt.recover_codex_usage=lambda **kwargs:None
provider=config['settings']['provider'];model=config['settings']['model']
intelligence_index.discover_available_providers=lambda:[provider]
intelligence_index.select_efficient_route=lambda **kwargs:SimpleNamespace(provider=provider,model_name=model,catalog_revision='authored')
llm_router.get_llm_provider=lambda *args,**kwargs:object()
calls=[]
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def generate(model_prompt,**kwargs):
    assert prompt in model_prompt and kwargs['side_effecting'] is True
    assert kwargs['allow_cross_provider_fallback'] is kwargs['allow_local_fallback'] is False
    raw=b'';cursor=None
    for _ in range(32):
        response=client.request_context_page(descriptor,digest,expected_workspace_generation=generation,
            artifact='formal',cursor=cursor,max_bytes=client.MAX_FRAME_BYTES)
        assert response['ok'],response
        page=response['page'];assert page['offset']==len(raw)
        raw+=page['content'].encode() if page['encoding']=='utf-8' else base64.b64decode(page['content'])
        if page['exhausted']:break
        cursor=page['next_cursor']
    else:raise AssertionError('native artifact pagination did not terminate')
    formal=json.loads(raw)['reports'][config['target']]['projections']['lean']
    definitions={row['definition'] for row in formal['bindings']}
    assert 'SelectionMin' in definitions and 'SelectionMax' not in definitions
    assert formal['status']=='prepared_not_built'
    path=Path(config['target']);path.write_bytes(path.read_bytes()+b'\n# second coding round\n')
    observation=dict(exit_code=0,prompt_tokens=123,completion_tokens=7,cached_tokens=19)
    if provider=='codex_cli':
        schema=canonical(kwargs['codex_output_schema'])
        observation.update(codex_output_schema_sha256=hashlib.sha256(schema).hexdigest(),codex_output_schema_bytes=len(schema))
    else:
        from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
        observation.update(native_grok_observation(dict(stopReason='end_turn',num_turns=1,
            usage=dict(input_tokens=123,cache_read_input_tokens=19,output_tokens=7,total_tokens=149)),exit_code=0))
    set_last_cli_observation(provider,observation)
    calls.append(dict(input_sha256=hashlib.sha256(model_prompt.encode()).hexdigest(),timeout=kwargs['timeout']))
    desc=client.validate_descriptor(json.loads(descriptor.read_bytes()))
    return json.dumps(dict(schema=reply.SCHEMA,status='candidate_ready',**{key:desc[key] for key in reply.BINDING_FIELDS}))
llm_router.generate_text=generate
output,receipt=router.run(prompt=prompt,timeout=config['timeout'],purpose='coding',
    semantic_repository=Path(config['repository']),container_boundary=Path(config['boundary']),
    container_boundary_sha256=config['boundary_sha256'],candidate_dispatch_handoff=Path(config['handoff']),
    candidate_dispatch_handoff_sha256=config['handoff_sha256'],candidate_round_descriptor=descriptor,
    candidate_round_descriptor_sha256=digest,**config['settings'])
assert len(calls)==1 and receipt['router_calls']==1 and receipt['status']=='provider_returned'
assert receipt['candidate_reply_contract']['response_validated'] is True
assert receipt['completion_authority'] is False and receipt['semantic_translation'] is None
assert receipt['model_prompt_sha256']==calls[0]['input_sha256']
assert receipt['usage']['prompt_tokens']==123 and receipt['usage']['completion_tokens']==7
assert receipt['candidate_round_dispatch']['workspace_generation']==generation
assert client.DESCRIPTOR_ENV not in os.environ and client.DIGEST_ENV not in os.environ
print(json.dumps(dict(status='passed',output=json.loads(output),model_calls=len(calls),
    provider=provider,source_changed=True,formalization_status='prepared_not_built',
    receipt=receipt),sort_keys=True))
'''


@pytest.mark.parametrize("provider,model,use_public", [("codex_cli", "gpt-6.1-sol", False), ("grok_cli", "grok-4.7", True)])
def test_owner_to_native_candidate_router_with_real_child_and_artifact_retrieval(request, monkeypatch, tmp_path, provider, model, use_public):
    extra = []
    if use_public:
        service, public_args, _ = request.getfixturevalue("public_service")
        extra = ["--public-instruction-artifact", str(public_args["artifact"]),
            "--public-instruction-sha256", public_args["expected_sha256"],
            "--public-instruction-task-cid", public_args["task_cid"]]
        target = "answer.py"
    else:
        service, target = request.getfixturevalue("source_service"), "mod.py"
    baseline = (service.repository / target).read_bytes()
    original = _prepared(service, extra)
    launch = owner.capture_original_coding_launch(source_broker=original,
        argv=["--provider", provider, "--model", model, "--timeout", "90", "--max-output-tokens", "128",
              "--semantic-repository", str(original.repository), "--source-view-profile", "python-comments@1", *extra],
        absolute_deadline=time.monotonic()+89)
    first = subprocess.Popen([sys.executable, "-I", "-c",
        "import sys;from pathlib import Path;sys.stdin.read();Path(sys.argv[1]).write_bytes(bytes.fromhex(sys.argv[2]));sys.exit(7)",
        str(original.workspace / target), CHANGED.hex()], stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    original.note_spawned(first)
    first.communicate(b"authored initial failure", timeout=10)
    assert first.returncode == 7
    cleanup_events = []
    native_run = subprocess.run
    def cleanup():
        cleanup_events.append("cleanup")
        return native_run([sys.executable, "-I", "-c", "pass"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    monkeypatch.setattr(owner, "_isolated_cleanup", cleanup)
    retired = owner.retire_coding_worker(launch=launch, process=first, source_broker=original)
    prepared = owner.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    broker = prepared.source_broker
    native_identity, native_peer = client.process_identity, broker._peer
    def identity(pid):
        value = native_identity(pid)
        return value if pid == os.getpid() else dict(value, uid=broker.worker_uid)
    def peer(channel):
        class Channel:
            def getsockopt(self, level, option, size):
                pid, _, group = struct.unpack("3i", channel.getsockopt(level, option, size))
                return struct.pack("3i", pid, broker.worker_uid, group)
        return native_peer(Channel())
    monkeypatch.setattr(client, "process_identity", identity)
    monkeypatch.setattr(broker, "_peer", peer)
    imports = [str(Path(owner.__file__).resolve().parents[3])]
    imports += [str(Path(importlib.util.find_spec(name).origin).resolve().parents[1])
        for name in ("ipfs_datasets_py", "ipfs_kit_py")]
    output_path, error_path = tmp_path / "child-output.json", tmp_path / "child-error.txt"
    def spawn(command, *, stdin, workspace):
        assert command[:6] == list(owner._PREFIX)
        assert "--candidate-dispatch-handoff" in command and "--source-view-descriptor" not in command
        assert "--public-instruction-artifact" not in command and "--source-formalization-handoff" not in command
        timeout = int(command[command.index("--timeout")+1])
        config = dict(imports=imports, settings=launch.launch_settings, timeout=timeout, target=target,
            descriptor=str(broker.descriptor_path), descriptor_sha256=broker.descriptor_sha256,
            handoff=str(prepared.handoff.path), handoff_sha256=prepared.handoff.sha256,
            repository=str(broker.repository), boundary=str(broker.boundary),
            boundary_sha256=broker.descriptor["boundary_manifest_sha256"],
            owner_uid=broker.owner_uid, worker_uid=broker.worker_uid)
        with output_path.open("wb") as out, error_path.open("wb") as err:
            return subprocess.Popen([sys.executable, "-I", "-c", CHILD, json.dumps(config)],
                stdin=stdin, cwd=workspace, stdout=out, stderr=err)
    monkeypatch.setattr(owner, "_spawn_worker", spawn)
    running = None
    try:
        running = prepared.start()
        assert running.process.pid != first.pid
        code = running.process.wait(timeout=launch.remaining_seconds)
        assert code == 0, error_path.read_text()[-12000:]
        records = [json.loads(line) for line in output_path.read_text().splitlines()]
        results = [row for row in records if row.get("status") == "passed"]
        receipts = [row for row in records if row.get("schema") == "router-implementation-invocation@1"]
        assert len(results) == len(receipts) == 1
        result = results[0]
        assert receipts[0] == result["receipt"]
        assert result["status"] == "passed" and result["model_calls"] == 1
        assert result["receipt"]["candidate_round_dispatch"]["handoff_sha256"] == prepared.handoff.sha256
        assert result["receipt"]["candidate_round_preflight"]["provider_input_bytes"] == 0
        assert result["receipt"]["native_prompt_sha256"] == hashlib.sha256(original.prompt.encode()).hexdigest()
        assert result["receipt"]["prompt_sha256"] == hashlib.sha256(prepared.prompt.provider_prompt).hexdigest()
        assert (original.repository / target).read_bytes() == baseline
        assert (original.workspace / target).read_bytes() == CHANGED+b"\n# second coding round\n"
        assert bool(json.loads(prepared.prompt.provider_prompt)["original_public_instruction_block"]) is use_public
        with pytest.raises(ValueError):
            prepared.snapshot.revalidate()
        with pytest.raises(ValueError, match="consumed"):
            prepared.start()
        with pytest.raises(ValueError, match="consumed"):
            owner.prepare_second_coding_worker(launch=launch, retired_worker=retired)
        owner.retire_coding_worker(launch=launch, process=running.process, source_broker=broker)
        assert len(cleanup_events) == 2
        audit = json.loads(broker.audit_path.read_bytes())
        assert audit["accounting_complete"] and audit["accepted_requests"] >= 2
        assert all(row["delivered"] for row in audit["requests"])
    finally:
        if running is not None and running.process.poll() is None:
            running.process.kill(); running.process.wait()
            owner.retire_coding_worker(launch=launch, process=running.process, source_broker=broker)
        prepared.close()
        prepared.handoff.close()
        original.close()
