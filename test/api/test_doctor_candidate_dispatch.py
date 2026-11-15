"""Actual first/second router children retain Doctor under an owner attempt policy.

The host UID boundary and provider are doubles. Native Doctor, signed task,
candidate producers, child wait/registration, socket preflight and handoff bytes
run locally; no paid provider or benchmark task validation is invoked.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_context as gate
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_doctor_dispatch as doctor_gate
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source_client
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as candidate_client
from test.api.test_doctor_candidate_context import doctor_service, scenario  # noqa: F401
from test.api.test_doctor_residual_context import retained as retained_factory
from test.api.test_candidate_round_native_pipeline import _prepared
from test.api.test_candidate_dispatch_context import worker_seams


@pytest.fixture(params=["secret_analysis"])
def retained(scenario, tmp_path, request):
    return retained_factory.__wrapped__(scenario, tmp_path, request)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


CHILD = r'''
import hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
config=json.loads(sys.argv[1]);sys.path[:0]=config['imports']
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as candidate
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_reply_contract as reply
from ipfs_accelerate_py.agent_supervisor.runtime import codex_usage_receipt
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.llm_allocation import intelligence_index
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
prompt=sys.stdin.buffer.read().decode(); second=config['attempt']==2
if second and config.get('dispatch_clock_advance_ns'):
    from time import monotonic_ns as native_monotonic_ns
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_context as dispatch_gate
    dispatch_gate.time=SimpleNamespace(monotonic_ns=lambda:
        native_monotonic_ns()+config['dispatch_clock_advance_ns'])
protocol=candidate if second else source
protocol.transport.os=SimpleNamespace(**{name:getattr(os,name) for name in dir(os)
    if name not in {'getuid','geteuid','access'}},getuid=lambda:config['worker_uid'],
    geteuid=lambda:config['worker_uid'],access=lambda *args:False)
protocol.transport._protected_parent=lambda parent,owner:parent.stat()
guard._protected_directory=lambda *args,**kwargs:None
boundary.verify_container_worker_boundary=lambda **kwargs:dict(owner_uid=config['owner_uid'],
    worker_uid=config['worker_uid'],manifest_sha256=config['boundary_sha256'])
codex_usage_receipt.recover_codex_usage=lambda **kwargs:None
provider=config['settings']['provider'];model=config['settings']['model']
intelligence_index.discover_available_providers=lambda:[provider]
intelligence_index.select_efficient_route=lambda **kwargs:SimpleNamespace(provider=provider,model_name=model,catalog_revision='authored')
llm_router.get_llm_provider=lambda *args,**kwargs:object()
calls=[]; descriptor=Path(config['descriptor']); digest=config['descriptor_sha256']
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def generate(model_prompt,**kwargs):
    if second and prompt not in model_prompt:
        from ipfs_accelerate_py.agent_supervisor.runtime.candidate_prompt_projection import project_candidate_prompt, restore_candidate_prompt
        view=project_candidate_prompt(prompt)
        assert view.provider_prompt in model_prompt
        assert restore_candidate_prompt(view.provider_prompt,view.receipt)==prompt
    else:
        assert prompt in model_prompt if second else config['doctor_advisory'] in model_prompt
    assert kwargs['side_effecting'] is True
    assert kwargs['allow_cross_provider_fallback'] is kwargs['allow_local_fallback'] is False
    if provider=='grok_cli':
        assert kwargs['grok_max_turns']==config['settings']['grok_coding_max_turns']
    if second:
        wire=json.loads(prompt)
        assert wire['doctor_advisory']==config['doctor_advisory']
        assert wire['doctor_context']['schema']=='supervisor-doctor-candidate-context-binding@1'
        response=candidate.request_context_page(descriptor,digest,artifact='formal',
            expected_workspace_generation=wire['candidate_binding']['workspace_generation'],max_bytes=8192)
        assert response['ok'],response
    else:
        response=source.request_page(descriptor,digest,path='answer.py',view='original',max_bytes=8192)
        assert response['ok'],response
    observation=dict(exit_code=0,prompt_tokens=123,completion_tokens=7,cached_tokens=19)
    if provider=='codex_cli' and 'codex_output_schema' in kwargs:
        raw=canonical(kwargs['codex_output_schema'])
        observation.update(codex_output_schema_sha256=hashlib.sha256(raw).hexdigest(),codex_output_schema_bytes=len(raw))
    if provider=='grok_cli':
        from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
        observation.update(native_grok_observation(dict(stopReason='end_turn',num_turns=1,
            usage=dict(input_tokens=123,cache_read_input_tokens=19,output_tokens=7,total_tokens=149)),exit_code=0))
    set_last_cli_observation(provider,observation)
    calls.append(True)
    target=Path('answer.py');target.write_bytes(target.read_bytes()+b'\n# coding attempt '+str(config['attempt']).encode()+b'\n')
    if not second:raise RuntimeError('authored first provider failure after an edit')
    desc=candidate.validate_descriptor(json.loads(descriptor.read_bytes()))
    return json.dumps(dict(schema=reply.SCHEMA,status='candidate_ready',**{key:desc[key] for key in reply.BINDING_FIELDS}))
llm_router.generate_text=generate
kwargs=dict(prompt=prompt,timeout=config['timeout'],purpose='coding',semantic_repository=Path(config['repository']),
    container_boundary=Path(config['boundary']),container_boundary_sha256=config['boundary_sha256'],**config['settings'])
if second:
    kwargs.update(candidate_dispatch_handoff=Path(config['handoff']),candidate_dispatch_handoff_sha256=config['handoff_sha256'],
        candidate_round_descriptor=descriptor,candidate_round_descriptor_sha256=digest)
    if config.get('drift_after_preflight'):
        native=router._candidate_round_preflight
        def preflight(**kwargs):
            result=native(**kwargs)
            path=Path(config['drift_after_preflight']);path.write_bytes(path.read_bytes()+b'\nlate dependency drift\n')
            return result
        router._candidate_round_preflight=preflight
else:
    kwargs.update(source_view_profile=config['profile'],source_view_descriptor=descriptor,source_view_descriptor_sha256=digest,
        doctor_residual_artifact=Path(config['doctor_artifact']),doctor_residual_sha256=config['doctor_sha256'],
        doctor_residual_task_cid=config['doctor_task_cid'])
try:
    output,receipt=router.run(**kwargs)
except (RuntimeError,ValueError):
    if second and config.get('drift_after_preflight'):
        assert not calls
        print(json.dumps(dict(status='preflight_refused',model_calls=0)))
        raise SystemExit(9)
    if second or len(calls)!=1:raise
    print(json.dumps(dict(status='first_failed',model_calls=1)))
    raise SystemExit(7)
assert second and len(calls)==receipt['router_calls']==1
assert receipt['candidate_round_dispatch']['schema']=='supervisor-candidate-dispatch-observation@2'
assert receipt['candidate_round_dispatch']['doctor_candidate_dispatch']['proposal_budget']==config['proposal_budget']
assert receipt['candidate_reply_contract']['response_validated'] is True
assert receipt['doctor_residual_context'] is None and 'doctor_budget_capability' not in receipt
print(json.dumps(dict(status='second_returned',model_calls=1,receipt=receipt),sort_keys=True))
'''


def _peer_seam(monkeypatch, broker, protocol):
    native_identity, native_peer = protocol.process_identity, broker._peer
    def identity(pid):
        value = native_identity(pid)
        return value if pid == os.getpid() else dict(value, uid=broker.worker_uid)
    def peer(channel):
        class Channel:
            def getsockopt(self, level, option, size):
                pid, _, group = struct.unpack("3i", channel.getsockopt(level, option, size))
                return struct.pack("3i", pid, broker.worker_uid, group)
        return native_peer(Channel())
    monkeypatch.setattr(protocol, "process_identity", identity)
    monkeypatch.setattr(broker, "_peer", peer)


def _config(broker, launch, imports, attempt):
    doctor = launch._original_requirements._doctor_requirements
    return dict(imports=imports, attempt=attempt, settings=launch.launch_settings,
        timeout=max(1, int(launch.remaining_seconds)),
        descriptor=str(broker.descriptor_path), descriptor_sha256=broker.descriptor_sha256,
        repository=str(broker.repository), boundary=str(broker.boundary),
        boundary_sha256=broker.descriptor["boundary_manifest_sha256"],
        owner_uid=broker.owner_uid, worker_uid=broker.worker_uid,
        doctor_artifact=doctor._request[0], doctor_sha256=doctor._request[1], doctor_task_cid=doctor._request[2],
        doctor_advisory=doctor.advisory_bytes.decode(), proposal_budget=doctor.metadata["residual_budget"])


@pytest.fixture
def doctor_dispatch_case(doctor_service, monkeypatch, tmp_path, request):
    service, doctor_request = doctor_service
    selected = getattr(request, "param", "codex_cli")
    selected = selected if isinstance(selected, tuple) else (selected, 128)
    provider, turns = selected[:2]
    timeout = selected[2] if len(selected) > 2 else 180
    dispatch_clock_advance_ns = selected[3]*1_000_000_000 if len(selected) > 3 else 0
    extra = ["--doctor-residual-artifact", str(doctor_request["artifact"]),
        "--doctor-residual-sha256", doctor_request["expected_sha256"],
        "--doctor-residual-task-cid", doctor_request["task_cid"]]
    original = _prepared(service, extra)
    request.addfinalizer(original.close)
    argv = ["--provider", provider, "--model", "grok-4.7" if provider == "grok_cli" else "gpt-6.1-sol",
        "--timeout", str(timeout), "--max-output-tokens", "128", "--semantic-repository", str(original.repository),
        "--source-view-profile", original._binding["profile"], *extra]
    if turns != 128:
        argv.extend(("--grok-coding-max-turns", str(turns)))
    launch = owner.capture_original_coding_launch(source_broker=original, argv=argv,
        absolute_deadline=time.monotonic()+timeout-1, doctor_coding_policy="doctor-coding-two-attempts@1")
    imports = [str(Path(owner.__file__).resolve().parents[3])]
    imports += [str(Path(importlib.util.find_spec(name).origin).resolve().parents[1])
        for name in ("ipfs_datasets_py", "ipfs_kit_py")]
    _peer_seam(monkeypatch, original, source_client)
    config = _config(original, launch, imports, 1)
    config["profile"] = original._binding["profile"]
    launch.reserve_first_attempt([*owner._PREFIX, *argv, *original.extra_argv])
    first = subprocess.Popen([sys.executable, "-I", "-c", CHILD, json.dumps(config)],
        cwd=original.workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    original.note_spawned(first)
    stdout, stderr = first.communicate(original.prompt.encode(), timeout=launch.remaining_seconds)
    assert first.returncode == 7, stderr.decode()[-12000:]
    first_rows = [json.loads(line) for line in stdout.decode().splitlines()]
    first_receipt = next(row for row in first_rows if row.get("schema") == "router-implementation-invocation@1")
    assert first_receipt["router_calls"] == 1 and first_receipt["status"] == "failed"
    assert first_receipt["doctor_residual_context"]["schema"] == "doctor-residual-router-context@2"
    monkeypatch.setattr(owner, "_isolated_cleanup", lambda: subprocess.CompletedProcess(["authored-cleanup"], 0))
    retired = owner.retire_coding_worker(launch=launch, process=first, source_broker=original)
    prepared = owner.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    worker_seams(monkeypatch, prepared)
    broker = prepared.source_broker
    args = dict(prompt=prepared.handoff.provider_prompt.decode(), repository=broker.repository,
        workspace=broker.workspace, boundary=broker.boundary,
        boundary_sha256=broker.descriptor["boundary_manifest_sha256"], handoff=prepared.handoff.path,
        handoff_sha256=prepared.handoff.sha256, descriptor=broker.descriptor,
        descriptor_sha256=broker.descriptor_sha256, descriptor_path=broker.descriptor_path,
        launch_settings=launch.launch_settings)
    case = SimpleNamespace(original=original, launch=launch, retired=retired, prepared=prepared,
        request=args, first_receipt=first_receipt, imports=imports, tmp_path=tmp_path,
        dispatch_clock_advance_ns=dispatch_clock_advance_ns)
    try:
        yield case
    finally:
        child = prepared.source_broker._process
        if child is not None and child.poll() is None:
            child.kill(); child.wait(timeout=5)
        if child is not None and launch._state.second_retirement is None:
            owner.retire_coding_worker(launch=launch, process=child, source_broker=prepared.source_broker)
        prepared.close()
        prepared.handoff.close()


def test_doctor_handoff_preserves_exact_constraints_and_policy(doctor_dispatch_case):
    case = doctor_dispatch_case
    observed = gate.validate_candidate_dispatch_context(**case.request)
    projection = case.prepared.handoff.metadata["doctor"]
    assert observed["schema"] == gate.DOCTOR_OBSERVATION_SCHEMA
    assert observed["doctor_candidate_dispatch"] == doctor_gate.doctor_dispatch_observation(projection)
    assert doctor_gate.validate_doctor_dispatch_observation(observed["doctor_candidate_dispatch"])
    for malformed in ({1: 1}, {"max_calls": True}, {"max_calls": 1.0}, {}):
        receipt = copy.deepcopy(observed["doctor_candidate_dispatch"])
        receipt["proposal_budget"] = malformed
        with pytest.raises(ValueError):
            doctor_gate.validate_doctor_dispatch_observation(receipt)
    assert projection["coding_policy"] == case.launch.doctor_coding_policy
    assert projection["coding_policy"]["proposal_budget"] == dict(max_calls=1, max_tokens=4096, max_rounds=1, max_cost_units=1)
    wire = json.loads(case.request["prompt"])
    assert wire["doctor_advisory"].encode() == case.launch._original_requirements._doctor_requirements.advisory_bytes
    assert wire["doctor_context"] == dict(schema="supervisor-doctor-candidate-context-binding@1",
        sha256=sha(gate.canonical_json(projection["context"])), bytes=len(gate.canonical_json(projection["context"])))
    assert observed["doctor_candidate_dispatch"]["source_files"] > observed["source_files"]


def test_doctor_lifetime_cannot_exceed_original_deadline_descriptor_or_absolute_ceiling(
        doctor_dispatch_case, monkeypatch):
    case = doctor_dispatch_case
    path = case.prepared.handoff.path
    original = case.prepared.handoff.metadata_bytes
    metadata = case.prepared.handoff.metadata
    admitted_deadline = metadata["admission"]["deadline_monotonic_ns"]
    descriptor_deadline = case.prepared.source_broker.descriptor["expires_monotonic_ns"]
    assert metadata["expires_monotonic_ns"] == min(admitted_deadline, descriptor_deadline)
    assert gate.MAX_DOCTOR_LIFETIME_NS == 1800_000_000_000
    for change in ("original_expired", "descriptor_expired", "beyond_original", "beyond_descriptor",
                   "excessive_lifetime", "future_issuance", "missing_policy", "wrong_policy"):
        value, request = copy.deepcopy(metadata), dict(case.request)
        with monkeypatch.context() as patch:
            if change in ("original_expired", "descriptor_expired"):
                now = (admitted_deadline if change == "original_expired" else descriptor_deadline)+1
                patch.setattr(gate, "time", SimpleNamespace(monotonic_ns=lambda: now))
            elif change == "beyond_original": value["expires_monotonic_ns"] = admitted_deadline+1
            elif change == "beyond_descriptor": value["expires_monotonic_ns"] = descriptor_deadline+1
            elif change == "excessive_lifetime":
                value["issued_monotonic_ns"] = value["expires_monotonic_ns"]-gate.MAX_DOCTOR_LIFETIME_NS-1
            elif change == "future_issuance": value["issued_monotonic_ns"] = time.monotonic_ns()+30_000_000_000
            elif change == "missing_policy": value["doctor"].pop("coding_policy")
            else: value["doctor"]["coding_policy"]["deadline_monotonic_ns"] += 1
            try:
                raw = gate.canonical_json(value)
                path.chmod(0o640); path.write_bytes(raw); path.chmod(0o440)
                request["handoff_sha256"] = sha(raw)
                with pytest.raises(ValueError):
                    gate.validate_candidate_dispatch_context(**request)
            finally:
                path.chmod(0o640); path.write_bytes(original); path.chmod(0o440)
    assert gate.validate_candidate_dispatch_context(**case.request)["doctor_candidate_dispatch"]["original_obligations_preserved"]


def test_doctor_worker_refuses_rehashed_context_policy_and_dependency_changes(doctor_dispatch_case):
    case = doctor_dispatch_case
    original = case.prepared.handoff.metadata
    path = case.prepared.handoff.path
    def reject(value):
        raw = gate.canonical_json(value)
        path.chmod(0o640); path.write_bytes(raw); path.chmod(0o440)
        with pytest.raises((ValueError, OSError)):
            gate.validate_candidate_dispatch_context(**(case.request | dict(handoff_sha256=sha(raw))))
    try:
        for change in ("budget", "policy", "deadline", "advice", "context", "artifact", "mode"):
            value = copy.deepcopy(original)
            if change == "budget": value["doctor"]["context"]["residual_budget"]["max_calls"] = 2
            elif change == "policy": value["doctor"]["coding_policy"]["max_child_attempts"] = 3
            elif change == "deadline": value["doctor"]["coding_policy"]["deadline_monotonic_ns"] += 1
            elif change == "advice": value["doctor"]["advisory_sha256"] = "f"*64
            elif change == "context": value["doctor"]["context"]["original_obligations_preserved"] = False
            elif change == "artifact": value["doctor"]["artifact"]["sha256"] = "f"*64
            else: value["admission"]["doctor_advisory_bytes"] = float(value["admission"]["doctor_advisory_bytes"])
            reject(value)
    finally:
        path.chmod(0o640); path.write_bytes(gate.canonical_json(original)); path.chmod(0o440)
    outside = original["doctor"]["context"]["unchanged_inventory_paths"][0]
    target = case.original.workspace / outside
    raw, mode = target.read_bytes(), target.stat().st_mode & 0o777
    for change in ("body", "executable"):
        try:
            if change == "body": target.write_bytes(raw+b"\nDoctor dependency drift\n")
            else: target.chmod(mode ^ 0o100)
            with pytest.raises(ValueError, match="complete Doctor source inventory"):
                gate.validate_candidate_dispatch_context(**case.request)
        finally:
            target.write_bytes(raw); target.chmod(mode)
    assert gate.validate_candidate_dispatch_context(**case.request)["doctor_candidate_dispatch"]["original_obligations_preserved"]


@pytest.mark.parametrize("doctor_dispatch_case", ["codex_cli", "grok_cli", ("grok_cli", 8),
    pytest.param(("grok_cli", 8, 600, 95), id="grok8-late-worker")], indirect=True)
@pytest.mark.parametrize("drift", [False, True])
def test_two_native_router_children_and_late_doctor_preflight(doctor_dispatch_case, monkeypatch, drift):
    case = doctor_dispatch_case
    prepared, broker = case.prepared, case.prepared.source_broker
    _peer_seam(monkeypatch, broker, candidate_client)
    output_path, error_path = case.tmp_path / "second-output.jsonl", case.tmp_path / "second-error.txt"
    def spawn(command, *, stdin, workspace):
        assert command[:6] == list(owner._PREFIX)
        assert "--doctor-residual-artifact" not in command and "--doctor-coding-policy" not in command
        assert command[command.index("--grok-coding-max-turns")+1] == str(
            case.launch.launch_settings["grok_coding_max_turns"])
        config = _config(broker, case.launch, case.imports, 2)
        config.update(handoff=str(prepared.handoff.path), handoff_sha256=prepared.handoff.sha256,
            timeout=int(command[command.index("--timeout")+1]),
            dispatch_clock_advance_ns=case.dispatch_clock_advance_ns)
        if case.dispatch_clock_advance_ns:
            value = prepared.handoff.metadata
            advanced = time.monotonic_ns()+case.dispatch_clock_advance_ns
            assert value["issued_monotonic_ns"]+90_000_000_000 < advanced
            assert advanced < min(broker.descriptor["expires_monotonic_ns"],
                value["admission"]["deadline_monotonic_ns"])
        if drift:
            config["drift_after_preflight"] = prepared.handoff.metadata["doctor"]["context"]["unchanged_inventory_paths"][0]
        with output_path.open("wb") as out, error_path.open("wb") as err:
            return subprocess.Popen([sys.executable, "-I", "-c", CHILD, json.dumps(config)],
                stdin=stdin, cwd=workspace, stdout=out, stderr=err)
    monkeypatch.setattr(owner, "_spawn_worker", spawn)
    running = prepared.start()
    code = running.process.wait(timeout=case.launch.remaining_seconds)
    assert code == (9 if drift else 0), error_path.read_text()[-12000:]
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    if case.launch.launch_settings["provider"] == "grok_cli":
        turns = case.launch.launch_settings["grok_coding_max_turns"]
        assert case.first_receipt["provider_invocation_policy"]["max_turns"] == turns
        assert receipt["provider_invocation_policy"]["max_turns"] == turns
    assert receipt["router_calls"] == (0 if drift else 1)
    assert receipt["candidate_round_preflight"]["requests"] == 1
    assert receipt["candidate_round_preflight"]["provider_input_bytes"] == 0
    assert receipt["candidate_round_dispatch"]["doctor_candidate_dispatch"]["proposal_budget"] == case.launch.doctor_coding_policy["proposal_budget"]
    assert prepared.handoff.metadata["expires_monotonic_ns"] <= min(
        broker.descriptor["expires_monotonic_ns"], case.launch.doctor_coding_policy["deadline_monotonic_ns"])
    if drift:
        assert receipt["failure_phase"] == "candidate_dispatch_context"
    else:
        assert receipt["candidate_reply_contract"]["response_validated"] is True
        assert case.first_receipt["router_calls"] + receipt["router_calls"] == 2
    owner.retire_coding_worker(launch=case.launch, process=running.process, source_broker=broker)
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_lifecycle
    from benchmarks.agent_supervisor.container_coding import terminal_candidate_attempt_accounting
    lifecycle = dict(schema="owner-candidate-rounds-observation@1", max_rounds=2,
        elapsed_seconds=1.0, outcome="workers_returned", failure_type=None,
        completion_authority=False, proof_authority=False, source_obligations_discharged=0,
        accepted_index_published=False, rounds=[dict(round_number=number,
            process_started=True, session_id=item.descriptor["session_id"],
            descriptor_sha256=item.descriptor_sha256, returncode=item._process.returncode,
            wait_completed=True, retirement_observed=True)
            for number, item in enumerate((case.original, broker), 1)])
    candidate_round_lifecycle.persist(source_broker=case.original, observation=lifecycle)
    report = dict(candidate_rounds=2, doctor_coding_policy="doctor-coding-two-attempts@1",
        arm="full", worker_cleanup_returncode=0,
        source_view_profile=case.original.descriptor["profile"],
        provider_invocations=[dict(case.first_receipt, phase="coding"), dict(receipt, phase="coding")])
    accounting = terminal_candidate_attempt_accounting.collect(parent=broker.audit_path.parent, report=report)
    assert accounting["status"] == "matched_round_receipts", accounting
    assert accounting["doctor_attempts_reserved"] == [1, 2]
    assert accounting["doctor_coding_policy_sha256"] == sha(gate.canonical_json(case.launch.doctor_coding_policy))
    from benchmarks.agent_supervisor.container_coding import terminal_candidate_round_observation as observation
    # Keep the same authored host directory admission as the native worker
    # fixture; all socket records, closed audits and router receipts are real.
    monkeypatch.setattr(observation, "_protected_directory", lambda *args, **kwargs: None)
    audit = json.loads(broker.audit_path.read_bytes())
    assert audit["accounting_complete"]
    observed = observation.collect(parent=broker.audit_path.parent,
        scope=dict(task_alias=broker.descriptor["task_id"]), report=report)
    assert observed["status"] == "observed", observed
    assert observed["sessions"][0]["prior_router_receipt_matched"] is True
    assert observed["counts"]["accepted_requests"] == audit["accepted_requests"]
    assert observed["counts"]["response_bytes"] == audit["response_bytes"]
    assert observed["preflight_counts"]["requests"] == 1
    assert observed["preflight_counts"]["response_bytes"] == receipt["candidate_round_preflight"]["response_bytes"]
    assert observed["provider_tokens"] is None and observed["model_input_replayed"] is False
    report["provider_invocations"].pop()
    incomplete = terminal_candidate_attempt_accounting.collect(parent=broker.audit_path.parent, report=report)
    assert incomplete["status"] == "incomplete" and incomplete["potential_unreceipted_attempt"]
    with pytest.raises(ValueError, match="consumed"):
        prepared.start()
    with pytest.raises(ValueError, match="consumed"):
        owner.prepare_second_coding_worker(launch=case.launch, retired_worker=case.retired)
