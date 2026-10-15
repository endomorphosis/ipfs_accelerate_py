"""Signed public checks can continue an actually successful, retired worker.

Both router children, public/Doctor capture, candidate producers and socket
preflight are real. Provider and host privilege/validation execution boundaries
are authored doubles; no model or hidden benchmark verifier is called.
"""
import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_public_validation as public
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_context as gate
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source_client
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as candidate_client
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
from test.api.public_validation_fixture import public_doctor_service
from test.api.test_doctor_candidate_dispatch import CHILD, _peer_seam, _config
from test.api.test_candidate_dispatch_context import worker_seams
from benchmarks.agent_supervisor.container_coding.test_terminal_public_acceptance_profile import measured_result


PUBLIC_CHILD = CHILD.replace("answer.py", "eigen.py").replace(
    "if not second:raise RuntimeError('authored first provider failure after an edit')",
    "if not second:return 'completed'").replace(
    "kwargs.update(source_view_profile=config['profile']",
    "kwargs.update(public_instruction_artifact=Path(config['public_artifact']),"
    "public_instruction_sha256=config['public_sha256'],public_instruction_task_cid=config['doctor_task_cid'],"
    "source_view_profile=config['profile']").replace(
    "assert second and len(calls)==receipt['router_calls']==1",
    "if not second:\n"
    "    assert len(calls)==receipt['router_calls']==1\n"
    "    print(json.dumps(dict(status='first_returned',model_calls=1,receipt=receipt)))\n"
    "    raise SystemExit(0)\n"
    "assert second and len(calls)==receipt['router_calls']==1")


def _runner_seams(monkeypatch, broker):
    runner = dict(schema=execution.SCHEMA, argv=[str(execution.ROOT / 'bin/validation-worker')],
        files={str(execution.ROOT / name): ('b' if name == 'container-boundary.json' else 'a')*64
            for name in ('bin/validation-worker', 'bin/worker-entry', 'container-boundary.json')},
        owner_uid=os.getuid(), worker_uid=1001, namespaces={key:'authored' for key in ('pid','mnt','net')},
        completion_authority=False)
    monkeypatch.setattr(execution, 'candidate_runner_from_environment', lambda: copy.deepcopy(runner))
    def verify(value):
        assert value == runner
        return value
    monkeypatch.setattr(execution, 'verify_candidate_runner', verify)
    monkeypatch.setattr(owner, '_isolated_cleanup', lambda: subprocess.run(
        [sys.executable, '-I', '-c', 'pass'], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, timeout=5))


@pytest.fixture
def successful_first(tmp_path, monkeypatch):
    with public_doctor_service(tmp_path, monkeypatch) as native:
        broker = native.original
        _runner_seams(monkeypatch, broker)
        argv = ['--provider','grok_cli','--model','grok-4.7','--timeout','600',
            '--max-output-tokens','128','--grok-coding-max-turns','8',
            '--semantic-repository',str(broker.repository),'--source-view-profile',broker._binding['profile'],
            *native.extra_argv]
        launch = owner.capture_original_coding_launch(source_broker=broker, argv=argv,
            absolute_deadline=time.monotonic()+599, doctor_coding_policy=policy.PUBLIC_VALIDATION_POLICY)
        imports = [str(Path(owner.__file__).resolve().parents[3])]
        imports += [str(Path(importlib.util.find_spec(name).origin).resolve().parents[1])
            for name in ('ipfs_datasets_py','ipfs_kit_py')]
        _peer_seam(monkeypatch, broker, source_client)
        config = _config(broker, launch, imports, 1)
        config.update(profile=broker._binding['profile'], public_artifact=str(native.public_request['artifact']),
            public_sha256=native.public_request['expected_sha256'])
        launch.reserve_first_attempt([*owner._PREFIX,*argv,*broker.extra_argv])
        process = subprocess.Popen([sys.executable,'-I','-c',PUBLIC_CHILD,json.dumps(config)],
            cwd=broker.workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        broker.note_spawned(process)
        out, err = process.communicate(broker.prompt.encode(), timeout=launch.remaining_seconds)
        assert process.returncode == 0, err.decode()[-12000:]
        receipts = [json.loads(line) for line in out.decode().splitlines()]
        first_receipt = next(row for row in receipts if row.get('schema') == 'router-implementation-invocation@1')
        assert first_receipt['router_calls'] == 1 and first_receipt['status'] == 'provider_returned'
        retired = owner.retire_coding_worker(launch=launch, process=process, source_broker=broker)
        calls = []
        def check(binding, argv, *, cwd, timeout, **_):
            assert binding == launch.public_validation_contract.metadata['runner'] and cwd == broker.workspace
            assert 0 < timeout <= launch.remaining_seconds+0.1
            selected = argv[-1]
            assert argv in [row['argv'] for row in launch.public_validation_contract.metadata['task_spec']['validations']]
            second = launch._state.second_retirement is not None
            result = measured_result(candidate=1e-6 if second else 3e-6, reference=2e-6) if selected == 'performance' else dict(
                schema=profiles.PUBLIC_ACCEPTANCE_RESULT_SCHEMA, profile=profiles.PUBLIC_VALIDATION_PROFILE,
                check=selected, status='passed', reason='obligations_satisfied', measurements=[])
            code = {'passed':0,'failed':1,'inconclusive':2}[result['status']]
            calls.append((2 if second else 1, selected))
            source = 'import sys;print('+repr(json.dumps(result))+');raise SystemExit('+str(code)+')'
            return subprocess.run([sys.executable,'-I','-c',source], cwd=cwd,
                stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
        monkeypatch.setattr(execution, 'run_candidate_bounded', check)
        yield SimpleNamespace(native=native, launch=launch, retired=retired, calls=calls,
            imports=imports, first_receipt=first_receipt, tmp_path=tmp_path)


def test_real_exit_zero_public_failure_admits_one_separate_router_child(successful_first, monkeypatch):
    case = successful_first
    launch, original = case.launch, case.native.original
    first = public.validate_retired_candidate(launch=launch, retired_worker=case.retired)
    launch._state.public_validations[1] = first
    assert first.metadata['status'] == 'failed' and case.retired.returncode == 0
    prepared = owner.prepare_second_coding_worker(launch=launch, retired_worker=case.retired, public_validation=first)
    try:
        worker_seams(monkeypatch, prepared)
        broker = prepared.source_broker
        _peer_seam(monkeypatch, broker, candidate_client)
        wire = json.loads(prepared.prompt.provider_prompt)
        assert wire['schema'] == 'supervisor-owner-candidate-router-input@3'
        assert wire['prior_failure']['prior_returncode'] == 0
        assert wire['public_validation']['feedback']['failed_obligations'] == [profiles.PUBLIC_VALIDATION_KEYS[-1]]
        observation_path, error_path = case.tmp_path/'second.jsonl', case.tmp_path/'second-error.txt'
        def spawn(command, *, stdin, workspace):
            assert '--doctor-coding-policy' not in command and '--doctor-residual-artifact' not in command
            config = _config(broker, launch, case.imports, 2)
            config.update(handoff=str(prepared.handoff.path), handoff_sha256=prepared.handoff.sha256,
                timeout=int(command[command.index('--timeout')+1]))
            with observation_path.open('wb') as out, error_path.open('wb') as err:
                return subprocess.Popen([sys.executable,'-I','-c',PUBLIC_CHILD,json.dumps(config)],
                    stdin=stdin,cwd=workspace,stdout=out,stderr=err)
        monkeypatch.setattr(owner,'_spawn_worker',spawn)
        running = prepared.start()
        assert running.process.wait(timeout=launch.remaining_seconds) == 0, error_path.read_text()[-12000:]
        second = owner.retire_coding_worker(launch=launch,process=running.process,source_broker=broker)
        rows = [json.loads(line) for line in observation_path.read_text().splitlines()]
        receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
        assert receipt['router_calls'] == 1 and receipt['candidate_round_preflight']['requests'] == 1
        nested = receipt['candidate_round_dispatch']['doctor_candidate_dispatch']
        assert nested['policy_id'] == policy.PUBLIC_VALIDATION_POLICY
        assert nested['actual_prior_returncode'] == 0 and nested['public_validation_status'] == 'failed'
        final = public.validate_retired_candidate(launch=launch,retired_worker=second)
        launch._state.public_validations[2] = final
        assert final.metadata['status'] == 'passed' and second.returncode == 0
        assert launch._state.reserved_attempts == {1,2}
        assert case.calls == [(number,check) for number in (1,2) for check in profiles.PUBLIC_CHECKS]
        with pytest.raises(ValueError):
            owner.prepare_second_coding_worker(launch=launch,retired_worker=second,public_validation=final)
        from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_lifecycle as lifecycle
        observation = dict(schema='owner-candidate-rounds-observation@2', max_rounds=2,elapsed_seconds=1,
            outcome='workers_returned',failure_type=None,completion_authority=False,proof_authority=False,
            source_obligations_discharged=0,accepted_index_published=False,
            public_validation_observations=[first.metadata,final.metadata],owner_validation_outcome='passed',owner_returncode=0,
            rounds=[dict(round_number=n,process_started=True,session_id=b.descriptor['session_id'],
                descriptor_sha256=b.descriptor_sha256,returncode=0,wait_completed=True,retirement_observed=True)
                for n,b in enumerate((original,broker),1)])
        retained = lifecycle.persist(source_broker=original,observation=observation)
        payload = lifecycle.parse_retention((original.descriptor_path.parent/retained['path']).read_bytes())
        assert payload['schema'] == lifecycle.PUBLIC_SCHEMA
        assert [row['returncode'] for row in payload['observation']['rounds']] == [0,0]
        from benchmarks.agent_supervisor.container_coding import terminal_candidate_attempt_accounting as accounting
        report = dict(candidate_rounds=2,doctor_coding_policy=policy.PUBLIC_VALIDATION_POLICY,
            provider_invocations=[dict(case.first_receipt,phase='coding'),dict(receipt,phase='coding')])
        result = accounting.collect(parent=original.descriptor_path.parent,report=report)
        assert result['status'] == 'matched_round_receipts', result
        assert result['owner_validation_outcome'] == 'passed' and result['owner_returncode'] == 0
        for key, changed in [('public_validation_contract_sha256','f'*64),
            ('public_validation_observation_sha256','e'*64),('actual_prior_returncode',7),
            ('public_validation_status','passed')]:
            forged = copy.deepcopy(report)
            forged['provider_invocations'][1]['candidate_round_dispatch']['doctor_candidate_dispatch'][key] = changed
            assert accounting.collect(parent=original.descriptor_path.parent,report=forged)['status'] == 'unavailable'
        from benchmarks.agent_supervisor.container_coding import terminal_candidate_input_audit as audit
        _, _, checks = audit.replay(rendered=original.prompt,receipt=receipt,parent=original.descriptor_path.parent)
        assert all(checks.values()), checks
    finally:
        child = prepared.source_broker._process
        if child is not None and child.poll() is None:
            child.kill();child.wait(timeout=5)
        if child is not None and launch._state.second_retirement is None:
            owner.retire_coding_worker(launch=launch,process=child,source_broker=prepared.source_broker)
        prepared.close();prepared.handoff.close()


def test_public_failure_and_policy_cannot_be_forged_or_detached_from_candidate(successful_first,monkeypatch):
    case = successful_first
    launch, retired = case.launch, case.retired
    with pytest.raises(ValueError,match='typed retired public validation'):
        owner.prepare_second_coding_worker(launch=launch,retired_worker=retired)
    observed = public.validate_retired_candidate(launch=launch,retired_worker=retired)
    for fake in (observed.metadata, replace(observed), object()):
        with pytest.raises(ValueError):
            owner.prepare_second_coding_worker(launch=launch,retired_worker=retired,public_validation=fake)
    original = launch._original_requirements
    options = dict(doctor_requirements=original._doctor_requirements.metadata,
        original_requirements_sha256=owner._sha(original.metadata_bytes),
        deadline_monotonic_ns=int(launch.absolute_deadline*1e9),policy_id=policy.PUBLIC_VALIDATION_POLICY)
    for key in ('original_requirements_sha256','deadline_monotonic_ns','task_cid'):
        contract = launch.public_validation_contract.metadata
        contract[key] = contract[key]+1 if type(contract[key]) is int else ('f'*64 if key.endswith('sha256') else contract[key]+'-other')
        if key == 'task_cid':contract['public_instruction_request']['task_cid'] = contract[key]
        with pytest.raises(ValueError):
            policy.build_doctor_coding_policy(**options,public_validation_contract=contract)
    with monkeypatch.context() as patch:
        native_states = public._states
        def cross_deadline(root,names):
            values = native_states(root,names)
            if set(names) == set(launch.public_validation_contract.metadata['sources']):
                patch.setattr(public,'time',SimpleNamespace(monotonic_ns=lambda:int(launch.absolute_deadline*1e9)+1))
            return values
        patch.setattr(public,'_states',cross_deadline)
        with pytest.raises(ValueError,match='deadline'):
            observed.revalidate(launch=launch,retired_worker=retired)
    assert observed.revalidate(launch=launch,retired_worker=retired)['status'] == 'failed'
    target = case.native.workspace/'eigen.py'
    raw, mode = target.read_bytes(), target.stat().st_mode & 0o777
    # Body drift deliberately discards the working generation. Exercise the
    # independent deadline and mode guards before that irreversible refusal.
    for change, refusal in (('mode','validated candidate generation or modes changed'),
                            ('body','admitted source is stale')):
        try:
            if change == 'body':target.write_bytes(raw+b'\n# changed after public checks\n')
            else:target.chmod(mode ^ 0o100)
            with pytest.raises(ValueError,match=refusal):
                owner.prepare_second_coding_worker(launch=launch,retired_worker=retired,public_validation=observed)
        finally:
            target.write_bytes(raw);target.chmod(mode)
        if change == 'mode':
            assert observed.revalidate(launch=launch,retired_worker=retired)['status'] == 'failed'
    assert launch._state.reserved_attempts == {1} and launch._state.phase == 'captured'
    assert policy.validate_doctor_coding_policy_shape(launch.doctor_coding_policy) == launch.doctor_coding_policy


def test_public_check_cleanup_failure_cannot_mint_observation_or_authorize_continuation(successful_first,monkeypatch):
    case = successful_first
    monkeypatch.setattr(owner,'_isolated_cleanup',lambda:subprocess.CompletedProcess(['authored-failed-cleanup'],1))
    with pytest.raises(ValueError,match='cleanup'):
        public.validate_retired_candidate(launch=case.launch,retired_worker=case.retired)
    assert case.calls == [(1,'structural')]
    assert case.launch._state.public_validations == {} and case.launch._state.reserved_attempts == {1}
    with pytest.raises(ValueError,match='typed retired public validation'):
        owner.prepare_second_coding_worker(launch=case.launch,retired_worker=case.retired)

