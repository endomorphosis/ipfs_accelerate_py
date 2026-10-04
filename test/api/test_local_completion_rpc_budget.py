"""Bounded owner completion and typed transport; no model or benchmark claim."""
from dataclasses import replace
import hashlib
import json
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario, _start
from test.api.test_local_completion_bridge import typed_claim, native_published_transition, complete
from test.api.test_owner_completion_service_retirement import owner_fixture
from ipfs_accelerate_py.agent_supervisor.runtime import local_completion_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as budgets
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as transport


def _bind(server, repo, *, scope=None, deadline=None, root=None):
    root = root or repo.parent
    branch = (subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
        if (repo/'.git/HEAD').exists() else 'main')
    return bridge.bind_owner_local_completion_service(server=server, repo_root=repo,
        portal_attempt_root=root/'portal-attempts', merge_queue_dir=root/'merge-queue',
        board_namespace='local-publication-test', target_branch=branch,
        **({'replay_scope':scope,'deadline_monotonic':deadline} if scope is not None or deadline is not None else {}))


def test_owner_captures_explicit_local_scope_without_changing_retirement(tmp_path, monkeypatch):
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','local-benchmark@1')
    server,gateway,repo = owner_fixture(tmp_path)
    with budgets.local_benchmark_applicability_budget(deadline_monotonic=time.monotonic()+60):
        scope=budgets.capture_applicability_budget()
    result=_bind(server,repo,scope=scope,deadline=time.monotonic()+50)
    assert type(result) is dict and result['completion_authority'] is False
    assert gateway._local_task_validation_binding is None
    assert not gateway.unbind_local_task_validation_handler(gateway._local_task_validation_handler,None)


@pytest.mark.parametrize('change',['missing_scope','missing_lifetime','expired','nonfinite','wrong_profile'])
def test_completion_binding_rejects_invalid_scope_pair(tmp_path,monkeypatch,change):
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','local-benchmark@1')
    server,gateway,repo=owner_fixture(tmp_path)
    with budgets.local_benchmark_applicability_budget(deadline_monotonic=time.monotonic()+60):
        scope=budgets.capture_applicability_budget()
    deadline=time.monotonic()+50
    if change=='missing_scope':scope=None
    elif change=='missing_lifetime':deadline=None
    elif change=='expired':deadline=time.monotonic()-1
    elif change=='nonfinite':deadline=float('nan')
    else:monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE')
    with pytest.raises((ValueError,TimeoutError)):_bind(server,repo,scope=scope,deadline=deadline)
    assert gateway._local_task_validation_handler is None


@pytest.mark.parametrize('left',[True,None,0,-1,float('nan'),float('inf'),121])
def test_invalid_remaining_budget_cannot_launch_or_publish(left):
    with pytest.raises(TimeoutError):local._local_validation_budget(60,lambda:left)


def test_command_budget_can_only_shorten_original_timeout():
    assert local._local_validation_budget(60,None)==60
    assert local._local_validation_budget(60,lambda:100)==60
    assert local._local_validation_budget(60,lambda:2.5)==2.5


def test_validation_checks_budget_after_signing_before_publication(scenario,monkeypatch):
    _start(scenario);(scenario['repository']/'answer.py').write_text('def answer():\n    return 2\n')
    remaining=[10.]
    signed=local._signed
    def signing(payload,manifest):
        result=signed(payload,manifest)
        if payload.get('schema')==local.RESULT_SCHEMA:remaining[0]=0
        return result
    monkeypatch.setattr(local,'_signed',signing)
    with pytest.raises(TimeoutError):
        local.run_local_task_validations(intent=scenario['intent'],task_cid=scenario['task_cid'],
            attempt_id='authored-budget-check',budget_guard=lambda:remaining[0])
    with scenario['intent']._connection() as cx:
        assert cx.execute('SELECT count(*) FROM validation_results').fetchone()[0]==0
    assert scenario['intent'].get_task(scenario['task_cid'])['status']=='in_progress'


def test_expired_prefix_prevents_command_launch(scenario,monkeypatch):
    _start(scenario)
    calls=[]
    monkeypatch.setattr(local.subprocess,'run',lambda *a,**k:calls.append((a,k)))
    with pytest.raises(TimeoutError):
        local.run_local_task_validations(intent=scenario['intent'],task_cid=scenario['task_cid'],
            attempt_id='authored-expired',budget_guard=lambda:0)
    assert calls==[]


@pytest.mark.parametrize('extended',[False,True])
def test_actual_owner_rpc_keeps_scope_and_completion_transport(scenario,tmp_path,monkeypatch,record_property,extended):
    if extended:
        monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','local-benchmark@1')
        monkeypatch.setenv(transport.LOCAL_COMPLETION_RPC_TIMEOUT_ENV,'125')
    else:
        monkeypatch.delenv(transport.LOCAL_COMPLETION_RPC_TIMEOUT_ENV,raising=False)
    with typed_claim(scenario,tmp_path) as (owner,daemon,attempt):
        task=owner.source.get_task(attempt.task_cid)
        native_published_transition(scenario,tmp_path,owner,attempt)
        connection=owner.client._adapter.raw
        assert connection._socket.gettimeout()==30
        assert connection._local_completion_timeout==(125. if extended else None)
        scope=None;deadline=None
        if extended:
            with budgets.local_benchmark_applicability_budget(deadline_monotonic=time.monotonic()+100):
                scope=budgets.capture_applicability_budget()
            deadline=time.monotonic()+90
        original=bridge.authorize_owner_portal_source_transition; seen=[]
        def observed(*args,**kwargs):
            captured=budgets.capture_applicability_budget()
            seen.append((threading.get_ident(),budgets.applicability_replay_timeout(),
                captured.deadline_monotonic if captured else None))
            if extended:time.sleep(31.)
            return original(*args,**kwargs)
        monkeypatch.setattr(bridge,'authorize_owner_portal_source_transition',observed)
        _bind(owner.server,scenario['repository'],scope=scope,deadline=deadline,root=tmp_path)
        # Changing ambient env after bootstrap cannot enlarge or shrink the
        # captured completion transport configuration.
        monkeypatch.setenv(transport.LOCAL_COMPLETION_RPC_TIMEOUT_ENV,'999')
        started=time.monotonic()
        generic='sha256:'+hashlib.sha256(b'authored observed Portal publication').hexdigest()
        owner.source.record_validation_result(task_cid=task.task_cid,outcome='passed',
            evidence_digest=generic,argv=['portal-supervisor-gates'],attempt_id=attempt.attempt_id,
            body={'validator':'DatabasePortalExecutionBridge@1'})
        elapsed=time.monotonic()-started
        complete(owner,attempt,generic)
        assert owner.source.get_task(task.task_cid).status=='completed'
        assert len(seen)==1 and seen[0][0]!=threading.get_ident()
        assert seen[0][1]==(120. if extended else 45.)
        if extended:
            assert elapsed>=31 and seen[0][2]<=min(scope.deadline_monotonic,deadline)
        else:assert seen[0][2] is None
        assert connection._socket.gettimeout()==30 and not connection._closed
        assert owner.server._command_gateway._local_task_validation_active==0
        record_property('completion_rpc_observation',json.dumps({'extended':extended,
            'seconds':elapsed,'authored_prefix_delay_seconds':31 if extended else 0,
            'actual_native_rpc':True,'actual_public_validation':True,'typed_completed':True,
            'original_socket_seconds':30,'completion_receive_seconds':125 if extended else 30,
            'provider_model_calls':0},sort_keys=True))


def test_receive_deadline_is_not_renewed_by_partial_frames():
    client,server=socket.socketpair();finished=threading.Event()
    def stream():
        try:
            server.sendall((20).to_bytes(4,'big'))
            for _ in range(20):
                time.sleep(.025);server.sendall(b' ')
        except OSError:pass
        finally:server.close();finished.set()
    thread=threading.Thread(target=stream);thread.start();started=time.monotonic()
    try:
        with pytest.raises(TimeoutError):transport._receive_frame(client,deadline_monotonic=started+.08)
        assert time.monotonic()-started<.3
    finally:
        client.close();thread.join(timeout=2)
    assert finished.is_set() and not thread.is_alive()


@pytest.mark.parametrize('partial',[False,True])
def test_completion_timeout_poisons_without_second_dispatch(partial):
    client_socket,server_socket=socket.socketpair();client_socket.settimeout(30)
    client=object.__new__(transport.TypedStateOwnerConnection)
    client._socket=client_socket;client._request_lock=threading.RLock()
    client._closed=False;client._active=True;client._prepared_command=object();client._request_index=0
    client._local_completion_timeout=.08;client._local_completion_deadline=None
    client._initial_grant_deadline=time.monotonic()+2
    requests=[];release=threading.Event()
    def peer():
        try:
            requests.append(transport._receive_frame(server_socket))
            if partial:server_socket.sendall((64).to_bytes(4,'big')+b'{')
            release.wait(1)
        finally:server_socket.close()
    thread=threading.Thread(target=peer);thread.start()
    try:
        with pytest.raises(TimeoutError):client.run_local_task_validation(task_cid='task:authored',attempt_id='attempt:authored',expected_revision=1)
        assert client._closed and not client._active and client._prepared_command is None
        assert client._local_completion_deadline is None and client_socket.fileno()==-1
        with pytest.raises(transport.TypedStateOwnerProtocolError,match='closed'):
            client.run_local_task_validation(task_cid='task:authored',attempt_id='attempt:authored',expected_revision=1)
        assert len(requests)==1 and set(requests[0])=={'schema','action','request_id','task_cid','attempt_id','expected_revision'}
    finally:release.set();thread.join(timeout=2)
    assert not thread.is_alive()


def test_well_formed_completion_refusal_restores_normal_transport_timeout():
    client_socket,server_socket=socket.socketpair();client_socket.settimeout(30)
    client=object.__new__(transport.TypedStateOwnerConnection)
    client._socket=client_socket;client._request_lock=threading.RLock()
    client._closed=False;client._active=False;client._prepared_command=None;client._request_index=0
    client._local_completion_timeout=.5;client._local_completion_deadline=None
    client._initial_grant_deadline=time.monotonic()+2
    def peer():
        try:
            request=transport._receive_frame(server_socket)
            transport._send_frame(server_socket,{'schema':transport.TYPED_STATE_OWNER_SCHEMA,
                'request_id':request['request_id'],'ok':False,'error_code':'operation_failed','error_type':'TimeoutError'})
        finally:server_socket.close()
    thread=threading.Thread(target=peer);thread.start()
    try:
        with pytest.raises(transport.TypedStateOwnerRemoteError):client.run_local_task_validation(task_cid='task:authored',attempt_id='attempt:authored',expected_revision=1)
        assert not client._closed and client._socket.gettimeout()==30
        assert client._local_completion_deadline is None
    finally:client._socket.close();thread.join(timeout=2)
    assert not thread.is_alive()


def test_expiry_between_checks_keeps_only_predeadline_observed_evidence(scenario,monkeypatch):
    from copy import deepcopy
    graph=scenario['graph'];task=graph.tasks[0]
    second=replace(task.validations[0],validation_key='public-answer-second')
    task=replace(task,validations=(*task.validations,second));graph=replace(graph,tasks=(task,))
    specs=deepcopy(scenario['manifest']['payload']['tasks'])
    specs[0]['validations'].append({key:local._plain(getattr(second,key)) for key in
        ('validation_key','argv','cwd','expected_exit_codes','policy_cid')})
    manifest=local.author_local_benchmark_manifest(repository=scenario['repository'],
        profile_dir=scenario['profile'],lifecycle_dir=scenario['lifecycle'],task_specs=specs,
        planning_roots={key:getattr(graph,key) for key in ('request_cid','scan_cid','program_root')})
    case={**scenario,'graph':graph,'manifest':manifest,'task_cid':task.task_cid}
    _start(case);(case['repository']/'answer.py').write_text('def answer():\n    return 2\n')
    left=[10.];original=case['intent'].record_validation_result;writes=[]
    def publish(**kwargs):
        result=original(**kwargs);writes.append(kwargs['outcome']);left[0]=0;return result
    monkeypatch.setattr(case['intent'],'record_validation_result',publish)
    with pytest.raises(TimeoutError):
        local.run_local_task_validations(intent=case['intent'],task_cid=task.task_cid,
            attempt_id='authored-two-check-budget',budget_guard=lambda:left[0])
    assert writes==['passed']
    with case['intent']._connection() as cx:
        assert [row[0] for row in cx.execute('SELECT outcome FROM validation_results WHERE task_cid=?',[task.task_cid]).fetchall()]==['passed']
    assert case['intent'].get_task(task.task_cid)['status']=='in_progress'


def test_prefix_cost_shortens_actual_validation_timeout(scenario,monkeypatch):
    _start(scenario);(scenario['repository']/'answer.py').write_text('def answer():\n    return 2\n')
    left=[20.];original_contract=local._contract;original_run=local.subprocess.run;timeouts=[]
    def contract(*args,**kwargs):
        result=original_contract(*args,**kwargs);left[0]=.75;return result
    def run(argv,**kwargs):
        if argv[-1]=='test_answer.py':timeouts.append(kwargs['timeout'])
        return original_run(argv,**kwargs)
    monkeypatch.setattr(local,'_contract',contract);monkeypatch.setattr(local.subprocess,'run',run)
    result=local.run_local_task_validations(intent=scenario['intent'],task_cid=scenario['task_cid'],
        attempt_id='authored-prefix-cost',budget_guard=lambda:left[0])
    assert result['passed'] and timeouts==[.75]


@pytest.mark.parametrize('ceiling',['work','lifetime','grant_renewal','request_queue','revocation'])
def test_native_callback_cannot_renew_ceilings_or_publish_after_authority_loss(
        scenario,tmp_path,monkeypatch,record_property,ceiling):
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','local-benchmark@1')
    monkeypatch.setenv(transport.LOCAL_COMPLETION_RPC_TIMEOUT_ENV,'125')
    with typed_claim(scenario,tmp_path) as (owner,daemon,attempt):
        task=owner.source.get_task(attempt.task_cid)
        native_published_transition(scenario,tmp_path,owner,attempt)
        connection=owner.client._adapter.raw;gateway=owner.server._command_gateway
        initial_now=time.monotonic()
        with budgets.local_benchmark_applicability_budget(deadline_monotonic=initial_now+(8 if ceiling=='work' else 200)):
            scope=budgets.capture_applicability_budget()
        lifetime=initial_now+(8 if ceiling=='lifetime' else 180)
        grant_id=connection.grant['grant_id']
        if ceiling=='grant_renewal':
            with gateway._grants_lock:
                matches=[(key,value) for key,value in gateway._grants.items() if value.grant_id==grant_id]
                assert len(matches)==1
                key,value=matches[0]
                gateway._grants[key]=replace(value,expires_at=int(time.time()*1000)+8000)
        original=bridge.authorize_owner_portal_source_transition;observations=[]
        actual_time=time.monotonic
        clock=[0.]
        monkeypatch.setattr(bridge,'time',SimpleNamespace(monotonic=lambda:actual_time()+clock[0],time=time.time))
        def boundary(*args,**kwargs):
            captured=budgets.capture_applicability_budget()
            request_deadline=transport._LOCAL_COMPLETION_REQUEST_DEADLINE.get()
            assert request_deadline is not None
            assert captured.deadline_monotonic<=min(scope.deadline_monotonic,lifetime,request_deadline)
            if ceiling=='work':assert captured.deadline_monotonic==scope.deadline_monotonic
            elif ceiling=='lifetime':assert captured.deadline_monotonic==lifetime
            elif ceiling=='request_queue':assert captured.deadline_monotonic==request_deadline
            elif ceiling=='grant_renewal':assert captured.deadline_monotonic<=initial_now+8.1
            transition=original(*args,**kwargs)
            if ceiling=='revocation':
                gateway.revoke_grant(grant_id)
            else:
                if ceiling=='grant_renewal':
                    # Owner-only renewal of the active grant cannot move the
                    # callback's already captured expiry.
                    renewed=gateway.renew_grant(grant_id,ttl_seconds=180)
                    assert renewed.expires_at>int((time.time()+100)*1000)
                clock[0]=captured.deadline_monotonic-actual_time()+.1
            observations.append({'ceiling':ceiling,'captured_deadline':captured.deadline_monotonic,
                'request_deadline':request_deadline,'simulated_elapsed':ceiling!='revocation'})
            return transition
        monkeypatch.setattr(bridge,'authorize_owner_portal_source_transition',boundary)
        if ceiling=='request_queue':
            invoke=gateway._invoke_local_task_validation_handler_locked
            def queued(*args,**kwargs):
                received=transport._LOCAL_COMPLETION_REQUEST_DEADLINE.get()
                time.sleep(.04)
                assert received<time.monotonic()+120
                return invoke(*args,**kwargs)
            monkeypatch.setattr(gateway,'_invoke_local_task_validation_handler_locked',queued)
        _bind(owner.server,scenario['repository'],scope=scope,deadline=lifetime,root=tmp_path)
        with pytest.raises(transport.TypedStateOwnerRemoteError):
            connection.run_local_task_validation(task_cid=task.task_cid,attempt_id=attempt.attempt_id,
                expected_revision=task.revision)
        assert len(observations)==1
        assert gateway._local_task_validation_active==0
        assert not connection._closed and connection._socket.gettimeout()==30
        with owner.server._lock:
            assert owner.server._connection.execute('SELECT count(*) FROM validation_results').fetchone()[0]==0
        if ceiling!='revocation':assert owner.source.get_task(task.task_cid).status=='in_progress'
        record_property('completion_refusal_boundary',json.dumps(observations[0],sort_keys=True))
