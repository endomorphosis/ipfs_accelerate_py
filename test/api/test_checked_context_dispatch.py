"""Protocol/refusal units with explicitly mocked native authority and births.

No test below qualifies Docker, admitted START, SQL custody or a real worker.
Local socket framing and current-process kernel facts are real where stated.
"""
import ast
from copy import copy, deepcopy
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'ipfs_accelerate_py/agent_supervisor/runtime/checked_context_dispatch.py'
spec = importlib.util.spec_from_file_location('checked_dispatch_client_unit', SOURCE)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def frame(value):
    raw = d._wire(value)
    return len(raw).to_bytes(4,'big') + raw


class Channel:
    def __init__(self, incoming=b'', limit=7):
        self.incoming, self.sent, self.closed, self.limit = incoming,bytearray(),False,limit
    def recv(self,n):
        n=min(n,self.limit);value,self.incoming=self.incoming[:n],self.incoming[n:];return value
    def sendall(self,raw): self.sent.extend(raw)
    def close(self): self.closed=True
    def settimeout(self,_): pass
    def connect(self,_): pass
    def getsockopt(self,*_):
        import struct
        return struct.pack('3i',os.getpid(),os.geteuid(),os.getegid())


def base():
    return ['/opt/ipfs-supervisor/bin/router-worker','--observe-model-input-only',d.OWNER_FLAG,'cid:scope']


def derived():
    return d._owner_recipe(base(),'cid:scope') + ['--checked-context-artifact','/opt/public/checked.json',
        '--checked-context-sha256','a'*64,'--checked-context-task-cid','cid:task']


def request():
    return {'schema':d.SCHEMA,'operation':'prepare','context_cid':'cid:scope','command':base(),
        'worktree':'/opt/worktrees/task','input_text':'whole native stdin\nwith Unicode λ',
        'task_id':'TASK','database_task_cid':'cid:task','database_attempt_id':'attempt:1',
        'database_claim_id':'claim:1','database_attempt_number':1}


def prepared(*, command=None):
    raw=request()['input_text'].encode()
    return {'schema':d.SCHEMA,'ok':True,'phase':'prepared','handoff_id':'n'*32,
        'command':derived() if command is None else command,'input_sha256':hashlib.sha256(raw).hexdigest(),
        'input_bytes':len(raw)}


def released():
    return {'schema':d.SCHEMA,'ok':True,'phase':'released','handoff_id':'n'*32}


def test_canonical_frame_roundtrip_handles_fragmented_unicode():
    assert d._receive(Channel(frame(request()),limit=1))==request()


@pytest.mark.parametrize('raw',[
    b'{"x":1,"x":2}',b'{ "x":1}',b'[]',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":"\xff"}',b'{',
])
def test_malformed_duplicate_noncanonical_frames_refuse(raw):
    with pytest.raises(d.CheckedContextDispatchError): d._receive(Channel(len(raw).to_bytes(4,'big')+raw))


@pytest.mark.parametrize('size',[0,1,d.MAX_BYTES+1,2**32-1])
def test_size_bound_refuses_before_body_read(size):
    with pytest.raises(d.CheckedContextDispatchError,match='size'):d._receive(Channel(size.to_bytes(4,'big')))


def test_truncated_frame_and_send_bound_refuse():
    with pytest.raises(d.CheckedContextDispatchError):d._receive(Channel((20).to_bytes(4,'big')+b'{}'))
    channel=Channel()
    with pytest.raises(d.CheckedContextDispatchError):d._send(channel,{'x':'a'*d.MAX_BYTES})
    assert not channel.sent


@pytest.mark.parametrize('value',[None,False,1,[],{},'', 'a'*(d.MAX_INPUT_BYTES+1)],ids=['none','false','integer','list','dict','empty','over-bound'])
def test_request_requires_complete_bounded_string_stdin(value):
    x=request();x['input_text']=value
    with pytest.raises(d.CheckedContextDispatchError):d._request(x)


@pytest.mark.parametrize('field,value',[
    ('database_attempt_number',True),('database_attempt_number',0),('database_attempt_number',2**32),
    ('task_id',''),('task_id','bad\nidentity'),('context_cid','cid:other'),('schema',[]),('operation','spawned')])
def test_closed_native_binding_refusals(field,value):
    x=request();x[field]=value
    with pytest.raises(d.CheckedContextDispatchError):d._request(x)


@pytest.mark.parametrize('mutation',['extra','missing','duplicate','equals','checked','wrong-context'])
def test_request_recipe_and_closed_fields_refuse(mutation):
    x=request()
    if mutation=='extra':x['tokens']=1
    elif mutation=='missing':x.pop('input_text')
    elif mutation=='duplicate':x['command'] += [d.OWNER_FLAG,'cid:scope']
    elif mutation=='equals':x['command'][-2:]=[d.OWNER_FLAG+'=cid:scope']
    elif mutation=='checked':x['command'] += ['--checked-context-sha256','a'*64]
    else:x['command'][-1]='cid:other'
    with pytest.raises(d.CheckedContextDispatchError):d._request(x)


def test_absent_route_keeps_ordinary_dispatch_and_present_routing_never_falls_back():
    assert d.require_checked_context_worker_dispatch(command=['/ordinary'],worktree='/work',
        environment={},dispatch_binding=None,input_text=None)is None
    with pytest.raises(d.CheckedContextDispatchError,match='fall back'):
        d.require_checked_context_worker_dispatch(command=['/ordinary'],worktree='/work',
            environment={d.CONTEXT_ENV:'cid:scope'},dispatch_binding=None,input_text=None)
    assert d.require_checked_context_isolated_worker_dispatch(command=['/ordinary'],worktree='/work',environment={})is None


@pytest.mark.parametrize('mutation',['prefix','extra','missing','bad-hash','relative-path','wrong-order','owner-remains'])
def test_derived_command_has_only_exact_permitted_augmentation(mutation):
    x=derived()
    if mutation=='prefix':x[0]='/other'
    elif mutation=='extra':x+=['--free-option','x']
    elif mutation=='missing':x=x[:-2]
    elif mutation=='bad-hash':x[-3]='z'*64
    elif mutation=='relative-path':x[-5]='relative.json'
    elif mutation=='wrong-order':x[-6],x[-4]=x[-4],x[-6]
    else:x+= [d.OWNER_FLAG,'cid:scope']
    with pytest.raises(d.CheckedContextDispatchError):d._derived(x,base(),'cid:scope')


def test_client_guard_binds_actual_stdin_and_rejects_copy_mutation_replay(monkeypatch):
    channel=Channel(frame(prepared())+frame(released()))
    monkeypatch.setattr(d,'_endpoint',lambda *_:Path('/literal/socket'))
    monkeypatch.setattr(d.socket,'socket',lambda *_:channel)
    guard=d._connect(request(),{d.OWNER_ENV:d._wire(d.owner_dispatch_identity()).decode()})
    assert guard.command==derived() and guard.child_environment()=={d.WRAPPER_ENV:'n'*32}
    clone=copy(guard)
    with pytest.raises(d.CheckedContextDispatchError):clone.command
    guard.command.append('cannot mutate original')
    assert guard.command==derived()
    guard.abort();assert channel.closed
    with pytest.raises(d.CheckedContextDispatchError):guard.command
    with pytest.raises(d.CheckedContextDispatchError):guard.note_spawned(SimpleNamespace(pid=1))


@pytest.mark.parametrize('field,value',[
    ('input_sha256','b'*64),('input_bytes',True),('input_bytes',1),('ok',False),('phase','released'),
    ('command',['/wrong']),('handoff_id',''),('input_sha256',[]),
])
def test_client_refuses_incorrect_prepared_response(monkeypatch,field,value):
    x=prepared();x[field]=value;channel=Channel(frame(x))
    monkeypatch.setattr(d,'_endpoint',lambda *_:Path('/literal/socket'))
    monkeypatch.setattr(d.socket,'socket',lambda *_:channel)
    with pytest.raises(d.CheckedContextDispatchError):d._connect(request(),{d.OWNER_ENV:d._wire(d.owner_dispatch_identity()).decode()})
    assert channel.closed


def test_client_rejects_another_task_even_when_recipe_shape_matches(monkeypatch):
    x=prepared();x['command'][-1]='cid:other';channel=Channel(frame(x))
    monkeypatch.setattr(d,'_endpoint',lambda *_:Path('/literal/socket'))
    monkeypatch.setattr(d.socket,'socket',lambda *_:channel)
    with pytest.raises(d.CheckedContextDispatchError,match='another native task'):d._connect(request(),{d.OWNER_ENV:d._wire(d.owner_dispatch_identity()).decode()})


def test_real_private_socket_mode_checks_and_kernel_peer(tmp_path):
    # pytest's namespaced base may exceed Linux's 108-byte sockaddr limit.
    # Only the real socket lives in this short, owned, automatically reaped dir.
    with tempfile.TemporaryDirectory(prefix='aa-ipc-',dir='/tmp') as name:
        private=Path(name);private.chmod(0o700);path=private/'owner.sock'
        (tmp_path/'socket-fixture.json').write_text(json.dumps({
            'directory':name,'socket_path':str(path),'owner_uid':private.stat().st_uid,
            'directory_mode':oct(private.stat().st_mode & 0o777),'ephemeral_real_socket':True}))
        channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            channel.bind(str(path));path.chmod(0o600)
            assert d._endpoint({d.SOCKET_ENV:str(path),d.CONTEXT_ENV:'cid:scope'},'cid:scope')==path
            path.chmod(0o660)
            with pytest.raises(d.CheckedContextDispatchError):d._endpoint({d.SOCKET_ENV:str(path),d.CONTEXT_ENV:'cid:scope'},'cid:scope')
            left,right=socket.socketpair()
            try:assert d._peer(left)==(os.getpid(),os.geteuid())
            finally:left.close();right.close()
        finally:channel.close()
    assert not private.exists()


def test_real_current_process_birth_uses_native_domain_without_owner_imports():
    value=d._kernel_identity(os.getpid())
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    raw=f"{os.getpid()}:{value['start_time_ticks']}:{boot}:{value['parent_pid']}".encode()
    assert value['process_birth_id']=='birth:'+hashlib.sha256(raw).hexdigest()[:32]
    assert value['uid']==os.geteuid()


def mocked_broker(monkeypatch):
    """Explicit mock authority: exercises ordering, never native origin."""
    events=[];b=object.__new__(d.OwnerCheckedContextDispatchBroker)
    b._lock=d.threading.RLock();b._pending=b._consumed=b._wrapper_spawned=b._isolated_spawned=False
    b._stop=d.threading.Event()
    b._ambiguous=b._prepared_ack=False;b._context='cid:scope';b._observations=[]
    b._prepared=b._wrapper=None;b._server=SimpleNamespace(_lock=d.threading.RLock())
    b._git_metadata_guard=b._git_metadata_original=b._git_metadata_type=None
    b._git_metadata_id=b._git_metadata_wire=None;b._git_metadata_closed=False
    b._identity_current=lambda:events.append('identity')
    claim={'native':'mock'}
    b._closing_authority=lambda *a:events.append('native-current')or claim
    b._execute_with_attempt_fence=lambda request,peer,callback:callback(claim,
        SimpleNamespace(expires_at_ms=d.time.time_ns()//1_000_000+60_000))
    b._run_lease_current=lambda:events.append('run-lease-current')
    b._scope=SimpleNamespace(
        prepare_dispatch=lambda **kw:events.append('reserve')or {'schema':'checked-observer-budget-dispatch@1',
            'command':derived(),'context_cid':'cid:scope'},
        require_current=lambda:events.append('scope-current'),
        capture_allocated=lambda:events.append('capture-allocated'),
        abort_pending=lambda:events.append('abort-nonrefund'))
    b._grant_worker_git_metadata_current=lambda *_:events.append('git-grant')or {'explicit_mock':True}
    b._git_metadata_current=lambda:events.append('git-current')or {'explicit_mock':True}
    b._git_metadata_readonly_environment=lambda:nullcontext()  # Explicit mock authority.
    b._native=SimpleNamespace(typed=SimpleNamespace(_kernel_peer_identity=lambda _: (11,os.geteuid(),101)),
        _require_wrapper_kernel_command=lambda **kw:{'pid':22,'parent_pid':11,'start_time_ticks':202,'process_birth_id':'birth:22'},
        _same_native_dispatch_claim=lambda *args:True,
        _require_native_grant_ack_budget=lambda **_:events.append('grant-current'),
        _kernel_identity=lambda pid:{'pid':22,'parent_pid':11,'start_time_ticks':202,'process_birth_id':'birth:22'})
    monkeypatch.setattr(d.uuid,'uuid4',lambda:SimpleNamespace(hex='n'*32))
    return b,events


def spawned_ack():
    return {'schema':d.SCHEMA,'operation':'spawned','handoff_id':'n'*32,'pid':22,'process_birth_id':'birth:22'}


def test_mocked_native_prepare_reserves_before_ack_and_retains_pending_budget(monkeypatch):
    b,events=mocked_broker(monkeypatch);channel=Channel(frame(request())+frame(spawned_ack()))
    send=channel.sendall
    def checked_send(raw):
        if b'"phase":"prepared"'in raw:assert 'reserve'in events and 'scope-current'in events
        send(raw)
    channel.sendall=checked_send
    b._handoff(channel)
    assert events.index('reserve')<events.index('scope-current')
    assert b._consumed and b._wrapper_spawned and not b._isolated_spawned and not b._pending
    assert 'abort-nonrefund'not in events
    output=Channel(bytes(channel.sent));assert d._receive(output)['command']==derived()
    assert d._receive(output)==released()
    with pytest.raises(d.CheckedContextDispatchError,match='consumed'):
        b._handoff(Channel(frame(request())))


def test_mocked_explicit_abort_charges_and_blocks_another_prepare(monkeypatch):
    b,events=mocked_broker(monkeypatch)
    abort={'schema':d.SCHEMA,'operation':'abort','handoff_id':'n'*32}
    b._handoff(Channel(frame(request())+frame(abort)))
    assert b._consumed and b._ambiguous and not b._wrapper_spawned
    assert events.count('reserve')==events.count('abort-nonrefund')==1
    with pytest.raises(d.CheckedContextDispatchError):b._handoff(Channel(frame(request())))


def test_mocked_budget_refusal_never_sends_prepared_or_observes_birth(monkeypatch):
    b,events=mocked_broker(monkeypatch)
    def reject(**_):events.append('reserve-refused');raise d.CheckedContextDispatchError('budget exhausted')
    b._scope.prepare_dispatch=reject;channel=Channel(frame(request()))
    with pytest.raises(d.CheckedContextDispatchError,match='budget exhausted'):b._handoff(channel)
    assert not channel.sent and not b._wrapper_spawned and b._consumed and not b._pending


@pytest.mark.parametrize('mutation',['uid','claim','context','wrapper-birth'])
def test_mocked_native_authority_failures_cannot_authorize_worker(monkeypatch,mutation):
    b,events=mocked_broker(monkeypatch);x=request()
    if mutation=='uid':b._native.typed._kernel_peer_identity=lambda _:(11,os.geteuid()+1,101)
    elif mutation=='claim':
        def reject(*_):raise d.CheckedContextDispatchError('claim mismatch')
        b._closing_authority=reject
    elif mutation=='context':x['context_cid']='cid:other';x['command'][-1]='cid:other'
    else:b._native._require_wrapper_kernel_command=lambda **_:{'parent_pid':99,'start_time_ticks':202,'process_birth_id':'birth:22'}
    with pytest.raises(d.CheckedContextDispatchError):b._handoff(Channel(frame(x)+frame(spawned_ack())))
    assert not b._wrapper_spawned and not b._isolated_spawned
    if mutation!='wrapper-birth':assert 'reserve'not in events


def test_isolated_request_before_registered_wrapper_refuses(monkeypatch):
    b,_=mocked_broker(monkeypatch)
    x={'schema':d.SCHEMA,'operation':'prepare_isolated','context_cid':'cid:scope','command':['/usr/bin/sudo'],
        'worktree':'/work','wrapper_ticket':'n'*32}
    with pytest.raises(d.CheckedContextDispatchError,match='registered'):b._isolated_handoff(Channel(),x,(22,os.geteuid(),202))


def test_client_source_has_no_eager_native_database_proof_or_benchmark_imports():
    tree=ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node,ast.ImportFrom)and node.module not in {'__future__','pathlib','contextlib'}:pytest.fail('native import at client module top level')
        if isinstance(node,ast.Import):assert all(not x.name.startswith(('ipfs','duckdb','benchmarks'))for x in node.names)
    assert 'from .checked_observer_budget import CheckedObserverBudgetScope'in SOURCE.read_text()


@pytest.mark.parametrize('marker',[None,'{}','{"pid":true,"process_birth_id":"birth:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
    '{ "pid":1,"process_birth_id":"birth:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
    '{"pid":1,"pid":1,"process_birth_id":"birth:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'])
def test_missing_or_malformed_original_owner_marker_refuses_before_request(monkeypatch,marker):
    channel=Channel(frame(prepared()))
    monkeypatch.setattr(d,'_endpoint',lambda *_:Path('/literal/socket'))
    monkeypatch.setattr(d.socket,'socket',lambda *_:channel)
    with pytest.raises(d.CheckedContextDispatchError):d._connect(request(),{d.OWNER_ENV:marker})
    assert not channel.sent and channel.closed


@pytest.mark.parametrize('field',['pid','process_birth_id'])
def test_same_uid_server_with_another_original_birth_is_refused(monkeypatch,field):
    identity=d.owner_dispatch_identity()
    identity[field]=identity[field]+1 if field=='pid'else'birth:'+'0'*32
    channel=Channel(frame(prepared()))
    monkeypatch.setattr(d,'_endpoint',lambda *_:Path('/literal/socket'))
    monkeypatch.setattr(d.socket,'socket',lambda *_:channel)
    with pytest.raises(d.CheckedContextDispatchError,match='original owner birth'):
        d._connect(request(),{d.OWNER_ENV:d._wire(identity).decode()})
    assert not channel.sent


def test_owner_birth_helper_explicitly_selects_parent_zero_policy(monkeypatch):
    observed=[]
    def kernel(pid,*,allow_parent_zero=False):
        observed.append(allow_parent_zero)
        if not allow_parent_zero:raise d.CheckedContextDispatchError('positive child ancestry required')
        return {'pid':pid,'process_birth_id':'birth:'+'a'*32}
    monkeypatch.setattr(d,'_kernel_identity',kernel)
    assert d.owner_dispatch_identity()['pid']==os.getpid() and observed==[True]
    guard=d._DispatchGuard(Channel(),prepared())
    with pytest.raises(d.CheckedContextDispatchError,match='positive child ancestry'):
        guard.note_spawned(SimpleNamespace(pid=12))
    assert observed==[True,False]


@pytest.mark.parametrize('mutation',['copy','fork','command','ticket'])
def test_guard_original_identity_refuses_tampering_before_kernel_observation(monkeypatch,mutation):
    guard=d._DispatchGuard(Channel(),prepared())
    if mutation=='copy':guard=copy(guard)
    elif mutation=='fork':guard._pid+=1
    elif mutation=='command':guard._command=('/other',)
    else:guard._handoff_id='other'
    monkeypatch.setattr(d,'_kernel_identity',lambda *_:pytest.fail('kernel callback before guard authentication'))
    with pytest.raises(d.CheckedContextDispatchError):guard.note_spawned(SimpleNamespace(pid=12))


def test_mocked_isolated_birth_requires_post_chmod_capture_and_keeps_reservation(monkeypatch):
    b,events=mocked_broker(monkeypatch)
    b._handoff(Channel(frame(request())+frame(spawned_ack())))
    command=['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry',*derived()[1:]]
    x={'schema':d.SCHEMA,'operation':'prepare_isolated','context_cid':'cid:scope','command':command,
       'worktree':request()['worktree'],'wrapper_ticket':'n'*32}
    b._native._isolated_kernel_birth=lambda **_:{'sudo':{'process_birth_id':'birth:33'},
        'worker':{'pid':44,'process_birth_id':'birth:44'},'parent_chain':[{'pid':44},{'pid':33}]}
    # This legacy fixture tests post-chmod ordering and retained reservation;
    # complete kernel/helper joins are exercised by isolated_worktree_fixture.
    b._worker_worktree_current=lambda worker,worktree,**_:events.append('mock-cwd-observation')or {'explicit_mock':True}
    ack={'schema':d.SCHEMA,'operation':'spawned','handoff_id':'n'*32,'pid':33,'process_birth_id':'birth:33'}
    channel=Channel(frame(ack));b._isolated_handoff(channel,x,(22,os.geteuid(),202))
    capture=events.index('capture-allocated')
    assert b._isolated_spawned and not b._ambiguous
    assert events[capture:capture+4]==['capture-allocated','scope-current','native-current','git-current']
    assert events[-4:]==['mock-cwd-observation','git-current','run-lease-current','grant-current']
    assert 'abort-nonrefund'not in events
    with pytest.raises(d.CheckedContextDispatchError):b._isolated_handoff(Channel(),x,(22,os.geteuid(),202))


@pytest.mark.parametrize('mutation',['ticket','pid','start','command','cwd'])
def test_mocked_isolated_identity_recipe_and_cwd_controls(monkeypatch,mutation):
    if mutation=='cwd':
        b,events,x,channel,_=isolated_worktree_fixture(monkeypatch,'worktree')
        with pytest.raises(d.CheckedContextDispatchError):b._isolated_handoff(channel,x,(22,os.geteuid(),202))
        assert not b._isolated_spawned
        return
    b,events=mocked_broker(monkeypatch)
    b._handoff(Channel(frame(request())+frame(spawned_ack())))
    command=['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry',*derived()[1:]]
    x={'schema':d.SCHEMA,'operation':'prepare_isolated','context_cid':'cid:scope','command':command,
       'worktree':request()['worktree'],'wrapper_ticket':'n'*32};peer=(22,os.geteuid(),202)
    if mutation=='ticket':x['wrapper_ticket']='forged'
    elif mutation=='pid':peer=(23,peer[1],peer[2])
    elif mutation=='start':peer=(peer[0],peer[1],203)
    elif mutation=='command':x['command'][3]='root'
    ack={'schema':d.SCHEMA,'operation':'spawned','handoff_id':'n'*32,'pid':33,'process_birth_id':'birth:33'}
    with pytest.raises(d.CheckedContextDispatchError):b._isolated_handoff(Channel(frame(ack)),x,peer)
    assert not b._isolated_spawned


@pytest.mark.parametrize('corrupt',[None,'status','binding','portal-task'])
def test_owner_projection_helper_uses_current_native_data_with_mocked_api(monkeypatch,corrupt):
    """Exercise owner helper imports/joins using fake API objects, no databases."""
    from dataclasses import make_dataclass
    import sys
    from types import ModuleType
    monkeypatch.setattr(d,'__package__','ipfs_accelerate_py.agent_supervisor.runtime')
    receipt={key:'mock:'+key for key in ('attempt_id','claim_id','owner_session_id','lease_id')}
    receipt.update(attempt_number=1,fencing_token=1,fence_epoch=1,attempt_execution_phase='claimed',attempt_execution_revision=1)
    record=SimpleNamespace(task_cid='cid:task',task_alias='TASK',status='in_progress',revision=3,
                           body={'completion_receipt':receipt})
    if corrupt=='status':record.status='ready'
    Attempt=make_dataclass('MockNativeAttempt',[(name,object)for name in
        ('task_cid','task_alias','attempt_id','claim_id','attempt_number','owner_session_id',
         'fencing_token','fence_epoch','lease_id','committed_phase','status','started_at_ms','revision','body')])
    class Source:
        def __init__(self,**_):pass
        def get_task(self,_):return record
    paths=SimpleNamespace(root=Path('/private/attempt'),binding=Path('/private/attempt/binding.json'),
                          task_projection=Path('/private/attempt/task.md'))
    class Bridge:
        def __init__(self,**_):pass
        def _paths(self,_):return paths
        def _seal_attempt_directory(self,_,**kwargs):assert kwargs['create']is False
        def _strict_binding(self,_):return {'schema':'mock-binding','value':1}
        def _binding(self,*_,**__):return {'schema':'mock-binding','value':2 if corrupt=='binding'else 1}
        def _render_projection(self,*_):return {'mock':'projection'}
        def _prior_projection_identity(self,*_):return {'task_id':'OTHER'if corrupt=='portal-task'else'TASK'}
    modules={
        'task_sources.intent_repository':{'IntentRepository':lambda **_:object()},
        'task_sources.database_task_source':{'DatabaseTaskSource':Source},
        'task_sources.board_control_plane':{'infer_board_namespace':lambda **_:'mock-board'},
        'todo_daemon.database_portal_bridge':{'DatabasePortalExecutionBridge':Bridge},
        'todo_daemon.implementation_daemon':{'DatabaseTaskAttempt':Attempt,
            'DatabaseImplementationDaemon':SimpleNamespace(_control_claim_binding=lambda *_:{'mock':'claim'})}}
    for name,attributes in modules.items():
        full='ipfs_accelerate_py.agent_supervisor.'+name;module=ModuleType(full)
        for key,value in attributes.items():setattr(module,key,value)
        monkeypatch.setitem(sys.modules,full,module)
    monkeypatch.setattr(d,'_owner_regular',lambda path,*args,**kwargs:(None,b'ref: refs/heads/main\n'))
    broker=SimpleNamespace(_runtime=SimpleNamespace(state=Path('/private'),manifest={'repository_root':'/repo'}),
        _server=SimpleNamespace(_connection=object(),config=SimpleNamespace(database_path='/private/control.duckdb')))
    if corrupt:
        with pytest.raises(d.CheckedContextDispatchError):d._native_projection(broker,request(),{'task_revision':3},lambda:None)
    else:
        actual=d._native_projection(broker,request(),{'task_revision':3},lambda:None)
        assert actual[0]is paths and actual[1]=={'schema':'mock-binding','value':1} and actual[3]=='main'


@pytest.mark.parametrize('mutation',[None,'manifest','scope','socket','wrapper','preparation'])
def test_original_broker_identity_uses_owner_local_runtime_verification_under_server_lock(mutation):
    """Real broker checkpoint/lock/socket; native runtime authority is mocked."""
    calls=[]
    with tempfile.TemporaryDirectory(prefix='aa-local-',dir='/tmp') as name:
        state=Path(name);state.chmod(0o700);path=state/'checked-context.sock'
        channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            channel.bind(str(path));path.chmod(0o600)
            lock=d.threading.RLock();server=SimpleNamespace(_connection=object(),_lock=lock)
            runtime=SimpleNamespace(server=server,owner_identity=object(),client_id='mock-native-client',
                manifest={'mock':'native-manifest'},manifest_id='mock-native-manifest-id')
            def verify(*,_owner_local_tasks=False):
                assert lock._is_owned(), 'regression must exercise the held original owner lock'
                if not _owner_local_tasks:
                    pytest.fail('remote Quack RPC would wait for the held owner lock')
                calls.append('complete-owner-local-native-verification')
            runtime._verify=verify
            scope=SimpleNamespace(bound_runtime=runtime,material_binding={'mock':'scope'},context_cid='mock-context')
            broker=object.__new__(d.OwnerCheckedContextDispatchBroker)
            broker._seal=d._SEAL;broker._pid=os.getpid();broker._closed=False
            broker._scope,broker._runtime=scope,runtime;broker._owner_identity=d.owner_dispatch_identity()
            broker._manifest_id=runtime.manifest_id;broker.socket_path=broker._original_socket_path=path
            broker._population={'mock':'before-START-population'};broker._population_bytes=d._wire(broker._population)
            broker._wrapper=broker._wrapper_wire=broker._prepared=broker._prepared_wire=None
            broker._state_path=state;entry=state.lstat()
            broker._state_identity=(entry.st_dev,entry.st_ino,entry.st_uid,entry.st_mode & 0o777)
            entry=path.lstat();broker._socket_identity=(entry.st_dev,entry.st_ino)
            broker._fields=(scope,runtime,server,server._connection,lock,runtime.owner_identity,runtime.client_id,
                d._wire(runtime.manifest),d._wire(scope.material_binding),scope.context_cid)
            with d._REGISTRY_LOCK:d._BROKERS[id(broker)]=broker
            try:
                # A metadata-read checkpoint authenticates the original object
                # and endpoint without reconstructing current model context.
                with lock:broker._identity_checkpoint()
                assert calls==[]
                with lock:broker._identity_current()
                assert calls==['complete-owner-local-native-verification']
                if mutation is not None:
                    if mutation=='manifest':runtime.manifest['changed']=True
                    elif mutation=='scope':scope.bound_runtime=SimpleNamespace()
                    elif mutation=='socket':path.chmod(0o660)
                    elif mutation=='wrapper':broker._wrapper={'changed':True}
                    else:broker._prepared={'changed':True}
                    with lock,pytest.raises(d.CheckedContextDispatchError):broker._identity_checkpoint()
                    assert calls==['complete-owner-local-native-verification']
            finally:
                with d._REGISTRY_LOCK:d._BROKERS.pop(id(broker),None)
        finally:channel.close()


def batch_closing_fixture(monkeypatch, tmp_path, *, late=None, after_reserve=False):
    """Real closing/handoff methods with explicit mutable mock native authority."""
    broker,events=mocked_broker(monkeypatch)
    state={'closing':0,'whole_in_closing':0,'claim':1,'grant':True,'worktree':True,
           'bootstrap':True,'run_lease':True,'proof':True}
    allocation=tmp_path/'authored-native-allocation.json'
    allocation.write_bytes(b'{"mock_allocated_task":"TASK"}');allocation.chmod(0o600)
    allocation_raw=allocation.read_bytes()
    captured={'claim':1}
    def whole():
        events.append('full-current')
        state['whole_in_closing']+=1
        selected=2 if after_reserve else 1
        if late is not None and state['closing']==selected and state['whole_in_closing']==2:
            state[late]=2 if late=='claim'else False
            if late=='worktree':allocation.write_bytes(b'{"mock_allocated_task":"OTHER"}')
        d._need(state['proof'],'mock retained proof custody changed')
    broker._identity_current=whole
    broker._identity_checkpoint=lambda:events.append('identity-only')
    def claim(*_):
        events.append('claim-current');return {'claim':state['claim']}
    def bootstrap(_):
        events.append('bootstrap-current');d._need(state['bootstrap'],'mock bootstrap changed')
    def worktree(*_):
        events.append('worktree-current')
        # The helper's bounded-read checkpoints carry no effect authority.
        for _ in range(6):broker._identity_checkpoint()
        _,raw=d._owner_regular(allocation,broker._identity_checkpoint)
        d._need(raw==allocation_raw,'mock worktree changed')
    def run_lease():
        events.append('run-lease-current');d._need(state['run_lease'],'mock run lease changed')
    def grant(**_):
        events.append('grant-current');d._need(state['grant'],'mock grant changed')
    broker._claim=claim;broker._bootstrap_current=bootstrap;broker._worktree_current=worktree
    broker._run_lease_current=run_lease
    broker._native._same_native_dispatch_claim=lambda current,original:current==original
    broker._native._require_native_grant_ack_budget=grant
    def closing(*args):
        state['closing']+=1;state['whole_in_closing']=0
        return d.OwnerCheckedContextDispatchBroker._closing_authority(broker,*args)
    broker._closing_authority=closing
    lease=SimpleNamespace(expires_at_ms=d.time.time_ns()//1_000_000+60_000)
    broker._execute_with_attempt_fence=lambda req,peer,callback:callback(captured,lease)
    return broker,events,state,captured,lease


def test_full_current_surrounds_batch_then_native_authority_is_reread_before_ack(monkeypatch,tmp_path):
    broker,events,state,captured,lease=batch_closing_fixture(monkeypatch,tmp_path)
    assert broker._closing_authority(request(),(11,os.geteuid(),101),captured,lease)==captured
    important=[x for x in events if x!='identity-only']
    assert important==['full-current','claim-current','bootstrap-current','worktree-current',
        'run-lease-current','grant-current','full-current','claim-current','bootstrap-current',
        'worktree-current','run-lease-current','grant-current']
    assert events.count('full-current')==2 and events.count('identity-only')==14
    assert broker._closing_authority(request(),(11,os.geteuid(),101),captured,lease)==captured
    assert events.count('full-current')==4


@pytest.mark.parametrize('late',['claim','grant','worktree','bootstrap','run_lease','proof'])
@pytest.mark.parametrize('after_reserve',[False,True],ids=['before-reserve','after-reserve'])
def test_final_full_callback_native_drift_refuses_before_prepared_ack(monkeypatch,tmp_path,late,after_reserve):
    broker,events,state,captured,lease=batch_closing_fixture(monkeypatch,tmp_path,late=late,after_reserve=after_reserve)
    channel=Channel(frame(request())+frame(spawned_ack()))
    with pytest.raises(d.CheckedContextDispatchError):broker._handoff(channel)
    assert not channel.sent and not broker._wrapper_spawned and not broker._isolated_spawned
    assert not broker._pending
    assert ('reserve'in events)is after_reserve
    if after_reserve:
        assert broker._consumed and 'abort-nonrefund'not in events
        # Retirement is a separate original-owner boundary and never refunds.
        broker._retire(ambiguous=False)
        assert events.count('abort-nonrefund')==1


def test_attempt_ack_budget_is_checked_after_final_full_callback(monkeypatch,tmp_path):
    broker,events,state,captured,lease=batch_closing_fixture(monkeypatch,tmp_path)
    whole=broker._identity_current
    def expire_after_full():
        whole()
        if state['whole_in_closing']==2:lease.expires_at_ms=0
    broker._identity_current=expire_after_full
    channel=Channel(frame(request()))
    with pytest.raises(d.CheckedContextDispatchError,match='acknowledgement time'):broker._handoff(channel)
    assert not channel.sent and 'reserve'not in events and events[-1]=='grant-current'


def isolated_worktree_fixture(monkeypatch, mutation=None):
    """Real second handoff, authored kernel/helper authority; no UID transition."""
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    broker,events=mocked_broker(monkeypatch)
    broker._handoff(Channel(frame(request())+frame(spawned_ack())))
    command=['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry',*derived()[1:]]
    worker={'pid':44,'uid':1001,'parent_pid':33,'start_time_ticks':404,'process_birth_id':'birth:44',
        'command':['/opt/ipfs-supervisor/venv/bin/python','/opt/ipfs-supervisor/bin/worker-entry',*derived()[1:]]}
    namespaces={key:key+':[123]'for key in ('pid','mnt','net')}
    broker._scope.material_binding={'boundary':{'namespaces':namespaces}}
    observation={'schema':'checked-worker-kernel-worktree-observation@1',
        **{key:worker[key]for key in ('pid','uid','parent_pid','start_time_ticks','process_birth_id')},
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'worktree':request()['worktree'],'command_sha256':hashlib.sha256(d._wire(worker['command'])).hexdigest(),
        'namespaces':dict(namespaces),'metadata_only':True,'execution_authority':False,'proof_authority':False,
        'completion_authority':False,'source_body_executed':False,'provider_calls':0}
    def kernel(pid):
        if pid==22:return {'pid':22,'parent_pid':11,'start_time_ticks':202,'process_birth_id':'birth:22'}
        assert pid==worker['pid'];events.append('kernel-reread')
        current=deepcopy(worker)
        if mutation=='reread-birth':current['process_birth_id']='birth:reused'
        elif mutation=='reread-command':current['command'].append('--other-command')
        elif mutation=='reread-parent':current['parent_pid']+=1
        elif mutation=='reread-uid':current['uid']=1000
        return current
    broker._native._kernel_identity=kernel
    broker._native._isolated_kernel_birth=lambda **_: {'worker':worker,'sudo':{'process_birth_id':'birth:33'},
        'parent_chain':[{key:worker[key]for key in ('pid','uid','parent_pid','start_time_ticks','process_birth_id')},
            {'pid':33,'uid':1001,'parent_pid':22,'start_time_ticks':303,'process_birth_id':'birth:33'}]}
    def helper(*,worker_identity,expected_worktree,timeout_seconds=3.0):
        assert worker_identity==worker and worker_identity is not worker and expected_worktree==request()['worktree']
        assert 0 < timeout_seconds <= 3.0
        events.append('metadata-helper')
        if mutation=='helper-refusal':raise PermissionError('authored helper refusal')
        response=deepcopy(observation)
        if mutation in {'pid','start_time_ticks','uid','parent_pid'}:response[mutation]+=1
        elif mutation=='process_birth_id':response[mutation]='birth:other'
        elif mutation=='command_sha256':response[mutation]='0'*64
        elif mutation=='worktree':response[mutation]='/opt/worktrees/another-task'
        elif mutation=='boot_id':response[mutation]='another-kernel-boot'
        elif mutation=='namespaces':response[mutation]['pid']='pid:[124]'
        elif mutation=='provider_calls':response[mutation]=True
        elif mutation=='metadata_only':response[mutation]=False
        elif mutation=='authority':response['execution_authority']=True
        elif mutation=='source_body':response['source_body_executed']=True
        elif mutation=='schema':response[mutation]='another-observation'
        elif mutation=='boolean-pid':response['pid']=True
        elif mutation=='extra':response['extra']=True
        return response
    monkeypatch.setattr(deployment,'observe_worker_kernel_worktree',helper,raising=False)
    x={'schema':d.SCHEMA,'operation':'prepare_isolated','context_cid':'cid:scope','command':command,
        'worktree':request()['worktree'],'wrapper_ticket':'n'*32}
    acknowledgement={'schema':d.SCHEMA,'operation':'spawned','handoff_id':'n'*32,'pid':33,'process_birth_id':'birth:33'}
    return broker,events,x,Channel(frame(acknowledgement)),observation


def test_same_worker_uid_worktree_metadata_is_joined_then_kernel_identity_reread_before_release(monkeypatch):
    broker,events,x,channel,observation=isolated_worktree_fixture(monkeypatch)
    # Cross-UID cwd access is not performed by the owner. Kernel metadata and
    # namespace helper authority are authored mocks in this control-flow test.
    monkeypatch.setattr(d.os,'readlink',lambda *_:pytest.fail('owner attempted direct cross-UID cwd read'))
    broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
    output=Channel(bytes(channel.sent));assert d._receive(output)['phase']=='prepared'
    assert d._receive(output)==released()
    assert events.index('metadata-helper')<events.index('kernel-reread')
    assert broker._isolated_spawned and not broker._pending
    assert broker._observations[-1]['worker_worktree_observation']==observation
    observation['worktree']='cannot mutate the published observation'
    assert broker._observations[-1]['worker_worktree_observation']['worktree']==request()['worktree']


@pytest.mark.parametrize('mutation',['pid','start_time_ticks','process_birth_id','command_sha256','worktree',
    'uid','parent_pid','boot_id','namespaces','provider_calls','metadata_only','authority','source_body','schema',
    'boolean-pid','extra','helper-refusal','reread-birth','reread-command','reread-parent','reread-uid'])
def test_worktree_helper_or_posthelper_kernel_drift_cannot_release_existing_child(monkeypatch,mutation):
    broker,events,x,channel,observation=isolated_worktree_fixture(monkeypatch,mutation)
    with pytest.raises((d.CheckedContextDispatchError,PermissionError)):
        broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
    output=Channel(bytes(channel.sent));assert d._receive(output)['phase']=='prepared'
    assert not output.incoming and not broker._isolated_spawned and not broker._pending
    assert events.count('reserve')==1 and 'abort-nonrefund'not in events
    broker._retire(ambiguous=True)
    assert broker._ambiguous and events.count('abort-nonrefund')==1


@pytest.mark.parametrize('elapsed,helper_elapsed,accepted',[(0.1,0.1,True),(2.5,0.5,True),(3.9,0.2,False)])
def test_birth_and_worker_metadata_share_original_ack_deadline(monkeypatch,elapsed,helper_elapsed,accepted):
    broker,events,x,channel,_=isolated_worktree_fixture(monkeypatch)
    clock={'now':0.0};monkeypatch.setattr(d.time,'monotonic',lambda:clock['now'])
    kernel=broker._native._isolated_kernel_birth
    def born(**kwargs):
        result=kernel(**kwargs);clock['now']+=elapsed;return result
    broker._native._isolated_kernel_birth=born
    observe=broker._worker_worktree_current;budgets=[]
    def observed(*args,timeout_seconds):
        budgets.append(timeout_seconds)
        assert timeout_seconds==min(3.0,4.0-elapsed)
        result=observe(*args,timeout_seconds=timeout_seconds)
        clock['now']+=helper_elapsed
        return result
    broker._worker_worktree_current=observed
    if accepted:
        broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
        assert broker._isolated_spawned
        output=Channel(bytes(channel.sent));assert d._receive(output)['phase']=='prepared'
        assert d._receive(output)==released()
    else:
        with pytest.raises(d.CheckedContextDispatchError,match='original acknowledgement deadline'):
            broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
        assert not broker._isolated_spawned
        output=Channel(bytes(channel.sent));assert d._receive(output)['phase']=='prepared'and not output.incoming
        broker._retire(ambiguous=True)
        assert events.count('abort-nonrefund')==1
    assert len(budgets)==1 and events.count('reserve')==1


@pytest.mark.parametrize('expiry',['attempt','grant','run-lease'])
def test_release_native_expiry_is_rechecked_after_metadata_without_policy_callbacks(monkeypatch,expiry):
    broker,events,x,channel,_=isolated_worktree_fixture(monkeypatch)
    lease=SimpleNamespace(expires_at_ms=d.time.time_ns()//1_000_000+60_000)
    broker._execute_with_attempt_fence=lambda req,peer,callback:callback({'native':'mock'},lease)
    observed={'done':False};helper=broker._worker_worktree_current
    def metadata(*args,**kwargs):
        result=helper(*args,**kwargs);observed['done']=True
        if expiry=='attempt':lease.expires_at_ms=0
        return result
    broker._worker_worktree_current=metadata
    def run_lease():
        events.append('run-lease-current')
        d._need(not observed['done']or expiry!='run-lease','mock runtime lease expired after metadata')
    def grant(**_):
        events.append('grant-current')
        d._need(not observed['done']or expiry!='grant','mock native grant expired after metadata')
    broker._run_lease_current=run_lease;broker._native._require_native_grant_ack_budget=grant
    with pytest.raises(d.CheckedContextDispatchError,match='expired|acknowledgement time'):
        broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
    output=Channel(bytes(channel.sent));assert d._receive(output)['phase']=='prepared'and not output.incoming
    assert observed['done']and not broker._isolated_spawned and not broker._pending
    broker._retire(ambiguous=True)
    assert events.count('reserve')==events.count('abort-nonrefund')==1


def git_grant_fixture(monkeypatch,tmp_path,corrupt=None):
    """Real broker grant join, explicitly authored guard/permission authority."""
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    broker,events=mocked_broker(monkeypatch)
    repository=tmp_path/'canonical';git=repository/'.git'
    (git/'refs/heads').mkdir(parents=True)
    (git/'HEAD').write_text('ref: refs/heads/main\n');(git/'refs/heads/main').write_text('a'*40+'\n')
    broker._runtime=SimpleNamespace(manifest={'repository_root':str(repository)})
    broker._allocation=d._wire({'lifecycle':{'branch':'implementation/admitted-task','merge_target':'main'}})
    broker._identity_checkpoint=lambda:events.append('identity-only')
    receipt={'schema':'checked-worker-git-metadata-read-grant@1','worktree':request()['worktree'],
        'canonical_repository':str(repository),'selected_branch':'implementation/admitted-task',
        'selected_commit':'a'*40,'execution_authority':False,'proof_authority':False,'completion_authority':False,
        'source_body_executed':False,'parent_history_traversed':False,'packed_objects_supported':False,'provider_calls':0}
    if corrupt in {'selected_branch','selected_commit','worktree','canonical_repository'}:receipt[corrupt]='different'
    elif corrupt=='authority':receipt['execution_authority']=True
    elif corrupt=='history':receipt['parent_history_traversed']=True
    elif corrupt=='provider-bool':receipt['provider_calls']=False
    state={'valid':True,'closed':False}
    class Guard:
        @property
        def receipt(self):return deepcopy(receipt)
        def require_current(self):
            events.append('git-FD-current');d._need(state['valid']and not state['closed'],'mock retained Git custody changed')
        @contextmanager
        def readonly_query_environment(self):
            # Authored environment lifecycle fixture, not deployment authority.
            previous=os.environ.get('GIT_OPTIONAL_LOCKS');present='GIT_OPTIONAL_LOCKS'in os.environ
            events.append('git-env-enter');os.environ['GIT_OPTIONAL_LOCKS']='0'
            try:
                yield
                d._need(os.environ.get('GIT_OPTIONAL_LOCKS')=='0','mock readonly Git environment changed')
            finally:
                if present:os.environ['GIT_OPTIONAL_LOCKS']=previous
                else:os.environ.pop('GIT_OPTIONAL_LOCKS',None)
                events.append('git-env-restored')
        def close(self):events.append('git-close');state['closed']=True
    guard=Guard()
    def factory(*,worktree,canonical_repository):
        assert worktree==Path(request()['worktree']) and canonical_repository==repository
        events.append('git-read-grant')
        if corrupt=='foreign-type':return SimpleNamespace()
        return guard
    monkeypatch.setattr(deployment,'WorkerGitMetadataReadGuard',Guard)
    monkeypatch.setattr(deployment,'grant_worker_git_metadata_readonly',factory)
    broker._grant_worker_git_metadata_current=d.OwnerCheckedContextDispatchBroker._grant_worker_git_metadata_current.__get__(broker)
    broker._git_metadata_current=d.OwnerCheckedContextDispatchBroker._git_metadata_current.__get__(broker)
    broker._git_metadata_readonly_environment=d.OwnerCheckedContextDispatchBroker._git_metadata_readonly_environment.__get__(broker)
    return broker,events,guard,state,receipt


def test_git_grant_is_joined_to_current_canonical_commit_and_exact_native_allocation(monkeypatch,tmp_path):
    broker,events,guard,state,receipt=git_grant_fixture(monkeypatch,tmp_path)
    assert broker._grant_worker_git_metadata_current(request()['worktree'])==receipt
    assert broker._git_metadata_guard is guard and events[-1]=='git-FD-current'
    with pytest.raises(d.CheckedContextDispatchError,match='one native'):broker._grant_worker_git_metadata_current(request()['worktree'])
    receipt['selected_branch']='late mutation'
    with pytest.raises(d.CheckedContextDispatchError,match='rebound'):broker._git_metadata_current()


@pytest.mark.parametrize('corrupt',['selected_branch','selected_commit','worktree','canonical_repository',
    'authority','history','provider-bool','foreign-type'])
def test_git_grant_wrong_native_join_or_authority_cannot_reach_worker_ack(monkeypatch,tmp_path,corrupt):
    broker,events,guard,state,receipt=git_grant_fixture(monkeypatch,tmp_path,corrupt)
    with pytest.raises(d.CheckedContextDispatchError):broker._grant_worker_git_metadata_current(request()['worktree'])
    assert not broker._isolated_spawned and 'git-FD-current'not in events
    if corrupt!='foreign-type':
        assert broker._git_metadata_guard is guard  # Retained for fenced STOP even on join failure.
        broker._close_git_metadata_after_stop()
        assert state['closed']and events[-1]=='git-close'


@pytest.mark.parametrize('drift',['copy','remove','identity','receipt'])
def test_original_git_guard_rebinding_cannot_replace_retained_custody(monkeypatch,tmp_path,drift):
    broker,events,guard,state,receipt=git_grant_fixture(monkeypatch,tmp_path)
    broker._grant_worker_git_metadata_current(request()['worktree'])
    if drift=='copy':broker._git_metadata_guard=copy(guard)
    elif drift=='remove':broker._git_metadata_guard=None
    elif drift=='identity':broker._git_metadata_id+=1
    else:receipt['selected_commit']='b'*40
    with pytest.raises(d.CheckedContextDispatchError):broker._git_metadata_current()


@pytest.mark.parametrize('phase',['final-native-callback','observed-birth'])
def test_retained_git_drift_refuses_before_worker_ack_or_release(monkeypatch,tmp_path,phase):
    broker,events,x,channel,_=isolated_worktree_fixture(monkeypatch)
    guarded,_,guard,state,receipt=git_grant_fixture(monkeypatch,tmp_path)
    for name in ('_runtime','_allocation','_identity_checkpoint','_git_metadata_guard','_git_metadata_original',
        '_git_metadata_type','_git_metadata_id','_git_metadata_wire','_git_metadata_closed'):
        setattr(broker,name,getattr(guarded,name))
    broker._grant_worker_git_metadata_current=d.OwnerCheckedContextDispatchBroker._grant_worker_git_metadata_current.__get__(broker)
    broker._git_metadata_current=d.OwnerCheckedContextDispatchBroker._git_metadata_current.__get__(broker)
    current=broker._closing_authority;calls=[]
    def closing(*args):
        calls.append('native-close');result=current(*args)
        if phase=='final-native-callback'and len(calls)==2:state['valid']=False
        return result
    broker._closing_authority=closing
    observe=broker._worker_worktree_current
    def observed(*args,**kwargs):
        result=observe(*args,**kwargs)
        if phase=='observed-birth':state['valid']=False
        return result
    broker._worker_worktree_current=observed
    with pytest.raises(d.CheckedContextDispatchError,match='retained Git'):broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
    assert not broker._isolated_spawned and not broker._pending
    if phase=='final-native-callback':assert not channel.sent
    else:
        output=Channel(bytes(channel.sent));assert d._receive(output)['phase']=='prepared' and not output.incoming
    broker._retire(ambiguous=phase=='observed-birth')
    assert events.count('reserve')==1 and events.count('abort-nonrefund')==1


def isolated_readonly_query_fixture(monkeypatch, tmp_path):
    broker, events, request_value, channel, _ = isolated_worktree_fixture(monkeypatch)
    guarded, grant_events, guard, state, receipt = git_grant_fixture(monkeypatch, tmp_path)
    for name in ('_runtime','_allocation','_identity_checkpoint','_git_metadata_guard','_git_metadata_original',
        '_git_metadata_type','_git_metadata_id','_git_metadata_wire','_git_metadata_closed'):
        setattr(broker,name,getattr(guarded,name))
    for name in ('_grant_worker_git_metadata_current','_git_metadata_current','_git_metadata_readonly_environment'):
        setattr(broker,name,getattr(d.OwnerCheckedContextDispatchBroker,name).__get__(broker))
    capture = broker._scope.capture_allocated
    current = broker._scope.require_current
    closing = broker._closing_authority
    def captured():
        assert os.environ['GIT_OPTIONAL_LOCKS'] == '0';capture()
    def scope_current():
        assert os.environ['GIT_OPTIONAL_LOCKS'] == '0';current()
    calls = []
    def closed(*args):
        calls.append('closing')
        if len(calls) == 2:assert os.environ['GIT_OPTIONAL_LOCKS'] == '0'
        return closing(*args)
    broker._scope.capture_allocated = captured
    broker._scope.require_current = scope_current
    broker._closing_authority = closed
    return broker, events, request_value, channel, guard, grant_events, calls


@pytest.mark.parametrize('previous',[None,'0','1'])
def test_postgrant_snapshot_queries_use_optional_lock_prevention_and_restore_before_ack(monkeypatch,tmp_path,previous):
    """Actual broker sequencing; native grant/kernel/proof authority is mocked."""
    if previous is None:monkeypatch.delenv('GIT_OPTIONAL_LOCKS',raising=False)
    else:monkeypatch.setenv('GIT_OPTIONAL_LOCKS',previous)
    broker, events, x, channel, guard, grant_events, calls = isolated_readonly_query_fixture(monkeypatch,tmp_path)
    send = d._send
    def checked_send(target,value):
        assert os.environ.get('GIT_OPTIONAL_LOCKS') == previous
        assert grant_events[-1] in {'git-env-restored','git-FD-current'}
        assert 'git-env-restored' in grant_events
        return send(target,value)
    monkeypatch.setattr(d,'_send',checked_send)
    broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
    assert os.environ.get('GIT_OPTIONAL_LOCKS') == previous
    output = Channel(bytes(channel.sent));assert d._receive(output)['phase'] == 'prepared'
    assert d._receive(output) == released() and broker._isolated_spawned
    assert len(calls) == 2 and grant_events.count('git-env-enter') == grant_events.count('git-env-restored') == 1
    assert events.count('reserve') == 1


@pytest.mark.parametrize('failure',['callback-exception','environment-drift','guard-rebound','STOP'])
def test_postgrant_query_failure_restores_exact_prior_environment_before_refusal_and_nonrefund(monkeypatch,tmp_path,failure):
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS','prior exact value')
    broker, events, x, channel, guard, grant_events, calls = isolated_readonly_query_fixture(monkeypatch,tmp_path)
    original = broker._closing_authority
    def callback(*args):
        result = original(*args)
        if len(calls) == 2:
            if failure == 'callback-exception':raise d.CheckedContextDispatchError('mock source/proof refusal')
            if failure == 'environment-drift':os.environ['GIT_OPTIONAL_LOCKS'] = '1'
            if failure == 'guard-rebound':broker._git_metadata_guard = copy(guard)
            if failure == 'STOP':broker._stop.set()
        return result
    broker._closing_authority = callback
    with pytest.raises(d.CheckedContextDispatchError):broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
    assert os.environ['GIT_OPTIONAL_LOCKS'] == 'prior exact value'
    assert grant_events.count('git-env-restored') == 1
    assert not channel.sent and not broker._isolated_spawned and not broker._pending
    broker._retire(ambiguous=True)
    assert events.count('reserve') == events.count('abort-nonrefund') == 1


@pytest.mark.parametrize('expiry',['attempt','grant','run-lease'])
def test_native_ack_expiry_is_rechecked_after_query_restoration_without_policy_callbacks(monkeypatch,tmp_path,expiry):
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS','1')
    broker, events, x, channel, guard, grant_events, calls = isolated_readonly_query_fixture(monkeypatch,tmp_path)
    lease = SimpleNamespace(expires_at_ms=d.time.time_ns()//1_000_000+60_000)
    broker._execute_with_attempt_fence=lambda req,peer,callback:callback({'native':'mock'},lease)
    original_query = guard.readonly_query_environment
    restored = {'done':False}
    @contextmanager
    def restored_query():
        with original_query():yield
        restored['done']=True
        if expiry=='attempt':lease.expires_at_ms=0
    guard.readonly_query_environment=restored_query
    def run_lease():
        assert restored['done'] and os.environ['GIT_OPTIONAL_LOCKS']=='1'
        events.append('run-lease-current')
        d._need(expiry!='run-lease','mock run lease expired after restore')
    def grant(**_):
        assert restored['done'] and os.environ['GIT_OPTIONAL_LOCKS']=='1'
        events.append('grant-current')
        d._need(expiry!='grant','mock native grant expired after restore')
    broker._run_lease_current=run_lease
    broker._native._require_native_grant_ack_budget=grant
    with pytest.raises(d.CheckedContextDispatchError,match='expired|acknowledgement time'):
        broker._isolated_handoff(channel,x,(22,os.geteuid(),202))
    assert not channel.sent and not broker._isolated_spawned and not broker._pending
    assert os.environ['GIT_OPTIONAL_LOCKS']=='1' and len(calls)==2
    broker._retire(ambiguous=True)
    assert events.count('reserve')==events.count('abort-nonrefund')==1


def test_refusal_call_locations_are_bounded_and_exclude_sensitive_exception_context():
    def sensitive_failure(depth):
        private_local='not public credential material'
        if depth:return sensitive_failure(depth-1)
        raise RuntimeError('unexportable exception message '+private_local)
    try:sensitive_failure(24)
    except RuntimeError as error:
        frames=d._public_failure_frames(error)
    assert len(frames)==16 and all(set(row)=={'file','function','line'}for row in frames)
    assert all(row['file']==Path(__file__).name and row['function']=='sensitive_failure'
        and type(row['line'])is int and row['line']>0 for row in frames)
    wire=d._wire(frames)
    assert b'private_local'not in wire and b'credential'not in wire and b'unexportable'not in wire
    assert all(len(row['file'])<=256 and len(row['function'])<=128 for row in frames)


def git_drift_diagnostic_fixture(monkeypatch, tmp_path):
    """Source-owned error schema, with explicitly mocked original grant authority."""
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    broker, events, guard, state, receipt = git_grant_fixture(monkeypatch, tmp_path)
    relative = '.git/worktrees/allocated'
    receipt['files'] = [{'path':receipt['worktree']+'/.git','kind':'metadata',
                         'sha256':'a'*64,'granted_mode':0o440}]
    receipt['directories'] = [{'path':str(Path(receipt['canonical_repository'])/relative),
                              'granted_mode':0o750}]
    broker._grant_worker_git_metadata_current(request()['worktree'])
    expected = dict(zip(('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink',
                         'st_size','st_mtime_ns','st_ctime_ns'),
                        (1,2,0o40750,1000,1000,2,4096,100,101)))
    current = {**expected,'st_mtime_ns':102,'st_ctime_ns':103}
    diagnostic = {'schema':'checked-worker-git-metadata-drift@1','phase':'retained',
        'path_role':'canonical_git_metadata','relative_path':relative,'kind':'directory',
        'changed_fields':['st_ctime_ns','st_mtime_ns'],'expected_stat':expected,
        'current_fd_stat':current,'current_path_stat':dict(current),'current_fresh_stat':None,
        'expected_sha256':None,'observed_sha256':None,'canonical_path_changed':False,
        'metadata_only':True,'execution_authority':False,'proof_authority':False,
        'completion_authority':False,'source_body_executed':False,'provider_calls':0}
    error = deployment.WorkerWorktreeObservationError('private exception text must stay hashed')
    error.git_metadata_drift = diagnostic
    return broker, events, guard, receipt, error, diagnostic


def test_selected_source_owned_git_drift_is_detached_and_has_no_authority_callback(monkeypatch, tmp_path):
    broker, events, guard, receipt, error, diagnostic = git_drift_diagnostic_fixture(monkeypatch, tmp_path)
    before = list(events)
    exported = broker._public_git_metadata_drift(error)
    assert exported == diagnostic and events == before
    exported['current_fd_stat']['st_ino'] = 999
    assert diagnostic['current_fd_stat']['st_ino'] == 2
    assert not broker._isolated_spawned and b'private exception' not in d._wire(exported)


@pytest.mark.parametrize('mutation',[
    'foreign-error','subclass-error','missing-diagnostic','extra','schema-list','phase-list','relative-absolute',
    'relative-parent','relative-control','unselected-path','wrong-role','wrong-kind','stat-extra',
    'stat-boolean','stat-negative','stat-huge','missing-expected-stat','wrong-granted-mode','fields-list',
    'fields-unsorted','fields-inconsistent','canonical-inconsistent','authority','provider-boolean',
    'hash-malformed','directory-hash','closed-guard','guard-copy','receipt-mutation'])
def test_invalid_or_foreign_git_drift_cannot_be_exported_or_replace_refusal(monkeypatch, tmp_path, mutation):
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    broker, events, guard, receipt, error, diagnostic = git_drift_diagnostic_fixture(monkeypatch, tmp_path)
    if mutation == 'foreign-error':
        class Foreign(RuntimeError):
            @property
            def git_metadata_drift(self):pytest.fail('foreign diagnostic getter was accessed')
        error = Foreign('private text')
    elif mutation == 'subclass-error':
        class Foreign(deployment.WorkerWorktreeObservationError):pass
        error = Foreign();error.git_metadata_drift = diagnostic
    elif mutation == 'missing-diagnostic':del error.git_metadata_drift
    elif mutation == 'extra':diagnostic['private_body'] = 'never export'
    elif mutation == 'schema-list':diagnostic['schema'] = []
    elif mutation == 'phase-list':diagnostic['phase'] = []
    elif mutation == 'relative-absolute':diagnostic['relative_path'] = '/private/owner.duckdb'
    elif mutation == 'relative-parent':diagnostic['relative_path'] = '.git/../../private'
    elif mutation == 'relative-control':diagnostic['relative_path'] = '.git/worktrees/\nprivate'
    elif mutation == 'unselected-path':diagnostic['relative_path'] = '.git/private-state'
    elif mutation == 'wrong-role':diagnostic['path_role'] = 'owner_database'
    elif mutation == 'wrong-kind':diagnostic['kind'] = 'objects'
    elif mutation == 'stat-extra':diagnostic['current_fd_stat']['private_body'] = 'never export'
    elif mutation == 'stat-boolean':diagnostic['current_fd_stat']['st_ino'] = True
    elif mutation == 'stat-negative':diagnostic['current_fd_stat']['st_ino'] = -1
    elif mutation == 'stat-huge':diagnostic['current_fd_stat']['st_ino'] = 2**128
    elif mutation == 'missing-expected-stat':diagnostic['expected_stat'] = None
    elif mutation == 'wrong-granted-mode':diagnostic['expected_stat']['st_mode'] = 0o40777
    elif mutation == 'fields-list':diagnostic['changed_fields'] = {'st_mtime_ns'}
    elif mutation == 'fields-unsorted':diagnostic['changed_fields'].reverse()
    elif mutation == 'fields-inconsistent':diagnostic['changed_fields'] = ['st_ino']
    elif mutation == 'canonical-inconsistent':diagnostic['canonical_path_changed'] = True
    elif mutation == 'authority':diagnostic['proof_authority'] = True
    elif mutation == 'provider-boolean':diagnostic['provider_calls'] = False
    elif mutation == 'hash-malformed':diagnostic['observed_sha256'] = 'private text'
    elif mutation == 'directory-hash':diagnostic['expected_sha256'] = 'a'*64
    elif mutation == 'closed-guard':broker._git_metadata_closed = True
    elif mutation == 'guard-copy':broker._git_metadata_guard = copy(guard)
    elif mutation == 'receipt-mutation':receipt['selected_commit'] = 'b'*40
    before = list(events)
    assert broker._public_git_metadata_drift(error) is None and events == before
    assert not broker._isolated_spawned


def test_selected_pointer_hash_drift_joins_original_selected_file(monkeypatch, tmp_path):
    broker, events, guard, receipt, error, diagnostic = git_drift_diagnostic_fixture(monkeypatch, tmp_path)
    expected = {**diagnostic['expected_stat'],'st_mode':0o100440,'st_size':32}
    diagnostic.update(phase='retained_bytes',path_role='allocated_git_pointer',relative_path='.git',
        kind='metadata',changed_fields=['sha256'],expected_stat=expected,current_fd_stat=dict(expected),
        current_path_stat=dict(expected),expected_sha256='a'*64,observed_sha256='b'*64)
    assert broker._public_git_metadata_drift(error) == diagnostic
    diagnostic['expected_sha256'] = 'c'*64
    assert broker._public_git_metadata_drift(error) is None


def test_real_selected_directory_fd_drift_diagnostic_matches_broker_export_schema(monkeypatch, tmp_path):
    """Real FD/stat/error emitter; internal fixture is not a native grant claim."""
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    repository = tmp_path/'real-canonical';selected = repository/'.git'
    selected.mkdir(parents=True);selected.chmod(0o750)
    descriptor = os.open(selected,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    original = deployment._git_stat(os.fstat(descriptor))
    receipt = {'worktree':request()['worktree'],'canonical_repository':str(repository),
        'files':[],'directories':[{'path':str(selected),'granted_mode':0o750}]}
    guard = deployment.WorkerGitMetadataReadGuard(
        [(str(selected),descriptor,original,original,None,'directory')],receipt,
        _key=deployment._GIT_METADATA_READ_KEY)
    broker, events = mocked_broker(monkeypatch)
    broker._git_metadata_guard = broker._git_metadata_original = guard
    broker._git_metadata_type = type(guard);broker._git_metadata_id = id(guard)
    broker._git_metadata_wire = d._wire(guard.receipt)
    try:
        os.utime(selected,ns=(original[7],original[7]+1_000_000))
        with pytest.raises(deployment.WorkerWorktreeObservationError) as caught:
            guard.require_current()
        exported = broker._public_git_metadata_drift(caught.value)
        assert exported == caught.value.git_metadata_drift
        assert exported['relative_path'] == '.git' and 'st_mtime_ns' in exported['changed_fields']
        assert exported['expected_stat']['st_mode'] & 0o7777 == receipt['directories'][0]['granted_mode']
        assert exported['metadata_only'] is True and exported['execution_authority'] is False
        assert exported['proof_authority'] is exported['completion_authority'] is False
    finally:
        guard.close()


def test_real_refusal_transport_retains_nonrefund_and_exports_only_selected_git_diagnostic(monkeypatch, tmp_path):
    broker, events, guard, receipt, error, diagnostic = git_drift_diagnostic_fixture(monkeypatch, tmp_path)
    owner, client = socket.socketpair()
    broker._listener = SimpleNamespace(accept=lambda:(owner,None))
    broker._active_channel = broker._active_channel_identity = None
    broker._boundary = 'isolated_worker'
    broker._wrapper_spawned = True
    def refuse(channel):
        broker._active_charged = broker._pending = True
        broker._stop.set()
        raise error
    broker._handoff = refuse
    try:
        broker._serve()
        assert d._receive(client) == {'schema':d.SCHEMA,'ok':False,'error':'checked_dispatch_refused'}
        observation = broker.observations[-1]
        assert observation['status'] == 'refused' and observation['git_metadata_drift'] == diagnostic
        assert observation['message_sha256'] == hashlib.sha256(str(error).encode()).hexdigest()
        assert observation['unacknowledged_child_birth_possible'] is True
        assert events.count('abort-nonrefund') == 1 and not broker._isolated_spawned
        assert broker._active_channel is None and not broker._pending
        assert all(observation[key] is False for key in d._FALSE)
        assert b'private exception' not in d._wire(observation)
    finally:
        owner.close();client.close()


def worker_metadata_failure_fixture(monkeypatch):
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    broker,events=mocked_broker(monkeypatch);broker._worker_metadata_timeout=1.25
    flags={'metadata_only':True,'execution_authority':False,'proof_authority':False,
        'completion_authority':False,'source_body_executed':False,'provider_calls':0}
    child={'schema':'checked-worker-kernel-worktree-refusal@1','stage':'target_status','error_type':'SystemExit',
        'message_sha256':'b'*64,'failure_frames':[{'file':'worker-entry','function':'observe','line':24}],**flags}
    raw=d._wire(child)+b'\n';helper=deployment.WORKER_ENTRY.encode()
    value={'schema':'checked-worker-kernel-worktree-helper-failure@1','phase':'subprocess_response',
        'returncode':1,'stdout_bytes':len(raw),'stdout_sha256':hashlib.sha256(raw).hexdigest(),
        'helper_sha256':hashlib.sha256(helper).hexdigest(),'helper_bytes':len(helper),'timeout_seconds':1.25,
        'helper_refusal':child,**flags}
    error=deployment.WorkerWorktreeObservationError('private exception stdout must not be copied')
    error.worker_worktree_failure=value
    return broker,events,error,value


def test_source_owned_worker_helper_failure_is_bounded_detached_and_never_a_success(monkeypatch):
    broker,events,error,value=worker_metadata_failure_fixture(monkeypatch)
    before=list(events);exported=broker._public_worker_worktree_failure(error)
    assert exported==value and events==before and not broker._isolated_spawned
    exported['helper_refusal']['failure_frames'][0]['line']=999
    assert value['helper_refusal']['failure_frames'][0]['line']==24
    assert b'private exception'not in d._wire(exported)


@pytest.mark.parametrize('mutation',['foreign','extra','helper-pin','helper-size','timeout','boolean-returncode',
    'negative-size','child-extra','child-stage','child-authority','child-frame-path','child-too-many-frames',
    'stdout-count','stdout-hash','successful-child','timeout-child','authority','provider-boolean'])
def test_invalid_helper_failure_or_foreign_error_cannot_be_exported(monkeypatch,mutation):
    broker,events,error,value=worker_metadata_failure_fixture(monkeypatch)
    if mutation=='foreign':
        class Foreign(RuntimeError):
            @property
            def worker_worktree_failure(self):pytest.fail('foreign getter accessed')
        error=Foreign()
    elif mutation=='extra':value['private_body']='never export'
    elif mutation=='helper-pin':value['helper_sha256']='a'*64
    elif mutation=='helper-size':value['helper_bytes']+=1
    elif mutation=='timeout':value['timeout_seconds']=3.0
    elif mutation=='boolean-returncode':value['returncode']=True
    elif mutation=='negative-size':value['stdout_bytes']=-1
    elif mutation=='child-extra':value['helper_refusal']['private_body']='never export'
    elif mutation=='child-stage':value['helper_refusal']['stage']='private-owner-file'
    elif mutation=='child-authority':value['helper_refusal']['execution_authority']=True
    elif mutation=='child-frame-path':value['helper_refusal']['failure_frames'][0]['file']='/private/secret'
    elif mutation=='child-too-many-frames':value['helper_refusal']['failure_frames']*=17
    elif mutation=='stdout-count':value['stdout_bytes']+=1
    elif mutation=='stdout-hash':value['stdout_sha256']='a'*64
    elif mutation=='successful-child':value['returncode']=0
    elif mutation=='timeout-child':value.update(phase='subprocess_timeout',returncode=None)
    elif mutation=='authority':value['proof_authority']=True
    elif mutation=='provider-boolean':value['provider_calls']=False
    before=list(events);assert broker._public_worker_worktree_failure(error)is None and events==before
    assert not broker._isolated_spawned


def test_worker_helper_timeout_metadata_is_not_a_child_refusal_or_live_birth(monkeypatch):
    broker,events,error,value=worker_metadata_failure_fixture(monkeypatch)
    value.update(phase='subprocess_timeout',returncode=None,stdout_bytes=0,
        stdout_sha256=hashlib.sha256(b'').hexdigest(),helper_refusal=None)
    assert broker._public_worker_worktree_failure(error)==value and not broker._isolated_spawned


def owner_task_verifier_fixture(monkeypatch,corrupt=None):
    """Actual native verifier; task transport/signatures are authored mock data."""
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as native
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository,database_task_source
    calls=[];lock=d.threading.RLock();connection=object()
    expected=SimpleNamespace(task_cid='mock-task-cid',goal_cid='mock-goal')
    contract={'intent_owner_id':'mock-owner','task_key':'TASK','dependencies':[],'plan_id':'mock-plan'}
    envelope={'explicit_mock_signature':True}
    row={'body':{native.CONTRACT_KEY:envelope},'identity':{'local_contract_cid':native.content_identity(envelope)},
        'task_alias':'TASK','dependencies':[],'goal_cid':'mock-goal','plan_cid':'mock-plan'}
    task=SimpleNamespace(task_cid=expected.task_cid,task_alias='TASK',dependencies=[],body=row['body'])
    page=SimpleNamespace(tasks=[task],next_cursor='')
    if corrupt=='population':page=SimpleNamespace(tasks=[],next_cursor='')
    elif corrupt=='contract':contract['plan_id']='foreign-plan'
    elif corrupt=='identity':row['identity']['local_contract_cid']='foreign-cid'
    elif corrupt=='dependencies':row['dependencies']=['foreign-dependency']
    elif corrupt=='goal':row['goal_cid']='foreign-goal'
    elif corrupt=='plan':row['plan_cid']='foreign-plan'
    class Intent:
        def __init__(self,*,bound_connection,install_schema):
            assert bound_connection is connection and install_schema is False and lock._is_owned()
        def get_task(self,key):assert key==expected.task_cid;return row
    class LocalSource:
        def __init__(self,*,intent,install_schema):assert type(intent)is Intent and install_schema is False
        def list_tasks(self,*,limit):
            assert limit==17 and lock._is_owned();calls.append('local');return page
    def remote(*,limit):
        assert limit==17 and not lock._is_owned(), 'remote source cannot service RPC under owner lock'
        calls.append('remote');return page
    monkeypatch.setattr(intent_repository,'IntentRepository',Intent)
    monkeypatch.setattr(database_task_source,'DatabaseTaskSource',LocalSource)
    monkeypatch.setattr(native,'_verify_signature',lambda value,profile:dict(contract))
    monkeypatch.setattr(native,'_pending_contract_payload',lambda **kwargs:{
        'intent_owner_id':'mock-owner','task_key':'TASK','dependencies':[],'plan_id':'mock-plan'})
    runtime=SimpleNamespace(server=SimpleNamespace(_connection=connection,_lock=lock),
        source=SimpleNamespace(list_tasks=remote),admission={'explicit_mock_admission':True})
    verified={'graph':SimpleNamespace(tasks=[expected]),'profile':{'explicit_mock_profile':True}}
    return native,runtime,verified,calls,lock


@pytest.mark.parametrize('corrupt',[None,'population','contract','identity','dependencies','goal','plan'])
def test_real_native_task_verifier_avoids_rpc_under_owner_lock_and_keeps_all_contract_checks(monkeypatch,corrupt):
    native,runtime,verified,calls,lock=owner_task_verifier_fixture(monkeypatch,corrupt)
    with lock:
        if corrupt:
            with pytest.raises(ValueError):native.AdmittedBenchmarkRuntime._verify_tasks(runtime,verified,owner_local=True)
        else:native.AdmittedBenchmarkRuntime._verify_tasks(runtime,verified,owner_local=True)
    assert calls==['local']


def test_real_native_task_verifier_legacy_remote_path_stays_outside_owner_lock(monkeypatch):
    native,runtime,verified,calls,lock=owner_task_verifier_fixture(monkeypatch)
    native.AdmittedBenchmarkRuntime._verify_tasks(runtime,verified)
    assert calls==['remote']


@pytest.mark.parametrize('selected',[None,0,1,'true'])
def test_real_native_task_verifier_requires_exact_boolean_mode_before_any_task_access(monkeypatch,selected):
    native,runtime,verified,calls,lock=owner_task_verifier_fixture(monkeypatch)
    with pytest.raises(ValueError,match='exact owner-local'):
        native.AdmittedBenchmarkRuntime._verify_tasks(runtime,verified,owner_local=selected)
    assert calls==[]


def test_nonblocking_diagnostic_samples_original_thread_locations_while_all_authority_locks_are_held():
    """Real held locks/thread; original native authority explicitly mocked."""
    ready=d.threading.Event();release=d.threading.Event()
    locks=[d.threading.Lock()for _ in range(3)]
    broker=object.__new__(d.OwnerCheckedContextDispatchBroker)
    broker._seal=d._SEAL;broker._pid=os.getpid();broker._closed=False;broker._context='mock-context'
    broker._pending=broker._consumed=broker._wrapper_spawned=True
    broker._isolated_spawned=broker._ambiguous=broker._prepared_ack=False
    broker._boundary='isolated_worker';broker._lock=locks[0]
    broker._runtime=SimpleNamespace(_verify=lambda **_:pytest.fail('diagnostic invoked live verification'))
    broker._scope=SimpleNamespace(require_current=lambda:pytest.fail('diagnostic invoked proof authority'))
    def locked_owner():
        with locks[0],locks[1],locks[2],d._REGISTRY_LOCK:
            ready.set();release.wait(5)
    with d._REGISTRY_LOCK:d._BROKERS[id(broker)]=broker
    broker._thread=d.threading.Thread(target=locked_owner);broker._thread.start();assert ready.wait(2)
    try:
        # Another thread holds registry and all authority locks concurrently.
        value=broker.diagnostic_snapshot()
        assert value['pending']is value['thread_alive']is value['diagnostic_only']is True
        assert value['boundary']=='isolated_worker' and value['runtime_current_checked']is False
        assert value['source_current_checked']is value['grant_current_checked']is False
        assert 1<=len(value['thread_frames'])<=16 and any(x['function']=='locked_owner'for x in value['thread_frames'])
        assert all(set(x)=={'file','function','line'}for x in value['thread_frames'])
        assert all(value[k]is False for k in d._FALSE)
        with pytest.raises(d.CheckedContextDispatchError):copy(broker).diagnostic_snapshot()
        broker._closed=True
        with pytest.raises(d.CheckedContextDispatchError):broker.diagnostic_snapshot()
    finally:
        release.set();broker._thread.join(2)
        with d._REGISTRY_LOCK:d._BROKERS.pop(id(broker),None)


def blocked_socket_broker(entered,done):
    """Real transport, authored native STOP predicate and reservation ledger."""
    directory=tempfile.TemporaryDirectory(prefix='aa-drain-',dir='/tmp');path=Path(directory.name)/'owner.sock'
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(path));path.chmod(0o600)
    listener.listen(1);listener.settimeout(0.2)
    broker=object.__new__(d.OwnerCheckedContextDispatchBroker)
    broker._seal=d._SEAL;broker._pid=os.getpid();broker._closed=False;broker._stop=d.threading.Event();broker._lock=d.threading.RLock()
    broker._listener=listener;broker._original_socket_path=path;info=path.lstat();broker._socket_identity=(info.st_dev,info.st_ino)
    broker._active_channel=broker._active_channel_identity=None
    broker._pending=False;broker._consumed=True;broker._wrapper_spawned=True;broker._isolated_spawned=False
    broker._ambiguous=broker._prepared_ack=False;broker._observations=[];broker._boundary='isolated_worker';broker._context='mock-context'
    state={'stopped':False,'status':'reserved','charged_tokens':123,'aborts':0}
    runtime=SimpleNamespace(_context_refresh_stopped=lambda:state['stopped'])
    def abort():state.update(status='aborted',aborts=state['aborts']+1)
    scope=SimpleNamespace(_runtime=runtime,bound_runtime=runtime,abort_pending=abort)
    broker._runtime,broker._scope=runtime,scope;broker._fields=(scope,runtime)
    def handoff(channel):
        broker._pending=True;broker._active_charged=True;entered.set()
        try:d._receive(channel)  # Wait for a real, absent acknowledgement.
        finally:broker._pending=False;done.set()
    broker._handoff=handoff
    with d._REGISTRY_LOCK:d._BROKERS[id(broker)]=broker
    broker._thread=d.threading.Thread(target=broker._serve);broker._thread.start()
    client=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);client.connect(str(path));assert entered.wait(2)
    return broker,state,client,directory,path


def test_native_stop_drains_real_blocked_ack_without_waiving_pending_or_refunding():
    entered,done=d.threading.Event(),d.threading.Event();broker,state,client,directory,path=blocked_socket_broker(entered,done)
    try:
        with pytest.raises(d.CheckedContextDispatchError,match='native STOP'):broker.close_after_native_stop()
        assert broker._pending and path.exists() and state['status']=='reserved'
        state['stopped']=True;receipt=broker.close_after_native_stop()
        assert done.is_set() and not broker._thread.is_alive() and not broker._pending
        assert broker._active_channel is None and not path.exists()
        assert receipt['socket_removed']is receipt['thread_joined']is True
        assert state['status']=='aborted' and state['charged_tokens']==123
        before=dict(state);assert broker.close_after_native_stop()==receipt and state==before
        assert all(receipt[k]is False for k in d._FALSE)
        assert not broker.worker_birth_observed
    finally:
        client.close();broker._stop.set();broker._listener.close();broker._thread.join(2)
        with d._REGISTRY_LOCK:d._BROKERS.pop(id(broker),None)
        directory.cleanup()


@pytest.mark.parametrize('abort_error',[False,True])
def test_original_git_custody_closes_only_after_native_stop_even_when_budget_abort_raises(monkeypatch,tmp_path,abort_error):
    """Real cancellation/close order; native STOP and Git grant are mock authority."""
    entered,done=d.threading.Event(),d.threading.Event()
    broker,state,client,directory,path=blocked_socket_broker(entered,done)
    try:
        guarded,events,guard,git_state,receipt=git_grant_fixture(monkeypatch,tmp_path)
        guarded._grant_worker_git_metadata_current(request()['worktree'])
        for name in ('_git_metadata_guard','_git_metadata_original','_git_metadata_type','_git_metadata_id',
                     '_git_metadata_wire','_git_metadata_closed'):
            setattr(broker,name,getattr(guarded,name))
        original_abort=broker._scope.abort_pending
        def abort():
            if abort_error:raise RuntimeError('authored budget abort refusal')
            original_abort()
        broker._scope.abort_pending=abort
        with pytest.raises(d.CheckedContextDispatchError,match='native STOP'):broker.close_after_native_stop()
        assert not git_state['closed']and 'git-close'not in events
        state['stopped']=True
        if abort_error:
            with pytest.raises(RuntimeError,match='budget abort'):broker.close_after_native_stop()
            assert git_state['closed']and not broker._thread.is_alive()and path.exists()
            broker._scope.abort_pending=original_abort
        result=broker.close_after_native_stop()
        assert git_state['closed']and result['git_metadata_custody_closed']is True
        assert result['git_metadata_permission_reversal']is False and state['charged_tokens']==123
        assert state['status']=='aborted'and not path.exists()
    finally:
        client.close();broker._stop.set();broker._listener.close();broker._thread.join(2)
        with d._REGISTRY_LOCK:d._BROKERS.pop(id(broker),None)
        directory.cleanup()


def test_stop_drain_failure_preserves_socket_and_retries_original_cleanup(monkeypatch):
    entered,done=d.threading.Event(),d.threading.Event();broker,state,client,directory,path=blocked_socket_broker(entered,done)
    try:
        state['stopped']=True;original=broker._thread.join
        monkeypatch.setattr(broker._thread,'join',lambda **_:None)
        monkeypatch.setattr(broker._thread,'is_alive',lambda:True)
        with pytest.raises(d.CheckedContextDispatchError,match='not closed safely'):broker.close_after_native_stop()
        assert path.exists() and not broker._closed  # No unlink/qualified close while thread may live.
        assert done.wait(2)
        monkeypatch.setattr(broker._thread,'join',original)
        monkeypatch.setattr(broker._thread,'is_alive',lambda:False)
        receipt=broker.close_after_native_stop()
        assert receipt['thread_joined']is True and not path.exists() and state['charged_tokens']==123
    finally:
        client.close();broker._stop.set();broker._listener.close();original(timeout=2)
        with d._REGISTRY_LOCK:d._BROKERS.pop(id(broker),None)
        directory.cleanup()
