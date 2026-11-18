"""Protocol/refusal units with explicitly mocked native authority and births.

No test below qualifies Docker, admitted START, SQL custody or a real worker.
Local socket framing and current-process kernel facts are real where stated.
"""
import ast
from copy import copy, deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
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
    tmp_path.chmod(0o700);path=tmp_path/'owner.sock'
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
    b._ambiguous=b._prepared_ack=False;b._context='cid:scope';b._observations=[]
    b._prepared=b._wrapper=None;b._server=SimpleNamespace(_lock=d.threading.RLock())
    b._identity_current=lambda:events.append('identity')
    claim={'native':'mock'}
    b._closing_authority=lambda *a:events.append('native-current')or claim
    b._execute_with_attempt_fence=lambda request,peer,callback:callback(claim,SimpleNamespace())
    b._scope=SimpleNamespace(
        prepare_dispatch=lambda **kw:events.append('reserve')or {'schema':'checked-observer-budget-dispatch@1',
            'command':derived(),'context_cid':'cid:scope'},
        require_current=lambda:events.append('scope-current'),
        capture_allocated=lambda:events.append('capture-allocated'),
        abort_pending=lambda:events.append('abort-nonrefund'))
    b._native=SimpleNamespace(typed=SimpleNamespace(_kernel_peer_identity=lambda _: (11,os.geteuid(),101)),
        _require_wrapper_kernel_command=lambda **kw:{'pid':22,'parent_pid':11,'start_time_ticks':202,'process_birth_id':'birth:22'},
        _same_native_dispatch_claim=lambda *args:True,
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
        if isinstance(node,ast.ImportFrom)and node.module not in {'__future__','pathlib'}:pytest.fail('native import at client module top level')
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
    monkeypatch.setattr(d.os,'readlink',lambda _:request()['worktree'])
    ack={'schema':d.SCHEMA,'operation':'spawned','handoff_id':'n'*32,'pid':33,'process_birth_id':'birth:33'}
    channel=Channel(frame(ack));b._isolated_handoff(channel,x,(22,os.geteuid(),202))
    assert b._isolated_spawned and not b._ambiguous and events[-3:]==['capture-allocated','scope-current','native-current']
    assert 'abort-nonrefund'not in events
    with pytest.raises(d.CheckedContextDispatchError):b._isolated_handoff(Channel(),x,(22,os.geteuid(),202))


@pytest.mark.parametrize('mutation',['ticket','pid','start','command','cwd'])
def test_mocked_isolated_identity_recipe_and_cwd_controls(monkeypatch,mutation):
    b,events=mocked_broker(monkeypatch)
    b._handoff(Channel(frame(request())+frame(spawned_ack())))
    command=['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry',*derived()[1:]]
    x={'schema':d.SCHEMA,'operation':'prepare_isolated','context_cid':'cid:scope','command':command,
       'worktree':request()['worktree'],'wrapper_ticket':'n'*32};peer=(22,os.geteuid(),202)
    if mutation=='ticket':x['wrapper_ticket']='forged'
    elif mutation=='pid':peer=(23,peer[1],peer[2])
    elif mutation=='start':peer=(peer[0],peer[1],203)
    elif mutation=='command':x['command'][3]='root'
    else:
        b._native._isolated_kernel_birth=lambda **_:{'sudo':{'process_birth_id':'birth:33'},
            'worker':{'pid':44,'process_birth_id':'birth:44'},'parent_chain':[]}
        monkeypatch.setattr(d.os,'readlink',lambda _:'/different-worktree')
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
