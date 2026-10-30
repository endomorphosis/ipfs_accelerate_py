"""Real sealed-file/pipe/refusal controls only; no fake Grok capture row.

Inert metadata never supplies native/source/credential/task authority. These
cases never construct NamedSessionRequest, dispatch, or seed capture. The two
allocation cases use only a test-owned genuine store and empty public reads.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import sys
import types

import pytest
import ipfs_accelerate_py as actual_root
import ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_worker as worker
import ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_transport as transport
import ipfs_accelerate_py.llm_allocation.duckdb_store as allocation


def pin(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def capture_request(tmp_path):
    metadata=tmp_path/'metadata';metadata.mkdir(mode=0o700)
    owner_capsule=metadata/'owner-capsule.json';owner_capsule.write_bytes(b'{}\n');owner_capsule.chmod(0o600)
    profile=metadata/'sources.json';profile.write_bytes(b'{}\n');profile.chmod(0o600)
    root=Path(worker.__file__).parents[2]
    now=time.monotonic()
    return dict(schema=worker.SCHEMA,mode='fresh-task-capture-diagnostic',one_shot=True,attempt_cap=1,
        provider_dispatch_allowed=True,execution_environment='isolated-loopback',accelerator_root=str(root),
        workspace=str(tmp_path/'workspace'),receipt_directory=str(tmp_path/'receipts'),task_alias='fresh-test-task',
        task_cid='inert-task-cid',database_attempt_id='inert-attempt',database_attempt_number=1,
        allocation_session_id='inert-allocation',source_generation='grok-automatic-named-session/fresh-test',
        work_deadline_monotonic=now+10,native_deadline_monotonic=now+20,owner_pid=2,owner_birth=1,
        owner_request_sha256='1'*64,context_path=str(tmp_path/'receipts'/'owner-context.json'),
        capsule_pin=pin(owner_capsule),source_profile_pin=pin(profile),max_output_tokens=128,maximum_timeout_seconds=10)


def sealed(raw,flags=worker.SEALS):
    descriptor=os.memfd_create('fresh-capture-test',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    os.write(descriptor,raw)
    if flags:fcntl.fcntl(descriptor,fcntl.F_ADD_SEALS,flags)
    return descriptor


def test_actual_fully_sealed_memfd_roundtrip_is_metadata_only(capture_request):
    raw=worker.canonical(capture_request);descriptor=sealed(raw)
    try:
        assert worker.read_sealed_request(descriptor,hashlib.sha256(raw).hexdigest())==capture_request
        assert fcntl.fcntl(descriptor,fcntl.F_GET_SEALS)&worker.SEALS==worker.SEALS
        with pytest.raises(OSError):os.pwrite(descriptor,b'x',0)
    finally:os.close(descriptor)


def test_unsealed_actual_memfd_is_not_accepted(capture_request):
    raw=worker.canonical(capture_request);descriptor=sealed(raw,0)
    try:
        with pytest.raises(worker.CaptureRefusal,match='fully_sealed_private_memfd_required'):
            worker.read_sealed_request(descriptor,hashlib.sha256(raw).hexdigest())
    finally:os.close(descriptor)


def test_actual_sealed_bytes_cannot_be_rebound_to_different_digest(capture_request):
    raw=worker.canonical(capture_request);descriptor=sealed(raw)
    try:
        with pytest.raises(worker.CaptureRefusal,match='sealed_request_changed'):
            worker.read_sealed_request(descriptor,'0'*64)
    finally:os.close(descriptor)


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}\n',b'{"a":NaN}\n'])
def test_duplicate_or_nonfinite_actual_JSON_bytes_refuse(raw):
    with pytest.raises(worker.CaptureRefusal):worker.decode(raw)


@pytest.mark.parametrize('mutation',['cap_bool','attempt_bool','output_bool','unknown_fallback',
    'provider_false','overlap','nonfinite','generation'])
def test_inert_request_cannot_relax_closed_shape_or_limits(capture_request,mutation):
    value=copy.deepcopy(capture_request)
    if mutation=='cap_bool':value['attempt_cap']=True
    elif mutation=='attempt_bool':value['database_attempt_number']=True
    elif mutation=='output_bool':value['max_output_tokens']=True
    elif mutation=='unknown_fallback':value['allow_local_fallback']=True
    elif mutation=='provider_false':value['provider_dispatch_allowed']=False
    elif mutation=='overlap':value['workspace']=value['receipt_directory']
    elif mutation=='nonfinite':value['work_deadline_monotonic']=float('inf')
    else:value['source_generation']='not-a-named-session'
    with pytest.raises(worker.CaptureRefusal):worker.validate_request(value)


def test_real_file_pin_tampering_refuses_without_repin(tmp_path):
    path=tmp_path/'input.json';path.write_bytes(b'{"public":1}\n');path.chmod(0o600)
    frozen=pin(path);assert worker.read_pin(frozen,private=True)==path.read_bytes()
    path.write_bytes(b'{"public":2}\n')
    with pytest.raises(worker.CaptureRefusal,match='input_changed_or_pin_mismatch'):
        worker.read_pin(frozen,private=True)


def test_actual_source_symlink_is_refused(tmp_path):
    path=tmp_path/'real.py';path.write_bytes(b'pass\n');path.chmod(0o600)
    alias=tmp_path/'alias.py';alias.symlink_to(path)
    frozen=pin(path);frozen['path']=str(alias)
    with pytest.raises(worker.CaptureRefusal,match='input_path_alias'):worker.read_pin(frozen)


def test_actual_FIFO_is_refused_without_blocking_reader(tmp_path):
    path=tmp_path/'fifo';os.mkfifo(path,0o600)
    frozen=dict(path=str(path),bytes=1,sha256='1'*64)
    with pytest.raises(worker.CaptureRefusal,match='input_custody_required'):worker.read_pin(frozen)


def test_private_snapshot_checks_real_byte_bound_before_read(tmp_path):
    path=tmp_path/'large';path.write_bytes(b'x'*32);path.chmod(0o600)
    with pytest.raises(worker.CaptureRefusal,match='bounded_private_snapshot_required'):
        transport._private_pin(path,8)


def pipe_exchange(action,deadline):
    read,write=os.pipe();stopped=threading.Event();errors=[]
    def writer():
        try:action(write,stopped)
        except BrokenPipeError:pass
        except BaseException as error:errors.append(error)
        finally:os.close(write)
    thread=threading.Thread(target=writer,name='fresh-capture-stdin-fixture');thread.start()
    try:
        with os.fdopen(read,'rb',buffering=0)as stream:
            return worker.read_original_stdin(stream,deadline=deadline)
    finally:
        stopped.set();thread.join(timeout=3)
        assert not thread.is_alive() and not errors


def test_actual_fragmented_original_stdin_uses_one_absolute_clock():
    def fragments(fd,stopped):
        for raw in (b'fresh ',b'private ',b'task\n'):
            os.write(fd,raw)
            if stopped.wait(0.005):break
    assert pipe_exchange(fragments,time.monotonic()+2)==b'fresh private task\n'


def test_actual_stdin_drip_cannot_renew_original_deadline():
    def drip(fd,stopped):
        for _ in range(100):
            if stopped.wait(0.01):return
            os.write(fd,b'x')
    with pytest.raises(worker.CaptureRefusal,match='original_stdin_deadline_expired'):
        pipe_exchange(drip,time.monotonic()+0.05)


@pytest.mark.parametrize('mode',['empty','oversize'])
def test_real_stdin_EOF_or_byte_bound_refuses(mode):
    def write(fd,stopped):
        if mode=='oversize':
            remaining=worker.MAX_STDIN+1
            while remaining and not stopped.is_set():
                count=os.write(fd,b'x'*min(4096,remaining));remaining-=count
    with pytest.raises(worker.CaptureRefusal):pipe_exchange(write,time.monotonic()+2)


@pytest.mark.parametrize('remaining,maximum,expected',[(1.9,90,1),(200.0,90,90),(7.1,3,3)])
def test_timeout_is_floor_of_original_remaining_with_fixed_ceiling(remaining,maximum,expected):
    assert transport.bounded_timeout(remaining,maximum)==expected


@pytest.mark.parametrize('remaining,maximum',[(True,90),(0.9,90),(float('inf'),90),(3.0,True)])
def test_invalid_or_exhausted_dispatch_budget_refuses(remaining,maximum):
    with pytest.raises(worker.CaptureRefusal):transport.bounded_timeout(remaining,maximum)


def capsule_shape(capture_request):
    # Inert pin data tests metadata only; never a selected executable/config.
    root=Path(capture_request['accelerator_root'])
    files={role:dict(path=str(root/relative),bytes=1,sha256='2'*64)
        for role,relative in transport.RELATIVE.items()}
    dummy=dict(path=str(root/'unselected-fixture-input'),bytes=1,sha256='3'*64)
    return dict(schema=transport.CAPSULE_SCHEMA,source_generation=capture_request['source_generation'],files=files,
        named_session=dict(native_home='/tmp/inert-native',audit_directory='/tmp/inert-audit',
        preparation_home='/tmp/inert-preparation',binary=dict(dummy),configuration=dict(dummy),
        owner_sampling_policy=dict(dummy),system_requirements=None,cleanup_reserve_seconds=3,bootstrap_max_seconds=10))


def test_generic_V4_capsule_shape_keeps_fresh_task_separate(capture_request):
    capsule=capsule_shape(capture_request)
    assert transport.validate_capsule_shape(capsule,capture_request)==capsule


def test_capsule_cannot_add_arbitrary_source_role(capture_request):
    capsule=capsule_shape(capture_request);capsule['files']['arbitrary']=dict(capsule['files']['worker'])
    with pytest.raises(worker.CaptureRefusal):transport.validate_capsule_shape(capsule,capture_request)


def test_bool_cleanup_reserve_is_not_a_finite_numeric_budget(capture_request):
    capsule=capsule_shape(capture_request);capsule['named_session']['cleanup_reserve_seconds']=True
    with pytest.raises(worker.CaptureRefusal):transport.validate_capsule_shape(capsule,capture_request)


def test_real_durable_metadata_record_never_overwrites_existing_bytes(tmp_path):
    tmp_path.chmod(0o700)
    first=transport.fresh_record(tmp_path,'actual-record.json',b'{"public":1}\n')
    assert worker.read_pin(first,private=True)==b'{"public":1}\n'
    with pytest.raises(FileExistsError):transport.fresh_record(tmp_path,'actual-record.json',b'{"public":2}\n')
    assert worker.read_pin(first,private=True)==b'{"public":1}\n'


def test_genuine_private_allocation_query_initializes_before_empty_acceptance(tmp_path,capture_request):
    path=tmp_path/'private-allocation.duckdb'
    store=allocation.AllocationStore(path=path)
    assert store._ready is False and store._duckdb is None and not path.exists()
    started=time.monotonic()
    observed=transport._store_row(store,capture_request,dict(allocation_store_path=str(path)),empty=True)
    assert store._ready is True and type(store._duckdb)is types.ModuleType
    assert store._duckdb is sys.modules['duckdb'] and path.is_file()
    assert observed['allocation_store_path']==str(path) and observed['row']=={}
    assert started<=observed['observed_monotonic']<=time.monotonic()
    assert store.get_cli_session(capture_request['allocation_session_id'],'grok_cli')=={}
    # A real read-only connection independently confirms no seeded capture row.
    connection=store._duckdb.connect(str(path),read_only=True)
    try:assert connection.execute('SELECT COUNT(*) FROM cli_sessions').fetchone()==(0,)
    finally:connection.close()


def test_genuine_store_path_mismatch_refuses_before_lazy_database_creation(tmp_path,capture_request):
    actual=tmp_path/'unselected-allocation.duckdb';expected=tmp_path/'selected-allocation.duckdb'
    store=allocation.AllocationStore(path=actual)
    with pytest.raises(worker.CaptureRefusal,match='actual_bound_allocation_store_required'):
        transport._store_row(store,capture_request,dict(allocation_store_path=str(expected)),empty=True)
    assert store._ready is False and store._duckdb is None
    assert not actual.exists() and not expected.exists()


def test_genuine_normal_root_lazy_module_subclass_is_accepted(capture_request):
    root=Path(worker.__file__).parents[2];path=root/'ipfs_accelerate_py/__init__.py'
    profile=dict(sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
    observed=transport.normal_module('ipfs_accelerate_py',capture_request,profile)
    assert observed is actual_root and sys.modules['ipfs_accelerate_py']is observed
    assert type(observed)is vars(actual_root)['_IPFSAccelerateModule']
    assert type(observed).__bases__==(types.ModuleType,)


def test_genuine_ordinary_worker_module_keeps_exact_module_type(capture_request):
    path=Path(worker.__file__)
    profile=dict(sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
    observed=transport.normal_module(worker.__name__,capture_request,profile)
    assert observed is worker and type(observed)is types.ModuleType
    assert observed.__file__==observed.__spec__.origin==str(path)


def test_genuine_root_still_refuses_wrong_actual_source_pin(capture_request):
    root=Path(worker.__file__).parents[2];path=root/'ipfs_accelerate_py/__init__.py'
    profile=dict(sources={str(path):'0'*64})
    with pytest.raises(worker.CaptureRefusal):
        transport.normal_module('ipfs_accelerate_py',capture_request,profile)
