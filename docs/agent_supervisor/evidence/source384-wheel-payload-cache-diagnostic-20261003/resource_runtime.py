"""External bounded source-free native-phase memory diagnostic, never a producer."""
import functools as _resource_functools
import json as _resource_json
import pathlib as _resource_pathlib
import sys as _resource_sys
import time as _resource_time
from dataclasses import asdict as _resource_asdict

_RESOURCE_PATH='/opt/ipfs-supervisor/state/source384-resource-events.jsonl'
_RESOURCE_MAX_EVENTS=64
_RESOURCE_MAX_BYTES=131072
_resource_state=None

def _resource_emit(stage,event,error=None):
    state=_resource_state
    if state['events']>=_RESOURCE_MAX_EVENTS:return
    try:
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
        base=_resource_pathlib.Path('/sys/fs/cgroup')
        for line in _resource_pathlib.Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                relative=_resource_pathlib.Path(line[3:].lstrip('/'))
                if '..' not in relative.parts and (base/relative).is_dir():base=base/relative
                break
        def bounded(path):
            with path.open('rb') as stream:raw=stream.read(16385)
            if len(raw)>16384:raise ValueError('resource event bound exceeded')
            return raw.decode()
        stat=dict(line.split() for line in bounded(base/'memory.stat').splitlines())
        fields=('anon','file','kernel','file_mapped','file_dirty','file_writeback','inactive_anon','active_anon','inactive_file','active_file','slab')
        value=dict(stage=stage,event=event,error_type=error,elapsed_seconds=_resource_time.monotonic()-state['started'],
            host=_resource_asdict(collect_proof_host_resources()),memory_current_bytes=int(bounded(base/'memory.current')),
            memory_stat={key:int(stat[key]) for key in fields if key in stat},
            process_statm=bounded(_resource_pathlib.Path('/proc/self/statm')).strip())
        raw=_resource_json.dumps(value,sort_keys=True,allow_nan=False)+'\n'
        if state['bytes']+len(raw.encode())>_RESOURCE_MAX_BYTES:return
        state['stream'].write(raw);state['stream'].flush();state['events']+=1;state['bytes']+=len(raw.encode())
    except TimeoutError:
        raise  # Preserve the native process alarm; never consume its deadline.
    except Exception:
        state['collection_errors']+=1

def _resource_wrap(owner,name,label,install_units=False):
    original=getattr(owner,name)
    @_resource_functools.wraps(original)
    def wrapped(*args,**kwargs):
        if install_units and not _resource_state['units']:
            module=_resource_sys.modules.get('ipfs_datasets_py.logic.software_contracts.codebase_source_units_384')
            if module is None:raise RuntimeError('native source-unit import expected before preparation')
            _resource_wrap(module,'_worker','numerical_worker')
            _resource_state['units']=True
            index=_resource_sys.modules['ipfs_datasets_py.logic.software_contracts.codebase_ir']
            if index._manifest_producer_key()!=_resource_state['producer']:
                raise RuntimeError('worker wrapper changed native reconstruction guard')
        _resource_emit(label,'enter')
        try:result=original(*args,**kwargs)
        except BaseException as exc:
            _resource_emit(label,'error',type(exc).__name__);raise
        _resource_emit(label,'return');return result
    setattr(owner,name,wrapped);_resource_state['restore'].append((owner,name,original))

def _resource_install():
    global _resource_state
    index=_resource_sys.modules['ipfs_datasets_py.logic.software_contracts.codebase_ir']
    key=index._manifest_producer_key()
    if key is None:raise RuntimeError('native immutable producer guard unavailable')
    stream=_resource_pathlib.Path(_RESOURCE_PATH).open('x',buffering=1)
    _resource_state=dict(started=_resource_time.monotonic(),stream=stream,events=0,bytes=0,
        collection_errors=0,units=False,restore=[],producer=key)
    _resource_wrap(index.RepositoryCodebaseIndex,'prepare_current','initial_index',install_units=True)
    _resource_wrap(index.RepositoryCodebaseIndex,'observe_current','source_observation')
    if index._manifest_producer_key()!=key:raise RuntimeError('instrumentation changed native reconstruction guard')
    _resource_emit('context','installed')

def _resource_finish():
    state=_resource_state
    if state is None:return
    try:
        _resource_emit('context','finished')
        index=_resource_sys.modules['ipfs_datasets_py.logic.software_contracts.codebase_ir']
        if index._manifest_producer_key()!=state['producer']:raise RuntimeError('native reconstruction guard changed')
    finally:
        try:state['stream'].close()
        finally:
            for owner,name,original in reversed(state['restore']):setattr(owner,name,original)
