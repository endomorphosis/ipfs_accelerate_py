"""Authored response drives the actual UID1001 router and protected helper."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

ROOT = Path('/opt/ipfs-supervisor')
CONTROL = ROOT/'fixture-ipc'


def wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def signal_owner(name, **fields):
    print(json.dumps(dict(fixture_event=name,**fields),sort_keys=True),flush=True)
    deadline=time.monotonic()+30
    while not (CONTROL/(name+'.ack')).exists():
        if time.monotonic()>deadline:raise TimeoutError('qualification owner did not acknowledge '+name)
        time.sleep(.05)


def install():
    assert os.getuid()==os.geteuid()==1001
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
    from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_client as protocol
    intelligence_index.discover_available_providers=lambda:['codex_cli']
    intelligence_index.select_efficient_route=lambda **kwargs:SimpleNamespace(provider='codex_cli',model_name='source-ipc-authored',catalog_revision='authored-fixture')
    llm_router.get_llm_provider=lambda *args,**kwargs:object()
    calls=[]

    def generate(prompt, **kwargs):
        assert not calls and kwargs['allow_local_fallback']is False and kwargs['allow_cross_provider_fallback']is False
        path=os.environ[protocol.DESCRIPTOR_ENV]
        digest=os.environ[protocol.DIGEST_ENV]
        descriptor=protocol.load_descriptor(path,digest,workspace=Path.cwd())
        assert '/opt/ipfs-supervisor/bin/source-lookup' in prompt and 'source_refs' in prompt
        calls.append(dict(prompt_utf8_bytes=len(prompt.encode()),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            descriptor_sha256=digest,authored_response=True))
        status=dict(line.split(':',1)for line in Path('/proc/self/status').read_text().splitlines()if ':' in line)
        assert status['NoNewPrivs'].strip()=='1' and all(int(status[name].strip(),16)==0 for name in('CapEff','CapPrm','CapAmb'))
        denied=[]
        for name in json.loads((CONTROL/'private-paths.json').read_bytes()):
            try:
                with Path(name).open('rb'):pass
            except PermissionError:denied.append(name)
            else:raise AssertionError('worker could access a private owner locator')
        subprocesses=[]

        def query(ref, *, offset=0, continuation=None):
            payload=dict(schema=protocol.SOURCE_REQUEST_SCHEMA,binding_cid=descriptor['source_binding_cid'],
                raw_path_hex=ref['raw_path_hex'],source_cid=ref['source_cid'],offset=offset,
                max_bytes=min(4096,descriptor['max_slice_bytes']),continuation=continuation)
            process=subprocess.Popen([str(ROOT/'bin/source-lookup')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            identity=protocol.process_identity(process.pid)
            stdout,stderr=process.communicate(input=wire(payload)+b'\n',timeout=30)
            response=json.loads(stdout)
            assert stdout==wire(response)+b'\n' and stderr==b''
            assert process.returncode==(0 if response['ok']else 2)
            protocol.validate_response(response,descriptor=descriptor,payload=payload,request_id=response['request_id'])
            assert response['response_bytes']==len(stdout)<=descriptor['max_response_bytes']
            assert response['response_wire_bytes']==len(stdout)+3
            assert str(ROOT/'state/control.duckdb').encode()not in stdout and str(ROOT/'state/native.sock').encode()not in stdout
            subprocesses.append(dict(pid=identity['pid'],parent_pid=identity['parent_pid'],uid=identity['uid'],
                start_time_ticks=identity['start_time_ticks'],response_bytes=len(stdout),
                response_wire_bytes=response['response_wire_bytes'],ok=response['ok'],
                source_binding_cid=descriptor['source_binding_cid'],native_grant_id=descriptor['native_grant_id']))
            return response

        signal_owner('worker_ready',descriptor=path,descriptor_sha256=digest)
        pages={}
        for ref in descriptor['source_refs']:
            name=bytes.fromhex(ref['raw_path_hex']).decode()
            selected,offset,cursor=[],0,None
            for _ in range(32):
                response=query(ref,offset=offset,continuation=cursor)
                assert response['ok'],response['reason_code']
                selected.append(response)
                if response['result']['source_complete']:break
                offset,cursor=response['result']['end_offset'],response['result']['next_continuation']
            else:raise AssertionError('source pagination did not terminate')
            pages[name]=selected
        assert len(pages['module.py'])>1
        ref=next(ref for ref in descriptor['source_refs']if bytes.fromhex(ref['raw_path_hex'])==b'module.py')
        negatives=[]
        for scope in('canonical','allocated'):
            signal_owner('mutate_'+scope)
            response=query(ref)
            assert response['ok']is False and response['result']is None
            negatives.append(dict(case='stale_'+scope,reason_code=response['reason_code']))
            signal_owner('restore_'+scope)
        assert query(ref)['ok']
        assert len({item['pid'] for item in subprocesses})==len(subprocesses)
        assert {item['native_grant_id']for item in subprocesses}=={descriptor['native_grant_id']}
        assert all(item['uid']==1001 and item['parent_pid']==os.getpid()for item in subprocesses)
        result=dict(schema='authored-source-worker-probe@1',uid=os.getuid(),gid=os.getgid(),groups=os.getgroups(),
            workspace=str(Path.cwd()),no_new_privileges=True,worker_effective_capabilities=0,
            private_read_denials=denied,pages=pages,negatives=negatives,helper_processes=subprocesses,
            provider_calls=0,model_calls=0,native_provider_tokens=None,total_token_savings_measured=False)
        (CONTROL/'worker-result.json').write_bytes(wire(result))
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        set_last_cli_observation('codex_cli',dict(exit_code=0,authored_offline_fixture=True))
        return 'Authored source retrieval completed; no source was executed or edited by the worker.'

    llm_router.generate_text=generate
    actual_run=router.run

    def capture_run(*args,**kwargs):
        output,receipt=actual_run(*args,**kwargs)
        assert len(calls)==1
        assert protocol.DESCRIPTOR_ENV not in os.environ and protocol.DIGEST_ENV not in os.environ
        retained={key:value for key,value in receipt.items()if key in('schema','source_lookup_retrieval','completion_authority',
            'usage','provider','model','model_prompt_bytes','model_prompt_sha256','native_prompt_bytes','native_prompt_sha256',
            'invocation_id','purpose','workspace','external_container_boundary','status')}
        (CONTROL/'router-observation.json').write_bytes(wire(dict(schema='authored-source-router-probe@1',
            authored_generate_calls=calls,descriptor_environment_restored=True,receipt=retained,
            provider_calls=0,model_calls=0,native_provider_tokens=None,coding_reply_mode='legacy',
            modern_sealed_semantic_route_qualified=False)))
        return output,receipt

    router.run=capture_run
