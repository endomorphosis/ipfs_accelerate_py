"""Off-tree archive and four pinned public binary cache advice diagnostic."""
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

BASE = Path(__file__).resolve().parent
WORK = Path('/home/barberb/lift_coding')
A = WORK / '.worktrees/ir-release-accelerate-20261002'
D = WORK / '.worktrees/ir-release-datasets-20261002'
ARCHIVE = WORK / 'artifacts/source384-full-supervisor-preflight-20261003/bundle'
TASK = WORK / '.benchmarks/terminal-bench-2/fix-code-vulnerability'
EXPECTED_ARCHIVE = 'a83b157f478dbcb869f134e71bfd02b17bdb7f82f3de8d281cc6202f66464d0b'
EXPECTED_MANIFEST = '4d13cfeb3ad2daf67a50c135babbf58c6a197a71f70376bc730051508ce8fb27'

PROBE = r'''
from dataclasses import asdict
import json,pathlib,time
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
base=pathlib.Path('/sys/fs/cgroup'); current=base
for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines():
 if line.startswith('0::'):
  relative=pathlib.Path(line[3:].lstrip('/'))
  if '..' not in relative.parts and (base/relative).is_dir():current=base/relative
  break
def bounded(path):
 try:
  with path.open('rb') as stream:raw=stream.read(32769)
  if len(raw)>32768:raise ValueError('telemetry exceeds byte bound')
  return raw.decode('utf-8')
 except OSError as exc:return dict(error_type=type(exc).__name__)
result=dict(schema='source384-lean-resource-diagnostic@1',timestamp=time.time(),
 cgroup_path=str(current),host=asdict(collect_proof_host_resources()),
 cgroup={name:bounded(current/name) for name in ('memory.current','memory.max','memory.high','memory.stat','cpu.max','cpu.stat','cpu.pressure','memory.pressure','io.pressure')},
 proc_pressure={name:bounded(pathlib.Path('/proc/pressure')/name) for name in ('cpu','memory','io')},
 memory_policy_changed=False,global_drop_caches=False,provider_calls=0)
raw=json.dumps(result,sort_keys=True,allow_nan=False)
if len(raw.encode())>262144:raise ValueError('diagnostic too large')
print(raw)
'''

def write(name, value):
    (BASE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(raw)
    return h.hexdigest()

def live_pins():
    files = [
        A / 'benchmarks/agent_supervisor/container_coding/terminal_deployment.py',
        A / 'benchmarks/agent_supervisor/container_coding/terminal_source384_qualification.py',
        A / 'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',
        A / 'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
        D / 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py',
        D / 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py',
        D / 'ipfs_datasets_py/logic/software_contracts/codebase_source_384.py',
        D / 'ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py',
    ]
    return {str(path):sha(path) for path in files}

def containers():
    run = subprocess.run(['docker','ps','-a','--filter','name=ipfs-native-deploy-',
        '--format','{{.ID}} {{.Names}} {{.Status}}'],capture_output=True,text=True,timeout=15)
    return dict(returncode=run.returncode,rows=run.stdout.splitlines())

def native_cache_loader(base_path,base_sha256,candidate_path,candidate_sha256,receipt_path,receipt_sha256,receipt_bytes):
    # Explicit SHA-bound modules work under -I -S; no ambient import path is added.
    selection=dict(base_path=base_path,base_sha256=base_sha256,candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,receipt_path=receipt_path,receipt_sha256=receipt_sha256,receipt_bytes=receipt_bytes)
    return 'selection='+repr(selection)+'\n'+r'''
import hashlib,os,stat,sys,types
def code(path,pin):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as stream:
  info=os.fstat(stream.fileno())
  if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=65536:raise ValueError('bounded code file required')
  raw=stream.read(65537)
 if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('cache helper code digest differs')
 return compile(raw,path,'exec')
base=types.ModuleType('cache_candidate');base.__file__=selection['base_path']
exec(code(selection['base_path'],selection['base_sha256']),base.__dict__)
sys.modules['cache_candidate']=base
sys.argv=[selection['candidate_path'],selection['receipt_path'],selection['receipt_sha256']]
native=types.ModuleType('codex_cache_candidate');native.__file__=selection['candidate_path']
exec(code(selection['candidate_path'],selection['candidate_sha256']),native.__dict__)
native.protect_receipt(os.path.dirname(selection['receipt_path']),selection['receipt_bytes'])
native.main()
'''

async def main():
    from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
    from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualification
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import deploy_worker_boundary
    original = qualification.observe_resources
    original_deploy = deployment.deploy_supervisor

    async def sample(environment, name):
        response = await environment.exec(command=deployment.PYTHON+' -P -c '+shlex.quote(PROBE),
            cwd='/app',user='supervisor',env=deployment.runtime_environment(),timeout_sec=15)
        raw=response.stdout or ''
        if response.return_code!=0 or len(raw.encode())>262144:
            raise ValueError('bounded resource sample failed')
        value=json.loads(raw)
        (BASE/'docker-01'/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
        return value

    async def deployed(environment, **kwargs):
        result = await original_deploy(environment, **kwargs)
        await deploy_worker_boundary(environment, output=BASE/'docker-01'/'worker-boundary')
        await environment.upload_file(ARCHIVE/'manifest.json',deployment.ROOT+'/cache-diagnostic-manifest.json')
        candidate=BASE/'cache_candidate.py'
        await environment.upload_file(candidate,deployment.ROOT+'/cache-diagnostic.py')
        before=await sample(environment,'cache-resources-before.json')
        loader=('import hashlib,pathlib,sys;'
            'p=pathlib.Path("/opt/ipfs-supervisor/cache-diagnostic.py");r=p.read_bytes();'
            'assert hashlib.sha256(r).hexdigest()=='+repr(sha(candidate))+';'
            'sys.argv=[str(p),"/opt/ipfs-supervisor/cache-diagnostic-manifest.json"];'
            'exec(compile(r,str(p),"exec"),{"__name__":"__main__"})')
        response=await environment.exec(command=deployment.PYTHON+' -I -S -B -c '+shlex.quote(loader),
            cwd='/',user='root',timeout_sec=60)
        raw=response.stdout or ''
        (BASE/'docker-01'/'cache-advice.stdout').write_text(raw)
        (BASE/'docker-01'/'cache-advice.stderr').write_text(response.stderr or '')
        write('cache-advice-status.json',dict(returncode=response.return_code,candidate_sha256=sha(candidate)))
        if response.return_code!=0 or len(raw.encode())>262144:
            raise ValueError('bounded manifest-only cache advice failed')
        advice=json.loads(raw)
        if advice.get('schema')!='manifest-cache-advice@1':raise ValueError('unexpected cache advice schema')
        write('cache-advice.json',advice)
        after=await sample(environment,'cache-resources-after.json')
        print(json.dumps(dict(phase='cache_advice_complete',at=time.time(),
            advised_files=advice['advised_files'],advised_bytes=advice['advised_bytes'],
            available_before=before['host']['available_memory_mb'],
            available_after=after['host']['available_memory_mb'],freed_bytes_claimed=False)),flush=True)
        # The worker boundary recopies binaries; bind its final installer receipt.
        from codex_cache_candidate import require_receipt
        exposure=BASE/'docker-01/worker-boundary/installation/native-codex-binary.log'
        if exposure.stat().st_size>32768:raise ValueError('native installer receipt too large')
        require_receipt(json.loads(exposure.read_text()))
        receipt_sha256=sha(exposure)
        await environment.upload_file(exposure,deployment.ROOT+'/codex-cache-exposure.json')
        native_candidate=BASE/'codex_cache_candidate.py'
        await environment.upload_file(native_candidate,deployment.ROOT+'/codex-cache-diagnostic.py')
        native_before=await sample(environment,'codex-cache-resources-before.json')
        loader=native_cache_loader(deployment.ROOT+'/cache-diagnostic.py',sha(candidate),
            deployment.ROOT+'/codex-cache-diagnostic.py',sha(native_candidate),
            deployment.ROOT+'/codex-cache-exposure.json',receipt_sha256,exposure.stat().st_size)
        response=await environment.exec(command=deployment.PYTHON+' -I -S -B -c '+shlex.quote(loader),
            cwd='/',user='root',timeout_sec=60)
        raw=response.stdout or ''
        (BASE/'docker-01/codex-cache-advice.stdout').write_text(raw)
        (BASE/'docker-01/codex-cache-advice.stderr').write_text(response.stderr or '')
        write('codex-cache-advice-status.json',dict(returncode=response.return_code,
            candidate_sha256=sha(native_candidate),post_boundary_receipt_sha256=receipt_sha256))
        if response.return_code!=0 or len(raw.encode())>262144:
            raise ValueError('bounded pinned native binary cache advice failed')
        native_advice=json.loads(raw)
        if native_advice.get('schema')!='pinned-native-codex-cache-advice@1':raise ValueError('unexpected native advice schema')
        write('codex-cache-advice.json',native_advice)
        native_after=await sample(environment,'codex-cache-resources-after.json')
        print(json.dumps(dict(phase='native_binary_cache_advice_complete',at=time.time(),
            advised_files=native_advice['advised_files'],advised_bytes=native_advice['advised_bytes'],
            body_read_bytes=native_advice['body_read_bytes'],
            available_before=native_before['host']['available_memory_mb'],
            available_after=native_after['host']['available_memory_mb'],freed_bytes_claimed=False)),flush=True)
        return result

    async def observed(environment, *, output, profile):
        result = await original(environment, output=output, profile=profile)
        observed_at = time.time()
        response = await environment.exec(command=deployment.PYTHON+' -P -c '+shlex.quote(PROBE),
            cwd='/app',user='supervisor',env=deployment.runtime_environment(),timeout_sec=15)
        raw = response.stdout or ''
        if response.return_code != 0 or len(raw.encode()) > 262144:
            raise ValueError('bounded read-only resource diagnostic failed')
        value = json.loads(raw)
        if value.get('schema') != 'source384-lean-resource-diagnostic@1':
            raise ValueError('unexpected diagnostic schema')
        (Path(output)/'detailed-resources.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
        print(json.dumps(dict(phase='native_resource_observation_complete',at=observed_at,
            output=str(output),host=value['host'],next_phase='source384_context')),flush=True)
        return result

    qualification.observe_resources = observed
    deployment.deploy_supervisor = deployed
    try:
        return await asyncio.wait_for(deployment.qualify_original_container(task_dir=TASK,
            archive_dir=ARCHIVE,output=BASE/'docker-01',install_codex=True,
            auth_json=Path('/home/barberb/.codex/auth.json'),keep_container=False,
            resource_profile='source384-5cpu-12gib@1',source384_context=True),2300)
    finally:
        qualification.observe_resources = original
        deployment.deploy_supervisor = original_deploy

if __name__ == '__main__':
    assert sha(ARCHIVE/'manifest.json') == EXPECTED_MANIFEST
    assert sha(ARCHIVE/'runtime.tar.gz') == EXPECTED_ARCHIVE
    before=live_pins();write('before-pins.json',before)
    write('public-task-pins.json',{name:sha(TASK/name) for name in ('instruction.md','task.toml','environment/Dockerfile')})
    write('containers-before.json',containers())
    write('scope.json',dict(instrumented_diagnostic=True,archive_sha256=EXPECTED_ARCHIVE,
        manifest_sha256=EXPECTED_MANIFEST,install_codex=True,credential_contents_recorded=False,
        provider_calls=0,official_verifier_executed=False,benchmark_result=False,
        worker_boundary_installed=True,source384_context=True,
        resource_profile='source384-5cpu-12gib@1',setup_operation_timeouts_unchanged=True,
        qualification_outer_seconds=2300,qualification_exec_seconds=300,
        qualification_inner_seconds=270,source384_native_seconds=90,
        source_mutations=False,admission_changes=False,global_cache_drop=False,
        archive_advice_scope='manifest_regular_files_only',
        combined_advice_scope='archive_manifest_plus_four_pinned_public_native_binaries',
        cache_advice_is_best_effort=True,
        candidate_sha256=sha(BASE/'cache_candidate.py'),cache_diagnostic_timeout_seconds=60,
        exact_native_codex_binary_copies=4,native_binary_body_reads_required=True,
        native_candidate_sha256=sha(BASE/'codex_cache_candidate.py')))
    started=time.monotonic();code=0
    print(json.dumps(dict(phase='diagnostic_setup_started',at=time.time())),flush=True)
    try:
        write('result.json',asyncio.run(main()))
    except BaseException as exc:
        code=1;traceback.print_exc()
        write('failure.json',dict(error_type=type(exc).__name__,error=str(exc)[:2048]))
    finally:
        after=live_pins();write('after-pins.json',after)
        write('containers-after.json',containers())
        write('exit.json',dict(returncode=code,seconds=time.monotonic()-started,
            source_pins_unchanged=before==after,provider_calls=0,official_verifier_executed=False))
    raise SystemExit(code)
