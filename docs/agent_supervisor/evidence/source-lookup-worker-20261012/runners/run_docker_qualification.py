"""Provider-free source channel qualification in one fresh owned container."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

WORKSPACE = Path('/home/barberb/lift_coding')
SOURCE = WORKSPACE / '.worktrees/token-selector-accelerate-20261011'
DATASETS = WORKSPACE / '.worktrees/source-worker-datasets-20261012'
KIT = WORKSPACE / '.worktrees/publication-main-20261004/kit'
EVIDENCE = Path(__file__).resolve().parent.parent
IMAGE = 'sha256:771900c18b663b33a5bf96e066346b2d5c842506523efb167aebe8d3a99b5c07'
ROOT = '/opt/ipfs-supervisor'
PYTHON = ROOT + '/venv/bin/python'
PRODUCTS = [
    'ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_broker.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_client.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/repository_source_inventory.py',
    'ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_broker.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/container_worker_boundary.py',
    'ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py',
    'ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py',
    'benchmarks/agent_supervisor/container_coding/container_worker_deployment.py',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


SETUP = r'''
import hashlib,json,os,pathlib,pwd,shutil,subprocess,sys
root=pathlib.Path('/opt/ipfs-supervisor')
def runtime_ignore(directory,names):
 result=[]
 for name in names:
  path=pathlib.Path(directory)/name
  if path.is_dir():
   if name in ('__pycache__','.git','node_modules','llvm','artifacts','.venv','venv','.venvs','model_cache'):result.append(name)
  elif path.suffix not in ('.py','.sql'):result.append(name)
 return result
ignore=runtime_ignore
for name in ('source','datasets','kit'):
 target=root/name
 if target.exists():shutil.rmtree(target)
 target.mkdir(mode=0o755);os.chown(target,0,0)
 if name=='source':
  for package in ('ipfs_accelerate_py','benchmarks'):shutil.copytree(pathlib.Path('/mounted/source')/package,target/package,ignore=ignore)
 elif name=='datasets':shutil.copytree(pathlib.Path('/mounted/datasets/ipfs_datasets_py'),target/'ipfs_datasets_py',ignore=ignore)
 else:shutil.copytree(pathlib.Path('/mounted/kit/ipfs_kit_py'),target/'ipfs_kit_py',ignore=ignore)
 for base,dirs,files in os.walk(target):
  os.chown(base,0,0);pathlib.Path(base).chmod(0o755)
  for filename in files:
   path=pathlib.Path(base)/filename
   if path.is_symlink():raise ValueError('qualification source copy contains a link')
   os.chown(path,0,0);path.chmod(path.stat().st_mode&~0o022)
sys.path[:0]=[str(root/name) for name in ('source','datasets','kit')]
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY,WORKER_ENTRY,SOURCE_LOOKUP_ENTRY
assert pwd.getpwnam('supervisor').pw_uid==1000 and pwd.getpwnam('benchmarkworker').pw_uid==1001
subprocess.run(['usermod','-a','-G',pwd.getpwuid(1000).pw_name,'benchmarkworker'],check=True)
anchor='else:\n from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main'
assert WORKER_ENTRY.count(anchor)==1
replacement="else:\n if args.model=='source-ipc-authored':\n  import runpy\n  runpy.run_path('/qualification/worker_probe.py')['install']()\n from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main"
worker=WORKER_ENTRY.replace(anchor,replacement)
for name,text in (('owner-worker',OWNER_ENTRY),('worker-entry',worker),('source-lookup',SOURCE_LOOKUP_ENTRY)):
 path=root/'bin'/name;path.write_text(text);os.chown(path,0,0);path.chmod(0o555)
probe='#!/opt/ipfs-supervisor/venv/bin/python -I\nimport ctypes,os,sys\nlibc=ctypes.CDLL(None,use_errno=True)\nif libc.prctl(38,1,0,0,0)!=0:raise OSError("no-new-privileges failed")\nos.execv("/opt/ipfs-supervisor/bin/source-lookup",["/opt/ipfs-supervisor/bin/source-lookup",*sys.argv[1:]])\n'
path=root/'bin/source-probe-client';path.write_text(probe);os.chown(path,0,0);path.chmod(0o555)
sudo=pathlib.Path('/etc/sudoers.d/source-lookup-qualification')
sudo.write_text('supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *, /opt/ipfs-supervisor/bin/source-probe-client *\n')
os.chown(sudo,0,0);sudo.chmod(0o440)
for name,mode in (('state',0o700),('repository',0o755),('worktrees',0o755),('worker-observations',0o750),('fixture-ipc',0o750)):
 path=root/name;path.mkdir(exist_ok=True);os.chown(path,1000,1000);path.chmod(mode)
for name in ('worker-result.json','router-observation.json'):
 path=root/'fixture-ipc'/name;path.touch();os.chown(path,1001,1000);path.chmod(0o660)
for name in ('.local','.duckdb','.ipfs_accelerate'):
 path=pathlib.Path('/home/supervisor')/name;path.mkdir(exist_ok=True);os.chown(path,1000,1000);path.chmod(0o700)
boundary=dict(schema='supervisor-container-worker-boundary@1',container_id=sys.argv[1],image_id=sys.argv[2],owner_uid=1000,worker_uid=1001,single_worker=True,provider='codex_cli',namespaces={name:os.readlink('/proc/self/ns/'+name) for name in ('pid','mnt','net')},allowed_worktree_roots=[str(root/'worktrees')],owner_private_paths=[str(root/'state')])
path=root/'container-boundary.json';path.write_text(json.dumps(boundary,sort_keys=True));os.chown(path,0,0);path.chmod(0o444)
print(json.dumps(dict(schema='source-lookup-root-setup@1',boundary=boundary,
 owner_entry_sha256=hashlib.sha256(OWNER_ENTRY.encode()).hexdigest(),
 worker_entry_production_sha256=hashlib.sha256(WORKER_ENTRY.encode()).hexdigest(),
 worker_entry_authored_hook_sha256=hashlib.sha256(worker.encode()).hexdigest(),
 source_lookup_entry_sha256=hashlib.sha256(SOURCE_LOOKUP_ENTRY.encode()).hexdigest(),
 unrelated_peer_probe_entry_sha256=hashlib.sha256(probe.encode()).hexdigest(),
 qualification_hook='Only fixture model inserts authored generate_text; OWNER_ENTRY and SOURCE_LOOKUP_ENTRY unchanged',
 source_copy='Root-owned .py/.sql runtime trees only; no tests/plugins, dataset blobs, Venv, weights, Git owner or LLVM copy')))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    assert output.parent == EVIDENCE / 'docker' and not output.exists()
    output.mkdir(parents=True, mode=0o700)
    results = output / 'results'
    results.mkdir(mode=0o777); results.chmod(0o777)
    runners = {str(path): sha(path) for path in (Path(__file__), Path(__file__).with_name('owner_probe.py'), Path(__file__).with_name('worker_probe.py'))}
    bindings = {str(SOURCE / path): sha(SOURCE / path) for path in PRODUCTS}
    for path in ('logic/software_contracts/content.py','logic/software_contracts/cache.py',
                 'logic/software_contracts/semantic_index/committed_snapshot.py'):
        selected = DATASETS / 'ipfs_datasets_py' / path
        bindings[str(selected)] = sha(selected)
    snapshot = output / 'runner-snapshot'; snapshot.mkdir()
    for path in runners:
        shutil.copyfile(path, snapshot / Path(path).name)
    name = 'source-lookup-worker-' + str(time.time_ns())
    record = dict(schema='source-worker-offline-docker-qualification@1', complete=False,
        image_id=IMAGE, container_name=name, provider_calls=0, model_calls=0, solver_calls=0,
        native_provider_tokens=None, paid_token_savings_measured=False, tool_choice_qualified=False,
        full_automatic_terminal_binding_qualified=False, network='none', cpu_limit=5,
        memory_limit_bytes=12*1024**3, pids_limit=128, privileged=False, read_only_base=False,
        root_setup_required_for_existing_sudo_worker_transition=True,
        accelerator_revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        datasets_revision=subprocess.check_output(['git','-C',str(DATASETS),'rev-parse','HEAD'],text=True).strip())
    env = dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',IPFS_KIT_DISABLE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_TELEMETRY='1',
        PYTHONHASHSEED='0',
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=ROOT+'/state/orchestration',
        IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=ROOT+'/state/seal.duckdb')
    command = ['docker','create','--name',name,'--network','none','--cpus','5','--memory','12g','--pids-limit','128','--user','0']
    mounts = [(SOURCE,'/mounted/source',True),(DATASETS,'/mounted/datasets',True),(KIT,'/mounted/kit',True),
        (Path('/home/barberb/.duckdb/extensions/v1.5.5/linux_arm64'),'/extensions',True),
        (Path(__file__).with_name('owner_probe.py'),'/qualification/owner_probe.py',True),
        (Path(__file__).with_name('worker_probe.py'),'/qualification/worker_probe.py',True),(results,'/results',False)]
    for host, target, readonly in mounts:
        assert host.exists()
        command += ['--mount',f'type=bind,src={host},dst={target}' + (',readonly' if readonly else '')]
    for key, value in env.items(): command += ['--env',key+'='+value]
    command += ['--entrypoint','/bin/sleep',IMAGE,'900']
    write(output/'command.json',dict(command=command, environment=env, runner_bindings=runners, source_bindings=bindings,
        scope='Actual generated owner/worker/helper with authored provider response; no native model decision.'))
    (output/'root-setup.py').write_text(SETUP)
    container, started = None, time.monotonic()
    try:
        assert subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()==IMAGE
        container = subprocess.check_output(command,text=True,timeout=30).strip()
        record['container_id'] = container
        subprocess.run(['docker','start',container],check=True,capture_output=True,timeout=30)
        setup = subprocess.run(['docker','exec','--user','0',container,PYTHON,'-I','-B','-c',SETUP,container,IMAGE],capture_output=True,text=True,timeout=120)
        (output/'root-setup.stdout').write_text(setup.stdout); (output/'root-setup.stderr').write_text(setup.stderr)
        setup.check_returncode()
        actual = json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]
        controls = {key: actual['HostConfig'][key] for key in ('NanoCpus','Memory','PidsLimit','NetworkMode','Privileged','ReadonlyRootfs')}
        assert controls == dict(NanoCpus=5_000_000_000,Memory=12*1024**3,PidsLimit=128,NetworkMode='none',Privileged=False,ReadonlyRootfs=False)
        write(output/'container-controls.json',dict(controls=controls,mounts=[{key:row[key] for key in ('Type','Source','Destination','RW')} for row in actual['Mounts']]))
        argv=['docker','exec','--user','1000:1000','--env','HOME=/home/supervisor',container,PYTHON,'-I','-B','/qualification/owner_probe.py']
        write(output/'owner-command.json',dict(argv=argv))
        with (output/'owner.stdout').open('w') as stdout,(output/'owner.stderr').open('w') as stderr:
            result = subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=600)
        record['owner_returncode'] = result.returncode
        if (results/'probe-qualification.json').exists():
            record['probe'] = json.loads((results/'probe-qualification.json').read_bytes())
            record['complete'] = result.returncode == 0 and record['probe']['complete']
    except BaseException as error:
        record.update(error_type=type(error).__name__,error=str(error)[:500])
        (output/'failure.txt').write_text(traceback.format_exc())
    finally:
        if container is not None:
            diagnostics=output/'private-diagnostics';diagnostics.mkdir(mode=0o700)
            for relative in ('probe-failure.txt','owner-entry.stderr','owner-entry.stdout','owner-launch.json','unrelated-peer-diagnostic.json'):
                copied=subprocess.run(['docker','cp',container+':'+ROOT+'/state/'+relative,str(diagnostics/relative)],capture_output=True,text=True,timeout=20)
                record[relative+'_retained']=copied.returncode==0
            copied=subprocess.run(['docker','cp',container+':'+ROOT+'/fixture-ipc/worker-result.json',str(diagnostics/'worker-result.json')],capture_output=True,text=True,timeout=20)
            record['private_worker_result_retained']=copied.returncode==0
            audits=subprocess.run(['docker','exec','--user','0',container,PYTHON,'-I','-B','-c',
                'import json,pathlib; print(json.dumps([str(p) for p in pathlib.Path("/opt/ipfs-supervisor/worker-observations").glob("source-*-audit.json")]))'],capture_output=True,text=True,timeout=20)
            if audits.returncode==0:
                for path in json.loads(audits.stdout):
                    subprocess.run(['docker','cp',container+':'+path,str(diagnostics/Path(path).name)],capture_output=True,text=True,timeout=20)
            record['owner_database_cas_tokens_dumped']=False
            subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=30)
        record['container_removed'] = not subprocess.check_output(['docker','container','ls','-a','--filter','name='+name,'--format','{{.Names}}'],text=True).strip()
        record['source_unchanged'] = all(sha(path)==digest for path,digest in bindings.items())
        record['runner_unchanged'] = all(sha(path)==digest for path,digest in runners.items())
        record['complete'] = record['complete'] and record['container_removed'] and record['source_unchanged'] and record['runner_unchanged']
        record['seconds'] = time.monotonic()-started
        write(output/'qualification.json',record)
        print(json.dumps({key:record.get(key) for key in ('complete','seconds','owner_returncode','error_type','source_unchanged','container_removed')}),flush=True)
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
