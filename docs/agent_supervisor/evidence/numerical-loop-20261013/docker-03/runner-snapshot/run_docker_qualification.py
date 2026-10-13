"""Root-released offline Docker probe; owns only a fresh named container."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

W = Path('/home/barberb/lift_coding')
R = W / 'artifacts/numerical-loop-coverage-20261013'
P = W / '.worktrees/supervisor-comparison-contracts-20261011'
D = W / '.worktrees/indexed-coverage-datasets-20261013'
K = W / '.worktrees/publication-main-20261004/kit'
IMAGE = 'sha256:771900c18b663b33a5bf96e066346b2d5c842506523efb167aebe8d3a99b5c07'
ROOT = '/opt/ipfs-supervisor'
PYTHON = ROOT + '/venv/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


SETUP = r'''
import json,os,pathlib,pwd,subprocess,sys
root=pathlib.Path('/opt/ipfs-supervisor')
sys.path[:0]=[str(root/n) for n in ('source','datasets','kit')]
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY,WORKER_ENTRY
assert pwd.getpwnam('supervisor').pw_uid==1000
assert pwd.getpwnam('benchmarkworker').pw_uid==1001
subprocess.run(['usermod','-a','-G',pwd.getpwuid(1000).pw_name,'benchmarkworker'],check=True)
anchor='else:\n from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main'
assert WORKER_ENTRY.count(anchor)==1
replacement="else:\n if args.model=='numerical-ipc-authored':\n  import runpy\n  runpy.run_path('/qualification/worker_probe.py')['install']()\n from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main"
worker=WORKER_ENTRY.replace(anchor,replacement)
client=''' + repr('''#!/opt/ipfs-supervisor/venv/bin/python -I
import ctypes,os,sys
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(38,1,0,0,0)!=0:raise OSError('no-new-privileges failed')
os.execv('/opt/ipfs-supervisor/venv/bin/python',['/opt/ipfs-supervisor/venv/bin/python','-I','-B','/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py',*sys.argv[1:]])
''') + r'''
for name,text in (('owner-worker',OWNER_ENTRY),('worker-entry',worker),('probe-client',client)):
 path=root/'bin'/name;path.write_text(text);os.chown(path,0,0);path.chmod(0o555)
sudo=pathlib.Path('/etc/sudoers.d/numerical-ipc-qualification')
sudo.write_text('supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *, /opt/ipfs-supervisor/bin/probe-client *\n')
os.chown(sudo,0,0);sudo.chmod(0o440)
for name,mode in (('state',0o700),('repository',0o755),('worktrees',0o755),('worker-observations',0o750),('fixture-ipc',0o750)):
 path=root/name;path.mkdir(exist_ok=True);os.chown(path,1000,1000);path.chmod(mode)
for name in ('worker-result.json','router-observation.json'):
 path=root/'fixture-ipc'/name;path.touch();os.chown(path,1001,1000);path.chmod(0o660)
for name in ('.local','.duckdb','.ipfs_accelerate'):
 path=pathlib.Path('/home/supervisor')/name;path.mkdir(exist_ok=True);os.chown(path,1000,1000);path.chmod(0o700)
boundary=dict(schema='supervisor-container-worker-boundary@1',container_id=sys.argv[1],image_id=sys.argv[2],owner_uid=1000,worker_uid=1001,single_worker=True,provider='codex_cli',namespaces={name:os.readlink('/proc/self/ns/'+name) for name in ('pid','mnt','net')},allowed_worktree_roots=[str(root/'worktrees')],owner_private_paths=[str(root/'state')])
path=root/'container-boundary.json';path.write_text(json.dumps(boundary,sort_keys=True));os.chown(path,0,0);path.chmod(0o444)
print(json.dumps(dict(schema='numerical-ipc-root-setup@1',boundary=boundary,owner_entry_sha256=__import__('hashlib').sha256(OWNER_ENTRY.encode()).hexdigest(),worker_entry_production_sha256=__import__('hashlib').sha256(WORKER_ENTRY.encode()).hexdigest(),worker_entry_authored_provider_hook_sha256=__import__('hashlib').sha256(worker.encode()).hexdigest(),qualification_hook='Only fixture model inserts authored generate_text before actual router main; real OWNER_ENTRY unchanged')))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    assert output.parent == R and not output.exists()
    release = json.loads(args.release.read_bytes())
    assert release['provider_free_docker_qualification_authorized'] is True
    assert release['runner_bindings'] == {str(path): sha(path) for path in (Path(__file__), R/'owner_probe.py', R/'worker_probe.py')}
    assert all(sha(path) == digest for path, digest in release['source_bindings'].items())
    output.mkdir(mode=0o700)
    results = output/'results'; results.mkdir(mode=0o777); results.chmod(0o777)
    snapshot = output/'runner-snapshot'; snapshot.mkdir()
    for path in release['runner_bindings']:
        shutil.copyfile(path, snapshot/Path(path).name)
    container = None
    name = 'numerical-loop-ipc-' + str(time.time_ns())
    started = time.monotonic()
    record = dict(schema='numerical-worker-offline-docker-qualification@1', complete=False,
        image_id=IMAGE, provider_calls=0, model_calls=0, network='none', cpu_limit=5,
        memory_limit_bytes=12*1024**3, pids_limit=128, privileged=False, read_only_base=False,
        root_setup_required_for_existing_sudo_worker_transition=True,
        worker_no_new_privileges_and_zero_caps_required=True, container_name=name,
        full_runtime_deployment_qualified=False, official_terminal_bench_score=None,
        native_provider_tokens=None, paid_token_savings_measured=False)
    env = dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',
        IPFS_KIT_DISABLE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_DISABLE_TELEMETRY='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONHASHSEED='0',
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=ROOT+'/state/orchestration',
        IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=ROOT+'/state/seal.duckdb')
    command = ['docker','create','--name',name,'--network','none','--cpus','5','--memory','12g',
               '--pids-limit','128','--user','0']
    source = W/'.benchmarks/terminal-bench-2/largest-eigenval'
    mounts = [(P,ROOT+'/source',True),(D,ROOT+'/datasets',True),(K,ROOT+'/kit',True),
        (Path('/home/barberb/.duckdb/extensions/v1.5.5/linux_arm64'),'/extensions',True),
        (R/'owner_probe.py','/qualification/owner_probe.py',True),
        (R/'worker_probe.py','/qualification/worker_probe.py',True),
        (source/'environment/src/eigen.py','/permitted/eigen.py',True),
        (source/'environment/src/eval.py','/permitted/eval.py',True),
        (source/'instruction.md','/permitted/instruction.md',True),
        (W/'artifacts/terminal-suite-eigenval-pilot-20261004/largest-eigenval-profile.json','/permitted/profile.json',True),
        (results,'/results',False)]
    for host,target,readonly in mounts:
        assert host.exists()
        command += ['--mount',f'type=bind,src={host},dst={target}'+(',readonly' if readonly else '')]
    for key,value in env.items(): command += ['--env',key+'='+value]
    command += ['--entrypoint','/bin/sleep',IMAGE,'900']
    write(output/'command.json',dict(command=command,environment=env,release_sha256=sha(args.release),
        runner_bindings=release['runner_bindings'],source_bindings=release['source_bindings'],
        scope='Actual generated owner/worker launch with authored provider hook; no native model decision.'))
    (output/'root-setup.py').write_text(SETUP)
    try:
        actual_image = subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
        assert actual_image == IMAGE
        container = subprocess.check_output(command,text=True,timeout=30).strip()
        record['container_id'] = container
        subprocess.run(['docker','start',container],check=True,capture_output=True,timeout=30)
        setup = subprocess.run(['docker','exec','--user','0',container,PYTHON,'-I','-B','-c',SETUP,container,IMAGE],capture_output=True,text=True,timeout=60)
        (output/'root-setup.stdout').write_text(setup.stdout)
        (output/'root-setup.stderr').write_text(setup.stderr)
        setup.check_returncode()
        inspect = json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]
        controls = {key:inspect['HostConfig'][key] for key in ('NanoCpus','Memory','PidsLimit','NetworkMode','Privileged','ReadonlyRootfs')}
        assert controls == dict(NanoCpus=5_000_000_000,Memory=12*1024**3,PidsLimit=128,NetworkMode='none',Privileged=False,ReadonlyRootfs=False)
        write(output/'container-controls.json',dict(controls=controls,mounts=[{key:row[key] for key in ('Type','Source','Destination','RW')} for row in inspect['Mounts']]))
        argv = ['docker','exec','--user','1000:1000','--env','HOME=/home/supervisor',container,PYTHON,'-I','-B','/qualification/owner_probe.py']
        write(output/'owner-command.json',dict(argv=argv))
        with (output/'owner.stdout').open('w') as stdout,(output/'owner.stderr').open('w') as stderr:
            completed = subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=600)
        record['owner_returncode'] = completed.returncode
        if (results/'probe-qualification.json').exists():
            probe=json.loads((results/'probe-qualification.json').read_bytes());record['probe']=probe
            record['complete']=completed.returncode==0 and probe['complete']
    except BaseException as error:
        record.update(error_type=type(error).__name__,error=str(error)[:500])
        (output/'failure.txt').write_text(traceback.format_exc())
    finally:
        if container is not None:
            for path,target in ((ROOT+'/state','private-state'),(ROOT+'/fixture-ipc','private-fixture-ipc'),
                                (ROOT+'/worker-observations','private-worker-observations')):
                result=subprocess.run(['docker','cp',container+':'+path,str(output/target)],capture_output=True,text=True,timeout=45)
                record[target+'_retained']=result.returncode==0
            subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=30)
        remaining=subprocess.check_output(['docker','container','ls','-a','--filter','name='+name,'--format','{{.Names}}'],text=True).strip()
        record['container_removed']=remaining==''
        record['source_unchanged']=all(sha(path)==digest for path,digest in release['source_bindings'].items())
        record['runner_unchanged']=all(sha(path)==digest for path,digest in release['runner_bindings'].items())
        record['complete']=record['complete'] and record['container_removed'] and record['source_unchanged'] and record['runner_unchanged']
        record['seconds']=time.monotonic()-started
        write(output/'qualification.json',record)
        print(json.dumps({key:record.get(key) for key in ('complete','seconds','owner_returncode','error_type','source_unchanged','container_removed')}))
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
