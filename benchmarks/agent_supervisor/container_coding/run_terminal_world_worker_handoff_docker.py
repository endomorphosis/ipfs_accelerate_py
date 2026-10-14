"""Retain and execute one offline worker observation under an existing host lease.

The caller must hold the native host resource lease. This runner never creates
another host scheduler, installs dependencies, pulls images, or enables models.
Its container-local scheduler serves only the authored native fixture.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import uuid

IMAGE = 'sha256:771900c18b663b33a5bf96e066346b2d5c842506523efb167aebe8d3a99b5c07'
DS_HEAD = '8fd82826e8ee935270457acfeee185054eb44aed'
KIT_HEAD = '40c98c71b95be750509ffdb21e7e5494d91b4e79'
SOURCE = Path(__file__).resolve().parents[3]
PYTHON = '/opt/ipfs-supervisor/venv/bin/python'
ENTRY = '/usr/local/libexec/ipfs-world-handoff-observe-worker'
EXPERIMENT = 'benchmarks/agent_supervisor/container_coding/terminal_world_worker_handoff_experiment.py'
ENV = dict(IPFS_ACCEL_SKIP_CORE='1', IPFS_KIT_DISABLE='1',
    IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS='0', HF_HUB_OFFLINE='1',
    HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='')


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def _snapshot(package, destination, prefixes):
    """Selected source copies; pins do not claim a transitive import census."""
    destination.mkdir(mode=0o755)
    rows = []
    for prefix in prefixes:
        base = package / prefix
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(package)
            if any(part in {'__pycache__', '.git', '.venv', 'node_modules', 'workspace', 'evidence'}
                   for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file() or path.suffix not in {
                    '.py', '.json', '.yaml', '.yml', '.toml', '.sql', '.lean', '.lgt'}:
                continue
            if path.stat().st_size > 4 * 1024**2:
                continue
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError('snapshot_source_not_regular')
                with os.fdopen(fd, 'rb', closefd=False) as stream:
                    raw = stream.read(4 * 1024**2 + 1)
                identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
                if len(raw) != before.st_size or identity(before) != identity(os.fstat(fd)) or identity(before) != identity(path.lstat()):
                    raise ValueError('snapshot_source_changed')
            finally:
                os.close(fd)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            target.chmod(0o444)
            rows.append(dict(relative=relative.as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest()))
    if not rows:
        raise ValueError('empty_package_snapshot')
    for path in (destination, *(p for p in destination.rglob('*') if p.is_dir())):
        path.chmod(0o755)
    return rows


def _worker_source():
    tree = ast.parse((SOURCE / EXPERIMENT).read_bytes())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WORKER_ENTRY_SOURCE' for t in node.targets):
            value = ast.literal_eval(node.value)
            if type(value) is str and len(value.encode()) < 8192:
                return value
    raise ValueError('fixed_worker_source_missing')


def _remove_owned_container(*, name, run_id, container_id, output):
    observed = subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
    if observed.returncode != 0:
        check = subprocess.run(['docker','ps','-a','-q','--filter','name=^/' + name + '$'],
                               capture_output=True,text=True,timeout=10)
        return check.returncode == 0 and not check.stdout.strip(), 'absence_observed'
    row = json.loads(observed.stdout)[0]
    labels = row['Config'].get('Labels') or {}
    mounts = {m['Destination']: m['Source'] for m in row['Mounts'] if m['Type'] == 'bind'}
    if (row['Name'] != '/' + name or row['Image'] != IMAGE
            or labels.get('ipfs.world-handoff.run') != run_id
            or (container_id is not None and row['Id'] != container_id)
            or mounts.get('/qualification') != str(output / 'assets')):
        return False, 'foreign_or_uncertain_container_preserved'
    owned_id = row['Id']
    subprocess.run(['docker','rm','-f',owned_id],capture_output=True,timeout=20)
    check = subprocess.run(['docker','ps','-a','-q','--no-trunc','--filter','id=' + owned_id],
                           capture_output=True,text=True,timeout=10)
    # An unrelated replacement of the old name is left alone.
    return check.returncode == 0 and not check.stdout.strip(), 'exact_owned_id_cleanup'


def run(*, output, datasets, kit):
    if os.getuid() != 1000 or output.exists() or not output.parent.is_dir():
        raise ValueError('fresh_owned_output_required')
    for root, head in ((datasets, DS_HEAD), (kit, KIT_HEAD)):
        observed = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if observed != head:
            raise ValueError('dependency_head_changed')
    output.mkdir(mode=0o700)
    assets = output / 'assets'; assets.mkdir(mode=0o755)
    results = output / 'results'; results.mkdir(mode=0o700)
    scratch = output / 'scratch'; scratch.mkdir(mode=0o700)
    selected = {
        'acc': _snapshot(SOURCE, assets / 'acc', ('ipfs_accelerate_py', 'benchmarks', 'test')),
        'ds': _snapshot(datasets, assets / 'ds', ('ipfs_datasets_py',)),
        'kit': _snapshot(kit, assets / 'kit', ('ipfs_kit_py',)),
    }
    _write(output / 'source-snapshots.json', dict(scope='selected_readonly_copies_not_complete_provenance',
        dependency_heads=dict(datasets=DS_HEAD, kit=KIT_HEAD), files=selected))
    worker = _worker_source()
    (assets / 'worker.py').write_text(worker); (assets / 'worker.py').chmod(0o444)
    owner = '''import json,os,pathlib,sys,types,importlib.machinery
sys.path[:0]=['/opt/ipfs-supervisor/source','/opt/ipfs-supervisor/datasets','/opt/ipfs-supervisor/kit']
os.environ['IPFS_DATASETS_RESOURCE_SCHEDULER_PATH']='/tmp/world-handoff-fixture-resources-1000.json'
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as native
config=native.ResourceSchedulerConfig(total_cpu_slots=1,total_memory_mb=1536,total_child_process_slots=3,
 lane_reservations={'validation':{'cpu_slots':1,'memory_mb':0}},state_path=os.environ['IPFS_DATASETS_RESOURCE_SCHEDULER_PATH'])
native.default_resource_scheduler_config=lambda:config
native._GLOBAL_SCHEDULERS[str(pathlib.Path(config.state_path).resolve())]=native.GlobalResourceScheduler(config)
namespace=types.ModuleType('benchmarks');namespace.__path__=['/opt/ipfs-supervisor/source/benchmarks'];namespace.__package__='benchmarks'
namespace.__spec__=importlib.machinery.ModuleSpec('benchmarks',loader=None,is_package=True)
namespace.__spec__.submodule_search_locations=namespace.__path__;sys.modules['benchmarks']=namespace
from benchmarks.agent_supervisor.container_coding import terminal_world_worker_handoff_experiment as experiment
original_cycle=experiment._cycle
def retain_failure(fixture):
 try: return original_cycle(fixture)
 except Exception:
  import traceback
  diagnostic=pathlib.Path('/world-qualification/results/private-failure-traceback.txt')
  diagnostic.write_text(traceback.format_exc());diagnostic.chmod(0o600)
  raise
experiment._cycle=retain_failure
report=experiment.run_experiment(scratch_root=pathlib.Path('/world-qualification/scratch'))
path=pathlib.Path('/world-qualification/results/measured.json');path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\\n');path.chmod(0o600)
pin=experiment.report_sha256(report);experiment.validate_report(report,expected_report_sha256=pin)
pathlib.Path('/world-qualification/results/report-pin.txt').write_text(pin+'\\n')
if report.get('qualified') is not True: raise RuntimeError('inner_worker_handoff_unqualified')
'''
    (assets / 'owner.py').write_text(owner); (assets / 'owner.py').chmod(0o444)
    setup = f'''import os,pathlib,pwd,subprocess
entry=pathlib.Path({ENTRY!r});entry.parent.mkdir(mode=0o755,parents=True,exist_ok=True)
entry.write_bytes(pathlib.Path('/qualification/worker.py').read_bytes());entry.chmod(0o555);os.chown(entry,0,0)
rule=pathlib.Path('/etc/sudoers.d/world-handoff-observer');rule.write_text('supervisor ALL=(benchmarkworker:supervisor) NOPASSWD: {ENTRY} ""\\n');rule.chmod(0o440)
subprocess.run(['/usr/sbin/visudo','-cf',str(rule)],check=True,capture_output=True)
for name in ('results','scratch'):
 path=pathlib.Path('/world-qualification')/name;path.mkdir(mode=0o700,parents=True);os.chown(path,1000,1000)
os.initgroups('supervisor',1000);os.setgid(1000);os.setuid(1000)
environment=dict(os.environ);environment['HOME']=pwd.getpwnam('supervisor').pw_dir
os.execve({PYTHON!r},[{PYTHON!r},'-I','/qualification/owner.py'],environment)
'''
    (assets / 'setup.py').write_text(setup); (assets / 'setup.py').chmod(0o444)
    run_id = uuid.uuid4().hex
    name = 'world-handoff-qualification-' + run_id
    container_id = None
    record = dict(schema='world-worker-container-qualification@1', image=IMAGE, container_name=name,
        cpu_slots=1, memory_mb=2048, child_process_slots=3, network='none', packages_installed=False,
        image_pulled=False, model_calls=0, training_steps=0, host_lease_required=True,
        container_local_scheduler_scope='authored_native_fixture_only_not_host_resource_authority',
        source_mounts_readonly=True, host_home_mounted=False, docker_socket_mounted=False,
        private_failed_scratch_retained=False, container_results_exported=False,
        container_removed=False, returncode=None, worker_entry_sha256=hashlib.sha256(worker.encode()).hexdigest())
    try:
        inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE], text=True))[0]
        if inspected['Id'] != IMAGE:
            raise ValueError('cached_image_pin_changed')
        command = ['docker', 'create', '--name', name, '--label', 'ipfs.world-handoff.qualification=observation',
            '--label', 'ipfs.world-handoff.run=' + run_id,
            '--network', 'none', '--cpus', '1', '--memory', '2g', '--memory-swap', '2g', '--pids-limit', '48',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m', '--user', '0:0',
            '--mount', f'type=bind,src={assets},dst=/qualification,readonly',
            '--entrypoint', PYTHON]
        for key, value in ENV.items():
            command += ['--env', key + '=' + value]
        for key, destination in (('acc', 'source'), ('ds', 'datasets'), ('kit', 'kit')):
            command += ['--mount', f'type=bind,src={assets / key},dst=/opt/ipfs-supervisor/{destination},readonly']
        command += [IMAGE, '-I', '/qualification/setup.py']
        container_id = subprocess.check_output(command, text=True, timeout=30).strip()
        record['container_id'] = container_id
        inspection = json.loads(subprocess.check_output(['docker', 'inspect', container_id], text=True))[0]
        if (inspection['Id'] != container_id or inspection['Name'] != '/' + name
                or inspection['Image'] != IMAGE
                or inspection['Config']['Labels'].get('ipfs.world-handoff.run') != run_id):
            raise ValueError('created_container_identity_changed')
        record['created_container_identity_verified'] = True
        host = inspection['HostConfig']
        record['observed_engine_limits'] = dict(network=host['NetworkMode'], nano_cpus=host['NanoCpus'],
            memory_bytes=host['Memory'], memory_swap_bytes=host['MemorySwap'], pids_limit=host['PidsLimit'])
        if record['observed_engine_limits'] != dict(network='none',nano_cpus=1000000000,
                memory_bytes=2147483648,memory_swap_bytes=2147483648,pids_limit=48):
            raise ValueError('engine_envelope_changed')
        with (output / 'container.stdout.log').open('xb') as out, (output / 'container.stderr.log').open('xb') as err:
            result = subprocess.run(['docker','start','-a',container_id],stdout=out,stderr=err,timeout=180)
        record['returncode'] = result.returncode
    finally:
        if record.get('created_container_identity_verified') is True:
            # Quiesce only the exact owned ID before exporting failed evidence.
            state = json.loads(subprocess.check_output(['docker','inspect',container_id],text=True))[0]['State']
            if state['Running']:
                subprocess.run(['docker','stop','--time','3',container_id],capture_output=True,timeout=10)
            state = json.loads(subprocess.check_output(['docker','inspect',container_id],text=True))[0]['State']
            record['container_stopped_before_export'] = state['Running'] is False
            for destination, field in ((results,'container_results_exported'),(scratch,'private_failed_scratch_retained')):
                if record['container_stopped_before_export']:
                    copied = subprocess.run(['docker','cp',container_id + ':/world-qualification/' + destination.name + '/.',
                        str(destination)],capture_output=True,timeout=30)
                    record[field] = copied.returncode == 0
        if (record.get('created_container_identity_verified') is True
                and (not record['container_results_exported'] or not record['private_failed_scratch_retained'])):
            record['container_cleanup_scope'] = 'owned_container_preserved_for_uncertain_evidence_export'
        else:
            record['container_removed'], record['container_cleanup_scope'] = _remove_owned_container(
                name=name,run_id=run_id,container_id=container_id,output=output)
        _write(output / 'container-execution-final.json',record)
    if (record['returncode'] != 0 or not record['container_removed']
            or not record['container_results_exported'] or not record['private_failed_scratch_retained']):
        raise ValueError('worker_container_qualification_failed')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--datasets',type=Path,required=True)
    parser.add_argument('--kit',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(output=args.output.absolute(),datasets=args.datasets.absolute(),kit=args.kit.absolute()),sort_keys=True))


if __name__ == '__main__':
    main()
