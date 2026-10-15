"""Run the full300 native preview and signed companion in an offline container."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import time

from . import run_codebase_inventory_resume_docker as deployment
from . import run_codebase_signed_successor_docker as signed
from . import source_successor_full_scan_fixture as shared

SOURCE = Path(__file__).resolve().parents[3]
MODULE = 'benchmarks.agent_supervisor.container_coding.qualify_codebase_full300_preview'
DEFAULT_HOST = Path('/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/successor-expansion-resources-20261003-01/configuration.json')
_RETAINED_HOST_LEASES = {}


def _write(path, value):
    signed._write(path, value)


def _observe_resources(container):
    observation_program = (
        "import json,os,pathlib;root=pathlib.Path('/sys/fs/cgroup');"
        "names=('memory.current','memory.peak','memory.events','memory.swap.current','pids.current','pids.peak','cpu.stat');"
        "values={name:(root/name).read_text().strip() for name in names};"
        "print(json.dumps({'schema':'full300-preview-container-resource-observation@1',"
        "'uid':os.getuid(),'values':values,'namespaces':{name:os.readlink('/proc/self/ns/'+name)"
        " for name in ('pid','mnt','net')}},sort_keys=True))")
    observation = subprocess.run(['docker', 'exec', '--user', '1000:1000', container,
        deployment.WORKER_ROOT + '/venv/bin/python', '-I', '-B', '-c', observation_program],
        capture_output=True, text=True, check=True, timeout=15)
    if len(observation.stdout.encode()) > 16384:
        raise ValueError('bounded owned-container resource observation required')
    return json.loads(observation.stdout)


def run(output, *, image, datasets, kit, lean, host_configuration=DEFAULT_HOST, native_seconds=3600):
    if type(native_seconds) is not int or not 950 <= native_seconds <= 3600:
        raise ValueError('bounded fresh native deadline required')
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve(strict=True) != output.parent:
        raise ValueError('fresh canonical container output required')
    datasets, kit, lean = [Path(value).resolve(strict=True) for value in (datasets, kit, lean)]
    sys.path.insert(0, str(datasets))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as scheduling
    if Path(scheduling.__file__).resolve(strict=True) != datasets / 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py':
        raise ValueError('host scheduler must come from the selected datasets source')
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
    from .container_worker_deployment import OWNER_ENTRY, VALIDATION_ENTRY, WORKER_ENTRY
    host_raw = host_configuration.read_bytes()
    host = json.loads(host_raw)
    if hashlib.sha256(host_raw).hexdigest() != 'c60d91943187515915b8c4c0d0ace3f3703f7de4da1960f09e49371a8cdf6575':
        raise ValueError('shared host configuration differs')
    state_path = Path(host['state_path'])
    expected_configuration = host['persisted_config']
    for path in (state_path, state_path.with_name(state_path.name + '.lock')):
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink()
                or path.resolve(strict=True) != path or metadata.st_uid != os.getuid()):
            raise ValueError('existing canonical owner-controlled MAIN state and lock required')

    class ExactExistingScheduler(scheduling.GlobalResourceScheduler):
        def _new_state(self):
            raise scheduling.ResourceConfigurationError('MAIN state absent; pool creation refused')

        def _validate_state_configuration(self, state, *, allow_reconfigure=False):
            if (type(state) is not dict
                    or state.get('schema_version') != scheduling.RESOURCE_SCHEDULER_SCHEMA_VERSION
                    or state.get('config') != expected_configuration):
                raise scheduling.ResourceConfigurationError('MAIN configuration differs; reconfiguration refused')
            return super()._validate_state_configuration(state, allow_reconfigure=False)

    scheduler = ExactExistingScheduler(scheduling.ResourceSchedulerConfig(
        **expected_configuration, state_path=state_path,
        lease_ttl_seconds=host['lease_ttl_seconds'], auto_renew_leases=host['auto_renew_leases']))
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True, timeout=15).strip()
    output.mkdir()
    started = time.monotonic()
    selected = {
        'source': deployment._snapshot(SOURCE, output / 'source', prefixes=('ipfs_accelerate_py', 'benchmarks', 'test')),
        'datasets': deployment._snapshot(datasets, output / 'datasets', prefixes=('ipfs_datasets_py',)),
        'kit': deployment._snapshot(kit, output / 'kit', prefixes=('ipfs_kit_py',)),
    }
    _write(output / 'selected-source-snapshot.json', selected)
    deployment._verify_snapshot(selected, output)
    _write(output / 'shared-host-configuration.json', host)
    scripts = output / 'deployment'
    scripts.mkdir()
    worker_entry = WORKER_ENTRY
    for name, content in (('owner-worker', OWNER_ENTRY), ('validation-worker', VALIDATION_ENTRY), ('worker-entry', worker_entry)):
        (scripts / name).write_text(content)
    (scripts / 'sudoers').write_text('supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *\n')
    name = 'ipfs-full300-preview-' + str(time.time_ns())
    record = {'schema': 'full300-preview-fresh-offline-container-execution@1', 'qualified': False,
        'image_id': image_id, 'container_name': name, 'network': 'none', 'privileged': False,
        'cpu_limit': 12, 'memory_limit_bytes': 8 * 1024**3, 'pids_limit': 512,
        'native_total_timeout_seconds': native_seconds, 'host_scheduler_state_path': str(scheduler.state_path),
        'host_configuration_pin': shared.pin(host_raw), 'inherited_scan_pages': 0, 'inherited_setup_epochs': 0,
        'scope': 'fresh300_member_cpu8d_full_default_optout_pages_and_signed_preview_without_worker',
        'proof_authority': False, 'execution_attestation': False, 'production_activated': False,
        'large300_member_scan_qualified': False, 'worker_dispatch_qualified': False,
        'task_installation_qualified': False, 'worker_launched': False}
    lease = container = None
    try:
        record['host_resources_before_reservation'] = scheduler.snapshot()
        lease = scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=12, memory_mb=8192,
            child_process_slots=12, timeout=90, request_id='fresh-preview-successor-container')
        record['host_reservation'] = lease.to_dict()
        record['container_creation_attempted'] = True
        mounts = ((output / 'source', deployment.WORKER_ROOT + '/source'),
            (output / 'datasets', deployment.WORKER_ROOT + '/datasets'),
            (output / 'kit', deployment.WORKER_ROOT + '/kit'), (lean, '/toolchains/lean'))
        command = ['docker', 'create', '--network', 'none', '--cpus', '12', '--memory', '8g',
            '--pids-limit', '512', '--user', '0', '--name', name]
        for local, destination in mounts:
            command += ['-v', str(local) + ':' + destination + ':ro']
        command += ['--entrypoint', '/bin/sleep', image_id, str(native_seconds + 180)]
        container = subprocess.check_output(command, text=True, timeout=30).strip()
        record['container_id'] = container
        subprocess.run(['docker', 'start', container], check=True, capture_output=True, timeout=30)
        record['actual_container_limits'] = signed._observe_container_limits(container, name)
        record['actual_container_resources_before_native'] = _observe_resources(container)
        observed_mounts = record['actual_container_limits']['inspection']['Mounts']
        for local, destination in mounts:
            matching = [row for row in observed_mounts if row['Destination'] == destination]
            if len(matching) != 1 or matching[0]['Type'] != 'bind' or matching[0]['Source'] != str(local) or matching[0]['RW'] is not False:
                raise ValueError('actual selected read-only mount differs')
        for filename in ('owner-worker', 'validation-worker', 'worker-entry'):
            subprocess.run(['docker', 'cp', str(scripts / filename), container + ':' + deployment.WORKER_ROOT + '/bin/' + filename],
                check=True, capture_output=True, timeout=30)
        subprocess.run(['docker', 'cp', str(scripts / 'sudoers'), container + ':/etc/sudoers.d/finite-worker'],
            check=True, capture_output=True, timeout=30)
        setup = (
            "import json,os,pathlib;root=pathlib.Path('/opt/ipfs-supervisor');"
            "[(os.chown(root/'bin'/n,0,0),(root/'bin'/n).chmod(0o755)) for n in ('owner-worker','validation-worker','worker-entry')];"
            "p=pathlib.Path('/etc/sudoers.d/finite-worker');os.chown(p,0,0);p.chmod(0o440);"
            "results=pathlib.Path('/results');results.mkdir();os.chown(results,1000,1000);results.chmod(0o755);"
            "boundary={'schema':'supervisor-container-worker-boundary@1',"
            f"'container_id':{container!r},'image_id':{image_id!r},"
            "'owner_uid':1000,'worker_uid':1001,'single_worker':True,"
            "'namespaces':{k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')},"
            "'allowed_worktree_roots':['/opt/ipfs-supervisor/worktrees'],"
            "'owner_private_paths':['/opt/ipfs-supervisor/state','/results/native/private','/home/supervisor'],"
            "'validation_repository_roots':['/results/native/repository']};"
            "p=root/'container-boundary.json';p.write_text(json.dumps(boundary,sort_keys=True));p.chmod(0o444)")
        subprocess.run(['docker', 'exec', '--user', '0', container, '/usr/local/bin/python', '-I', '-c', setup],
            check=True, capture_output=True, timeout=30)
        record['offline_preflight'] = deployment._preflight(container, output, image_id)
        parent = {'host_scheduler_state_path': str(scheduler.state_path), 'host_configuration_pin': shared.pin(host_raw),
            'host_reservation': lease.to_dict(), 'container_id': container, 'cpu_limit': 12,
            'memory_limit_bytes': 8 * 1024**3, 'pids_limit': 512}
        authority_program = (
            "import sys,pathlib,json,os;root=pathlib.Path('/opt/ipfs-supervisor');"
            "sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')];"
            "from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceSchedulerConfig,GlobalResourceScheduler;"
            "state=pathlib.Path('/results/container-resource-admission.json');"
            "config=ResourceSchedulerConfig.for_proof_host(state_path=state,lane_reservations={},auto_renew_leases=True);"
            "config.total_cpu_slots=min(config.total_cpu_slots,12);config.total_memory_mb=min(config.total_memory_mb,8192);"
            "config.total_child_process_slots=min(config.total_child_process_slots,12);config.total_gpu_memory_mb=None;"
            "config.total_unified_memory_mb=config.total_memory_mb;config.validate();scheduler=GlobalResourceScheduler(config);"
            f"parent={parent!r};"
            "authority={'schema':'successor-dispatch-container-resource-authority@1','state_path':str(state),"
            "'persisted_config':config.persisted_dict(),'auto_renew_leases':True,'lease_ttl_seconds':config.lease_ttl_seconds,"
            "'parent_host_envelope':parent,'namespace':{k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')},"
            "'scope':'local PID accounting inside held host envelope and actual CPU RAM PID cgroup',"
            "'independent_whole_host_pool':False,'shares_host_pid_state':False,'proof_authority':False,"
            "'initial_resources':scheduler.snapshot()};"
            "path=pathlib.Path('/results/container-resource-authority.json');"
            "raw=json.dumps(authority,sort_keys=True,indent=2,allow_nan=False).encode()+b'\\n';"
            "stream=path.open('xb');stream.write(raw);stream.flush();os.fsync(stream.fileno());stream.close();path.chmod(0o444);"
            "print(json.dumps(authority,sort_keys=True),flush=True)")
        authority = subprocess.run(['docker', 'exec', '--user', '1000:1000', container,
            deployment.WORKER_ROOT + '/venv/bin/python', '-I', '-B', '-c', authority_program],
            check=True, capture_output=True, text=True, timeout=30)
        record['container_resource_authority'] = json.loads(authority.stdout)
        with (output / 'run.stdout').open('xb') as stdout, (output / 'run.stderr').open('xb') as stderr:
            result = subprocess.run(['docker', 'exec', '--user', '1000:1000',
                '-e', 'DOCTOR_COMPOSITION_LEAN=/toolchains/lean/bin/lean', '-e', 'IPFS_ACCELERATE_RUN_LIVE_QUACK=1',
                '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
                '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'NUMEXPR_NUM_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                container, deployment.WORKER_ROOT + '/venv/bin/python', '-B', '-m', MODULE,
                '--output', '/results/native', '--overall-seconds', str(native_seconds - 50)],
                stdout=stdout, stderr=stderr, timeout=native_seconds)
        record['returncode'] = result.returncode
        resources = _observe_resources(container)
        record['actual_container_resources_after_native'] = resources
        if (resources['uid'] != 1000 or resources['namespaces'] != record['actual_container_limits']['cgroup']['namespaces']):
            raise ValueError('owned-container resource observation namespace differs')
        events = dict((name, int(value)) for name, value in
            (line.split() for line in resources['values']['memory.events'].splitlines()))
        peak = int(resources['values']['memory.peak'])
        if peak <= 0 or peak > 8 * 1024**3 or events.get('oom') != 0 or events.get('oom_kill') != 0:
            raise ValueError('owned-container peak memory or OOM evidence exceeds envelope')
        record['actual_container_peak_memory_bytes'] = peak
        record['actual_container_oom_events'] = events
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        def cleanup_error(stage, error):
            record['cleanup_error_count'] = record.get('cleanup_error_count', 0) + 1
            errors = record.setdefault('cleanup_errors', [])
            if len(errors) < 32:
                errors.append({'stage': stage,
                    'error_type': type(error).__name__, 'error': str(error)[:2048]})

        try:
            if container is not None:
                try:
                    copied = subprocess.run(['docker', 'cp', container + ':/results/.', str(output)],
                        capture_output=True, text=True, timeout=60)
                    record['container_results_copied'] = copied.returncode == 0
                except (OSError, subprocess.TimeoutExpired) as error:
                    cleanup_error('copy_results', error)
        finally:
            if record.get('container_creation_attempted'):
                # Keep this process and its renewing reservation alive until
                # the exact owned container is proved absent, even if copying
                # evidence or the Docker engine temporarily fails.
                _RETAINED_HOST_LEASES[name] = lease
                while True:
                    try:
                        removed = deployment._remove_container(name)
                        absent = subprocess.run(['docker', 'ps', '-a', '-q', '--no-trunc',
                            '--filter', 'name=^/' + name + '$'], capture_output=True, text=True, timeout=15)
                        record['container_removed'] = removed and absent.returncode == 0 and not absent.stdout.strip()
                        record['container_absence_observation'] = {'returncode': absent.returncode,
                            'stdout': absent.stdout[:16384], 'stderr': absent.stderr[:16384]}
                    except BaseException as error:
                        record['container_removed'] = False
                        cleanup_error('remove_owned_container', error)
                    if record.get('container_removed') is True:
                        break
                    record['host_reservation_retained_for_live_container'] = True
                    try:
                        progress = output / 'cleanup-progress.json'
                        progress.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n')
                    except BaseException as error:
                        cleanup_error('persist_cleanup_progress', error)
                    try:
                        time.sleep(15)
                    except BaseException as error:
                        cleanup_error('wait_for_owned_container_cleanup', error)
            if lease is not None:
                lease.release()
                _RETAINED_HOST_LEASES.pop(name, None)
        record['host_reservation_released'] = lease is not None and lease.released
        record['host_resources_after_cleanup'] = scheduler.snapshot()
        record['elapsed_seconds'] = time.monotonic() - started
        try:
            deployment._verify_snapshot(selected, output)
            record['retained_source_verified_after_execution'] = True
        except BaseException as error:
            record['retained_source_verified_after_execution'] = False
            cleanup_error('verify_retained_sources', error)
        changes = [row['path'] for rows in selected.values() for row in rows
            if not Path(row['path']).is_file() or hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']]
        record['original_selected_source_changes_after_execution'] = changes
        native_result = output / 'native/result.json'
        record['qualified'] = (record.get('returncode') == 0 and record.get('container_removed') is True
            and record.get('container_results_copied') is True
            and record['host_reservation_released'] and not changes and native_result.is_file()
            and record['retained_source_verified_after_execution'] and not record.get('cleanup_errors')
            and type(record.get('actual_container_peak_memory_bytes')) is int
            and json.loads(native_result.read_bytes()).get('qualified') is True)
        if record['qualified']:
            evidence = json.loads(native_result.read_bytes())
            record['large300_member_scan_qualified'] = evidence.get('large300_member_scan_qualified') is True
            record['signed_preview_companion_qualified'] = evidence.get('signed_preview_companion_qualified') is True
            record['qualified'] = (record['large300_member_scan_qualified']
                and record['signed_preview_companion_qualified']
                and evidence.get('worker_launched') is False
                and evidence.get('task_installation_qualified') is False
                and all(evidence.get(key) is False for key in ('native_worker_qualified',
                    'publication_qualified', 'semantic_reproof_join_qualified', 'cuda_qualified',
                    '384d_qualified', '768d_qualified', '4096d_qualified', 'proof_authority')))
        copied_root = output / 'native'
        copied_files = copied_bytes = copied_symlinks = 0
        if copied_root.is_dir():
            for directory, names, filenames in os.walk(copied_root, followlinks=False):
                for filename in filenames:
                    metadata = (Path(directory) / filename).lstat()
                    if stat.S_ISREG(metadata.st_mode):
                        copied_files += 1
                        copied_bytes += metadata.st_size
                    elif stat.S_ISLNK(metadata.st_mode):
                        copied_symlinks += 1
        record['copied_native_artifact_accounting'] = {'regular_files': copied_files,
            'regular_file_bytes': copied_bytes, 'symlink_files': copied_symlinks,
            'scope': 'measured copied native tree; no enforced disk byte cap claimed'}
        _write(output / 'container-execution-final.json', record)
    if not record['qualified']:
        raise RuntimeError('fresh native preview qualification failed; retained evidence identifies the stage')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--image', default='ipfs-supervisor-preview-worker:20261005-r3')
    parser.add_argument('--datasets', type=Path, default=SOURCE.parent / 'ipfs_datasets')
    parser.add_argument('--kit', type=Path, default=SOURCE.parent / 'ipfs_kit')
    parser.add_argument('--lean', type=Path, default=Path('/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1'))
    parser.add_argument('--host-configuration', type=Path, default=DEFAULT_HOST)
    parser.add_argument('--native-seconds', type=int, default=3600)
    args = parser.parse_args()
    report = run(args.output, image=args.image, datasets=args.datasets, kit=args.kit, lean=args.lean,
        host_configuration=args.host_configuration, native_seconds=args.native_seconds)
    print(json.dumps({key: report.get(key) for key in ('qualified', 'returncode', 'container_removed', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
