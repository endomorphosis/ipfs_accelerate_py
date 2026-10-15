"""Own fresh full300 selected-successor scans and paired warm/cold receiving in an offline container."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import signal
import threading
from unittest.mock import patch
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time

from . import run_codebase_inventory_resume_docker as deployment
from . import run_codebase_signed_successor_docker as signed
from . import source_successor_full_scan_fixture as shared

SOURCE = Path(__file__).resolve().parents[3]
MODULE = 'benchmarks.agent_supervisor.container_coding.qualify_codebase_full300_shared_successor_receiving'
DEFAULT_HOST = Path('/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/successor-expansion-resources-20261003-01/configuration.json')
_RETAINED_HOST_LEASES = {}
SOURCE_ROOT = Path('/home/barberb/lift_coding/.worktrees/codebase-shared-qualification-20261007-04')
ISOLATION_MANIFEST = SOURCE_ROOT / 'qualification-source-manifest.json'
EXPECTED_IMAGE = 'sha256:88463ed5d1f72a79fdd4bac5e2ad1c2a0490c1ae24b21365d14eacd5e2a0a879'
CONFIG_SHA = 'c60d91943187515915b8c4c0d0ace3f3703f7de4da1960f09e49371a8cdf6575'
ORIGINAL_ROOT = Path('/home/barberb/lift_coding')
MANIFEST_SCOPE = ('chosen regular-file snapshot; original shared checkout currentness recorded '
    'at capture only; sequential observations')


def _read(path, maximum=4 * 1024**2):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError('canonical bounded ordinary source locator required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.getuid() or not 0 <= before.st_size <= maximum):
            raise ValueError('bounded owned single-link regular source required')
        raw = stream.read(maximum + 1)
        after, named = os.fstat(stream.fileno()), path.lstat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink')
        if len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key)
                or getattr(before, key) != getattr(named, key) for key in fields):
            raise ValueError('ordinary source changed during its held read')
    return raw


def _pin(path):
    path = Path(path).absolute()
    maximum = 16 * 1024**2 if path == ISOLATION_MANIFEST else (65536 if path == DEFAULT_HOST else 4 * 1024**2)
    raw = _read(path, maximum)
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _same_pin(item):
    try:
        return _pin(Path(item['path'])) == item
    except (OSError, ValueError):
        return False


def _parse(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate qualification JSON key')
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError('nonfinite qualification JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def _pinned_json(path, maximum=4 * 1024**2):
    item = _pin(path)
    raw = _read(path, maximum)
    if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('recorded JSON changed between bounded observations')
    return _parse(raw), item


def _manifest():
    value, before = _pinned_json(ISOLATION_MANIFEST, 16 * 1024**2)
    if (type(value) is not dict or set(value) != {'schema', 'source_root', 'original_workspace_root',
            'created_utc', 'scope', 'packages'}
            or value['schema'] != 'isolated-codebase-shared-source-capture@1'
            or value['source_root'] != str(SOURCE_ROOT) or value['original_workspace_root'] != str(ORIGINAL_ROOT)
            or value['scope'] != MANIFEST_SCOPE or type(value['created_utc']) is not str
            or not value['created_utc'] or type(value['packages']) is not dict
            or set(value['packages']) != {'ipfs_accelerate', 'ipfs_datasets', 'ipfs_kit'}):
        raise ValueError('fixed chosen source capture contract required')
    files = {}
    for package, groups in value['packages'].items():
        if type(groups) is not dict or set(groups) != {'selected', 'metadata'}:
            raise ValueError('closed selected source/metadata groups required')
        for group, rows in groups.items():
            if type(rows) is not list or len(rows) > (20000 if group == 'selected' else 128):
                raise ValueError('bounded chosen source population required')
            for row in rows:
                if (type(row) is not dict or set(row) != {'path', 'relative', 'bytes', 'sha256'}
                        or type(row['relative']) is not str or not row['relative']
                        or type(row['bytes']) is not int or not 0 <= row['bytes'] <= 4 * 1024**2
                        or type(row['sha256']) is not str or len(row['sha256']) != 64
                        or any(character not in '0123456789abcdef' for character in row['sha256'])):
                    raise ValueError('closed chosen ordinary source pin required')
                relative = Path(row['relative'])
                if (relative.is_absolute() or relative.as_posix() != row['relative']
                        or any(part in {'', '.', '..'} for part in relative.parts)
                        or row['path'] != str(ORIGINAL_ROOT / 'external' / package / relative)):
                    raise ValueError('canonical captured provenance mapping required')
                path = SOURCE_ROOT / 'external' / package / relative
                if path in files:
                    raise ValueError('duplicate chosen source member')
                files[path] = row
    if not _same_pin(before):
        raise ValueError('chosen manifest changed during parsing')
    return value, files, before


def _selected_pin(path, files):
    path = Path(path).absolute()
    row = files.get(path)
    if row is None:
        raise ValueError('required source must belong to the exact chosen manifest')
    item = _pin(path)
    if any(item[key] != row[key] for key in ('bytes', 'sha256')):
        raise ValueError('chosen source differs from its captured member')
    return item


def _snapshot(package, destination, files, prefixes):
    destination.mkdir()
    rows, total = [], 0
    wanted = [path for path in files if path.is_relative_to(package)
        and any(path.relative_to(package).as_posix().startswith(prefix + '/') for prefix in prefixes)]
    for path in sorted(wanted):
        item = _selected_pin(path, files)
        raw = _read(path)
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('chosen source changed while staging its bytes')
        total += len(raw)
        if len(rows) >= 20000 or total > 2 * 1024**3:
            raise ValueError('selected package staging exceeds unchanged bounds')
        relative = path.relative_to(package)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        target.chmod(0o444)
        rows.append({**item, 'relative': relative.as_posix()})
    if not rows:
        raise ValueError('empty chosen source snapshot')
    return rows


def _verify_snapshot(selected, output):
    for name, rows in selected.items():
        root = output / name
        expected, actual, entries = {row['relative'] for row in rows}, set(), 0
        for directory, names, filenames in os.walk(root, followlinks=False):
            entries += 1 + len(names) + len(filenames)
            if entries > 100000 or any((Path(directory) / part).is_symlink() for part in names):
                raise ValueError('retained source traversal exceeds bounded ordinary membership')
            actual.update((Path(directory) / part).relative_to(root).as_posix() for part in filenames)
        if actual != expected:
            raise ValueError('retained source membership changed')
        for row in rows:
            path = root / row['relative']
            item = _pin(path)
            if path.lstat().st_mode & 0o222 or any(item[key] != row[key] for key in ('bytes', 'sha256')):
                raise ValueError('retained source bytes changed')


class _OwnedCancellation(BaseException):
    pass


@contextmanager
def _owned_signals(record):
    state, stopped = {'cleanup': False}, threading.Event()
    previous = {}
    def interrupt(number, frame):
        stopped.set()
        signals = record.setdefault('requested_shutdown_signals', [])
        if len(signals) < 32:
            signals.append(number)
        if not state['cleanup']:
            raise _OwnedCancellation('owned successor campaign shutdown requested')
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, interrupt)
    try:
        yield state, stopped
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _write(path, value):
    signed._write(path, value)


def _observe_resources(container):
    observation_program = (
        "import json,os,pathlib;root=pathlib.Path('/sys/fs/cgroup');"
        "names=('memory.current','memory.peak','memory.events','memory.swap.current','pids.current','pids.peak','cpu.stat');"
        "values={name:(root/name).read_text().strip() for name in names};"
        "print(json.dumps({'schema':'full300-paired-shared-successor-container-resource-observation@1',"
        "'uid':os.getuid(),'values':values,'namespaces':{name:os.readlink('/proc/self/ns/'+name)"
        " for name in ('pid','mnt','net')}},sort_keys=True))")
    observation = subprocess.run(['docker', 'exec', '--user', '1000:1000', container,
        deployment.WORKER_ROOT + '/venv/bin/python', '-I', '-B', '-c', observation_program],
        capture_output=True, text=True, check=True, timeout=15)
    if len(observation.stdout.encode()) > 16384:
        raise ValueError('bounded owned-container resource observation required')
    return json.loads(observation.stdout)


def _native_gates(evidence):
    modes = ('legacy_default', 'default', 'reference')
    required = ('full_campaign_qualified', 'final_resource_cleanup_verified',
        'full300_refusal_controls_qualified', 'complete_scan_qualified', 'large300_member_scan_qualified',
        'actual294_numerical_rows_per_mode_qualified', 'actual_sampled_rss_join_qualified',
        'shared_transport_streaming_qualified', 'retained_shared_worker_inputs',
        'warm_paired_receiver_qualified', 'cold_paired_receiver_qualified',
        'paired_shared_successor_receiving_qualified')
    forbidden = ('worker_launched', 'task_installation_qualified', 'worker_profile_integrated',
        'native_worker_qualified', 'publication_qualified', 'semantic_reproof_join_qualified',
        'cuda_qualified', '384d_qualified', '768d_qualified', '4096d_qualified',
        'proof_authority', 'signed_admission_qualified', 'throughput_qualified')
    if (type(evidence) is not dict or evidence.get('schema') != 'full300-paired-shared-successor-native-qualification@1'
            or evidence.get('qualified') is not True or any(evidence.get(key) is not True for key in required)
            or any(evidence.get(key) is not False for key in forbidden)
            or type(evidence.get('new_fitting_epochs')) is not int or evidence['new_fitting_epochs'] != 2
            or any(type(evidence.get(key)) is not int or evidence[key] != 0 for key in
                ('post_setup_fit_attempt_count', 'receive_forward_attempt_count',
                 'receiving_extra_numerical_attempt_count', 'receiving_mutation_attempt_count'))):
        return False
    requests = evidence.get('actual_numerical_requests_by_mode')
    if type(requests) is not dict or set(requests) != set(modes) or any(type(value) is not int or value <= 0 for value in requests.values()):
        return False
    dispositions = {'inferred':294, 'opaque':2, 'parse_failed':1, 'unindexed':1, 'unsupported_target':2}
    for mode in modes:
        coverage = evidence.get(mode + '_coverage', {})
        if (type(evidence.get('new_' + mode + '_scan_pages')) is not int or evidence['new_' + mode + '_scan_pages'] != 5
                or any(type(coverage.get(key)) is not int or coverage[key] != value for key,value in
                    (('inventory_entries',300), ('pages',5), ('inferred_rows',294), ('projection_records',588)))
                or json.dumps(coverage.get('dispositions'), sort_keys=True, separators=(',', ':'), allow_nan=False)
                    != json.dumps(dispositions, sort_keys=True, separators=(',', ':'), allow_nan=False)):
            return False
    resumes = evidence.get('fresh_resume_processes')
    cold = evidence.get('fresh_cold_receiver_processes')
    phases = evidence.get('paired_receiver_phase_records')
    if (type(resumes) is not list or len(resumes) != 3 or {row.get('mode') for row in resumes} != set(modes)
            or type(cold) is not list or len(cold) != 2 or {row.get('mode') for row in cold} != {'default','reference'}
            or type(phases) is not list or len(phases) != 4
            or {(row.get('mode'),row.get('stage')) for row in phases} !=
                {('default','warm'),('default','cold'),('reference','warm'),('reference','cold')}):
        return False
    births = []
    for row in resumes + cold:
        identity = row.get('process_identity', {})
        if type(row.get('pid')) is not int or row['pid'] <= 0 or identity.get('pid') != row['pid'] or not identity.get('birth'):
            return False
        births.append((row['pid'],identity['birth']))
    if len(set(births)) != 5:
        return False
    for row in phases:
        memory = row.get('rss_observation', {})
        observed, invoked = memory.get('identity', {}), row.get('invoking_process_identity', {})
        if (row.get('schema') != 'full300-paired-shared-successor-receive-phase@1'
                or row.get('profile') != 'paired-shared512-successor-native-receiving@1'
                or row.get('accepted') is not True or row.get('native_scope_exit_success') is not True
                or any(type(row.get(key)) is not int or row[key] != 1 for key in ('callback_invocations','successful_close_calls'))
                or memory.get('monitor_joined') is not True or memory.get('errors') != []
                or type(memory.get('sample_count')) is not int or memory['sample_count'] <= 0
                or type(memory.get('peak_rss_bytes')) is not int or not 0 < memory['peak_rss_bytes'] <= 8 * 1024**3
                or observed.get('pid') != invoked.get('pid') or observed.get('parent_pid') != invoked.get('parent_pid')
                or observed.get('process_group_id') != invoked.get('group_pid') or observed.get('session_id') != invoked.get('session_pid')
                or type(invoked.get('birth')) is not str or not invoked['birth'].isdecimal()
                or observed.get('birth_ticks') != int(invoked['birth'])):
            return False
    return (evidence.get('all_page_opt_out_equivalence', {}).get('per_source_numerical_values_exact') is True
        and evidence.get('legacy_default_equivalence', {}).get('per_source_numerical_values_within_tolerance') is True
        and evidence.get('parent_rss_observation', {}).get('monitor_joined') is True
        and type(evidence.get('parent_rss_observation', {}).get('sample_count')) is int
        and evidence['parent_rss_observation']['sample_count'] > 0)


def _run_selected(output, *, image, datasets, kit, lean, host_configuration=DEFAULT_HOST,
                  native_seconds=6000, docker_endpoint_guard):
    if type(native_seconds) is not int or native_seconds != 6000:
        raise ValueError('exact6000 native and5950 producer deadlines required')
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve(strict=True) != output.parent:
        raise ValueError('fresh canonical container output required')
    if SOURCE != SOURCE_ROOT / 'external/ipfs_accelerate':
        raise ValueError('new wrapper must execute from its exact frozen chosen source')
    datasets, kit, lean = [Path(value).resolve(strict=True) for value in (datasets, kit, lean)]
    if datasets != SOURCE_ROOT / 'external/ipfs_datasets' or kit != SOURCE_ROOT / 'external/ipfs_kit':
        raise ValueError('exact chosen datasets and kit roots required')
    host_configuration = Path(host_configuration).absolute()
    if host_configuration != DEFAULT_HOST:
        raise ValueError('exact existing MAIN configuration locator required')
    manifest, manifest_files, manifest_pin = _manifest()
    bootstrap = [_selected_pin(Path(module.__file__).resolve(), manifest_files)
        for module in (deployment, signed, shared)]
    bootstrap.append(_selected_pin(Path(__file__).resolve(), manifest_files))
    sys.path.insert(0, str(datasets))
    scheduler_pin = _selected_pin(datasets / 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py', manifest_files)
    process_pin = _selected_pin(datasets / 'ipfs_datasets_py/logic/backends/process.py', manifest_files)
    worker_deployment_pin = _selected_pin(SOURCE / 'benchmarks/agent_supervisor/container_coding/container_worker_deployment.py', manifest_files)
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as scheduling
    from ipfs_datasets_py.logic.backends import process as owned_process
    from . import container_worker_deployment as worker_deployment
    for module, expected_pin in ((scheduling, scheduler_pin), (owned_process, process_pin),
            (worker_deployment, worker_deployment_pin)):
        if Path(module.__file__).resolve() != Path(expected_pin['path']) or _pin(Path(expected_pin['path'])) != expected_pin:
            raise ValueError('exact chosen imported scheduler/process/bootstrap source required')
    bootstrap.extend((scheduler_pin, process_pin, worker_deployment_pin))
    host_raw = _read(host_configuration, 65536)
    if hashlib.sha256(host_raw).hexdigest() != CONFIG_SHA:
        raise ValueError('existing MAIN configuration bytes differ')
    host = _parse(host_raw)
    if host['auto_renew_leases'] is not True:
        raise ValueError('renewing existing MAIN leases required')
    configuration_pin = _pin(host_configuration)
    if configuration_pin['bytes'] != len(host_raw) or configuration_pin['sha256'] != hashlib.sha256(host_raw).hexdigest():
        raise ValueError('MAIN configuration changed between held reads')
    state_path, expected_configuration = Path(host['state_path']), host['persisted_config']
    for path in (state_path, state_path.with_name(state_path.name + '.lock')):
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink()
                or path.resolve(strict=True) != path or metadata.st_uid != os.getuid()):
            raise ValueError('existing canonical owner-controlled MAIN state and lock required')
    class ExactExistingScheduler(scheduling.GlobalResourceScheduler):
        def _new_state(self):
            raise scheduling.ResourceConfigurationError('MAIN state absent; pool creation refused')
        def _validate_state_configuration(self, state, *, allow_reconfigure=False):
            if (type(state) is not dict or state.get('schema_version') != scheduling.RESOURCE_SCHEDULER_SCHEMA_VERSION
                    or state.get('config') != expected_configuration):
                raise scheduling.ResourceConfigurationError('MAIN configuration differs; reconfiguration refused')
            return super()._validate_state_configuration(state, allow_reconfigure=False)
    scheduler = ExactExistingScheduler(scheduling.ResourceSchedulerConfig(**expected_configuration,
        state_path=state_path, lease_ttl_seconds=host['lease_ttl_seconds'], auto_renew_leases=True))
    output.mkdir()
    started = time.monotonic()
    name = 'ipfs-full300-paired-successor-' + str(time.time_ns())
    record = {'schema':'full300-paired-shared-successor-offline-container-execution@1', 'qualified':False,
        'container_name':name, 'network':'none', 'privileged':False, 'cpu_limit':12,
        'memory_limit_bytes':8 * 1024**3, 'pids_limit':512, 'host_child_process_slots':12,
        'native_total_timeout_seconds':6000, 'producer_overall_deadline_seconds':5950,
        'host_scheduler_state_path':str(scheduler.state_path), 'native_producer_module':MODULE,
        'host_configuration_pin':shared.pin(host_raw), 'configuration':configuration_pin,
        'source_root':str(SOURCE_ROOT), 'original_workspace_root':str(ORIGINAL_ROOT),
        'isolation_manifest':manifest_pin, 'bootstrap_source_pins':bootstrap,
        'original_shared_workspace_currentness_checked':False,
        'source_currentness_scope':'chosen source pins at start and source-close; sequential observations only',
        'inherited_scan_pages':0, 'inherited_setup_epochs':0, 'proof_authority':False,
        'execution_attestation':False, 'production_activated':False, 'large300_member_scan_qualified':False,
        'worker_dispatch_qualified':False, 'task_installation_qualified':False, 'worker_launched':False,
        'worker_profile_integrated':False, 'paired_shared_successor_receiving_qualified':False,
        'scope':'fresh CPU structural8 selected-successor scans, two warm and two fresh cold paired receives; no signed worker or proof authority'}
    record['docker_client_endpoint_guard'] = docker_endpoint_guard
    record['docker_client_endpoint_unchanged'] = False
    lease = container = held_cleanup = None
    selected = {}
    signals = _owned_signals(record)
    signal_state, shutdown = signals.__enter__()
    try:
        record['host_resources_before_reservation'] = scheduler.snapshot()
        lease = scheduler.acquire(lane=scheduling.ResourceLane.ORCHESTRATION, cpu_slots=12,
            memory_mb=8192, child_process_slots=12, timeout=90, request_id='fresh-paired-shared-successor-container')
        record['host_reservation'] = lease.to_dict()
        if not lease.renew() or not _same_pin(manifest_pin) or not all(_same_pin(item) for item in bootstrap):
            raise ValueError('live MAIN reservation and unchanged chosen bootstrap required')
        selected = {
            'source':_snapshot(SOURCE, output / 'source', manifest_files, ('ipfs_accelerate_py','benchmarks','test')),
            'datasets':_snapshot(datasets, output / 'datasets', manifest_files, ('ipfs_datasets_py',)),
            'kit':_snapshot(kit, output / 'kit', manifest_files, ('ipfs_kit_py',)),
        }
        producer_path = SOURCE / 'benchmarks/agent_supervisor/container_coding/qualify_codebase_full300_shared_successor_receiving.py'
        producer_pin = _selected_pin(producer_path, manifest_files)
        producer_copy = output / 'source/benchmarks/agent_supervisor/container_coding/qualify_codebase_full300_shared_successor_receiving.py'
        copied_pin = _pin(producer_copy)
        if any(producer_pin[key] != copied_pin[key] for key in ('bytes','sha256')):
            raise ValueError('new selected producer copy differs')
        record['native_producer_pin'] = {'path':str(producer_path),
            'copy':producer_copy.relative_to(output).as_posix(), 'bytes':producer_pin['bytes'], 'sha256':producer_pin['sha256']}
        _write(output / 'selected-source-snapshot.json', selected)
        _verify_snapshot(selected, output)
        _write(output / 'shared-host-configuration.json', host)
        image_id = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image], text=True, timeout=15).strip()
        if image_id != EXPECTED_IMAGE:
            raise ValueError('exact approved offline image SHA required')
        record['image_id'] = image_id
        scripts = output / 'deployment'
        scripts.mkdir()
        for filename, content in (('owner-worker',worker_deployment.OWNER_ENTRY),
                ('validation-worker',worker_deployment.VALIDATION_ENTRY), ('worker-entry',worker_deployment.WORKER_ENTRY)):
            with (scripts / filename).open('x') as stream:
                stream.write(content)
        with (scripts / 'sudoers').open('x') as stream:
            stream.write('supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *\n')
        if not lease.renew() or lease.cancellation_signal.is_set() or shutdown.is_set():
            raise ValueError('live MAIN reservation required before owned container creation')
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
        native_command = ['docker', '--host', docker_endpoint_guard['endpoint'], 'exec', '--user', '1000:1000',
                '-e', 'DOCTOR_COMPOSITION_LEAN=/toolchains/lean/bin/lean', '-e', 'IPFS_ACCELERATE_RUN_LIVE_QUACK=1',
                '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
                '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'NUMEXPR_NUM_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                container, deployment.WORKER_ROOT + '/venv/bin/python', '-B', '-m', MODULE,
                '--output', '/results/native', '--overall-seconds', str(native_seconds - 50)]
        class Cancellation:
            def is_set(self):
                return shutdown.is_set() or lease.cancellation_signal.is_set()
        result = owned_process.run_bounded_stdin_tool(native_command, b'',
            runner=owned_process.BoundedToolRunner(base_environment=dict(os.environ)),
            limits=owned_process.ToolRunLimits(timeout_seconds=native_seconds,
                resident_memory_bytes=8 * 1024**3, max_output_bytes=16 * 1024**2,
                max_input_bytes=1, max_workspace_bytes=128 * 1024**2,
                max_file_bytes=64 * 1024**2, max_workspace_entries=65536),
            cancellation=Cancellation(), owned_process_observation=True)
        with (output / 'run.stdout').open('xb') as stdout:
            stdout.write(result.stdout.encode('utf-8'))
        with (output / 'run.stderr').open('xb') as stderr:
            stderr.write(result.stderr.encode('utf-8'))
        record['native_command'] = native_command
        record['host_docker_client_process_observation'] = result.to_dict()
        record['host_process_observation_scope'] = 'owned docker exec client and its observed host descendants; container cgroup is the native containment boundary'
        if (result.native_lifecycle is None or not result.native_lifecycle.cleanup_confirmed
                or any((result.timed_out, result.cancelled, result.unavailable,
                    result.output_truncated, result.resource_exhausted))):
            raise ValueError('owned native Docker client failed its bounds or drain')
        record['returncode'] = result.returncode
        resources = _observe_resources(container)
        record['actual_container_resources_after_native'] = resources
        if (resources['uid'] != 1000 or resources['namespaces'] != record['actual_container_limits']['cgroup']['namespaces']):
            raise ValueError('owned-container resource observation namespace differs')
        events = dict((name, int(value)) for name, value in
            (line.split() for line in resources['values']['memory.events'].splitlines()))
        peak = int(resources['values']['memory.peak'])
        pids_peak = int(resources['values']['pids.peak'])
        if (peak <= 0 or peak > 8 * 1024**3 or not 0 < pids_peak <= 512
                or events.get('oom') != 0 or events.get('oom_kill') != 0):
            raise ValueError('owned-container peak memory or OOM evidence exceeds envelope')
        record['actual_container_peak_memory_bytes'] = peak
        record['actual_container_peak_pids'] = pids_peak
        record['actual_container_oom_events'] = events
    except BaseException as error:
        record.update(qualified=False, error_type=type(error).__name__, error=str(error))
        if isinstance(error, owned_process.OwnedToolCleanupError):
            held_cleanup = error.cleanup_handle
    finally:
        signal_state['cleanup'] = True
        def cleanup_error(stage, error):
            record['cleanup_error_count'] = record.get('cleanup_error_count', 0) + 1
            errors = record.setdefault('cleanup_errors', [])
            if len(errors) < 32:
                errors.append({'stage':stage, 'error_type':type(error).__name__, 'error':str(error)[:2048]})
        try:
            if container is not None:
                try:
                    copied = subprocess.run(['docker','cp',container + ':/results/.',str(output)],
                        capture_output=True,text=True,timeout=60)
                    record['container_results_copied'] = copied.returncode == 0
                except BaseException as error:
                    cleanup_error('copy_results',error)
            if record.get('container_creation_attempted'):
                _RETAINED_HOST_LEASES[name] = lease
                while True:
                    try:
                        if lease is not None:
                            lease.renew()
                        removed = deployment._remove_container(name)
                        absent = subprocess.run(['docker','ps','-a','-q','--no-trunc','--filter','name=^/' + name + '$'],
                            capture_output=True,text=True,timeout=15)
                        record['container_removed'] = removed and absent.returncode == 0 and not absent.stdout.strip()
                        record['container_absence_observation'] = {'returncode':absent.returncode,
                            'stdout':absent.stdout[:16384],'stderr':absent.stderr[:16384]}
                    except BaseException as error:
                        record['container_removed'] = False
                        cleanup_error('remove_owned_container',error)
                    if record.get('container_removed') is True:
                        break
                    record['host_reservation_retained_for_live_container'] = True
                    try:
                        _write(output / 'cleanup-progress.json',record)
                    except BaseException as error:
                        cleanup_error('persist_cleanup_progress',error)
                    time.sleep(.25)
            if held_cleanup is not None:
                while True:
                    try:
                        if lease is not None:
                            lease.renew()
                        observed = held_cleanup.retry_cleanup()
                        if observed.cleanup_confirmed:
                            record['host_docker_client_cleanup_retry'] = observed.to_dict()
                            break
                    except BaseException as error:
                        cleanup_error('drain_owned_docker_client',error)
                    time.sleep(.25)
            try:
                _verify_snapshot(selected,output)
                record['retained_source_verified_after_execution'] = bool(selected)
                changes = [row['path'] for rows in selected.values() for row in rows
                    if not _same_pin({key:row[key] for key in ('path','bytes','sha256')})]
                record['chosen_selected_source_changes_after_execution'] = changes
                record['isolation_manifest_unchanged'] = _same_pin(manifest_pin)
                record['bootstrap_source_pins_unchanged'] = all(_same_pin(item) for item in bootstrap)
                record['configuration_unchanged'] = _same_pin(configuration_pin)
                record['docker_client_endpoint_unchanged'] = (
                    _docker_socket_guard(docker_endpoint_guard['context'],docker_endpoint_guard['endpoint']) == docker_endpoint_guard)
                if not record['docker_client_endpoint_unchanged']:
                    raise ValueError('selected Docker endpoint changed during owned execution')
                record['source_close_completed_under_reservation'] = lease is not None and not lease.released
                native_path = output / 'native/result.json'
                if native_path.exists():
                    evidence, native_pin = _pinned_json(native_path)
                    record['native_result_pin'] = native_pin
                    record['native_campaign_gates_passed'] = _native_gates(evidence)
                else:
                    record['native_campaign_gates_passed'] = False
            except BaseException as error:
                record['retained_source_verified_after_execution'] = False
                record['native_campaign_gates_passed'] = False
                cleanup_error('close_selected_source_and_native_result',error)
            if lease is not None:
                while not lease.released:
                    try:
                        lease.release()
                    except BaseException as error:
                        cleanup_error('release_MAIN',error)
                        try:
                            lease.renew()
                        except BaseException as renewal_error:
                            cleanup_error('renew_pending_MAIN_release',renewal_error)
                        time.sleep(.25)
            record['host_reservation_released'] = lease is not None and lease.released
            if record['host_reservation_released']:
                _RETAINED_HOST_LEASES.pop(name,None)
            record['host_resources_after_cleanup'] = scheduler.snapshot()
            record['elapsed_seconds'] = time.monotonic() - started
            record['qualified'] = (record.get('returncode') == 0 and record.get('container_removed') is True
                and record.get('container_results_copied') is True and record['host_reservation_released']
                and record.get('retained_source_verified_after_execution') is True
                and record.get('chosen_selected_source_changes_after_execution') == []
                and record.get('isolation_manifest_unchanged') is True and record.get('bootstrap_source_pins_unchanged') is True
                and record.get('configuration_unchanged') is True and record.get('source_close_completed_under_reservation') is True
                and record.get('native_campaign_gates_passed') is True
                and record['docker_client_endpoint_unchanged']
                and _same_pin(record.get('native_result_pin', {})) and not record.get('cleanup_errors')
                and 'error_type' not in record and not shutdown.is_set()
                and type(record.get('actual_container_peak_memory_bytes')) is int)
            record['large300_member_scan_qualified'] = record['qualified']
            record['paired_shared_successor_receiving_qualified'] = record['qualified']
            copied_root = output / 'native'
            copied_files = copied_bytes = copied_symlinks = 0
            if copied_root.is_dir():
                for directory,names,filenames in os.walk(copied_root,followlinks=False):
                    for filename in filenames:
                        metadata = (Path(directory) / filename).lstat()
                        if stat.S_ISREG(metadata.st_mode):
                            copied_files += 1
                            copied_bytes += metadata.st_size
                        elif stat.S_ISLNK(metadata.st_mode):
                            copied_symlinks += 1
            record['copied_native_artifact_accounting'] = {'regular_files':copied_files,
                'regular_file_bytes':copied_bytes,'symlink_files':copied_symlinks,
                'scope':'measured copied native tree; no enforced disk byte cap claimed'}
            _write(output / 'container-execution-final.json',record)
        finally:
            signals.__exit__(None,None,None)
    if not record['qualified']:
        raise RuntimeError('fresh paired successor campaign failed; retained evidence identifies the stage')
    return record



def _docker_socket_guard(context, endpoint):
    if not endpoint.startswith('unix:///') or len(endpoint) > 4096:
        raise ValueError('selected Docker endpoint must be a local Unix socket')
    path=Path(endpoint[len('unix://'):])
    metadata=path.lstat()
    if (path.resolve(strict=True)!=path or not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid!=os.getuid() or metadata.st_nlink!=1):
        raise ValueError('canonical owner-controlled Docker Unix socket required')
    return {'schema':'full300-paired-shared-successor-docker-endpoint@1',
        'context':context,'endpoint':endpoint,'socket_path':str(path),'device':metadata.st_dev,
        'inode':metadata.st_ino,'uid':metadata.st_uid,'mode':metadata.st_mode,
        'metadata_ctime_ns':metadata.st_ctime_ns}


@contextmanager
def _selected_docker_endpoint(guard):
    """Bind wrapper/helper clients without altering production native Popen."""
    real_subprocess=subprocess
    class SelectedClient:
        def __getattr__(self,name):
            return getattr(real_subprocess,name)
        def command(self,value):
            if isinstance(value,(list,tuple)) and value and Path(str(value[0])).name=='docker':
                value=list(value)
                if '--host' in value:
                    offset=value.index('--host')
                    if offset+1>=len(value) or value[offset+1]!=guard['endpoint']:
                        raise ValueError('Docker client attempted another daemon endpoint')
                else:
                    value[1:1]=['--host',guard['endpoint']]
            return value
        def run(self,command,*args,**kwargs):
            return real_subprocess.run(self.command(command),*args,**kwargs)
        def check_output(self,command,*args,**kwargs):
            return real_subprocess.check_output(self.command(command),*args,**kwargs)
        def Popen(self,command,*args,**kwargs):
            return real_subprocess.Popen(self.command(command),*args,**kwargs)
    selected=SelectedClient()
    import importlib
    owner=importlib.import_module(__name__)
    with patch.object(owner,'subprocess',selected), patch.object(deployment,'subprocess',selected), \
            patch.object(signed,'subprocess',selected):
        yield


def run(output, *, image, datasets, kit, lean, host_configuration=DEFAULT_HOST, native_seconds=6000):
    context=subprocess.check_output(['docker','context','show'],text=True,timeout=15).strip()
    if not context or len(context)>128 or any(not(c.isalnum() or c in '._-') for c in context):
        raise ValueError('bounded selected Docker context required')
    endpoint=subprocess.check_output(['docker','context','inspect',context,
        '--format','{{.Endpoints.docker.Host}}'],text=True,timeout=15).strip()
    guard=_docker_socket_guard(context,endpoint)
    with _selected_docker_endpoint(guard):
        return _run_selected(output,image=image,datasets=datasets,kit=kit,lean=lean,
            host_configuration=host_configuration,native_seconds=native_seconds,docker_endpoint_guard=guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--image', default=EXPECTED_IMAGE)
    parser.add_argument('--datasets', type=Path, default=SOURCE.parent / 'ipfs_datasets')
    parser.add_argument('--kit', type=Path, default=SOURCE.parent / 'ipfs_kit')
    parser.add_argument('--lean', type=Path, default=Path('/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1'))
    parser.add_argument('--host-configuration', type=Path, default=DEFAULT_HOST)
    parser.add_argument('--native-seconds', type=int, choices=(6000,), default=6000)
    args = parser.parse_args()
    report = run(args.output, image=args.image, datasets=args.datasets, kit=args.kit, lean=args.lean,
        host_configuration=args.host_configuration, native_seconds=args.native_seconds)
    print(json.dumps({key: report.get(key) for key in ('qualified', 'returncode', 'container_removed', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
