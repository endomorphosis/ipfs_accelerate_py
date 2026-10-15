"""Fresh complete 300-source CPU8D scans and signed preview qualification.

Only the two explicit setup epochs fit a model. Both five-page scans include
three pages resumed in fresh processes. Native @6 signing, the genuine preview
and companion, and @7 receiving are qualified without opening an Intent owner,
installing tasks, reserving a worker or invoking START. Larger heads, CUDA,
semantic proof and full worker execution remain separate gates.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from unittest.mock import patch

from . import qualify_codebase_inventory_resume as native
from . import qualify_codebase_inventory_scan as base
from . import qualify_codebase_signed_successor as worker
from . import source_successor_full_scan_fixture as shared

MODULE = 'benchmarks.agent_supervisor.container_coding.qualify_codebase_full300_preview'
SCHEMA = 'codebase-full300-preview-fresh-native-qualification@1'
RESUME_REQUEST = 'full300-preview-native-resume-request@1'
RESUME_RESULT = 'full300-preview-native-resume-result@1'
REPOSITORY_ID = 'qualification:source-successor'
ENTRIES, PAGE_ENTRIES, PAGES, PARENT_PAGES, RESUME_PAGES = 300, 64, 5, 2, 3


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')


def _persist(output, name, record):
    base._write(output / name, {'artifact_cid': record.artifact_cid, 'value': record.to_dict()})


def _states(registry, versions):
    return {name: native._state(registry, version, include_identity=True)
        for name, version in versions.items()}


def _process(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    native.require(len(fields) >= 22, 'complete process identity required')
    return {'pid': pid, 'birth': fields[19], 'parent_pid': int(fields[1]),
        'group_pid': int(fields[2]), 'session_pid': int(fields[3]), 'state': fields[0]}


def _group_members(identity):
    """Observe only the captured isolated session inside the finite container."""
    pids = [int(path.name) for path in Path('/proc').iterdir() if path.name.isdecimal()]
    native.require(len(pids) <= 1024, 'bounded fresh container process population required')
    members = []
    for pid in pids:
        observed = _process(pid)
        if observed is not None and observed['group_pid'] == identity['group_pid']:
            native.require(observed['session_pid'] == identity['session_pid'],
                'fresh process group now belongs to another session')
            if observed['state'] not in {'Z', 'X', 'x'}:
                members.append(observed)
    return sorted(members, key=lambda item: item['pid'])


def _signal_member(identity, selected_signal):
    native.require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'),
        'Linux PID-bound fresh process cleanup required')
    try:
        descriptor = os.pidfd_open(identity['pid'], 0)
    except ProcessLookupError:
        return
    try:
        current = _process(identity['pid'])
        if current is None or current['birth'] != identity['birth'] or current['state'] in {'Z', 'X', 'x'}:
            return
        native.require(current['group_pid'] == identity['group_pid']
            and current['session_pid'] == identity['session_pid'], 'owned resume member changed session before signal')
        try:
            signal.pidfd_send_signal(descriptor, selected_signal)
        except ProcessLookupError:
            pass
    finally:
        os.close(descriptor)


def _drain_resume_group(child, identity):
    """Keep this live launch-lease owner until its exact session has drained.

    The outer owned container deadline handles an unresolvable custody failure.
    Returning early would incorrectly release the local launch reservation.
    """
    observations, failures = [], []
    started = time.monotonic()
    while True:
        try:
            native.require(identity is not None and identity['group_pid'] == identity['session_pid'] == child.pid,
                'captured isolated resume session required for shutdown')
            members = _group_members(identity)
            if len(observations) < 32:
                observations.append({'members': members, 'elapsed_seconds': time.monotonic() - started})
            if not members:
                child.wait(timeout=5)
                return {'launch_leader_reaped': True, 'live_group_members': 0,
                    'group_pid': identity['group_pid'], 'session_pid': identity['session_pid'],
                    'observations': observations, 'cleanup_failures': failures,
                    'scope': 'captured fresh resume session only; launch lease held until group drained'}
            selected_signal = signal.SIGINT if time.monotonic() - started < 10 else signal.SIGKILL
            for member in members:
                _signal_member(member, selected_signal)
            child.poll()
            time.sleep(.1)
        except BaseException as error:
            if len(failures) < 32:
                failures.append({'error_type': type(error).__name__, 'error': str(error)[:256]})
            try:
                print(json.dumps({'status': 'fresh_resume_guardian_retained_until_group_drains',
                    'group_pid': child.pid, 'error_type': type(error).__name__}, sort_keys=True), flush=True)
            except BaseException:
                pass
            try:
                time.sleep(5)
            except BaseException:
                pass


def _launch_resume(command, log, timeout):
    """Reap the exact captured fresh process before its launch lease closes."""
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = _process(child.pid)
    try:
        native.require(identity is not None and identity['parent_pid'] == os.getpid()
            and identity['group_pid'] == identity['session_pid'] == child.pid, 'captured isolated fresh resume required')
        returncode = child.wait(timeout=timeout)
    finally:
        cleanup = _drain_resume_group(child, identity)
        log.flush()
        os.fsync(log.fileno())
    return returncode, identity, cleanup


def _resume(request_path, expected_sha256):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
    request_path = Path(request_path)
    raw = shared.read_absolute(request_path, 1_048_576)
    native.require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'resume request bytes differ')
    request = json.loads(raw)
    native.require(set(request) == {'schema', 'output', 'mode', 'root_cid', 'cursor', 'versions',
        'checkpoint_states', 'expected_head', 'authority', 'authority_sha256', 'orchestration',
        'parent_identity', 'result', 'max_pages', 'timeout_seconds'}, 'closed full300 resume request required')
    native.require(request['schema'] == RESUME_REQUEST and request['mode'] in {'default', 'reference'}
        and request['max_pages'] == RESUME_PAGES and set(request['versions']) == {'root', 'child'}
        and type(request['timeout_seconds']) in {int, float} and math.isfinite(request['timeout_seconds'])
        and 0 < request['timeout_seconds'] <= 1200, 'bounded exact full300 continuation required')
    output = Path(request['output'])
    native.require(output.is_absolute() and output.resolve(strict=True) == output
        and request_path == output / (request['mode'] + '-resume-request.json')
        and Path(request['result']) == output / (request['mode'] + '-resume-result.json'),
        'exact fresh continuation output paths required')
    authority_raw = shared.read_absolute(Path(request['authority']), 1_048_576)
    native.require(hashlib.sha256(authority_raw).hexdigest() == request['authority_sha256'],
        'container resource authority changed')
    authority = json.loads(authority_raw)
    native.require(authority['namespace'] == {name: os.readlink('/proc/self/ns/' + name)
        for name in ('pid', 'mnt', 'net')}, 'resume authority belongs to another namespace')
    scheduler = shared.shared_scheduler(authority['state_path'], authority['persisted_config'])
    parent = _process(os.getppid())
    native.require(parent is not None and all(parent[key] == request['parent_identity'][key]
        for key in ('pid', 'birth', 'parent_pid', 'group_pid', 'session_pid'))
        and parent['state'] not in {'Z', 'X', 'x'}, 'fresh resume parent identity changed')
    with scheduler._locked_state(persist=False) as state:
        held = state['leases'].get(request['orchestration']['lease_id'])
        native.require(held is not None and held['owner_pid'] == os.getppid()
            and held['owner_birth_marker'] == parent['birth'] and held['cpu_slots'] == 1
            and held['memory_mb'] == 512 and held['child_process_slots'] == 1
            and not held.get('cancelled') and not held.get('release_requested')
            and held['expires_at'] > time.time(), 'fresh process has no live parent launch reservation')
    selected = json.loads(shared.read_absolute(output / 'generation-inputs.json', 1_048_576))
    native._require_pins(output, selected)
    deadline = time.monotonic() + request['timeout_seconds']
    index = registry = connection = None
    report = {'schema': RESUME_RESULT, 'qualified': False, 'pid': os.getpid(),
        'process_identity': _process(os.getpid()), 'mode': request['mode'],
        'request_sha256': expected_sha256, 'pages': [], 'post_setup_fit_attempt_count': 0,
        'new_fitting_epochs': 0, 'proof_authority': False}
    try:
        index, registry, connection = native._open(output)
        native.require(index.current(REPOSITORY_ID).to_dict() == request['expected_head'], 'resume head changed')
        native.require(_states(registry, request['versions']) == request['checkpoint_states'],
            'resume weights or Adam state changed')
        root = scan.load_codebase_scan_resume_root(index.artifacts, request['root_cid'])
        native.require(len(root.to_dict()['members']) == ENTRIES
            and root.to_dict()['limits']['page_entries'] == PAGE_ENTRIES
            and root.to_dict()['model']['version_id'] == request['versions']['child'],
            'fresh full300 root population or model differs')
        cursor = scan.CodebaseScanResumeCursor.from_dict(request['cursor'])
        with shared.no_fit(report):
            for number in range(RESUME_PAGES):
                remaining = deadline - time.monotonic()
                native.require(remaining > 0, 'full300 fresh continuation deadline exceeded')
                page = scan.scan_current_codebase_page(index, output / 'repository', root=root,
                    registry=registry, cursor=cursor, scheduler=scheduler, memory_mb=1024,
                    timeout_seconds=min(300, remaining), admission_timeout_seconds=min(30, remaining))
                report['pages'].append({'artifact_cid': page.artifact_cid, 'value': page.to_dict()})
                cursor = page.next_cursor
                if cursor is None:
                    break
        native.require(cursor is None and len(report['pages']) == RESUME_PAGES,
            'full300 continuation must finish exactly three pages')
        report['checkpoint_states_after'] = _states(registry, request['versions'])
        native.require(report['checkpoint_states_after'] == request['checkpoint_states'],
            'fresh continuation changed numerical state')
        native._require_pins(output, selected)
        report['qualified'] = True
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        primary_error_active = sys.exc_info()[0] is not None
        cleanup_error = None
        for name, action in (('close_native_owners', lambda: native._close(registry, connection)),
                ('observe_owned_resources', lambda: shared.assert_clean(scheduler))):
            try:
                value = action()
                if name == 'observe_owned_resources':
                    report['resources_after'] = value
            except BaseException as error:
                report['qualified'] = False
                report.setdefault('cleanup_errors', []).append({'stage': name,
                    'error_type': type(error).__name__, 'error': str(error)})
                cleanup_error = cleanup_error or error
        base._write(Path(request['result']), report)
        if cleanup_error is not None and not primary_error_active:
            raise cleanup_error
    return report


def _no_install_guards(report):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        codebase_inventory_execution as execution, codebase_successor_dispatch_admission as boundary,
        codebase_preview_successor_worker_admission as preview, local_planning_admission as local,
    )
    counters = {'intent_open_attempt_count': 0, 'task_installation_attempt_count': 0,
        'worker_launch_attempt_count': 0}
    report['no_install_guards'] = {'counters': counters, 'guarded_apis': []}
    stack = ExitStack()
    targets = [(IntentRepository, '__init__', 'intent_open_attempt_count')]
    targets.extend((IntentRepository, name, 'task_installation_attempt_count')
        for name in ('upsert_objective', 'upsert_goal', 'upsert_plan', 'upsert_task'))
    targets.extend((owner, name, 'task_installation_attempt_count') for owner, name in (
        (local, 'materialize_local_benchmark_plan'), (local, '_materialize_local_benchmark_plan'),
        (boundary, 'materialize_current_successor_plan'),
        (preview, 'materialize_current_preview_successor_worker_plan')))
    targets.extend((owner, name, 'worker_launch_attempt_count') for owner, name in (
        (boundary, 'reserve_current_successor_execution'), (preview, 'reserve_current_preview_successor_execution'),
        (execution, 'reserve_inventory_execution'), (AdmittedBenchmarkRuntime, 'create'),
        (AdmittedBenchmarkRuntime, 'start')))
    for owner, name, counter in targets:
        qualified_name = getattr(owner, '__name__', str(owner)) + '.' + name
        report['no_install_guards']['guarded_apis'].append(qualified_name)

        def refuse(*args, _counter=counter, _name=qualified_name, **kwargs):
            counters[_counter] += 1
            raise AssertionError('preview-only qualification forbids ' + _name)

        stack.enter_context(patch.object(owner, name, refuse))
    return stack


def _replace_owned_bytes(path, raw):
    """Mutate only this newly authored fixture, restoring its original mode."""
    before = path.lstat()
    native.require(stat.S_ISREG(before.st_mode) and not path.is_symlink()
        and before.st_uid == os.getuid() and len(raw) <= 1_048_576, 'bounded owned drift fixture required')
    mode = stat.S_IMODE(before.st_mode)
    path.chmod(mode | stat.S_IWUSR)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'wb') as stream:
            opened = os.fstat(stream.fileno())
            native.require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                'owned drift fixture inode changed')
            os.ftruncate(stream.fileno(), 0)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        path.chmod(mode)


def _qualify_preview_only(output, *, current, scheduler, phase, report, options):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        codebase_successor_dispatch_admission as boundary,
        codebase_preview_successor_worker_admission as preview_boundary,
        local_planning_admission as local,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import MAX_BODY_BYTES
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    from .preview_successor_native_profile import prepare_native_preview_successor

    repository, private = current['repository'], output / 'private'
    profile, lifecycle = private / 'profile', private / 'lifecycle'
    with _no_install_guards(report):
        bootstrap = phase('initialize_private_signed_preview_owner', lambda: Supervisor.init_local(
            repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle))
        base._write(output / 'signed-owner-bootstrap.json', bootstrap)
        graph, specs, roots = worker._native_graph(current['index'], current['index'].current(REPOSITORY_ID),
            current['completion'])
        base._write(output / 'graph.json', graph.to_dict())
        base._write(output / 'authored-task-population.json', {'tasks': specs,
            'task_cids': sorted(task.task_cid for task in graph.tasks), 'planning_roots': roots,
            'checker_scripts': [{'path': name, **shared.pin(shared.read_absolute(repository / name))}
                for name in ('check_type.py', 'check_offset.py')], 'checker_executed_here': False})
        manifest = phase('sign_full300_parent_manifest', lambda: boundary.author_current_successor_manifest(
            current['selection'], current['root'], current['completion'], current['index'], repository,
            registry=current['registry'], profile_dir=profile, lifecycle_dir=lifecycle,
            task_specs=specs, planning_roots=roots, **options(300)))
        base._write(output / 'parent-signed-manifest.json', manifest)
        parent = phase('admit_full300_parent_graph', lambda: boundary.admit_current_successor_plan(
            **current, manifest=manifest, graph=graph, **options(300)))
        base._write(output / 'parent-admission.json', parent)
        parent_wire = _wire(local._plain(parent))
        received_parent = phase('receive_full300_parent_admission', lambda: boundary.verify_current_successor_admission(
            **current, admission=parent, **options(300)))
        base._write(output / 'parent-current-admission-verification.json', received_parent)
        verified_parent = local.verify_local_benchmark_admission(parent)
        native.require(len(verified_parent['manifest']['sources']) == ENTRIES
            and verified_parent['graph'].to_dict() == graph.to_dict()
            and not received_parent['current_facts'] and not received_parent['removed_task_cids'],
            'full300 parent discarded sources or independent tasks')
        prepared = phase('prepare_genuine_full300_preview_and_companion', lambda: prepare_native_preview_successor(
            **current, parent_admission=parent, graph=graph, owner_state_dir=private / 'preview-owner-state',
            **options(600, memory_mb=2048)))
        base._write(output / 'signed-manifest.json', prepared['manifest'])
        base._write(output / 'admission.json', prepared['admission'])
        base._write(output / 'typed-request.json', prepared['request'].to_dict())
        base._write(output / 'material-snapshot.json', prepared['bridge']['payload']['input_snapshot'])
        base._write(output / 'preview.json', prepared['preview'].to_dict())
        base._write(output / 'signed-bridge.json', prepared['bridge'])
        base._write(output / 'preview-native-profile.json', prepared['records'])
        native.require(prepared['records']['preview_memory_mb'] == 512
            and prepared['records']['preview_owner_memory_mb'] == 512
            and prepared['records']['preview_lease']['memory_mb'] == 512
            and prepared['records']['signed_memory_mb'] == 2048
            and prepared['records']['capture_limits'] == {'max_entries': 512, 'max_file_bytes': 65536},
            'full300 native preview reservation or capture limits differ')
        captured_inputs = preview_boundary.capture_preview_successor_bridge_inputs(prepared['bridge_inputs'])
        metadata_sizes = {'worker_admission_bytes': len(preview_boundary._wire(prepared['admission'])),
            'parent_admission_bytes': len(preview_boundary._wire(local._plain(parent))),
            'signed_bridge_bytes': len(preview_boundary._wire(prepared['bridge'])),
            'private_input_wire_bytes': [len(raw) for raw in captured_inputs[1]]}
        metadata_bytes = sum(metadata_sizes[key] for key in
            ('worker_admission_bytes', 'parent_admission_bytes', 'signed_bridge_bytes'))
        metadata_bytes += sum(metadata_sizes['private_input_wire_bytes'])
        report['signed_metadata_reservation'] = {**metadata_sizes, 'total_metadata_bytes': metadata_bytes,
            'required_bytes': 248 * 1024 * 1024 + 4 * metadata_bytes,
            'permitted_bytes': 2048 * 1024 * 1024 // 4, 'caller_memory_mb': 2048,
            'guard_rewritten_here': False}
        native.require(report['signed_metadata_reservation']['required_bytes']
            <= report['signed_metadata_reservation']['permitted_bytes'], 'full300 signed metadata exceeds authorized envelope')

        def receive():
            return preview_boundary.verify_current_preview_successor_worker_admission(**current,
                worker_admission=prepared['admission'], bridge_inputs=prepared['bridge_inputs'],
                **options(600, memory_mb=2048))

        try:
            received = phase('fresh_receive_full300_preview_child', receive)
            base._write(output / 'current-admission-verification.json', received)
            verified = local.verify_local_benchmark_admission(prepared['admission'])
            context = verified['manifest']['codebase_preview_successor_context']
            bindings = {key: received[key] for key in ('parent_admission_cid', 'signed_bridge_cid',
                'bridge_cid', 'codebase_preview_successor_context_cid')}
            native.require(verified['manifest']['schema'] == local.PREVIEW_SUCCESSOR_MANIFEST_SCHEMA
                and len(verified['manifest']['sources']) == ENTRIES
                and verified['graph'].to_dict() == graph.to_dict() and received['worker_profile_integrated'] is True
                and not received['current_facts'] and not received['removed_task_cids']
                and bindings == {'parent_admission_cid': content_identity(parent),
                    'signed_bridge_cid': content_identity(prepared['bridge']),
                    'bridge_cid': prepared['bridge']['payload']['bridge_cid'],
                    'codebase_preview_successor_context_cid': content_identity(context)}
                and all(value is False for value in received['authority'].values()),
                'full300 native preview identity, complete graph or authority differs')
            report['preview_successor_binding'] = bindings
            report['signed_metadata_memory_mb'] = 2048
            report['preview_memory_mb'] = 512
            report['native_preview_source_members'] = ENTRIES
            report['native_preview_signed_manifest_bytes'] = len(_wire(prepared['manifest']))
            report['native_preview_admission_bytes'] = len(_wire(prepared['admission']))
            report['task_body_incompatibility'] = {'embedded_parent_bytes': len(parent_wire),
                'body_limit_bytes': MAX_BODY_BYTES, 'embedded_parent_alone_exceeds_body_limit': len(parent_wire) > MAX_BODY_BYTES,
                'prospective_pending_body_constructed_here': False, 'task_population_installed_here': False}
            native.require(len(parent_wire) > MAX_BODY_BYTES, 'retained full300 inline parent unexpectedly fits task body')
            cancelled = threading.Event()
            cancelled.set()
            before = shared.assert_clean(scheduler)
            try:
                preview_boundary.verify_current_preview_successor_worker_admission(**current,
                    worker_admission=prepared['admission'], bridge_inputs=prepared['bridge_inputs'],
                    cancel_event=cancelled, **options(600, memory_mb=2048))
            except LeaseCancelledError as error:
                report['controls'].append({'name': 'precancelled_full300_preview_receive', 'refused': True,
                    'error_type': type(error).__name__, 'resources_before': before,
                    'resources_after': shared.assert_clean(scheduler)})
            else:
                raise AssertionError('pre-cancelled full300 receiving succeeded')
            policy_path = private / 'preview-owner-state' / 'checker-policy.json'
            policy_raw = shared.read_absolute(policy_path, 1_048_576)
            policy = json.loads(policy_raw)
            policy['outcomes_recorded'] = True
            for name, path, changed in (
                    ('current_source_bytes_drift', repository / 'calc.py',
                     b'def increment(n: int) -> int:\n    return n + 999\n'),
                    ('persisted_checker_policy_drift', policy_path, _wire(policy))):
                original = shared.read_absolute(path, 1_048_576)
                before = shared.assert_clean(scheduler)
                try:
                    _replace_owned_bytes(path, changed)
                    def refuse_drift():
                        try:
                            receive()
                        except ValueError as error:
                            return error
                        raise AssertionError('drifted full300 preview receiving succeeded')

                    error = phase('refuse_full300_' + name, refuse_drift)
                    if isinstance(error, ValueError):
                        report['controls'].append({'name': name, 'refused': True,
                            'error_type': type(error).__name__, 'error': str(error),
                            'original_sha256': hashlib.sha256(original).hexdigest(),
                            'changed_sha256': hashlib.sha256(changed).hexdigest(),
                            'resources_before': before, 'resources_after': shared.assert_clean(scheduler),
                            'scope': 'genuine current fixture owner drift; no install or worker'})
                finally:
                    _replace_owned_bytes(path, original)
                    native.require(shared.read_absolute(path, 1_048_576) == original, 'drift fixture restore differs')
            restored = phase('fresh_receive_full300_preview_after_restore', receive)
            base._write(output / 'current-admission-verification-after-restore.json', restored)
            native.require(_wire(restored) == _wire(received) and _wire(local._plain(parent)) == parent_wire,
                'restored full300 receiving or immutable parent differs')
        finally:
            base._write(output / 'preview-native-profile-after-verification.json', prepared['records'])
        absent = {name: not path.exists() for name, path in (
            ('intent_database_absent', private / 'intent.duckdb'), ('worker_launch_directory_absent', private / 'launch'),
            ('worker_worktrees_absent', repository / '.git' / 'worktrees'))}
        native.require(all(absent.values()) and all(value == 0 for value in report['no_install_guards']['counters'].values()),
            'preview-only qualification crossed an installation or worker boundary')
        report['no_install_guards'].update(absent, preview_only_guarded=True)
        base._write(output / 'no-install-guards.json', report['no_install_guards'])
        report['signed_preview_companion_qualified'] = True


def run(output, *, overall_seconds=3550, host_configuration='/results/container-resource-authority.json'):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes

    native.require(type(overall_seconds) is int and 900 <= overall_seconds <= 3550,
        'bounded full300 preview deadline required')
    output = Path(output).absolute()
    native.require(not output.exists() and output.parent.resolve(strict=True) == output.parent,
        'fresh exact full300 output required')
    output.mkdir()
    scheduler, authority, authority_raw = worker._container_scheduler(Path(host_configuration))
    started = time.monotonic()
    deadline = started + overall_seconds
    report = {'schema': SCHEMA, 'qualified': False, 'pid': os.getpid(), 'process_identity': _process(os.getpid()),
        'phases': [], 'controls': [], 'scope': 'fresh300_all_pages_cpu8d_and_signed_preview_only',
        'inventory_entries': ENTRIES, 'page_entries': PAGE_ENTRIES, 'new_fitting_epochs': 0,
        'new_default_scan_pages': 0, 'new_reference_scan_pages': 0, 'fresh_resume_processes': [],
        'post_setup_fit_attempt_count': 0, 'inference_attempt_count': 0, 'provider_calls': 0,
        'worker_launched': False, 'task_installation_qualified': False, 'native_worker_qualified': False,
        'full_administrator_population_completed': False, 'publication_qualified': False,
        'signed_preview_companion_qualified': False, 'large300_member_scan_qualified': False,
        'proof_authority': False, 'source_execution_attested': False, 'scan_execution_attested': False,
        'cuda_qualified': False, '384d_qualified': False, '768d_qualified': False, '4096d_qualified': False,
        'production_default_activated': False, 'semantic_reproof_join_qualified': False,
        'new_source_generation_after_worker_qualified': False, 'overall_deadline_seconds': overall_seconds,
        'local_pool_bounded_by_host_envelope': True, 'shares_host_pid_state': False,
        'scheduler_configuration': scheduler.config.persisted_dict(), 'container_resource_authority_pin': shared.pin(authority_raw)}
    index = registry = connection = None
    base._write(output / 'container-resource-authority.json', authority)

    def remaining(cap=300):
        seconds = deadline - time.monotonic()
        native.require(seconds > 0, 'full300 preview qualification deadline exceeded')
        return min(cap, seconds)

    def options(cap=300, *, memory_mb=1024):
        return {'scheduler': scheduler, 'timeout_seconds': remaining(cap), 'memory_mb': memory_mb,
            'admission_timeout_seconds': min(30, remaining(cap))}

    def phase(name, action):
        row = {'name': name, 'status': 'running'}
        report['phases'].append(row)
        base._progress(output / 'progress.json', report)
        begin = time.monotonic()
        try:
            value = action()
            row['status'] = 'completed'
            return value
        except BaseException as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic() - begin
            base._progress(output / 'progress.json', report)

    def forbidden_forward(*args, **kwargs):
        report['inference_attempt_count'] += 1
        raise AssertionError('full300 signed receiving attempted forward inference')

    try:
        selected = native._pins(output, sorted(set(successor._implementation()['files']) | {
            MODULE, worker.__name__, shared.__name__, base.__name__,
            'benchmarks.agent_supervisor.container_coding.preview_successor_native_profile',
            'ipfs_accelerate_py.agent_supervisor.runtime.codebase_preview_successor_worker_admission',
            'ipfs_accelerate_py.agent_supervisor.runtime.codebase_preview_successor_worker_context',
            'ipfs_accelerate_py.agent_supervisor.runtime.codebase_preview_successor_admission',
            'ipfs_accelerate_py.agent_supervisor.runtime.codebase_preview_successor_contract',
            'ipfs_accelerate_py.agent_supervisor.runtime.codebase_successor_dispatch_admission',
            'ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission',
            'ipfs_accelerate_py.agent_supervisor.runtime.codebase_inventory_execution',
            'ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository',
            'ipfs_accelerate_py.agent_supervisor.entrypoints.facade',
            'ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile',
            'ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime',
            'ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview',
            'ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context',
            'ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler',
            'ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts',
            'ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service'}))
        repository = output / 'repository'
        native._sources(repository)
        native.require(len([name for name in native._git(repository, 'ls-files', '-z').split('\0') if name]) == ENTRIES,
            'original native300 fixture population differs')
        index, registry, connection = phase('open_fresh_native_source_model_owners', lambda: native._open(output))
        preparation = {'repository_id': REPOSITORY_ID, 'limits': CodebaseScanLimits(512, 65536), 'exclusions': ('.runtime',)}
        previous = phase('publish_native_previous_source', lambda: index.prepare_current(repository,
            operation_id='initial', expected_head=None, **preparation, **options()))
        base._write(output / 'previous-publication-receipt.json', previous.to_dict())
        base._write(output / 'previous-source-manifest.json', index.load(previous.head.manifest_cid).to_dict())
        selections = [training.CodebaseTrainingSelection(path, role) for path, role in
            (('calc.py', 'train'), ('tune.py', 'tune'), ('canary.py', 'canary'))]

        def fit(head, name, parent=None):
            before = 0 if parent is None else native._state(registry, parent)['completed_epochs']
            result = training.train_current_codebase_features(index, repository, expected_head=head,
                registry=registry, selections=selections, operation_id=name, parent_version_id=parent,
                epochs=1, learning_rate=.002, seed=1729, **options(300))
            version = result.to_dict()['version_id']
            native.require(native._state(registry, version)['completed_epochs'] - before == 1,
                'actual explicit full300 setup epoch differs')
            report['new_fitting_epochs'] += 1
            _persist(output, name + '-training-record.json', result)
            return version

        parent = phase('explicit_native_root_training_one_epoch', lambda: fit(previous.head, 'root'))
        (repository / 'calc.py').write_bytes(b'def increment(n: int) -> int:\n    return n + 2\n')
        (repository / 'bulk000.py').unlink()
        (repository / 'bulk001.py').rename(repository / 'renamed.py')
        (repository / 'added.py').write_bytes(b'def added(n: int) -> int:\n    return n + 700\n')
        native._git(repository, 'add', '-A')
        native._git(repository, 'commit', '--quiet', '--no-verify', '-m', 'Explicit full300 native source successor')
        native.require(len([name for name in native._git(repository, 'ls-files', '-z').split('\0') if name]) == ENTRIES,
            'full300 successor lost original population')
        current = phase('publish_native_current_source', lambda: index.prepare_current(repository,
            operation_id='successor', expected_head=previous.head, **preparation, **options()))
        base._write(output / 'current-publication-receipt.json', current.to_dict())
        manifest = index.load(current.head.manifest_cid)
        native.require(len(manifest.snapshot.entries) == ENTRIES and manifest.snapshot.max_entries == 512
            and manifest.snapshot.max_file_bytes == 65536, 'full300 capture population or original limits differ')
        base._write(output / 'current-source-manifest.json', manifest.to_dict())
        transition = phase('build_native_source_delta', lambda: delta.build_current_codebase_source_delta(
            index, repository, previous_head=previous.head, expected_head=current.head, **options()))
        _persist(output, 'source-delta.json', transition)
        child = phase('explicit_native_child_training_one_epoch', lambda: fit(current.head, 'child', parent))
        versions = {'root': parent, 'child': child}
        frozen = _states(registry, versions)
        base._write(output / 'checkpoint-states-before.json', frozen)
        completed = []
        with shared.no_fit(report):
            for optimized in (True, False):
                mode = 'default' if optimized else 'reference'
                selection = phase('select_native_' + mode + '_successor', lambda: successor.start_current_codebase_successor_scan(
                    index, repository, source_delta=transition, registry=registry, previous_version_id=parent, version_id=child,
                    limits=scan.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=PAGE_ENTRIES),
                    optimized=optimized, **options(300)))
                root = scan.load_codebase_scan_resume_root(index.artifacts, selection.to_dict()['root_cid'])
                _persist(output, mode + '-selection.json', selection)
                _persist(output, mode + '-root.json', root)
                cursor, pages = None, []
                for number in range(1, PARENT_PAGES + 1):
                    page = phase('infer_' + mode + '_page_' + str(number), lambda: scan.scan_current_codebase_page(
                        index, repository, root=root, registry=registry, cursor=cursor, **options(300)))
                    pages.append(page)
                    cursor = page.next_cursor
                    _persist(output, mode + '-page-' + str(number) + '.json', page)
                native.require(cursor is not None, 'full300 fresh-process cursor missing')
                native._close(registry, connection)
                index = registry = connection = None
                result_path = output / (mode + '-resume-result.json')
                request_path = output / (mode + '-resume-request.json')
                from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
                with scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=512,
                        child_process_slots=1, timeout=min(30, remaining()),
                        request_id='full300-preview-' + mode + '-fresh-process') as orchestration:
                    request = {'schema': RESUME_REQUEST, 'output': str(output), 'mode': mode, 'root_cid': root.artifact_cid,
                        'cursor': cursor.to_dict(), 'versions': versions, 'checkpoint_states': frozen,
                        'expected_head': current.head.to_dict(), 'authority': str(host_configuration),
                        'authority_sha256': hashlib.sha256(authority_raw).hexdigest(), 'orchestration': orchestration.to_dict(),
                        'parent_identity': _process(os.getpid()), 'result': str(result_path),
                        'max_pages': RESUME_PAGES, 'timeout_seconds': remaining(1200)}
                    base._write(request_path, request)
                    digest = hashlib.sha256(shared.read_absolute(request_path, 1_048_576)).hexdigest()
                    command = [sys.executable, '-B', '-m', MODULE, '--resume-request', str(request_path),
                        '--resume-request-sha256', digest]
                    with (output / (mode + '-resume.log')).open('xb') as log:
                        returncode, identity, process_cleanup = phase('fresh_process_' + mode + '_resume', lambda: _launch_resume(
                            command, log, min(remaining(1220), request['timeout_seconds'] + 20)))
                native.require(returncode == 0, 'fresh full300 continuation failed')
                resumed = json.loads(shared.read_absolute(result_path, 4 * 1024 * 1024))
                native.require(resumed['qualified'] and resumed['request_sha256'] == digest
                    and resumed['post_setup_fit_attempt_count'] == resumed['new_fitting_epochs'] == 0
                    and resumed['checkpoint_states_after'] == frozen and resumed['pid'] == identity['pid']
                    and resumed['process_identity']['birth'] == identity['birth'], 'fresh full300 continuation evidence differs')
                report['fresh_resume_processes'].append({'mode': mode, 'pid': resumed['pid'],
                    'process_identity': identity, 'request_sha256': digest, 'result': result_path.name,
                    'launch_group_cleanup': process_cleanup})
                shared.assert_clean(scheduler, owner_pids=[os.getpid(), resumed['pid']])
                index, registry, connection = phase('cold_reopen_after_' + mode, lambda: native._open(output))
                for number, row in enumerate(resumed['pages'], PARENT_PAGES + 1):
                    page = scan.load_codebase_scan_resume_page(index.artifacts, row['artifact_cid'])
                    native.require(page.to_dict() == row['value'], 'fresh full300 page bytes differ')
                    pages.append(page)
                    _persist(output, mode + '-page-' + str(number) + '.json', page)
                completion = phase('complete_all_' + mode + '_pages', lambda: scan.complete_current_codebase_scan(
                    index, repository, root=root, registry=registry, tail_page_cid=pages[-1].artifact_cid, **options(300)))
                _persist(output, mode + '-completion.json', completion)
                native.require(completion.to_dict()['coverage']['inventory_entries'] == ENTRIES
                    and completion.to_dict()['coverage']['pages'] == len(pages) == PAGES,
                    'complete native full300 coverage differs')
                report['new_' + ('default' if optimized else 'reference') + '_scan_pages'] = len(pages)
                completed.append((selection, root, completion, pages))
            for first, reference in zip(completed[0][3], completed[1][3]):
                native.require(all(shared.inert.observation_same(first.to_dict()[key], reference.to_dict()[key])
                    for key in ('entries', 'coverage', 'inference')), 'full300 optimized/opt-out outcomes differ')
            report['all_page_opt_out_equivalence'] = {'pages': PAGES, 'inventory_entries': ENTRIES,
                'entries_coverage_and_inference_exact': True, 'throughput_qualified': False}
            selection, root, completion, _ = completed[0]
            _persist(output, 'successor-selection.json', selection)
            _persist(output, 'scan-root.json', root)
            _persist(output, 'scan-completion.json', completion)
            with ExitStack() as guards:
                for owner, name in ((scan, '_worker'), (features, 'infer_projection_features'),
                        (runtimes.SourceBoundCodebaseFeatureRuntime, 'infer')):
                    guards.enter_context(patch.object(owner, name, forbidden_forward))
                _qualify_preview_only(output, current={'selection': selection, 'root': root, 'completion': completion,
                    'index': index, 'registry': registry, 'repository': repository},
                    scheduler=scheduler, phase=phase, report=report, options=options)
            after = _states(registry, versions)
            native.require(after == frozen and index.current(REPOSITORY_ID) == current.head,
                'full300 scans or signed preview changed model/Adam or source head')
            base._write(output / 'checkpoint-states-after.json', after)
            native._require_pins(output, selected)
            native.require(report['new_fitting_epochs'] == 2 and report['post_setup_fit_attempt_count'] == 0
                and report['inference_attempt_count'] == 0 and report['signed_preview_companion_qualified']
                and len({row['pid'] for row in report['fresh_resume_processes']}) == 2,
                'full300 explicit setup, fresh receiving or process accounting differs')
            report.update(qualified=True, complete_scan_qualified=True, large300_member_scan_qualified=True,
                selected_producers_unchanged=True, native_source_generation_advanced_by_worker=False)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        worker._finalize_qualification_report(output, report, registry=registry, connection=connection,
            scheduler=scheduler, started=started, primary_error_active=sys.exc_info()[0] is not None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--overall-seconds', type=int, default=3550)
    parser.add_argument('--host-configuration', default='/results/container-resource-authority.json')
    parser.add_argument('--resume-request')
    parser.add_argument('--resume-request-sha256')
    args = parser.parse_args()
    if args.resume_request:
        native.require(args.resume_request_sha256 is not None and args.output is None,
            'exact standalone full300 resume request required')
        _resume(args.resume_request, args.resume_request_sha256)
    else:
        native.require(args.output is not None and args.resume_request_sha256 is None,
            'fresh full300 preview output required')
        run(args.output, overall_seconds=args.overall_seconds, host_configuration=args.host_configuration)


if __name__ == '__main__':
    main()
