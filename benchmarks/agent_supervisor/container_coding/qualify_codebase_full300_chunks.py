"""Fresh full300 sequential CPU structural 8D chunks, without signed dispatch.

This additive producer creates its own sources, native owners and two explicit
setup epochs. Five pages in each mode include three pages in a genuinely fresh
process. All 294 eligible members must produce actual numerical rows; the six
other dispositions remain explicit. Chunk bytes and process observations stay
advisory. Signed admission, task installation, larger heads, CUDA and proof are
separate unqualified gates.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from . import qualify_codebase_full300_companion as custody
from . import qualify_codebase_inventory_resume as native
from . import qualify_codebase_inventory_scan as base
from . import qualify_codebase_signed_successor as deployment
from . import source_successor_full_scan_fixture as shared

MODULE = 'benchmarks.agent_supervisor.container_coding.qualify_codebase_full300_chunks'
SCHEMA = 'codebase-full300-sequential-chunks-fresh-native-qualification@1'
RESUME_REQUEST = 'full300-sequential-chunks-native-resume-request@1'
RESUME_RESULT = 'full300-sequential-chunks-native-resume-result@1'
REPOSITORY_ID = 'qualification:source-successor'
ENTRIES, PAGE_ENTRIES, PAGES, PARENT_PAGES, RESUME_PAGES = 300, 64, 5, 2, 3
EXPECTED_DISPOSITIONS = {'inferred': 294, 'opaque': 2, 'parse_failed': 1,
                         'unindexed': 1, 'unsupported_target': 2}
EXPECTED_PAGE_ROWS = [63, 64, 64, 64, 39]


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _persist(output, name, record):
    custody._persist(output, name, record)


def _states(registry, versions):
    return custody._states(registry, versions)


def _launch_resume(command, log, timeout, orchestration):
    """Hold the launch lease until the sampled leader and descendants drain.

    Numerical grandchildren start separate sessions. The new owned native
    runner tracks their birth identities through ancestry rather than relying
    on the resume leader's process group. The enclosing dedicated container
    remains the kernel containment boundary at its independent outer deadline.
    """
    from ipfs_datasets_py.logic.backends.process import (
        BoundedToolRunner, OwnedToolCleanupError, ToolRunLimits, run_bounded_stdin_tool,
    )
    runner = BoundedToolRunner(base_environment=dict(os.environ))
    bounds = ToolRunLimits(timeout_seconds=timeout, termination_grace_seconds=1,
        max_input_bytes=1, max_output_bytes=16 * 1024 * 1024,
        max_workspace_bytes=128 * 1024 * 1024, max_file_bytes=64 * 1024 * 1024,
        resident_memory_bytes=8 * 1024 * 1024 * 1024)
    try:
        process = run_bounded_stdin_tool(command, b'', runner=runner, limits=bounds,
            cancellation=orchestration.combined_cancellation_signal(None), owned_process_observation=True)
    except OwnedToolCleanupError as error:
        # No caller context exit while cleanup ownership is unresolved. Each
        # attempt remains bounded; the parent process and host container lease
        # stay alive until actual cleanup or the outer container guardian.
        attempts, begin, last_notice = 0, time.monotonic(), 0.0
        while True:
            attempts += 1
            cleanup = None
            try:
                orchestration.renew()
            except BaseException:
                pass
            try:
                cleanup = error.cleanup_handle.retry_cleanup()
            except BaseException:
                pass
            if cleanup is not None and cleanup.cleanup_confirmed:
                log.write(_wire({'status': 'fresh_resume_cleanup_confirmed_after_failed_run',
                    'attempts': attempts, 'elapsed_seconds': time.monotonic() - begin,
                    'native_lifecycle': cleanup.to_dict()}) + b'\n')
                log.flush()
                os.fsync(log.fileno())
                raise
            now = time.monotonic()
            if now - last_notice >= 5:
                last_notice = now
                try:
                    notice = {'status': 'fresh_resume_launch_lease_retained_until_actual_cleanup',
                        'attempts': attempts, 'elapsed_seconds': now - begin,
                        'native_lifecycle': None if cleanup is None else cleanup.to_dict()}
                    log.write(_wire(notice) + b'\n')
                    log.flush()
                    print(json.dumps({'status': notice['status'], 'attempts': attempts}, sort_keys=True), flush=True)
                except BaseException:
                    pass
            try:
                time.sleep(.25)
            except BaseException:
                pass
    lifecycle = process.native_lifecycle
    native.require(lifecycle is not None and lifecycle.cleanup_confirmed
        and lifecycle.parent_pid == os.getpid() and lifecycle.leader.parent_pid == os.getpid(),
        'fresh resume has no actual parent-owned drained lifecycle')
    leader = lifecycle.leader
    identity = {'pid': leader.pid, 'birth': str(leader.birth_ticks), 'parent_pid': leader.parent_pid,
        'group_pid': leader.process_group_id, 'session_pid': leader.session_id, 'state': 'reaped'}
    native.require(identity['group_pid'] == identity['session_pid'] == identity['pid'],
                   'fresh resume leader did not own its isolated native session')
    cleanup = {'schema': 'full300-chunks-resume-owned-lifecycle@1',
        'launch_leader_reaped': lifecycle.leader_reaped,
        'live_group_members': len(lifecycle.remaining_group_members),
        'live_tracked_descendants': len(lifecycle.remaining_tracked_descendants),
        'group_pid': leader.process_group_id, 'session_pid': leader.session_id,
        'native_lifecycle': lifecycle.to_dict(), 'sampled_ownership_only': True,
        'kernel_containment_attested': False,
        'scope': 'native sampled ancestry and birth-bound descendants; launch lease held until actual cleanup',
        'runner': {'returncode': process.returncode, 'timed_out': process.timed_out,
            'cancelled': process.cancelled, 'unavailable': process.unavailable,
            'resource_exhausted': process.resource_exhausted, 'output_truncated': process.output_truncated,
            'workspace_cleaned': process.workspace_cleaned, 'elapsed_ms': process.elapsed_ms,
            'max_output_bytes': bounds.max_output_bytes, 'max_workspace_bytes': bounds.max_workspace_bytes,
            'max_file_bytes': bounds.max_file_bytes, 'resident_memory_bytes': bounds.resident_memory_bytes}}
    log.write(process.stdout.encode('utf-8'))
    log.write(process.stderr.encode('utf-8'))
    log.write(b'\n' + _wire({'parent_owned_launch_lifecycle': cleanup}) + b'\n')
    log.flush()
    os.fsync(log.fileno())
    native.require(not any((process.timed_out, process.cancelled, process.unavailable,
        process.resource_exhausted, process.output_truncated)), 'fresh resume native runner exceeded its bounds')
    return process.returncode, identity, cleanup


def _export_page(output, mode, number, page, index):
    """Persist every actual chunk separately; retain only bounded descriptors."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunks as records
    _persist(output, mode + '-page-' + str(number) + '.json', page)
    p, previous, rows, projections = page.to_dict(), None, 0, 0
    for ordinal, descriptor in enumerate(p['chunks']):
        chunk = scan.load_codebase_scan_numerical_chunk(index.artifacts, descriptor['chunk_cid'])
        c = chunk.to_dict()
        native.require(records.chunk_descriptor(chunk) == descriptor and c['chunk_ordinal'] == ordinal
            and c['previous_chunk_cid'] == previous and c['worker_counters']['training_calls'] == 0
            and c['worker_counters']['rows'] == len(c['inference']['rows']) == descriptor['rows'],
            'actual full300 chunk mapping, prefix or worker counters differ')
        native.require(c['worker_receipt']['input_bytes'] <= 32 * 1024 * 1024
            and c['worker_receipt']['output_bytes'] <= 16 * 1024 * 1024
            and len(chunk._payload) <= 16 * 1024 * 1024,
            'full300 actual request, response or wrapper exceeds unchanged cap')
        for row_index, member_index in enumerate(c['member_indices']):
            entry = p['entries'][member_index - p['start']]
            native.require(entry['chunk_ordinal'] == ordinal and entry['row_index'] == row_index
                and entry['target_sha256'] == c['target_sha256s'][row_index]
                and entry['source_digest'] == c['source_digests'][row_index]
                == c['inference']['rows'][row_index]['source_digest'], 'actual full300 row mapping differs')
        name = mode + '-page-' + str(number) + '-chunk-' + str(ordinal + 1) + '.json'
        _persist(output, name, chunk)
        previous = chunk.artifact_cid
        rows += descriptor['rows']
        projections += descriptor['projection_records']
        del chunk, c
    native.require(rows == p['coverage']['inferred_rows'] == EXPECTED_PAGE_ROWS[number - 1]
        and projections == p['coverage']['projection_records'] == 2 * rows
        and p['coverage']['chunks'] == len(p['chunks']), 'actual full300 page coverage differs')
    return records.page_descriptor(page)


def _verify_page_exports(output, mode, number, page, index):
    """Join fresh-child ordinary exports back to the independently read CAS."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    name = mode + '-page-' + str(number) + '.json'
    saved = json.loads(shared.read_absolute(output / name, 32 * 1024 * 1024))
    native.require(shared.inert.observation_same(saved,
        {'artifact_cid': page.artifact_cid, 'value': page.to_dict()}), 'fresh child page export bytes differ')
    for ordinal, descriptor in enumerate(page.to_dict()['chunks']):
        chunk = scan.load_codebase_scan_numerical_chunk(index.artifacts, descriptor['chunk_cid'])
        name = mode + '-page-' + str(number) + '-chunk-' + str(ordinal + 1) + '.json'
        saved = json.loads(shared.read_absolute(output / name, 32 * 1024 * 1024))
        native.require(shared.inert.observation_same(saved,
            {'artifact_cid': chunk.artifact_cid, 'value': chunk.to_dict()}), 'fresh child chunk export bytes differ')
        del chunk, saved


@contextmanager
def _no_forward(report):
    """Receiving replays native targets; no numerical child, forward or fit."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    real_popen = subprocess.Popen

    def refuse(*args, **kwargs):
        report['receive_forward_attempt_count'] += 1
        raise AssertionError('chunk receiving attempted numerical inference')

    def popen(command, *args, **kwargs):
        if isinstance(command, (tuple, list)) and any(
                'codebase_inventory_resume_worker.py' in str(value) for value in command):
            return refuse(command, *args, **kwargs)
        return real_popen(command, *args, **kwargs)

    with ExitStack() as stack:
        for owner, name in ((scan, '_worker'), (resume, '_worker'),
                (features, 'infer_projection_features'), (runtimes.SourceBoundCodebaseFeatureRuntime, 'infer')):
            stack.enter_context(patch.object(owner, name, refuse))
        stack.enter_context(patch.object(subprocess, 'Popen', popen))
        yield


def _resume(request_path, expected_sha256):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunks as records
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    request_path = Path(request_path)
    raw = shared.read_absolute(request_path, 1_048_576)
    native.require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'chunk resume request bytes differ')
    request = json.loads(raw)
    native.require(set(request) == {'schema', 'output', 'mode', 'root_cid', 'cursor', 'versions',
        'checkpoint_states', 'expected_head', 'authority', 'authority_sha256', 'orchestration',
        'parent_identity', 'result', 'max_pages', 'timeout_seconds'}, 'closed chunk continuation request required')
    native.require(request['schema'] == RESUME_REQUEST and request['mode'] in {'default', 'reference'}
        and type(request['max_pages']) is int and request['max_pages'] == RESUME_PAGES
        and set(request['versions']) == {'root', 'child'}
        and type(request['timeout_seconds']) in {int, float} and math.isfinite(request['timeout_seconds'])
        and 0 < request['timeout_seconds'] <= 1800, 'bounded exact chunk continuation required')
    output, mode = Path(request['output']), request['mode']
    native.require(output.is_absolute() and output.resolve(strict=True) == output
        and request_path == output / (mode + '-resume-request.json')
        and Path(request['result']) == output / (mode + '-resume-result.json'), 'exact chunk continuation paths required')
    authority_raw = shared.read_absolute(Path(request['authority']), 1_048_576)
    native.require(hashlib.sha256(authority_raw).hexdigest() == request['authority_sha256'], 'chunk resource authority changed')
    authority = json.loads(authority_raw)
    native.require(authority['namespace'] == {name: os.readlink('/proc/self/ns/' + name)
        for name in ('pid', 'mnt', 'net')}, 'chunk continuation namespace differs')
    scheduler = shared.shared_scheduler(authority['state_path'], authority['persisted_config'])
    parent = custody._process(os.getppid())
    native.require(parent is not None and all(parent[key] == request['parent_identity'][key]
        for key in ('pid', 'birth', 'parent_pid', 'group_pid', 'session_pid'))
        and parent['state'] not in {'Z', 'X', 'x'}, 'chunk continuation parent changed')
    with scheduler._locked_state(persist=False) as state:
        held = state['leases'].get(request['orchestration']['lease_id'])
        native.require(held is not None and held['owner_pid'] == os.getppid()
            and held['owner_birth_marker'] == parent['birth'] and held['cpu_slots'] == 1
            and held['memory_mb'] == 512 and held['child_process_slots'] == 1
            and not held.get('cancelled') and not held.get('release_requested')
            and held['expires_at'] > time.time(), 'chunk continuation launch reservation missing')
    selected = json.loads(shared.read_absolute(output / 'generation-inputs.json', 1_048_576))
    native._require_pins(output, selected)
    deadline = time.monotonic() + request['timeout_seconds']
    index = registry = connection = None
    report = {'schema': RESUME_RESULT, 'qualified': False, 'pid': os.getpid(),
        'process_identity': custody._process(os.getpid()), 'mode': mode,
        'request_sha256': expected_sha256, 'pages': [], 'post_setup_fit_attempt_count': 0,
        'new_fitting_epochs': 0, 'receive_forward_attempt_count': 0, 'proof_authority': False}

    def options():
        remaining = deadline - time.monotonic()
        native.require(remaining > 0, 'chunk continuation deadline exceeded')
        return {'scheduler': scheduler, 'memory_mb': 1024, 'timeout_seconds': min(400, remaining),
                'admission_timeout_seconds': min(30, remaining)}

    try:
        index, registry, connection = native._open(output)
        native.require(index.current(REPOSITORY_ID).to_dict() == request['expected_head']
            and _states(registry, request['versions']) == request['checkpoint_states'], 'chunk resume source or state changed')
        root = resume.load_codebase_scan_resume_root(index.artifacts, request['root_cid'])
        native.require(len(root.to_dict()['members']) == ENTRIES
            and root.to_dict()['limits']['page_entries'] == PAGE_ENTRIES
            and root.to_dict()['model']['version_id'] == request['versions']['child'], 'chunk resume root differs')
        cursor = records.CodebaseScanChunkedCursor.from_dict(request['cursor'])
        native.require(cursor.next_offset == PARENT_PAGES * PAGE_ENTRIES, 'fresh chunk cursor is not page boundary128')
        with shared.no_fit(report):
            prefix = scan.load_codebase_scan_chunked_page(index.artifacts, cursor.previous_page_cid)
            with _no_forward(report):
                native.require(scan.validate_current_codebase_chunked_scan_page(prefix, index, output / 'repository',
                    root=root, registry=registry, **options()) is prefix, 'cold prefix receiving changed selection')
            for number in range(PARENT_PAGES + 1, PAGES + 1):
                page = scan.scan_current_codebase_page_chunked(index, output / 'repository', root=root,
                    registry=registry, cursor=cursor, **options())
                report['pages'].append(_export_page(output, mode, number, page, index))
                cursor = records.CodebaseScanChunkedCursor.from_page(root, page)
            native.require(cursor is None and len(report['pages']) == RESUME_PAGES, 'fresh chunk continuation incomplete')
        report['checkpoint_states_after'] = _states(registry, request['versions'])
        native.require(report['checkpoint_states_after'] == request['checkpoint_states'], 'fresh chunks altered weights or Adam')
        native._require_pins(output, selected)
        report['qualified'] = True
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        primary = sys.exc_info()[0] is not None
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
        if cleanup_error is not None and not primary:
            raise cleanup_error
    return report


def _compare_pages(index, first_pages, reference_pages):
    """Stream paired saved chunk rows; never reconstruct a giant page tensor."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    totals = {'rows': 0, 'projection_records': 0, 'pages': 0, 'chunks_per_mode': 0}
    for first, reference in zip(first_pages, reference_pages):
        f, r = first.to_dict(), reference.to_dict()
        native.require(all(shared.inert.observation_same(f[key], r[key])
            for key in ('entries', 'coverage')), 'default/reference chunk entries or coverage differ')
        native.require(len(f['chunks']) == len(r['chunks']), 'default/reference numerical partitions differ')
        for fd, rd in zip(f['chunks'], r['chunks']):
            fc = scan.load_codebase_scan_numerical_chunk(index.artifacts, fd['chunk_cid']).to_dict()
            rc = scan.load_codebase_scan_numerical_chunk(index.artifacts, rd['chunk_cid']).to_dict()
            native.require(fc['member_indices'] == rc['member_indices']
                and fc['target_sha256s'] == rc['target_sha256s'] and fc['source_digests'] == rc['source_digests']
                and shared.inert.observation_same(fc['inference'], rc['inference']),
                'default/reference full300 actual per-source numerical outcomes differ')
            totals['rows'] += len(fc['inference']['rows'])
            totals['projection_records'] += len(fc['inference']['coverage'])
            totals['chunks_per_mode'] += 1
            del fc, rc
        totals['pages'] += 1
    native.require(totals['rows'] == 294 and totals['projection_records'] == 588 and totals['pages'] == PAGES,
        'equal-work full300 chunk row totals differ')
    return {**totals, 'inventory_entries': ENTRIES, 'per_source_numerical_values_exact': True,
            'entries_and_coverage_exact': True, 'throughput_qualified': False}


def run(output, *, overall_seconds=5950, host_configuration='/results/container-resource-authority.json'):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunk_scan as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_chunks as records
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    native.require(type(overall_seconds) is int and 1200 <= overall_seconds <= 6000,
                   'bounded full300 chunk qualification deadline required')
    output = Path(output).absolute()
    native.require(not output.exists() and output.parent.resolve(strict=True) == output.parent, 'fresh exact chunk output required')
    output.mkdir()
    scheduler, authority, authority_raw = deployment._container_scheduler(Path(host_configuration))
    started = time.monotonic()
    deadline = started + overall_seconds
    report = {'schema': SCHEMA, 'qualified': False, 'pid': os.getpid(),
        'process_identity': custody._process(os.getpid()), 'phases': [], 'controls': [],
        'scope': 'fresh300_cpu_structural8d_294_actual_rows_sequential_chunks_without_signed_dispatch',
        'inventory_entries': ENTRIES, 'page_entries': PAGE_ENTRIES, 'new_fitting_epochs': 0,
        'new_default_scan_pages': 0, 'new_reference_scan_pages': 0, 'fresh_resume_processes': [],
        'post_setup_fit_attempt_count': 0, 'receive_forward_attempt_count': 0, 'provider_calls': 0,
        'worker_launched': False, 'task_installation_qualified': False, 'native_worker_qualified': False,
        'worker_launch_flag_scope': 'supervisor task worker; numerical child launches are recorded separately',
        'actual_numerical_requests_by_mode': {},
        'worker_profile_integrated': False, 'signed_admission_qualified': False,
        'signed_chunked_context_integration_qualified': False, 'successor_dispatch_qualified': False,
        'signed_preview_companion_qualified': False, 'publication_qualified': False,
        'proof_authority': False, 'source_execution_attested': False, 'scan_execution_attested': False,
        'cuda_qualified': False, '384d_qualified': False, '768d_qualified': False, '4096d_qualified': False,
        'semantic_reproof_join_qualified': False, 'new_source_generation_after_worker_qualified': False,
        'production_default_activated': False, 'overall_deadline_seconds': overall_seconds,
        'operation_deadline_ceiling_seconds': 400, 'local_pool_bounded_by_host_envelope': True,
        'shares_host_pid_state': False, 'scheduler_configuration': scheduler.config.persisted_dict(),
        'container_resource_authority_pin': shared.pin(authority_raw)}
    index = registry = connection = None
    base._write(output / 'container-resource-authority.json', authority)

    def remaining(cap=400):
        seconds = deadline - time.monotonic()
        native.require(seconds > 0, 'full300 chunk qualification deadline exceeded')
        return min(cap, seconds)

    def options():
        return {'scheduler': scheduler, 'timeout_seconds': remaining(), 'memory_mb': 1024,
                'admission_timeout_seconds': min(30, remaining())}

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

    try:
        selected = native._pins(output, sorted(set(delta._implementation()['files']) | {
            MODULE, custody.__name__, deployment.__name__, shared.__name__, base.__name__,
            records.__name__, scan.__name__, 'ipfs_datasets_py.logic.backends.process'}))
        repository = output / 'repository'
        native._sources(repository)
        native.require(len([name for name in native._git(repository, 'ls-files', '-z').split('\0') if name]) == ENTRIES,
                       'original300 fixture population differs')
        index, registry, connection = phase('open_fresh_native_owners', lambda: native._open(output))
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
                epochs=1, learning_rate=.002, seed=1729, **options())
            version = result.to_dict()['version_id']
            native.require(native._state(registry, version)['completed_epochs'] - before == 1, 'explicit setup epoch differs')
            report['new_fitting_epochs'] += 1
            _persist(output, name + '-training-record.json', result)
            return version

        parent = phase('explicit_root_training_one_epoch', lambda: fit(previous.head, 'root'))
        (repository / 'calc.py').write_bytes(b'def increment(n: int) -> int:\n    return n + 2\n')
        (repository / 'bulk000.py').unlink()
        (repository / 'bulk001.py').rename(repository / 'renamed.py')
        (repository / 'added.py').write_bytes(b'def added(n: int) -> int:\n    return n + 700\n')
        native._git(repository, 'add', '-A')
        native._git(repository, 'commit', '--quiet', '--no-verify', '-m', 'Explicit full300 native chunk successor')
        native.require(len([name for name in native._git(repository, 'ls-files', '-z').split('\0') if name]) == ENTRIES,
                       'full300 successor lost original population')
        current = phase('publish_native_current_source', lambda: index.prepare_current(repository,
            operation_id='successor', expected_head=previous.head, **preparation, **options()))
        base._write(output / 'current-publication-receipt.json', current.to_dict())
        manifest = index.load(current.head.manifest_cid)
        native.require(len(manifest.snapshot.entries) == ENTRIES and manifest.snapshot.max_entries == 512
            and manifest.snapshot.max_file_bytes == 65536 and current.head.generation == 2, 'original full300 capture profile differs')
        base._write(output / 'current-source-manifest.json', manifest.to_dict())
        transition = phase('build_native_source_delta', lambda: delta.build_current_codebase_source_delta(
            index, repository, previous_head=previous.head, expected_head=current.head, **options()))
        _persist(output, 'source-delta.json', transition)
        child = phase('explicit_child_training_one_epoch', lambda: fit(current.head, 'child', parent))
        versions, frozen = {'root': parent, 'child': child}, _states(registry, {'root': parent, 'child': child})
        native.require(frozen['root']['completed_epochs'] == 1 and frozen['child']['completed_epochs'] == 2
            and all(row['latent_width'] == 8 and row['feature_columns'] == 53 for row in frozen.values()),
            'native CPU structural8D setup state or feature basis differs')
        base._write(output / 'checkpoint-states-before.json', frozen)
        completed = []
        with shared.no_fit(report), custody._no_install_guards(report):
            for optimized in (True, False):
                mode = 'default' if optimized else 'reference'
                root = phase('start_fresh_' + mode + '_root', lambda: scan.start_current_codebase_chunked_scan(
                    index, repository, expected_head=current.head, registry=registry, version_id=child,
                    limits=resume.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=PAGE_ENTRIES),
                    optimized=optimized, **options()))
                _persist(output, mode + '-root.json', root)
                cursor, pages = None, []
                for number in range(1, PARENT_PAGES + 1):
                    page = phase('infer_' + mode + '_page_' + str(number), lambda: scan.scan_current_codebase_page_chunked(
                        index, repository, root=root, registry=registry, cursor=cursor, **options()))
                    pages.append(page)
                    cursor = records.CodebaseScanChunkedCursor.from_page(root, page)
                    _export_page(output, mode, number, page, index)
                native.require(cursor is not None and cursor.next_offset == 128, 'full300 fresh-process cursor missing')
                native._close(registry, connection)
                index = registry = connection = None
                result_path, request_path = output / (mode + '-resume-result.json'), output / (mode + '-resume-request.json')
                from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
                with scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=512,
                        child_process_slots=1, timeout=min(30, remaining()),
                        request_id='full300-chunks-' + mode + '-fresh-process') as orchestration:
                    request = {'schema': RESUME_REQUEST, 'output': str(output), 'mode': mode, 'root_cid': root.artifact_cid,
                        'cursor': cursor.to_dict(), 'versions': versions, 'checkpoint_states': frozen,
                        'expected_head': current.head.to_dict(), 'authority': str(host_configuration),
                        'authority_sha256': hashlib.sha256(authority_raw).hexdigest(), 'orchestration': orchestration.to_dict(),
                        'parent_identity': custody._process(os.getpid()), 'result': str(result_path),
                        'max_pages': RESUME_PAGES, 'timeout_seconds': remaining(1800)}
                    base._write(request_path, request)
                    digest = hashlib.sha256(shared.read_absolute(request_path, 1_048_576)).hexdigest()
                    command = [sys.executable, '-B', '-m', MODULE, '--resume-request', str(request_path),
                               '--resume-request-sha256', digest]
                    with (output / (mode + '-resume.log')).open('xb') as log:
                        returncode, identity, cleanup = phase('fresh_process_' + mode + '_resume',
                            lambda: _launch_resume(command, log,
                                min(remaining(1820), request['timeout_seconds'] + 20), orchestration))
                native.require(returncode == 0, 'fresh chunk continuation failed')
                resumed = json.loads(shared.read_absolute(result_path, 1_048_576))
                native.require(resumed['qualified'] and resumed['request_sha256'] == digest
                    and resumed['post_setup_fit_attempt_count'] == resumed['new_fitting_epochs']
                    == resumed['receive_forward_attempt_count'] == 0 and resumed['checkpoint_states_after'] == frozen
                    and resumed['pid'] == identity['pid'] and resumed['process_identity']['birth'] == identity['birth'],
                    'fresh chunk continuation evidence differs')
                report['fresh_resume_processes'].append({'mode': mode, 'pid': resumed['pid'],
                    'process_identity': identity, 'request_sha256': digest, 'result': result_path.name,
                    'launch_group_cleanup': cleanup})
                shared.assert_clean(scheduler, owner_pids=[os.getpid(), resumed['pid']])
                index, registry, connection = phase('cold_reopen_after_' + mode, lambda: native._open(output))
                for number, descriptor in enumerate(resumed['pages'], PARENT_PAGES + 1):
                    page = scan.load_codebase_scan_chunked_page(index.artifacts, descriptor['page_cid'])
                    native.require(records.page_descriptor(page) == descriptor, 'fresh full300 page descriptor differs')
                    pages.append(page)
                    # Child persisted the ordinary page/chunk exports. Join
                    # each back to this process's independently read CAS.
                    _verify_page_exports(output, mode, number, page, index)
                with _no_forward(report):
                    completion = phase('complete_all_' + mode + '_pages', lambda: scan.complete_current_codebase_chunked_scan(
                        index, repository, root=root, registry=registry, tail_page_cid=pages[-1].artifact_cid, **options()))
                    phase('receive_current_' + mode + '_completion', lambda: scan.validate_current_codebase_chunked_scan_completion(
                        completion, index, repository, root=root, registry=registry, **options()))
                _persist(output, mode + '-completion.json', completion)
                coverage = completion.to_dict()['coverage']
                native.require(coverage['inventory_entries'] == ENTRIES and coverage['pages'] == len(pages) == PAGES
                    and coverage['inferred_rows'] == 294 and coverage['projection_records'] == 588
                    and coverage['dispositions'] == EXPECTED_DISPOSITIONS, 'full300 numerical completion coverage differs')
                report['new_' + mode + '_scan_pages'] = len(pages)
                report[mode + '_coverage'] = coverage
                report['actual_numerical_requests_by_mode'][mode] = coverage['chunks']
                completed.append((root, completion, pages))
            report['all_page_opt_out_equivalence'] = phase('compare_equal294_actual_numerical_rows',
                lambda: _compare_pages(index, completed[0][2], completed[1][2]))
            _persist(output, 'scan-root.json', completed[0][0])
            _persist(output, 'scan-completion.json', completed[0][1])
            after = _states(registry, versions)
            native.require(after == frozen and index.current(REPOSITORY_ID) == current.head, 'chunks changed model, Adam or source')
            base._write(output / 'checkpoint-states-after.json', after)
            native._require_pins(output, selected)
            absent = {'intent_database_absent': not (output / 'private' / 'intent.duckdb').exists(),
                'worker_launch_directory_absent': not (output / 'private' / 'launch').exists(),
                'worker_worktrees_absent': not (repository / '.git' / 'worktrees').exists()}
            native.require(all(absent.values()) and all(value == 0 for value in report['no_install_guards']['counters'].values())
                and report['new_fitting_epochs'] == 2 and report['post_setup_fit_attempt_count']
                == report['receive_forward_attempt_count'] == 0
                and len({row['pid'] for row in report['fresh_resume_processes']}) == 2,
                'chunk setup, receiving, fresh process or no-install accounting differs')
            report['no_install_guards'].update(absent)
            base._write(output / 'no-install-guards.json', report['no_install_guards'])
            report.update(qualified=True, complete_scan_qualified=True, large300_member_scan_qualified=True,
                actual294_numerical_rows_per_mode_qualified=True, current_chunked_receiving_qualified=True,
                selected_producers_unchanged=True, native_source_generation_advanced_by_worker=False)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        deployment._finalize_qualification_report(output, report, registry=registry, connection=connection,
            scheduler=scheduler, started=started, primary_error_active=sys.exc_info()[0] is not None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--overall-seconds', type=int, default=5950)
    parser.add_argument('--host-configuration', default='/results/container-resource-authority.json')
    parser.add_argument('--resume-request')
    parser.add_argument('--resume-request-sha256')
    args = parser.parse_args()
    if args.resume_request:
        native.require(args.resume_request_sha256 is not None and args.output is None, 'exact standalone chunk request required')
        _resume(args.resume_request, args.resume_request_sha256)
    else:
        native.require(args.output is not None and args.resume_request_sha256 is None, 'fresh full300 chunk output required')
        run(args.output, overall_seconds=args.overall_seconds, host_configuration=args.host_configuration)


if __name__ == '__main__':
    main()
