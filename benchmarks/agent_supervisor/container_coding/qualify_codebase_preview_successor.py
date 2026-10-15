"""Fresh native 32-member scan and signed-preview child worker qualification.

This explicit CPU8D fixture trains two setup epochs. Every optimized and
opt-out page executes here; a fresh process resumes each halfway cursor.
The real signed worker preserves two authored tasks and their public checks.
It does not qualify a 300-member scan, other widths, CUDA or semantic proof.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from . import qualify_codebase_inventory_resume as native
from . import qualify_codebase_inventory_scan as base
from . import qualify_codebase_signed_successor as worker
from . import source_successor_full_scan_fixture as shared

MODULE = 'benchmarks.agent_supervisor.container_coding.qualify_codebase_preview_successor'
REPOSITORY_ID = 'qualification:source-successor'
SCHEMA = 'codebase-preview-successor-fresh-native-qualification@1'


def _persist(output, name, record):
    base._write(output / name, {'artifact_cid': record.artifact_cid, 'value': record.to_dict()})


def _resume(request_path, expected_sha256):
    """Own fresh-process continuation of a parent-authored exact cursor."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
    request_path = Path(request_path)
    raw = shared.read_absolute(request_path, 1_048_576)
    native.require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'resume request bytes differ')
    request = json.loads(raw)
    native.require(set(request) == {'schema', 'output', 'root_cid', 'cursor', 'version_id',
        'expected_head', 'state', 'authority', 'authority_sha256', 'orchestration',
        'result', 'max_pages'}, 'closed native resume request required')
    native.require(request['schema'] == 'preview-successor-native-resume-request@1'
        and request['max_pages'] == 2, 'bounded exact native continuation required')
    output = Path(request['output'])
    authority_raw = shared.read_absolute(Path(request['authority']), 1_048_576)
    native.require(hashlib.sha256(authority_raw).hexdigest() == request['authority_sha256'],
        'parent-verified container resource authority changed')
    authority = json.loads(authority_raw)
    native.require(authority['namespace'] == {name: os.readlink('/proc/self/ns/' + name)
        for name in ('pid', 'mnt', 'net')}, 'resume authority belongs to another namespace')
    scheduler = shared.shared_scheduler(authority['state_path'], authority['persisted_config'])
    with scheduler._locked_state(persist=False) as state:
        held = state['leases'].get(request['orchestration']['lease_id'])
        native.require(held is not None and held['owner_pid'] == os.getppid()
            and held['cpu_slots'] == 1 and held['memory_mb'] == 512
            and held['child_process_slots'] == 1, 'fresh process has no held parent orchestration reservation')
    index = registry = connection = None
    report = {'schema': 'preview-successor-native-resume-result@1', 'qualified': False,
        'pid': os.getpid(), 'request_sha256': expected_sha256, 'pages': [], 'proof_authority': False,
        'post_setup_fit_attempt_count': 0}
    try:
        index, registry, connection = native._open(output)
        native.require(index.current(REPOSITORY_ID).to_dict() == request['expected_head'], 'resume source head changed')
        native.require(native._state(registry, request['version_id'], include_identity=True) == request['state'],
            'resume weights or Adam state changed')
        root = scan.load_codebase_scan_resume_root(index.artifacts, request['root_cid'])
        cursor = scan.CodebaseScanResumeCursor.from_dict(request['cursor'])
        with shared.no_fit(report):
            for number in range(request['max_pages']):
                page = scan.scan_current_codebase_page(index, output / 'repository', root=root,
                    registry=registry, cursor=cursor, scheduler=scheduler, memory_mb=1024,
                    timeout_seconds=300, admission_timeout_seconds=30)
                report['pages'].append({'artifact_cid': page.artifact_cid, 'value': page.to_dict()})
                cursor = page.next_cursor
                if cursor is None:
                    break
        native.require(cursor is None and len(report['pages']) == 2, 'native continuation did not finish exactly two pages')
        report['state_after'] = native._state(registry, request['version_id'], include_identity=True)
        native.require(report['state_after'] == request['state'], 'resume numerical state changed')
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
                result = action()
                if name == 'observe_owned_resources':
                    report['resources_after'] = result
            except BaseException as error:
                report['qualified'] = False
                report.setdefault('cleanup_errors', []).append({'stage': name,
                    'error_type': type(error).__name__, 'error': str(error)})
                cleanup_error = cleanup_error or error
        base._write(Path(request['result']), report)
        if cleanup_error is not None and not primary_error_active:
            raise cleanup_error
    return report


def run(output, *, overall_seconds=2950, host_configuration='/results/container-resource-authority.json'):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features

    native.require(type(overall_seconds) is int and 900 <= overall_seconds <= 3550,
        'bounded native qualification deadline required')
    output = Path(output).absolute()
    native.require(not output.exists() and output.parent.resolve(strict=True) == output.parent,
        'fresh exact native output required')
    output.mkdir()
    scheduler, authority, authority_raw = worker._container_scheduler(Path(host_configuration))
    started = time.monotonic()
    deadline = started + overall_seconds
    report = {'schema': SCHEMA, 'qualified': False, 'pid': os.getpid(), 'phases': [], 'controls': [],
        'scope': 'fresh32_member_all_pages_cpu8d_and_signed_preview_two_task_native_worker',
        'inventory_entries': 32, 'page_entries': 8, 'new_fitting_epochs': 0,
        'new_default_scan_pages': 0, 'new_reference_scan_pages': 0, 'fresh_resume_processes': [],
        'post_setup_fit_attempt_count': 0, 'inference_attempt_count': 0, 'provider_calls': 0,
        'proof_authority': False, 'source_execution_attested': False, 'scan_execution_attested': False,
        'cuda_qualified': False, '384d_qualified': False, '768d_qualified': False, '4096d_qualified': False,
        'production_default_activated': False, 'large300_member_scan_qualified': False,
        'semantic_reproof_join_qualified': False, 'overall_deadline_seconds': overall_seconds,
        'local_pool_bounded_by_host_envelope': True, 'shares_host_pid_state': False,
        'scheduler_configuration': scheduler.config.persisted_dict(),
        'container_resource_authority_pin': shared.pin(authority_raw)}
    index = registry = connection = None
    base._write(output / 'container-resource-authority.json', authority)

    def remaining(cap=180):
        seconds = deadline - time.monotonic()
        native.require(seconds > 0, 'fresh native qualification deadline exceeded')
        return min(cap, seconds)

    def options(cap=180):
        return {'scheduler': scheduler, 'timeout_seconds': remaining(cap), 'memory_mb': 1024,
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

    def forbidden_fit(*args, **kwargs):
        report['post_setup_fit_attempt_count'] += 1
        raise AssertionError('scan/worker attempted fitting after explicit setup')

    def forbidden_forward(*args, **kwargs):
        report['inference_attempt_count'] += 1
        raise AssertionError('receiving or worker attempted forward inference')

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
            'ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction',
            'ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview',
            'ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context',
            'ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler',
            'ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts',
            'ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service',
            'ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime'}))
        repository = output / 'repository'
        native._sources(repository)
        for number in range(23, 291):
            (repository / f'bulk{number:03d}.py').unlink()
        native._git(repository, 'add', '-A')
        native._git(repository, 'commit', '--quiet', '--no-verify', '-m', 'Explicit 32-member native preview fixture')
        native.require(len([item for item in native._git(repository, 'ls-files', '-z').split('\0') if item]) == 32,
            'native fixture population differs')
        index, registry, connection = phase('open_fresh_native_source_model_owners', lambda: native._open(output))
        preparation = {'repository_id': REPOSITORY_ID, 'limits': CodebaseScanLimits(64, 65536),
            'exclusions': ('.runtime',)}
        previous = phase('publish_native_previous_source', lambda: index.prepare_current(repository,
            operation_id='initial', expected_head=None, **preparation, **options()))
        base._write(output / 'previous-publication-receipt.json', previous.to_dict())
        selections = [training.CodebaseTrainingSelection(path, role) for path, role in
            (('calc.py', 'train'), ('tune.py', 'tune'), ('canary.py', 'canary'))]

        def fit(head, name, parent=None):
            before = 0 if parent is None else native._state(registry, parent)['completed_epochs']
            result = training.train_current_codebase_features(index, repository, expected_head=head,
                registry=registry, selections=selections, operation_id=name, parent_version_id=parent,
                epochs=1, learning_rate=.002, seed=1729, **options(300))
            version = result.to_dict()['version_id']
            native.require(native._state(registry, version)['completed_epochs'] - before == 1,
                'actual explicit setup epoch differs')
            report['new_fitting_epochs'] += 1
            _persist(output, name + '-training-record.json', result)
            return version

        parent = phase('explicit_native_root_training_one_epoch', lambda: fit(previous.head, 'root'))
        (repository / 'calc.py').write_bytes(b'def increment(n: int) -> int:\n    return n + 2\n')
        (repository / 'bulk000.py').unlink()
        (repository / 'bulk001.py').rename(repository / 'renamed.py')
        (repository / 'added.py').write_bytes(b'def added(n: int) -> int:\n    return n + 700\n')
        native._git(repository, 'add', '-A')
        native._git(repository, 'commit', '--quiet', '--no-verify', '-m', 'Explicit native source successor')
        current = phase('publish_native_current_source', lambda: index.prepare_current(repository,
            operation_id='successor', expected_head=previous.head, **preparation, **options()))
        base._write(output / 'current-publication-receipt.json', current.to_dict())
        transition = phase('build_native_source_delta', lambda: delta.build_current_codebase_source_delta(
            index, repository, previous_head=previous.head, expected_head=current.head, **options()))
        _persist(output, 'source-delta.json', transition)
        child = phase('explicit_native_child_training_one_epoch', lambda: fit(current.head, 'child', parent))
        frozen = {name: native._state(registry, version, include_identity=True)
            for name, version in (('root', parent), ('child', child))}
        base._write(output / 'checkpoint-states-before.json', frozen)
        completed = []
        with ExitStack() as guards:
            guards.enter_context(base._patch(training, 'train_current_codebase_features', forbidden_fit))
            guards.enter_context(base._patch(features, 'train_projection_features', forbidden_fit))
            for optimized in (True, False):
                mode = 'default' if optimized else 'reference'
                selection = phase('select_native_' + mode + '_successor', lambda: successor.start_current_codebase_successor_scan(
                    index, repository, source_delta=transition, registry=registry,
                    previous_version_id=parent, version_id=child,
                    limits=scan.CodebaseScanResumeLimits(max_inventory_entries=64, page_entries=8),
                    optimized=optimized, **options(300)))
                root = scan.load_codebase_scan_resume_root(index.artifacts, selection.to_dict()['root_cid'])
                _persist(output, mode + '-selection.json', selection)
                _persist(output, mode + '-root.json', root)
                cursor = None
                pages = []
                for number in (1, 2):
                    page = phase('infer_' + mode + '_page_' + str(number), lambda: scan.scan_current_codebase_page(
                        index, repository, root=root, registry=registry, cursor=cursor, **options(300)))
                    pages.append(page)
                    cursor = page.next_cursor
                    _persist(output, mode + '-page-' + str(number) + '.json', page)
                native.require(cursor is not None, 'resumable halfway cursor missing')
                native._close(registry, connection)
                index = registry = connection = None
                result_path = output / (mode + '-resume-result.json')
                request = {'schema': 'preview-successor-native-resume-request@1', 'output': str(output),
                    'root_cid': root.artifact_cid, 'cursor': cursor.to_dict(), 'version_id': child,
                    'expected_head': current.head.to_dict(), 'state': frozen['child'],
                    'authority': str(host_configuration), 'result': str(result_path), 'max_pages': 2}
                request_path = output / (mode + '-resume-request.json')
                from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
                with scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=512,
                        child_process_slots=1, timeout=min(30, remaining()),
                        request_id='preview-native-' + mode + '-fresh-process') as orchestration:
                    request.update(authority_sha256=hashlib.sha256(authority_raw).hexdigest(),
                        orchestration=orchestration.to_dict())
                    base._write(request_path, request)
                    digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
                    command = [sys.executable, '-B', '-m', MODULE, '--resume-request', str(request_path),
                        '--resume-request-sha256', digest]
                    with (output / (mode + '-resume.log')).open('xb') as log:
                        resumed = phase('fresh_process_' + mode + '_resume', lambda: subprocess.run(command,
                            stdout=log, stderr=subprocess.STDOUT, timeout=remaining(650), check=False))
                native.require(resumed.returncode == 0, 'fresh native continuation failed')
                resumed_report = json.loads(result_path.read_bytes())
                native.require(resumed_report['qualified'] and resumed_report['request_sha256'] == digest,
                    'fresh continuation receipt differs')
                native.require(resumed_report['post_setup_fit_attempt_count'] == 0,
                    'fresh continuation attempted model fitting')
                report['fresh_resume_processes'].append({'mode': mode, 'pid': resumed_report['pid'],
                    'request_sha256': digest, 'result': result_path.name})
                index, registry, connection = phase('cold_reopen_after_' + mode, lambda: native._open(output))
                for number, row in enumerate(resumed_report['pages'], 3):
                    page = scan.load_codebase_scan_resume_page(index.artifacts, row['artifact_cid'])
                    native.require(page.to_dict() == row['value'], 'resumed native page bytes differ')
                    pages.append(page)
                    _persist(output, mode + '-page-' + str(number) + '.json', page)
                completion = phase('complete_all_' + mode + '_pages', lambda: scan.complete_current_codebase_scan(
                    index, repository, root=root, registry=registry, tail_page_cid=pages[-1].artifact_cid,
                    **options(300)))
                _persist(output, mode + '-completion.json', completion)
                native.require(completion.to_dict()['coverage']['inventory_entries'] == 32
                    and completion.to_dict()['coverage']['pages'] == len(pages) == 4, 'full native scan population differs')
                report['new_' + ('default' if optimized else 'reference') + '_scan_pages'] = len(pages)
                completed.append((selection, root, completion, pages))
            for first, reference in zip(completed[0][3], completed[1][3]):
                native.require(all(shared.inert.observation_same(first.to_dict()[key], reference.to_dict()[key])
                    for key in ('entries', 'coverage', 'inference')), 'full default/opt-out numerical outcomes differ')
            report['all_page_opt_out_equivalence'] = {'pages': 4, 'inventory_entries': 32,
                'entries_coverage_and_inference_exact': True, 'throughput_qualified': False}
            selection, root, completion, _ = completed[0]
            _persist(output, 'successor-selection.json', selection)
            _persist(output, 'scan-root.json', root)
            _persist(output, 'scan-completion.json', completion)
            with base._patch(scan, '_worker', forbidden_forward), base._patch(features, 'infer_projection_features', forbidden_forward):
                worker._qualify_admission_worker(output, index, registry, selection, root, completion,
                    scheduler, phase, report, options, preview_bound=True, fresh_setup=True)
            after = {name: native._state(registry, version, include_identity=True)
                for name, version in (('root', parent), ('child', child))}
            native.require(after == frozen, 'native scans or worker changed weights/Adam state')
            base._write(output / 'checkpoint-states-after.json', after)
            native._require_pins(output, selected)
            native.require(report['new_fitting_epochs'] == 2 and report['post_setup_fit_attempt_count'] == 0
                and report['inference_attempt_count'] == 0, 'explicit setup/receiving accounting differs')
            native.require(len({row['pid'] for row in report['fresh_resume_processes']}) == 2,
                'fresh native resume process identities repeat')
            report.update(qualified=True, native_worker_qualified=True, full_administrator_population_completed=True,
                complete_scan_qualified=True, selected_producers_unchanged=True,
                native_source_generation_advanced_by_worker=False)
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
    parser.add_argument('--overall-seconds', type=int, default=2950)
    parser.add_argument('--host-configuration', default='/results/container-resource-authority.json')
    parser.add_argument('--resume-request')
    parser.add_argument('--resume-request-sha256')
    args = parser.parse_args()
    if args.resume_request:
        native.require(args.resume_request_sha256 is not None and args.output is None,
            'exact standalone resume request required')
        _resume(args.resume_request, args.resume_request_sha256)
    else:
        native.require(args.output is not None and args.resume_request_sha256 is None,
            'explicit fresh native output required')
        run(args.output, overall_seconds=args.overall_seconds, host_configuration=args.host_configuration)


if __name__ == '__main__':
    main()
