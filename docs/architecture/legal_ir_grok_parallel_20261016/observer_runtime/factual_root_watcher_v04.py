"""Root-owned observer for the unchanged native Database/Portal pipeline.

Callable factory only; import executes no product/provider/native job. Actual
loopback clients use the original private broker environment in a scoped parent
context. The caller supplies the genuine existing independent-validation owner
runner. This helper never replaces native execution callbacks or marks tasks
complete. It publishes a factual gate only after actual native evidence readback.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType


@contextmanager
def native_environment(environment):
    previous = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update({str(key): str(value) for key, value in environment.items()})
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def checked_module(pin, name):
    import hashlib
    path = Path(pin['path'])
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('exact pinned helper path required')
    before = path.lstat()
    raw = path.read_bytes()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']
            or before != path.lstat()):
        raise ValueError('helper source changed')
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ''
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def _save_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError('short watcher snapshot write')
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _projection_candidates(root):
    """Search only the declared native state namespace, with finite traversal."""
    root = Path(root)
    if not root.exists():
        return []
    found = []
    visited = 0
    for directory, children, files in os.walk(root, followlinks=False):
        visited += 1
        if visited > 512:
            raise ValueError('native projection search bound exceeded')
        current = Path(directory)
        children[:] = sorted(name for name in children if not (current / name).is_symlink())
        if len(current.relative_to(root).parts) >= 12:
            children[:] = []
        if 'task-projection.runtime.todo.md' in files:
            found.append(current / 'task-projection.runtime.todo.md')
            if len(found) > 32:
                raise ValueError('native projection candidate bound exceeded')
    return sorted(found)



def _native_portal_attempt(v, portal_state, observed):
    """Wait only for the native selected-but-not-started state publication.

    Native selection publishes the exact task tuple before implementation
    atomically publishes its positive attempt, phase and in-progress marker.
    This observational wait grants no attempt, evidence or completion authority.
    """
    v.need(portal_state.get('active_task_id') == v.TASK
           and portal_state.get('active_task_cid') == observed['native_portal_task_cid']
           and portal_state.get('active_task_key') == observed['native_portal_task_key'],
           'actual current Portal state identity differs')
    local_attempt = portal_state.get('active_attempt')
    v.need(type(local_attempt) is int and local_attempt >= 0,
           'actual local Portal attempt ordinal missing')
    if local_attempt == 0:
        v.need(portal_state.get('active_phase') == ''
               and portal_state.get('implementation_in_progress') is False,
               'actual local Portal attempt ordinal missing')
        return None
    return local_attempt


def make_factual_watcher(*, source_root, controller_prefix, worker_receipt_root,
                         projection_search_root, worktree_root, watcher_root,
                         gate_path, validator_path, gate_helper_pin,
                         projection_observer_pin, projection_owner_sha256,
                         private_state_observer_pin, private_state_owner_sha256,
                         source_guard, record_operation, independent_validation_runner):
    """Return watcher(live, child, plan) for run_retained_supervisor.

    independent_validation_runner(request) must use the real separate existing
    native validation owner, retain its full request/result/closeout, and return
    {'checked_result_pin': {path,bytes,sha256},
     'native_job_receipt_pin': {path,bytes,sha256}} after actual success.
    The request contains argv/native_environment and exact frozen input pins.
    Do not substitute an in-process self-authored passing receipt for that job.
    """
    if not all(callable(value) for value in (
            source_guard, record_operation, independent_validation_runner)):
        raise ValueError('actual native guard/recorder/independent job runner required')
    root = Path(source_root).resolve(strict=True)
    prefix = Path(controller_prefix).resolve(strict=True)
    receipts = Path(worker_receipt_root).resolve(strict=True)
    state_root = Path(projection_search_root)
    worktrees = Path(worktree_root)
    own = Path(watcher_root).resolve(strict=True)
    gate_path = Path(gate_path)
    if (own.lstat().st_uid != os.geteuid() or own.lstat().st_mode & 0o022
            or not gate_path.is_relative_to(own)):
        raise ValueError('native-owned watcher/gate root required')
    helper = checked_module(gate_helper_pin, 'legal_ir_native_gate_checked')
    observer = checked_module(projection_observer_pin, 'legal_ir_projection_observer_checked')
    private_state = checked_module(private_state_observer_pin, 'legal_ir_private_state_observer_checked')
    v = helper.load_validator(Path(validator_path))
    capsule = v.load_capsule(prefix)
    last = {'phase': None, 'attempt_id': None, 'finished': False, 'job_started': False}

    def status(phase, **extra):
        if phase != last['phase']:
            record_operation('factual_watcher', {'phase': phase, **extra,
                'task_status_mutated': False, 'completion_authority': False})
            last['phase'] = phase

    def watcher(live, child, plan):
        # All operations remain under the original live server/native custody.
        # The server's private environment is never printed or persisted.
        if last['finished']:
            return
        source_guard()
        with native_environment(live['environment']):
            module = importlib.import_module(
                'ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source')
            v.need(Path(module.__file__).resolve(strict=True).is_relative_to(root),
                   'foreign native mutation client')
            with module.DatabaseTaskSource(live['program'].quack_endpoint,
                    install_schema=False, owner_id=f'database-task-source:{os.getpid()}') as source:
                task = source.get('LIR-9001')
                if task is None or task.status == 'todo':
                    status('waiting_for_actual_native_claim')
                    return
                if task.status in {'completed', 'complete', 'done'}:
                    status('native_task_completed', task_cid=task.task_cid)
                    last['finished'] = True
                    return
                if task.status != 'in_progress':
                    status('native_task_pending_or_blocked', native_status=task.status)
                    return
                claim = helper.plain(task.body.get('completion_receipt') or {})
                if claim.get('operation') not in {'database_claim', 'database_attempt_admitted'}:
                    status('waiting_for_admitted_native_claim_receipt')
                    return
                v.need(task.task_cid == capsule['contract']['selected_task_cid'],
                       'actual native task CID differs from protected packet')
                attempt_id = claim['attempt_id']
                v.need(last['attempt_id'] in (None, attempt_id),
                       'native attempt changed; independent retry reconciliation required')
                last['attempt_id'] = attempt_id
                expected_claim = {'task_cid': task.task_cid, 'task_revision': task.revision,
                    'board_namespace': v.NAMESPACE,
                    **{key: claim[key] for key in ('attempt_id', 'attempt_number', 'claim_id',
                                                  'lease_id', 'fencing_token', 'fence_epoch')}}
                matches = []
                for projection in _projection_candidates(state_root):
                    # Reject another attempt before calling the genuine observer.
                    binding_file = projection.parent / 'database-attempt-binding.json'
                    if not binding_file.exists():
                        continue
                    binding_raw, _ = v.read_checked(binding_file)
                    binding = v.strict_json(binding_raw)
                    if binding.get('attempt_id') != attempt_id:
                        continue
                    matches.append(observer.observe_portal_projection(
                        source_root=root, projection_path=projection,
                        expected_claim=expected_claim, source_sha256=projection_owner_sha256))
                v.need(len(matches) <= 1, 'actual native projection is ambiguous')
                if not matches:
                    status('waiting_for_actual_native_portal_projection', native_attempt_id=attempt_id)
                    return
                observed = matches[0]
                # Native portal_factory binds this one-attempt state to a
                # private DuckDB only in the daemon process. The original
                # process-local binding cannot be inferred from root's ambient
                # Quack environment. Observe the source-derived private target
                # read-only under the native writer's existing file lock.
                try:
                    state_raw, state_observation = private_state.observe_private_portal_state(
                        source_root=root, projection_observation=observed,
                        expected_claim=expected_claim, source_sha256=private_state_owner_sha256)
                except private_state.PrivatePortalStatePending:
                    status('waiting_for_actual_native_portal_state', native_attempt_id=attempt_id)
                    return
                state_binding_id = state_observation['observation_id']
                portal_state = v.strict_json(state_raw)
                if not portal_state.get('active_task_id'):
                    status('waiting_for_actual_native_portal_active_tuple', native_attempt_id=attempt_id)
                    return
                local_attempt = _native_portal_attempt(v, portal_state, observed)
                if local_attempt is None:
                    status('waiting_for_actual_native_portal_attempt', native_attempt_id=attempt_id)
                    return
                workspace = Path(portal_state['active_worktree_path'])
                v.need(workspace.is_absolute() and workspace.resolve(strict=True) == workspace
                       and workspace.is_relative_to(worktrees.resolve(strict=True)),
                       'native Portal workspace escapes declared worktree root')
                report_path = workspace / 'data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json'
                if not report_path.exists():
                    status('waiting_for_frozen_grok_candidate', native_attempt_id=attempt_id,
                           portal_attempt=local_attempt)
                    return
                if last['job_started']:
                    status('independent_validation_already_started_reconciliation_required')
                    return
                report_raw, report_pin = v.read_checked(report_path)
                report = v.strict_json(report_raw)
                v.need(report['task_cid'] == observed['native_portal_task_cid']
                       and report['attempt'] == local_attempt,
                       'candidate is not the independently observed native Portal attempt')
                invocation_pin = report['invocation_receipt']
                invocation_raw, actual_invocation_pin = v.read_checked(
                    receipts / invocation_pin['name'], pin=invocation_pin)
                invocation = v.strict_json(invocation_raw)
                prompt_pin = invocation['native_prompt_artifact']
                _, actual_prompt_pin = v.read_checked(receipts / prompt_pin['name'], maximum=128000, pin=prompt_pin)
                native = {'task_cid': task.task_cid, 'portal_task_cid': observed['native_portal_task_cid'],
                    'task_revision': capsule['contract']['execution_packet_revision'],
                    'attempt': local_attempt, 'native_attempt_id': attempt_id,
                    'database_attempt_number': claim['attempt_number'],
                    'report_sha256': report_pin['sha256'],
                    'invocation_sha256': actual_invocation_pin['sha256'],
                    'native_prompt_sha256': actual_prompt_pin['sha256']}
                # This is a real native-state snapshot, not a model-authored
                # identity assertion. It remains nonauthoritative by itself.
                state_snapshot = own / 'LIR-9001-actual-portal-state.json'
                _save_new(state_snapshot, state_raw)
                _, state_pin = v.read_checked(state_snapshot)
                observation = {
                    'database_task_cid': task.task_cid, 'portal_task_cid': observed['native_portal_task_cid'],
                    'native_attempt_id': attempt_id, 'database_attempt_number': claim['attempt_number'],
                    'portal_attempt_number': local_attempt, 'claim_id': claim['claim_id'],
                    'lease_id': claim['lease_id'], 'fencing_token': claim['fencing_token'],
                    'fence_epoch': claim['fence_epoch'],
                    'execution_packet_revision': capsule['contract']['execution_packet_revision'],
                    'projection_pin': observed['projection_file'],
                    'projection_binding_pin': observed['binding_file'],
                    'portal_state_pin': {'path': str(state_snapshot), **state_pin}}
                environment = {name: str(native[key]) for key, name in v.NATIVE_FIELDS.items()}
                request = {'schema': 'legal-ir-independent-factual-validation-request/v2',
                    'argv': ['/usr/bin/python3', '-B', str(validator_path), '--check-report', str(report_path),
                             '--controller-prefix', str(prefix), '--receipt-root', str(receipts)],
                    'native_environment': environment, 'native_bindings': native,
                    'report_pin': {'path': str(report_path), **report_pin},
                    'invocation_pin': {'path': str(receipts / invocation_pin['name']), **actual_invocation_pin},
                    'original_prompt_pin': {'path': str(receipts / prompt_pin['name']), **actual_prompt_pin},
                    'portal_projection_observation': observed,
                    'native_portal_state_binding_id': state_binding_id,
                    'private_native_portal_state_observation': state_observation,
                    'actual_claim_and_portal_observation': observation,
                    'factual_evidence_kind': v.EVIDENCE_KIND,
                    'task_completion_authority': False}
                last['job_started'] = True
                status('independent_native_factual_validation_started', native_attempt_id=attempt_id,
                       portal_attempt=local_attempt)
                source_guard()
                result = independent_validation_runner(request)
                v.need(type(result) is dict and set(result) == {
                    'checked_result_pin', 'native_job_receipt_pin'},
                    'actual independent native validation output pins required')
                v.read_checked(Path(result['native_job_receipt_pin']['path']),
                               pin=result['native_job_receipt_pin'])
                record_operation('independent_native_factual_job_returned', result)
                source_guard()
                published = helper.publish_gate_after_checked_result(
                    source, validator_path=Path(validator_path), controller_prefix=prefix,
                    checked_result_pin=result['checked_result_pin'], report_path=report_path,
                    receipt_root=receipts, actual_claim_and_portal_observation=observation,
                    gate_path=gate_path, source_guard=source_guard, record_operation=record_operation)
                status('actual_factual_evidence_recorded_gate_published', gate=published['gate'])
                last['finished'] = True
    return watcher
