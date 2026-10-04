"""Manual metadata-only projection of the parent's completed trial receipt.

No polling or process control. Reads only the named outer receipt; never follows
paths into verifier output, credential stores, model traces or source bodies.
"""
import json
import hashlib
import math
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
INPUT = OUT / 'trial-01/receipt.json'
MAX_BYTES = 16 * 1024 * 1024


def scalar(value):
    if value is None or type(value) is bool:
        return value
    if type(value) in (int,float) and math.isfinite(value):
        return value
    if type(value) is str and len(value) <= 200 and re.fullmatch(r'[A-Za-z0-9_:/@.+=-]*',value):
        return value
    return '[excluded_non_scalar_or_body]'


def pick(value, fields):
    return {key:scalar(value[key]) for key in fields if type(value) is dict and key in value}


def numbers(value):
    if type(value) is not dict or len(value) > 64:
        return {}
    return {key:item for key,item in value.items() if type(key) is str
        and re.fullmatch(r'[A-Za-z0-9_]{1,80}',key) and type(item) in (int,float)
        and math.isfinite(item)}


CONTROL_STATUSES = {
    'succeeded', 'failed', 'denied', 'conflict', 'not_found', 'cancelled',
    'timed_out', 'unavailable',
}
CONTROL_ERROR_CODES = {
    'invalid_request', 'unknown_operation', 'unauthorized', 'forbidden', 'not_found',
    'conflict', 'stale_tree', 'stale_lease', 'bounds_exceeded', 'unsupported_version',
    'unsupported_capability', 'invalid_cursor', 'cursor_expired', 'path_escape',
    'idempotency_required', 'idempotency_conflict', 'authority_violation',
    'invalid_lifecycle_transition', 'unavailable', 'timed_out', 'cancelled', 'internal_error',
}
TRANSACTION_PHASES = {
    'prepared', 'dispatching', 'committed', 'compensation_required',
    'repair_required', 'compensated', 'repaired',
}
CONTROL_REASONS = {
    'startup did not prove sustained health before its deadline': 'sustained_health_not_proved',
    'another lifecycle transition is still active': 'lifecycle_transition_active',
}
PROJECTOR_MEMBER = 'source/benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py'
SUCCESSOR_MEMBERS = (
    'source/ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
    'source/ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py',
    'source/ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py',
    'source/ipfs_accelerate_py/agent_supervisor/runtime/published_retrieval.py',
    'source/ipfs_accelerate_py/agent_supervisor/runtime/published_learned_retrieval.py',
)
REFRESH_FAILURE_PHASES = {'rebuilder_selection', 'cache_reload', 'successor_refresh'}


def source384_successor_projection(value, *, projected=False):
    """Only closed advisory receipt identities and bounded inference observations."""
    if type(value) is not dict or value.get('schema') != 'supervisor-source384-successor-observation@2':
        raise ValueError('closed Source384 successor observation required')
    expected = {'schema', 'status', 'predecessor_receipt_sha256', 'receipt_sha256',
        'config_sha256', 'checkpoint_sha256', 'inference_sha256', 'source_head', 'version_id',
        'preserved_paths', 'declared_outputs_outside_index_scope', 'requires_independent_manifest',
        'header_nomination_inherited', 'historical_currentness_verified', 'source_semantics_verified',
        'proof_authority', 'planning_authority', 'dispatch_authority', 'execution_authority',
        'completion_authority', 'training_steps', 'text_generation_calls',
        'inference_activity_scope', 'inference_execution'}
    if projected:
        expected -= {'preserved_paths', 'declared_outputs_outside_index_scope'}
        expected |= {'preserved_path_count', 'declared_outputs_outside_index_scope_count'}
    if set(value) != expected or value['status'] != 'current_advice':
        raise ValueError('Source384 successor observation fields differ')
    for key in ('predecessor_receipt_sha256', 'receipt_sha256', 'config_sha256',
                'checkpoint_sha256', 'inference_sha256'):
        if type(value[key]) is not str or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('Source384 observation identity is not a SHA256')
    flags = ('header_nomination_inherited', 'historical_currentness_verified',
        'source_semantics_verified', 'proof_authority', 'planning_authority',
        'dispatch_authority', 'execution_authority', 'completion_authority')
    if (any(value[key] is not False for key in flags)
            or value['requires_independent_manifest'] is not True
            or any(type(value[key]) is not int or value[key] != 0
                   for key in ('training_steps', 'text_generation_calls'))
            or value['inference_activity_scope'] != 'successor_artifact_preparation'):
        raise ValueError('Source384 observation cannot inherit authority')
    execution = value['inference_execution']
    if (type(execution) is not dict or set(execution) !=
            {'native_worker_executed', 'inference_executed', 'model_loads'}
            or execution['native_worker_executed'] is not True
            or execution['inference_executed'] is not True
            or type(execution['model_loads']) is not int or not 0 <= execution['model_loads'] <= 1024):
        raise ValueError('bounded Source384 inference activity required')
    head = value['source_head']
    if (type(head) is not dict or set(head) != {'schema', 'repository_id', 'generation',
            'manifest_cid', 'snapshot_cid', 'ast_revision_id', 'receipt_cid'}
            or type(head['generation']) is not int or not 1 <= head['generation'] < 2**63
            or any(type(item) is not str or not 1 <= len(item) <= 1024
                or not re.fullmatch(r'[A-Za-z0-9_:/@.+=-]+', item)
                for key, item in head.items() if key != 'generation')):
        raise ValueError('bounded Source384 source head required')
    if type(value['version_id']) is not str or not re.fullmatch(r'[A-Za-z0-9_:/@.+=-]{1,512}', value['version_id']):
        raise ValueError('bounded Source384 model version required')
    if projected:
        if any(type(value[key]) is not int or not 0 <= value[key] <= 256 for key in
               ('preserved_path_count', 'declared_outputs_outside_index_scope_count')):
            raise ValueError('bounded Source384 population counts required')
        return dict(value)
    for key in ('preserved_paths', 'declared_outputs_outside_index_scope'):
        paths = value[key]
        if (type(paths) is not list or len(paths) > 256
                or any(type(path) is not str or not 1 <= len(path) <= 1024 for path in paths)
                or paths != sorted(set(paths))):
            raise ValueError('bounded Source384 population required')
    return {key: item for key, item in value.items()
            if key not in {'preserved_paths', 'declared_outputs_outside_index_scope'}} | {
        'preserved_path_count': len(value['preserved_paths']),
        'declared_outputs_outside_index_scope_count': len(value['declared_outputs_outside_index_scope'])}


def successor_rows_projection(context):
    values = context.get('results', []) if type(context) is dict else []
    if type(values) is not list or len(values) > 64:
        raise ValueError('bounded refreshed context results required')
    result = []
    for value in values:
        if type(value) is not dict or 'source384_refresh' not in value:
            continue
        if type(value.get('source384_refresh_reused')) is not bool:
            raise ValueError('exact successor reuse observation required')
        result.append({'task_cid': scalar(value.get('task_cid')),
            'source384_refresh_reused': value['source384_refresh_reused'],
            'source384_refresh': source384_successor_projection(value['source384_refresh'])})
    return result


def published_context_row_projection(value, *, projected=False):
    """Retain each derivative's outcome even when no successor was produced."""
    fields = {'task_cid', 'status', 'error_type', 'error_phase', 'completion_authority'}
    if type(value) is not dict or (projected and not set(value) <= fields):
        raise ValueError('closed published context outcome required')
    task_cid = value.get('task_cid')
    if type(task_cid) is not str or not re.fullmatch(r'[A-Za-z0-9_:/@.+=-]{1,512}', task_cid):
        raise ValueError('bounded published context task identity required')
    if type(value.get('status')) is not str or value['status'] not in {'refreshed', 'unavailable', 'pending_completion',
            'pending_native_stop', 'retry_budget_exhausted', 'successor_unavailable'}:
        raise ValueError('closed published context status required')
    if 'error_type' in value and (type(value['error_type']) is not str
            or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', value['error_type'])):
        raise ValueError('bounded published context error class required')
    if 'error_phase' in value and (type(value['error_phase']) is not str
            or value['error_phase'] not in REFRESH_FAILURE_PHASES
            or value['status'] != 'unavailable'):
        raise ValueError('closed published context failure phase required')
    if 'completion_authority' in value and value['completion_authority'] is not False:
        raise ValueError('derivative outcome cannot grant completion authority')
    return {key: value[key] for key in fields if key in value}


def published_context_rows_projection(context):
    values = context.get('results', []) if type(context) is dict else []
    if type(values) is not list or len(values) > 64:
        raise ValueError('bounded published context result sequence required')
    return [published_context_row_projection(value) for value in values]


def control_result_projection(value):
    """Project typed control metadata; classify only exact known messages."""
    if type(value) is not dict:
        return {}
    result = {}
    if type(value.get('status')) is str and value['status'] in CONTROL_STATUSES:
        result['status'] = value['status']
    error = value.get('error')
    if type(error) is dict:
        projected = {}
        if type(error.get('code')) is str and error['code'] in CONTROL_ERROR_CODES:
            projected['code'] = error['code']
        if type(error.get('retryable')) is bool:
            projected['retryable'] = error['retryable']
        details = error.get('details')
        exception_type = details.get('exception_type') if type(details) is dict else None
        if type(exception_type) is str and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', exception_type):
            projected['exception_type'] = exception_type
        message = error.get('message')
        projected['reason'] = (CONTROL_REASONS.get(message, 'unclassified_control_error')
            if type(message) is str else 'unclassified_control_error')
        result['error'] = projected
    data = value.get('data')
    transaction = data.get('transaction') if type(data) is dict else None
    if type(transaction) is dict:
        projected = {}
        for key, allowed in (
                ('phase', TRANSACTION_PHASES),
                ('recovery_action', {'none', 'compensate', 'repair'}),
                ('failure_code', CONTROL_ERROR_CODES)):
            if type(transaction.get(key)) is str and transaction[key] in allowed:
                projected[key] = transaction[key]
        result['transaction'] = projected
    return result


def startup_budget_projection(supervisor):
    result = {}
    for key, lower, upper in (
            ('native_start_timeout_seconds', 2, 120),
            ('native_start_timeout_ms', 2000, 120000),
            ('native_stop_timeout_ms', 2000, 30000)):
        value = supervisor.get(key)
        if type(value) is int and lower <= value <= upper:
            result[key] = value
    if supervisor.get('native_startup_error') == 'collection_unavailable':
        result['native_startup_error'] = 'collection_unavailable'
    return result


def load_startup_projector():
    """Bind the imported closed projector to the complete frozen archive map."""
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as module
    source = Path(module.__file__).resolve()
    expected = OUT.parent.parent / '.worktrees/ir-release-accelerate-20261002' / PROJECTOR_MEMBER.removeprefix('source/')
    assert source == expected.resolve(), 'startup projector came from an unexpected checkout'
    assert source.is_file() and not source.is_symlink() and source.stat().st_size <= 262144
    source_raw = source.read_bytes()
    assert len(source_raw) <= 262144
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    pins_path = OUT / 'frozen-production-pins.json'
    assert pins_path.is_file() and not pins_path.is_symlink() and pins_path.stat().st_size <= MAX_BYTES
    pins = json.loads(pins_path.read_text())
    assert pins.get(PROJECTOR_MEMBER) == source_sha256, 'startup projector differs from the frozen archive'
    checkout = expected.parents[3]
    successor_pins = {}
    for member in SUCCESSOR_MEMBERS:
        path = checkout / member.removeprefix('source/')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert pins.get(member) == digest, 'successor producer differs from the frozen archive'
        successor_pins[member] = digest
    provenance = {
        'schema': 'bounded-full-trial-inspection-provenance@1',
        'inspector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_startup_projector': {
            'source_member': PROJECTOR_MEMBER,
            'sha256': source_sha256,
            'frozen_source_match': True,
        },
        'source384_successor_producers': successor_pins,
    }
    return module._project_native_startup, provenance


def begin_inspection():
    """Claim this inspection before any receipt body is opened; failures stay claimed."""
    assert not (OUT/'full-trial-metadata.json').exists(), 'completed inspection already exists'
    marker = {'schema': 'bounded-full-trial-inspection-start@1',
              'inspector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'receipt_path': INPUT.relative_to(OUT).as_posix(),
              'automatic_retry_authorized': False}
    with (OUT/'full-trial-inspection-start.json').open('x') as stream:
        stream.write(json.dumps(marker, indent=2, sort_keys=True)+'\n')


def main():
    begin_inspection()
    assert INPUT.is_file() and not INPUT.is_symlink() and INPUT.stat().st_size <= MAX_BYTES
    with INPUT.open('rb') as stream:
        raw = stream.read(MAX_BYTES+1)
    assert len(raw) <= MAX_BYTES
    receipt = json.loads(raw)
    project_startup, provenance = load_startup_projector()
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    receipt_bytes = len(raw)
    del raw
    result = dict(schema='bounded-full-trial-metadata-inspection@1', receipt_sha256=receipt_sha256, receipt_bytes=receipt_bytes,
        inspection_provenance=provenance,
        receipt=pick(receipt,('schema','arm','task','trial_count','parallel_workers','harbor_returncode',
            'invocation_seconds','original_task_inputs_unchanged','planning_strategy','benchmark_advantage_claimed')),
        exported_verifier_bodies=False,exported_model_bodies=False,exported_credentials=False,trials=[])
    assert type(receipt.get('trials')) is list and len(receipt['trials']) <= 3
    for trial in receipt['trials']:
        supervisor = trial.get('supervisor',{})
        row = dict(trial=scalar(trial.get('trial')),reward=numbers(trial.get('reward')),
            durations_seconds=numbers(trial.get('durations_seconds')),
            supervisor=pick(supervisor,('schema','arm','task_completed','production_activation',
                'benchmark_advantage_claimed','max_total_agent_seconds','reserved_cleanup_seconds',
                'work_cutoff_seconds','seconds','error_phase','error_type','worker_cleanup_returncode')),
            phases=numbers(supervisor.get('phases')),
            provider_invocation_count=len(supervisor.get('provider_invocations',[])),
            doctor_invocation_count=(len(supervisor['doctor_invocations'])
                if type(supervisor.get('doctor_invocations')) is list else None),
            doctor_dispatch_present=type(supervisor.get('doctor_dispatch')) is dict)
        row['supervisor']['error_type']=scalar(supervisor.get('error',{}).get('type'))
        traceback=supervisor.get('error_traceback',{})
        frames=traceback.get('frames',[]) if type(traceback) is dict else []
        if type(frames) is list and len(frames)<=20:
            row['traceback_frames']=[pick(frame,('file','line','function')) for frame in frames]
        row['supervisor'].update(startup_budget_projection(supervisor))
        row['start'] = control_result_projection(supervisor.get('start'))
        row['stop'] = control_result_projection(supervisor.get('stop'))
        row['task_state']=pick(supervisor.get('task_state'),('status','revision'))
        row['remaining_processes']=scalar(supervisor.get('remaining_processes'))
        row['post_publication_context']=pick(supervisor.get('post_publication_context'),('status','reason','budget_seconds','refresh_seconds','embedding_calls','error_type'))
        row['source384_successor_refreshes'] = successor_rows_projection(supervisor.get('post_publication_context'))
        row['published_context_results'] = published_context_rows_projection(supervisor.get('post_publication_context'))
        row['post_publication_proof_index']=pick(supervisor.get('post_publication_proof_index'),('status','reason','proof_authority','completion_authority'))
        row['post_publication_security_advice']=pick(supervisor.get('post_publication_security_advice'),('status','reason','training_steps','provider_calls','download_calls','proof_authority','completion_authority'))
        row['planning']=pick(supervisor.get('planning'),('qualified','goals','tasks','provider_calls','elapsed_seconds','planning_strategy'))
        doctor = supervisor.get('doctor_dispatch',{})
        row['doctor']=pick(doctor,('status','route','provider_calls','task_cid'))
        workflow=doctor.get('contract_workflow',{})
        row['contract']=pick(workflow,('status','whole_program_proved','canonical_source_edits','publication_authority','completion_authority'))
        row['proof']=pick(workflow.get('proof'),('status','whole_program_proved','security_ir_obligations_discharged','provider_calls'))
        row['proof_receipt']=pick(workflow.get('proof',{}).get('proof'),('disposition','receipt_id','property_id','write_authority','uniqueness_satisfied'))
        initial=supervisor.get('initial_context',{})
        row['initial_context']=pick(initial,('seconds','indexed_symbols','full_capsules'))
        row['initial_context_phases']=numbers(initial.get('nonoverlapping_seconds'))
        row['source384']=pick(initial.get('source384_context'),('seconds','inference_executed','native_worker_executed'))
        row['failure_admission']=pick(supervisor.get('failure_admission'),('status','reason','error_type','observation_boundary','causal_proof'))
        row['failure_resources']=numbers(supervisor.get('failure_resources'))
        # These are already closed, bounded metadata emitted by the production
        # failure projector. Keep its nested primary-gate/pressure evidence apart
        # from later resource values, without reading raw exception bodies.
        admission=supervisor.get('failure_admission')
        if (type(admission) is dict
                and admission.get('schema')=='terminal-admission-failure-observation@1'
                and len(json.dumps(admission).encode('utf-8')) <= 4096):
            row['failure_admission_observation']=admission
        checker = supervisor.get('failure_header_checker')
        if (type(checker) is dict and checker.get('schema') in {
                'bounded-header-checker-failure@1', 'header-model-check-refusal@1'}
                and len(json.dumps(checker).encode('utf-8')) <= 4096):
            row['failure_header_checker'] = checker
        startup = supervisor.get('native_startup')
        if (type(startup) is dict and startup.get('schema') == 'admitted-native-startup-observation@1'
                and len(json.dumps(startup).encode('utf-8')) <= 8192):
            # Reuse the production closed-schema projector, never export raw logs.
            row['native_startup'] = project_startup(startup)
        result['trials'].append(row)
    with (OUT/'full-trial-metadata.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
