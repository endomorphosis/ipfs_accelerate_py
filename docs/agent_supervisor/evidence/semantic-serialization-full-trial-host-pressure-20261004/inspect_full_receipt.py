"""Manual metadata-only projection of the parent's completed trial receipt.

No polling or process control. Reads only the named outer receipt; never follows
paths into verifier output, credential stores, model traces or source bodies.
"""
import json
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


assert INPUT.is_file() and not INPUT.is_symlink() and INPUT.stat().st_size <= MAX_BYTES
with INPUT.open('rb') as stream:
    raw = stream.read(MAX_BYTES+1)
assert len(raw) <= MAX_BYTES
receipt = json.loads(raw)
del raw
result = dict(schema='bounded-full-trial-metadata-inspection@1',
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
        doctor_invocation_count=len(supervisor.get('doctor_invocations',[])))
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
    result['trials'].append(row)
with (OUT/'full-trial-metadata.json').open('x') as stream:
    stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
