"""Freeze and observe full/no-index trials using retained metadata only.

No task/verifier/solution contents, transcript, credential, database or provider
is opened. Receipt hashes bind observations, not authenticity or proof. A lock
must be created before launch. Missing/contradictory usage remains unknown.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import native_usage_accounting as native_usage

MAX_JSON = 64 * 1024 * 1024
TOKENS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')
LOCK_SCHEMA = 'numerical-live-supervisor-pair-lock@1'
SUMMARY_SCHEMA = 'numerical-live-supervisor-pair-observation@1'
COMMON = ('task','dataset','archive','archive_sha256','manifest_sha256','provider_profile','model',
          'reasoning_effort','cli_version','resource_profile','agent_timeout_seconds',
          'task_input_sha256','host_source_sha256','intent_action_384','source384',
          'planning_strategy','intent_requirement_contract_cid','intent_requirement_contract_sha256',
          'numerical_formalization_profile')
CONFIG_DIFFS = {'/agents/0/kwargs/arm','/job_name','/jobs_dir'}


def need(condition, reason):
    if not condition: raise ValueError(reason)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def obj(value): return value if type(value) is dict else {}
def count(value): return value if type(value) is int and value >= 0 else None

def identity(info):
    return tuple(getattr(info, key) for key in ('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns'))


def read(path, *, optional=False, binary=False, maximum=MAX_JSON):
    path = Path(path).absolute()
    try: before = path.lstat()
    except FileNotFoundError:
        if optional: return None, dict(path=str(path), availability='missing')
        raise
    need(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
         and not path.is_symlink() and before.st_size <= maximum, 'bounded canonical regular artifact required')
    digest = hashlib.sha256(); pieces = []
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        need(identity(before) == identity(os.fstat(stream.fileno())), 'artifact changed before read')
        length = 0
        while block := stream.read(1024 * 1024):
            length += len(block); need(length <= maximum, 'artifact exceeds bound')
            digest.update(block)
            if not binary: pieces.append(block)
        after = os.fstat(stream.fileno())
    need(identity(before) == identity(after) == identity(path.lstat()), 'artifact changed during read')
    binding = dict(path=str(path), availability='observed', bytes=length, sha256=digest.hexdigest())
    if binary: return None, binding
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate JSON field'); result[key] = value
        return result
    def nonfinite(_): raise ValueError('nonfinite JSON value')
    value = json.loads(b''.join(pieces), object_pairs_hook=unique, parse_constant=nonfinite)
    need(type(value) is dict, 'object artifact required')
    return value, binding


def scalar(value):
    return value is None or type(value) in (bool,int) or type(value) is float and math.isfinite(value) or type(value) is str and len(value.encode()) <= 2048


def fields(value, names):
    value = obj(value)
    return {key:value[key] for key in names if key in value and scalar(value[key])}


def counters(value):
    if value is None: return None
    need(type(value) is dict and len(value) <= 64, 'bounded counter object required')
    need(all(type(k) is str and re.fullmatch(r'[A-Za-z0-9_:@.-]{1,128}', k)
             and count(v) is not None for k,v in value.items()), 'invalid counters')
    return dict(value)


def differences(a,b,path=''):
    if type(a) is dict and type(b) is dict:
        return sum((differences(a.get(k, _MISSING), b.get(k, _MISSING), path+'/'+k)
                    for k in sorted(a.keys() | b.keys())), [])
    if type(a) is list and type(b) is list and len(a)==len(b):
        return sum((differences(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))), [])
    return [] if type(a) is type(b) and a==b else [path]
_MISSING = object()


def implementation_bindings():
    return {key:read(path,binary=True,maximum=4*1024*1024)[1] for key,path in {
        'collector':Path(__file__).resolve(), 'comparison':Path(comparison.__file__).resolve(),
        'controls':Path(controls.__file__).resolve(), 'native_usage':Path(native_usage.__file__).resolve()}.items()}


def prepared(base, arm, *, require_unlaunched=False):
    base=Path(base).absolute(); prep,pbind=read(base/'preparation.json'); config,cbind=read(base/'config.json')
    need(prep.get('prepared') is True and prep.get('arm')==arm, 'preparation arm differs')
    need(cbind['sha256']==prep.get('config_sha256'), 'configuration differs from preparation')
    need(controls.validate_controls(prep.get('comparison_controls')), 'invalid declared controls')
    expected_controls=controls.build_controls(config,task_input_sha256=prep['task_input_sha256'],
        task=prep['task'],model=prep['model'],reasoning_effort=prep['reasoning_effort'],cli_version=prep['cli_version'])
    need(expected_controls==prep['comparison_controls'],'controls do not match exact configuration')
    agents=config.get('agents');need(type(agents)is list and len(agents)==1, 'one native supervisor agent required')
    kwargs=obj(agents[0].get('kwargs'))
    need(kwargs.get('arm')==arm and kwargs.get('runtime_archive')==prep.get('archive'), 'native agent differs from preparation')
    need(config.get('job_name')=='supervisor-'+arm+'-'+prep['task'], 'job task/arm differs')
    need(Path(config['jobs_dir']).is_relative_to(base) and config.get('n_attempts')==1,
         'isolated one-attempt trial directory required')
    need(kwargs.get('numerical_formalization_profile')==prep.get('numerical_formalization_profile'), 'numerical selection differs')
    need(agents[0].get('model_name')==prep.get('model') and kwargs.get('provider_profile')==prep.get('provider_profile'), 'provider/model selection differs')
    if require_unlaunched:
        need(all(not (base/name).exists() for name in ('invocation.json','receipt.json','launch-result.json','launch-start.json','launch.json','execute.stdout')), 'freeze must precede launch')
        need(not Path(config['jobs_dir']).exists(), 'freeze must precede jobs creation')
    return prep,config,dict(directory=str(base),arm=arm,preparation=pbind,configuration=cbind)


def freeze(full, no_index):
    need(0<len(full)<=16 and 0<len(no_index)<=16, 'bounded nonempty arms required')
    declared=[('full',p) for p in full]+[('no-index',p) for p in no_index]
    need(len({str(Path(p).absolute()) for _,p in declared})==len(declared), 'duplicate trial directory')
    prepared_rows=[(*prepared(p,arm,require_unlaunched=True),) for arm,p in declared]
    reference,refconfig,_=prepared_rows[0]
    inventory=[]
    for prep,config,row in prepared_rows:
        need(all(prep.get(k)==reference.get(k) for k in COMMON), 'common preparation differs')
        changed=differences(refconfig,config);need(set(changed)<=CONFIG_DIFFS,'unexpected pair configuration difference')
        if row['arm']=='no-index':need(config['agents'][0]['kwargs'].get('model_revision')=='','no-index retrieval must be disabled')
        row['configuration_difference_paths']=changed
        row['declared_controls_sha256']=prep['comparison_controls']['sha256']
        inventory.append(row)
    archive=Path(reference['archive']);manifest,mb=read(archive/'manifest.json')
    _,ab=read(archive/'runtime.tar.gz',binary=True,maximum=8*1024**3)
    need(ab['sha256']==reference['archive_sha256']==manifest.get('archive_sha256')
         and mb['sha256']==reference['manifest_sha256'],'archive/manifest differs from preparation')
    host_root=Path(comparison.__file__).resolve().parents[3]
    host=obj(reference.get('host_source_sha256'));need(0<len(host)<=4096,'host closure unavailable')
    for name,digest in host.items():
        path=Path(name);need(path.is_relative_to(host_root),'host pin outside imported adapter')
        need(read(path,binary=True,maximum=4*1024*1024)[1]['sha256']==digest,'host adapter differs')
    launch=launch_inputs()
    recipe,_=read(launch['recipe']['path']); environment,_=read(launch['environment']['path'])
    need(recipe['environment_manifest_sha256']==launch['environment']['sha256'],'recipe environment differs')
    need(recipe['cwd']==str(host_root),'recipe host root differs')
    for row in inventory:
        command=recipe['commands_by_provider'][reference['provider_profile']][row['arm']]['execute_argv']
        need(command==[environment['python'],'-B','-m','benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark',
            'execute','--output',row['directory']],'recipe execution command differs')
    return dict(schema=LOCK_SCHEMA,launch_inputs=launch,created_at_utc=datetime.now(timezone.utc).isoformat(),
        attempts=inventory,common={k:reference.get(k) for k in COMMON},runtime_archive=ab,
        runtime_manifest=mb,host_root=str(host_root),implementations=implementation_bindings(),
        frozen_before_launch=True,task_contents_read=False,hidden_verifier_contents_read=False,
        provider_calls=0,scope='full versus no-index; native supervisor in both; no native CLI baseline',
        allowed_config_difference_paths=sorted(CONFIG_DIFFS),authority='local operator freeze; not cryptographic authenticity')


def usage_projection(report):
    rows=report.get('provider_invocations',[]);need(type(rows)is list and len(rows)<=128,'bounded invocation list required')
    unique={};issues=[]
    for row in rows:
        need(type(row)is dict and type(row.get('invocation_id'))is str
             and re.fullmatch('[0-9a-f]{32}',row['invocation_id']),'invocation identity required')
        key=row['invocation_id']
        need(key not in unique or unique[key]==row,'conflicting duplicate invocation')
        unique[key]=row
    projected=[]
    for key,row in unique.items():
        native=obj(row.get('native_rollout_usage'));raw=obj(native.get('usage'))
        complete,scope,cache=native_usage.native_usage_contract(row,native)
        observed={k:count(raw.get(k)) for k in TOKENS}
        valid=all(value is not None for value in observed.values())
        if scope=='native_session_cumulative':
            valid=valid and observed['total_tokens']==observed['input_tokens']+observed['output_tokens'] and observed['cached_input_tokens']<=observed['input_tokens']
        elif scope=='native_final_envelope':
            write=count(raw.get('cache_write_input_tokens'))
            valid=valid and write is not None and observed['total_tokens']==observed['input_tokens']+observed['cached_input_tokens']+write+observed['output_tokens']
        else: valid=False
        if not valid: issues.append(dict(invocation_id=key,reason='native_usage_missing_or_inconsistent'));complete=None
        tokens={k:v if valid and complete is True else None for k,v in observed.items()}
        projected.append(dict(**fields(row,('invocation_id','purpose','phase','provider','model','reasoning_effort','status','error_type','seconds','router_calls','model_prompt_bytes','model_prompt_sha256','native_prompt_bytes','native_prompt_sha256')),
            native_usage_complete_observed=complete,token_count_scope=scope,cache_included_in_input=cache,
            tokens=tokens,reported_native_counts=observed,
            cache_write_input_tokens=count(raw.get('cache_write_input_tokens')),
            source_formalization_context=fields(row.get('source_formalization_context'),('schema','task_id','task_cid','context_bundle_sha256','payload_sha256','payload_bytes','source_semantics_verified','proof_authority')),
            numerical_obligation_retrieval=fields(row.get('numerical_obligation_retrieval'),('schema','descriptor_sha256','session_id','task_id','task_cid','context_bundle_sha256','context_receipt_sha256','source_tree_id','boundary_manifest_sha256','requests_observed'))))
    unreceipted=bool(report.get('unreceipted_provider_attempt'))
    complete=bool(projected) and not unreceipted and all(r['native_usage_complete_observed'] is True for r in projected)
    sums={k:sum(r['tokens'][k] for r in projected) if complete else None for k in TOKENS}
    contracts={r['token_count_scope'] for r in projected};cache={r['cache_included_in_input'] for r in projected}
    return dict(invocations=projected,observed_router_sessions=len(projected),
        router_sessions_scope='unique retained supervisor invocation IDs, not internal model turns',
        unreceipted_provider_attempt=unreceipted,native_usage_complete_observed=complete,
        token_count_scope=next(iter(contracts)) if len(contracts)==1 else None,
        cache_included_in_input=next(iter(cache)) if len(cache)==1 else None,
        tokens=sums,issues=issues,billing_total_verified=False)


def retrieval_projection(value):
    if value is None:return None
    need(type(value)is dict,'retrieval observation object required')
    result=fields(value,('schema','status','reason','observation_only','scope','source_obligations_discharged','provider_calls','provider_tokens','token_estimate','proof_authority','execution_authority','completion_authority','response_byte_scope','total_token_savings_measured'))
    result['counts']=counters(value.get('counts'))
    result['candidate_context_counts']=counters(value.get('candidate_context_counts'))
    sessions=value.get('sessions');need(sessions is None or type(sessions)is list and len(sessions)<=16,'bounded retrieval sessions required')
    result['sessions']=None if sessions is None else [dict(**fields(row,('session_id_sha256','invocation_id_sha256','audit_sha256','descriptor_sha256','context_bundle_sha256','context_receipt_sha256','source_tree_id_sha256','boundary_manifest_sha256','unique_request_hash_count','unique_response_hash_count')),
        counts=counters(row.get('counts')),status_counts=counters(row.get('status_counts'))) for row in sessions]
    return result


def source384_projection(report):
    ctx=obj(obj(report.get('initial_context')).get('source384_context'));summary=obj(ctx.get('summary'));coverage=obj(summary.get('coverage'))
    return dict(context=fields(ctx,('schema','seconds','checkpoint_sha256','inference_sha256','neural_inference_replayed','training_steps','formalization_authority','proof_authority')),
        summary=fields(summary,('schema','mode','source_files','program_source_files','harness_support_files','inference_python_files','excluded_source_files','training_steps','provider_calls','nomination_only','model_promotion_performed','omitted_candidates','formalization_authority','proof_authority')),
        coverage={**fields(coverage,('files','functions','selected_units')),
            **{k:counters(coverage.get(k)) for k in ('candidate_statuses','file_dispositions','unit_dispositions')}})


def project_trial(receipt, trial, binding):
    need(type(trial)is dict and trial.get('task') in (receipt.get('task'),'terminal-bench/'+str(receipt.get('task'))) and trial.get('exact_trial_task_matches') is True,'trial task identity differs')
    row=comparison._trial(receipt,trial,source=binding)
    report=obj(trial.get('supervisor'))
    if report:
        need(report.get('arm')==receipt.get('arm'),'supervisor arm differs')
        selected=obj(report.get('provider_profile'))
        need(all(selected.get(k)==receipt.get(k) for k in ('model','reasoning_effort')),'supervisor provider model differs')
        need(selected.get('id')==receipt.get('provider_profile'),'supervisor provider profile differs')
    result={k:row[k] for k in ('arm','trial','task','model','reasoning_effort','cli_version','official_reward','native_task_completed','outcome','exception_type','supervisor_error_type','durations_seconds','native_total_seconds','supervisor_seconds','supervisor_phases_seconds','context_input_audit','post_publication_context','declared_output_observation','initial_indexed_context_observation','indexed_context_observation')}
    result['usage']=usage_projection(report)
    selected=obj(report.get('provider_profile'))
    for invocation in result['usage']['invocations']:
        need(invocation.get('provider')==selected.get('provider') and invocation.get('model')==receipt.get('model') and invocation.get('reasoning_effort')==receipt.get('reasoning_effort'),'invocation route differs')
    numerical=obj(obj(report.get('initial_context')).get('numerical_formalization')); numerical_summary=obj(numerical.get('summary'))
    result['numerical_coverage']=dict(receipt=fields(numerical,('profile','receipt_sha256','seconds','source_tree_id')),
        summary=fields(numerical_summary,('schema','profile','source_tree_id','files','functions',
            'structurally_complete_functions','partial_or_unsupported_functions','pending_obligation_count',
            'omitted_function_summaries','omitted_obligation_summaries','compound_contract_count',
            'omitted_compound_contract_summaries','assertion_requirement_count','omitted_assertion_requirement_summaries',
            'control_reference_count','omitted_control_reference_summaries','source_unit_count',
            'linked_source_unit_count','unmatched_source_unit_count','omitted_source_unit_link_summaries',
            'proof_authority','completion_authority','execution_authority','source_semantics_verified',
            'whole_program_semantics_verified','source_obligations_discharged','reachability_verified',
            'loop_invariants_verified','termination_verified','control_flow_semantics_verified')),
        pending_obligation_kinds=counters(numerical_summary.get('pending_obligation_kinds')),
        inline_pending_hint_count=len(numerical_summary['pending_obligations']) if type(numerical_summary.get('pending_obligations'))is list else None,
        canonical_summary_bytes=len(json.dumps(numerical_summary,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()) if numerical_summary else None)
    result.update(numerical_retrieval=retrieval_projection(report.get('numerical_obligation_retrieval_observation')),
        source_retrieval=retrieval_projection(report.get('source_view_retrieval_observation')),
        numerical_planning=fields(report.get('numerical_planning_observation'),('schema','profile','arm','status','reason','request_checked','planner_request_sha256','planner_request_bytes','model_prompt_sha256','request_hash_join_verified','planning_invocation_join_verified','planning_invocation_id','planning_execution_receipt_sha256','numerical_summary_count','numerical_summary_sha256','numerical_receipt_sha256','numerical_source_tree_id','exact_summary_match','live_planner_delivery_verified','no_index_absence_verified','proof_authority','source_obligations_discharged')),
        numerical_selection=fields(report.get('numerical_formalization_selection'),('requested_profile','enabled','effective_profile','disabled_reason')),
        source384=source384_projection(report),
        intent_ir=fields(report.get('intent_preplanning'),('status','seconds','before_goal_decomposition','instruction_sha256','advice_sha256','artifact_sha256','execution_authority','completion_authority')),
        lifecycle={key:fields(report.get(key),('status','succeeded','returncode','seconds')) for key in ('start','stop','runtime_close','start_cleanup')},
        cleanup=fields(report,('worker_cleanup_returncode','remaining_processes','coding_dispatch_possible','implementation_route','error_phase')),
        native_result_sha256=trial.get('native_result_sha256'))
    return result


def launch_inputs():
    root=Path(__file__).resolve().parent
    return {name:read(root/file,binary=True,maximum=MAX_JSON)[1] for name,file in (
        ('runner','run_trial.py'),('recipe','launch-recipe-candidates.json'),('environment','environment.json'))}


def validate_launch_start(start, *, lock, lock_binding, declared):
    need(type(lock_binding)is dict,'launch requires exact lock file binding')
    expected=dict(schema='numerical-live-pair-launch@1',arm=declared['arm'],
        pair_lock_sha256=lock_binding['sha256'],recipe_sha256=lock['launch_inputs']['recipe']['sha256'],
        script_sha256=lock['launch_inputs']['runner']['sha256'],
        environment_manifest_sha256=lock['launch_inputs']['environment']['sha256'],
        preparation_sha256=declared['preparation']['sha256'],inherited_environment_values_retained=False)
    need(all(start.get(k)==v for k,v in expected.items()),'launch start binding differs')
    recipe,_=read(lock['launch_inputs']['recipe']['path'])
    need(start.get('cwd')==recipe['cwd'] and start.get('argv')==recipe['commands_by_provider'][lock['common']['provider_profile']][declared['arm']]['execute_argv'],'launch command differs')
    return {**expected,**fields(start,('started_at_utc',))}


def controls_projection(control, expected):
    control=obj(control);declared=control.get('declared')
    valid=controls.validate_controls(declared) and declared==expected
    status=control.get('status') if control.get('status') in ('observed','mismatch','invalid','unavailable') else 'invalid'
    reasons=(None,'legacy_preparation_without_controls','invalid_frozen_controls','configuration_or_task_inputs_invalid','configuration_or_task_inputs_changed')
    reason=control.get('reason') if control.get('reason') in reasons else 'invalid_observation'
    unchanged=control.get('configuration_unchanged') if type(control.get('configuration_unchanged'))is bool else None
    return dict(status=status,reason=reason,declared_matches_frozen=valid,
        declared_sha256=declared['sha256'] if valid else None,configuration_unchanged=unchanged,
        matched_controls=valid and status=='observed' and unchanged is True)


def collect(lock, *, lock_binding=None):
    need(lock.get('schema')==LOCK_SCHEMA and lock.get('frozen_before_launch')is True,'frozen pair lock required')
    need(lock.get('implementations')==implementation_bindings(),'collector implementation changed after freeze')
    need(lock.get('launch_inputs')==launch_inputs(),'frozen launch input changed')
    common=lock['common'];attempts=[];seen=set();all_invocation_ids=set()
    # Rehash runtime bytes and frozen host producers, never task/verifier files.
    for key in ('runtime_manifest','runtime_archive'):
        binding=lock[key];actual=read(binding['path'],binary=True,maximum=8*1024**3)[1]
        need(actual==binding,'frozen runtime input changed')
    for name,digest in common['host_source_sha256'].items():
        need(Path(name).is_relative_to(Path(lock['host_root'])),'host path outside frozen root')
        need(read(name,binary=True,maximum=4*1024*1024)[1]['sha256']==digest,'frozen host changed')
    for declared in lock['attempts']:
        base=Path(declared['directory']);prep,config,current=prepared(base,declared['arm'])
        need(all(current[key]==declared[key] for key in ('arm','directory','preparation','configuration')),'frozen trial input changed')
        receipt,rbind=read(base/'receipt.json',optional=True);launch,lbind=read(base/'launch-result.json',optional=True)
        invocation,ibind=read(base/'invocation.json',optional=True)
        launch_start,sbind=read(base/'launch-start.json',optional=True)
        start_projection=validate_launch_start(launch_start,lock=lock,lock_binding=lock_binding,declared=declared) if launch_start is not None else None
        need(launch_start is not None or receipt is None and launch is None and invocation is None,'executed attempt lacks frozen launch start')
        attempt=dict(directory=str(base),arm=declared['arm'],receipt=rbind,launch_result=lbind,invocation=ibind,launch_start=sbind,launch_start_projection=start_projection,
            launch=fields(launch,('schema','name','exit_code','seconds','preparation_unchanged','receipt_sha256','complete_single_trial_receipt','trial_count','exception_type','error_type')),
            status='awaiting_receipt' if receipt is None else 'observed',trials=[])
        if receipt is not None:
            need(receipt.get('schema')==comparison.SUPERVISOR and receipt.get('arm')==declared['arm'],'receipt schema/arm differs')
            need(all(receipt.get(k)==prep.get(k) for k in ('task','provider_profile','model','reasoning_effort','cli_version','resource_profile','numerical_formalization_profile')),'receipt identity/profile differs')
            control=receipt.get('comparison_controls')
            if type(control)is dict and control.get('status')=='observed':
                need(control.get('declared')==prep['comparison_controls'],'receipt declared controls differ')
            attempt['receipt_status']=fields(receipt,('harbor_returncode','invocation_seconds','complete_single_trial_receipt','trial_count','native_job_result_present','original_task_inputs_unchanged','intent_selection_config_unchanged','numerical_formalization_config_unchanged'))
            attempt['controls']=controls_projection(control,prep['comparison_controls'])
            flags=['original_task_inputs_unchanged','intent_selection_config_unchanged']
            if prep.get('numerical_formalization_profile') is not None: flags.append('numerical_formalization_config_unchanged')
            flags.extend(k for k in receipt if k.endswith('_config_unchanged'))
            attempt['control_integrity']=attempt['controls']['matched_controls'] and all(receipt.get(k) is True for k in flags)
            trials=receipt.get('trials');need(type(trials)is list and len(trials)<=128,'bounded trial list required')
            for trial in trials:
                identity=(declared['arm'],trial.get('trial'));need(identity not in seen,'duplicate retained native trial');seen.add(identity)
                projected=project_trial(receipt,trial,rbind)
                for invocation_row in projected['usage']['invocations']:
                    identity=invocation_row['invocation_id'];need(identity not in all_invocation_ids,'invocation identity reused across registered pair');all_invocation_ids.add(identity)
                attempt['trials'].append(projected)
            if launch and launch.get('receipt_sha256') is not None:need(launch['receipt_sha256']==rbind['sha256'],'launch receipt binding differs')
        if launch is not None:
            need(launch.get('schema')=='numerical-live-pair-launch-result@1' and launch.get('arm')==declared['arm'],'launch result arm/schema differs')
            if launch.get('preparation_unchanged') is not True:attempt['control_integrity']=False
        else:attempt['control_integrity']=False
        attempts.append(attempt)
    arms={}
    for arm in ('full','no-index'):
        selected=[a for a in attempts if a['arm']==arm];rows=[t for a in selected for t in a['trials']]
        identities=[i['invocation_id'] for t in rows for i in t['usage']['invocations']]
        need(len(identities)==len(set(identities)),'invocation identity reused across trials')
        complete=bool(rows) and all(a['status']=='observed' and a['trials'] for a in selected) and all(t['usage']['native_usage_complete_observed'] for t in rows)
        arms[arm]=dict(retained_attempts=len(selected),retained_native_trials=len(rows),
            official_passes=sum(t['official_reward']==1 for t in rows),unknown_rewards=sum(t['official_reward'] is None for t in rows),
            observed_router_sessions=sum(t['usage']['observed_router_sessions'] for t in rows),
            native_usage_complete_observed=complete,
            tokens={k:sum(t['usage']['tokens'][k] for t in rows) if complete else None for k in TOKENS})
    matched_controls=all(a.get('control_integrity') is True for a in attempts)
    delta={k:arms['full']['tokens'][k]-arms['no-index']['tokens'][k]
        if matched_controls and arms['full']['tokens'][k] is not None and arms['no-index']['tokens'][k] is not None else None for k in TOKENS}
    return dict(schema=SUMMARY_SCHEMA,observed_at_utc=datetime.now(timezone.utc).isoformat(),common=common,
        attempts=attempts,arms=arms,matched_controls=matched_controls,full_minus_no_index_tokens=delta,
        complete_native_usage_for_all_registered_attempts=all(a['native_usage_complete_observed'] for a in arms.values()),
        observer_provider_calls=0,model_calls=0,task_contents_read=False,hidden_verifier_contents_read=False,
        raw_model_transcripts_read=False,credential_contents_read=False,
        billing_total_verified=False,causal_advantage_claimed=False,source_proofs_established=False,
        scope='Full versus no-index native supervisors. All registered attempts retained. Native usage excludes unreported auxiliary usage; cache is not added again. Nested stage times are not added to agent time.')


def write(path,value):
    path=Path(path).absolute();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path.read_bytes()),schema=value['schema'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    freeze_parser=sub.add_parser('freeze');freeze_parser.add_argument('--full',type=Path,action='append',required=True);freeze_parser.add_argument('--no-index',type=Path,action='append',required=True);freeze_parser.add_argument('--output',type=Path,required=True)
    collect_parser=sub.add_parser('collect');collect_parser.add_argument('--lock',type=Path,required=True);collect_parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.operation=='freeze':value=freeze(args.full,args.no_index)
    else:
        lock,binding=read(args.lock);value=collect(lock,lock_binding=binding);value['pair_lock']=binding
    write(args.output,value)

if __name__=='__main__':main()
