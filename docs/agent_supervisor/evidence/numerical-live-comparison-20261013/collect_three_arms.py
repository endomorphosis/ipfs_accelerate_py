"""Closed, artifact-only native/full/no-index observation; no model or transcript I/O.

The existing pair collector and locks remain unchanged. This observer replays
that collector, verifies its saved observation, then joins a separately locked
native receipt. Intentional experiment differences stay explicit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('frozen_pair_observer',ROOT/'collect_pair.py')
pair=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(pair)
controls=pair.controls
TOKENS=pair.TOKENS
SCHEMA='numerical-three-arm-observation@1'
EXPERIMENT='experimental_controls_sha256.numerical_formalization_profile'
PROFILE='python-numerical-control-structure@3'
HEX=re.compile(r'[0-9a-f]{64}')
read,need,obj,count=pair.read,pair.need,pair.obj,pair.count


def number(v):return v if type(v)in(int,float) and math.isfinite(v) and v>=0 else None

def boolean(v):return v if type(v)is bool else None

def label(v):return v if type(v)is str and re.fullmatch(r'[A-Za-z0-9_.:@/+ -]{1,160}',v) else None

def hashed(v):return pair.sha(v.encode()) if type(v)is str and 0<len(v)<=2048 else None

def binding(v):return {k:v[k] for k in ('availability','bytes','sha256') if k in v}


def native_usage(raw, *, model, effort, cli_version):
    """Validate final cumulative per-session counters without reading rollouts."""
    raw=obj(raw);sessions=raw.get('sessions',[])
    need(type(sessions)is list and len(sessions)<=128,'bounded native session list required')
    rows=[];seen=set();issues=[]
    for session in sessions:
        need(type(session)is dict,'native session object required')
        sid=session.get('session_id');sid_hash=hashed(sid)
        need(sid_hash is not None and sid not in seen,'missing or duplicate native session identity');seen.add(sid)
        values={k:count(obj(session.get('usage')).get(k)) for k in TOKENS}
        coherent=all(v is not None for v in values.values())
        coherent=coherent and values['total_tokens']==values['input_tokens']+values['output_tokens'] and values['cached_input_tokens']<=values['input_tokens']
        route=(session.get('cli_version')==cli_version and session.get('observed_models')==[model]
               and session.get('observed_reasoning_efforts')==[effort])
        complete=(session.get('task_complete_observed') is True and session.get('task_complete_event_observed') is True
                  and type(session.get('malformed_lines'))is int and session['malformed_lines']==0)
        if not coherent:issues.append('missing_or_incoherent_session_counts')
        if not route:issues.append('session_route_not_exact')
        if not complete:issues.append('session_completion_not_established')
        rows.append(dict(session_id_sha256=sid_hash,transcript_sha256=session.get('sha256') if type(session.get('sha256'))is str and HEX.fullmatch(session['sha256']) else None,
            reported_counts=values,coherent_counts=coherent,route_matches=route,completion_observed=complete,
            token_count_events=count(session.get('token_count_events'))))
    reported={k:count(obj(raw.get('usage')).get(k)) for k in TOKENS}
    subtotal={k:sum(v['reported_counts'][k] for v in rows) if rows and all(v['coherent_counts'] for v in rows) else None for k in TOKENS}
    aggregate_matches=bool(rows) and reported==subtotal
    if not aggregate_matches:issues.append('aggregate_missing_or_differs_from_sessions')
    complete=(bool(rows) and raw.get('source')=='native_rollout_observed_cumulative_totals'
              and raw.get('task_complete_observed') is True and aggregate_matches
              and all(v['coherent_counts'] and v['route_matches'] and v['completion_observed'] for v in rows))
    return dict(observed_native_sessions=len(rows),sessions=rows,reported_counts=reported,
        known_coherent_subtotals=subtotal,tokens={k:subtotal[k] if complete else None for k in TOKENS},
        native_usage_complete_observed=complete,cache_included_in_input=True,
        counter_scope='one_final_cumulative_total_per_distinct_native_session',
        issues=sorted(set(issues)),billing_total_verified=False)


def native_trial(receipt,trial):
    need(type(trial)is dict,'native trial object required')
    identity=(trial.get('task') in (receipt.get('task'),'terminal-bench/'+str(receipt.get('task'))) and trial.get('exact_trial_task_matches') is True
              and trial.get('exact_trial_profile_matches') is True)
    reward=number(obj(trial.get('reward')).get('reward'))
    exception=label(trial.get('exception_type'))
    outcome=('official_verifier_passed' if reward==1 else 'official_verifier_not_passed' if reward is not None
             else 'failed_without_verifier_reward' if exception else 'outcome_unknown')
    usage=native_usage(trial.get('raw_usage'),model=receipt.get('model'),effort=receipt.get('reasoning_effort'),cli_version=receipt.get('cli_version'))
    return dict(arm='native-codex',trial=label(trial.get('trial')),task=label(receipt.get('task')),
        identity_matches=identity,official_reward=reward,outcome=outcome,exception_type=exception,
        durations_seconds={k:number(obj(trial.get('seconds')).get(k)) for k in ('environment_setup','agent_setup','agent_execution','verifier')},
        native_total_seconds=number(trial.get('total_seconds')),usage=usage,
        declared_output_observation={'status':'unavailable','reason':'native_collector_does_not_capture_declared_output_hashes'})


def experiment_audit(native_declared,supervisor_declared):
    need(controls.validate_controls(native_declared) and controls.validate_controls(supervisor_declared),'valid declared controls required')
    common_sections=('identity','task_input_sha256','job_controls_sha256','agent_controls_sha256')
    differences=[section+'.'+key for section in common_sections for key in sorted(set(native_declared[section])|set(supervisor_declared[section]))
                 if native_declared[section].get(key)!=supervisor_declared[section].get(key)]
    left=native_declared.get('experimental_controls_sha256',{})
    right=supervisor_declared.get('experimental_controls_sha256',{})
    expected={'numerical_formalization_profile':controls._digest(PROFILE)}
    intended=left=={} and right==expected
    # These wrappers test declarations only, not receipt/runtime integrity.
    production=controls.compare_controls({'status':'observed','configuration_unchanged':True,'declared':native_declared},
                                       {'status':'observed','configuration_unchanged':True,'declared':supervisor_declared})
    return dict(common_controls_equal=not differences,common_control_differences=differences,
        intentional_experimental_difference_valid=intended,
        explicit_three_arm_declaration_matches=not differences and intended,
        allowed_intentional_difference=[EXPERIMENT],
        production_comparator={k:production[k] for k in ('matches','status','differences')},
        production_comparator_unchanged=True,runtime_enforcement_verified=False)


def require_bound(path,digest,*,maximum=pair.MAX_JSON):
    _,observed=read(path,binary=True,maximum=maximum)
    need(observed['sha256']==digest,'frozen native file changed')
    return observed


def validate_native_launch(start, *, command, cwd, lock_sha256, runner_sha256, recipe_sha256, environment_sha256):
    expected=dict(schema='numerical-native-baseline-launch@1',argv=command,cwd=cwd,
        execution_lock_sha256=lock_sha256,script_sha256=runner_sha256,
        recipe_sha256=recipe_sha256,environment_sha256=environment_sha256)
    need(type(start)is dict and all(start.get(k)==v for k,v in expected.items()),'native launch binding differs')


def native_attempt(native_lock):
    lock,lock_binding=read(native_lock)
    need(lock.get('schema')=='numerical-native-baseline-execution-lock@1' and lock.get('frozen_before_launch') is True,'native prelaunch execution lock required')
    root=Path(native_lock).absolute().parent
    need(root==ROOT/'next-native-baseline','native release directory differs')
    release,release_binding=read(root/'release.json');recipe,rb=read(root/'recipe.json');env,eb=read(root/'environment.json');pins,_=read(root/'source-pins.json')
    need(release.get('schema')=='native-codex-separate-baseline-release@1','native release schema differs')
    need(recipe.get('environment_sha256')==eb['sha256'],'native recipe environment differs')
    files=lock.get('files');need(type(files)is dict and 8<=len(files)<=1024,'bounded native lock file inventory required')
    release_names={'prepare_native.py','environment.json','environment-notes.json','recipe.json','host-cli-receipt.json',
        'prepare-result.json','source-pins.json','native-codex-01/preparation.json','native-codex-01/config.json',
        'native-codex-01/native-dry-run.stdout','native-codex-01/native-dry-run.stderr','host-version.stdout','host-version.stderr'}
    need(type(release.get('files'))is dict and set(release['files'])==release_names,'native release file inventory differs')
    expected_files=({str(root/name) for name in release_names|{'release.json','run_native.py','native-preparation-independent-review.json','bin/codex'}}
        |set(pins['files'])|{str(ROOT/'pair-lock.json'),str(ROOT/'pair-observation-final.json')}
        |{str(ROOT/f'live-trials/codex-{arm}-01/launch-result.json') for arm in ('full','no-index')})
    need(set(files)==expected_files,'native lock file inventory differs')
    allowed_roots=(ROOT,Path(recipe['cwd']),Path(env['python']).parents[1]/'lib')
    for name,digest in files.items():
        path=Path(name)
        need(path.is_absolute() and any(path.is_relative_to(parent) for parent in allowed_roots),'lock path outside reviewed roots')
        need(type(digest)is str and HEX.fullmatch(digest),'invalid frozen digest')
        require_bound(path,digest,maximum=512*1024*1024)
    for name in ('release.json','recipe.json','environment.json','source-pins.json','run_native.py','native-codex-01/config.json','native-codex-01/preparation.json'):
        need(str(root/name)in files,'native lock omits release or launch input')
    for name,entry in release['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts,'invalid release member')
        require_bound(root/name,entry['sha256'])
    for name,digest in pins['files'].items():
        need(files.get(name)==digest,'native lock omits source pin');require_bound(name,digest)
    need(files.get(str(root/'bin/codex'))==release['host_cli']['sha256'],'native lock omits pinned host CLI')
    base=root/'native-codex-01';prep,pb=read(base/'preparation.json');config,cb=read(base/'config.json')
    need(prep.get('schema')=='native-codex-harbor-baseline-preparation@1' and prep.get('prepared')is True,'native preparation required')
    need(cb['sha256']==prep.get('config_sha256'),'native prepared config changed')
    declared=controls.build_controls(config,task_input_sha256=prep['task_input_sha256'],task=prep['task'],model=prep['model'],reasoning_effort=prep['reasoning_effort'],cli_version=prep['cli_version'])
    need(declared==prep.get('comparison_controls'),'native declaration differs from config')
    start,sb=read(base/'launch-start.json',optional=True);result,lb=read(base/'launch-result.json',optional=True)
    receipt,receipt_binding=read(base/'receipt.json',optional=True);invocation,ib=read(base/'invocation.json',optional=True)
    expected_command=[env['python'],'-B','-m','benchmarks.agent_supervisor.container_coding.native_codex_baseline','execute','--output',str(base)]
    need(recipe.get('execute_argv')==expected_command,'native execute command differs')
    need(start is not None or all(v is None for v in (result,receipt,invocation)),'native artifacts lack launch start')
    if start is not None:
        validate_native_launch(start,command=expected_command,cwd=recipe['cwd'],lock_sha256=lock_binding['sha256'],runner_sha256=files[str(root/'run_native.py')],recipe_sha256=rb['sha256'],environment_sha256=eb['sha256'])
    if invocation is not None:need(invocation.get('command')==prep.get('command'),'native invocation differs from preparation')
    attempt=dict(arm='native-codex',status='awaiting_receipt' if receipt is None else 'observed',
        bindings={k:binding(v) for k,v in [('receipt',receipt_binding),('launch_start',sb),('launch_result',lb),('invocation',ib),('preparation',pb),('config',cb)]},
        launch_exit_code=result.get('exit_code') if result is not None and type(result.get('exit_code'))is int else None,
        launch_seconds=number(obj(result).get('seconds')),control_integrity=False,trials=[])
    if result is not None:
        need(result.get('schema')=='numerical-native-baseline-launch-result@1','native launch result schema differs')
        need(result.get('receipt_sha256')==(receipt_binding.get('sha256') if receipt is not None else None),'native launch receipt digest differs')
    if receipt is not None:
        need(receipt.get('schema')=='native-codex-harbor-baseline-receipt@1','native receipt schema differs')
        need(all(receipt.get(k)==prep.get(k) for k in ('task','model','reasoning_effort','cli_version')),'native receipt identity differs')
        observed=pair.controls_projection(receipt.get('comparison_controls'),declared)
        trials=receipt.get('trials');need(type(trials)is list and len(trials)<=128 and receipt.get('trial_count')==len(trials),'native trial inventory differs')
        seen=set();session_ids=set()
        for trial in trials:
            identity=trial.get('trial');need(type(identity)is str and identity not in seen,'native trial identity missing or repeated');seen.add(identity)
            projected=native_trial(receipt,trial)
            for session in projected['usage']['sessions']:
                need(session['session_id_sha256'] not in session_ids,'native session reused across trials');session_ids.add(session['session_id_sha256'])
            attempt['trials'].append(projected)
        attempt['control_integrity']=(observed['matched_controls'] and receipt.get('original_task_inputs_unchanged') is True and result is not None and len(trials)==1 and all(t['identity_matches'] for t in attempt['trials']))
        attempt['receipt_complete_single_trial']=boolean(receipt.get('complete_single_trial_receipt'))
        attempt['controls']=observed
        attempt['native_job_usage_corroborated']=boolean(receipt.get('usage_corroborated_by_unscrubbed_native_job'))
    if receipt is None and result is not None:attempt['status']='finished_without_receipt'
    return attempt,declared,binding(lock_binding)


def supervisor_trial(t):
    usage=obj(t.get('usage'));tokens={k:count(obj(usage.get('tokens')).get(k)) for k in TOKENS}
    return dict(arm=t['arm'],trial=label(t.get('trial')),task=label(t.get('task')),official_reward=number(t.get('official_reward')),
        outcome=label(t.get('outcome')),exception_type=label(t.get('exception_type')),supervisor_error_type=label(t.get('supervisor_error_type')),
        durations_seconds={k:number(obj(t.get('durations_seconds')).get(k)) for k in ('environment_setup','agent_setup','agent_execution','verifier')},
        native_total_seconds=number(t.get('native_total_seconds')),supervisor_seconds=number(t.get('supervisor_seconds')),
        usage=dict(tokens=tokens,native_usage_complete_observed=usage.get('native_usage_complete_observed')is True,
            observed_router_sessions=count(usage.get('observed_router_sessions')),cache_included_in_input=boolean(usage.get('cache_included_in_input'))),
        # Keep the detailed, already audited coverage/output observations in the separately bound pair file.
        detail_scope='full coverage, index, output and retrieval details retained in the separately bound pair observation')


def aggregate(attempts,arm):
    selected=[a for a in attempts if a['arm']==arm];rows=[t for a in selected for t in a['trials']]
    complete=bool(rows) and all(a['status']=='observed' and a['trials'] for a in selected) and all(t['usage']['native_usage_complete_observed'] for t in rows)
    return dict(retained_attempts=len(selected),retained_native_trials=len(rows),official_passes=sum(t['official_reward']==1 for t in rows),
        unknown_rewards=sum(t['official_reward']is None for t in rows),native_usage_complete_observed=complete,
        tokens={k:sum(t['usage']['tokens'][k] for t in rows) if complete else None for k in TOKENS})


def collect(pair_observation,pair_lock,native_lock):
    lock,lock_binding=read(pair_lock);saved,saved_binding=read(pair_observation)
    need(saved.get('schema')==pair.SUMMARY_SCHEMA,'frozen pair observation schema required')
    fresh=pair.collect(lock,lock_binding=lock_binding)
    normalized=lambda x:{k:v for k,v in x.items() if k not in ('observed_at_utc','pair_lock')}
    need(normalized(saved)==normalized(fresh),'saved pair observation differs from independent replay')
    need(saved.get('pair_lock')==lock_binding,'saved pair observation lock differs')
    native,declared,nb=native_attempt(native_lock)
    attempts=[];audits=[]
    for a,registered in zip(saved['attempts'],lock['attempts'],strict=True):
        prep,_=read(Path(registered['directory'])/'preparation.json')
        audit=experiment_audit(declared,prep['comparison_controls']);audits.append(dict(arm=a['arm'],**audit))
        attempts.append(dict(arm=a['arm'],status=a['status'],control_integrity=a.get('control_integrity')is True,
            launch_exit_code=obj(a.get('launch')).get('exit_code') if type(obj(a.get('launch')).get('exit_code'))is int else None,
            launch_seconds=number(obj(a.get('launch')).get('seconds')),receipt_binding=binding(a['receipt']),trials=[supervisor_trial(t) for t in a['trials']]))
    attempts.append(native)
    integrity=all(a['control_integrity'] for a in attempts) and all(a['explicit_three_arm_declaration_matches'] for a in audits)
    arms={arm:aggregate(attempts,arm) for arm in ('native-codex','no-index','full')}
    deltas={}
    for arm in ('no-index','full'):
        deltas[arm]={k:arms[arm]['tokens'][k]-arms['native-codex']['tokens'][k] if integrity and arms[arm]['tokens'][k]is not None and arms['native-codex']['tokens'][k]is not None else None for k in TOKENS}
    return dict(schema=SCHEMA,observed_at_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(pair_observation=binding(saved_binding),pair_lock=binding(lock_binding),native_execution_lock=nb),
        implementation_sha256={k:read(p,binary=True)[1]['sha256'] for k,p in [('collector',Path(__file__)),('pair_collector',ROOT/'collect_pair.py'),('controls',Path(controls.__file__))]},
        attempts=attempts,arms=arms,explicit_experiment_control_audit=audits,
        all_registered_attempts_control_integrity=integrity,production_generic_comparator_match=False,
        supervisor_minus_native_tokens=deltas,full_minus_no_index_tokens=fresh['full_minus_no_index_tokens'],
        pair_controls_matched=fresh['matched_controls'],billing_total_verified=False,causal_advantage_claimed=False,
        randomized_trial_order=False,provider_work_budget_equality_claimed=False,
        budget_note='Same 6480-second outer agent cap; supervisor separately reserves preparation and caps planning at 600 and coding at 1800 seconds, while native coding can use its outer agent budget.',
        observer_provider_calls=0,model_calls=0,raw_transcripts_read=False,task_or_hidden_verifier_contents_read=False,
        scope='All registered native/full/no-index attempts. Native session counters and supervisor router invocation counters have distinct call-count scopes. Cached input is included in input. No proof discharge or broad efficiency claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-observation',type=Path,required=True);p.add_argument('--pair-lock',type=Path,required=True)
    p.add_argument('--native-lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pair.write(a.output,collect(a.pair_observation,a.pair_lock,a.native_lock))

if __name__=='__main__':main()
