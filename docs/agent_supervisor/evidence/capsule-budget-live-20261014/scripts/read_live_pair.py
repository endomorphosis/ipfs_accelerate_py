"""Bounded artifact-only capsule-budget pair reader; never launches providers.

Usage: PYTHONPATH=<frozen checkout> python -B read_live_pair.py --pair-lock PATH --output NEW_PATH
Only registered metadata is read, never transcripts, source payloads or verifiers.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

OLD = Path('/home/barberb/lift_coding/artifacts/repeated-compact-edits-20261013')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

r = module('capsule_budget_metadata_reader', OLD / 'collect_live_readout.py')
shape = module('capsule_budget_native_shape', OLD / 'collect_native_shape.py')
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_request_observation as operations
from ipfs_accelerate_py.cli_runtime import grok_stream_capture, grok_stream_usage

LABELS = ('legacy', 'one-capsule')
BUDGETS = dict(zip(LABELS, ('legacy', 'one-capsule@1')))
DIFFS = {'/agents/0/kwargs/capsule_budget', '/jobs_dir'}
CANDIDATE = ('prepared_pages', 'delivered_pages', 'prepared_bytes', 'delivered_bytes')
DISCOVERY = ('prepared_pages', 'delivered_pages', 'prepared_rows', 'delivered_rows')


def required_sources():
    return sorted({Path(__file__).resolve(), Path(r.__file__).resolve(), Path(shape.__file__).resolve(),
        Path(r.comparison.__file__).resolve(), Path(r.controls.__file__).resolve(),
        Path(__import__(r.native_usage_contract.__module__, fromlist=['']).__file__).resolve(),
        Path(operations.__file__).resolve(), Path(grok_stream_capture.__file__).resolve(),
        Path(grok_stream_usage.__file__).resolve(), Path(__file__).resolve().parent/'observe_initial_context.py'})


def bound(binding):
    value, actual = r.read(binding['path'])
    r.need(actual == binding, 'frozen metadata binding changed')
    return value, actual


def validate_operation_counts(value):
    fields = {'schema','rows','accepted_requests','unparsed_requests','classification_scope',
              'raw_query_fields_exported','admission_authority'}
    r.need(type(value) is dict and set(value) == fields and value['schema'] == operations.COUNTS_SCHEMA,
           'closed operation count schema required')
    r.need(value['classification_scope'] == 'validated_request_shape_before_peer_and_context_admission'
        and value['raw_query_fields_exported'] is False and value['admission_authority'] is False,
        'operation authority differs')
    r.need(type(value['rows']) is list and len(value['rows']) <= len(operations.COMBINATIONS), 'bounded operation rows required')
    seen = set()
    for row in value['rows']:
        r.need(type(row) is dict and set(row) == {'operation','source_state','view','status_counts'}, 'closed operation row required')
        key = tuple(row[k] for k in ('operation','source_state','view'))
        r.need(all(type(k) is str for k in key) and key in operations.COMBINATIONS and key not in seen,
               'unknown or duplicate operation row'); seen.add(key)
        statuses = r.counters(row['status_counts'], r.STATUS_CODES)
        r.need(statuses is not None and statuses and all(v > 0 for v in statuses.values()), 'nonempty operation statuses required')
    r.need(all(r.count(value[k]) is not None for k in ('accepted_requests','unparsed_requests')),
           'nonnegative operation totals required')
    r.need(operations.combine_operation_counts([value]) == value, 'operation totals/order differ')
    return value


def source_use(report, invocations, profile):
    raw = report.get('source_view_retrieval_observation')
    if raw is None:
        return dict(status='unavailable', reason='source_session_observation_missing', counts=None)
    r.need(type(raw) is dict and raw.get('schema') == 'terminal-source-view-observation@1', 'source observation schema differs')
    if raw.get('status') != 'observed':
        r.need(raw.get('status') in {'unavailable','not_selected','not_applicable'}, 'unknown source status')
        return dict(status=raw['status'], reason=r.tag(raw.get('reason')), counts=None,
                    absence_is_zero_usage=False)
    result = r.source_usage(report, invocations, profile)
    for name, keys in (('candidate_context_counts', CANDIDATE), ('discovery_counts', DISCOVERY)):
        selected = []
        for src, dst in zip(raw['sessions'], result['sessions']):
            value = r.counters(src.get(name), keys)
            r.need(value is None or set(value) == set(keys), 'partial source detail counters')
            if value is not None:
                r.need(value['delivered_pages'] <= value['prepared_pages'], 'delivered page count exceeds prepared')
                suffix = 'bytes' if name.startswith('candidate') else 'rows'
                r.need(value['delivered_'+suffix] <= value['prepared_'+suffix], 'delivered detail count exceeds prepared')
            dst[name] = value; selected.append(value)
        expected = {key: sum(item[key] for item in selected) for key in keys} if all(item is not None for item in selected) else None
        r.need(raw.get(name) == expected, 'source detail aggregate differs')
        result[name] = expected
    attributed = []
    for src, dst in zip(raw['sessions'], result['sessions']):
        if 'request_operation_counts' in src:
            value = validate_operation_counts(src['request_operation_counts'])
            r.need(value['accepted_requests'] == dst['counts']['accepted_requests'], 'operation session total differs')
            statuses = {}
            for row in value['rows']:
                for key,n in row['status_counts'].items(): statuses[key] = statuses.get(key,0)+n
            r.need({k:v for k,v in dst['status_counts'].items() if v} == statuses, 'operation statuses differ')
            dst['request_operation_counts'] = value; attributed.append(value)
    if attributed:
        expected = operations.combine_operation_counts(attributed)
        requests = expected['accepted_requests']
        coverage = dict(observed_sessions=len(attributed), historical_sessions=len(raw['sessions'])-len(attributed),
            observed_requests=requests, historical_unattributed_requests=result['counts']['accepted_requests']-requests,
            all_sessions_observed=len(attributed)==len(raw['sessions']))
        r.need(raw.get('request_operation_counts') == expected and raw.get('request_operation_coverage') == coverage,
               'operation aggregate or historical coverage differs')
        result.update(request_operation_counts=expected, request_operation_coverage=coverage)
    else:
        r.need('request_operation_counts' not in raw and 'request_operation_coverage' not in raw,
               'operation aggregate lacks session observations')
        result.update(request_operation_counts=None, request_operation_coverage=None)
    result.update(response_and_nested_payload_bytes_are_additive=False,
        source_byte_scope='Source slices including repeated reads; excludes candidate context and discovery metadata.',
        request_attribution_scope='Requested shape before admission; successful status and delivered counts are separate evidence.',
        ordinary_tool_usage=None)
    return result


def components(call):
    value = call.get('semantic_prompt_components')
    if value is None:
        return dict(status='unavailable', reason='not_recorded')
    if value == dict(schema='supervisor-semantic-prompt-component-census@1', status='unavailable',raw_content_exported=False):
        return dict(status='unavailable',reason='producer_diagnostic_unavailable')
    constants = dict(schema='supervisor-semantic-prompt-component-census@1',
        measurement_scope='canonical-serialized-prompt-utf8-bytes@1',
        partition_rule='member_bytes_plus_outer_punctuation_equals_object_bytes',
        nested_partitions_overlap=True, token_estimate=False, provider_tokens=None,
        model_prompt_wrappers_included=False, provider_calls=0, raw_content_exported=False,
        dynamic_labels_exported=False,source_freshness_verified=False,proof_authority=False,
        omission_authority=False,completion_authority=False,execution_authority=False,
        exact_native_round_trip_verified=True)
    partitions = {'native_object': {'evidence','goal','scope','acceptance','authority','other_fields'},
        'provider_object': {'native_context','translated_semantic','translation_table','instructions','native_suffix','other_fields'},
        'remaining_native_context': {'evidence','goal','scope','acceptance','authority','other_fields'},
        'translated_semantic': {'capsules','admissions','raw_sources','manifest','other_fields'}}
    numeric = {'native_prompt_bytes','native_suffix_bytes','provider_prompt_bytes'}
    hashes = {'native_prompt_sha256','provider_prompt_sha256'}
    evidence = {'native_evidence','remaining_native_evidence'}
    r.need(type(value) is dict and set(value) == set(constants)|set(partitions)|numeric|hashes|evidence,
        'closed semantic component census required')
    r.need(all(type(value[k]) is type(v) and value[k] == v for k,v in constants.items()), 'component authority differs')
    r.need(all(r.count(value[k]) is not None for k in numeric) and all(r.digest(value[k]) for k in hashes), 'invalid component identity')
    r.need(len(json.dumps(value).encode()) <= 32768, 'bounded component census required')
    for key, labels in partitions.items():
        part = value[key]
        r.need(type(part) is dict and set(part) == {'object_bytes','outer_punctuation_bytes','members'}
            and type(part['members']) is dict and set(part['members']) == labels, 'closed component partition required')
        for member in part['members'].values():
            r.need(type(member) is dict and set(member) == {'count','value_bytes','member_bytes'}
                and all(r.count(n) is not None for n in member.values())
                and member['member_bytes'] >= member['value_bytes'], 'invalid component member')
        r.need(r.count(part['object_bytes']) is not None and r.count(part['outer_punctuation_bytes']) is not None
            and sum(row['member_bytes'] for row in part['members'].values())+part['outer_punctuation_bytes']==part['object_bytes'],
            'component byte sum differs')
    for key in evidence:
        part = value[key]
        r.need(type(part) is dict and set(part) == {'array_bytes','outer_punctuation_bytes','groups'}
            and type(part['groups']) is dict and set(part['groups']) == {'semantic_context','source_formalization_context','other_evidence'},
            'closed evidence partition required')
        for row in part['groups'].values():
            r.need(type(row) is dict and set(row) == {'count','row_bytes','summary_utf8_bytes','summary_json_string_bytes',
                'summary_json_string_overhead_bytes','row_envelope_bytes'} and all(r.count(n) is not None for n in row.values()),
                'invalid evidence member')
            r.need(row['summary_json_string_bytes'] == row['summary_utf8_bytes']+row['summary_json_string_overhead_bytes']
                and row['row_bytes'] == row['summary_json_string_bytes']+row['row_envelope_bytes'], 'evidence byte sum differs')
        r.need(r.count(part['array_bytes']) is not None and r.count(part['outer_punctuation_bytes']) is not None
            and sum(row['row_bytes'] for row in part['groups'].values())+part['outer_punctuation_bytes']==part['array_bytes'],
            'evidence array differs')
    native = value['native_object']; provider = value['provider_object']; remaining = value['remaining_native_context']
    r.need(native['object_bytes']+value['native_suffix_bytes']==value['native_prompt_bytes']
        and provider['object_bytes']==value['provider_prompt_bytes']
        and native['members']['evidence']['value_bytes']==value['native_evidence']['array_bytes']
        and remaining['members']['evidence']['value_bytes']==value['remaining_native_evidence']['array_bytes']
        and provider['members']['native_context']['value_bytes']==remaining['object_bytes']
        and provider['members']['translated_semantic']['value_bytes']==value['translated_semantic']['object_bytes'],
        'nested component accounting differs')
    translation = r.obj(call.get('semantic_translation'))
    r.need(value['native_prompt_sha256']==call.get('native_prompt_sha256')
        and value['native_prompt_bytes']==call.get('native_prompt_bytes')
        and value['provider_prompt_sha256']==translation.get('provider_prompt_sha256')
        and value['provider_prompt_bytes']==translation.get('provider_prompt_bytes'), 'component dispatch identity differs')
    return dict(status='observed', measurement=value, census_replayed_here=False,
        scope='Producer census before public instruction and model wrappers; nested partitions overlap. Actual complete router task input is model_prompt_bytes, not this census.')


def project_trial(receipt, raw, receipt_binding, prep, seen):
    r.need(type(raw) is dict and raw.get('exact_trial_task_matches') is True
        and raw.get('task') in (prep['task'],'terminal-bench/'+prep['task']), 'native task binding differs')
    report = r.obj(raw.get('supervisor')); measured = r.usage(report)
    provider = r.obj(report.get('provider_profile'))
    if report:
        r.need(provider.get('id') == prep['provider_profile'] and provider.get('model')==prep['model']
            and provider.get('reasoning_effort')==prep['reasoning_effort'] and report.get('arm')=='full', 'runtime route differs')
    for call in measured['invocations']:
        r.need(call['invocation_id'] not in seen and call['model']==prep['model']
            and call['provider']==provider.get('provider'), 'duplicate invocation or model/provider differs')
        seen.add(call['invocation_id'])
    native_shape = shape.project(report, [row['invocation_id'] for row in measured['invocations']])
    for call, dst in zip(report.get('provider_invocations',[]), measured['invocations']):
        dst['input_observation'] = r.comparison._input_observation(call)
        dst['semantic_prompt_components'] = components(call)
        dst['semantic_input_sha256'] = r.digest(r.obj(call.get('semantic_translation')).get('semantic_sha256'))
    # Unknown completeness is separate from observed execution failure.
    measured['failed_invocation_count'] = sum(row['error_type'] is not None or row['status'] in {'failed','error','timeout'} for row in measured['invocations'])
    measured['accounting_not_complete_invocation_count'] = sum(row['native_usage_complete_observed'] is not True for row in measured['invocations'])
    row = r.comparison._trial(receipt,raw,source=receipt_binding)
    keys = ('trial','official_reward','outcome','exception_type','supervisor_error_type','native_task_completed',
        'durations_seconds','native_total_seconds','supervisor_seconds','supervisor_phases_seconds','runtime_archive_sha256',
        'initial_indexed_context_observation','indexed_context_observation','doctor_observation','context_input_audit',
        'code_learning_observation','post_publication_context','post_publication_proof_index','declared_output_observation')
    return dict({key:row.get(key) for key in keys},
        runtime_matches_preparation=row.get('runtime_archive_sha256')==prep['archive_sha256'] if row.get('runtime_archive_sha256') else None,
        usage=measured,native_shape=native_shape,source_use=source_use(report,measured['invocations'],prep['source_view_profile']),
        prompt_measurement_scope='Actual complete router task prompt sent to provider; excludes provider harness/system context and subsequent native history.')


def receipt_integrity(receipt, prep):
    selected = r.obj(receipt.get('comparison_controls'))
    flags = {'source_view_config_unchanged','source_edit_config_unchanged'}|{k for k in receipt if k.endswith('_config_unchanged')}
    budget = prep.get('capsule_budget','legacy')
    if budget != 'legacy': flags |= {'capsule_budget_config_unchanged','capsule_budget_runtime_declared_matches'}
    return (selected.get('status')=='observed' and selected.get('declared')==prep['comparison_controls']
        and selected.get('configuration_unchanged') is True and receipt.get('original_task_inputs_unchanged') is True
        and all(receipt.get(k) is True for k in flags) and receipt.get('capsule_budget','legacy')==budget)


def aggregates(attempts):
    results = {}
    for label in LABELS:
        selected = [a for a in attempts if a['label']==label]; trials=[t for a in selected for t in a['trials']]
        complete = (bool(selected) and all(a['status']=='observed' and a['launch_result'].get('availability')!='missing'
            and a['complete_single_trial_receipt'] is True and len(a['trials'])==1 for a in selected)
            and all(t['usage']['native_usage_complete_observed'] is True for t in trials))
        results[label]=dict(registered_attempts=len(selected),retained_trials=len(trials),
            absent_trial_attempts=sum(not a['trials'] for a in selected),official_passes=sum(t['official_reward']==1 for t in trials),
            official_nonpasses=sum(t['official_reward'] is not None and t['official_reward']!=1 for t in trials),
            unknown_rewards=sum(t['official_reward'] is None for t in trials),
            observed_router_sessions=sum(t['usage']['observed_router_sessions'] for t in trials),
            complete_native_usage_for_registered_attempts=complete,
            tokens={key:sum(t['usage']['tokens'][key] for t in trials) if complete and all(t['usage']['tokens'][key] is not None for t in trials) else None for key in r.TOKENS})
    return results



def initial_context_observation(base, label, lock_binding, launch_binding, reporter_pins, trials):
    raw,binding=r.read(base/'initial-context-observation.json',optional=True)
    if raw is None:return dict(status='unavailable',reason='not_recorded',binding=binding,dispatch_joined=False)
    expected={'schema','label','observed_at_utc','container_id','observer_sha256','pair_lock_sha256','launch',
              'raw_content_exported','owner_state_read','provider_calls','observation','dispatch_join_performed'}
    r.need(set(raw)==expected and raw['schema']=='capsule-budget-initial-context-observation@1'
        and raw['label']==label and raw['pair_lock_sha256']==lock_binding['sha256'] and raw['launch']==launch_binding
        and raw['observer_sha256']==reporter_pins[str(Path(__file__).resolve().parent/'observe_initial_context.py')]['sha256']
        and raw['raw_content_exported'] is False and raw['owner_state_read'] is False and type(raw['provider_calls']) is int
        and raw['provider_calls']==0 and raw['dispatch_join_performed'] is False, 'initial context observation custody differs')
    value=raw['observation'];counts=('max_capsules','selected_capsule_count','full_capsule_count','omitted_capsule_count',
        'max_bytes','artifact_bytes','required_raw_paths','source_manifest_paths')
    fields={'status','artifact_sha256','semantic_sha256','capsule_index_cid','source_changed_during_read',
            'semantic_equivalence_claimed','completion_authority',*counts}
    r.need(type(value) is dict and set(value)==fields and value['status']=='observed'
        and all(r.count(value[k]) is not None for k in counts) and r.digest(value['artifact_sha256']) and r.digest(value['semantic_sha256'])
        and r.tag(value['capsule_index_cid']) and all(value[k] is False for k in
            ('source_changed_during_read','semantic_equivalence_claimed','completion_authority')), 'initial context numeric projection differs')
    r.need(value['max_capsules']==(8 if label=='legacy' else 1)
        and value['selected_capsule_count']<=value['max_capsules']
        and value['full_capsule_count']==value['selected_capsule_count']+value['omitted_capsule_count'], 'initial capsule counts differ')
    matched=[call['invocation_id'] for trial in trials for call in trial['usage']['invocations']
        if call['phase']=='coding' and call['semantic_input_sha256']==value['semantic_sha256']]
    return dict(status='observed',binding=binding,observation=value,dispatch_joined=bool(matched),matched_coding_invocation_ids=matched,
        join_scope='Canonical initial semantic artifact hash joined only to recorded coding translation semantic hash; mismatch remains preparation-only.',
        source_or_proof_authority=False)


def collect(lock_path):
    lock, lb = r.read(lock_path)
    r.need(lock.get('schema')=='capsule-budget-live-pair-lock@1' and lock.get('frozen_before_launch') is True
        and set(lock.get('allowed_config_diff_paths',[]))==DIFFS, 'capsule pair prelaunch lock required')
    frozen, frozen_binding = bound(lock['reporting_inputs'])
    r.need(frozen.get('schema')=='capsule-budget-live-reporting-code-freeze@1'
        and type(frozen.get('files')) is list and len(frozen['files'])<=64, 'bounded reporting source freeze required')
    bypath={item['path']:item for item in frozen['files']}
    r.need(len(bypath)==len(frozen['files']) and all(str(p) in bypath for p in required_sources()), 'incomplete reporting code closure')
    for item in bypath.values(): r.need(r.source_binding(item['path'])==item['sha256'], 'frozen reporting code changed')
    for item in lock['launch_inputs'].values(): r.need(r.source_binding(item['path'])==item['sha256'], 'frozen launch input changed')
    pins=lock.get('host_source_sha256'); r.need(type(pins) is dict and 0<len(pins)<=4096,'bounded source closure required')
    root=Path(r.comparison.__file__).resolve().parents[3]
    for path,digest in pins.items():
        r.need(Path(path).is_relative_to(root) and r.digest(digest) and r.source_binding(path)==digest,'host source pin differs')
    manifest, _ = bound(lock['runtime']['manifest'])
    verification, _ = bound(lock['verification']); audit, _ = bound(lock['control_audit'])
    r.need(verification.get('archive')==lock['runtime']['archive'] and verification.get('manifest')==lock['runtime']['manifest']
        and verification.get('all_archive_members_match_manifest') is True
        and verification.get('current_source_capabilities_match') is True, 'archive verification differs')
    r.need(audit.get('complete') is True and set(audit.get('actual_config_diff_paths',[]))==DIFFS,'prelaunch config audit differs')
    registered=lock.get('attempts');r.need(type(registered) is list and len(registered)==2
        and {a.get('label') for a in registered}==set(LABELS), 'exact registered pair required')
    rows=[];seen=set();directories=set();refprep=refconfig=None
    for entry in registered:
        label=entry['label'];base=Path(entry['directory']).absolute()
        r.need(str(base) not in directories and entry.get('arm')=='full' and entry.get('capsule_budget')==BUDGETS[label], 'attempt registration differs');directories.add(str(base))
        prep,pb=bound(entry['preparation']);config,cb=bound(entry['configuration'])
        r.need(base/'preparation.json'==Path(pb['path']) and base/'config.json'==Path(cb['path']), 'attempt directory binding differs')
        r.need(prep.get('prepared') is True and prep.get('arm')=='full' and prep.get('config_sha256')==cb['sha256']
            and prep.get('capsule_budget','legacy')==BUDGETS[label] and prep.get('host_source_sha256')==pins,'prepared capsule inputs differ')
        r.need(prep.get('archive_sha256')==lock['runtime']['archive']['sha256']
            and prep.get('manifest_sha256')==lock['runtime']['manifest']['sha256'],'prepared runtime differs')
        r.need(type(config.get('agents')) is list and len(config['agents'])==1 and config.get('n_attempts')==1,'one-attempt configuration required')
        kwargs=r.obj(config['agents'][0].get('kwargs'))
        r.need(kwargs.get('capsule_budget','legacy')==BUDGETS[label] and kwargs.get('arm')=='full','configured budget differs')
        common=r.obj(lock.get('common'))
        for key in ('task','model','reasoning_effort','cli_version','provider_profile','resource_profile','source_view_profile','source_edit_mode'):
            r.need(prep.get(key)==common.get(key), 'locked common preparation differs: '+key)
        for key in ('source_view_profile','source_edit_mode','provider_profile','resource_profile'):
            r.need(kwargs.get(key)==common.get(key), 'locked configuration selection differs: '+key)
        r.need(r.controls.validate_controls(prep.get('comparison_controls')) and r.controls.build_controls(config,
            task_input_sha256=prep['task_input_sha256'],task=prep['task'],model=prep['model'],reasoning_effort=prep['reasoning_effort'],
            cli_version=prep['cli_version'])==prep['comparison_controls'],'prepared controls differ')
        if refprep is None:refprep,refconfig=prep,config
        changes=r.differences(refconfig,config)
        r.need(set(changes)<=DIFFS,'configuration differs beyond capsule treatment and output')
        r.need(all(prep.get(k)==refprep.get(k) for k in ('task','task_input_sha256','archive_sha256','manifest_sha256','host_source_sha256',
            'provider_profile','model','reasoning_effort','cli_version','resource_profile','source384','intent_action_384','intent_action_384_readiness')),'common preparation differs')
        receipt,rb=r.read(base/'receipt.json',optional=True);launch,lbresult=r.read(base/'launch-result.json',optional=True)
        started,sb=r.read(base/'launch-start.json',optional=True);invocation,ib=r.read(base/'invocation.json',optional=True)
        if any(x is not None for x in (receipt,launch,invocation)):r.need(started is not None,'executed attempt lacks locked launch')
        if started is not None:
            r.need(started.get('schema')=='capsule-budget-live-launch@1' and started.get('label')==label
                and started.get('capsule_budget')==BUDGETS[label] and started.get('pair_lock_sha256')==lb['sha256']
                and started.get('preparation_sha256')==pb['sha256']
                and started.get('runner_sha256')==lock['launch_inputs']['run_locked_trial.py']['sha256']
                and started.get('recipe_sha256')==lock['launch_inputs']['recipe.json']['sha256']
                and started.get('environment_manifest_sha256')==lock['launch_inputs']['live-environment.json']['sha256'],'locked launch differs')
        row=dict(label=label,capsule_budget=BUDGETS[label],directory=str(base),preparation=pb,configuration=cb,receipt=rb,
            launch_start=sb,launch_result=lbresult,invocation=ib,configuration_difference_paths=changes,
            status='awaiting_receipt' if receipt is None else 'observed',trials=[],comparison_control_integrity=False,
            generic_control_comparison=r.controls.compare_controls(
                dict(status='observed',configuration_unchanged=True,declared=refprep['comparison_controls']),
                dict(status='observed',configuration_unchanged=True,declared=prep['comparison_controls'])),
            complete_single_trial_receipt=None,launch_seconds=r.number(r.obj(launch).get('seconds')),
            launch_exit_code=shape.exit_code(r.obj(launch).get('exit_code')))
        if receipt is not None:
            r.need(receipt.get('schema')==r.comparison.SUPERVISOR and receipt.get('arm')=='full'
                and all(receipt.get(k)==prep.get(k) for k in ('task','model','provider_profile','reasoning_effort','cli_version')), 'receipt route differs')
            trial_rows=receipt.get('trials');r.need(type(trial_rows) is list and len(trial_rows)<=16,'bounded trials required')
            row['comparison_control_integrity']=receipt_integrity(receipt,prep)
            row['complete_single_trial_receipt']=shape.boolean(receipt.get('complete_single_trial_receipt'))
            row['capsule_budget_config_unchanged']=shape.boolean(receipt.get('capsule_budget_config_unchanged'))
            row['capsule_budget_runtime_declared_matches']=shape.boolean(receipt.get('capsule_budget_runtime_declared_matches'))
            row['trials']=[project_trial(receipt,raw,rb,prep,seen) for raw in trial_rows]
        if launch is not None:
            r.need(launch.get('schema')=='capsule-budget-live-launch-result@1' and launch.get('label')==label
                and launch.get('capsule_budget')==BUDGETS[label] and launch.get('receipt_sha256')==rb.get('sha256'),'launch result differs')
            row['comparison_control_integrity'] &= all(launch.get(k) is True for k in ('preparation_unchanged','configuration_unchanged','host_source_unchanged'))
        else:row['comparison_control_integrity']=False
        row['initial_context_observation']=initial_context_observation(base,label,lb,sb,bypath,row['trials'])
        rows.append(row)
    r.need(set(rows[-1]['configuration_difference_paths'])==DIFFS,'pair lacks exact capsule and output difference')
    matched=all(a['comparison_control_integrity'] and a['complete_single_trial_receipt'] is True
        and len(a['trials'])==1 and all(t['runtime_matches_preparation'] is True for t in a['trials']) for a in rows)
    totals=aggregates(rows)
    return dict(schema='capsule-budget-live-readout@1',pair_lock=lb,reporting_inputs=frozen_binding,attempts=rows,aggregates=totals,
        frozen_host_source_count=len(pins),frozen_reporting_files=len(bypath),frozen_source_pins_verified=True,
        all_receipt_controls_match_preparation=matched,
        one_capsule_minus_legacy_tokens={k:totals['one-capsule']['tokens'][k]-totals['legacy']['tokens'][k]
            if matched and all(totals[label]['tokens'][k] is not None for label in LABELS) else None for k in r.TOKENS},
        provider_calls_by_reader=0,raw_model_transcripts_read=False,hidden_verifier_or_solution_contents_read=False,
        billing_total_verified=False,causal_advantage_claimed=False,
        limitations=['Only the two explicitly registered attempts are included; missing/failing attempts are retained. No unregistered usage is inferred.',
            'Sequential one-task full/full comparison; capsule cap can affect planning and coding. It is not a randomized general benchmark.',
            'Grok final counters remain reported observations until explicit usage completeness is true. Cache categories are not added twice.',
            'Bounded Grok stream usage events have unverified counter semantics; never sum them or add them to final usage.',
            'Complete prompt means the actual router task input, excluding provider harness/system context and subsequent native history.',
            'Semantic component partitions overlap; response bytes already contain nested source/candidate/discovery payloads.',
            'Provider-free coding does not imply provider-free planning. Missing receipts do not establish zero cost.',
            'Supervisor phase timings overlap native enclosing agent time. Source delivery is not source semantics, proof or official correctness.',
            'This reader checks frozen metadata bindings and runtime digest claims; the separate launch/archive verifier owns execution custody and archive inspection.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair-lock',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=collect(args.pair_lock)
    with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(args.output),'sha256':r.sha(args.output.read_bytes())}))


if __name__=='__main__':main()
