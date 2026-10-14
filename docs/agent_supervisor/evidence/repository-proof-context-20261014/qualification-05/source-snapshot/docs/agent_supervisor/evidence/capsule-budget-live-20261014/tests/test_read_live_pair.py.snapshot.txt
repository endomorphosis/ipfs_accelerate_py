"""Authored offline accounting contracts; no models, network, Docker or benchmark verifier."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capsule_live_reader_test',HERE/'read_live_pair.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=m.r.sha(raw))


def source_binding(path):
    raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=m.r.sha(raw))


def invocation(*,complete=None,identity='a'*32):
    native=dict(schema='native-grok-final-usage@1',usage_available=True,
        totals_scope='native_final_envelope',cache_included_in_input=False,
        task_complete_observed=True,usage=dict(input_tokens=10,cached_input_tokens=20,
        cache_write_input_tokens=3,output_tokens=4,total_tokens=37))
    if complete is not None:native['usage_complete_observed']=complete
    return dict(invocation_id=identity,phase='coding',purpose='coding',provider='grok_cli',model='grok-4.7',status='ok',
        model_prompt_bytes=1000,model_prompt_sha256='b'*64,native_rollout_usage=native)


def test_grok_unknown_completeness_retains_native_counters_without_total():
    result=m.r.usage(dict(provider_invocations=[invocation()]))
    assert result['invocations'][0]['reported_native_counts']['total_tokens']==37
    assert result['invocations'][0]['native_usage_complete_observed'] is None
    assert result['tokens']['total_tokens'] is None


@pytest.mark.parametrize('complete',[True,False])
def test_grok_explicit_completeness_and_cache_categories(complete):
    result=m.r.usage(dict(provider_invocations=[invocation(complete=complete)]))
    assert result['invocations'][0]['native_usage_complete_observed'] is complete
    assert result['tokens']['total_tokens']==(37 if complete else None)
    assert result['cache_categories_added_to_reported_total'] is False


def test_unreceipted_mapping_prevents_complete_all_attempt_total():
    result=m.r.usage(dict(provider_invocations=[invocation(complete=True)],unreceipted_provider_attempt={'type':'timeout'}))
    assert result['tokens']['total_tokens'] is None and result['unreceipted_provider_attempt'] is True


def test_duplicate_invocations_refused():
    with pytest.raises(ValueError,match='duplicate'):
        m.r.usage(dict(provider_invocations=[invocation(),invocation()]))


def test_bounded_grok_events_retained_without_inferred_summation():
    parser=m.grok_stream_capture.GrokStreamParser()
    for i in range(131):
        parser.feed((json.dumps(dict(type='usage',stopReason='tool_use',usage={'input_tokens':i+1}))+'\n').encode(),elapsed_seconds=.5)
    parser.feed(b'{"type":"end","stopReason":"end_turn","usage":{"input_tokens":7}}\n',elapsed_seconds=.6)
    parser.eof('stdout',elapsed_seconds=1);parser.eof('stderr',elapsed_seconds=1);parser.process_exit(0,elapsed_seconds=1)
    call=invocation();call['native_grok_progress']=parser.snapshot(elapsed_seconds=1)
    projected=m.shape.project(dict(provider_invocations=[call]),[call['invocation_id']])
    events=projected['invocations'][0]['native_stream']['usage_observations']
    assert events['retained_events']==128 and events['omitted_events']==3
    assert events['rows'][0]['usage']['input_tokens']==1 and events['rows'][-1]['usage']['input_tokens']==128
    assert events['summation_permitted'] is False and events['ordinals_are_turn_ids'] is False


@pytest.mark.parametrize('status',['unavailable','not_selected','not_applicable'])
def test_source_status_is_preserved_not_zero(status):
    value=m.source_use(dict(source_view_retrieval_observation=dict(schema='terminal-source-view-observation@1',
        status=status,reason='selected_reason')),[],'identity@1')
    assert value['status']==status and value['counts'] is None and value['absence_is_zero_usage'] is False


def source_observation():
    call=invocation();call['source_view_retrieval']=dict(schema='worker-source-dispatch-observation@1',
        profile='identity@1',source_edit_mode='mapped-span@2',session_id='c'*32,descriptor_sha256='d'*64)
    counts=dict.fromkeys(m.r.SOURCE_COUNTS,0);counts.update(sessions=1,accepted_requests=2,response_bytes=200,
        delivered_response_bytes=180,prepared_pages=1,delivered_pages=1,delivered_responses=1,
        request_hash_count=2,response_hash_count=2,unique_request_hash_count=2,unique_response_hash_count=2)
    ops=m.operations.combine_operation_counts([dict(rows=[dict(operation='candidate_context',source_state='working-candidate',
        view='not_applicable',status_counts={'ok':1}),dict(operation='read',source_state='baseline',view='original',status_counts={'stale_context':1})])])
    candidate=dict(prepared_pages=1,delivered_pages=1,prepared_bytes=30,delivered_bytes=30)
    discovery=dict.fromkeys(m.DISCOVERY,0);edits=dict.fromkeys(m.r.EDIT_COUNTS,0)
    session=dict(invocation_id_sha256=m.r.sha(call['invocation_id'].encode()),profile='identity@1',source_edit_mode='mapped-span@2',
        descriptor_sha256='d'*64,session_id_sha256=m.r.sha(('c'*32).encode()),audit_sha256='e'*64,
        counts={k:counts[k] for k in m.r.SESSION_SOURCE_COUNTS},source_edit_counts=edits,status_counts={'ok':1,'stale_context':1},
        candidate_context_counts=candidate,discovery_counts=discovery,request_operation_counts=ops)
    raw=dict(schema='terminal-source-view-observation@1',status='observed',source_edit_mode='mapped-span@2',
        proof_authority=False,execution_authority=False,completion_authority=False,raw_queries_exported=False,raw_sources_exported=False,
        counts=counts,source_edit_counts=edits,sessions=[session],candidate_context_counts=candidate,discovery_counts=discovery,
        request_operation_counts=ops,request_operation_coverage=dict(observed_sessions=1,historical_sessions=0,observed_requests=2,
            historical_unattributed_requests=0,all_sessions_observed=True))
    report=dict(provider_invocations=[call],source_view_retrieval_observation=raw)
    return report,m.r.usage(report)['invocations']


def test_source_candidate_bytes_and_requested_original_refusal_remain_separate():
    report,calls=source_observation();result=m.source_use(report,calls,'identity@1')
    assert result['counts']['response_bytes']==200 and result['candidate_context_counts']['delivered_bytes']==30
    assert result['response_and_nested_payload_bytes_are_additive'] is False
    assert result['request_operation_counts']['rows'][1]['status_counts']=={'stale_context':1}
    assert result['ordinary_tool_usage'] is None


def test_historical_source_operation_absence_stays_unknown():
    report,calls=source_observation();raw=report['source_view_retrieval_observation']
    del raw['request_operation_counts'];del raw['request_operation_coverage'];del raw['sessions'][0]['request_operation_counts']
    result=m.source_use(report,calls,'identity@1')
    assert result['request_operation_counts'] is None and result['request_operation_coverage'] is None


@pytest.mark.parametrize('mutation',['candidate','operation','identity','private'])
def test_source_bad_join_or_aggregate_refused(mutation):
    report,calls=source_observation();raw=report['source_view_retrieval_observation'];raw['candidate_context_counts']=dict(raw['candidate_context_counts'])
    if mutation=='candidate':raw['candidate_context_counts']['delivered_bytes']+=1
    elif mutation=='operation':raw['request_operation_counts']=deepcopy(raw['request_operation_counts']);raw['request_operation_counts']['accepted_requests']+=1
    elif mutation=='identity':raw['sessions'][0]['invocation_id_sha256']='0'*64
    else:raw['sessions'][0]['request_operation_counts']['rows'][0]['private_query']='secret'
    with pytest.raises(ValueError):m.source_use(report,calls,'identity@1')


@pytest.fixture
def pair(tmp_path):
    root=tmp_path.resolve();files=[source_binding(p) for p in m.required_sources()]
    frozen=write(root/'reporting-freeze.json',dict(schema='capsule-budget-live-reporting-code-freeze@1',files=files))
    pin=Path(m.r.comparison.__file__).resolve();pins={str(pin):m.r.source_binding(pin)}
    manifest=write(root/'manifest.json',dict(files=[]));archive=dict(path=str(root/'archive.tar.gz'),bytes=100,sha256='a'*64)
    verification=write(root/'verification.json',dict(archive=archive,manifest=manifest,all_archive_members_match_manifest=True,current_source_capabilities_match=True))
    audit=write(root/'control-audit.json',dict(complete=True,actual_config_diff_paths=sorted(m.DIFFS)))
    inputs={name:write(root/name,{}) for name in ('run_locked_trial.py','recipe.json','live-environment.json')}
    common=dict(task='authored-task',model='grok-4.7',reasoning_effort='high',cli_version='1.0.46',provider_profile='grok-4.7-cli-1.0.46@1',
        resource_profile=None,source_view_profile='identity@1',source_edit_mode='mapped-span@2')
    entries=[]
    for label in m.LABELS:
        base=root/label;kwargs=dict(arm='full',provider_profile=common['provider_profile'],source_view_profile='identity@1',source_edit_mode='mapped-span@2')
        if label!='legacy':kwargs['capsule_budget']=m.BUDGETS[label]
        config=dict(n_attempts=1,jobs_dir=str(base/'jobs'),tasks=[{'path':'authored-task'}],agents=[dict(
            import_path='benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent:FullSupervisorAgent',
            model_name='grok-4.7',kwargs=kwargs)])
        cb=write(base/'config.json',config)
        prep=dict(**common,arm='full',prepared=True,config_sha256=cb['sha256'],host_source_sha256=pins,
            archive_sha256=archive['sha256'],manifest_sha256=manifest['sha256'],task_input_sha256={'instruction.md':'f'*64})
        if label!='legacy':prep['capsule_budget']=m.BUDGETS[label]
        prep['comparison_controls']=m.r.controls.build_controls(config,task_input_sha256=prep['task_input_sha256'],
            **{k:prep[k] for k in ('task','model','reasoning_effort','cli_version')})
        pb=write(base/'preparation.json',prep)
        entries.append(dict(label=label,arm='full',capsule_budget=m.BUDGETS[label],directory=str(base),preparation=pb,configuration=cb))
    lock=dict(schema='capsule-budget-live-pair-lock@1',frozen_before_launch=True,allowed_config_diff_paths=sorted(m.DIFFS),
        reporting_inputs=frozen,launch_inputs=inputs,host_source_sha256=pins,runtime=dict(archive=archive,manifest=manifest),
        verification=verification,control_audit=audit,attempts=entries,common=common)
    lb=write(root/'pair-lock.json',lock)
    def launch(label,*,complete=None,reward=0,budget_match=True,missing_receipt=False):
        entry=next(x for x in entries if x['label']==label);base=Path(entry['directory']);prep,_=m.r.read(entry['preparation']['path'])
        write(base/'launch-start.json',dict(schema='capsule-budget-live-launch@1',label=label,capsule_budget=m.BUDGETS[label],
            pair_lock_sha256=lb['sha256'],preparation_sha256=entry['preparation']['sha256'],
            runner_sha256=inputs['run_locked_trial.py']['sha256'],recipe_sha256=inputs['recipe.json']['sha256'],environment_manifest_sha256=inputs['live-environment.json']['sha256']))
        if missing_receipt:return
        call=invocation(complete=complete,identity=('a' if label=='legacy' else 'b')*32)
        report=dict(arm='full',provider_profile=dict(id=common['provider_profile'],model=common['model'],provider='grok_cli',reasoning_effort='high'),
            provider_invocations=[call],task_completed=bool(reward))
        raw=dict(trial='authored-trial',task='authored-task',exact_trial_task_matches=True,reward={'reward':reward},supervisor=report,
            exception_type=None,agent_context=dict(metadata=dict(runtime_archive_sha256=archive['sha256'])))
        receipt=dict(schema=m.r.comparison.SUPERVISOR,arm='full',**common,trials=[raw],complete_single_trial_receipt=True,
            comparison_controls=dict(status='observed',declared=prep['comparison_controls'],configuration_unchanged=True),
            original_task_inputs_unchanged=True,source_view_config_unchanged=True,source_edit_config_unchanged=True)
        if label!='legacy':receipt.update(capsule_budget=m.BUDGETS[label],capsule_budget_config_unchanged=True,capsule_budget_runtime_declared_matches=budget_match)
        rb=write(base/'receipt.json',receipt)
        write(base/'launch-result.json',dict(schema='capsule-budget-live-launch-result@1',label=label,capsule_budget=m.BUDGETS[label],
            receipt_sha256=rb['sha256'],preparation_unchanged=True,configuration_unchanged=True,host_source_unchanged=True,
            exit_code=0 if reward else 1,seconds=2))
    return root/'pair-lock.json',launch,lock


def test_missing_attempt_is_retained(pair):
    path,launch,_=pair;launch('legacy',complete=True,reward=1);launch('one-capsule',missing_receipt=True)
    result=m.collect(path)
    assert len(result['attempts'])==2 and result['attempts'][1]['status']=='awaiting_receipt'
    assert result['aggregates']['one-capsule']['absent_trial_attempts']==1
    assert result['one_capsule_minus_legacy_tokens']['total_tokens'] is None


def test_failure_and_unknown_usage_not_filtered(pair):
    path,launch,_=pair;launch('legacy',reward=0);launch('one-capsule',reward=1)
    result=m.collect(path)
    assert result['aggregates']['legacy']['official_nonpasses']==1
    assert result['attempts'][0]['trials'][0]['usage']['invocations'][0]['reported_native_counts']['total_tokens']==37
    assert result['all_receipt_controls_match_preparation'] is True
    assert result['one_capsule_minus_legacy_tokens']['total_tokens'] is None


def test_complete_pair_total_does_not_double_count_cache(pair):
    path,launch,_=pair
    for label in m.LABELS:launch(label,complete=True,reward=1)
    result=m.collect(path)
    assert result['aggregates']['legacy']['tokens']['total_tokens']==37
    assert result['one_capsule_minus_legacy_tokens']['total_tokens']==0
    assert result['attempts'][0]['capsule_budget_runtime_declared_matches'] is None


def test_runtime_capsule_mismatch_keeps_cost_but_invalidates_comparison(pair):
    path,launch,_=pair;launch('legacy',complete=True,reward=1);launch('one-capsule',complete=True,reward=1,budget_match=False)
    result=m.collect(path)
    assert result['aggregates']['one-capsule']['tokens']['total_tokens']==37
    assert result['all_receipt_controls_match_preparation'] is False
    assert result['one_capsule_minus_legacy_tokens']['total_tokens'] is None


def test_changed_frozen_config_refused(pair):
    path,_,lock=pair;cfg=Path(lock['attempts'][1]['configuration']['path']);data=json.loads(cfg.read_text());data['n_attempts']=2;write(cfg,data)
    with pytest.raises(ValueError,match='binding changed'):m.collect(path)


def test_private_component_fields_refused():
    with pytest.raises(ValueError):m.components(dict(semantic_prompt_components={'raw_prompt':'secret'}))


def test_empty_invocations_never_become_zero_tokens():
    assert m.r.usage({})['tokens']['total_tokens'] is None


def test_initial_context_observation_matches_only_same_semantic_hash(tmp_path):
    observer=HERE/'observe_initial_context.py';pins={str(observer):source_binding(observer)}
    launch={'path':str(tmp_path/'launch-start.json'),'bytes':1,'sha256':'a'*64};lock={'sha256':'b'*64}
    value=dict(status='observed',artifact_sha256='c'*64,semantic_sha256='d'*64,artifact_bytes=100,
        max_capsules=1,selected_capsule_count=1,full_capsule_count=9,omitted_capsule_count=8,max_bytes=32768,
        required_raw_paths=3,source_manifest_paths=4,capsule_index_cid='capsule:index',source_changed_during_read=False,
        semantic_equivalence_claimed=False,completion_authority=False)
    raw=dict(schema='capsule-budget-initial-context-observation@1',label='one-capsule',observed_at_utc='2026-10-14',container_id='e'*64,
        observer_sha256=pins[str(observer)]['sha256'],pair_lock_sha256=lock['sha256'],launch=launch,raw_content_exported=False,
        owner_state_read=False,provider_calls=0,observation=value,dispatch_join_performed=False)
    write(tmp_path/'initial-context-observation.json',raw)
    call=dict(invocation_id='f'*32,phase='coding',semantic_input_sha256='d'*64)
    trials=[{'usage':{'invocations':[call]}}]
    result=m.initial_context_observation(tmp_path,'one-capsule',lock,launch,pins,trials)
    assert result['dispatch_joined'] is True and result['observation']['selected_capsule_count']==1
    call['semantic_input_sha256']='0'*64
    assert m.initial_context_observation(tmp_path,'one-capsule',lock,launch,pins,trials)['dispatch_joined'] is False


def test_native_producer_census_joins_to_actual_prompt_measurement(tmp_path):
    # Reuse a native producer fixture, not a fabricated JSON prompt-size estimate.
    from test.api.test_semantic_router_translation import native, codec
    root,_,prompt,_=native.__wrapped__(tmp_path)
    encoded=codec.encode_semantic_router_prompt(prompt=prompt,repository=root)
    census=codec.census_semantic_router_prompt(encoded=encoded)
    call=dict(native_prompt_sha256=encoded.receipt['native_prompt_sha256'],native_prompt_bytes=encoded.receipt['native_prompt_bytes'],
        semantic_translation=encoded.receipt,semantic_prompt_components=census,model_prompt_bytes=len(encoded.provider_prompt.encode())+200)
    result=m.components(call)
    assert result['status']=='observed' and result['measurement']['model_prompt_wrappers_included'] is False
    assert result['measurement']['provider_prompt_bytes']<call['model_prompt_bytes']
    call['semantic_translation']=dict(encoded.receipt,provider_prompt_sha256='0'*64)
    with pytest.raises(ValueError,match='dispatch identity'):m.components(call)
