"""Modeled metadata joins only; original reader/native/descriptor custody is modeled.

The generic parser owns actual graph/body validation in its independent tests.
These fixtures do not reproduce a genuine accepted original read or prove live
read health. They exercise the new Main joins and preserve historical gaps.
"""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as science


@pytest.fixture
def metadata(tmp_path,monkeypatch):
    docs={};pins={}
    def add(name,value):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));path.chmod(0o400)
        # Generated fixtures have real descriptors. Accepted original-reader
        # custody and current-review provenance remain explicitly modeled.
        pin={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        docs[str(path)]=value;pins[name]=pin;return pin
    def keys(value,expected,reason):science.need(type(value)is dict and set(value)==set(expected),reason)
    h=SimpleNamespace(_document=lambda r:docs[r['path']],descriptor=lambda p:next(r for r in pins.values()if r['path']==str(p)),
        _keys=keys,_integer=lambda v,lo=0,hi=2**63-1:type(v)is int and lo<=v<=hi,
        _text=lambda v:type(v)is str and bool(v),_path=lambda p:Path(p),_decode=json.loads)
    monkeypatch.setattr(science,'health_owner',lambda:h)
    ids=['core%d'%i for i in range(64)];tasks=[]
    for i,cid in enumerate(ids):
        tasks.append(dict(task_cid=cid,task_alias='LIR%d'%i,goal_cid='goal%d'%(i%34),
            plan_cid='plan',objective_id='objective',dependencies=ids[:2]if i<51 else['core0']))
    # 51*2+13 =115; self/cycle semantics intentionally belong to parser tests.
    authored=dict(tasks=tasks,goals=[{'goal_cid':'goal%d'%i}for i in range(34)])
    authored_ref=add('authored',authored)
    node=dict(task_cid='child',task_alias='child',goal_cid='goal0',plan_cid='plan',objective_id='objective',
        minimum_revision=1,namespace_binding={'present':False,'value':None},observation_role='root_dependency',
        body_binding={'profile':'literal-prerequisite/v1'},dependencies=[])
    roots=[dict(task_cid='core0',dependency_task_cid='child',kind='depends_on')]
    boundary=dict(schema='legal-ir-native-dependency-boundary/v2',board_namespace='board',plan_root_cid='plan',
        repository_tree_id='tree',authored_population_sha256=authored_ref['sha256'],authored_task_count=64,
        authored_goal_count=34,authored_dependency_count=115,root_dependencies=roots,observed_tasks=[node],completion_authority=False)
    boundary_ref=add('boundary',boundary)
    envelope_ref=add('envelope',dict(schema='legal-ir-native-catalogue-boundary/v1',authored_population_ref=authored_ref,
        dependency_boundary_ref=boundary_ref))
    parser_ref=add('parser',{'modeled_parser_source':True})
    # Actual G execution-source path differs from its same-body D5 projection.
    projected_parser_ref=dict(parser_ref,path=str(tmp_path/'projected/native_catalogue_probe.py'))
    monkeypatch.setattr(science,'catalogue_parser_reference',lambda profile:projected_parser_ref)
    oldcat=dict(population_path=authored_ref['path'],population_sha256=authored_ref['sha256'],database_path='/db',
        board_namespace='board',plan_root_cid='plan',repository_tree_id='tree')
    cat=dict(oldcat,population_path=envelope_ref['path'],population_sha256=envelope_ref['sha256'])
    prior=dict(task_scope=ids,catalogue=dict(oldcat));profile=dict(task_scope=ids,catalogue=dict(cat))
    plan_ref=add('plan',{});request_ref=add('request',{})
    original_ref=add('original',dict(schema='legal-ir-native-executor-health-original-capture/v1',returncode=0,
        received_signals=[],cleanup_errors=[],completed=True,child_reaped=True,original_CLI_exit_observed=True,
        source_READ_closed=True,source_custody_closed=True,source_pins_unchanged=True,original_native_compute_closed=True,
        native_task_or_service_mutations=False,plan=plan_ref,request=request_ref))
    rows=[dict({k:t[k]for k in ('task_cid','task_alias','goal_cid','plan_cid','objective_id')},
               status='completed',revision=7,board_namespace_present=True,board_namespace='board')for t in tasks]
    histories=[dict(schema='ipfs_accelerate_py/agent-supervisor/task-revision-history-projection@1',task_cid=t['task_cid'],
        current_task_revision=7,current_task_status='completed',current_revision_history_matches=True,
        history_has_revisions_above_current=False,revisions=[dict(revision=7,status='completed')])for t in tasks]
    snapshot=dict(schema='legal-ir-atomic-task-dependency-boundary-observation/v2',original_population=authored_ref,
        original_owned_validation_launch={'role':'modeled'},readonly_physical_connection=True,
        transaction='BEGIN TRANSACTION; ROLLBACK',native_claim_status_completion_provider_or_evidence_writes=0,
        database_and_wal_unchanged=True,database_before={'database':{'path':'/db'}},database_after={'database':{'path':'/db'}},
        completion_authority=False,dispatch_authority=False,skipped_is_not_satisfied=True,event_watermark=415,
        current_root_dependency_rows=roots,original_tasks=rows,task_revision_history_headers=histories,
        noncore_dependency_tasks=[dict(task_cid='child',status='blocked',revision=1)],
        dependency_closure=dict(schema='legal-ir-atomic-dependency-closure-observation/v1',all_outgoing_targets_observed=True,
            all_reads_share_transaction_watermark=415,seed_task_count=64,total_task_count=65,maximum_dependency_depth=16,
            maximum_dependency_edges=10000,maximum_observed_depth=1,noncore_task_count=1,completion_authority=False,dispatch_authority=False))
    snapshot_ref=add('snapshot',snapshot);driver_ref=add('driver',{})
    original_review_ref=add('originalreview',dict(schema='legal-ir-original-bounded-closure-read-independent-result-review/v1',
        result='accepted_original_closed_read_and_complete_observed_dependency_graph',snapshot=snapshot_ref,
        original_closeout=original_ref,actual_driver=driver_ref,actual_plan=plan_ref,actual_request=request_ref,
        checks=dict(source_native_process_custody_closed=True,native_released=True,storage_status='released'),
        acceptance_limits=dict(completion_authority=False,dispatch_authority=False)))
    evidence_ref=add('evidence',dict(schema='legal-ir-proposed-boundary-observation-evidence/v1',state='proposed',
        raw_actual_snapshot_ref=snapshot_ref,original_owner_closeout_ref=original_ref,driver_source_ref=driver_ref,
        observer_source_ref=parser_ref,original_launch=snapshot['original_owned_validation_launch'],completion_authority=False,
        dispatch_authority=False,publication_or_adoption_performed=False,status_or_skipped_satisfaction_inferred=False,
        status_observations_come_only_from_raw_snapshot=True))
    review_ref=add('review',dict(schema='legal-ir-proposed-boundary-DATA-independent-result-review/v1',
        result='accepted_as_exact_PROPOSED_observational_DATA_only',population_envelope=envelope_ref,dependency_boundary=boundary_ref,
        observation_evidence=evidence_ref,observer_source=parser_ref,original_snapshot=snapshot_ref,original_owner_closeout=original_ref,
        driver_source=driver_ref,prior_independent_closed_read_review=original_review_ref,
        checks=dict(actual_source_and_artifact_joins_match_original_closed_read=True,current_status_not_pinned_by_certificate=True),
        limits=dict(completion_authority=False,dispatch_authority=False,state='proposed')))
    graph=dict(schema='native-observational-dependency-graph/v1',authored_population_ref=authored_ref,
        dependency_boundary_ref=boundary_ref,population_envelope_ref=envelope_ref,authored_dependency_count=115,
        native_outgoing_dependency_count=116,observed_task_count=1,prerequisite_body_semantics_qualified=False,
        completion_authority=False,dispatch_authority=False,observed_task_completion_established=False,
        observed_tasks=[dict({k:node[k]for k in ('task_cid','task_alias','goal_cid','plan_cid','objective_id','namespace_binding','observation_role')},
            status='blocked',revision=1,body_binding_profile='literal-prerequisite/v1')])
    closing_ref=add('closing',dict(tasks=copy.deepcopy(rows)))
    fresh=dict(event_cursor=415,tasks=copy.deepcopy(rows),population_schema='legal-ir-native-catalogue-boundary/v1',dependency_boundary=graph)
    recognition=dict(to_cursor=271,native_diagnostic_ref=closing_ref)
    return dict(docs=docs,pins=pins,prior=prior,profile=profile,oldconfig={'catalogue':oldcat},config={'catalogue':cat},
        boundary=boundary,snapshot=snapshot,fresh=fresh,recognition=recognition,add=add)


def joined(c):
    return science._catalogue_race_population_boundary(c['pins']['boundary'],c['pins']['review'],c['pins']['evidence'],
        c['prior'],c['profile'],c['oldconfig'],c['config'])


def test_exact_overlay_joins_preserve_authored_population_and_original_credit(metadata):
    c=metadata;before=copy.deepcopy(c['docs']);bundle=joined(c)
    assert bundle['authored_ref']==c['pins']['authored']
    assert c['docs']==before
    assert science._catalogue_race_current_progress(bundle,c['recognition'],c['profile'],c['fresh'])is c['fresh']


@pytest.mark.parametrize('fault',['extra_envelope_field','boundary_v1','source_review_crossed','authored_digest',
    'authority','snapshot_crossed','original_unclosed','unknown_signal','writes_bool_alias','wrong_database','partial_closure',
    'mixed_watermarks','missing_descendant','root_edge_changed'])
def test_unreviewed_or_unclosed_overlay_stops(metadata,fault):
    c=metadata;d=c['docs'];p=c['pins']
    if fault=='extra_envelope_field':d[p['envelope']['path']]['tasks']=[]
    elif fault=='boundary_v1':c['boundary']['schema']='legal-ir-native-dependency-boundary/v1'
    elif fault=='source_review_crossed':d[p['review']['path']]['observer_source']={'path':'elsewhere'}
    elif fault=='authored_digest':p['authored']['sha256']='f'*64
    elif fault=='authority':d[p['evidence']['path']]['completion_authority']=True
    elif fault=='snapshot_crossed':d[p['evidence']['path']]['raw_actual_snapshot_ref']={'path':'elsewhere'}
    elif fault=='original_unclosed':d[p['original']['path']]['child_reaped']=False
    elif fault=='unknown_signal':d[p['original']['path']]['received_signals']=[15]
    elif fault=='writes_bool_alias':c['snapshot']['native_claim_status_completion_provider_or_evidence_writes']=False
    elif fault=='wrong_database':c['snapshot']['database_before']['database']['path']='/elsewhere'
    elif fault=='partial_closure':c['snapshot']['dependency_closure']['all_outgoing_targets_observed']=False
    elif fault=='mixed_watermarks':c['snapshot']['dependency_closure']['all_reads_share_transaction_watermark']=414
    elif fault=='missing_descendant':c['boundary']['observed_tasks']=[]
    else:c['boundary']['root_dependencies']=[]
    with pytest.raises((science.ExecutorHold,KeyError)):joined(c)


def test_current415_is_distinct_from_historical271_without_health_credit(metadata):
    c=metadata;bundle=joined(c)
    assert science._catalogue_race_current_progress(bundle,c['recognition'],c['profile'],c['fresh'])['event_cursor']==415


@pytest.mark.parametrize('fault',['unobserved_future_cursor','regressed_revision','current_status_unobserved','runnable_core',
    'changed_identity','header_current_mismatch','missing_relevant_revision','boundary_ref_crossed',
    'runnable_dependency','dependency_below_minimum','unknown_dependency','semantic_authority','unobserved_quiet_dependency_status'])
def test_current_progress_requires_exact_observed_quiet_rows_and_history(metadata,fault):
    c=metadata;bundle=joined(c);fresh=c['fresh'];snapshot=c['snapshot']
    if fault=='unobserved_future_cursor':fresh['event_cursor']=416
    elif fault=='regressed_revision':fresh['tasks'][0]['revision']=6
    elif fault=='current_status_unobserved':fresh['tasks'][0]['status']='blocked'
    elif fault=='runnable_core':fresh['tasks'][0]['status']='ready'
    elif fault=='changed_identity':snapshot['original_tasks'][0]['goal_cid']='elsewhere'
    elif fault=='header_current_mismatch':snapshot['task_revision_history_headers'][0]['current_task_status']='blocked'
    elif fault=='missing_relevant_revision':snapshot['task_revision_history_headers'][0]['revisions']=[]
    elif fault=='boundary_ref_crossed':fresh['dependency_boundary']['dependency_boundary_ref']={'path':'elsewhere'}
    elif fault=='runnable_dependency':fresh['dependency_boundary']['observed_tasks'][0]['status']='in_progress'
    elif fault=='dependency_below_minimum':fresh['dependency_boundary']['observed_tasks'][0]['revision']=0
    elif fault=='unknown_dependency':fresh['dependency_boundary']['observed_tasks'][0]['task_cid']='elsewhere'
    elif fault=='unobserved_quiet_dependency_status':fresh['dependency_boundary']['observed_tasks'][0]['status']='completed'
    else:fresh['dependency_boundary']['prerequisite_body_semantics_qualified']=True
    with pytest.raises(science.ExecutorHold):science._catalogue_race_current_progress(bundle,c['recognition'],c['profile'],fresh)


def test_unchanged_starting7_preserves_old_missing4_and_past_busy_status(metadata):
    c=metadata;bundle=joined(c);header=c['snapshot']['task_revision_history_headers'][0]
    header['revisions']=[dict(revision=3,status='blocked'),dict(revision=5,status='in_progress'),dict(revision=7,status='completed')]
    assert science._catalogue_race_current_progress(bundle,c['recognition'],c['profile'],c['fresh'])is c['fresh']


def test_legitimate_changed_interval_requires_all_revisions_but_no_invented_old_bodies(metadata):
    c=metadata;bundle=joined(c);row=c['snapshot']['original_tasks'][0];row['revision']=9
    c['fresh']['tasks'][0]['revision']=9;header=c['snapshot']['task_revision_history_headers'][0]
    header['current_task_revision']=9;header['revisions']=[dict(revision=7,status='completed'),dict(revision=8,status='blocked'),dict(revision=9,status='completed')]
    assert science._catalogue_race_current_progress(bundle,c['recognition'],c['profile'],c['fresh'])is c['fresh']


@pytest.fixture
def authored_data_file(tmp_path):
    import hashlib,stat
    path=tmp_path/'historical-authored.json';raw=b'{"schema":"modeled-authored-DATA/v1","tasks":[]}\n'
    path.write_bytes(raw);path.chmod(0o664);initial=path.lstat()
    try:
        yield dict(path=path,raw=raw,ref=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    finally:
        # Only restore the identity-checked owned ordinary fixture's mode.
        # The two alias cases below model metadata and create no OS aliases.
        info=path.lstat()
        assert (info.st_dev,info.st_ino,stat.S_IFMT(info.st_mode))==(initial.st_dev,initial.st_ino,stat.S_IFREG)
        if stat.S_IMODE(info.st_mode)!=0o664:path.chmod(0o664)


def test_real_FD_reader_accepts_exact_authenticated_historical0664_without_chmod(authored_data_file):
    c=authored_data_file;before=c['path'].stat()
    value=science._catalogue_race_authored_population_document(c['ref'])
    after=c['path'].stat()
    assert value=={'schema':'modeled-authored-DATA/v1','tasks':[]}
    assert (before.st_dev,before.st_ino,before.st_mode,before.st_mtime_ns,before.st_ctime_ns)==(
        after.st_dev,after.st_ino,after.st_mode,after.st_mtime_ns,after.st_ctime_ns)
    assert after.st_mode&0o777==0o664 and c['path'].read_bytes()==c['raw']


@pytest.mark.parametrize('fault',['changed_sha','world_write','symlink','hardlink','size_bool',
    'changed_bytes','identity_changed_during_read'])
def test_real_FD_reader_refuses_unauthenticated_or_aliased_or_raced_DATA(authored_data_file,monkeypatch,fault):
    import os
    from types import SimpleNamespace
    c=authored_data_file;pin=dict(c['ref']);path=c['path']
    if fault=='changed_sha':pin['sha256']='f'*64
    elif fault=='world_write':path.chmod(0o666)
    elif fault=='symlink':
        # MODELED alias metadata: the unchanged H._path uses is_symlink,
        # not resolve. The actual owned path remains a single-link regular file.
        original=Path.is_symlink
        def modeled_symlink(candidate):
            return True if candidate==path else original(candidate)
        monkeypatch.setattr(Path,'is_symlink',modeled_symlink)
    elif fault=='hardlink':
        # MODELED alias metadata on this actual owned FD only. No OS hardlink
        # exists for the parent's strict live usage sampler to encounter.
        original=os.fstat;target=path.stat()
        def modeled_hardlink(fd):
            value=original(fd)
            if (value.st_dev,value.st_ino)!=(target.st_dev,target.st_ino):return value
            values={key:getattr(value,key)for key in ('st_dev','st_ino','st_mode','st_uid','st_gid',
                'st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
            values['st_nlink']=2
            return SimpleNamespace(**values)
        monkeypatch.setattr(os,'fstat',modeled_hardlink)
    elif fault=='size_bool':pin['bytes']=True
    elif fault=='changed_bytes':path.write_bytes(c['raw'].replace(b'tasks',b'other'))
    else:
        original=os.fstat;target_inode=path.stat().st_ino;calls=[]
        def changed(fd):
            value=original(fd)
            if value.st_ino!=target_inode:return value
            calls.append(True)
            if len(calls)<2:return value
            return SimpleNamespace(**{key:getattr(value,key)+(1 if key=='st_ino'else 0)
                for key in ('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')})
        monkeypatch.setattr(os,'fstat',changed)
    with pytest.raises(ValueError):science._catalogue_race_authored_population_document(pin)
