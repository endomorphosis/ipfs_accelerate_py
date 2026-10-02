"""Explicit bounded administrative task-population selection.

The independently signed universe and groundings precede fresh finite checks.
Rich conditional/alternative syntax is retained; no saved Boolean supplies proof
or execution authority. Existing nonempty graph/admission contracts are untouched.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import time
import uuid

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.intent_ir.formalize import rich_grammar
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ..proof.formal_verification_contracts import content_identity
from ..prompt.prompt_workflow import PromptGoalGraph
from ..task_sources.intent_repository import IntentRepository
from . import local_planning_admission as local
from . import repository_behavioral_admission as evidence_owner
from . import repository_finite_handoff as finite
from . import repository_successor_context as successor

SCHEMA='reviewed-intent-task-population@1'
DECISION_SCHEMA='reviewed-intent-task-population-decision@1'
SCOPE='independently_grounded_administrative_task_coverage_with_finite_source_observations'
FALSE=dict(proof_authority=False,execution_authority=False,completion_authority=False,
    source_equivalence_proved=False,free_text_semantics_verified=False,model_inference_used=False)


def _require(value,message):
    if not value:raise ValueError(message)


def _closed(value,fields,name):
    _require(type(value) is dict and set(value)==set(fields),'exact '+name+' fields required')
    return value


def _strings(value,name):
    _require(type(value) is list and len(value)<=16 and all(type(v) is str and 0<len(v)<=256 for v in value)
        and len(set(value))==len(value),'bounded unique '+name+' required')
    return sorted(value)


def _pins():
    names={__name__,rich_grammar.__name__,IntegerOffsetContract.__module__,IntentRepository.__module__,successor.__name__}
    return {**evidence_owner._pins(),**{name:hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in sorted(names)}}


def _shape(graph):
    _require(type(graph) is PromptGoalGraph and len(graph.goals)==2 and 1<=len(graph.tasks)<=16
        and not graph.evidence and not graph.unresolved_questions and not graph.uncertainty_debt,
        'two-goal bounded independently declared universe required')
    root=graph.root_goal
    child=next(g for g in graph.goals if g.goal_cid!=root.goal_cid)
    _require(child.parent_goal_cid==root.goal_cid and not root.dependency_goal_cids and not child.dependency_goal_cids
        and all(t.goal_cid==child.goal_cid for t in graph.tasks),'one native execution subgoal required')
    return root,child


def _atoms(ast):
    if ast['kind']=='atom':return {'root':ast}
    if ast['kind'] in ('and','or'):return {'left':ast['left'],'right':ast['right']}
    if ast['kind']=='if':return {'body':ast['body']}
    raise ValueError('temporal ordering is outside this population profile')


def _request(value,allowed_paths):
    if value is None:return None
    _closed(value,{'contract','inputs'},'finite evidence request')
    contract=IntegerOffsetContract.from_dict(value['contract']);inputs=value['inputs']
    _require(contract.path in allowed_paths,'finite property path is outside independently declared scope')
    _require(type(inputs) is list and 1<=len(inputs)<=32 and all(type(v) is int for v in inputs)
        and inputs==sorted(set(inputs)),'complete sorted finite integer input domain required')
    return dict(contract=contract.to_dict(),inputs=inputs)


def _declarations(base_admission,instruction,groundings,guard_grounding,allowed_task_sets,review_ref,tool_policy):
    verified=local.verify_local_benchmark_admission(base_admission,initial=True)
    graph=verified['graph'];_shape(graph)
    _require(type(instruction) is str and 0<len(instruction.encode())<=16384
        and graph.root_goal.objective==instruction,'exact independently signed original instruction required')
    ast=rich_grammar.parse_instruction(instruction);atoms=_atoms(ast)
    _require(all(a['actor']=='agent' for a in atoms.values()),
        'this explicit profile binds only the declared agent actor to the native authorization owner')
    _require(all(a['modality'] in ('required','intended','prohibited') for a in atoms.values()),
        'permission/recommendation scope is outside this explicit population profile')
    _require(ast['kind']!='or' or all(a['modality'] in ('required','intended') for a in atoms.values()),
        'alternative branches require two mandatory norms')
    _require(type(review_ref) is str and 0<len(review_ref)<=512,'explicit independent grounding review reference required')
    tasks={t.task_key:t for t in graph.tasks};paths={p for t in graph.tasks for p in t.scope_paths}
    _require(type(groundings) is list and len(groundings)==len(atoms),'complete grounded atom population required')
    rows={}
    for row in groundings:
        _closed(row,{'atom_path','atom','task_keys','proof_request','forbidden_outputs'},'atom grounding')
        key=row['atom_path'];_require(key in atoms and key not in rows and row['atom']==atoms[key],
            'grounding must retain its exact unique source atom')
        keys=_strings(row['task_keys'],'grounded task keys')
        _require(set(keys)<=set(tasks),'grounded task is outside signed universe')
        from ..prompt.intent_plan_coverage import _outputs
        forbidden=_outputs(row['forbidden_outputs'])
        if atoms[key]['modality']=='prohibited':
            _require(forbidden and row['proof_request'] is None,'prohibition needs explicit forbidden future effects, not a proof marker')
            expected=sorted(t.task_key for t in graph.tasks if any((o.path,o.effect)==(f['path'],f['effect'])
                for o in t.outputs for f in forbidden))
            _require(keys==expected,'prohibition grounding must include every affected potential task')
        else:
            _require(keys and not forbidden,'mandatory atom needs explicit task grounding')
        rows[key]=dict(atom_path=key,atom=atoms[key],task_keys=keys,
            proof_request=_request(row['proof_request'],paths),forbidden_outputs=forbidden)
    _require(set(rows)==set(atoms) and {k for r in rows.values() for k in r['task_keys']}==set(tasks),
        'every original requirement and potential task must be grounded')
    if ast['kind']=='if':
        _closed(guard_grounding,{'guard','proof_request'},'conditional guard grounding')
        _require(guard_grounding['guard']==ast['guard'],'guard subject/property/polarity differs from original Intent')
        guard_grounding=dict(guard=ast['guard'],proof_request=_request(guard_grounding['proof_request'],paths))
    else:_require(guard_grounding is None,'unguarded Intent cannot acquire an invented condition')
    _require(type(allowed_task_sets) is list and 1<=len(allowed_task_sets)<=32,'bounded independent selection authorization required')
    selections=[_strings(v,'authorized task set') for v in allowed_task_sets]
    _require(len({tuple(v) for v in selections})==len(selections) and all(set(v)<=set(tasks) for v in selections),
        'authorized populations must be unique subsets of the entire signed universe')
    _require(type(tool_policy) is dict,'explicit native tool policy required')
    profile=local._manifest(base_admission['manifest'],initial=True)[1]
    value=dict(schema=SCHEMA,interpretation_scope=SCOPE,
        actor_binding=dict(source_actor='agent',native_owner_did=profile.identity_did),
        guard_scope='signed_finite_property_at_initial_source_snapshot',
        prohibition_scope='declared_future_task_output_effects',instruction=instruction,
        instruction_sha256=finite._sha(instruction.encode()),ast=ast,
        base_admission_json=finite._wire(local._plain(base_admission)).decode(),
        universe=sorted(tasks),groundings=[rows[k] for k in sorted(rows)],guard_grounding=guard_grounding,
        allowed_task_sets=sorted(selections),review_ref=review_ref,tool_policy=tool_policy,producers=_pins(),**FALSE)
    _require(len(finite._wire(value))<=512*1024,'bounded population policy required')
    return value,verified


def author_population_policy(*,base_admission,instruction,groundings,guard_grounding=None,
        allowed_task_sets,review_ref,tool_policy):
    """Sign the complete reviewed universe and allowed choices before evidence."""
    value,verified=_declarations(base_admission,instruction,groundings,guard_grounding,allowed_task_sets,review_ref,tool_policy)
    return local._signed(value,verified['manifest'])


def _load(policy):
    _require(type(policy) is dict and set(policy)=={'payload','binding'},'signed population policy required')
    raw=policy['payload'];_require(type(raw) is dict and len(finite._wire(raw))<=512*1024,'bounded policy required')
    base=json.loads(raw['base_admission_json'])
    expected,verified=_declarations(base,raw['instruction'],raw['groundings'],raw['guard_grounding'],
        raw['allowed_task_sets'],raw['review_ref'],raw['tool_policy'])
    _require(raw==expected,'closed population policy/source/producers differ')
    profile=local._manifest(base['manifest'],initial=True)[1]
    _require(local._verify_signature(policy,profile)==raw,'independent population signature differs')
    return raw,base,verified


def _coverage(value,selected,observations):
    """Only checked complete query results decide guards or preexisting facts."""
    ast=value['ast'];active=True;unresolved=[]
    guard=None
    if ast['kind']=='if':
        row=observations.get('guard')
        guard=None if row is None else row['truth']
        if guard is None:unresolved.append('guard')
        else:active=(not guard) if ast['guard']['negated'] else guard
    rows=[];positive_tasks=set();violations=[]
    for grounding in value['groundings']:
        key=grounding['atom_path'];atom=grounding['atom'];keys=set(grounding['task_keys'])
        evidence=observations.get(key);known=evidence is not None and evidence['truth'] is True
        selected_here=sorted(keys&set(selected))
        prohibited=atom['modality']=='prohibited'
        if active and prohibited and selected_here:violations.append(key)
        if active and not prohibited:positive_tasks.update(keys)
        covered=(not selected_here) if prohibited else (known or keys<=set(selected))
        rows.append(dict(atom_path=key,atom=atom,active=active if guard is not None or ast['kind']!='if' else None,
            grounded_task_keys=sorted(keys),selected_task_keys=selected_here,evidence=evidence,covered=covered,
            disposition=('unresolved_guard' if unresolved else 'condition_inactive' if not active else
                'prohibition_respected' if prohibited and covered else 'prohibition_violated' if prohibited else
                'existing_finite_property' if known else 'selected_for_execution' if covered else 'not_selected_alternative_or_uncovered')))
    booleans=[r['covered'] for r in rows]
    satisfied=(any(booleans) if ast['kind']=='or' else all(booleans)) if active else True
    orphaned=sorted(set(selected)-positive_tasks)
    accepted=not unresolved and not violations and not orphaned and satisfied
    return dict(schema='reviewed-task-population-coverage@1',accepted=accepted,requirements=rows,
        complete_atom_paths=[r['atom_path'] for r in rows],guard_truth=guard,unresolved=unresolved,
        prohibited_atom_paths=violations,uncovered_selected_tasks=orphaned,
        formula_kind=ast['kind'],original_requirement_population_retained=True,
        task_execution_success_proved=False,**FALSE)


def _preserved_properties(coverage,graph,selected):
    """The closed self-contained property cannot be elided then overwritten.

    Conditional guards deliberately describe the initial source snapshot. They
    are not permanent invariants. Positive required properties used instead of
    their complete task population must survive the selected future writes.
    """
    writes={o.path for t in graph.tasks if t.task_key in selected for o in t.outputs}
    protected=[]
    for row in coverage['requirements']:
        if row['active'] is not True or row['disposition']!='existing_finite_property' \
                or set(row['grounded_task_keys'])<=set(selected):continue
        evidence=row['evidence']
        path=evidence['query']['contract']['path']
        _require(path not in writes,'selected output would invalidate a proof-elided property source')
        protected.append(dict(atom_path=row['atom_path'],path=path,source_head=evidence['source_head'],
            scope='exact_closed_integer_function_source_preserved_by_declared_outputs'))
    return protected


def _project(graph,selected,policy_cid,head):
    root,child=_shape(graph);selected=set(selected)
    tasks=[t for t in graph.tasks if t.task_key in selected];by_cid={t.task_cid:t for t in graph.tasks}
    _require(tasks and all(by_cid[d].task_key in selected for t in tasks for d in t.dependency_task_cids),
        'selected execution population must retain complete task dependencies')
    revision=content_identity(dict(policy=policy_cid,selected=sorted(selected),head=head.to_dict()))
    acceptances={a.criterion_key:a for t in tasks for a in t.acceptance}
    for task in tasks:
        _require(all(acceptances[a.criterion_key]==a for a in task.acceptance),'conflicting selected acceptance declarations')
    scope=tuple(sorted({p for t in tasks for p in t.scope_paths}))
    goals=[]
    for old,parent in ((root,''),(child,None)):
        goals.append(replace(old,goal_key='POPULATION-'+finite._sha((revision+old.goal_cid).encode())[:24],
            parent_goal_cid=parent if parent is not None else goals[0].goal_cid,
            scope_paths=scope,acceptance=tuple(acceptances[k] for k in sorted(acceptances))))
    projected=[];ids={}
    pending=list(tasks)
    while pending:
        ready=[t for t in pending if set(t.dependency_task_cids)<=set(ids)]
        _require(ready,'selected task dependency cycle')
        for task in ready:
            new=replace(task,task_key='POPULATION-'+finite._sha((revision+task.task_cid).encode())[:24],
                goal_cid=goals[1].goal_cid,dependency_task_cids=tuple(ids[d] for d in task.dependency_task_cids))
            projected.append(new);ids[task.task_cid]=new.task_cid;pending.remove(task)
    roots=dict(request_cid=revision,scan_cid=graph.scan_cid,program_root=graph.program_root)
    result=replace(graph,**roots,goals=tuple(goals),tasks=tuple(projected))
    return PromptGoalGraph.from_dict(result.to_dict()),ids


def _choice_identity(policy,head):
    _require(type(head) is CodebaseHead,'exact native population source head required')
    key=dict(schema='reviewed-task-population-choice@1',policy_cid=content_identity(policy),source_head=head.to_dict())
    cid=content_identity(key)
    return dict(key=key,choice_cid=cid,goal_cid=content_identity(dict(kind='population-choice-goal',choice=cid)),
        plan_cid=content_identity(dict(kind='population-choice-plan',choice=cid)))


def _choice_record(owner,choice,verified,selected):
    """Read an inert native choice, never a saved proof Boolean."""
    plan=owner.get_plan(choice['plan_cid']);goal=owner.get_goal(choice['goal_cid'])
    if plan is None:
        _require(goal is None,'partial population choice parent cannot be overwritten')
        return None
    body=plan['body'];_closed(body,{'schema','choice','signed_decision','materialized'},'durable population choice')
    _require(body['schema']=='reviewed-task-population-choice-record@1' and body['choice']==choice,
        'durable population choice identity differs')
    profile=verified['profile']
    value=local._verify_signature(body['signed_decision'],profile)
    _require(value['schema']==DECISION_SCHEMA and value['policy_cid']==choice['key']['policy_cid']
        and value['source_head']==choice['key']['source_head'] and value['selected_task_keys']==selected,
        'population choice is already committed to another task set or source')
    _require(value['decision_cid']==cid_for_structured({k:v for k,v in value.items() if k!='decision_cid'})
        and value['producers']==_pins() and all(value[k] is False for k in FALSE),
        'durable population decision integrity or producer differs')
    def exact(row,expected):
        return row is not None and all(local._plain(row[k])==local._plain(v) for k,v in expected.items())
    expected_goal=dict(goal_cid=choice['goal_cid'],goal_alias='POPULATION-CHOICE-GOAL-'+choice['goal_cid'][-20:],
        title=verified['graph'].root_goal.title,objective_id='',parent_goal_cid='',ordinal=0,status='open',revision=1,
        body=dict(choice_cid=choice['choice_cid'],policy_cid=choice['key']['policy_cid'],**FALSE))
    expected_plan=dict(plan_cid=choice['plan_cid'],goal_cid=choice['goal_cid'],
        plan_alias='POPULATION-CHOICE-'+choice['plan_cid'][-20:],status='proposed',revision=1,body=body)
    _require(exact(goal,expected_goal) and exact(plan,expected_plan)
        and owner.get_plan_head(choice['goal_cid']) is None,'native population choice parent/plan changed')
    return body


def _decision_semantics(value):
    """Compare fresh checks with the retained choice without reusing evidence."""
    result={k:v for k,v in value.items() if k not in ('decision_cid','observations','coverage')}
    def evidence(row):
        return None if row is None else {k:v for k,v in row.items() if k not in ('record_cid','request_key')}
    result['observations']={key:evidence(row) for key,row in value['observations'].items()}
    result['coverage']={**value['coverage'],'requirements':[
        {**row,'evidence':evidence(row['evidence'])} for row in value['coverage']['requirements']]}
    return result


def _publish_decision(repository,signed):
    """Atomically expose or exactly replay the existing public artifact slot.

    A killed writer may leave a complete temporary, never a partial file under
    the public content name. This is separate from the native Intent commit.
    """
    raw=finite._wire(signed);_require(len(raw)<=2_000_000,'bounded public population artifact required')
    current=repository
    for name in ('.runtime','repository-finite-handoffs'):
        current=current/name
        try:current.mkdir(mode=0o755)
        except FileExistsError:pass
        info=current.lstat()
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid==os.geteuid() and not info.st_mode&0o022
            and stat.S_IMODE(info.st_mode)&0o005==0o005,'owner-controlled worker-readable artifact parent required')
    digest=finite._sha(raw);path=current/(digest+'.json')
    directory=evidence_owner._directory(current)
    temporary=current/('.population-'+uuid.uuid4().hex)
    try:
        try:
            observed,info=evidence_owner._read(directory,path.name)
        except FileNotFoundError:
            finite._persist(temporary,raw)
            try:os.link(temporary.name,path.name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
            except FileExistsError:pass
            finally:os.unlink(temporary.name,dir_fd=directory)
            os.fsync(directory)
            observed,info=evidence_owner._read(directory,path.name)
        _require(info.st_uid==os.geteuid() and stat.S_IMODE(info.st_mode)==0o444 and observed==raw,
            'existing public population artifact differs')
    finally:os.close(directory)
    return path,digest


def prepare_population_admission(*,policy,selected_task_keys,catalog,checked_cache,expected_head,intent,state,
        scheduler=None,parent_lease=None,cancel_event=None,timeout_seconds=180):
    """Fresh checks plus one durable choice per signed policy/source/Intent owner."""
    _,_,verified=_load(policy)
    state=Path(state).absolute();repository=Path(verified['manifest']['repository'])
    _require(state.resolve()==state and not state.is_relative_to(repository),'private canonical population state outside source required')
    with successor._state_lock(state):
        return _prepare_population_admission(policy=policy,selected_task_keys=selected_task_keys,
            catalog=catalog,checked_cache=checked_cache,expected_head=expected_head,intent=intent,state=state,
            scheduler=scheduler,parent_lease=parent_lease,cancel_event=cancel_event,timeout_seconds=timeout_seconds)


def _prepare_population_admission(*,policy,selected_task_keys,catalog,checked_cache,expected_head,intent,state,
        scheduler=None,parent_lease=None,cancel_event=None,timeout_seconds=180):
    value,base,verified=_load(policy);graph=verified['graph'];manifest=verified['manifest']
    selected=_strings(selected_task_keys,'selected task keys')
    _require(selected in value['allowed_task_sets'],'selected population was not independently authorized')
    _require(isinstance(intent,IntentRepository) and not intent.uses_bound_connection,'independent native Intent owner required')
    _require(all(intent.get_task(t.task_cid) is None for t in graph.tasks),
        'population profile requires an unmaterialized original potential task universe')
    _require(type(timeout_seconds) in (int,float) and math.isfinite(timeout_seconds) and 0<timeout_seconds<=300,'bounded population deadline required')
    _require(cancel_event is None or callable(getattr(cancel_event,'is_set',None)),'cooperative cancellation required')
    roots=evidence_owner._roots(catalog,checked_cache);repository=Path(manifest['repository'])
    _require(Path(intent.database_path).resolve()!=Path(roots['source_database']['path']),'independent source and Intent owners required')
    choice=_choice_identity(policy,expected_head)
    _choice_record(intent,choice,verified,selected)
    successor._journal(state/'request.json',dict(schema='reviewed-task-population-retry@1',choice=choice,
        selected_task_keys=selected,owner_roots=roots,intent_database=evidence_owner._entry(Path(intent.database_path)),
        producers=_pins()))
    deadline=time.monotonic()+timeout_seconds
    def resources():
        _require(cancel_event is None or not cancel_event.is_set(),'population preparation cancelled')
        left=deadline-time.monotonic();_require(left>0,'population preparation deadline expired')
        return dict(scheduler=scheduler,parent_lease=parent_lease,cancel_event=cancel_event,timeout_seconds=left)
    def fence():
        _require(_load(policy)[0]==value and evidence_owner._roots(catalog,checked_cache)==roots,
            'population source/admission/producer owner changed')
        catalog.index.observe_current(repository,expected_head=expected_head,**resources())
    fence();observations={}
    queries={r['atom_path']:r['proof_request'] for r in value['groundings']}
    if value['guard_grounding'] is not None:queries['guard']=value['guard_grounding']['proof_request']
    for name,query in queries.items():
        if query is None:continue
        controls=resources();remaining=controls.pop('timeout_seconds')
        checked=checked_cache.check_and_store(owner_inputs=dict(index=catalog.index,repository=repository,
            expected_head=expected_head,contract=IntegerOffsetContract.from_dict(query['contract']),
            inputs=query['inputs'],tool_policy=value['tool_policy'],**controls),timeout_seconds=remaining)
        _require(checked['status'] in ('positive','refuted') and checked['fresh_native_observation'],
            'guard/property requires a fresh complete native observation')
        observations[name]=dict(truth=checked['status']=='positive',status=checked['status'],
            query=query,record_cid=checked['record_cid'],request_key=checked['request_key'],
            fresh_native_observation=True,source_head=expected_head.to_dict(),**FALSE)
    coverage=_coverage(value,selected,observations);fence()
    protected=_preserved_properties(coverage,graph,selected) if coverage['accepted'] else []
    policy_cid=content_identity(policy)
    payload=dict(schema=DECISION_SCHEMA,interpretation_scope=SCOPE,policy=policy,policy_cid=policy_cid,
        source_head=expected_head.to_dict(),owner_roots=roots,selected_task_keys=selected,
        omitted_task_keys=sorted(set(value['universe'])-set(selected)),coverage=coverage,
        observations=observations,protected_property_sources=protected,task_population_authorized=coverage['accepted'],
        original_universe_materialized=False,no_worker=not selected,producers=_pins(),**FALSE)
    if not coverage['accepted']:
        return dict(status='unresolved_population',decision=payload,**FALSE)
    if selected:
        next_graph,mapping=_project(graph,selected,policy_cid,expected_head)
        new_manifest=local.author_local_benchmark_manifest(repository=repository,
            profile_dir=Path(manifest['profile_dir']),lifecycle_dir=Path(manifest['lifecycle_dir']),
            task_specs=successor._specs(next_graph),planning_roots={k:getattr(next_graph,k) for k in ('request_cid','scan_cid','program_root')})
        admission=local.admit_local_benchmark_plan(graph=next_graph,manifest=new_manifest)
        payload.update(selected_graph_json=finite._wire(next_graph.to_dict()).decode(),identity_mapping=mapping,
            selected_admission_json=finite._wire(local._plain(admission)).decode())
    else:
        admission=None;payload.update(selected_graph_json=None,identity_mapping={},selected_admission_json=None)
    payload['decision_cid']=cid_for_structured(payload)
    signed=local._signed(payload,manifest);_require(len(finite._wire(signed))<=2_000_000,'bounded signed population decision required')
    fence()
    with intent._connection(write=True) as cx:
        with IntentRepository(bound_connection=cx,owner_id=intent.owner_id,session_id=intent.session_id) as owner:
            _require(all(owner.get_task(t.task_cid) is None for t in graph.tasks),
                'original universe materialized during population selection')
            stored=_choice_record(owner,choice,verified,selected)
            if stored is not None:
                _require(_decision_semantics(stored['signed_decision']['payload'])==_decision_semantics(payload),
                    'fresh population checks or independent admission differ from committed choice')
            # The exact same native transaction owns both complete task
            # materialization and the one-choice record, including zero tasks.
            materialized=(successor._materialize_or_replay_owned(admission,owner) if selected else
                dict(plan_cid=choice['plan_cid'],task_cids=[],no_worker=True,**FALSE))
            if stored is None:
                owner.upsert_goal(goal_cid=choice['goal_cid'],goal_alias='POPULATION-CHOICE-GOAL-'+choice['goal_cid'][-20:],
                    title=graph.root_goal.title,body=dict(choice_cid=choice['choice_cid'],policy_cid=policy_cid,**FALSE),expected_revision=0)
                owner.upsert_plan(plan_cid=choice['plan_cid'],goal_cid=choice['goal_cid'],
                    plan_alias='POPULATION-CHOICE-'+choice['plan_cid'][-20:],status='proposed',set_head=False,
                    body=dict(schema='reviewed-task-population-choice-record@1',choice=choice,
                        signed_decision=signed,materialized=materialized),expected_revision=0)
            else:
                _require(stored['materialized']==local._plain(materialized),'committed population materialization differs')
                signed=stored['signed_decision']
            _choice_record(owner,choice,verified,selected)
            fence()
    path,digest=_publish_decision(repository,signed)
    return dict(status='population_admitted' if selected else 'zero_edit_admitted',decision=signed,
        decision_path=str(path),decision_sha256=digest,admission=admission,materialized=materialized,
        choice=choice,decision_replayed=stored is not None,fresh_observations=observations,
        saved_observations_used_as_proof=False,**FALSE)


__all__=['author_population_policy','prepare_population_admission']
