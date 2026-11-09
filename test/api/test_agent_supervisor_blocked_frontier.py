"""Verified refused work is visible even while read-only gate probes rotate."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume
import ipfs_accelerate_py.agent_supervisor.rescue as rescue_package
from ipfs_accelerate_py.agent_supervisor.rescue import planned_work_watchdog as planned
from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as dispatcher
from ipfs_accelerate_py.agent_supervisor.rescue.gate_recovery import gate_evidence, gate_source_contract
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import parse_todo_task_fields

source_root=Path(lane_resume.__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('_blocked_frontier_rig',source_root/'test/api/test_agent_supervisor_repair_dispatcher.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
rig=fixtures.rig


@pytest.fixture
def frontier(rig):
    config,now,lane,live,probe=rig
    root=config.repo_root
    (root/'producer.py').write_text('import sys\nraise SystemExit(75 if sys.argv[1]=="2" else 0)\n')
    (root/'validator.py').write_text('raise SystemExit(75)\n')
    (root/'research.md').write_text('Actual externally announced UTC release date.\n')
    (root/'data').mkdir()
    dependencies={'VOG-001':[],'VOG-002':['VOG-001'],'VOG-003':['VOG-001'],
                  'VOG-004':['VOG-001'],'VOG-005':['VOG-002','VOG-003','VOG-004']}
    board=''
    for task,depends in dependencies.items():
        status='completed' if task=='VOG-001' else 'planned' if task=='VOG-005' else 'pending'
        board+=f'## {task} Work\n- Status: {status}\n- Depends on: '+', '.join(depends)+'\n'
        if status=='pending':
            number=str(int(task[-3:]));(root/f'data/gate-{number}.json').write_text(json.dumps({'schema':'fixture.gate.v1','reason':'budget_or_capability_unavailable','generated_at':now}))
            board+=f'- Deterministic command: python3 producer.py {number}\n- Validation: python3 validator.py {number}\n- Outputs: data/gate-{number}.json\n'
    (root/'todo.md').write_text(board)
    (root/'goals.json').write_text(json.dumps({'tasks':[{'id':task,'goal_id':'G-1','depends_on':deps,'work_package':{}} for task,deps in dependencies.items()]}))
    manifest=json.loads(config.manifest_path.read_text())
    manifest['planned_work_policy']={'enabled':True,'goals_path':'goals.json','grace_seconds':60,
        'materialize_independent_frontier':True,'deferred_tasks':{'VOG-004':{'kind':'not_before',
        'not_before_utc':'2026-11-01T00:00:00Z','reason':'Actual future external release','evidence_path':'research.md'}}}
    fixtures.write_json(config.manifest_path,manifest)
    fixtures.status(lane,state='degraded',phase='gate_dependency_revalidation',active_task_id='VOG-002',
                    heartbeat_at_ms=int(now*1000),last_task_transition_at_ms=int((now-3000)*1000))
    fields=dict(parse_todo_task_fields(board,task_header_prefix='## VOG-'))
    ledger={'_validation_fixed_points':{}};gates={}
    for task in ['VOG-002','VOG-003','VOG-004']:
        number=str(int(task[-3:]))
        code=subprocess.run([sys.executable,'producer.py',number],cwd=root,timeout=3).returncode
        stage='command'
        if code==0:
            code=subprocess.run([sys.executable,'validator.py',number],cwd=root,timeout=3).returncode;stage='validation'
        assert code==75
        command=lane_resume.parse_deterministic_command(fields[task]['deterministic_command'],repo_root=root)
        validation=lane_resume.parse_deterministic_command(fields[task]['validation'],repo_root=root)
        contract=lane_resume._command_contract(fields[task],command,validation,repo_root=root)
        fingerprint,evidence=gate_evidence(task,fields,repo_root=root,todo_path=root/'todo.md',log_path=lane/'supervisor.log')
        ledger['_validation_fixed_points'][task]={'execution_progress':{'schema':'agent-supervisor/execution-progress@1',
            'contract':contract,'state':'gate_withheld','last_exit_code':75,'last_stage':stage,
            'gate_refusals':1,'finished_at_ms':int(now*1000)}}
        gates[task]={'fingerprint':fingerprint,'source_contract':gate_source_contract(evidence,contract),
                    'unchanged_probes':3,'first_seen_ms':int((now-3600)*1000),'last_seen_ms':int(now*1000),
                    'last_probe':'validation_only','last_validation_exit_code':75}
    fixtures.write_json(lane/'lane_resume_attempts.json',ledger)
    fixtures.write_json(lane/'lane_resume_attempts.gates.json',gates)
    return config,now,lane,live,probe,ledger,gates


def observe(frontier,state,offset=0):
    config,now,lane,live,probe,ledger,gates=frontier
    fixtures.status(lane,heartbeat_at_ms=int((now+offset)*1000))
    return dispatcher.observe(config,state,now=now+offset,probe=probe)


def test_real_source_current_command_and_validation_refusals_are_collected(frontier):
    result=observe(frontier,{})
    plan=result['planned_work']
    assert not plan['dependency_ready_planned_task_ids'] and plan['blocked_task_ids']==['VOG-002','VOG-003']
    assert plan['state']=='blocked_frontier_grace' and not result['incidents']
    assert {p['withheld_stage'] for p in plan['blocked_frontier_proofs']}=={'command','validation'}
    assert plan['deferred_tasks']['VOG-004']['state']=='waiting_external_gate'
    assert 'VOG-004' not in plan['runnable_task_ids'] and not plan['completion_authority']


def test_stopped_lane_without_proofs_keeps_the_frontier_clock(frontier):
    config, now, lane, live, probe, ledger, gates = frontier
    state = {}
    observe(frontier, state)
    original = state["blocked_frontier_observation"]["first_seen"]
    tasks = dispatcher._tasks(config.repo_root / "todo.md", "## VOG-")
    policy = json.loads(config.manifest_path.read_text())["planned_work_policy"]
    result = planned.observe_planned_work(
        policy=policy, repo_root=config.repo_root, tasks=tasks, lanes=[],
        state=state, now=now + 30, gate_blockers=[])
    assert state["blocked_frontier_observation"]["first_seen"] == original
    assert result["state"] != "blocked_frontier_grace"


def test_rotating_readonly_probes_and_json_clocks_cannot_renew_grace(frontier):
    config,now,lane,live,probe,ledger,gates=frontier;state={}
    observe(frontier,state)
    original=state['blocked_frontier_observation']['first_seen']
    for offset,task in [(20,'VOG-003'),(40,'VOG-004'),(60,'VOG-002')]:
        fixtures.status(lane,phase='gate_dependency_revalidation',active_task_id=task)
        for gate in gates.values():gate.update(last_seen_ms=int((now+offset)*1000),unchanged_probes=offset+1)
        fixtures.write_json(lane/'lane_resume_attempts.gates.json',gates)
        for p in (config.repo_root/'data').glob('*.json'):
            value=json.loads(p.read_text());value['generated_at']=now+offset;p.write_text(json.dumps(value))
        state=json.loads(json.dumps(state));result=observe(frontier,state,offset)
        assert state['blocked_frontier_observation']['first_seen']==original
    assert result['incidents'][0]['kind']=='planned_work_blocked_frontier'
    assert result['planned_work']['state']=='plan_replan_required'


@pytest.mark.parametrize('phase',['gate_recovery_needed','gate_retry_wait','gate_withheld'])
def test_gate_recovery_label_keeps_the_blocked_frontier_incident(frontier,phase):
    config,now,lane,live,probe,ledger,gates=frontier
    state={};observe(frontier,state)
    fixtures.status(lane,phase=phase,active_task_id='VOG-002')
    result=observe(frontier,state,80)
    assert 'planned_work_blocked_frontier' in [item['kind'] for item in result['incidents']]
    assert result['planned_work']['state']=='plan_replan_required'
    assert state['blocked_frontier_observation']['first_seen']==now


@pytest.mark.parametrize('phase',['deterministic_command','validation','gate_revalidation'])
def test_real_work_defers_maintenance_without_erasing_blockage_clock(frontier,phase):
    config,now,lane,live,probe,ledger,gates=frontier;state={};observe(frontier,state)
    fixtures.status(lane,phase=phase,active_task_id='VOG-002')
    progress=ledger['_validation_fixed_points']['VOG-002']['execution_progress'];saved=copy.deepcopy(progress)
    progress['state']='running';fixtures.write_json(lane/'lane_resume_attempts.json',ledger)
    active=observe(frontier,state,80)
    assert not active['incidents'] and active['planned_work']['state']=='blocked_frontier_deferred'
    assert state['blocked_frontier_observation']['first_seen']==now
    ledger['_validation_fixed_points']['VOG-002']['execution_progress']=saved
    fixtures.write_json(lane/'lane_resume_attempts.json',ledger)
    fixtures.status(lane,phase='gate_dependency_revalidation')
    assert observe(frontier,state,81)['incidents'][0]['kind']=='planned_work_blocked_frontier'


def test_work_in_a_different_lane_still_blocks_dispatch(frontier):
    config,now,lane,live,probe,ledger,gates=frontier;state={}
    first=observe(frontier,state)
    tasks=dispatcher._tasks(config.repo_root/'todo.md','## VOG-')
    policy=json.loads(config.manifest_path.read_text())['planned_work_policy']
    result=planned.observe_planned_work(policy=policy,repo_root=config.repo_root,tasks=tasks,
        lanes=[*first['lanes'],{'lane':'other','active_task_id':'OTHER-1','phase':'deterministic_command'}],
        state=state,now=now+90,gate_blockers=first['planned_work']['blocked_frontier_proofs'])
    assert result['state']=='blocked_frontier_deferred' and not result['incidents']


@pytest.mark.parametrize('defect',['contract','fingerprint','source','changed_source','changed_guard'])
def test_stale_or_changed_proof_is_review_needed_never_current_blockage(frontier,defect):
    config,now,lane,live,probe,ledger,gates=frontier
    if defect=='contract':ledger['_validation_fixed_points']['VOG-002']['execution_progress']['contract']='f'*64
    if defect=='fingerprint':gates['VOG-002']['fingerprint']='f'*64
    if defect=='source':gates['VOG-002']['source_contract']='f'*64
    if defect=='changed_source':(config.repo_root/'producer.py').write_text('raise SystemExit(0)\n')
    if defect=='changed_guard':(config.repo_root/'data/gate-2.json').write_text('{"reason":"actual capability changed"}')
    fixtures.write_json(lane/'lane_resume_attempts.json',ledger)
    fixtures.write_json(lane/'lane_resume_attempts.gates.json',gates)
    result=dispatcher.tick(config,{},now=now+80,probe=probe)
    assert result['action']=='gate_evidence_review_needed' and result['frontier_gate_reviews']
    assert not result['incidents'] and not result['planned_work'].get('blocked_task_ids')


@pytest.mark.parametrize('invalid',[{'last_exit_code':1},{'last_exit_code':'75'},{'state':'validated'},
                                    {'schema':'unknown'},{'gate_refusals':0},{'finished_at_ms':0}])
def test_unqualified_metadata_cannot_be_accepted_as_gate_proof(frontier,invalid):
    config,now,lane,live,probe,ledger,gates=frontier
    ledger['_validation_fixed_points']['VOG-002']['execution_progress'].update(invalid)
    fixtures.write_json(lane/'lane_resume_attempts.json',ledger)
    assert not observe(frontier,{},80)['incidents']


def test_actual_maintenance_after_grace_is_bounded_and_does_not_change_board_or_gate_history(frontier):
    config,now,lane,live,probe,ledger,gates=frontier
    config.worker_argv=[sys.executable,'-c',"from pathlib import Path;Path('maintenance-private-plan').write_text('real bounded local fixture')"]
    config.max_attempts=1;config.worker_timeout=3
    before={p:p.read_bytes() for p in [config.repo_root/'todo.md',lane/'lane_resume_attempts.json',lane/'lane_resume_attempts.gates.json']}
    state={};first=dispatcher.tick(config,state,apply=True,now=now,probe=probe)
    assert first['action']=='blocked_frontier_wait'
    result=dispatcher.tick(config,state,apply=True,now=now+60,probe=probe)
    assert result['action']=='worker_finished' and result['worker']['exit_code']==0
    assert (config.repo_root/'maintenance-private-plan').is_file()
    last=dispatcher.tick(config,state,apply=True,now=now+61,probe=probe)
    assert last['action']=='attention_required' and sum(x['count'] for x in state['attempts'].values())==1
    assert all(p.read_bytes()==body for p,body in before.items()) and 'recovery_observed_at' not in state


def test_known_future_only_wait_is_visible_and_does_not_refund_attempts(frontier):
    config,now,lane,live,probe,ledger,gates=frontier
    board=config.repo_root/'todo.md';text=board.read_text()
    for task in ['VOG-002','VOG-003']:
        text=text.replace(f'## {task} Work\n- Status: pending',f'## {task} Work\n- Status: completed')
    board.write_text(text);fixtures.status(lane,phase='gate_retry_wait',active_task_id='VOG-004')
    state={'attempts':{'prior':{'count':2}},'unresolved_signature':'prior','next_dispatch_at':now+500}
    result=dispatcher.tick(config,state,now=now,probe=probe)
    assert result['action']=='waiting_external_gate' and not result['incidents']
    assert state['attempts']['prior']['count']==2 and state['next_dispatch_at']==now+500


def test_operator_hold_suppresses_maintenance_and_keeps_native_artifacts_unchanged(frontier):
    config,now,lane,live,probe,ledger,gates=frontier;state={};observe(frontier,state)
    config.state_dir.mkdir(parents=True);(config.state_dir/'hold').write_text('operator')
    result=dispatcher.tick(config,state,apply=True,now=now+90,probe=probe,
                           worker=lambda *_:pytest.fail('held maintenance'))
    assert result['action']=='held' and not result['incidents']
    assert not state.get('blocked_frontier_observation')


def test_real_cached_policy_mismatch_is_actionable_not_false_healthy(frontier,monkeypatch):
    config,now,lane,live,probe,ledger,gates=frontier
    source=config.repo_root/'old_policy.py';source.write_bytes(Path(lane_resume.__file__).read_bytes())
    name=lane_resume.__name__;spec=importlib.util.spec_from_file_location(name,source)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    source.write_text(source.read_text()+'\ndef _command_contract(*args, **kwargs):\n    return "f" * 64\n')
    ledger['_validation_fixed_points']['VOG-002']['execution_progress']['contract']='f'*64
    fixtures.write_json(lane/'lane_resume_attempts.json',ledger)
    monkeypatch.setitem(sys.modules,name,old);monkeypatch.setattr(rescue_package,'lane_resume',old)
    result=dispatcher.tick(config,{},now=now,probe=probe)
    assert result['action']=='would_dispatch' and result['frontier_gate_reviews']
    assert all(x['kind']=='source_change_review_needed' for x in result['incidents'])
    assert all(x['evidence']['loaded_policy_matches_disk'] is False for x in result['incidents'])


def test_verified_gated_queue_cannot_hide_independent_planned_cpu_frontier(frontier):
    config,now,lane,live,probe,ledger,gates=frontier
    board=config.repo_root/'todo.md';board.write_text(board.read_text().replace(
        '## VOG-005 Work\n- Status: planned\n- Depends on: VOG-002, VOG-003, VOG-004',
        '## VOG-005 Work\n- Status: planned\n- Depends on:'))
    goals=config.repo_root/'goals.json';document=json.loads(goals.read_text())
    document['tasks'][-1]['depends_on']=[];goals.write_text(json.dumps(document))
    state={};observe(frontier,state)
    result=observe(frontier,state,60)
    assert result['planned_work']['materialization_task_ids']==['VOG-005']
    assert result['incidents'][0]['kind']=='planned_work_blocked_frontier'


def test_an_untried_runnable_task_still_gets_its_real_execution_opportunity(frontier):
    config,now,lane,live,probe,ledger,gates=frontier
    board=config.repo_root/'todo.md';board.write_text(board.read_text()+
        '\n## VOG-006 Genuine new CPU work\n- Status: pending\n- Depends on: VOG-001\n'
        '- Deterministic command: python3 producer.py 6\n- Validation: python3 validator.py 6\n')
    goals=config.repo_root/'goals.json';document=json.loads(goals.read_text())
    document['tasks'].append({'id':'VOG-006','depends_on':['VOG-001'],'work_package':{}})
    goals.write_text(json.dumps(document))
    result=observe(frontier,{},100)
    assert not result['incidents'] and result['planned_work']['state']=='execution_available'
    assert not result['planned_work'].get('blocked_task_ids')
