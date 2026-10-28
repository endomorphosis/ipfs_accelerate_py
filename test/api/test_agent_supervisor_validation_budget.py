"""Actual bounded native command fixtures; these grant no proof authority."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT=Path('/home/barberb/lift_coding/JevOps')
sys.path.insert(0,'/home/barberb/lift_coding/external/ipfs_accelerate')
import ipfs_accelerate_py.agent_supervisor.rescue.lane_resume as canonical

@pytest.fixture
def lane():
    path=Path(os.environ.get('JLR_VALIDATION_CANDIDATE',canonical.__file__))
    name='ipfs_accelerate_py.agent_supervisor.rescue._budget_candidate'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

@pytest.mark.parametrize('value,expected',[
    (None,300),('',300),('bad',300),('inf',300),('nan',1),
    ('-1',1),('0',1),('60',60),('900',900),('5000',1800)])
def test_explicit_budgets_keep_native_cap_and_legacy_default(lane,value,expected):
    fields={} if value is None else {'implementation_timeout_seconds':value}
    assert lane._validation_timeout(fields)==expected
    assert lane._validation_timeout(fields)<=lane._task_timeout(fields)

@pytest.mark.parametrize('valid',[True,False])
def test_actual_native_validator_receives_registered_budget_and_controls_completion(lane,tmp_path,monkeypatch,valid):
    (tmp_path/'produce.py').write_text("from pathlib import Path\nPath('result').write_text('bounded fixture')\n")
    (tmp_path/'validate.py').write_text("from pathlib import Path\nassert Path('result').read_text()=='bounded fixture'\nraise SystemExit("+str(0 if valid else 1)+")\n")
    board=tmp_path/'board.md'
    board.write_text('## TEST-001 Bounded validator fixture\n- Status: ready\n- Depends on:\n'
        '- Deterministic command: python3 produce.py\n- Validation: python3 validate.py\n'
        '- Outputs: result\n- Implementation timeout seconds: 900\n'
        '- Acceptance: Actual successful validator required\n')
    real=lane._run_argv;calls=[]
    def observed(argv,**kwargs):
        calls.append((list(argv),kwargs['timeout']))
        return real(argv,**kwargs)
    monkeypatch.setattr(lane,'_run_argv',observed)
    result=lane.run_resume_cycle(repo_root=tmp_path,todo_path=board,task_header_prefix='## TEST-',
        attempt_path=tmp_path/'attempts.json',log_path=tmp_path/'lane.log',heartbeat=lambda **kw:None)
    assert any(argv[-1]=='validate.py' and timeout==900 for argv,timeout in calls)
    assert ('- Status: completed' in board.read_text()) is valid
    assert result['completion_authority'] is False

def test_hold_refuses_actual_commands_before_budgeted_execution(lane,tmp_path):
    (tmp_path/'HOLD').write_text('operator fixture hold\n')
    (tmp_path/'board.md').write_text('## TEST-001 Held fixture\n- Status: ready\n'
        '- Deterministic command: python3 produce.py\n- Implementation timeout seconds: 900\n')
    result=lane.run_resume_cycle(repo_root=tmp_path,todo_path=tmp_path/'board.md',task_header_prefix='## TEST-',
        attempt_path=tmp_path/'attempts.json',log_path=tmp_path/'lane.log',heartbeat=lambda **kw:None)
    assert result['phase']=='operator_hold' and not result['work_complete']


@pytest.mark.parametrize('damage',[None,'output','producer_failure','semantic_failure','short_budget','repeat_timeout'])
def test_exhausted_validation_timeout_gets_one_review_without_repeating_producer(lane,tmp_path,monkeypatch,damage):
    (tmp_path/'produce.py').write_text("from pathlib import Path\np=Path('calls')\np.write_text(str(int(p.read_text())+1) if p.exists() else '1')\nPath('result').write_text('bounded fixture')\n")
    (tmp_path/'validate.py').write_text("from pathlib import Path\nassert Path('result').read_text()=='bounded fixture'\n")
    board=tmp_path/'board.md'
    budget=60 if damage=='short_budget' else 900
    board.write_text('## TEST-001 Exhausted validator fixture\n- Status: ready\n- Depends on:\n'
        '- Deterministic command: python3 produce.py\n- Validation: python3 validate.py\n'
        '- Outputs: result\n- Semantic failure limit: 1\n- Implementation timeout seconds: '+str(budget)+'\n'
        '- Acceptance: Actual independent validator required\n')
    real=lane._run_argv;validation_calls=[];first=[True]
    def observed(argv,**kwargs):
        if argv[-1]=='validate.py':
            validation_calls.append(kwargs['timeout'])
            if first[0]:
                first[0]=False
                return 1 if damage=='semantic_failure' else 124
            if damage=='repeat_timeout':return 124
        return real(argv,**kwargs)
    monkeypatch.setattr(lane,'_run_argv',observed)
    def cycle():
        return lane.run_resume_cycle(repo_root=tmp_path,todo_path=board,task_header_prefix='## TEST-',
            attempt_path=tmp_path/'attempts.json',log_path=tmp_path/'lane.log',heartbeat=lambda **kw:None)
    assert cycle()['phase']=='validation_failed'
    import json
    attempt_path=tmp_path/'attempts.json';before=json.loads(attempt_path.read_text())
    if damage=='output':(tmp_path/'result').write_text('tampered fixture')
    if damage=='producer_failure':
        before['_command_evidence']['TEST-001']['exit_code']=1
        attempt_path.write_text(json.dumps(before))
    outcome=cycle();after=json.loads(attempt_path.read_text())
    assert (tmp_path/'calls').read_text()=='1'
    assert after['TEST-001']==before['TEST-001']
    assert after['_command_evidence']==before['_command_evidence']
    assert ('- Status: completed' in board.read_text()) is (damage is None)
    if damage in (None,'repeat_timeout'):
        review=after['_validation_fixed_points']['TEST-001']['validation_timeout_recovery']
        assert review['timeout_seconds']==900 and review['producer_reexecuted'] is False
        assert review['prior_execution_progress']['semantic_failures']==1
        assert len(validation_calls)==2
        if damage=='repeat_timeout':
            cycle();assert len(validation_calls)==2
    else:
        assert len(validation_calls)==1
    assert outcome['completion_authority'] is False
