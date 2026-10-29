"""Declared upstream validation budgets share one fenced parent probe lifetime."""
import importlib.util
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import gate_recovery, lane_resume

CANONICAL=Path(gate_recovery.__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('_dependency_budget_fixtures',
    CANONICAL/'test/api/test_agent_supervisor_gate_recovery.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
lane=f.lane
actual_run=lane_resume._run_argv


@pytest.mark.parametrize('value,expected',[(None,30),('900',900),('9999',1800),
    ('nan',1),('inf',30),('invalid',30),('-1',1),('0.05',1)])
def test_only_explicit_valid_declared_timeout_expands_quick_default(value,expected):
    fields={'resource_class':'cpu-lean'}
    if value is not None:fields['implementation_timeout_seconds']=value
    assert lane_resume._dependency_validation_timeout(fields)==expected


def declared(lane,*,deep=False):
    text=lane.board.read_text()
    for task in ['TEST-010','TEST-040','TEST-060']:
        state='pending' if task=='TEST-060' else 'completed'
        text=text.replace(f'## {task} '+('Source' if task=='TEST-010' else 'Aggregate' if task=='TEST-040' else 'Closeout')+f'\n- Status: {state}\n',
                          f'## {task} '+('Source' if task=='TEST-010' else 'Aggregate' if task=='TEST-040' else 'Closeout')+f'\n- Status: {state}\n- Implementation timeout seconds: 900\n')
    if deep:
        text=text.replace('## TEST-010 Source\n- Status: completed\n- Implementation timeout seconds: 900\n- Depends on:',
                          '## TEST-010 Source\n- Status: completed\n- Implementation timeout seconds: 900\n- Depends on: TEST-005')
        text='## TEST-005 Deeper source\n- Status: completed\n- Depends on:\n- Validation: python3 deep.py\n\n'+text
        (lane.root/'deep.py').write_text('pass\n')
    lane.board.write_text(text)


def test_real_validator_receives_its_900_second_declaration_without_30_second_override(lane,monkeypatch):
    declared(lane)
    (lane.root/'source.py').write_text('import time\ntime.sleep(.04)\nprint("actual independent local validation")\n')
    observed=[]
    def run(argv,**kwargs):
        if argv[-1]=='source.py':
            observed.append(kwargs['timeout'])
            return actual_run(argv,**kwargs)
        return lane.run(argv,**kwargs)
    monkeypatch.setattr(lane_resume,'_run_argv',run)
    for _ in range(3):result=lane.cycle()
    assert observed and 30<observed[0]<=900
    packet=json.loads(Path(result['repair_packet']).read_text())
    actual=next(x for x in packet['dependency_validations'] if x['task_id']=='TEST-010')
    assert actual['executed'] and actual['exit_code']==0 and actual['timeout_seconds']>30
    assert 'actual independent local validation' in (lane.root/'lane.log').read_text()
    assert result['phase']=='gate_recovery_needed' and not actual['resolved_gate']
    assert '- Status: pending' in lane.board.read_text()


def test_all_dependencies_share_parent_budget_and_unstarted_checks_are_not_fake_timeouts(lane,monkeypatch):
    declared(lane,deep=True)
    clock=[50.0];calls=[]
    monkeypatch.setattr(lane_resume.time,'monotonic',lambda:clock[0])
    def run(argv,**kwargs):
        if argv[-1] in {'coverage.json','source.py','deep.py'}:
            calls.append((argv[-1],kwargs['timeout']))
            clock[0]+=600 if argv[-1]=='coverage.json' else 300
        return lane.run(argv,**kwargs)
    monkeypatch.setattr(lane_resume,'_run_argv',run)
    for _ in range(3):result=lane.cycle()
    assert calls==[('coverage.json',900),('source.py',300)]
    rows=json.loads(Path(result['repair_packet']).read_text())['dependency_validations']
    skipped=next(row for row in rows if row['task_id']=='TEST-005')
    assert skipped['executed'] is False and skipped['exit_code'] is None
    assert skipped['not_run_reason']=='dependency_probe_budget_exhausted'
    assert skipped['timeout_seconds'] is None and skipped['probe_elapsed_seconds']==900
    assert all(row['probe_budget_seconds']==900 and not row['resolved_gate'] for row in rows)


def test_dependency_probe_stays_once_per_source_across_cooldown_and_restart(lane,monkeypatch):
    declared(lane);calls=[]
    def run(argv,**kwargs):
        if argv[-1] in {'coverage.json','source.py'}:calls.append(argv[-1])
        return lane.run(argv,**kwargs)
    monkeypatch.setattr(lane_resume,'_run_argv',run)
    for _ in range(3):result=lane.cycle()
    assert calls==['coverage.json','source.py']
    for _ in range(3):
        lane.now[0]=result['next_retry_at_ms']/1000+1
        result=lane.cycle()
        assert calls==['coverage.json','source.py']
    persisted=json.loads((lane.root/'attempts.gates.json').read_text())['TEST-060']
    assert persisted['dependencies_revalidated'] is True and persisted['next_retry_at_ms']>int(lane.now[0]*1000)


def test_undeclared_dependency_keeps_30_seconds_and_holds_do_not_spend_a_probe(lane,monkeypatch):
    calls=[]
    def run(argv,**kwargs):
        if argv[-1] in {'coverage.json','source.py'}:calls.append(kwargs['timeout'])
        return lane.run(argv,**kwargs)
    monkeypatch.setattr(lane_resume,'_run_argv',run)
    for _ in range(2):lane.cycle()
    before=(lane.root/'attempts.gates.json').read_bytes()
    (lane.root/'HOLD').write_text('fixture operator pause')
    assert lane.cycle()['phase']=='operator_hold' and not calls
    assert (lane.root/'attempts.gates.json').read_bytes()==before
    (lane.root/'HOLD').unlink()
    result=lane.cycle()
    assert calls==[30,30] and result['phase']=='gate_recovery_needed'
