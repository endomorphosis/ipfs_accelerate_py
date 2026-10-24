"""Real producer/validator children: a probe is not a successful producer."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume
from ipfs_accelerate_py.agent_supervisor.rescue.gate_recovery import GateRecovery

@pytest.fixture
def work(tmp_path, monkeypatch):
    (tmp_path / "guard.json").write_text(json.dumps({"eligible": False, "generated_at": "initial"}))
    (tmp_path / "producer.py").write_text("""import json,pathlib,sys
r=pathlib.Path('.')
with (r/'events.log').open('a') as f:f.write('producer\\n')
guard=json.loads((r/'guard.json').read_text())
if not guard['eligible']:sys.exit(75)
(r/'result.json').write_text(json.dumps({'real_producer_ran':True}))
""")
    (tmp_path / "validator.py").write_text("""import json,pathlib,sys
r=pathlib.Path('.')
with (r/'events.log').open('a') as f:f.write('validator\\n')
if (r/'result.json').exists():sys.exit(0)
guard=json.loads((r/'guard.json').read_text())
if (r/'release').exists():guard['eligible']=True
guard['generated_at']='new-probe-clock'
(r/'guard.json').write_text(json.dumps(guard))
sys.exit(75)
""")
    clock=[1000.0]
    monkeypatch.setattr(lane_resume.time,'time',lambda:clock[0])
    monkeypatch.setattr(lane_resume,'_refill_module',lambda _:None)
    def configure(opt_in):
        (tmp_path/'board.md').write_text('## TASK-001 Real gated stage\n- Status: pending\n- Depends on:\n'
            '- Deterministic command: python3 producer.py\n- Validation: python3 validator.py\n'
            '- Outputs: guard.json, result.json\n- Resource class: cpu-small\n- Implementation timeout seconds: 10\n'
            + ('- Retry on changed evidence: true\n' if opt_in else ''))
    configure(True)
    def cycle():
        return lane_resume.run_resume_cycle(repo_root=tmp_path,todo_path=tmp_path/'board.md',
            task_header_prefix='## TASK-',attempt_path=tmp_path/'attempts.json',log_path=tmp_path/'lane.log',
            heartbeat=lambda **_:None)
    def events():return (tmp_path/'events.log').read_text().splitlines()
    return tmp_path,clock,configure,cycle,events

@pytest.mark.parametrize('restart',[False,True])
def test_validator_discovered_semantic_change_replays_real_producer_next_cycle(work,restart):
    root,clock,_,cycle,events=work
    assert cycle()['phase']=='gate_withheld'
    assert events()==['producer']
    (root/'release').touch()
    probe=cycle()
    assert probe['phase']=='gate_withheld' and events()==['producer','validator']
    assert '- Status: pending' in (root/'board.md').read_text()
    assert not (root/'result.json').exists(), 'probe must not manufacture producer output'
    record=GateRecovery(root/'attempts.json').records['TASK-001']
    assert record.get('producer_evidence_fingerprint')!=record['fingerprint']
    if restart:
        # Re-open the source and durable records in a genuinely fresh controller.
        script = """import importlib.util,json,pathlib,sys
import ipfs_accelerate_py.agent_supervisor.rescue as package
source=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2])
name=package.__name__+'.lane_resume';spec=importlib.util.spec_from_file_location(name,source)
m=importlib.util.module_from_spec(spec);sys.modules[name]=m;package.lane_resume=m;spec.loader.exec_module(m)
m._refill_module=lambda _:None
result=m.run_resume_cycle(repo_root=root,todo_path=root/'board.md',task_header_prefix='## TASK-',
    attempt_path=root/'attempts.json',log_path=root/'lane.log',heartbeat=lambda **_:None)
print(json.dumps(result))
"""
        child=subprocess.run([sys.executable,'-B','-c',script,lane_resume.__file__,str(root)],
            stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=20)
        assert child.returncode==0,child.stderr
        result=json.loads(child.stdout)
    else:
        result=cycle()
    assert result['advanced_task_id']=='TASK-001'
    assert events()==['producer','validator','producer','validator']
    assert json.loads((root/'result.json').read_text())['real_producer_ran'] is True
    assert '- Status: completed' in (root/'board.md').read_text()
    assert result['completion_authority'] is False


def test_changed_probe_guard_rearms_once_even_after_unchanged_gate_backoff(work):
    root,clock,_,cycle,events=work
    cycle();cycle();refusal=cycle()
    assert refusal['phase']=='gate_recovery_needed'
    assert events().count('producer')==1
    (root/'release').touch()
    clock[0]=refusal['next_retry_at_ms']/1000+1
    probe=cycle()
    assert probe['phase']=='gate_recovery_needed'
    record=GateRecovery(root/'attempts.json').records['TASK-001']
    assert record['next_retry_at_ms']>clock[0]*1000, 'new refusal budget remains persisted'
    assert record['source_refusals']>=4 and events().count('producer')==1
    result=cycle()  # A newly observed real guard can bypass that refusal cooldown once.
    assert result['advanced_task_id']=='TASK-001' and events().count('producer')==2


def test_clock_only_probe_changes_do_not_replay_producer(work):
    root,clock,_,cycle,events=work
    cycle();cycle();refusal=cycle()
    assert events().count('producer')==1 and not (root/'result.json').exists()
    record=GateRecovery(root/'attempts.json').records['TASK-001']
    assert record.get('producer_evidence_fingerprint',record['fingerprint'])==record['fingerprint']
    clock[0]=refusal['next_retry_at_ms']/1000+1
    cycle()
    assert events().count('producer')==1 and '- Status: pending' in (root/'board.md').read_text()


def test_default_policy_keeps_validation_only_retry(work):
    root,_,configure,cycle,events=work
    configure(False)
    cycle();(root/'release').touch();cycle();cycle()
    assert events().count('producer')==1 and not (root/'result.json').exists()
    assert '- Status: pending' in (root/'board.md').read_text()
    assert 'producer_evidence_fingerprint' not in GateRecovery(root/'attempts.json').records['TASK-001']


def test_operator_hold_prevents_replay_after_probe_changes_guard(work):
    root,_,_,cycle,events=work
    cycle();(root/'release').touch();cycle()
    before=events();(root/'HOLD').write_text('owned test hold')
    assert cycle()['phase']=='operator_hold' and events()==before
    assert '- Status: pending' in (root/'board.md').read_text()
