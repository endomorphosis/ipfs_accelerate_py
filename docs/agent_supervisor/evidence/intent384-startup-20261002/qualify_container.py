"""Offline archive-only benchmark preprocessing; no coding worker or provider."""
from pathlib import Path
import hashlib,json,subprocess,time
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from ipfs_accelerate_py.agent_supervisor.runtime.intent_advisor_selection import load_intent_384_selection,intent_384_planner_summary
ROOT=Path('/out'); started=time.monotonic()
selected=Path('/opt/ipfs-supervisor/models/intent-action-384/config.json')
config=json.loads(selected.read_text())
report=dict(schema='intent384-offline-container-preplanning/v1',qualified=False,network='none',provider_calls=0,
    coding_worker_launched=False,benchmark_result=False,checkpoint_sha256=config['checkpoint_sha256'],cases=[])
try:
    for label,instruction,expected in [
        ('supported','the calculator must compute result; requires left > 0; ensures result = old(right) + old(left) and returned.','semantic_candidate_advice'),
        ('terminal-bench',Path('/inputs/instruction.md').read_text(),'fail_open_input_out_of_scope')]:
        root=ROOT/label;root.mkdir();repo=root/'app';repo.mkdir()
        # This is an input transport fixture, not the Bottle benchmark source.
        (repo/'bottle.py').write_text('def application():\n    return "fixture"\n')
        for args in [('init','-q'),('add','bottle.py'),('-c','user.name=Qualification','-c','user.email=qualification@example.invalid','commit','-qm','Fixture')]:
            subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
        source=root/'instruction.md';source.write_text(instruction)
        before=hashlib.sha256((repo/'bottle.py').read_bytes()).hexdigest()
        prepared=prep.prepare(repository=repo,instruction=source,state=root/'state',intent_action_384_config=selected)
        recorded=prepared['intent_preplanning']
        assert recorded['status']==expected, recorded
        advice=load_intent_384_selection(path=root/'state/intent-advice.json',expected_sha256=recorded['artifact_sha256'],
            instruction=instruction,selection=recorded['intent_action_384_selection'])
        summary,checked=intent_384_planner_summary(advice,instruction=instruction)
        assert advice['status']==expected and checked['status']==expected, checked
        assert (summary is not None)==(expected=='semantic_candidate_advice')
        assert prepared['query']==instruction==(repo/prep.INSTRUCTION).read_text()
        assert hashlib.sha256((repo/'bottle.py').read_bytes()).hexdigest()==before
        report['cases'].append(dict(label=label,status=expected,preprocessing_seconds=recorded['seconds'],
            before_goal_decomposition=recorded['before_goal_decomposition'],raw_instruction_preserved=True,
            numerical_replay_verified=advice['numerical_replay_verified'],summary_present=summary is not None,
            checkpoint_sha256=advice['checkpoint_sha256'],completion_authority=False,execution_authority=False))
    report['qualified']=True
finally:
    report['seconds']=time.monotonic()-started
    (ROOT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
