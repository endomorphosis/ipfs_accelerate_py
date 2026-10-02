"""Published inference -> explicit native admission -> task context -> live Lake.

Admission is independently authored. This does not start the daemon or infer
an execution permission from the decoded contract.
"""
from pathlib import Path
import hashlib, json, subprocess, time, sys
from ipfs_accelerate_py.agent_supervisor.runtime.intent_advisor_selection import (
    prepare_intent_384_selection, load_intent_384_selection, intent_384_planner_summary)
from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task

ARTIFACTS = Path('/home/barberb/lift_coding/artifacts')
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else ARTIFACTS / 'intent384-startup-20261002/context-01'
ROOT.mkdir(exist_ok=False)
def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
prior = ARTIFACTS / 'intent-action-contracts-20261002'
intent_config = json.loads((prior / 'hub-consumer-advice.json').read_text())['config']
intent_path = ROOT / 'intent-config.json'
write(intent_path, intent_config)
security_config = json.loads((ARTIFACTS / 'intent-code-effects-20261002/run-03/security-config.json').read_text())
domains = {'capacity': {'lower': -1, 'upper': 1}, 'threshold': {'lower': -1, 'upper': 1}}
security_config['finite_state_domains'] = {'source.py': domains}
security_path = ROOT / 'security-config.json'
write(security_path, security_config)
weights = {str(Path(c['checkpoint_path'])): digest(Path(c['checkpoint_path'])) for c in [intent_config, security_config]}
assert all(weights[c['checkpoint_path']] == c['checkpoint_sha256'] for c in [intent_config, security_config])
rows = json.loads((prior / 'native-01/source-only-instructions.json').read_text())
source = json.loads((prior / 'native-01/source-inputs.json').read_text())[0]['source_text']
report = dict(schema='intent384-startup-context-qualification/v1', qualified=False,
    provider_calls=0, model_inference_executed=True, daemon_started=False, benchmark_result=False,
    admission_source='independently authored local single-goal/task; decoded candidate grants no authority',
    effects_configuration='generated from source-audited learned Intent plus caller-selected mapping and bounds',
    weight_sha256=weights, rows=[], controls=[])
write(ROOT / 'inputs.json', dict(cases=[dict(rows[i], expected_effect_status=s) for i,s in
    [(0,'satisfied'),(5,'refuted'),(1,'no_enabled_cases')]], source_text=source,
    intent_config=intent_config, security_config=security_config))
started = time.monotonic()
try:
    for index, expected in [(0,'satisfied'),(5,'refuted'),(1,'no_enabled_cases')]:
        instruction = rows[index]['instruction']
        directory = ROOT / ('case-' + str(index)); directory.mkdir()
        advice, selection, elapsed = prepare_intent_384_selection(instruction=instruction, config_path=intent_path)
        assert advice['status'] == 'semantic_candidate_advice', advice
        advice_file = directory / 'intent-advice.json'; write(advice_file, advice)
        checked = load_intent_384_selection(path=advice_file, expected_sha256=digest(advice_file),
            instruction=instruction, selection=selection)
        assert checked == advice
        summary, checked = intent_384_planner_summary(checked, instruction=instruction)
        assert summary is not None and checked == advice
        write(directory / 'selection.json', selection)
        write(directory / 'planner-summary.json', json.loads(summary))
        repo = directory / 'repository'; repo.mkdir()
        (repo / 'source.py').write_text(source)
        for argv in [('init','-q'),('add','source.py'),('-c','user.name=Qualification',
            '-c','user.email=qualification@example.invalid','commit','-qm','Original scalar source')]:
            subprocess.run(['git','-C',str(repo),*argv],check=True,capture_output=True)
        with IntentRepository(directory / 'intent.duckdb') as intent:
            declared = prepare_local_task(repository=repo, state=directory/'policy', intent=intent,
                scope_paths=['source.py'], output_path='source.py', validation_argv=['python3','-m','py_compile','source.py'],
                objective=instruction)
            cid = declared['task_cid']; before = intent.event_watermark()
            effect_config = dict(schema='supervisor-intent-code-effect-config/v2', contracts=[dict(
                id='declared-return',source_id='source.py',action_id='action',
                input_parameter_mapping={'left':'capacity','right':'threshold'},input_domains=domains)],
                lake=dict(executable='/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lake',timeout_seconds=60))
            context = prepare_supervised_task_context(repository=repo,intent=intent,task_cid=cid,
                paths=['source.py'],required_raw_paths=['source.py'],output=repo/'.runtime/context',
                security_source_program_config=security_path,intent_code_effect_instruction=instruction,
                intent_code_effect_intent_advice=checked,intent_code_effect_config=effect_config)
            effects = context['intent_code_effect_advice']
            assert effects['native'] is not None, effects
            assert effects['live_build_verified'] and effects['all_selected_contracts_checked'], effects
            assert effects['rows'][0]['effect_status'] == expected, effects
            assert effects['selected_bounded_effects_satisfied'] is (expected == 'satisfied')
            assert intent.event_watermark() == before and (repo/'source.py').read_text() == source
            assert intent.get_task(cid)['status'] == 'ready'
            report['rows'].append(dict(id=rows[index]['id'],intent_status=checked['status'],
                effect_status=expected,live_build_verified=True,association_replay_verified=effects['association_replay_verified'],
                task_status='ready',source_unchanged=True,task_revision_unchanged=True,
                intent_checkpoint_sha256=effects['intent_checkpoint_sha256'],
                security_checkpoint_sha256=effects['security_checkpoint_sha256'],
                semantic_root_cid=context['semantic_root_cid'],world_snapshot_cid=context['world_snapshot_cid'],
                case_count=effects['rows'][0]['case_count'],enabled_case_count=effects['rows'][0]['enabled_case_count'],
                preplanning_seconds=elapsed/1e9,proof_authority=False,execution_authority=False,completion_authority=False))
    for instruction in ['the agent may compute result.', 'Repair the vulnerabilities in bottle.py and write a report.']:
        advice, selection, _ = prepare_intent_384_selection(instruction=instruction,config_path=intent_path)
        summary, checked = intent_384_planner_summary(advice,instruction=instruction)
        assert advice['status'].startswith('fail_open_') and summary is None and advice['continue_planning']
        report['controls'].append(dict(instruction=instruction,status=advice['status'],continue_planning=True,summary=None))
    bad = dict(intent_config,checkpoint_sha256='0'*64); write(ROOT/'bad-config.json',bad)
    advice, _, _ = prepare_intent_384_selection(instruction=rows[0]['instruction'],config_path=ROOT/'bad-config.json')
    assert advice['status'].startswith('fail_open_') and advice['candidate_intent_ir'] is None
    report['controls'].append(dict(case='wrong_checkpoint_hash',status=advice['status'],continue_planning=True))
    report['inputs_and_weights_unchanged'] = all(digest(Path(p))==s for p,s in weights.items())
    assert report['inputs_and_weights_unchanged']
    report['qualified'] = True
finally:
    report['seconds'] = time.monotonic()-started
    write(ROOT/'result.json',report)
    print(json.dumps(report,indent=2))
