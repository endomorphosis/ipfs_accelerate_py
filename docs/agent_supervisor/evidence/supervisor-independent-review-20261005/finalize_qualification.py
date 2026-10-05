from pathlib import Path
import json, subprocess, xml.etree.ElementTree as ET
ROOT = Path('/home/barberb/lift_coding/.worktrees/supervisor-context-gaps-20261004')
SOURCE = Path('/home/barberb/lift_coding/artifacts/supervisor-independent-review-20261005')
DESTINATION = ROOT / 'docs/agent_supervisor/evidence/supervisor-independent-review-20261005'
run = json.loads((SOURCE/'final-02.qualification.json').read_text())
before = json.loads((SOURCE/'final-02-before.json').read_text())
after = json.loads((SOURCE/'final-02-after.json').read_text())
assert run['returncode'] == 0 and run['pins_unchanged'] and before == after
assert run['counts'] == {'passed':328,'failure':0,'error':0,'skipped':0}
assert run['ast_sealing']['records'] == 328 and not run['ast_sealing']['any_completion_authority']
assert before['datasets_tracked_status'] == ''
original = [case.attrib['classname']+'::'+case.attrib['name'] for case in ET.parse(SOURCE/'baseline-optional.xml').iter('testcase') if case.find('failure') is not None]
assert len(original) == 7 and all(run['case_outcomes'].get(case) == 'passed' for case in original)
subprocess.run(['python',str(SOURCE/'collect_public_evidence.py')],cwd=ROOT,check=True)
p = DESTINATION/'qualification.json'
qualification = json.loads(p.read_text())
qualification.update({
    'source_base':before['base_commit'],
    'upstream_integrated_commit':'cda8d05433b4c5453e63c54e121b6f49a992294e',
    'previous_frozen_local_run':'final-01.qualification.json',
    'final_local_run':'final-02.qualification.json',
    'counts':run['counts'],
    'source_assets_and_datasets_pins_unchanged':True,
    'ast_sealing':run['ast_sealing'],
    'original_seven_cases_all_passed':original,
    'selection':'final-02-selection.json',
    'retained_assets':before['retained_assets'],
    'current_source_pins':'final-02-before.json',
    'additional_controls':['26 native-parse and managed-child abbreviated-policy regressions','9 upstream literal-path, CRLF patch and bounded persistence controls'],
    'adoption_abbreviation_original_probe':'transport/adoption-abbreviation-probe-01.json',
    'hosted_ci':'not qualified by this local run',
})
p.write_text(json.dumps(qualification,indent=2)+'\n')
doc=ROOT/'docs/agent_supervisor/terminal_symbolic_capabilities.md'
s=doc.read_text().replace('The final local selection passes 293 tests','The final local selection passes 328 tests')
s=s.replace('rejects abbreviated, missing, duplicated or changed operator fields. Task metadata cannot select the policy or its trust\nroots.', 'rejects abbreviated, missing, duplicated or changed operator fields. Task\nmetadata cannot select the policy or its trust roots.')
s=s.replace('by the attestation. Another lane or a restarted daemon verifies these carriers against its\noperator-pinned policy', 'by the attestation. Another lane or a restarted daemon verifies these carriers\nagainst its operator-pinned policy')
doc.write_text(s)
print(json.dumps({'counts':run['counts'],'original_seven_passed':True,'assets_unchanged':True}))
