"""Seal the bounded existing-consumer regression evidence after controls pass."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
A = BASE.parent.parent / '.worktrees/ir-release-accelerate-20261002'
RUN = BASE / 'regression-01'
OUT = BASE / 'public-regression'
OWNERS = [
    'ipfs_accelerate_py/agent_supervisor/runtime/local_completion_bridge.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/intent_requirement_observation.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py',
]

def main():
    command = json.loads((RUN / 'command.json').read_text())
    result = json.loads((RUN / 'exit.json').read_text())
    assert result['returncode'] == 0 and result['source_pins_unchanged'] is True
    counts = result['xml_counts']
    assert counts['tests'] == 115 and counts['failures'] == counts['errors'] == 0 and counts['skipped'] == 1
    tree = ET.parse(RUN / 'results.xml')
    cases = list(tree.getroot().iter('testcase'))
    identities = [(c.get('classname'), c.get('name')) for c in cases]
    assert len(identities) == len(set(identities)) == counts['tests']
    skipped = [{'test': c.get('classname') + '::' + c.get('name'), 'reason': c.find('skipped').get('message')} for c in cases if c.find('skipped') is not None]
    assert skipped == [{'test': 'api.test_header_intent_applicability::test_normal_initialization_with_real_checkpoint_produces_nomination_before_planning', 'reason': 'explicit pinned checkpoint and cached GTE snapshot required'}]
    assert not list(tree.getroot().iter('system-out')) and not list(tree.getroot().iter('system-err'))
    binding = {}
    for name in OWNERS:
        current = hashlib.sha256((A / name).read_bytes()).hexdigest()
        assert command['source_pins']['source/' + name] == result['after_pins']['source/' + name] == current
        original = BASE / (Path(name).stem + '.before.py')
        binding[name] = {'before_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'tested_sha256': current}
    OUT.mkdir(exist_ok=False)
    for source, target in [(RUN/'command.json','regression-01-command.json'),
                           (RUN/'exit.json','regression-01-exit.json'),
                           (RUN/'results.xml','regression-01.xml'),
                           (BASE/'run_regressions.py','run_regressions.py'),
                           (Path(__file__),'package_regressions.py'),
                           (BASE/'diagnosis-before.json','diagnosis-before.json')]:
        shutil.copy2(source, OUT / target)
    (OUT/'scope.json').write_text(json.dumps({'schema':'header-observation-regression-scope@1', 'cases':115, 'passed':114, 'failed':0, 'errors':0, 'skipped':skipped, 'new_v3_component_passed':6, 'combined_distinct_passes':120, 'whole_package_neural_qualification_claimed':False},indent=2)+'\n')
    (OUT/'source-binding.json').write_text(json.dumps({'schema':'header-nomination-owner-binding@1', 'owners':binding,
        'all_owner_source_pins_unchanged': True, 'complete_repository_pin_claim': False}, indent=2)+'\n')
    patch = subprocess.check_output(['git','diff','--',*OWNERS],cwd=A,text=True)
    (OUT/'change.patch').write_text(patch)
    (OUT/'README.md').write_text(f'''# Completed header observation replay

114 existing consumer controls passed, with no failures or errors, at unchanged selected source pins. One optional real-checkpoint initialization control skipped because explicit checkpoint/GTE asset paths were not supplied. The exact skip is recorded in scope.json; this group does not qualify neural initialization. They cover local completion, captured-header applicability, signed planning admission, published task context, requirement observation, and public instructions. The separate six-case component retains original failures and actual v3 publication/completion controls; counts from preliminary runs are not added.

Three replay consumers now pass the header nomination from the immediately verified signed planning receipt. Exact receipt equality, captured source/solver checks, native publication signatures and current-source fences remain active. The owner diffs and before/tested source digests are retained here.

The preceding T benchmark scored reward 1 and the native task completed, but its post-publication observation failed. Its bounded error did not retain the swallowed per-row cause. An actual authored v3 fixture reproduced that exact observation boundary before these changes. T used the original generation; these host controls do not qualify a new Docker trial or demonstrate benchmark efficiency. The six new controls used no model/provider calls, and the checkpoint-dependent existing control did not execute. Source384 successor inference remains separately unavailable, and private captured-store custody has not been relaxed for worker-UID access.

The command records exact cwd, argv, selected source pins and isolated seal/key/orchestration paths. Only passing XML, bounded metadata, source diff and artifact harnesses are exported. Raw logs, private stores, credentials, model bodies and hidden verifier bodies are excluded. The diagnosis JSON is a bounded inline-produced projection of the named historical receipt, not a second benchmark inspector execution.
''')
    rows=[]
    for p in sorted(OUT.iterdir()):
        raw=p.read_bytes(); rows.append({'path':p.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    manifest={'schema':'bounded-public-evidence-manifest@1','files':rows,'member_count':len(rows),
              'member_bytes':sum(r['bytes'] for r in rows),'exported_public_source_diff':True,
              'exported_credentials':False,'exported_model_bodies':False,'exported_verifier_bodies':False}
    target=OUT/'manifest.json'; target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'package':str(OUT),'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'members':len(rows),'counts':counts}))

if __name__ == '__main__':
    main()
