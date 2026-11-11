"""Aggregate source-bound offline checks without conflating tested snapshots."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
B=Path(__file__).resolve().parent
W=Path('/home/barberb/lift_coding/.worktrees/supervisor-candidate-projection-20261015')
def bind(p):
 p=Path(p);r=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(r).hexdigest(),bytes=len(r))
def read(label):
 r=json.loads((B/(label+'-result.json')).read_text())
 assert r['exit_code']==0 and set(r['counts'])=={'passed'} and r['live_provider_calls']==0
 assert r['source_pins_unchanged'] and r['dependencies_unchanged'] and r['distribution_pins_unchanged']
 return r,json.loads((B/(label+'-source-pins.json')).read_text())['files']
finals=['codec-final02','router-final02','native-final02']
earlier=['codec-final01','router-final01','native-final01']
files={};runs=[]
for label in finals:
 r,pins=read(label)
 for name,digest in pins.items():
  assert bind(W/name)['sha256']==digest,(label,name)
  assert name not in files or files[name]==digest
  files[name]=digest
 runs.append(dict(label=label,passed=r['counts']['passed'],result=bind(B/(label+'-result.json')),pins=bind(B/(label+'-source-pins.json'))))
prior=[]
for label in earlier:
 r,pins=read(label)
 changed=[name for name,digest in pins.items() if bind(W/name)['sha256']!=digest]
 assert set(changed)<=set(['ipfs_accelerate_py/agent_supervisor/runtime/candidate_prompt_projection.py','test/api/test_candidate_prompt_projection.py'])
 prior.append(dict(label=label,passed=r['counts']['passed'],result=bind(B/(label+'-result.json')),pins=bind(B/(label+'-source-pins.json')),subsequently_changed_files=changed))
metrics=[]
for node in ET.parse(B/'native-final02.xml').getroot().iter('property'):
 if node.attrib.get('name')=='candidate_projection_native': metrics.append(json.loads(node.attrib['value']))
assert len(metrics)==1
value=dict(schema='candidate-projection-runtime-qualification@1',passed=True,passed_cases=sum(r['passed'] for r in runs),failures=0,errors=0,skipped=0,provider_calls=0,
 source_sha256=files,runs=runs,broader_prior_snapshot_qualification=prior,
 final_revision_change='Provider guidance preserves optional evidence status; added required/optional regression. Final checks requalify codec, both provider dispatch paths, bounds, archive and ordinary-route guards, plus full indexed public+Doctor continuation and retained-input audit.',
 authored_native_metrics=metrics[0],raw_content_exported=False,provider_token_usage_verified=False,complete_run_token_savings_claimed=False)
with (B/'runtime-qualification.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(passed_cases=value['passed_cases'],broader_prior_passed=sum(r['passed'] for r in prior),metrics=metrics[0])))
