"""Join passing frozen-source qualifications; no runtime/provider execution."""
import hashlib,json,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent
W=Path('/home/barberb/lift_coding/.worktrees/supervisor-tool-event-coverage-20261015')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
source={};external={};cases=0;batches=[]
for label in ('capture-final01','routes-dev01','codec-final01','external-final01'):
 result_path=A/(label+'-result.json');pins_path=A/(label+'-source-pins.json')
 result=json.loads(result_path.read_bytes());pins=json.loads(pins_path.read_bytes())
 assert result['exit_code']==0 and result['counts'].get('passed',0)>0
 assert all(result['counts'].get(key,0)==0 for key in ('failed','error','skipped'))
 assert all(result[key] is True for key in ('source_pins_unchanged','dependencies_unchanged','distribution_pins_unchanged'))
 assert result['live_provider_calls']==0
 for path,digest in pins['files'].items():
  target=Path(path) if Path(path).is_absolute() else W/path
  assert sha(target)==digest
  dest=external if Path(path).is_absolute() else source
  assert path not in dest or dest[path]==digest
  dest[path]=digest
 cases+=result['counts']['passed']
 batches.append(dict(label=label,passed_cases=result['counts']['passed'],result_sha256=sha(result_path),source_pins_sha256=sha(pins_path)))
record=dict(schema='tool-event-coverage-qualification@1',passed=True,passed_cases=cases,failures=0,errors=0,skipped=0,
 runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),
 source_sha256=source,external_sha256=external,batches=batches,live_provider_calls=0,
 test_runner_sha256=sha(A/'run-tests.py'),qualification_script_sha256=sha(Path(__file__)))
with (A/'qualification-final01.json').open('x') as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(passed=True,passed_cases=cases,source_files=len(source),external_files=len(external),runtime_commit=record['runtime_commit'])))
