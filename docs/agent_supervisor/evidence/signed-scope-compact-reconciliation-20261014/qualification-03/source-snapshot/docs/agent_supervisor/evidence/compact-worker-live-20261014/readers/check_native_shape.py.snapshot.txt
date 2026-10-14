"""Authored metadata checks for the separate closed native-shape supplement."""
import hashlib, importlib.util, json
from pathlib import Path
A=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('supplement',A/'collect_native_shape.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
report=dict(planning=dict(qualified=True,planning_strategy='direct',goals=2,tasks=1,provider_calls=1,private_prompt='SECRET'),provider_invocations=[dict(invocation_id='a'*32,phase='coding',usage=dict(num_turns=5,exit_code=-15,timed_out=True,private_prompt='SECRET'))])
x=m.project(report,['a'*32])
check('explicit graph counts retained with missing subgoals unknown',x['planning']['goals']==2 and x['planning']['tasks']==1 and x['planning']['subgoals'] is None)
check('native turn count and signed failure retained',x['invocations'][0]['num_turns']==5 and x['invocations'][0]['exit_code']==-15)
check('raw unknown text excluded', 'SECRET' not in json.dumps(x))
check('absent tool census remains unknown',x['invocations'][0]['ordinary_source_read_or_edit_count'] is None and x['invocations'][0]['native_stream'] is None)
check('absent dispatch count stays unknown',x['native_dispatch_count'] is None)
try:m.project(report,[])
except ValueError:checks.append('different invocation population rejected')
else:raise AssertionError('population must refuse')
report['provider_invocations'][0]['native_grok_progress']={'schema':'native-grok-stream-progress@1','private_prompt':'SECRET'}
try:m.project(report,['a'*32])
except ValueError:checks.append('unvalidated stream envelope rejected')
else:raise AssertionError('stream must refuse')
result=dict(schema='compact-native-shape-authored-checks@1',passed=len(checks),checks=checks,provider_calls=0,production_files_changed=False,source_sha256=hashlib.sha256((A/'collect_native_shape.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (A/'native-shape-checks.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(passed=len(checks),sha256=result['source_sha256'])))
