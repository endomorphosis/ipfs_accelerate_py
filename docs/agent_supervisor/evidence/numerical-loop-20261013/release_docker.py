"""Pin the selected production closure and authored offline fixture before execution."""
import ast,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
D=Path('/home/barberb/lift_coding/.worktrees/indexed-coverage-datasets-20261013')
policy=P/'benchmarks/agent_supervisor/container_coding/terminal_numerical_policy.py'
values={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(policy.read_text()).body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'SOURCE_PATHS','DATASET_PATHS','GUARDED_DATASET_PATHS','CONTROL_DATASET_PATHS'}}
assert len(values)==4
paths=[P/name for name in values['SOURCE_PATHS']]+[D/name for key in ('DATASET_PATHS','GUARDED_DATASET_PATHS','CONTROL_DATASET_PATHS') for name in values[key]]
paths += [policy]
W=Path('/home/barberb/lift_coding')
public=W/'.benchmarks/terminal-bench-2/largest-eigenval'
paths += [public/'environment/src/eigen.py',public/'environment/src/eval.py',public/'instruction.md',W/'artifacts/terminal-suite-eigenval-pilot-20261004/largest-eigenval-profile.json']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
record=dict(schema='numerical-loop-offline-docker-release@1',provider_free_docker_qualification_authorized=True,profile='python-numerical-control-structure@3',source_bindings={str(p):sha(p) for p in sorted(set(paths))},runner_bindings={str(R/name):sha(R/name) for name in ('run_docker_qualification.py','owner_probe.py','worker_probe.py')},scope='Provider-free authored route; real owner/worker/router/IPC; no live provider, learned inference, native START, hidden verifier or benchmark reward.')
with (R/(sys.argv[1] if len(sys.argv)>1 else 'docker-release-01.json')).open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(source_bindings=len(record['source_bindings']),runner_bindings=len(record['runner_bindings']))))
