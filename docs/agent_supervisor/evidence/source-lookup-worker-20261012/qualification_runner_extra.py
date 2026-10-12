import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/barberb/lift_coding/.worktrees/token-selector-accelerate-20261011')
DS = Path('/home/barberb/lift_coding/.worktrees/source-worker-datasets-20261012')
label = sys.argv[1]
prefix = Path('/tmp/source-lookup-worker-' + label + '-20261012')
files = [
 'ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_broker.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_client.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
 'benchmarks/agent_supervisor/container_coding/container_worker_deployment.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py',
 'ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/container_worker_boundary.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_broker.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/repository_source_inventory.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py',
 'test/api/test_source_formalization_router.py',
 'docs/agent_supervisor/evidence/source-lookup-worker-20261012/qualification_runner.py',
 'test/api/test_source_lookup_worker_channel.py',
 'test/api/test_source_lookup_worker_router.py',
 'test/api/test_numerical_obligation_client.py',
 'test/api/test_numerical_obligation_router.py',
 'benchmarks/agent_supervisor/container_coding/test_numerical_obligation_worker_wiring.py',
 'benchmarks/agent_supervisor/container_coding/test_terminal_worker_capability.py',
 'benchmarks/agent_supervisor/container_coding/test_source_formalization_worker_deployment.py',
 'test/api/test_repository_source_inventory.py',
 'test/api/causal_federation/test_bootstrap_runtime.py',
]

files += ['test/api/test_worker_source_views.py', 'test/api/test_worker_source_broker.py', 'test/api/test_worker_source_client.py', 'benchmarks/agent_supervisor/container_coding/test_source_view_worker_wiring.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_policy.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_obligation_broker.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_source_views.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_source_broker.py', 'ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py', 'benchmarks/agent_supervisor/container_coding/terminal_source_view_policy.py', 'benchmarks/agent_supervisor/container_coding/terminal_source_view_observation.py', 'docs/agent_supervisor/evidence/source-lookup-worker-20261012/qualification_runner_extra.py']
dsfiles = ['ipfs_datasets_py/logic/software_contracts/' + path for path in (
 'cache.py', 'content.py', 'semantic_index/committed_snapshot.py',
 'semantic_index/chunked_snapshot.py', 'semantic_index/paged_snapshot.py')]
dsfiles += ['ipfs_datasets_py/logic/software_verification/numerical_obligation_retrieval.py']
def manifest():
 result = []
 for repo, root, paths in [('accelerator', ROOT, files), ('datasets', DS, dsfiles)]:
  for path in paths:
   raw = (root / path).read_bytes()
   result.append(dict(repository=repo,path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 return result
before = manifest()
targets = ['test/api/test_worker_source_views.py', 'test/api/test_worker_source_broker.py', 'test/api/test_worker_source_client.py', 'benchmarks/agent_supervisor/container_coding/test_source_view_worker_wiring.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_policy.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_obligation_broker.py']
updates = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
 IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI='0',
 PYTHONPATH=str(ROOT) + ':' + str(DS) + ':/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages:/home/barberb/lift_coding/.worktrees/publication-main-20261004/kit:/home/barberb/.local/lib/python3.12/site-packages')
record = dict(schema='supervisor-source-worker-qualification-command@1',
 cwd=str(ROOT), interpreter=sys.executable, resolved_interpreter=str(Path(sys.executable).resolve()),
 python_version=platform.python_version(), targets=targets,
 accelerator_base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 datasets_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DS,text=True).strip(),
 before=before, invocations=[])
for phase in ('collection', 'tests'):
 env = os.environ.copy()
 env.update(updates)
 for key in ('IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET', 'IPFS_ACCELERATE_AGENT_QUACK_TOKEN'):
  env.pop(key, None)
 phase_updates = dict(updates,
  IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(prefix)+'-'+phase+'-state',
  IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(prefix)+'-'+phase+'-seal.duckdb')
 env.update(phase_updates)
 argv = [sys.executable, '-m', 'pytest', '-c', 'pytest.ini', *targets,
  '--color=no', '-o', 'log_cli=false']
 if phase == 'collection':
  argv += ['--collect-only', '-qq']
 else:
  argv += ['--maxfail=1', '-q', '--tb=short', '--junitxml='+str(prefix)+'-junit.xml']
 log = Path(str(prefix)+'-'+phase+'.log')
 started = time.monotonic()
 with log.open('wb') as out:
  process = subprocess.run(argv,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
 call = dict(phase=phase,argv=argv,environment_updates=phase_updates,log=str(log),
  exit_code=process.returncode, elapsed_seconds=round(time.monotonic()-started,3),
  log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
 record['invocations'].append(call)
 if phase == 'collection':
  ids = [line.strip() for line in log.read_text().splitlines()
         if line.startswith(('test/api/', 'benchmarks/')) and '::' in line]
  nodeids = dict(schema='pytest-qualified-nodeids@1',nodeids=ids,distinct_count=len(set(ids)))
  Path(str(prefix)+'-nodeids.json').write_text(json.dumps(nodeids,indent=2)+'\n')
  record['collected_distinct_tests'] = len(set(ids))
 if process.returncode:
  break
record['after'] = manifest()
record['source_before_after_equal'] = before == record['after']
junit = Path(str(prefix)+'-junit.xml')
if junit.is_file():
 suites = ET.parse(junit).getroot().findall('testsuite')
 record['junit'] = {key:sum(int(s.get(key,0)) for s in suites)
                    for key in ('tests','failures','errors','skipped')}
 record['junit_sha256'] = hashlib.sha256(junit.read_bytes()).hexdigest()
Path(str(prefix)+'-commands.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({key: record.get(key) for key in
 ('collected_distinct_tests','junit','source_before_after_equal')}))
print(log.read_text()[-1600:])
raise SystemExit(record['invocations'][-1]['exit_code'])
