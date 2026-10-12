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
DS = Path('/home/barberb/lift_coding/.worktrees/source-inventory-datasets-20261012')
label = sys.argv[1]
prefix = Path('/tmp/current-source-lookup-' + label + '-20261012')
files = [
 'ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup_cli.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py',
 'ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py',
 'ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py',
 'ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/repository_source_inventory.py',
 'test/api/test_current_source_lookup.py',
 'test/api/test_current_source_lookup_cli.py',
 'test/api/test_current_source_lookup_transport.py',
 'test/api/test_repository_source_inventory.py',
 'test/api/causal_federation/test_typed_state_owner.py',
 'test/api/causal_federation/test_bootstrap_runtime.py',
]
dsfiles = ['ipfs_datasets_py/logic/software_contracts/' + path for path in (
 'cache.py', 'content.py', 'semantic_index/committed_snapshot.py',
 'semantic_index/chunked_snapshot.py', 'semantic_index/paged_snapshot.py')]
def manifest():
 result = []
 for repo, root, paths in [('accelerator', ROOT, files), ('datasets', DS, dsfiles)]:
  for path in paths:
   raw = (root / path).read_bytes()
   result.append(dict(repository=repo,path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 return result
before = manifest()
targets = [
 'test/api/test_current_source_lookup.py',
 'test/api/test_current_source_lookup_cli.py',
 'test/api/test_current_source_lookup_transport.py',
 'test/api/causal_federation/test_typed_state_owner.py::test_grant_rejects_self_admission_unknown_mutation_and_scope_promotion',
 'test/api/causal_federation/test_typed_state_owner.py::test_live_grant_revocation_is_checked_on_every_request',
 'test/api/causal_federation/test_typed_state_owner.py::test_partial_response_timeout_poison_closes_the_client_transport',
 'test/api/causal_federation/test_typed_state_owner.py::test_grant_is_kernel_bound_to_an_independent_process',
]
updates = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
 PYTHONPATH=str(ROOT) + ':' + str(DS))
record = dict(schema='supervisor-current-source-qualification-command@1',
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
         if line.startswith('test/api/') and '::' in line]
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
