"""Frozen numerical worker protocol, deployment, retrieval and audit regression union."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/barberb/lift_coding')
P = ROOT/'.worktrees/supervisor-comparison-contracts-20261011'
D = ROOT/'.worktrees/numerical-obligation-datasets-20261012'
R = ROOT/'artifacts/numerical-obligation-worker-20261012'
FILES = ['test/api/test_current_source_lookup.py', 'test/api/test_current_source_lookup_cli.py', 'test/api/test_current_source_lookup_transport.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_obligation_retrieval.py', 'benchmarks/agent_supervisor/container_coding/test_numerical_obligation_worker_wiring.py', 'benchmarks/agent_supervisor/container_coding/test_source_formalization_worker_deployment.py', 'test/api/test_numerical_obligation_router.py', 'test/api/test_source_formalization_router.py', 'test/api/test_container_worker_timeout_contract.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_policy.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_worker_capability.py', 'test/api/test_numerical_obligation_client.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_obligation_broker.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_context.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_observation.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_preflight.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_obligation_integration.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_retrieval_observation.py']
FIXTURES = [
    'benchmarks/agent_supervisor/container_coding/test_terminal_indexed_preparation.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_initial_context.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_setup_cache_advice.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_deployment.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_semantic_metadata_policy.py',
    'test/api/test_semantic_router_integration.py',
    'test/api/test_agent_supervisor_local_planning_admission.py',
    'test/api/test_local_completion_bridge.py',
    'test/api/semantic_state/test_published_task_context.py',
]
SEAMS = ['ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py', 'ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py', 'ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py', 'ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup_cli.py', 
    'ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
    'ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py',
    'benchmarks/agent_supervisor/container_coding/terminal_setup_cache_advice.py',
    'benchmarks/agent_supervisor/container_coding/terminal_deployment.py',
]

def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()

run = R/sys.argv[1]
run.mkdir(parents=True, exist_ok=False)
prior = json.loads((ROOT/'artifacts/supervisor-numerical-runtime-20261011/deployment-final-01/command.json').read_text())
env = prior['environment_overrides'].copy()
env['PYTHONPATH'] = env['PYTHONPATH'].replace(str(ROOT/'.worktrees/formalization-datasets-20261011'), str(D))
env.pop('CODEBASE384_CHECKPOINT', None)
env.pop('CODEBASE384_EMBEDDING_SNAPSHOT', None)
env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(run/'seal.duckdb')
env['IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR'] = str(run/'orchestration')
env['IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI'] = '0'
tree = ast.parse((P/'benchmarks/agent_supervisor/container_coding/terminal_numerical_policy.py').read_text())
constants = {target.id: ast.literal_eval(node.value)
             for node in tree.body if isinstance(node, ast.Assign)
             for target in node.targets if isinstance(target, ast.Name)
             and target.id in {'SOURCE_PATHS', 'DATASET_PATHS'}}
tracked = {P/name for name in [*FILES, *FIXTURES, *SEAMS, *constants['SOURCE_PATHS'], 'benchmarks/agent_supervisor/container_coding/terminal_worker_capability.py']}
tracked.update(D/name for name in constants['DATASET_PATHS'])
tracked.add(Path(__file__).resolve())

def hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(tracked)}

selected = sys.argv[2:] or FILES
command = [prior['command'][0], '-B', '-m', 'pytest', '-q', '--color=no', *selected,
    '--junitxml='+str(run/'junit.xml'), '-o', 'cache_dir='+str(run/'pytest-cache'),
    '-o', 'log_cli=false', '--tb=short', '--maxfail=1']
before = hashes()
metadata = dict(command=command, cwd=str(P), accelerate_head=head(P), datasets_head=head(D),
    source_before=before, environment_overrides=env,
    scope='Native projections/lakes/admission/planner and coding dispatch with authored provider stubs; no paid calls, inference or benchmark-source execution. Lifecycle tests may run authored local fixture validations.')
(run/'command.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
start = time.monotonic()
with (run/'pytest.log').open('w') as stream:
    result = subprocess.run(command, cwd=P, env={**os.environ, **env}, stdout=stream, stderr=subprocess.STDOUT)
after = hashes()
suite = ET.parse(run/'junit.xml').getroot()
counts = {key: sum(int(node.get(key, 0)) for node in suite.iter('testsuite'))
          for key in ('tests', 'failures', 'errors', 'skipped')}
qualification = dict(exit_code=result.returncode, seconds=time.monotonic()-start,
    source_after=after, source_unchanged=before==after, counts=counts,
    accelerate_head_after=head(P), datasets_head_after=head(D))
(run/'qualification.json').write_text(json.dumps(qualification, indent=2, sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in qualification.items() if k != 'source_after'}))
raise SystemExit(result.returncode if before == after else 2)
