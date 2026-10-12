"""Scoped postmerge qualification, with isolated local stores and pinned sources."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/barberb/lift_coding')
P = ROOT/'.worktrees/supervisor-comparison-contracts-20261011'
D = ROOT/'.worktrees/census-tree-pin-datasets-20261012'
R = ROOT/'artifacts/supervisor-ranked-successor-20261012'
UPSTREAM = '1f6d998a92db4024f35b57bbdb183a6626a7fa31'
D_HEAD = 'c1b1f65f254f5f3ee8f307cdddf923c62450f78e'

def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()

groups = json.loads((R/'header_inventory_postmerge_nodeids.json').read_text())
assert groups['total'] == 25
selections = {f'inventory-authored-{i}': value for i, value in enumerate(groups['authored_groups'])}
selections['inventory-real'] = groups['real']
selections['bridge'] = [
    'test/api/test_doctor_header_cache_bridge.py::test_native_custody_replay_admits_only_local_receipt_and_native_cache_preserves_scope',
    'test/api/test_doctor_header_cache_bridge.py::test_repeat_native_replay_keeps_original_receipt_identity_without_quarantining',
]
selections['policy'] = [
    'benchmarks/agent_supervisor/container_coding/test_terminal_semantic_metadata_policy.py::test_each_opt_in_helper_requires_its_exact_current_archive_body',
    'benchmarks/agent_supervisor/container_coding/test_terminal_semantic_metadata_policy.py::test_missing_duplicate_or_boolean_archive_metadata_rejected',
    'benchmarks/agent_supervisor/container_coding/test_terminal_semantic_metadata_policy.py::test_current_inventory_passes_without_claiming_actual_archive_execution',
]
group = sys.argv[1]
assert len(sys.argv) == 2 and group in selections
subprocess.run(['git', '-C', str(P), 'merge-base', '--is-ancestor', UPSTREAM, 'HEAD'], check=True)
assert git(D, 'rev-parse', 'HEAD') == D_HEAD and not git(D, 'status', '--porcelain')
head = git(P, 'rev-parse', 'HEAD')
run = R/('header-inventory-postmerge-' + group + '-01')
run.mkdir(exist_ok=False)
prior = json.loads((ROOT/'artifacts/supervisor-numerical-runtime-20261011/deployment-final-01/command.json').read_text())
env = prior['environment_overrides'].copy()
env['PYTHONPATH'] = env['PYTHONPATH'].replace(str(ROOT/'.worktrees/formalization-datasets-20261011'), str(D))
env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(run/'seal.duckdb')
env['IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR'] = str(run/'orchestration')
env['IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI'] = '0'
policy = P/'benchmarks/agent_supervisor/container_coding/terminal_semantic_metadata_policy.py'
tree = ast.parse(policy.read_text())
source_paths = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'SOURCE_PATHS' for target in node.targets))
assert len(source_paths) == 41 and len(set(source_paths)) == 41
assert {'ipfs_accelerate_py/agent_supervisor/runtime/doctor_header_proof_inventory.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py',
    'ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py'} <= set(source_paths)
tracked = {P/name for name in source_paths}
tracked.update(P/name.split('::')[0] for name in selections[group])
tracked.update(P/name for name in ('test/api/test_doctor_header_cache_bridge.py',
    'test/api/test_doctor_contract_proof.py', 'test/api/test_doctor_contract_proof_custody.py',
    'test/api/test_doctor_header_contracts.py', 'test/api/test_doctor_task_workflow.py',
    'test/api/test_agent_supervisor_local_planning_admission.py'))
tracked.update(D/name for name in ('ipfs_datasets_py/logic/autoformal/tree_pin.py',
    'ipfs_datasets_py/logic/formalization/autoencoder/source_program_runtime_384_v2.py'))
tracked.add(Path(__file__).resolve())
def hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(tracked)}
command = [prior['command'][0], '-B', '-m', 'pytest', '-q', '--color=no', *selections[group],
    '--junitxml=' + str(run/'junit.xml'), '-o', 'cache_dir=' + str(run/'cache'),
    '-o', 'log_cli=false', '--tb=short', '--maxfail=1']
before = hashes()
meta = dict(command=command, cwd=str(P), accelerate_head=head, datasets_head=D_HEAD,
    upstream_ancestor=UPSTREAM, source_paths_count=len(source_paths), source_before=before,
    environment_overrides=env, scope='Offline header inventory, bridge defaults and archive closure; no training or live providers.')
(run/'command.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
started = time.monotonic()
with (run/'pytest.log').open('w') as output:
    result = subprocess.run(command, cwd=P, env={**os.environ, **env}, stdout=output, stderr=subprocess.STDOUT)
after = hashes()
record = dict(exit_code=result.returncode, seconds=time.monotonic()-started, source_after=after,
    source_unchanged=before==after, accelerate_head_after=git(P,'rev-parse','HEAD'),
    datasets_head_after=git(D,'rev-parse','HEAD'))
(run/'qualification.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps({key:value for key,value in record.items() if key != 'source_after'}))
raise SystemExit(result.returncode if before==after else 2)
