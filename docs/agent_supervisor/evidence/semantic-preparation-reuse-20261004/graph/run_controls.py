from pathlib import Path
import hashlib, json, os, subprocess, sys, time, xml.etree.ElementTree as E

B = Path(__file__).resolve().parent
W = B.parent.parent
A = W / '.worktrees/ir-release-accelerate-20261002'
D = W / '.worktrees/ir-admission-observation-datasets-20261004'
kind = sys.argv[1]
recipe = json.loads((W / 'artifacts/source384-construction-lifetime-20261004/dispatch-after-command.json').read_text())
env = {**recipe['environment_overrides'],
    'IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB': str(B / (kind + '-seal.duckdb')),
    'IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH': str(B / (kind + '-keys.db'))}
if kind == 'datasets':
    cwd = D
    env['PYTHONPATH'] = str(D)
    targets = ['tests/unit/logic/software_contracts/semantic_state']
else:
    assert kind == 'accelerate'
    cwd = A
    targets = recipe['argv'][5:-1] + [
        'test/api/semantic_state/test_semantic_context_runtime.py',
        'test/api/semantic_state/test_semantic_context_refresh.py',
        'test/api/semantic_state/test_semantic_capsule_selection.py',
        'test/api/semantic_state/test_semantic_worker_projection.py',
        'benchmarks/agent_supervisor/container_coding/test_terminal_initial_context.py',
        'benchmarks/agent_supervisor/container_coding/test_terminal_initial_context_lifetime.py']
owners = [D / p for p in ('ipfs_datasets_py/logic/software_contracts/semantic_state/capsules.py',
    'ipfs_datasets_py/logic/software_contracts/semantic_state/merkle.py',
    'ipfs_datasets_py/logic/ir_core/canonical.py')]
owners += [p for t in targets for p in (([cwd / t] if (cwd / t).is_file() else sorted((cwd / t).rglob('*.py'))))]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = {str(p):sha(p) for p in owners}
argv = [recipe['argv'][0], '-B', '-m', 'pytest', '-q', *targets, '--junitxml=' + str(B / (kind + '.xml'))]
command = dict(argv=argv, cwd=str(cwd), environment_overrides=env, source_pins=pins)
with (B / (kind + '-command.json')).open('x') as f: json.dump(command, f, indent=2)
started = time.monotonic()
with (B / (kind + '.stdout')).open('x') as out, (B / (kind + '.stderr')).open('x') as err:
    run = subprocess.run(argv, cwd=cwd, env={**os.environ, **env}, stdout=out, stderr=err, timeout=420)
tree = E.parse(B / (kind + '.xml'))
receipt = dict(returncode=run.returncode, seconds=time.monotonic()-started,
    tests=len(tree.findall('.//testcase')), failed=len(tree.findall('.//failure')),
    errors=len(tree.findall('.//error')), skipped=len(tree.findall('.//skipped')),
    source_pins_unchanged=pins=={str(p):sha(p) for p in owners})
with (B / (kind + '-exit.json')).open('x') as f: json.dump(receipt, f, indent=2)
print(json.dumps(receipt), flush=True)
assert not run.returncode and receipt['source_pins_unchanged'] and receipt['skipped']==0
