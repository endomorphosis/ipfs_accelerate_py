from pathlib import Path
import ast, cProfile, hashlib, json, os, pstats, shutil, subprocess, time
import importlib.abc, importlib.util, sys

B = Path(__file__).resolve().parent
W = B.parent.parent
A = W / '.worktrees/ir-release-accelerate-20261002'
D = W / '.worktrees/ir-admission-observation-datasets-20261004'
reference = os.environ.get('SEMANTIC_OWNER_MODE') == 'reference'
if reference:
    # Host diagnostic only: select archived producer source before any owner
    # imports. No runtime or benchmark entrypoint installs this reference loader.
    archived = {
        'ipfs_datasets_py.logic.software_contracts.semantic_state.merkle': B / 'merkle-before.py',
        'ipfs_datasets_py.logic.software_contracts.semantic_state.capsules': W / 'artifacts/capsule-compilation-reuse-20261004/before-capsules.py',
    }
    class ReferenceLoader(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in archived:
                return importlib.util.spec_from_file_location(fullname, archived[fullname])
    sys.meta_path.insert(0, ReferenceLoader())
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from benchmarks.agent_supervisor.container_coding.terminal_initial_context import _semantic_view

mode = os.environ.get('SEMANTIC_PROFILE_LABEL', 'before')
repo = B / 'repository'
if not repo.exists():
    repo.mkdir()
    source = W / 'artifacts/source384-construction-lifetime-20261004/trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__WCih55g/agent/public-output-evidence/before/bottle.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba'
    shutil.copyfile(source, repo / 'bottle.py')
    shutil.copyfile(W / '.benchmarks/terminal-bench-2/fix-code-vulnerability/instruction.md', repo / '.supervisor-instruction.md')
    tree = ast.parse((A / 'benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py').read_text())
    smoke = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PUBLIC_SMOKE' for t in n.targets))
    (repo / '.supervisor-public-smoke.py').write_text(smoke)
    for argv in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Qualification', '-c', 'user.email=local@example.invalid', 'commit', '-qm', 'Public input snapshot')]:
        subprocess.run(['git', '-C', str(repo), *argv], check=True, capture_output=True)
query = (repo / '.supervisor-instruction.md').read_text()
paths = ['.supervisor-instruction.md', '.supervisor-public-smoke.py', 'bottle.py']
pins = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths}
output = repo / '.runtime' / mode
profile = cProfile.Profile()
instrumented = os.environ.get('SEMANTIC_PROFILE_ENABLED', '1') == '1'
begun = time.monotonic()
if instrumented: profile.enable()
result = prepare_semantic_context(repository=repo, paths=paths, required_raw_paths=paths[:2],
    objective=' '.join(query.split()), task_id='TB-CODE-TASK', output=output,
    max_symbols=1024, worker_query=query, worker_max_bytes=32768)
build = time.monotonic() - begun
metadata = {'Semantic context artifact': str((output / 'worker-context.json').relative_to(repo)),
    'Semantic context sha256': result['worker_payload_sha256']}
payload, view, get_block = _semantic_view(repo, metadata, 'TB-CODE-TASK')
if instrumented: profile.disable()
elapsed = time.monotonic() - begun
if instrumented:
    profile.dump_stats(str(B / (mode + '.pstats')))
    with (B / (mode + '-profile.txt')).open('x') as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(65)
assert pins == {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths}
receipt = dict(mode=mode, build_seconds=build, build_and_reopen_seconds=elapsed,
    profiling_enabled=instrumented, reference_owner_injection=reference,
    container_measurement=False, provider_calls=0,
    source_pins=pins, source_pins_unchanged=True, result=result)
with (B / (mode + '-result.json')).open('x') as stream: json.dump(receipt, stream, indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k != 'result'}), flush=True)
