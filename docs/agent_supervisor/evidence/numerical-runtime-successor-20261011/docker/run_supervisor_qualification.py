"""Offline Docker qualification of the existing shared-IR supervisor tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/barberb/lift_coding')
OUT = ROOT / 'artifacts/supervisor-numerical-runtime-20261011'
P = ROOT / '.worktrees/supervisor-comparison-contracts-20261011'
D = ROOT / '.worktrees/formalization-datasets-20261011'
K = ROOT / '.worktrees/publication-main-20261004/kit'
CHECKPOINT = ROOT / 'artifacts/supervisor-numerical-training-20261011/release/checkpoint.json'
MODELS = Path('/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small')
TAG = 'ipfs-supervisor-numerical-dev:20261011-r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    attempt = sys.argv[1]
    assert attempt.isdecimal() and len(attempt) == 2
    run = OUT / ('supervisor-' + attempt)
    run.mkdir(exist_ok=False)
    results = run / 'results'
    results.mkdir()
    results.chmod(0o777)
    (run / 'entry.py').write_bytes((OUT / 'supervisor_pytest_entry.py').read_bytes())
    name = 'numerical-runtime-supervisor-20261011-' + attempt
    tag = sys.argv[2] if len(sys.argv) > 2 else TAG
    image_id = subprocess.check_output(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'], text=True).strip()
    command = ['docker', 'run', '--name', name, '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,nodev,mode=1777,size=2147483648',
        '--tmpfs', '/home/supervisor/.local:rw,nosuid,nodev,uid=1000,gid=1000,mode=0700,size=67108864',
        '--memory', '12g', '--cpus', '5',
        '--pids-limit', '128', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges']
    for source, target, readonly in ((P, '/code/source', True), (D, '/code/datasets', True),
            (K, '/code/kit', True), (CHECKPOINT, '/models/checkpoint.json', True),
            (MODELS, '/models/models--thenlper--gte-small', True),
            (run / 'entry.py', '/qualification/entry.py', True),
            (results, '/qualification-output', False)):
        command += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    environment = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='',
        IPFS_KIT_DISABLE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_DISABLE_TELEMETRY='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTHONPATH='/code/source:/code/datasets:/code/kit',
        IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB='/tmp/pytest-seal.duckdb',
        NUMERICAL_SELECTION384_CHECKPOINT='/models/checkpoint.json',
        CODEBASE384_EMBEDDING_SNAPSHOT='/models/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd')
    for key, value in environment.items():
        command += ['--env', f'{key}={value}']
    command += ['--entrypoint', '/opt/ipfs-supervisor/venv/bin/python', image_id, '-I', '-B', '/qualification/entry.py']
    sources = [run / 'entry.py', CHECKPOINT]
    for relative in (
            'conftest.py', 'test/api/test_semantic_router_integration.py',
            'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_selection_integration.py',
            'benchmarks/agent_supervisor/container_coding/test_terminal_indexed_preparation.py',
            'benchmarks/agent_supervisor/container_coding/test_terminal_initial_context.py',
            'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',
            'benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py',
            'benchmarks/agent_supervisor/container_coding/terminal_context_audit.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/numerical_selection_config.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/numerical_selection_context.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py',
            'ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py',
            'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
            'ipfs_accelerate_py/llm_router.py'):
        sources.append(P / relative)
    for relative in ('logic/software_contracts/codebase_numerical_selection_384.py',
            'logic/software_contracts/codebase_numerical_selection_384_worker.py',
            'logic/backends/codebase_process.py',
            'logic/formalization/autoencoder/numerical_selection_384.py',
            'optimizers/logic_theorem_optimizer/autoencoder_embedding_runtime.py'):
        sources.append(D / 'ipfs_datasets_py' / relative)
    before = {str(path): sha(path) for path in sources}
    record = dict(command=command, image_id=image_id, environment=environment, source_before=before,
        git_heads={str(repo): subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            for repo in (P, D, K)})
    (run / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
    started = time.monotonic()
    code = None
    error = None
    try:
        with (run / 'stdout.txt').open('w') as stdout, (run / 'stderr.txt').open('w') as stderr:
            code = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=240).returncode
    except subprocess.TimeoutExpired:
        error = 'qualification_timeout_240_seconds'
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    after = {str(path): sha(path) for path in sources}
    remaining = subprocess.check_output(['docker', 'container', 'ls', '-a', '--filter', 'name=' + name,
        '--format', '{{.Names}}'], text=True).strip()
    counts = {}
    if (results / 'junit.xml').is_file():
        tree = ET.parse(results / 'junit.xml')
        counts = {key: sum(int(s.attrib.get(key, 0)) for s in tree.iter('testsuite'))
            for key in ('tests', 'errors', 'failures', 'skipped')}
    qualification = dict(returncode=code, error=error, counts=counts, seconds=time.monotonic()-started,
        source_before=before, source_after=after, source_hashes_unchanged=before == after,
        container_removed=remaining == '', results={path.name: sha(path) for path in results.iterdir() if path.is_file()},
        scope='supervisor_consumer_planner_router_and_historical_audit_with_authored_provider_replies_in_container',
        live_provider_calls=0, model_downloads=0, training_steps=0, benchmark_score=None)
    (run / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    print(json.dumps({k: v for k, v in qualification.items() if k not in ('source_before', 'source_after')}))
    print((run / 'stdout.txt').read_text()[-12000:])
    print((run / 'stderr.txt').read_text()[-5000:])
    raise SystemExit(code if code is not None else 1)


if __name__ == '__main__':
    main()
