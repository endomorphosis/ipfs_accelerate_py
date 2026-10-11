"""Run the retained development image against read-only cached inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

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
    attempt = sys.argv[1] if len(sys.argv) == 2 else '02'
    assert attempt.isdecimal() and len(attempt) == 2
    run = OUT / ('inference-' + attempt)
    run.mkdir(exist_ok=False)
    results = run / 'results'
    results.mkdir()
    (run / 'inference_probe.py').write_bytes((OUT / 'inference_probe.py').read_bytes())
    # The rootless image's nonroot supervisor maps to a subordinate host UID.
    # This newly-created artifact-only export directory contains no private data.
    results.chmod(0o777)
    name = 'numerical-runtime-offline-20261011-inference-' + attempt
    image_id = subprocess.check_output(['docker', 'image', 'inspect', TAG, '--format', '{{.Id}}'], text=True).strip()
    command = ['docker', 'run', '--name', name, '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,nodev,mode=1777,size=2147483648', '--memory', '8g', '--cpus', '3',
        '--pids-limit', '128', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges']
    for source, target, readonly in ((P, '/code/source', True), (D, '/code/datasets', True),
            (K, '/code/kit', True), (CHECKPOINT, '/models/checkpoint.json', True),
            (MODELS, '/models/models--thenlper--gte-small', True),
            (OUT / 'inference_probe.py', '/qualification/inference_probe.py', True),
            (results, '/qualification-output', False)):
        command += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    environment = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='',
        IPFS_KIT_DISABLE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_DISABLE_TELEMETRY='1')
    for key, value in environment.items():
        command += ['--env', f'{key}={value}']
    command += ['--entrypoint', '/opt/ipfs-supervisor/venv/bin/python', image_id,
        '-I', '-B', '/qualification/inference_probe.py']
    sources = [OUT / 'inference_probe.py', CHECKPOINT]
    for repo, relative in (
            (D, 'ipfs_datasets_py/logic/software_contracts/codebase_numerical_selection_384.py'),
            (D, 'ipfs_datasets_py/logic/software_contracts/codebase_numerical_selection_384_worker.py'),
            (D, 'ipfs_datasets_py/logic/backends/codebase_process.py'),
            (D, 'ipfs_datasets_py/logic/formalization/autoencoder/numerical_selection_384.py'),
            (D, 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_embedding_runtime.py')):
        sources.append(repo / relative)
    before = {str(path): sha(path) for path in sources}
    record = dict(command=command, image_id=image_id, source_before=before,
        git_heads={str(repo): subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            for repo in (P, D, K)})
    (run / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
    started = time.monotonic()
    code = None
    error = None
    try:
        with (run / 'stdout.txt').open('w') as stdout, (run / 'stderr.txt').open('w') as stderr:
            code = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=300).returncode
    except subprocess.TimeoutExpired:
        error = 'qualification_timeout_300_seconds'
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    after = {str(path): sha(path) for path in sources}
    remaining = subprocess.check_output(['docker', 'container', 'ls', '-a', '--filter', 'name=' + name,
        '--format', '{{.Names}}'], text=True).strip()
    qualification = dict(returncode=code, error=error, seconds=time.monotonic()-started,
        source_before=before, source_after=after, source_hashes_unchanged=before == after,
        container_removed=remaining == '', results={path.name: sha(path) for path in results.iterdir() if path.is_file()})
    if (results / 'observation.json').is_file():
        qualification['observation'] = json.loads((results / 'observation.json').read_text())
    (run / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    print(json.dumps(qualification))
    print((run / 'stderr.txt').read_text()[-8000:])
    raise SystemExit(code if code is not None else 1)


if __name__ == '__main__':
    main()
