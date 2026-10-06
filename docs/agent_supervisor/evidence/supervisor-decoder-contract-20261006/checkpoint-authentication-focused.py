"""Fresh frozen source-bound, offline checkpoint/catalog qualification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path('/home/barberb/lift_coding/.worktrees/supervisor-decoder-contract-20261006')
OUT = Path(__file__).parent
DATASETS = Path('/home/barberb/lift_coding/.worktrees/ir-supervisor-contracts-datasets-20261004')
NAME = 'checkpoint-authentication-focused-03'
OWNERS = [
    'ipfs_accelerate_py/agent_supervisor/runtime/task_ir_checkpoint.py',
    'test/api/test_task_ir_checkpoint.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/task_ir_selection.py',
    'test/api/test_task_ir_selection.py',
    'ipfs_accelerate_py/model_catalog/sources/ir_persistent.py',
]

def snapshot():
    return {name: {'bytes': (ROOT / name).stat().st_size,
                   'sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
            for name in OWNERS}

before = snapshot()
seal = Path('/dev/shm') / (NAME + '-' + uuid.uuid4().hex + '.duckdb')
base = Path('/tmp') / (NAME + '-' + uuid.uuid4().hex)
overrides = {
    'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'IPFS_DATASETS_PY_MINIMAL_IMPORTS': '1',
    'PYTHONPATH': str(ROOT) + ':' + str(DATASETS),
    'IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB': str(seal), 'IPFS_ACCELERATE_PYTEST_SEAL': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
}
command = [sys.executable, '-u', '-B', '-m', 'pytest', '-c', 'pytest.ini',
           '--noconftest', '--import-mode=importlib', '-q', '--tb=short',
           '--show-capture=no', '--color=no', '-o', 'log_cli=false',
           '--junitxml=' + str(OUT / (NAME + '.xml')), '--basetemp=' + str(base),
           'test/api/test_task_ir_checkpoint.py', 'test/api/test_task_ir_selection.py']
with (OUT / (NAME + '.log')).open('w') as log:
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **overrides},
                            stdout=log, stderr=subprocess.STDOUT)
after = snapshot()
import duckdb
with duckdb.connect(str(seal), read_only=True) as connection:
    seal_summary = connection.execute('SELECT count(*),min(file_count),max(file_count),bool_or(completion_authority) FROM pytest_ast_seal').fetchone()
summary = {'command': command, 'environment': overrides, 'exit_code': result.returncode,
           'before': before, 'after': after, 'owners_unchanged': before == after,
           'seal_database': str(seal), 'seal_summary': seal_summary,
           'model_training_inference_and_runtime_admission_performed': False,
           'real_retained_asset_writes_performed': False}
(OUT / (NAME + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'exit_code': result.returncode, 'owners_unchanged': before == after,
                  'seal_summary': seal_summary, 'summary': str(OUT / (NAME + '.json'))}))
sys.exit(result.returncode if before == after else 2)
