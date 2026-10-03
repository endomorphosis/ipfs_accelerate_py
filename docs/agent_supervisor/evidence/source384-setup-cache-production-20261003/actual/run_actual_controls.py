"""Run the applied cache controls from the actual checkout, without overlays."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

B = Path(__file__).resolve().parent
A = Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
D = Path('/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002')
DRAFT = B.parent / 'source384-wheel-cache-production-draft-20261003/timeout-02'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(B / 'actual-seals.duckdb')
sys.path[:0] = [str(A), str(D), '/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages']
os.chdir(A)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = json.loads((DRAFT / 'draft-pins.json').read_text())['files']
before = {row['path']: sha(A / row['path']) for row in rows}
assert all(before[row['path']] == row['proposed_sha256'] for row in rows)
tests = ['test_terminal_setup_cache_advice.py', 'test_terminal_deployment.py',
         'test_terminal_source384_qualification.py', 'test_terminal_source384_transport.py',
         'test_terminal_torch_wheel.py', 'test_full_supervisor_harbor_agent.py']
test_paths = [A / 'benchmarks/agent_supervisor/container_coding' / name for name in tests]
argv = ['-q', *map(str, test_paths), '--junitxml=' + str(B / 'actual-01.xml')]
receipt = {'argv': [sys.executable, str(__file__)], 'pytest_argv': argv, 'cwd': str(A),
           'producer_pins': before, 'test_pins': {str(p.relative_to(A)): sha(p) for p in test_paths},
           'source_mode': 'actual on-disk A modules; no module-path or method overlays',
           'a_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=A, text=True).strip(),
           'd_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=D, text=True).strip(),
           'environment': {name: os.environ[name] for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE',
               'IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB')}}
(B / 'actual-01-command.json').write_text(json.dumps(receipt, indent=2) + '\n')
import pytest
started = time.monotonic()
code = pytest.main(argv)
after = {name: sha(A / name) for name in before}
owner_names = {Path(name).stem for name in before if not Path(name).name.startswith('test_')}
loaded = {name: str(getattr(module, '__file__', '')) for name, module in sys.modules.items()
          if name.startswith('benchmarks.agent_supervisor.container_coding.') and name.split('.')[-1] in owner_names}
actual_sources = len(loaded) == len(owner_names) and all(Path(p).is_relative_to(A) for p in loaded.values())
(B / 'actual-01-exit.json').write_text(json.dumps({'returncode': int(code),
    'seconds': time.monotonic() - started, 'production_pins_unchanged': before == after,
    'all_owners_loaded_from_actual_checkout': actual_sources, 'loaded_owners': loaded}, indent=2) + '\n')
assert before == after and actual_sources
raise SystemExit(code)
