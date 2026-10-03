from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time
import psutil
root = Path(__file__).resolve().parent.parent
base = Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
datasets = Path('/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002')
old = json.loads((root / 'final-producers-after.json').read_text())
def sources():
    rows = []
    for row in old:
        path = Path(row['path'].replace('/tmp/ir-release-accelerate-20261001', str(base)).replace('/tmp/ir-release-datasets-20261001', str(datasets)))
        rows.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'historical_sha256': row['sha256']})
    return rows
before = sources()
assert all(row['sha256'] == row['historical_sha256'] for row in before)
(root / 'restart-producers-before.json').write_text(json.dumps(before, indent=2) + '\n')
env = dict(os.environ)
env.pop('RPI_PROCESS_LIMITS_CONTROLLED_HOST', None)
env.update(PYTHONPATH=str(base) + ':' + str(datasets), IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(root / 'restart-seal.duckdb'))
argv = ['/usr/bin/python3.12', '-m', 'pytest', '-q', '--color=no', 'test/api/test_repository_process_limits.py', 'test/api/test_repository_pipeline_resources.py', '--basetemp=' + str(root / 'native-restart-02'), '--junitxml=' + str(root / 'native-restart-02.xml')]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
probe = {'disk_percent': psutil.disk_usage('/home/barberb/lift_coding').percent, 'available_memory_bytes': psutil.virtual_memory().available}
t0 = time.perf_counter()
with (root / 'native-restart-02.log').open('xb') as log:
    run = subprocess.run(argv, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
post = sources()
(root / 'restart-producers-after.json').write_text(json.dumps(post, indent=2) + '\n')
result = {'schema': 'repository-process-limits-restart-run@1', 'started_at': started, 'elapsed_seconds': time.perf_counter() - t0, 'argv': argv, 'returncode': run.returncode, 'producer_drift': before != post, 'all_sources_match_historical': all(row['sha256'] == row['historical_sha256'] for row in post), 'resource_probe': probe, 'new_suite_resource_mode': 'actual_default; missing-helper/platform controls explicitly injected at named seams', 'legacy_suite_resource_mode': 'unchanged mixed pure, injected-owner and actual-host fixtures'}
(root / 'native-restart-02-run.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
raise SystemExit(run.returncode or (1 if before != post else 0))
