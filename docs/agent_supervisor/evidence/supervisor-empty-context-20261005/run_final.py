import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

root = Path.cwd()
evidence = root / 'docs/agent_supervisor/evidence/supervisor-empty-context-20261005'
seed = json.loads((evidence / 'interrupted-pre-compatibility-final-run.json').read_text())
files = [*seed['test_files'], 'benchmarks/agent_supervisor/container_coding/test_terminal_legacy_profile_context_reuse.py']
assert all((root / name).is_file() for name in files)
datasets = Path('/home/barberb/lift_coding/.worktrees/ir-supervisor-contracts-datasets-20261004')

def git(*args, cwd=root):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

# Retain the qualified file inventory when rerunning a clean published tree.
# Also capture any extra current edits, while excluding generated evidence.
names = sorted(set(seed['source_pins_before']) | {files[-1]} |
    set(git('diff', '--name-only').splitlines() + git('ls-files', '--others', '--exclude-standard').splitlines()))
names = [name for name in names if name.endswith('.py') and not name.startswith('docs/agent_supervisor/evidence/')]

def pins():
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}

def asset_pins():
    result = {}
    for key, row in seed['asset_pins_before'].items():
        path = Path(row['path'])
        result[key] = dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result

env = dict(os.environ)
env.update(seed['environment'])
env.update(HF_HUB_DISABLE_TELEMETRY='1',
    IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB='/dev/shm/supervisor-empty-final-' + uuid.uuid4().hex + '.duckdb')
args = ['python', '-u', '-B', '-m', 'pytest', '-c', 'pytest.ini', '--noconftest', '--import-mode=importlib',
    '-q', '--tb=short', '--show-capture=no', '--color=no', '-o', 'log_cli=false',
    '--junitxml=' + str(evidence / 'final.xml'), *files]
record = dict(schema='supervisor-empty-context-test-run@1', head=git('rev-parse', 'HEAD'),
    branch=git('branch', '--show-current'), datasets_head=git('rev-parse', 'HEAD', cwd=datasets),
    datasets_tracked_dirty=git('status', '--porcelain', '--untracked-files=no', cwd=datasets),
    command=args, environment={key: env[key] for key in seed['environment']}, test_files=files,
    source_pins_before=pins(), asset_pins_before=asset_pins(), completion_authority=False)
assert not record['datasets_tracked_dirty']
assert record['datasets_head'] == seed['datasets_head']
(evidence / 'final-run.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
print(json.dumps(dict(test_files=len(files), source_pins=len(names), evidence=str(evidence))), flush=True)
started = time.monotonic()
with (evidence / 'final.log').open('w') as log:
    run = subprocess.run(args, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
record.update(exit_code=run.returncode, seconds=time.monotonic()-started,
    source_pins_after=pins(), asset_pins_after=asset_pins())
record['sources_stable'] = record['source_pins_before'] == record['source_pins_after']
record['assets_unchanged'] = record['asset_pins_before'] == record['asset_pins_after']
if (evidence / 'final.xml').exists():
    xml = ET.parse(evidence / 'final.xml').getroot()
    record['junit'] = {key: sum(int(s.get(key, '0')) for s in xml.findall('testsuite'))
        for key in ['tests', 'errors', 'failures', 'skipped']}
import duckdb
with duckdb.connect(env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'], read_only=True) as db:
    row = db.execute('SELECT count(*), min(file_count), max(file_count), bool_or(completion_authority) FROM pytest_ast_seal').fetchone()
record['ast_seals'] = dict(records=row[0], closure_files_min=row[1], closure_files_max=row[2], any_completion_authority=row[3])
(evidence / 'final-run.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
print(json.dumps({key: record.get(key) for key in ['exit_code', 'seconds', 'junit', 'sources_stable', 'assets_unchanged', 'ast_seals']}), flush=True)
raise SystemExit(run.returncode)
