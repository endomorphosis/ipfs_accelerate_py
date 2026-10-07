"""Record an isolated, frozen-source qualification without changing shared state."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path('/home/barberb/lift_coding/.worktrees/supervisor-stop-20261007')
D = Path('/home/barberb/lift_coding/.worktrees/schedule-datasets-20261007')
OUT = Path(__file__).parent
label, *targets = sys.argv[1:]
assert label and targets and all(c.isalnum() or c in '-_' for c in label)

def git(root, *args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', '-C', str(root), *args]).decode().strip()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    paths = set(git(P, 'diff', '--name-only').splitlines())
    paths.update(git(P, 'ls-files', '--others', '--exclude-standard').splitlines())
    paths.update(target.split('::', 1)[0] for target in targets if not target.startswith('-'))
    return {
        'accelerate_head': git(P, 'rev-parse', 'HEAD'),
        'accelerate_status': git(P, 'status', '--porcelain=v1', '--untracked-files=all'),
        'datasets_head': git(D, 'rev-parse', 'HEAD'),
        'datasets_status': git(D, 'status', '--porcelain=v1'),
        'source_sha256': {name: digest(P / name) for name in sorted(paths)
                          if name.endswith('.py') and (P / name).is_file()},
    }

for suffix in ('-command.json', '-exit.json', '.xml', '.log', '.duckdb'):
    if (OUT / (label + suffix)).exists():
        raise SystemExit('Fresh qualification label required')
xml = OUT / (label + '.xml')
argv = ['/home/barberb/.local/bin/python', '-B', '-m', 'pytest', '-q', '-o', 'addopts=',
        '-p', 'no:cacheprovider', '-p', 'ipfs_accelerate_py.testing.pytest_ast_seal',
        '--junitxml=' + str(xml), '--basetemp=' + str(OUT / (label + '-fixtures')), *targets]
env = dict(
    PYTHONPATH=f'{P}:{D}:/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages',
    PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
    IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(OUT / (label + '.duckdb')),
    IPFS_DATASETS_RESOURCE_SCHEDULER_PATH=str(OUT / (label + '-resources.json')),
)
before = snapshot()
(OUT / (label + '-command.json')).write_text(json.dumps(
    dict(argv=argv, cwd=str(P), environment_overrides=env, before=before), indent=2) + '\n')
started = time.monotonic()
with (OUT / (label + '.log')).open('xb') as stream:
    result = subprocess.run(argv, cwd=P, env={**os.environ, **env}, stdout=stream,
                            stderr=subprocess.STDOUT)
after = snapshot()
record = dict(exit_code=result.returncode, seconds=time.monotonic() - started,
              source_unchanged=before == after, after=after,
              log_sha256=digest(OUT / (label + '.log')),
              xml_sha256=digest(xml) if xml.is_file() else None)
(OUT / (label + '-exit.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'after'}, indent=2))
print((OUT / (label + '.log')).read_text()[-8000:])
raise SystemExit(result.returncode)
