import hashlib, json, os, pathlib, subprocess, sys, time
base = pathlib.Path(__file__).parent
cfg = json.loads((base / 'environment.json').read_text())
root = pathlib.Path('/home/barberb/lift_coding/.worktrees/supervisor-token-growth-20261013')
run = base / sys.argv[1]
run.mkdir(exist_ok=False)
nodes = sys.argv[2:]
env = os.environ.copy(); env.update(cfg['environment_overrides'])
env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(run / 'seal.duckdb')
cmd = [cfg['python'], '-m', 'pytest', '-q', '--color=no', '-o', 'log_cli=false', '--junitxml=' + str(run / 'junit.xml'), *nodes]
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=root).decode().splitlines()
untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root).decode().splitlines()
paths = sorted(set(changed + untracked + [n.split('::')[0] for n in nodes]))
def pins():
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths if (root / p).is_file()}
before = pins()
(run / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
(run / 'source-before.json').write_text(json.dumps(before, indent=2) + '\n')
start = time.monotonic(); timed_out = False
with (run / 'stdout').open('w') as out, (run / 'stderr').open('w') as err:
    try:
        outcome = subprocess.run(cmd, cwd=root, env=env, stdout=out, stderr=err, timeout=900)
        code = outcome.returncode
    except subprocess.TimeoutExpired:
        code = 124; timed_out = True
after = pins()
(run / 'source-after.json').write_text(json.dumps(after, indent=2) + '\n')
result = dict(exit_code=code, timeout=timed_out, elapsed_seconds=time.monotonic()-start,
    source_unchanged=before == after, changed_sources=[p for p in paths if before.get(p) != after.get(p)])
(run / 'exit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result)); sys.exit(code)
