"""Capture a fresh deterministic command, its environment and exact exit status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--cwd', required=True, type=Path)
parser.add_argument('argv', nargs=argparse.REMAINDER)
args = parser.parse_args()
assert args.output.parent == HERE and not args.output.exists()
args.output.mkdir()
config = json.loads((HERE/'environment.json').read_bytes())
env = dict(os.environ)
for name in ('OPENAI_API_KEY','XAI_API_KEY','ANTHROPIC_API_KEY','HF_TOKEN'):
    env.pop(name, None)
overrides = dict(config['environment_overrides'])
overrides['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(args.output/'seal.duckdb')
overrides['IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR'] = str(args.output/'orchestration')
overrides['IPFS_NUMERICAL_TEST_LAKE'] = '/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lake'
env.update(overrides)
argv = args.argv[1:] if args.argv and args.argv[0] == '--' else args.argv
argv = [config['python'], *argv]
(args.output/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(args.cwd),environment_overrides=overrides),indent=2)+'\n')
started = time.monotonic()
with (args.output/'stdout').open('w') as stdout, (args.output/'stderr').open('w') as stderr:
    result = subprocess.run(argv,cwd=args.cwd,env=env,stdout=stdout,stderr=stderr)
(args.output/'exit.json').write_text(json.dumps(dict(returncode=result.returncode,seconds=time.monotonic()-started),indent=2)+'\n')
print(json.dumps(dict(returncode=result.returncode,seconds=time.monotonic()-started,output=str(args.output))))
raise SystemExit(result.returncode)
