import asyncio, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
from benchmarks.agent_supervisor.container_coding.terminal_deployment import qualify_original_container
base=Path(__file__).parent
archive=base/'bundle'
manifest=json.loads((archive/'manifest.json').read_text())
consumer='source/ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py'
assert next(x['sha256'] for x in manifest['files'] if x['path']==consumer)=='5e7da432a206b07900822044326dde963b5bf8cb9c1f39d313e1a6ceed56e66f'
assert next(x['sha256'] for x in manifest['files'] if x['path']=='source/ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py')=='3c31ecff80baf3cb96324041a721d23ffc597da9aaa4841833f16c5799097fcc'
pins=json.loads((base/'prepared-a-pins.json').read_text())+json.loads((base/'frozen-d-pins.json').read_text())
archive_pins={row['path']:row['sha256'] for row in manifest['files']}
for row in pins:
 assert archive_pins.get(row['path'])==row['sha256'], 'archive producer differs from authorized freeze'
task=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/fix-code-vulnerability')
inputs={name:hashlib.sha256((task/name).read_bytes()).hexdigest() for name in ['instruction.md','task.toml','environment/Dockerfile']}
(base/'task-public-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
started=time.monotonic();code=0
try:
 result=asyncio.run(asyncio.wait_for(qualify_original_container(task_dir=task,archive_dir=archive,
  output=base/'docker-01',install_codex=False,resource_profile='source384-5cpu-12gib@1',source384_context=True),2300))
 (base/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
except BaseException as exc:
 code=1;traceback.print_exc()
 (base/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc)),indent=2)+'\n')
finally:
 (base/'exit.json').write_text(json.dumps(dict(returncode=code,seconds=time.monotonic()-started),indent=2)+'\n')
raise SystemExit(code)
