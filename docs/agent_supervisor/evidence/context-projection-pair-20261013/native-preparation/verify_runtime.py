from pathlib import Path
import hashlib,json,subprocess,os,tarfile,time
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
manifest=json.loads((R/'runtime-01/manifest.json').read_text());rows={r['path']:r for r in manifest['files']};assert len(rows)==len(manifest['files'])
start=time.monotonic();seen=set();total=0
with tarfile.open(R/'runtime-01/runtime.tar.gz','r|gz') as t:
 for m in t:
  assert m.name in rows and m.name not in seen and m.isfile()
  r=rows[m.name];assert m.size==r['bytes'] and m.mode==r['mode']
  with t.extractfile(m) as s:assert hashlib.file_digest(s,'sha256').hexdigest()==r['sha256'],m.name
  seen.add(m.name);total+=m.size
assert seen==set(rows)
env=json.loads((R/'bundle-command.json').read_text())['environment_overrides']
code="""import json
from pathlib import Path
from benchmarks.agent_supervisor.container_coding.terminal_source_view_policy import require_source_view_archive
from benchmarks.agent_supervisor.container_coding.terminal_semantic_metadata_policy import require_semantic_metadata_archive
from benchmarks.agent_supervisor.container_coding.terminal_deployment import validate_source384_binding, validate_torch_cpu_wheel_binding
from benchmarks.agent_supervisor.container_coding.terminal_grok_deployment import validate_grok_binding
manifest=json.loads(Path(%r).read_text())
require_source_view_archive(manifest,'identity@1','source384-5cpu-20gib-formalization@1')
require_semantic_metadata_archive(manifest,'common-bindings@1')
assert validate_source384_binding(manifest)
assert validate_torch_cpu_wheel_binding(manifest)
assert validate_grok_binding(manifest,required=True)
print('Current source-view, metadata, Source384, Torch and Grok bindings verified.')
""" % str(R/'runtime-01/manifest.json')
(R/'verify-runtime-bindings.py').write_text(code)
argv=['/home/barberb/lift_coding/.venvs/terminal-bench-harbor/bin/python','-B',str(R/'verify-runtime-bindings.py')]
(R/'verification-command.json').write_text(json.dumps(dict(argv=argv,cwd=str(P),environment_overrides=env,archive_stream_verification_script=str(Path(__file__))),indent=2,sort_keys=True)+'\n')
with (R/'verify.stdout').open('w') as out,(R/'verify.stderr').open('w') as err:
 c=subprocess.run(argv,cwd=P,env={**os.environ,**env},stdout=out,stderr=err,timeout=180)
result=dict(schema='source-context-live-runtime-verification@1',archive_sha256=manifest['archive_sha256'],all_archive_members_match_manifest=True,file_count=len(seen),uncompressed_bytes=total,current_source_capability_returncode=c.returncode,elapsed_seconds=time.monotonic()-start,provider_calls=0,model_calls=0,credentials_inspected=False)
(R/'verification-result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result));raise SystemExit(c.returncode)
