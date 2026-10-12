import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
ROOT=Path('/home/barberb/lift_coding')
P=ROOT/'.worktrees/supervisor-comparison-contracts-20261011'
D=ROOT/'.worktrees/formalization-datasets-20261011'
K=ROOT/'.worktrees/publication-main-20261004/kit'
R=ROOT/'artifacts/supervisor-token-efficiency-20261012/archive-01'
R.mkdir(exist_ok=False)
from benchmarks.agent_supervisor.container_coding.terminal_deployment import build_runtime_archive
from benchmarks.agent_supervisor.container_coding.terminal_semantic_metadata_policy import SOURCE_PATHS,require_semantic_metadata_archive

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
args=dict(output=R/'archive',source=P,datasets=D,kit=K,extension_dir=Path('/home/barberb/.duckdb/extensions/v1.5.5/linux_arm64'))
tracked={str(p):sha(p) for p in [P/name for name in SOURCE_PATHS]}
(R/'command.json').write_text(json.dumps(dict(function='terminal_deployment.build_runtime_archive',arguments={k:str(v) for k,v in args.items()},source_before=tracked,git_heads={str(p):subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in (P,D,K)},scope='Models-free source archive construction and actual archive-byte integrity, not full deployed runtime qualification.'),indent=2)+'\n')
start=time.monotonic()
manifest=build_runtime_archive(**args)
for mode in ('common-bindings@1','common-bindings-instruction@1'):
 require_semantic_metadata_archive(manifest,mode)
rows={row['path']:row for row in manifest['files']}
assert len(rows)==len(manifest['files'])
archive=R/'archive/runtime.tar.gz'
assert sha(archive)==manifest['archive_sha256']
checked=[]
with tarfile.open(archive) as stream:
 members=stream.getmembers()
 assert len(members)==len(rows) and len({m.name for m in members})==len(rows)
 for member in members:
  assert member.isfile() and member.name in rows and not member.name.startswith('/') and '..' not in Path(member.name).parts
  raw=stream.extractfile(member).read()
  row=rows[member.name]
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  top,relative=member.name.split('/',1)
  checkout={'source':P,'datasets':D,'kit':K,'extensions':args['extension_dir']}[top]
  assert sha(checkout/relative)==row['sha256']
  checked.append(member.name)
 extraction=R/'extracted'
 extraction.mkdir()
 stream.extractall(extraction,filter='data')
assert not any(name.startswith(('models/','toolchains/','runtime-wheels/')) for name in checked)
after={path:sha(Path(path)) for path in tracked}
qualification=dict(schema='instruction-metadata-archive-source-qualification@1',passed=tracked==after,seconds=time.monotonic()-start,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,manifest_sha256=sha(R/'archive/manifest.json'),verified_members=len(checked),all_actual_member_hashes_match_manifest_and_current_source=True,required_metadata_source_members=len(SOURCE_PATHS),qualified_modes=['common-bindings@1','common-bindings-instruction@1'],source_before=tracked,source_after=after,source_unchanged=tracked==after,model_assets_included=False,lean_assets_included=False,credentials_included=False,full_runtime_deployment_qualified=False,live_provider_calls=0,downloads=0)
(R/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
print(json.dumps({k:v for k,v in qualification.items() if k not in ('source_before','source_after')}))
