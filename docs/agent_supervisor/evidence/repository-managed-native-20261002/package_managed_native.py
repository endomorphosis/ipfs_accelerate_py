"""Closed public export of two managed runs; excludes all private capability stores."""
from pathlib import Path
import hashlib,json,subprocess
BASE=Path('/home/barberb/lift_coding/artifacts/repository-behavioral-supervision-20261002')
REPOS={'datasets':Path('/tmp/ir-release-datasets-20261001'),'accelerate':Path('/tmp/ir-release-accelerate-20261001')}
DEST=REPOS['accelerate']/'docs/agent_supervisor/evidence/repository-managed-native-20261002'
assert not DEST.exists();DEST.mkdir(parents=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def put(relative,data):
 p=DEST/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def val(relative,data):put(relative,(json.dumps(data,indent=2,sort_keys=True)+'\n').encode())
for name in ['managed-native-02','managed-native-03']:
 for filename in [name+'.log',name+'-producers.json']:put(Path(filename),(BASE/filename).read_bytes())
 put(Path(name)/'result.json',(BASE/name/'result.json').read_bytes())
 prov=json.loads((BASE/(name+'-producers.json')).read_text())
 for row in prov['files']:
  if name.endswith('02'):
   data=subprocess.check_output(['git','-C',str(REPOS[row['repository']]),'show',prov['heads'][row['repository']]+':'+row['path']])
  else:data=(BASE/(name+'-sources')/row['repository']/(row['path']+'.txt')).read_bytes()
  assert sha(data)==row['sha256'];put(Path('sources')/name/row['repository']/(row['path']+'.txt'),data)
root=BASE/'managed-native-03';r=json.loads((root/'result.json').read_text())
for filename in ['candidate-result.json','initial-preparation.json','published-successor-preparation.json','independent-cold-successor-preparation.json',
  'initial-checked-cache.json','successor-checked-cache.json','successor-checked-cache-replay.json','cold-match.json','successor-match.json',
  'semantic-manifest-comparison.json','successor-semantic-comparison.json','pipeline-disk-ledger.json']:
 put(Path('managed-native-03')/filename,(root/filename).read_bytes())
# Preserve all immutable source/structured objects only; no catalog files.
for casname in ['artifacts','cold-artifacts']:
 for family in ['source','structured']:
  for p in sorted((root/casname/family).glob('*/*')):
   assert p.is_file() and not p.is_symlink() and p.name.startswith(('bafk','bagu'))
   put(Path('managed-native-03')/p.relative_to(root),p.read_bytes())
for p in sorted((root/'repository/.runtime/repository-finite-handoffs').glob('*.json')):
 assert sha(p.read_bytes())==p.stem
 put(Path('managed-native-03/public-handoffs')/p.name,p.read_bytes())
put(Path('managed-native-03/context-bundle.json'),(root/'repository/.runtime/context-bundle.json').read_bytes())
val(Path('managed-native-03/worker-receipt.json'),r['public_context_worker_receipt'])
# Two successor/cold native observation bodies are bounded, complete and public.
observation_names={'FiniteInteger.lean','FiniteInteger.olean','captured_source.py','compiled.json','driver.py','lean_certificate.json','lean_process.json',
 'observations.json','python_process.json','request.json','result.json','tool_policy.json'}
for dirname in ['successor-observation','cold-observation']:
 files=list((root/dirname).iterdir());assert {p.name for p in files}==observation_names
 for p in files:
  assert p.is_file() and not p.is_symlink();put(Path('managed-native-03')/dirname/p.name,p.read_bytes())
# Export authored source/check scripts at each immutable Git commit, without .git.
for label,commit in [('baseline',r['baseline_commit']),('published',r['published_commit'])]:
 paths=subprocess.check_output(['git','-C',str(root/'repository'),'ls-tree','-r','--name-only',commit],text=True).splitlines()
 assert set(paths)=={'.gitignore','calc.py','decoy.py','instruction.txt','public_check.py','unsupported.py'}
 for relative in paths:
  assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
  data=subprocess.check_output(['git','-C',str(root/'repository'),'show',commit+':'+relative]);assert len(data)<=65536
  put(Path('managed-native-03/fixture-sources')/label/relative,data)
for name in ['managed-native-03-independent-audit.json','audit_managed_native.py']:
 put(Path(name),(BASE/name).read_bytes())
put(Path('package_managed_native.py'),Path(__file__).read_bytes())
# Key-based review is a supplement to the explicit directory/file allowlist.
forbidden={'parent_token','private_key','secret_key','signing_key','lease_key','capability_key','access_token','refresh_token','quack_token'}
def scan(x,path):
 if isinstance(x,dict):
  for key,value in x.items():
   assert key not in forbidden,(str(path),key)
   scan(value,path)
 elif isinstance(x,list):
  for value in x:scan(value,path)
for p in DEST.rglob('*.json'):scan(json.loads(p.read_text()),p)
audit=json.loads((DEST/'managed-native-03-independent-audit.json').read_text())
summary=dict(schema='repository-managed-native-qualification/v1',status='qualified_bounded_managed_profile',
 failure=dict(run='managed-native-02',qualified=False,seconds=2.7003437159582973,
  reason='preparation output owner is outside the exact named disk roots',source_provenance='Exact historical committed bytes; no relabeling as later exact-path implementation.'),
 success=dict(run='managed-native-03',qualified=True,seconds=r['seconds'],task=r['task'],start=r['start']['status'],stop=r['stop']['status'],
  publication_matches_checked_candidate=True,cold_semantic_and_behavioral_agree=True,stale_index_cache_dispatch_refused=True,
  public_before_after_fences=True,owned_processes_remaining=0,owned_leases_remaining=0,provider_calls=0,provider_tokens=0,training_steps=0),
 reviewed_provenance=dict(success_direct_source_pins=9,success_captured_working_bytes_match_current=True,historical_direct_source_pins=6,
  verified_public_signatures=3,verified_source_cas_objects=118,protected_phases=7,total_pipeline_phases=24,
  public_integrity_replay='Complete retained before/candidate artifact reconstruction without fresh checker invocation',
  historical_resource_chain='one native host root -> protected validation phase -> daemon reservation -> worker child',
  private_grant_body_excluded=True,complete_transitive_dependency_attestation=False),
 limits=['This is one authored, complete controlled-language finite integer fixture, with model inference explicitly off; it is not an official Terminal-Bench score.',
  'The successful run used captured working bytes for the exact-path cache changes. Baseline Git commits identify ancestry, not those changed bytes.',
  'CPU/process-memory accounting is cooperative/sampled, not hard aggregate enforcement. GPU/per-device and gradual recovery remain outside this qualification.',
  'Disk accounting covers exact named roots and conservative precharges. Ambient semantic temporary paths and external supervisor/worker RSS are explicitly not captured.',
  'The independent review verifies retained signatures, hashes and historical lease identities. It does not reopen private SQL state or re-admit expired leases.',
  'No ablation, token-saving comparison, learned Intent interpretation, new training, or proof of general Python runtime semantics is claimed.'],
 exclusions=['private policy and owner directories','private resource grant bodies','native DuckDB/SQLite/WAL/lock files','lifecycle process grants and private key material','Git administration'])
val(Path('qualification.json'),summary)
readme='''# Managed supervisor native qualification\n\nThe successful `managed-native-03` run completed the authored finite repair, published the exact checked source, stopped the native supervisor and recorded zero remaining owned processes and leases in 178.179405 seconds. Initial validation failed as intended; final validation passed. Fresh successor checks and independent cold reconstruction agreed. The run made zero provider calls and used no learned interpretation or training.\n\n`managed-native-02` is retained separately: it was admitted, then failed after 2.700344 seconds because cache placement escaped the exact named disk roots. Its six source pins reconstruct from the historical commits. The successful run retains nine exact working-source copies, including the explicit cache-path fix, rather than assigning their bytes to the older baseline commit.\n\nIndependent review verified all 15 historical/current source pins, three public signatures, complete portable initial/candidate artifact reconstruction, 118 source/structured CAS objects, signed edit versus baseline/published Git bytes, and the recorded host-root → protected-validation-phase → daemon-reservation → worker-child chain. All 24 pipeline phases completed, including seven protected validation/cleanup phases. This is historical integrity replay, with no new prover/daemon execution and no private SQL reopen or live capability reuse.\n\nThe closed export includes raw run logs/results, selected preparation/cache reports, public signed handoffs and worker receipt, immutable CAS, successor/cold checker artifacts, authored baseline/published fixture files, and exact selected sources. Private policies, grant bodies, database files, locks, private signing material and Git internals are excluded.\n\nThe profile remains cooperative and sampled. Hard aggregate enforcement, GPU/per-device authority, ambient temporary storage and external supervisor/worker RSS remain unqualified. Conservative named-root write precharges are not measured throughput. This model-off fixture is not an official Terminal-Bench result or an ablation/token-saving comparison.\n'''
put(Path('README.md'),readme.encode())
files=[]
for p in sorted(DEST.rglob('*')):
 if p.is_file():
  b=p.read_bytes();files.append(dict(path=p.relative_to(DEST).as_posix(),bytes=len(b),sha256=sha(b)))
val(Path('manifest.json'),dict(schema='closed-public-managed-evidence/v1',files=files,total_bytes=sum(f['bytes'] for f in files)))
attrs=REPOS['accelerate']/'.gitattributes';line='docs/agent_supervisor/evidence/repository-managed-native-20261002/** -whitespace'
text=attrs.read_text()
if line not in text.splitlines():attrs.write_text(text+('' if text.endswith('\n') else '\n')+line+'\n')
print(json.dumps(dict(destination=str(DEST),manifest_sha256=sha((DEST/'manifest.json').read_bytes()),members=len(files),bytes=sum(f['bytes'] for f in files)),indent=2))
