"""Bounded public Bottle replay; private local DBs are never exported."""
import __future__, ast, gc, hashlib, json, os, pathlib, statistics, subprocess, sys, time, tracemalloc
import duckdb
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.logic.software_contracts import codebase_ir as module
from ipfs_datasets_py.logic.software_contracts.ast_ir import ASTRecord
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig
B=pathlib.Path(__file__).resolve().parent
D=pathlib.Path(module.__file__).resolve().parents[3]
source=B.parent/'source384-construction-lifetime-20261004/trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__WCih55g/agent/public-output-evidence/before/bottle.py'
private=B/'private'/'paired-profile-bottle'
private.mkdir(parents=True,exist_ok=False)
repo=private/'repository';repo.mkdir();(repo/'bottle.py').write_bytes(source.read_bytes())
def git(*args):subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
git('init','-q');git('config','user.name','Public fixture');git('config','user.email','fixture@example.invalid');git('add','.');git('commit','-qm','public Bottle fixture')
original=B/'codebase_ir-before.py'
parsed=ast.parse(original.read_text())
cls=next(node for node in parsed.body if isinstance(node,ast.ClassDef) and node.name=='RepositoryCodebaseIndex')
method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='observe_current')
namespace=dict(vars(module))
exec(compile(ast.Module(body=[method],type_ignores=[]),str(original),'exec',flags=__future__.annotations.compiler_flag),namespace)
old=namespace['observe_current']
owner=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=private/'scheduler.json',proof_resource_sampler=lambda:ProofHostResources(8,8192,8192),lane_reservations={},auto_renew_leases=False,poll_interval_seconds=.005,proof_backoff_seconds=.02))
connection=duckdb.connect(str(private/'index.duckdb'),config={'threads':1,'memory_limit':'256MB'})
store=DuckDBASTStore(connection=connection);cas=ImmutableCAS(private/'cas');index=module.RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),artifacts=cas,catalog=CodebaseCatalog(store,cas))
pinpaths=['ipfs_datasets_py/logic/software_contracts/'+name for name in ['codebase_ir.py','ast_ir.py','cache.py','duckdb_ast_store.py','schema_versions.py']]
def pins():return {p:hashlib.sha256((D/p).read_bytes()).hexdigest() for p in pinpaths}
record={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_bytes':source.stat().st_size,'source_path':str(source),'baseline_owner_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'source_pins_before':pins(),'profile_scope':'Public Bottle source; native file-backed DuckDB/CAS; isolated fixture scheduler; no provider, checkpoint, Docker, verifier or production ledger','observations':[]}
started=time.monotonic()
try:
 head=index.prepare_current(repo,repository_id='repository:public-bottle-observation',operation_id='initial',expected_head=None,scheduler=owner,limits=module.CodebaseScanLimits(max_entries=4,max_file_bytes=256*1024)).head
 assert module._ast_observation_context(index,store) is not None
 manifest=index.load(head.manifest_cid)
 record['inventory_files']=len(manifest.units)
 record['ast_units']=sum(unit.ast_cid is not None for unit in manifest.units)
 assert record['ast_units']==1, 'diagnostic must exercise AST observation'
 record['ast_payload_bytes']=[cas.path_for(unit.ast_cid).stat().st_size for unit in manifest.units if unit.ast_cid]
 expected=None
 code=ASTRecord.from_dict.__func__.__code__
 def execute(mode,measure):
  global expected
  calls=[]
  def profile(frame,event,arg):
   if event=='call' and frame.f_code is code:calls.append(frame.f_locals['value']['provenance']['path'])
  if measure=='constructions':sys.setprofile(profile)
  if measure=='allocations':gc.collect();tracemalloc.start()
  c,w=time.process_time(),time.monotonic()
  try: result=(old(index,repo,expected_head=head,scheduler=owner) if mode=='before' else index.observe_current(repo,expected_head=head,scheduler=owner))
  finally:
   cpu,wall=time.process_time()-c,time.monotonic()-w
   sys.setprofile(None)
  allocation=None
  if measure=='allocations':
   current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();allocation={'current_bytes':current,'peak_bytes':peak,'measurement':'Python traced allocations only; not RSS or cgroup memory'}
  digest=hashlib.sha256(json.dumps(result.manifest.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
  identity=(result.head.manifest_cid,result.manifest.cid,digest)
  if expected is None:expected=identity
  assert identity==expected
  row={'mode':mode,'measure':measure,'process_cpu_seconds':cpu,'wall_seconds':wall,'output_identity':identity}
  if measure=='constructions':row['astrecord_from_dict_calls']=calls
  if allocation:row['traced_allocations']=allocation
  record['observations'].append(row)
  (B/'paired-progress.json').write_text(json.dumps(record,indent=2)+'\n')
 execute('before','constructions');execute('after','constructions')
 for mode in ['after','before','before','after']:
  if time.monotonic()-started>35:break
  execute(mode,'timing')
 # Tracing is deliberately separate from timing and optional within a bounded run.
 if time.monotonic()-started<20:
  execute('before','allocations')
  if time.monotonic()-started<45:execute('after','allocations')
 state=owner.snapshot();assert state['active_lease_count']==state['waiting_request_count']==0
 record['leases_after']={'active':state['active_lease_count'],'waiting':state['waiting_request_count']}
 record['source_pins_after']=pins();record['source_pins_unchanged']=record['source_pins_before']==pins()
 record['outputs_equal']=True;record['elapsed_seconds']=time.monotonic()-started
 record['limitations']=['No RSS/cgroup memory or admission recovery measurement.','Timing is a short local component diagnostic, not Terminal-Bench performance.','Final SQL identity fence preserves existing blob/file/revision checks; no new atomic relation snapshot guarantee.']
 (B/'paired-result.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({'elapsed_seconds':record['elapsed_seconds'],'observations':len(record['observations']),'ast_units':record['ast_units'],'counts':[len(row['astrecord_from_dict_calls']) for row in record['observations'] if 'astrecord_from_dict_calls' in row]}))
finally:connection.close()
