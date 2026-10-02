import hashlib,json,os,resource,subprocess,time
from pathlib import Path
import duckdb
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
from ipfs_datasets_py.logic.software_contracts.codebase_paged_staging import CodebasePagedStager,CodebaseStagingLimits
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler,ResourceSchedulerConfig
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
out=Path('/home/barberb/lift_coding/artifacts/codebase-foundation-20261002/staging-01');out.mkdir(parents=True,exist_ok=False)
root=out/'repository';root.mkdir()
def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
git('init','-q');git('config','user.name','Codebase staging qualification');git('config','user.email','fixture@example.invalid')
for i in range(320):(root/f'unit_{i:04d}.py').write_text(f'def increment_{i}(n: int) -> int:\n    return n + {i}\n')
git('add','.');git('commit','-qm','320 bounded source units')
scheduler=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=out/'resources.json',proof_resource_sampler=lambda:ProofHostResources(8,8192,8192),lane_reservations={},auto_renew_leases=False))
cas=ImmutableCAS(out/'artifacts');db=out/'staging.duckdb'
def open_owner():
 cx=duckdb.connect(str(db),config={'threads':1,'memory_limit':'64MB','max_temp_directory_size':'128MB'})
 store=DuckDBASTStore(connection=cx)
 index=RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),artifacts=cas,catalog=CodebaseCatalog(store,cas))
 return cx,index,CodebasePagedStager(index)
cx,index,stager=open_owner();started=time.monotonic()
state=stager.prepare(root,repository_id='qualification:320',expected_commit=git('rev-parse','HEAD'),expected_tree=git('rev-parse','HEAD^{tree}'),limits=CodebaseStagingLimits(max_entries=512,max_batch_entries=16),scheduler=scheduler)
prepare_elapsed=time.monotonic()-started;times=[];restarted=False
while not state['complete_inventory_staged']:
 batch_start=time.monotonic();state=stager.advance(root,generation_cid=state['generation_cid'],expected_cursor=state['cursor'],scheduler=scheduler)
 times.append({'cursor':state['cursor'],'seconds':time.monotonic()-batch_start})
 if state['cursor']==128:
  cx.close();cx,index,stager=open_owner();restored=stager.status(state['generation_cid']);assert restored['cursor']==128;restarted=True
elapsed=time.monotonic()-started
assert state['cursor']==320 and state['disposition_counts']=={'captured_ast_ok':320}
assert index.current('qualification:320') is None
assert index.ingestor.store.stats()['size']==0
idle=scheduler.snapshot();assert idle['active_lease_count']==idle['waiting_request_count']==0
report={'schema':'codebase-paged-staging-qualification@1','qualified':True,'scope':'320 committed tiny Python files staged in deterministic bounded shards; no current head or behavioral proof publication','source_files':320,'max_batch_entries':16,'batches':len(times),'database_reopened_mid_run':restarted,'prepare_seconds':prepare_elapsed,'elapsed_seconds':elapsed,'units_per_second':320/elapsed,'peak_owner_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'duckdb_memory_limit':'64MB','source_executed':False,'provider_calls':0,'training_executed':False,'inference_executed':False,'final':state,'batch_times':times,'module_sha256':hashlib.sha256(Path(__import__('ipfs_datasets_py.logic.software_contracts.codebase_paged_staging',fromlist=['x']).__file__).read_bytes()).hexdigest()}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');cx.close()
print(json.dumps({k:report[k] for k in ('qualified','source_files','batches','elapsed_seconds','units_per_second','database_reopened_mid_run','peak_owner_rss_kib')}))
