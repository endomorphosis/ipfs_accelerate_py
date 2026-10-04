from pathlib import Path
import hashlib, importlib.abc, importlib.util, json, os, shutil, sys, time

B=Path(__file__).resolve().parent
W=B.parent.parent
D=W/'.worktrees/ir-admission-observation-datasets-20261004'
mode=sys.argv[1]
assert mode in ('before','after')
label=os.environ.get('OBSERVATION_LABEL',mode)
owners={'ipfs_datasets_py.logic.software_contracts.codebase_ir':
            B/'codebase_ir.before.py' if mode=='before' else D/'ipfs_datasets_py/logic/software_contracts/codebase_ir.py',
        'ipfs_datasets_py.logic.software_contracts.duckdb_ast_store':
            W/'artifacts/ast-batch-observation-20261004/duckdb_ast_store.before.py' if mode=='before' else D/'ipfs_datasets_py/logic/software_contracts/duckdb_ast_store.py'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={str(p):sha(p) for p in owners.values()}
if mode=='before':
    class ReferenceLoader(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in owners:return importlib.util.spec_from_file_location(fullname,owners[fullname])
    sys.meta_path.insert(0,ReferenceLoader())
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import duckdb
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog,CodebaseHead
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler,ResourceSchedulerConfig

retained=W/'artifacts/source-combined-observation-performance-20261003/after-02'
receipt=json.loads((retained/'receipt.json').read_text())
repository=retained/'app'
original={name:sha(repository/name) for name in receipt['source_hashes']}
assert original==receipt['source_hashes']
state=B/(label+'-state');state.mkdir()
shutil.copyfile(retained/'state/source.duckdb',state/'source.duckdb')
artifact_root=retained/'state/source-artifacts'
artifact_hashes={str(p.relative_to(artifact_root)):sha(p) for p in artifact_root.rglob('*') if p.is_file()}
database_sha=sha(retained/'state/source.duckdb')
with duckdb.connect(str(state/'source.duckdb'),config={'threads':1,'memory_limit':'512MB'}) as native:
    # The copied database retains its declared artifact root. Observation only
    # reads that immutable store; do not change its binding to relocate it.
    store=DuckDBASTStore(connection=native);artifacts=ImmutableCAS(artifact_root)
    index=RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),artifacts=artifacts,catalog=CodebaseCatalog(store,artifacts))
    counts={};original_rows=store._rows
    def observed_rows(table,*args,**kwargs):
        counts[table]=counts.get(table,0)+1
        return original_rows(table,*args,**kwargs)
    store._rows=observed_rows
    scheduler=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=state/'host-scheduler.json',auto_renew_leases=False))
    head=CodebaseHead.from_dict(receipt['head']);results=[]
    for run in range(2):
        counts.clear();wall=time.monotonic();cpu=time.process_time()
        observed=index.observe_current(repository,expected_head=head,scheduler=scheduler,timeout_seconds=120.,memory_mb=4096)
        results.append(dict(run=run,seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
            sql_row_queries=dict(counts),manifest_cid=observed.manifest.cid,head=observed.head.to_dict()))
    assert scheduler.snapshot()['active_lease_count']==0
record=dict(schema='source-observation-batch-host-diagnostic@1',mode=mode,
    source_files=len(original),source_hashes=original,source_pins_unchanged=original=={name:sha(repository/name) for name in original},
    producers=pins,producer_pins_unchanged=pins=={str(p):sha(p) for p in owners.values()},
    reference_owner_injection=mode=='before',scheduler='actual host sampler, new isolated ledger',
    cpu_affinity_count=1,memory_reservation_mb=4096,results=results,provider_calls=0,training_steps=0,
    original_database_modified=database_sha!=sha(retained/'state/source.duckdb'),
    retained_artifacts_unchanged=artifact_hashes=={str(p.relative_to(artifact_root)):sha(p) for p in artifact_root.rglob('*') if p.is_file()},
    benchmark_result=False,container_measurement=False)
with (B/(label+'-result.json')).open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','producers')}),flush=True)
