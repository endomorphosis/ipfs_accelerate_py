"""Bounded host-only observation diagnostic with ordinary lease renewal.

Reference imports and query counters are diagnostic instrumentation. No native
benchmark entrypoint uses this runner, its timings, or its isolated ledger.
"""
from pathlib import Path
import hashlib, importlib.abc, importlib.util, json, os, shutil, sys, time

B = Path(__file__).resolve().parent
W = B.parent.parent
D = W / '.worktrees/ir-admission-observation-datasets-20261004'
mode, label = sys.argv[1:]
assert mode in ('before', 'after') and label.startswith('renewed-')
assert '/' not in label and '..' not in label
owners = {
    'ipfs_datasets_py.logic.software_contracts.codebase_ir':
        B / 'codebase_ir.before.py' if mode == 'before' else D / 'ipfs_datasets_py/logic/software_contracts/codebase_ir.py',
    'ipfs_datasets_py.logic.software_contracts.duckdb_ast_store':
        W / 'artifacts/ast-batch-observation-20261004/duckdb_ast_store.before.py' if mode == 'before'
        else D / 'ipfs_datasets_py/logic/software_contracts/duckdb_ast_store.py',
}
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
def write(suffix, value):
    with (B / (label + suffix)).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)

paths = list(owners.values()) + [D / name for name in (
    'ipfs_datasets_py/logic/software_contracts/codebase_resources.py',
    'ipfs_datasets_py/logic/software_contracts/cache.py',
    'ipfs_datasets_py/logic/software_contracts/ast_ir.py',
    'ipfs_datasets_py/logic/software_contracts/content.py',
    'ipfs_datasets_py/duckdb_control/codebase_catalog.py',
    'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py')]
pins = {str(path): sha(path) for path in paths}
if mode == 'before':
    class ReferenceLoader(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in owners:
                return importlib.util.spec_from_file_location(fullname, owners[fullname])
    sys.meta_path.insert(0, ReferenceLoader())
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import duckdb, multiformats
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseHead
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig

retained = W / 'artifacts/source-combined-observation-performance-20261003/after-02'
receipt = json.loads((retained / 'receipt.json').read_text())
repository = retained / 'app'
original = {name: sha(repository / name) for name in receipt['source_hashes']}
assert original == receipt['source_hashes']
state = B / (label + '-state')
state.mkdir()
shutil.copyfile(retained / 'state/source.duckdb', state / 'source.duckdb')
artifact_root = retained / 'state/source-artifacts'
artifact_hashes = {str(path.relative_to(artifact_root)): sha(path)
                   for path in artifact_root.rglob('*') if path.is_file()}
database_sha = sha(retained / 'state/source.duckdb')
command = dict(schema='recorded-host-observation-command@1', argv=[sys.executable, '-B', *sys.argv],
    script_sha256=sha(Path(__file__)), source_pins=pins, source_hashes=original,
    mode=mode, reference_owner_injection=mode == 'before', cpu_affinity=sorted(os.sched_getaffinity(0)),
    observation_timeout_seconds=120.0, memory_reservation_mb=4096,
    scheduler_state=str(state / 'host-scheduler.json'), auto_renew_leases=True,
    lease_ttl_seconds=120.0, python=sys.version, duckdb_version=duckdb.__version__,
    multiformats_version=multiformats.__version__, retained_receipt_sha256=sha(retained / 'receipt.json'),
    retained_database_sha256=database_sha, current_directory=str(Path.cwd()),
    environment_overrides={key: os.environ.get(key) for key in (
        'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
    benchmark_result=False, container_measurement=False, production_hooks_installed=False)
write('-command.json', command)
results = dict(schema='source-observation-renewed-host-diagnostic@1', mode=mode,
    benchmark_result=False, container_measurement=False, provider_calls=0, training_steps=0)
started = time.monotonic()
try:
    with duckdb.connect(str(state / 'source.duckdb'), config={'threads': 1, 'memory_limit': '512MB'}) as native:
        store = DuckDBASTStore(connection=native)
        artifacts = ImmutableCAS(artifact_root)
        index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
            catalog=CodebaseCatalog(store, artifacts))
        counts, timings = {}, {}
        original_rows = store._rows
        def observed_rows(table, *args, **kwargs):
            counts[table] = counts.get(table, 0) + 1
            begun = time.monotonic()
            try:
                return original_rows(table, *args, **kwargs)
            finally:
                timings[table] = timings.get(table, 0.0) + time.monotonic() - begun
        store._rows = observed_rows
        scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=state / 'host-scheduler.json'))
        assert scheduler.config.auto_renew_leases is True and scheduler.config.lease_ttl_seconds == 120.0
        results['scheduler_before'] = scheduler.snapshot()
        head = CodebaseHead.from_dict(receipt['head'])
        wall, cpu = time.monotonic(), time.process_time()
        try:
            observed = index.observe_current(repository, expected_head=head, scheduler=scheduler,
                timeout_seconds=120.0, memory_mb=4096)
        finally:
            results.update(observation_seconds=time.monotonic() - wall,
                observation_cpu_seconds=time.process_time() - cpu,
                sql_row_queries=counts, sql_row_query_seconds=timings,
                scheduler_after=scheduler.snapshot())
        assert results['scheduler_after']['active_lease_count'] == 0
        results.update(manifest_cid=observed.manifest.cid, head=observed.head.to_dict())
except BaseException as error:
    results['failure'] = dict(type=type(error).__name__, message=str(error)[:2048])
finally:
    results.update(controller_seconds=time.monotonic() - started,
        source_pins_unchanged=original == {name: sha(repository / name) for name in original},
        producer_pins_unchanged=pins == {str(path): sha(path) for path in paths},
        original_database_unchanged=database_sha == sha(retained / 'state/source.duckdb'),
        retained_artifacts_unchanged=artifact_hashes == {str(path.relative_to(artifact_root)): sha(path)
            for path in artifact_root.rglob('*') if path.is_file()})
    good = 'failure' not in results and all(results[key] for key in (
        'source_pins_unchanged', 'producer_pins_unchanged', 'original_database_unchanged', 'retained_artifacts_unchanged'))
    results['returncode'] = 0 if good else 1
    write('-result.json', results)
    print(json.dumps({key: value for key, value in results.items() if not key.startswith('scheduler_')}), flush=True)
raise SystemExit(results['returncode'])
