"""Read-only accounting of retained public host Bottle context objects, not RSS."""
from pathlib import Path
import dataclasses,hashlib,json,sys,types
import duckdb
from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot

B=Path(__file__).resolve().parent
ROOT=Path('/home/barberb/lift_coding/artifacts/terminal-source384-context-20261003/bottle-canary-04/app')
def account(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value)); total=sys.getsizeof(value)
    if isinstance(value,dict):
        return total+sum(account(k,seen)+account(v,seen) for k,v in value.items())
    if isinstance(value,(list,tuple,set,frozenset)):
        return total+sum(account(v,seen) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value,type):
        if hasattr(value,'__dict__'): total+=sys.getsizeof(value.__dict__)
        return total+sum(account(getattr(value,f.name),seen) for f in dataclasses.fields(value))
    return total
with duckdb.connect(str(ROOT/'.runtime/terminal-vectors/vectors.duckdb'),read_only=True,config={'threads':1}) as cx:
    row=cx.execute('SELECT payload FROM snapshots').fetchone()
snapshot=CodeVectorIndexSnapshot.from_dict(json.loads(row[0]))
context=ROOT/'.runtime/terminal-initial-context'
payload=json.loads((context/'semantic/worker-context.json').read_bytes())
world=json.loads((context/'initial-world.json').read_bytes())
indexed=json.loads((ROOT/'.runtime/terminal-vectors/result.json').read_bytes())
descriptor=json.loads((context/'descriptor.json').read_bytes())
objects={'serialized_snapshot_row':row,'native_vector_snapshot':snapshot,'unused_semantic_worker_payload':payload,
    'empty_world_capture':world,'indexed_qualification_result':indexed}
# These three are independently allocated in the actual prepare_initial_context
# construction and none is retained by the result passed to its fresh loader.
exact_three=('serialized_snapshot_row','native_vector_snapshot','unused_semantic_worker_payload')
seen=set(); union=sum(account(objects[k],seen) for k in exact_three)
value=dict(schema='initial-context-retained-input-accounting@1',source_root=str(ROOT),
    scope='Read-only reconstruction from retained prior public Bottle host artifacts. Object accounting is not current Docker RSS or proof of admission benefit.',
    interpreter=sys.version,native_snapshot=dict(rows=len(snapshot.rows),dimensions=snapshot.dimensions),
    serialized_snapshot_bytes=len(row[0].encode()),
    accounted_bytes={k:account(v) for k,v in objects.items()},
    three_disjoint_candidate_objects_accounted_bytes=union,
    other_objects_caveat='World/index totals are individually accounted; shared scalar aliases with result/descriptor are not subtracted, so do not sum these as exact freed memory.',
    omitted='Thin semantic view/reader; no loaded whole semantic graph. Descriptor shares construction summaries and is not added to their totals.',
    source_artifact_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
      [context/'semantic/worker-context.json',context/'initial-world.json',context/'descriptor.json',ROOT/'.runtime/terminal-vectors/result.json']},
    provider_calls=0,model_calls=0,database_read_only=True,rss_measured=False)
(B/'retained-input-accounting.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:value[k] for k in ['native_snapshot','accounted_bytes','three_disjoint_candidate_objects_accounted_bytes']}))
