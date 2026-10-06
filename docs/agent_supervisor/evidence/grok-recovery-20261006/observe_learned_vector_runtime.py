"""Read only closed learned-vector qualification fields, with no new inference.

Equivalent retained producer for the inline trial10 observation. No source,
query, symbol, router trace or hit bodies leave the selected container.
The coordinator/caller supplies the exact owned-container identity.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
REMOTE = '''from pathlib import Path
import json,os,stat,hashlib
p=Path('/app/.runtime/terminal-vectors/result.json')
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
try:
 before=os.fstat(fd);assert stat.S_ISREG(before.st_mode) and before.st_size<=2097152
 raw=os.read(fd,2097153);after=os.fstat(fd)
 assert len(raw)==before.st_size and len(raw)<=2097152 and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
finally:os.close(fd)
def pairs(items):
 result={}
 for key,value in items:
  if key in result:raise ValueError('duplicate metadata key')
  result[key]=value
 return result
v=json.loads(raw,object_pairs_hook=pairs);assert v['schema']=='native-local-learned-vector-qualification@1'
def num(v):return v if type(v)is int and 0<=v<=100000000 else None
def boo(v):return v if type(v)is bool else None
def enum(v,allowed):return v if type(v)is str and v in allowed else None
c=v.get('configuration') or {};can=v.get('canary') or {};d=v.get('ducklake') or {}
result={'schema':'closed-learned-vector-runtime-observation@1','producer_schema':v['schema'],
 'status':enum(v.get('status'),{'qualified'}),'retained_result_bytes':len(raw),'retained_result_sha256':hashlib.sha256(raw).hexdigest(),
 'model_revision':enum(v.get('model_revision'),{'17e1f347d17fe144873b1201da91788898c639cd'}),
 'configuration':{'device':enum(c.get('device'),{'cpu','cuda'}),'max_seq_length':num(c.get('max_seq_length')),
  **{k:boo(c.get(k))for k in ('local_files_only','trust_remote_code','use_safetensors')}},
 'canary':{'disposition':enum(can.get('disposition'),{'passed','failed','skipped'}),
  'vector_lane':enum(can.get('vector_lane'),{'enabled','disabled','lexical_only'}),
  **{k:num(can.get(k))for k in ('observed_dimensions','sample_count')},'semantic_authority':boo(can.get('semantic_authority'))},
 'ducklake_status':enum(d.get('status'),{'projected'}),
 **{k:num(v.get(k))for k in ('dimensions','symbols','native_fact_rows_replayed','local_model_calls','local_model_texts','remote_provider_calls')},
 **{k:boo(v.get(k))for k in ('complete_permitted_scope','learned_embeddings','nomination_only','semantic_authority','full_system_qualified')},
 'embedding_input':enum(v.get('embedding_input'),{'qualified symbol names only'}),
 'observer_database_connections':0,'observer_inference_calls':0,'raw_symbol_query_source_or_credential_bodies_exported':False}
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-id', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{64}', args.container_id)
    assert re.fullmatch('[0-9]{2}', args.attempt)
    target = A/'grok-container'/('learned-vector-runtime-observation-'+args.attempt+'.json')
    if target.exists():
        raise SystemExit('fresh observation output required')
    process = subprocess.run(['docker','exec',args.container_id,'python3','-I','-B','-c',REMOTE],
        capture_output=True, text=True, timeout=15)
    if process.returncode:
        raise SystemExit('bounded learned qualification observation unavailable')
    result = json.loads(process.stdout)
    result.update(trial_name='grok-tune-mjcf-'+args.attempt,
                  observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
