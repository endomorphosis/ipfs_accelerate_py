"""Read only bounded capsule metadata from one campaign container; no bodies."""
import argparse
import json
import subprocess
from campaign_common import A, LABELS, binding, need, now, read, sha, write_new

# No imports from the installed repository and no writes inside the container.
# This fixed artifact is separate from owner state/signing keys and transcripts.
OBSERVE = r'''
import hashlib,json,os,pathlib,re,stat
p=pathlib.Path('/app/.runtime/terminal-initial-context/semantic/worker-context.json')
if not p.exists():
    print(json.dumps({'status':'not_yet_available'}));raise SystemExit(0)
s=p.lstat()
assert stat.S_ISREG(s.st_mode) and p.resolve(strict=True)==p and 0<s.st_size<=1048576
def identity(s):return s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns
with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
    assert identity(os.fstat(f.fileno()))==identity(s)
    raw=f.read(1048577)
    assert identity(os.fstat(f.fileno()))==identity(s)
assert identity(p.lstat())==identity(s) and len(raw)==s.st_size
payload=json.loads(raw);projection=payload['worker_projection']
keys=('max_capsules','selected_capsule_count','full_capsule_count','omitted_capsule_count','max_bytes')
counts={k:projection[k] for k in keys}
assert all(type(v)is int and 0<=v<2**31 for v in counts.values())
assert counts['selected_capsule_count']==len(payload['capsules'])<=counts['max_capsules']
assert counts['full_capsule_count']==counts['selected_capsule_count']+counts['omitted_capsule_count']
assert projection.get('semantic_equivalence_claimed') is False and projection.get('completion_authority') is False
assert type(projection['capsule_index_cid']) is str and re.fullmatch(r'[A-Za-z0-9:._-]{1,160}',projection['capsule_index_cid'])
canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
print(json.dumps(dict(status='observed',artifact_sha256=hashlib.sha256(raw).hexdigest(),
    semantic_sha256=hashlib.sha256(canonical).hexdigest(),artifact_bytes=len(raw),
    **counts,required_raw_paths=len(payload['raw_sources']),source_manifest_paths=len(payload['manifest']),
    capsule_index_cid=projection['capsule_index_cid'],source_changed_during_read=False,
    semantic_equivalence_claimed=projection.get('semantic_equivalence_claimed'),
    completion_authority=projection.get('completion_authority'))))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label',choices=LABELS)
    parser.add_argument('--container',required=True)
    args=parser.parse_args()
    lock=read(A/'pair-lock.json')
    row=next(row for row in lock['attempts'] if row['label']==args.label)
    trial=A/'live-trials'/f'grok-{args.label}-01'
    need(str(trial)==row['directory'] and (trial/'launch-start.json').exists(), 'launched campaign attempt required')
    need(not (trial/'initial-context-observation.json').exists(),'fresh observation required')
    # The caller selects the campaign-owned container from Docker metadata.
    need(len(args.container)==64 and all(c in '0123456789abcdef' for c in args.container),'full container ID required')
    command=['docker','exec','-i',args.container,'python3','-B','-I','-']
    p=subprocess.run(command,input=OBSERVE,text=True,capture_output=True,timeout=20)
    need(p.returncode==0 and not p.stderr and len(p.stdout.encode())<=8192,'bounded read-only observation failed')
    observed=json.loads(p.stdout)
    if observed['status']=='not_yet_available':
        print(json.dumps(observed));return
    result=dict(schema='capsule-budget-initial-context-observation@1',label=args.label,
        observed_at_utc=now(),container_id=args.container,observer_sha256=sha(__file__),
        pair_lock_sha256=sha(A/'pair-lock.json'),launch=binding(trial/'launch-start.json'),
        raw_content_exported=False,owner_state_read=False,provider_calls=0,
        observation=observed,dispatch_join_performed=False)
    write_new(trial/'initial-context-observation.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
