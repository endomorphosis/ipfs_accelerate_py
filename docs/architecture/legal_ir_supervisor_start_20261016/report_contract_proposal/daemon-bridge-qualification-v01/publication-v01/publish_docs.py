"""Publish an exact additive documentation snapshot under genuine admission.

The original generic native owner covers this worker and its inherited-group
Git/gh children. No proposed runtime source is installed or executed.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat
import subprocess
import sys

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
REPO = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
BASE = 'f55c9597be0223fb0bd67334d0351cf5794d577f'
BRANCH = 'agent/legal-ir-ordinary-worker-20261016'
REMOTE = 'https://github.com/endomorphosis/ipfs_accelerate_py.git'
PREFIX = 'docs/architecture/legal_ir_supervisor_start_20261016/report_contract_proposal/daemon-bridge-qualification-v01'
RUNTIME = REPO/'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
RUNTIME_SHA = '23addb0dfc23e2cb093e7f74abbb7dc84a5308c24259e1eb0e25d721c0a80bed'


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum=16*1024**2):
    path = Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path,*path.parents)), 'unaliased_owned_path_required')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0,os.getuid()}
             and not before.st_mode&2 and before.st_size <= maximum, 'bounded_regular_owned_file_required')
        raw = stream.read(maximum+1)
        after = os.fstat(stream.fileno())
    witness=lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns,row.st_ctime_ns,row.st_mode,row.st_nlink)
    need(len(raw)==before.st_size and witness(before)==witness(after)==witness(path.stat()), 'original_source_changed')
    return raw


def command(argv, *, input=None, limit=32*1024**2):
    # Inherit the actual admitted worker's process group; no detached jobs.
    result=subprocess.run(argv,cwd=REPO,stdin=subprocess.DEVNULL if input is None else None,
        input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180,
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_CONFIG_NOSYSTEM='1'))
    need(len(result.stdout)+len(result.stderr)<=limit,'bounded_publication_command_output_required')
    need(result.returncode==0,'publication_command_failed:'+str(argv[1:3])+':'+result.stderr.decode(errors='replace')[-512:])
    return result.stdout


def git(*args,input=None):
    return command(['/usr/bin/git','-c','core.hooksPath=/dev/null',*args],input=input)


def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.exists() and not any(p.is_symlink()for p in(path,*path.parents)),'fresh_additive_document_required')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb')as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def main():
    need(len(sys.argv)==2,'exact_publication_packet_digest_required')
    packet_raw=read(HERE/'PACKET.json');need(sha(packet_raw)==sys.argv[1],'publication_packet_changed')
    packet=json.loads(packet_raw)
    need(packet['base_head']==BASE and packet['branch']==BRANCH and packet['prefix']==PREFIX
         and packet['github_qualification']=='skipped_by_user','reviewed_publication_scope_required')
    need(sha(read(RUNTIME))==RUNTIME_SHA,'original_runtime_source_changed')
    rows=packet['files'];need(0<len(rows)<=256,'bounded_publication_population_required')
    contents={}
    for row in rows:
        relative=PurePosixPath(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix()==row['path']
             and row['path'] not in contents,'canonical_unique_document_required')
        raw=read(ROOT/relative)
        need(len(raw)==row['bytes'] and sha(raw)==row['sha256'],'exact_publication_source_required')
        contents[row['path']]=raw
    need(sum(map(len,contents.values()))<=8*1024**2,'bounded_publication_bytes_required')
    need(git('symbolic-ref','--short','HEAD').decode().strip()==BRANCH
         and git('rev-parse','HEAD').decode().strip()==BASE,'original_owned_branch_head_required')
    need(git('status','--porcelain=v1','-z','--untracked-files=all')==b'','original_clean_owned_checkout_required')
    remote=git('ls-remote','--heads',REMOTE,'refs/heads/'+BRANCH).decode().strip().split()
    need(remote==[BASE,'refs/heads/'+BRANCH],'original_remote_branch_head_required')
    pr=json.loads(command(['/usr/bin/gh','api','repos/endomorphosis/ipfs_accelerate_py/pulls/292',
        '--jq','{state,draft,head_ref:.head.ref,head_sha:.head.sha}']))
    need(pr=={'state':'open','draft':True,'head_ref':BRANCH,'head_sha':BASE},'original_open_owned_draft_PR_required')
    need(not (REPO/PREFIX).exists(),'fresh_documentation_generation_required')
    for name,raw in contents.items():write(REPO/PREFIX/name,raw)
    names=sorted(PREFIX+'/'+name for name in contents)
    git('add','--sparse','-f','--',*names)
    staged=git('diff','--cached','--name-status','-z').split(b'\0')
    need(staged[-1]==b''and len(staged)==2*len(names)+1,'exact_staged_document_population_required')
    actual={}
    for i in range(0,len(staged)-1,2):
        need(staged[i]==b'A','publication_must_be_additive')
        actual[staged[i+1].decode()]=None
    need(set(actual)==set(names),'only_reviewed_documents_staged')
    # Verify all staged bytes through one genuine Git batch child.
    output=git('cat-file','--batch',input=''.join(':'+name+'\n'for name in names).encode())
    offset=0
    for name in names:
        end=output.index(b'\n',offset);header=output[offset:end].split();offset=end+1
        need(len(header)==3 and header[1]==b'blob','actual_staged_blob_required')
        size=int(header[2]);raw=output[offset:offset+size];offset+=size
        need(output[offset:offset+1]==b'\n'and raw==contents[name[len(PREFIX)+1:]],'actual_staged_document_bytes_differ');offset+=1
    need(offset==len(output),'exact_batch_blob_response_required')
    git('-c','user.name=Legal IR Engineering Agent','-c','user.email=legal-ir-agent@example.invalid',
        'commit','-m','docs: qualify Legal IR daemon bridge components and track activation gaps')
    head=git('rev-parse','HEAD').decode().strip()
    need(git('rev-parse','HEAD^').decode().strip()==BASE and len(head)==40,'exact_additive_commit_parent_required')
    committed=git('diff-tree','--no-commit-id','--name-status','-r','-z',head)
    need(committed==b''.join(b'A\0'+name.encode()+b'\0'for name in names),'exact_committed_document_paths_required')
    need(sha(read(RUNTIME))==RUNTIME_SHA and git('status','--porcelain=v1','-z','--untracked-files=all')==b'',
         'unchanged_runtime_and_clean_owned_checkout_required')
    git('push',REMOTE,'HEAD:refs/heads/'+BRANCH)
    remote=git('ls-remote','--heads',REMOTE,'refs/heads/'+BRANCH).decode().strip().split()
    need(remote==[head,'refs/heads/'+BRANCH],'exact_published_owned_head_required')
    result={'schema':'legal-ir-daemon-bridge-documentation-publication@1','passed':True,'base_head':BASE,'head':head,
        'branch':BRANCH,'draft_PR':'https://github.com/endomorphosis/ipfs_accelerate_py/pull/292',
        'document_prefix':PREFIX,'document_files':len(contents),'document_bytes':sum(map(len,contents.values())),
        'packet_sha256':sys.argv[1],'runtime_source_changed':False,'runtime_enabled':False,
        'github_qualification':'skipped_by_user','canonical_tasks_accepted':0,'acceptance_authority':False,
        'original_native_finalization_pending':True}
    attempt=Path(os.environ['TMPDIR']);fd=os.open(attempt/'publication.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w')as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    main()
