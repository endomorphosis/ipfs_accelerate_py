"""Source-only publisher freeze after root finalizes its reviewable selection.

Bounded immutable file reads and one fresh manifest write. No Git, network,
product import, execution of historical requests or native admission.
"""
from pathlib import Path
import hashlib
import json
import os
import stat

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
REPO = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
PREVIOUS = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-settlement-20261017-01/publication-v01')
PREVIOUS_OBSERVATION = PREVIOUS.parent/'publication-root-observation-r01.json'
BASE = 'a1b1f32826e58913837308330d8ec6cc8589222a'
FILES = ('CONTRACT.json','EXISTING_REPORT_INVENTORY.json','README.txt','SELECTION.json',
         'build_manifest.py','build_packet.py','build_request.py','parent_attachment_gate.py','publish_docs.py')


def need(value,reason):
    if not value:raise RuntimeError(reason)


def read(path,maximum=16*1024**2):
    path=Path(path)
    need(path.is_absolute() and str(path)==os.path.abspath(path)
         and not any(item.is_symlink() for item in (path,*path.parents)), 'unaliased_publisher_source_required')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as stream:
        before=os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_uid in {0,os.getuid()}
             and not before.st_mode&2 and 0<=before.st_size<=maximum,'bounded_owned_publisher_source_required')
        raw=stream.read(maximum+1);after=os.fstat(stream.fileno())
    witness=lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mode,row.st_uid,row.st_gid,
                        row.st_nlink,row.st_mtime_ns,row.st_ctime_ns)
    need(len(raw)==before.st_size and witness(before)==witness(after)==witness(path.lstat()),
         'publisher_source_changed_during_read')
    return raw


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    need(not (HERE/'SOURCE_MANIFEST.json').exists() and not (HERE/'PACKET.json').exists()
         and not (ROOT/'publication-request-r01.json').exists(),'fresh_root_final_source_freeze_required')
    selection=json.loads(read(HERE/'SELECTION.json'))
    need(selection.get('schema')=='legal-ir-disposition-publication-selection@1'
         and selection.get('status')=='root_finalized','root_final_selection_required')
    need({'READINESS.md','TASKS.json'}<=set(selection['files'])
         and selection.get('root_observations'),'root_final_docs_and_native_observations_required')
    previous_manifest=json.loads(read(PREVIOUS/'SOURCE_MANIFEST.json'))
    previous_observation=json.loads(read(PREVIOUS_OBSERVATION))
    need(previous_observation.get('published_head')==BASE
         and previous_observation.get('native_finalized') is True
         and previous_observation.get('document_files_independently_verified')==87
         and previous_observation.get('preexisting_document_files_independently_preserved')==506
         and previous_observation.get('runtime_enabled') is False,
         'actual_previous_published_generation_context_required')
    inventory=json.loads(read(HERE/'EXISTING_REPORT_INVENTORY.json'))
    need(inventory['base_head']==BASE and inventory['file_count']==len(inventory['files'])==593
         and inventory['total_bytes']==16547750,'exact_593_file_readonly_inventory_required')
    encoded=[];names=set()
    for row in inventory['files']:
        path=Path(row['path'])
        need(not path.is_absolute() and '..' not in path.parts and '.git' not in path.parts
             and row['path'] not in names,'unique_existing_document_path_required')
        names.add(row['path'])
        raw=read(REPO/path)
        oid=hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
        need(len(raw)==row['bytes'] and sha(raw)==row['sha256'] and oid==row['blob_oid'],
             'existing_readonly_document_bytes_changed')
        encoded.append(b'100644 blob '+oid.encode()+b'\t'+row['path'].encode()+b'\0')
    need(sha(b''.join(encoded))==inventory['git_tree_raw_sha256'],'readonly_inventory_encoding_changed')
    rows=[]
    for name in FILES:
        raw=read(HERE/name);rows.append(dict(path=name,bytes=len(raw),sha256=sha(raw)))
    # Reuse the finite executed source/tool/native closure, not any historical
    # request's nested source_sha256 mapping. Those request files are inert docs.
    pins=dict(previous_manifest['input_pins'])
    for path in (PREVIOUS/'SOURCE_MANIFEST.json',PREVIOUS/'PACKET.json',
                 PREVIOUS/'publish_docs.py',PREVIOUS_OBSERVATION):
        pins[str(path)]=sha(read(path))
    need(not any('.git' in Path(name).parts for name in pins),'mutable_Git_metadata_input_pins_refused')
    manifest=dict(schema='legal-ir-daemon-disposition-documentation-publisher-source@1',files=rows,
        input_pins=dict(sorted(pins.items())),source_only_authoring=True,
        author_tests_executed=0,author_Git_or_network_jobs_executed=0,
        github_qualification='skipped_by_user',acceptance_authority=False,
        original_native_owner_required=True,packet_excluded_to_avoid_cycle=True,
        future_publication_request_and_result_excluded=True,root_final_docs_authorized=True,
        immutable_existing_document_inputs='593 directly enumerated ordinary documentation files/16547750bytes. Their embedded historical requests remain inert; no nested request input closure is expanded.',
        mutations='Only exact additive documentation on the existing owned draft branch; no runtime installation, canonical board/state/cap/counter change, PR merge or qualification.')
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    need(len(raw)<=1024**2,'bounded_publisher_manifest_required')
    fd=os.open(HERE/'SOURCE_MANIFEST.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(manifest_sha256=sha(raw),source_rows=len(rows),input_pins=len(pins),
                         packet_frozen=False,request_frozen=False),sort_keys=True))


if __name__=='__main__':main()
