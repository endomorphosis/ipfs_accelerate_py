"""Real file/FD regressions for archived OWNED DATA; no native/health credit.

Link/UID/FD-race cases model only target-FD metadata, creating no OS aliases.
The separate original qualification integration must read the real Owner128
archive and unchanged H source; these fixtures do not impersonate that proof.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,os,stat
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as science


def archived(tmp_path,body=b'archived-data',mode=0o664):
    attempt=tmp_path/'attempt';attempt.mkdir()
    path=attempt/'exports/result.json';path.parent.mkdir();path.write_bytes(body);path.chmod(mode)
    root=attempt.stat();info=path.stat()
    binding=dict(path=str(attempt),device=root.st_dev,inode=root.st_ino)
    row=dict(path='exports/result.json',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),device=info.st_dev,inode=info.st_ino)
    return attempt,path,binding,row


@pytest.mark.parametrize('body,mode',[(b'',0o664),(b'x',0o664),(b'archived',0o600),(b'archived',0o400)])
def test_owned_archive_accepts_empty_and_group_writable_exact_manifest_data(tmp_path,body,mode):
    _,path,binding,row=archived(tmp_path,body,mode)
    before=path.stat()
    assert science._catalogue_race_owned_artifact(binding,row)==body
    after=path.stat()
    assert (after.st_ino,after.st_dev,stat.S_IMODE(after.st_mode))==(before.st_ino,before.st_dev,mode)


@pytest.mark.parametrize('field,value',[('bytes',0),('bytes',True),('sha256','0'*64),('device',1),('inode',1)])
def test_owned_archive_rejects_wrong_manifest_bytes_hash_or_inode(tmp_path,field,value):
    _,_,binding,row=archived(tmp_path)
    row[field]=value
    with pytest.raises(science.ExecutorHold,match='owned_data_'):
        science._catalogue_race_owned_artifact(binding,row)


@pytest.mark.parametrize('relative',['/tmp/outside','../outside','exports/../result.json','exports//result.json','./exports/result.json','exports/result.json/',''])
def test_owned_archive_rejects_nonliteral_or_escaping_relative_path(tmp_path,relative):
    _,_,binding,row=archived(tmp_path)
    row['path']=relative
    with pytest.raises(science.ExecutorHold,match='owned_data_manifest_row'):
        science._catalogue_race_owned_artifact(binding,row)


@pytest.mark.parametrize('mode',[0o602,0o666])
def test_owned_archive_rejects_world_writable_data(tmp_path,mode):
    _,_,binding,row=archived(tmp_path,mode=mode)
    with pytest.raises(science.ExecutorHold,match='owned_data_manifest_fd_identity_or_permissions'):
        science._catalogue_race_owned_artifact(binding,row)


@pytest.mark.parametrize('field',['device','inode','path'])
def test_owned_archive_requires_exact_released_attempt_identity(tmp_path,field):
    attempt,_,binding,row=archived(tmp_path)
    if field=='path':
        other=tmp_path/'other';other.mkdir();binding[field]=str(other)
    else:binding[field]+=1
    with pytest.raises(science.ExecutorHold,match='owned_data_attempt_identity'):
        science._catalogue_race_owned_artifact(binding,row)


@pytest.mark.parametrize('fault',['wrong_uid','hardlink_metadata','second_fd_stamp','path_stamp','midread_same_size_change','directory_replacement'])
def test_owned_archive_refuses_target_fd_metadata_identity_changes(tmp_path,monkeypatch,fault):
    _,path,binding,row=archived(tmp_path)
    original=science.os.fstat;calls=0
    def fstat(fd):
        nonlocal calls
        value=original(fd)
        if Path('/proc/self/fd/'+str(fd)).resolve()!=path:return value
        calls+=1
        values={name:getattr(value,name)for name in ('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
        if fault=='wrong_uid':values['st_uid']+=1
        elif fault=='hardlink_metadata':values['st_nlink']=2
        elif fault=='second_fd_stamp' and calls==2:values['st_ctime_ns']+=1
        return SimpleNamespace(**values)
    monkeypatch.setattr(science.os,'fstat',fstat)
    if fault=='path_stamp':
        lstat=Path.lstat
        def changed_lstat(selected,*args,**kwargs):
            value=lstat(selected,*args,**kwargs)
            if selected!=path:return value
            fields={name:getattr(value,name)for name in ('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
            fields['st_ino']+=1
            return SimpleNamespace(**fields)
        monkeypatch.setattr(Path,'lstat',changed_lstat)
    if fault in ('midread_same_size_change','directory_replacement'):
        read=science.os.read;changed=False
        def mutate_after_read(fd,size):
            nonlocal changed
            body=read(fd,size)
            if not changed and body and Path('/proc/self/fd/'+str(fd)).resolve()==path:
                changed=True
                if fault=='midread_same_size_change':path.write_bytes(b'X'*len(path.read_bytes()))
                else:
                    moved=path.parent.with_name('original-exports');path.parent.rename(moved)
                    path.parent.mkdir();path.write_bytes((moved/path.name).read_bytes());path.chmod(0o664)
            return body
        monkeypatch.setattr(science.os,'read',mutate_after_read)
    with pytest.raises(science.ExecutorHold,match='owned_data_'):
        science._catalogue_race_owned_artifact(binding,row)


def test_owned_archive_refuses_same_bytes_replaced_inode(tmp_path):
    _,path,binding,row=archived(tmp_path)
    replacement=path.with_name('replacement');replacement.write_bytes(path.read_bytes());replacement.chmod(0o664)
    os.replace(replacement,path)
    with pytest.raises(science.ExecutorHold,match='owned_data_manifest_fd_identity_or_permissions'):
        science._catalogue_race_owned_artifact(binding,row)


@pytest.mark.parametrize('size_offset',[0,1])
def test_owned_archive_enforces_exact_original_bound(tmp_path,size_offset):
    maximum=science.health_owner().MAX_JSON
    if size_offset==0:
        body=b'X'*maximum
        _,_,binding,row=archived(tmp_path,body=body)
        assert science._catalogue_race_owned_artifact(binding,row)==body
    else:
        _,_,binding,row=archived(tmp_path)
        row['bytes']=maximum+1
        with pytest.raises(science.ExecutorHold,match='owned_data_manifest_row'):
            science._catalogue_race_owned_artifact(binding,row)


@pytest.mark.parametrize('body,mode',[(b'',0o664),(b'owned',0o664)])
def test_archived_data_does_not_relax_original_H_code_reader(tmp_path,body,mode):
    _,path,binding,row=archived(tmp_path,body,mode)
    health=science.health_owner()
    with pytest.raises(health.ExecutorHealthError):health.descriptor(path)
    assert science._catalogue_race_owned_artifact(binding,row)==body


def test_owned_archive_uses_nofollow_for_every_intermediate_directory(tmp_path,monkeypatch):
    # MODEL only the kernel's intermediate-symlink ELOOP refusal; no OS alias is
    # created, so this gives no physical-alias/source-sampler qualification.
    import errno
    _,_,binding,row=archived(tmp_path)
    original=science.os.open;observed=[]
    def refuse_directory(path,flags,*args,**kwargs):
        if path=='exports' and kwargs.get('dir_fd') is not None:
            observed.append(flags)
            assert flags & os.O_DIRECTORY and flags & os.O_NOFOLLOW
            raise OSError(errno.ELOOP,'modeled intermediate symlink')
        return original(path,flags,*args,**kwargs)
    monkeypatch.setattr(science.os,'open',refuse_directory)
    with pytest.raises(OSError,match='modeled intermediate symlink'):
        science._catalogue_race_owned_artifact(binding,row)
    assert len(observed)==1
