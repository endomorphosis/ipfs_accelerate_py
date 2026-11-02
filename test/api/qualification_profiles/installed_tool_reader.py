"""Installed tool reads only; no tool/native/service invocation or authority.

The real-root executable positive is separate from private current-UID fixtures.
The ordinary project/control source reader retains its stricter ownership rule.
"""
from pathlib import Path
import hashlib
import os
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker

ROOT_TOOL_PIN = {'path': '/usr/bin/systemctl', 'bytes': 1585008, 'sha256': '121271332c5f1c95ad640ad676a9f271d1df0736641900a3828642d3596120fb'}
ROOT_TOOL = ROOT_TOOL_PIN['path']

def stamp(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_nlink,
            info.st_size,info.st_mtime_ns,info.st_ctime_ns)

def tool_pin(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def test_actual_root_owned_installed_tool_is_accepted_without_execution():
    path=Path(ROOT_TOOL);info=path.lstat();assert info.st_uid==0 and info.st_nlink==1 and info.st_mode&0o111
    pin=dict(ROOT_TOOL_PIN)
    assert tool_pin(path)==pin
    assert worker.checked_installed_tool_descriptor(pin)==pin
    assert stamp(path.lstat())==stamp(info)

def test_ordinary_source_reader_still_refuses_root_owned_tool():
    assert os.getuid()!=0
    with pytest.raises(ValueError,match='ordinary owner file'):
        worker.ordinary_bytes(Path(ROOT_TOOL),32*1024**2)

@pytest.mark.parametrize('failure',['boolean_size','zero_size','oversized','wrong_hash'])
def test_pinned_installed_tool_descriptor_refuses_invalid_original_identity(failure):
    pin=dict(ROOT_TOOL_PIN)
    if failure=='boolean_size':pin['bytes']=True
    elif failure=='zero_size':pin['bytes']=0
    elif failure=='oversized':pin['bytes']=32*1024**2+1
    else:pin['sha256']='f'*64
    with pytest.raises(ValueError):worker.checked_installed_tool_descriptor(pin)

def test_private_current_uid_executable_fixture_is_only_read(tmp_path):
    path=tmp_path/'never-run-tool';path.write_bytes(b'#!/bin/sh\nexit 99\n');path.chmod(0o700)
    pin=tool_pin(path);before=path.lstat()
    assert worker.checked_installed_tool_descriptor(pin)==pin
    assert stamp(path.lstat())==stamp(before) and path.read_bytes()==b'#!/bin/sh\nexit 99\n'

def test_private_nonexecutable_fixture_does_not_gain_tool_role(tmp_path):
    path=tmp_path/'not-executable';path.write_bytes(b'ordinary data\n');path.chmod(0o600)
    with pytest.raises(ValueError,match='executable identity'):
        worker.checked_installed_tool_descriptor(tool_pin(path))

@pytest.mark.parametrize('mutation',['opened_file','named_path'])
def test_actual_opened_or_named_file_change_is_not_a_stable_tool(tmp_path,monkeypatch,mutation):
    path=tmp_path/'changing-tool';path.write_bytes(b'#!/bin/sh\nexit 99\n');path.chmod(0o700)
    pin=tool_pin(path);identity=(path.stat().st_dev,path.stat().st_ino)
    actual_read=os.read;changed=[False]
    def mutate_after_original_read(fd,count):
        raw=actual_read(fd,count);info=os.fstat(fd)
        if raw and not changed[0] and (info.st_dev,info.st_ino)==identity:
            changed[0]=True
            if mutation=='opened_file':path.write_bytes(b'#!/bin/sh\nexit 98\n')
            else:
                path.rename(tmp_path/'retained-opened-file')
                path.write_bytes(b'#!/bin/sh\nexit 99\n');path.chmod(0o700)
        return raw
    with monkeypatch.context() as patch:
        patch.setattr(worker.os,'read',mutate_after_original_read)
        with pytest.raises(ValueError,match='changed'):
            worker.checked_installed_tool_descriptor(pin)
    assert changed[0]
