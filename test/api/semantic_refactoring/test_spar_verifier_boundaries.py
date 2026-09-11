"""Kernel/source boundary negatives, without a live board or fake issuer."""
from __future__ import annotations
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state import spar_native_verification as native


def test_sealed_input_rejects_actual_write_and_truncate():
    fd=native._sealed_fd("spar-seal-test",b"bound input")
    try:
        with pytest.raises(PermissionError):os.pwrite(fd,b"forged",0)
        with pytest.raises(PermissionError):os.ftruncate(fd,0)
        assert os.pread(fd,11,0)==b"bound input"
    finally:os.close(fd)


def test_parent_refuses_actual_child_forged_handshake():
    source=b"import os,json,sys\nprint(json.dumps({'pid':os.getpid()+1}),flush=True)\nsys.stdin.readline()\n"
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,"w") as output:output.writestr("__main__.py",source)
    with pytest.raises(native.NativeVerificationUnavailable,match="process_binding_changed"):
        native.execute_verifier({},archive.getvalue())


@pytest.mark.parametrize("changed",["worker","datasets"])
def test_committed_module_capture_refuses_working_byte_drift(changed):
    root=Path(tempfile.mkdtemp(prefix="spar-module-capture-",dir="/dev/shm"))
    worker=root/native.WORKER;worker.parent.mkdir(parents=True)
    worker.write_text("# fixed worker fixture; no execution in this capture test\n")
    dataset=root/"ipfs_datasets_py"
    module=dataset/"ipfs_datasets_py/logic/software_contracts/semantic_state/spar_verification.py"
    module.parent.mkdir(parents=True);module.write_text("# fixed verifier fixture; no execution\n")
    def git(p,*args):return subprocess.check_output(["git","-C",str(p),*args],stderr=subprocess.DEVNULL).decode().strip()
    def commit(p):
        git(p,"init","-q");git(p,"add",".");git(p,"-c","user.name=Test","-c","user.email=t@example.invalid","commit","-qm","source")
        return git(p,"rev-parse","HEAD")
    dataset_head=commit(dataset);head=commit(root)
    source={"source_forest":{"source_head":head,"nested_repositories":[dict(repository="ipfs_datasets",path="ipfs_datasets_py",head=dataset_head)]}}
    path=worker if changed=="worker" else module
    path.write_text(path.read_text()+"# changed bytes\n")
    with pytest.raises(native.NativeVerificationUnavailable,match="source_changed"):
        native.capture_verifier_archive(root,source)
