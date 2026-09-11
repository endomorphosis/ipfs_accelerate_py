"""Actual native owner/verifier custody, plus read-only negative admission."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest
from test.api.semantic_refactoring.test_spar_closeout_profile import population

HERE = Path(__file__).resolve()
ACCEL = HERE.parents[3]
DATASETS = Path(os.environ.get("SPAR_DATASETS_NATIVE_CHECKOUT", ACCEL / "ipfs_datasets_py"))
KIT = Path(os.environ.get("SPAR_KIT_NATIVE_CHECKOUT", ACCEL / "ipfs_kit_py"))


def git(root,*args):
    return subprocess.check_output(["/usr/bin/git","-c","core.hooksPath=/dev/null","-C",str(root),*args],stderr=subprocess.DEVNULL).decode().strip()


def commit(root):
    git(root,"init","-q"); git(root,"add",".")
    git(root,"-c","user.name=Qualification","-c","user.email=fixture@example.invalid","commit","-qm","real native source fixture")
    return git(root,"rev-parse","HEAD")


@pytest.fixture(scope="module")
def actual_native_report():
    # Native file locks/Quack/IPC are real. This qualification makes no disk
    # durability claim; all state belongs to this new explicit tmpfs directory.
    if os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize < 8 * 1024**3:
        pytest.fail("qualification tmpfs floor unavailable")
    directory = Path(tempfile.mkdtemp(prefix="spar-native-verifier-",dir="/dev/shm"))
    root = directory / "source"; root.mkdir()
    worker = "ipfs_accelerate_py/agent_supervisor/semantic_state/spar_verifier_worker.py"
    (root/worker).parent.mkdir(parents=True);shutil.copyfile(ACCEL/worker,root/worker)
    (root/"program.py").write_text("def add(a,b):\n    return a+b\n")
    datasets=root/"ipfs_datasets_py";relative=Path("ipfs_datasets_py/logic/software_contracts")
    (datasets/relative).parent.mkdir(parents=True)
    shutil.copytree(DATASETS/relative,datasets/relative,ignore=shutil.ignore_patterns("__pycache__"))
    dataset_head=commit(datasets)
    kit=root/"ipfs_kit_py";leaf=Path("ipfs_kit_py/mcp_server/mcplusplus");(kit/leaf).mkdir(parents=True)
    for name in ("coordination_storage.py","duckdb_coordination_storage.py","artifacts.py"):
        shutil.copyfile(KIT/leaf/name,kit/leaf/name)
    kit_head=commit(kit);commit(root)
    monkeypatch=pytest.MonkeyPatch()
    try:
        material,facts,snapshot,_=population.__wrapped__(monkeypatch)
    finally:monkeypatch.undo()
    material["nested_repositories"]=[
        dict(repository="ipfs_datasets",path="ipfs_datasets_py",planning_revision=dataset_head),
        dict(repository="ipfs_kit",path="ipfs_kit_py",planning_revision=kit_head)]
    (directory/"population.json").write_text(json.dumps([material,facts,snapshot]))
    process=subprocess.run([sys.executable,"-B",str(HERE.with_name("native_spar_verifier_qualification.py")),str(directory)],
        env={**os.environ,"PYTHONPATH":str(ACCEL),"PYTHONDONTWRITEBYTECODE":"1"},
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=300)
    assert process.returncode == 0, process.stdout + process.stderr + "\nfixture=" + str(directory)
    report=json.loads((directory/"qualification.json").read_text())
    report["fixture_directory"]=str(directory)
    return report


def test_actual_native_owner_and_independent_child(actual_native_report):
    r=actual_native_report
    assert r["owner_ready"] and r["kit"]["admitted"] and r["verified"]["admitted"],r
    e=r["execution"]
    assert e["process_identity"]["pid"] != e["process_identity"]["parent_pid"]
    assert e["process_identity"]["parent_pid"] != os.getpid()
    assert e["pidfd_exit_observed"] and e["sealed_result"] and e["returncode"] == 0


def test_real_component_stays_noncompleting(actual_native_report):
    r=actual_native_report
    assert not r["verified"]["accepted_root"] and not r["after"]["completion_authority"]
    assert len(r["verified"]["missing_coverage"]) == 7
    assert all(not goal["accepted"] for goal in r["after"]["goal_requirements"])
    assert "datasets_independent_accepted_root_producer_and_admission_required" in r["after"]["blockers"]


def test_replay_reuses_actual_execution(actual_native_report):
    r=actual_native_report
    assert r["replay"]["idempotent_replay"]
    assert r["replay"]["execution_cid"] == r["verified"]["execution_cid"]


@pytest.mark.parametrize("key",["read_only","rpc_writer_refused","goal_drift_refused","task_drift_refused",
    "source_drift_refused","forged_execution_refused","json_issuer_replay_refused","unknown_goals_preserved",
    "execution_boundary_drift_refused","stale_owner_refused","actual_owner_rollover_refused"])
def test_actual_native_admission_boundaries(actual_native_report,key):
    assert actual_native_report[key]
