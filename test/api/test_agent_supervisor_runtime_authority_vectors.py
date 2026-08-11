"""Independent vectors for content and process runtime authorities."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import profile_g_cid
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_dag_json,
    validate_cid,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import terminate_pid_tree
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import read_process_birth

KNOWN_RUNTIME_AUTHORITY_CID = "baguqeeracqtgb767f7cbbix3olmiskqv5bpva2aqp7darhmzwomawygz5ila"


def test_profile_g_runtime_authority_matches_real_cidv1_vector() -> None:
    artifact = {"schema": "runtime-authority@1", "fence": 7}

    assert profile_g_cid(artifact) == KNOWN_RUNTIME_AUTHORITY_CID
    assert cid_for_dag_json(artifact) == KNOWN_RUNTIME_AUTHORITY_CID
    assert (
        validate_cid(KNOWN_RUNTIME_AUTHORITY_CID, codecs=("dag-json",))
        == KNOWN_RUNTIME_AUTHORITY_CID
    )


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="process-birth authority vector requires Linux /proc",
)
def test_naturally_exited_direct_child_retains_empty_group_authority() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        birth = read_process_birth(child.pid)
        assert birth is not None
        assert birth.parent_pid == os.getpid()
        child.wait(timeout=3.0)

        assert terminate_pid_tree(
            child.pid,
            grace_seconds=0.0,
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=child.pid,
            expected_root_start_time_ticks=birth.start_time_ticks,
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1.0)
