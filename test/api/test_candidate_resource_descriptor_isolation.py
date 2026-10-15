"""A real bound child cannot inherit the private qualification capability.

The owner actually has the externally pinned sealed parent FD and routing
environment. The child reports booleans and the complete prompt digest only;
it never reads, serializes or logs parent capability contents. Native resource
lineage and Popen cleanup are genuine; deployment UID admission remains the
existing authored host seam, rather than a production container qualification.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import native_candidate_round_worker as worker
from test.api.candidate_resource_test_support import lease_record
from test.api.test_candidate_resource_pipeline import (
    candidate_case, candidate_resources, cleanup_actual, options, resource_parent,
)


PARENT_FD_ENV = "IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_FD"
PARENT_SHA_ENV = "IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_SHA256"
INSPECT_CHILD = r'''
import hashlib,json,os,sys
from pathlib import Path
raw=sys.stdin.buffer.read()
has_parent_fd=False
for path in Path('/proc/self/fd').iterdir():
    try:
        target=os.readlink(path)
    except OSError:
        # The descriptor used to enumerate /proc may already have closed.
        continue
    has_parent_fd=has_parent_fd or 'memfd:candidate-resource-qualification-parent' in target
observation=dict(has_parent_fd=has_parent_fd,
    has_parent_fd_env='IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_FD' in os.environ,
    has_parent_sha_env='IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_SHA256' in os.environ,
    prompt_sha256=hashlib.sha256(raw).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(observation,sort_keys=True))
'''


def test_real_bound_child_has_no_private_parent_fd_or_routing_environment(candidate_resources, tmp_path):
    value = candidate_resources
    # The resource_parent fixture already authenticates the independent pin,
    # closed envelope and native parent lineage. Verify actual owner possession
    # again without reading or exposing the capability bytes in this test.
    assert PARENT_FD_ENV in os.environ and PARENT_SHA_ENV in os.environ
    descriptor = int(os.environ[PARENT_FD_ENV])
    required_seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW
    assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals == required_seals
    assert "memfd:candidate-resource-qualification-parent" in os.readlink("/proc/self/fd/" + str(descriptor))
    selected, marker = options(value, tmp_path, cleanup=cleanup_actual)
    selected["command"] = [sys.executable, "-I", "-c", INSPECT_CHILD, str(marker)]
    assert PARENT_FD_ENV not in selected["environment"] and PARENT_SHA_ENV not in selected["environment"]
    leaf = value.binding.native_lease
    result = worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    assert result.returncode == 0 and value.gate.process.poll() == 0
    observed = json.loads(marker.read_bytes())
    assert set(observed) == {"has_parent_fd", "has_parent_fd_env", "has_parent_sha_env", "prompt_sha256"}
    assert all(observed[name] is False for name in ("has_parent_fd", "has_parent_fd_env", "has_parent_sha_env"))
    assert observed["prompt_sha256"] == hashlib.sha256(value.case.candidate_prompt.provider_prompt).hexdigest()
    assert leaf.released and lease_record(value.owner, leaf.lease_id) is None
    assert not value.owner.parent.released


@pytest.mark.parametrize("name", [PARENT_FD_ENV, PARENT_SHA_ENV])
def test_explicit_child_environment_refuses_private_resource_capability_names(name):
    selected = {"PATH":os.defpath, name:"authored-forbidden-routing-value"}
    with pytest.raises(ValueError, match="authority"):
        worker._environment(selected)
    assert selected[name] == "authored-forbidden-routing-value"
