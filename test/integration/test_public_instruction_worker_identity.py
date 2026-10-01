"""Offline qualification of public instruction delivery across distinct UIDs.

Run this explicit qualification in the disposable Docker test image as root;
root only allocates a temporary directory and starts children. Admission and
instruction preparation run as UID1000, consumption as UID1001. No provider is
loaded, no external state is read, and both identities are unprivileged.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


def _owner(base):
    from test.api.test_agent_supervisor_local_planning_admission import scenario
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import prepare_public_instruction_context

    assert os.getuid() == os.geteuid() == 1000
    fixture = scenario.__wrapped__(base)
    values = next(fixture)
    try:
        root = values["repository"]
        admission = local.admit_local_benchmark_plan(graph=values["graph"], manifest=values["manifest"])
        raw = (root / "test_answer.py").read_bytes()
        context = prepare_public_instruction_context(repository=root, admission=admission,
            task_cid=values["task_cid"], source_path="test_answer.py",
            expected_source_sha256=hashlib.sha256(raw).hexdigest())
        workspace = base / "allocated"
        subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
        # These are authored empty denial sentinels, not owner key paths.
        private = base / "private-owner"
        private.mkdir(mode=0o700)
        (private / "sentinel").write_text("authored private state")
        (base / "public-request.json").write_text(json.dumps({"context": context,
            "workspace": str(workspace), "private": str(private), "repository": str(root)}))
    finally:
        fixture.close()


def _worker(base):
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction

    assert os.getuid() == os.geteuid() == 1001
    request = json.loads((base / "public-request.json").read_text())
    context = request["context"]
    block, receipt = load_public_instruction(artifact=Path(context["artifact"]),
        expected_sha256=context["sha256"], task_cid=context["task_cid"],
        prompt=json.dumps({"objective_id": "LOCAL-TASK"}), workspace=Path(request["workspace"]))
    assert "assert answer() == 2" in block
    denied = []
    for path in (Path(request["private"]) / "sentinel", Path(context["artifact"])):
        try:
            with path.open("ab"):
                pass
        except PermissionError:
            denied.append(str(path))
        else:
            raise AssertionError("worker can mutate owner evidence")
    with pytest.raises(PermissionError):
        (Path(request["private"]) / "sentinel").read_bytes()
    # Only public metadata is exported; no source body or signing state.
    print(json.dumps({"uid": os.getuid(), "manifest_signature_verified": receipt["manifest_signature_verified"],
        "source_freshness_verified": receipt["source_freshness_verified"], "source_bytes": receipt["source_bytes"],
        "denied_count": len(denied), "provider_calls": 0, "completion_authority": False}))


@pytest.mark.skipif(os.geteuid() != 0, reason="explicit disposable Docker cross-UID qualification")
def test_actual_owner_and_worker_uids_can_share_only_public_instruction_evidence():
    base = Path(tempfile.mkdtemp(prefix="instruction-uid-", dir="/tmp"))
    os.chown(base, 1000, 0)
    base.chmod(0o755)
    def child(uid, action):
        def drop():
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
        return subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), action, str(base)],
            env={**os.environ, "HOME": str(base)}, preexec_fn=drop, cwd="/",
            capture_output=True, text=True, timeout=90)
    try:
        owner = child(1000, "owner")
        assert owner.returncode == 0, owner.stderr[-4000:]
        worker = child(1001, "worker")
        assert worker.returncode == 0, worker.stderr[-4000:]
        result = json.loads(worker.stdout.splitlines()[-1])
        assert result["uid"] == 1001 and result["denied_count"] == 2
        assert result["manifest_signature_verified"] and result["source_freshness_verified"]
        assert result["provider_calls"] == 0 and result["completion_authority"] is False
    finally:
        shutil.rmtree(base)


if __name__ == "__main__":
    {"owner": _owner, "worker": _worker}[sys.argv[1]](Path(sys.argv[2]))
