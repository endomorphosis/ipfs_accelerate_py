"""Execute the installed worker parser with controlled boundary/router owners."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY


def _run(tmp_path, arguments):
    root = tmp_path / "deployment"
    root.mkdir()
    (root / "container-boundary.json").write_text('{"provider":"grok_cli"}')
    script = tmp_path / "worker.py"
    # Execute the real deployed argparse and argv forwarding. UID admission and
    # router execution are controlled here; actual isolation is a Docker check.
    prelude = r'''
import os,sys,types,json
os.getuid=lambda:1001
os.geteuid=lambda:1001
for name in ('ipfs_accelerate_py','ipfs_accelerate_py.agent_supervisor',
             'ipfs_accelerate_py.agent_supervisor.runtime','ipfs_accelerate_py.agent_supervisor.todo_daemon'):
 module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
boundary=types.ModuleType('ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary')
boundary.verify_container_worker_boundary=lambda **kwargs:{'controlled_boundary':True}
sys.modules[boundary.__name__]=boundary
router=types.ModuleType('ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner')
router.MAX_IMPLEMENTATION_TIMEOUT_SECONDS=600
router.main=lambda:print(json.dumps({'argv':sys.argv[1:]})) or 0
sys.modules[router.__name__]=router
custody=types.ModuleType('ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper')
custody.run_with_child_custody=lambda callback:callback()
sys.modules[custody.__name__]=custody
'''
    script.write_text(prelude + WORKER_ENTRY.replace("pathlib.Path('/opt/ipfs-supervisor')", "pathlib.Path(" + repr(str(root)) + ")"))
    return subprocess.run([sys.executable, "-I", str(script), *arguments], cwd=tmp_path,
                          input=b"", capture_output=True, timeout=20)


def test_worker_forwards_both_trusted_handoff_pins_to_router(tmp_path):
    result = _run(tmp_path, ["--provider", "grok_cli", "--model", "grok-4.7",
        "--semantic-repository", "/app", "--source-formalization-handoff", "/opt/ipfs-supervisor/worker-observations/exact.json",
        "--source-formalization-handoff-sha256", "a"*64])
    assert result.returncode == 0, result.stderr.decode()
    argv = json.loads(result.stdout)["argv"]
    assert argv[argv.index("--source-formalization-handoff")+1] == "/opt/ipfs-supervisor/worker-observations/exact.json"
    assert argv[argv.index("--source-formalization-handoff-sha256")+1] == "a"*64
    assert "--container-boundary" in argv and "--container-boundary-sha256" in argv


@pytest.mark.parametrize("change", ["missing_digest", "missing_path", "missing_repository", "planning", "preflight"])
def test_worker_rejects_partial_or_non_coding_owner_handoff(tmp_path, change):
    args = ["--semantic-repository", "/app", "--source-formalization-handoff", "/sealed.json",
            "--source-formalization-handoff-sha256", "a"*64]
    if change == "missing_digest":
        args = args[:-2]
    elif change == "missing_path":
        args = args[:2]+args[4:]
    elif change == "missing_repository":
        args = args[2:]
    elif change == "planning":
        args += ["--purpose", "planning"]
    else:
        args += ["--preflight"]
    result = _run(tmp_path, args)
    assert result.returncode != 0
    assert b"exact owner formalization handoff" in result.stderr
    assert not result.stdout


def test_worker_legacy_route_does_not_add_a_handoff(tmp_path):
    result = _run(tmp_path, ["--provider", "grok_cli", "--model", "grok-4.7"])
    assert result.returncode == 0, result.stderr.decode()
    assert not any("formalization-handoff" in arg for arg in json.loads(result.stdout)["argv"])


def test_worker_preserves_explicit_grok_bound_with_candidate_handoff(tmp_path):
    args = ["--provider", "grok_cli", "--model", "grok-4.7", "--semantic-repository", "/app",
        "--grok-coding-max-turns", "8",
        "--candidate-dispatch-handoff", "/opt/ipfs-supervisor/worker-observations/candidate.json",
        "--candidate-dispatch-handoff-sha256", "a"*64,
        "--candidate-round-descriptor", "/opt/ipfs-supervisor/worker-observations/descriptor.json",
        "--candidate-round-descriptor-sha256", "b"*64]
    result = _run(tmp_path, args)
    assert result.returncode == 0, result.stderr.decode()
    forwarded = json.loads(result.stdout)["argv"]
    for flag in ("--grok-coding-max-turns", "--candidate-dispatch-handoff",
                 "--candidate-dispatch-handoff-sha256", "--candidate-round-descriptor",
                 "--candidate-round-descriptor-sha256"):
        assert forwarded.count(flag) == 1
        assert forwarded[forwarded.index(flag)+1] == args[args.index(flag)+1]


def _numerical_args():
    return ["--semantic-repository", "/app", "--source-formalization-handoff", "/sealed.json",
        "--source-formalization-handoff-sha256", "a" * 64,
        "--numerical-obligation-descriptor", "/opt/ipfs-supervisor/worker-observations/numerical.json",
        "--numerical-obligation-descriptor-sha256", "b" * 64]


def test_worker_forwards_exact_numerical_descriptor_pair_alongside_owner_handoff(tmp_path):
    result = _run(tmp_path, _numerical_args())
    assert result.returncode == 0, result.stderr.decode()
    argv = json.loads(result.stdout)["argv"]
    for flag in ("--numerical-obligation-descriptor", "--numerical-obligation-descriptor-sha256",
                 "--source-formalization-handoff", "--source-formalization-handoff-sha256"):
        assert argv.count(flag) == 1
        assert argv[argv.index(flag)+1] == _numerical_args()[_numerical_args().index(flag)+1]


@pytest.mark.parametrize("change", ["missing_descriptor", "missing_digest", "empty_digest",
    "missing_handoff", "missing_handoff_digest", "missing_repository", "planning", "preflight", "finite"])
def test_worker_refuses_unbound_or_wrong_route_numerical_descriptor(tmp_path, change):
    args = _numerical_args()
    remove = {"missing_descriptor": "--numerical-obligation-descriptor",
        "missing_digest": "--numerical-obligation-descriptor-sha256",
        "missing_handoff": "--source-formalization-handoff",
        "missing_handoff_digest": "--source-formalization-handoff-sha256",
        "missing_repository": "--semantic-repository"}
    if change in remove:
        index = args.index(remove[change]); del args[index:index+2]
    elif change == "empty_digest":
        args[-1] = ""
    elif change == "planning":
        args += ["--purpose", "planning"]
    elif change == "preflight":
        args += ["--preflight"]
    else:
        args += ["--finite-repository-artifact", "/finite.json"]
    result = _run(tmp_path, args)
    assert result.returncode != 0 and not result.stdout
    assert b"exact numerical obligation descriptor" in result.stderr
