"""Process-bound pipe credentials and exact public output over the real socket."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes
from test.api.test_current_source_lookup import _case, _payload, _data, DEFAULT_SOURCE


MODULE = "ipfs_accelerate_py.agent_supervisor.runtime.current_source_lookup_cli"


def _run_child(case, *, denial=None):
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    command = [sys.executable, "-m", MODULE, "--credential-fd", str(read_fd)]
    child_environment = os.environ.copy()
    child_environment.pop(typed.TYPED_STATE_OWNER_TOKEN_ENV, None)
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(read_fd,), env=child_environment)
    os.close(read_fd)
    grant = None
    try:
        pid = os.getpid() if denial == "wrong-process" else child.pid
        birth = typed.kernel_process_birth_id(pid)
        token, grant = case.server.issue_typed_client_grant_record(client_id="source-cli-child",
            process_birth_id=birth, peer_pid=pid, allowed_operations=("source.get",),
            entity_scopes={"task_cid": case.task_cid, "repository_id": case.request.repository_id,
                           "tree_id": case.request.expected_tree})
        # The token is minted after spawn and crosses only the anonymous pipe.
        assert token not in " ".join(command)
        assert token.encode() not in Path(f"/proc/{child.pid}/environ").read_bytes()
        assert token.encode() not in Path(f"/proc/{child.pid}/cmdline").read_bytes()
        if denial == "revoked":
            case.gateway.revoke_grant(grant.grant_id)
        credential = dict(schema="supervisor-source-lookup-cli-credential@1",
            socket_path=str(case.server.typed_command_socket_path()), token=token,
            client_id=grant.client_id, process_birth_id=birth, store_id=case.gateway.store_id)
        raw = canonical_json_bytes(credential) + b"\n"
        assert os.write(write_fd, raw) == len(raw)
        os.close(write_fd)
        write_fd = None
        stdout, stderr = child.communicate(input=canonical_json_bytes(_payload(case)) + b"\n", timeout=15)
        assert token.encode() not in stdout + stderr
        return child.returncode, stdout, stderr, grant
    finally:
        if write_fd is not None:
            os.close(write_fd)
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
        if grant is not None:
            case.gateway.revoke_grant(grant.grant_id)


def test_actual_child_pipe_grant_returns_one_exact_full_canonical_response(tmp_path):
    with _case(tmp_path) as case:
        code, stdout, stderr, grant = _run_child(case)
        assert code == 0 and stderr == b""
        response = json.loads(stdout)
        assert stdout == canonical_json_bytes(response) + b"\n"
        assert set(response) == {"schema", "request_id", "ok", "result"} and response["ok"] is True
        result = response["result"]
        assert result["response_bytes"] == len(stdout) <= case.binding["max_response_bytes"]
        assert result["response_wire_bytes"] == len(stdout) + 3
        assert result["caller_grant_id"] == grant.grant_id
        assert result["binding"]["task_cid"] == case.task_cid
        assert _data(response) == DEFAULT_SOURCE
        assert result["source_complete"] is True and result["working_source_verified"] is True
        assert result["completion_authority"] is False


@pytest.mark.parametrize("denial", ["wrong-process", "revoked"])
def test_child_cannot_use_parent_process_grant_or_revoked_pipe_credential(tmp_path, denial):
    with _case(tmp_path) as case:
        code, stdout, stderr, _ = _run_child(case, denial=denial)
        assert code == 1 and stdout == b""
        assert stderr == b"source_lookup_cli_refused\n"


def test_regular_file_descriptor_cannot_substitute_for_secret_pipe(tmp_path):
    path = tmp_path / "authored-credential.json"
    path.write_text(json.dumps(dict(schema="supervisor-source-lookup-cli-credential@1",
        socket_path="/tmp/unadmitted.sock", token="authored-dummy-pipe-token",
        client_id="authored-client", process_birth_id="authored-birth", store_id="authored-store")))
    with path.open("rb") as descriptor:
        result = subprocess.run([sys.executable, "-m", MODULE, "--credential-fd", str(descriptor.fileno())],
            pass_fds=(descriptor.fileno(),), input=b"{}\n", capture_output=True, timeout=15)
    assert result.returncode == 1 and result.stdout == b""
    assert result.stderr == b"source_lookup_cli_refused\n"
    assert b"authored-dummy-pipe-token" not in result.stdout + result.stderr
