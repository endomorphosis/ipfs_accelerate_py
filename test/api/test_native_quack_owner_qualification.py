"""Actual native owner, transport, claims and kernel-bound credential checks."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import (
    native_owner_session,
    qualify,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)


@pytest.fixture(autouse=True)
def require_installed_quack():
    report = probe_quack_capabilities()
    if not report.passes_health_check:
        pytest.skip(f"installed native Quack unavailable: {report.reason_code}")


def test_real_quack_tcp_and_typed_reservation_admission(tmp_path):
    output = tmp_path / "qualification"
    result = qualify(output)
    assert result["status"] == "passed"
    assert result["transport"] == "InProcessQuackTransport"
    assert result["fake_transport"] is False
    assert result["quack_tcp_read_verified"] is True
    assert result["typed_client_authenticated"] is True
    assert result["native_owner_ready"] is True
    assert result["native_owner_stopped"] is True
    assert result["ready_revision"] == 1
    assert result["reservation_revision"] == 2
    assert result["admission_revision"] == 3
    assert result["final_task_status"] == "in_progress"
    assert result["failing_acceptance_returncode"] != 0
    assert result["provider_calls"] == 0
    assert result["completion_authority_exercised"] is False
    assert result["duplicate_claim_refused"] is True
    assert json.loads((output / "qualification.json").read_text()) == result
    history = json.loads((output / "claim-history.json").read_text())
    operations = {
        row["revision"]: row["body"].get("completion_receipt", {}).get("operation")
        for row in history["revisions"]
    }
    assert operations[2] == "database_claim"
    assert operations[3] == "database_attempt_admitted"


def test_real_native_grant_cannot_be_reused_by_child_process(tmp_path):
    with native_owner_session(tmp_path / "owner") as session:
        # This intentionally presents the parent's real token from another
        # kernel peer. No token travels in argv or child output.
        child = subprocess.run(
            [sys.executable, "-c", """
import json, sys
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import QuackStateClient
request = json.load(sys.stdin)
client = QuackStateClient(owner_id=request['client_id'], store_id=request['store_id'],
                         process_birth_id=request['process_birth_id'])
try:
    client.attach(request['endpoint'], server_id=request['server_id'])
except Exception as error:
    print(json.dumps({'refused': True, 'reason': str(error.__cause__ or error)}))
else:
    print(json.dumps({'refused': False}))
finally:
    client.close()
"""],
            input=json.dumps({
                "client_id": session.credentials.client_id,
                "store_id": session.identity.store_id,
                "process_birth_id": session.credentials.process_birth_id,
                "endpoint": session.identity.listen_uri,
                "server_id": session.identity.server_id,
            }),
            env=dict(os.environ), text=True, capture_output=True, timeout=20,
            check=True,
        )
        assert session.credentials.token not in child.stdout + child.stderr
        response = json.loads(child.stdout)
        assert response["refused"] is True
        # The gateway closes the unauthorized channel before returning any
        # privileged diagnostic. The same token remains valid for its parent.
        assert response["reason"] == "typed state-owner channel closed"
        assert session.source.get_task(session.task_cid).status == "ready"


def test_native_session_restores_environment_and_stops_after_error(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_SOCKET_ENV,
        TYPED_STATE_OWNER_TOKEN_ENV,
    )

    keys = (TYPED_STATE_OWNER_SOCKET_ENV, TYPED_STATE_OWNER_TOKEN_ENV)
    before = {key: os.environ.get(key) for key in keys}
    with pytest.raises(RuntimeError, match="qualification interruption"):
        with native_owner_session(tmp_path / "owner") as session:
            assert session.server.ready()["ready"] is True
            raise RuntimeError("qualification interruption")
    assert session.server.status()["lifecycle"] == "stopped"
    assert {key: os.environ.get(key) for key in keys} == before
