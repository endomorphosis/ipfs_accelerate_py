"""EAAEF-172: in-process clean external handoff smoke.

Submit exported session + repository identity, disconnect, execute in-process
(not live containers), reattach, verify, typed terminal.
"""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import ExternalHandoffAPI
from ipfs_accelerate_py.agent_supervisor.api.external_run_handle import ExternalRunHandle
from ipfs_accelerate_py.agent_supervisor.runtime.external_fixed_point import terminate


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "release_smoke.json"
)

SOURCE_ROOT = "sha256:" + "e" * 64
SEMANTIC_ROOT = "sha256:" + "f" * 64


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_in_process_handoff_disconnect_execute_reattach_terminal() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "session_id": "session:exported-client",
            "repository_id": "repo:sha256-exported",
            "objective_id": "objective:release-smoke",
            "idempotency_key": "idem:smoke-1",
        }
    )
    assert started.verdict == "admitted"
    handle = ExternalRunHandle.from_receipt(started, api=api)
    handle.steer("execute in-process without live containers")
    snapshot = handle.detach()
    paused = api.pause(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "session_id": "session:exported-client",
        }
    )
    assert paused.run_status == "paused"
    assert started.run_id in api._runs

    attached = api.attach(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "session_id": "session:exported-client",
        }
    )
    assert attached.run_id == started.run_id
    assert attached.authority_id == started.authority_id
    resumed = api.resume(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "session_id": "session:exported-client",
        }
    )
    assert resumed.run_status == "running"
    restored = ExternalRunHandle.deserialize(snapshot, api=api)
    status = api.status(
        {
            "principal_id": "principal:operator",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
        }
    )
    assert status.run_status == "running"
    verified = api.report(
        {
            "principal_id": "principal:operator",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
        }
    )
    assert verified.reason_code == "reported"
    approved = api.approve(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "reviewer_principal_id": "principal:reviewer",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
        }
    )
    assert approved.run_status == "approved"
    terminal = terminate(
        goals_complete=True,
        tests_current=True,
        proofs_current=True,
        invalidations_empty=True,
        merge_queue_empty=True,
        claims_empty=True,
        source_root=SOURCE_ROOT,
        semantic_root=SEMANTIC_ROOT,
    )
    assert terminal["terminal"] == "completed"
    assert restored.run_id == started.run_id

    payload = _write_receipt(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-overlay-receipt@1",
            "task_id": "EAAEF-172",
            "evidence_mode": "contract_fail_closed",
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "live_containers": False,
            "session_id": started.session_id,
            "repository_id": started.repository_id,
            "run_id": started.run_id,
            "terminal": terminal["terminal"],
            "run_status": approved.run_status,
        }
    )
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_containers"] is False
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert payload["terminal"] == "completed"
