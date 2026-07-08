import time

import pytest


def _make_service_without_init(scheduler):
    # P2PWorkflowDiscoveryService.__init__ requires a working/authenticated GitHub CLI.
    # For unit tests we bypass __init__ and only set the fields needed by the merge helpers.
    from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService

    svc = P2PWorkflowDiscoveryService.__new__(P2PWorkflowDiscoveryService)
    svc.scheduler = scheduler
    return svc


def test_merge_workflow_payload_into_scheduler_adds_task():
    from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler

    scheduler = P2PWorkflowScheduler(peer_id="peer-test")
    svc = _make_service_without_init(scheduler)

    ok = svc._merge_workflow_payload_into_scheduler(
        {
            "owner": "octo",
            "repo": "demo",
            "workflow_id": "octo/demo",
            "workflow_name": "p2p.yml",
            "tags": ["p2p-only", "code-generation"],
            "priority": 8,
            "created_at": time.time(),
        }
    )

    assert ok is True
    assert "octo/demo/p2p.yml" in scheduler.pending_tasks


def test_merge_snapshot_payload_updates_peer_and_tasks():
    from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler, MerkleClock

    scheduler = P2PWorkflowScheduler(peer_id="peer-local")
    svc = _make_service_without_init(scheduler)

    remote_clock = MerkleClock(node_id="peer-remote")
    remote_clock.tick()

    merged = svc._merge_snapshot_payload_into_scheduler(
        {
            "peer_id": "peer-remote",
            "timestamp": time.time(),
            "merkle_clock": remote_clock.to_dict(),
            "pending_tasks": [
                {
                    "task_id": "octo/demo/p2p.yml",
                    "workflow_id": "octo/demo",
                    "name": "p2p.yml (octo/demo)",
                    "tags": ["p2p-only"],
                    "priority": 7,
                    "created_at": time.time(),
                    "task_hash": "",
                    "assigned_peer": None,
                }
            ],
        }
    )

    assert merged >= 1
    assert "peer-remote" in scheduler.known_peers
    assert "octo/demo/p2p.yml" in scheduler.pending_tasks


def test_merge_snapshot_ignores_self_peer_id():
    from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler

    scheduler = P2PWorkflowScheduler(peer_id="peer-self")
    svc = _make_service_without_init(scheduler)

    merged = svc._merge_snapshot_payload_into_scheduler({"peer_id": "peer-self", "pending_tasks": []})
    assert merged == 0
