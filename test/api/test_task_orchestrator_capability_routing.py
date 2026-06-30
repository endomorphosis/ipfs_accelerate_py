from ipfs_accelerate_py.p2p_tasks.orchestrator import OrchestratorConfig, TaskOrchestrator


class _Remote:
    def __init__(self, peer_id: str, multiaddr: str):
        self.peer_id = peer_id
        self.multiaddr = multiaddr


class _RegistryStub:
    def __init__(self, scores):
        self.scores = scores

    def score_peer_for_task(self, *, peer_id: str, task_type: str) -> float:
        return float(self.scores.get(peer_id, 0.0))


def test_orchestrator_claim_prioritizes_capability_registry(monkeypatch, tmp_path):
    queue_path = str(tmp_path / "tasks.duckdb")
    orchestrator = TaskOrchestrator(
        config=OrchestratorConfig(
            queue_path=queue_path,
            orchestrator_id="orch-test",
            base_worker_id="worker-test",
            min_workers=0,
            max_workers=0,
            mesh_peer_fanout=2,
            mesh_claim_batch=1,
        ),
        supported_task_types=["text-generation"],
    )

    remote_a = _Remote("peer-a", "/ip4/127.0.0.1/tcp/4001/p2p/peer-a")
    remote_b = _Remote("peer-b", "/ip4/127.0.0.1/tcp/4002/p2p/peer-b")

    monkeypatch.setattr(orchestrator, "_get_capability_registry", lambda: _RegistryStub({"peer-a": 1.0, "peer-b": 20.0}))

    call_order = []

    import ipfs_accelerate_py.p2p_tasks.client as client

    def _fake_claim_many_sync(*, remote, worker_id, supported_task_types, max_tasks, same_task_type, session_id, peer_id, clock):
        call_order.append(getattr(remote, "peer_id", ""))
        if getattr(remote, "peer_id", "") == "peer-b":
            return [
                {
                    "task_id": "remote-task-1",
                    "task_type": "text-generation",
                    "model_name": "model-a",
                    "payload": {"prompt": "hello"},
                }
            ]
        return []

    monkeypatch.setattr(client, "claim_many_sync", _fake_claim_many_sync)

    claimed = orchestrator._claim_from_peers(peers=[remote_a, remote_b], max_tasks=1)

    assert claimed
    assert call_order[0] == "peer-b"
    assert claimed[0][0].peer_id == "peer-b"
