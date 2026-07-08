from ipfs_accelerate_py.p2p_tasks.capability_registry import PeerCapabilityRegistry


def test_peer_capability_registry_persists_and_scores(tmp_path):
    registry_path = str(tmp_path / "peer_capability_registry.json")
    registry = PeerCapabilityRegistry(path=registry_path)

    record = registry.upsert_from_status(
        peer_id="peer-a",
        multiaddr="/ip4/127.0.0.1/tcp/4001/p2p/peer-a",
        status={
            "ok": True,
            "session": "sess-1",
            "queued": 1,
            "running": 0,
            "queued_by_type": {"text-generation": 1},
            "capabilities": {
                "supported_task_types": ["text-generation", "embedding"],
                "loaded_models": ["model-a"],
            },
            "detail": {
                "runtime": {
                    "cuda_available": True,
                }
            },
        },
    )

    assert record is not None
    assert record.peer_id == "peer-a"
    assert "text-generation" in record.supported_tasks

    score_supported = registry.score_peer_for_task(peer_id="peer-a", task_type="text-generation")
    score_unsupported = registry.score_peer_for_task(peer_id="peer-a", task_type="vision-generation")
    assert score_supported > score_unsupported

    reloaded = PeerCapabilityRegistry(path=registry_path)
    loaded = reloaded.get_record("peer-a")
    assert loaded is not None
    assert loaded.session == "sess-1"
    assert loaded.loaded_models == ["model-a"]
