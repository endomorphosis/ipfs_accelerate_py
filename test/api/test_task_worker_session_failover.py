import threading


def test_worker_session_failover_resubmits_locally_with_transcript(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks.cache_store import DiskTTLCache
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    # Enable copilot_cli tasks but stub actual LLM call.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SESSION", "S_A")

    # Enable worker-side failover and make it immediate for the test.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SESSION_FAILOVER", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SESSION_FAILOVER_AFTER_S", "0")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SESSION_FAILOVER_SCAN_INTERVAL_S", "0")

    # Enable cache and isolate it to tmp_path.
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR", str(cache_dir))

    import ipfs_accelerate_py.llm_router as llm_router

    def fake_generate_text(prompt: str, *args, **kwargs):
        # Return deterministic output to assert completion.
        return "ok"

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    # Seed transcript for the chat session.
    cache = DiskTTLCache(cache_dir)
    cache.set(
        "chat_history:chat-1",
        {"text": "User: hi\nAssistant: hello", "updated_at": 0.0},
        ttl_s=3600,
    )

    queue_path = str(tmp_path / "q.duckdb")
    q = TaskQueue(queue_path)

    # Submit a task that requires a different session; it should failover.
    tid_old = q.submit(
        task_type="llm.generate",
        model_name="gpt-5-mini",
        payload={
            "prompt": "next",
            "provider": "copilot_cli",
            "session_id": "S_B",
            "chat_session_id": "chat-1",
        },
    )

    t = threading.Thread(
        target=lambda: run_worker(
            queue_path=queue_path,
            worker_id="w1",
            once=True,
            supported_task_types=["llm.generate"],
        ),
        daemon=True,
    )
    t.start()
    t.join(timeout=10)

    old_row = q.get(tid_old)
    assert old_row is not None
    assert old_row["status"] in {"cancelled", "queued"}

    # Find the new, completed task.
    rows = q.list(limit=50)
    completed = [r for r in rows if r.get("status") == "completed" and (r.get("result") or {}).get("text") == "ok"]
    assert completed, "Expected a completed failover task"

    new_task = completed[-1]
    payload = new_task.get("payload") or {}
    assert payload.get("session_id") == "S_A"
    assert payload.get("chat_session_id") == "chat-1"

    # Prompt should include transcript (best-effort) when available.
    assert "Previous chat transcript" in str(payload.get("prompt") or "")
    assert "User: hi" in str(payload.get("prompt") or "")
    assert "Assistant: hello" in str(payload.get("prompt") or "")
    assert "User: next" in str(payload.get("prompt") or "")

    # Worker should append the new turn to the transcript.
    updated = cache.get("chat_history:chat-1")
    assert isinstance(updated, dict)
    assert "User: next" in str(updated.get("text") or "")
    assert "Assistant: ok" in str(updated.get("text") or "")
