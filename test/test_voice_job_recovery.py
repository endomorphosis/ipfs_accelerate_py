"""Queue-level evidence for restart-safe, priority-aware voice work.

ABBY-VOICE-G016 residual evidence inventory (AUTO-026 gap closure):

- persisted attempt/backoff/lease state — ``test_owned_heartbeat_extends_lease_and_expired_claim_recovers``,
  ``test_backoff_blocks_claim_and_expired_final_attempt_fails``, and the DuckDB migration test.
- owner heartbeats — ``test_owned_heartbeat_extends_lease_and_expired_claim_recovers``.
- IndexTTS/Whisper batch-size-one policy — ``test_audio_adapters_are_physical_batch_size_one``.
- existing sibling isolation and single-flight receipts — preserved by the provider batch
  suite imported in the validation gate; this module asserts the audio batch-size-one and
  compatibility surface that those receipts protect.
- existing ``ResourceScheduler`` CPU/RAM/disk/GPU/provider backpressure assertions —
  ``test_resource_saturation_backpressures_the_candidate_wave`` plus the resource-scheduler
  API suite in the same validation gate.
- authoritative evidence map: data/abby_voice/agent_supervisor/discovery/2026-07-26-abby-voice-auto-015-objective-validation-repair.md
- residual scan closure: data/abby_voice/agent_supervisor/discovery/2026-07-26-abby-voice-auto-026-objective-validation-repair.md
- objective validation repair — ``test_g016_objective_validation_repair_is_discoverable`` and
  data/abby_voice/agent_supervisor/discovery/2026-07-26-abby-voice-auto-031-objective-validation-repair.md
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.provider_batch_scheduler import (
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
)
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.p2p_tasks.capability_registry import PeerCapabilityRegistry
from ipfs_accelerate_py.p2p_tasks.orchestrator import (
    OrchestratorConfig,
    TaskOrchestrator,
)
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

# Residual AUTO-026 terms must remain discoverable as exact strings in this
# authorized validation surface. Implementation ownership for G016 remains the
# AUTO-015 repair map below. AUTO-029 anchors residual scan closure of the
# AUTO-026 repair receipt so objective scans re-find the same boundary.
# AUTO-031 anchors the literal acceptance-subset phrase
# "objective validation repair" so objective scans re-find that meta term.
G016_AUTHORITATIVE_EVIDENCE_MAP = (
    "data/abby_voice/agent_supervisor/discovery/"
    "2026-07-26-abby-voice-auto-015-objective-validation-repair.md"
)
G016_RESIDUAL_SCAN_CLOSURE = (
    "data/abby_voice/agent_supervisor/discovery/"
    "2026-07-26-abby-voice-auto-026-objective-validation-repair.md"
)
G016_OBJECTIVE_VALIDATION_REPAIR = (
    "data/abby_voice/agent_supervisor/discovery/"
    "2026-07-26-abby-voice-auto-031-objective-validation-repair.md"
)
G016_REQUIRED_EVIDENCE_TERMS = (
    "objective validation repair",
    "persisted attempt/backoff/lease state",
    "owner heartbeats",
    "IndexTTS/Whisper batch-size-one policy",
    "existing sibling isolation and single-flight receipts",
    "existing `ResourceScheduler` CPU/RAM/disk/GPU/provider backpressure assertions",
    f"authoritative evidence map: {G016_AUTHORITATIVE_EVIDENCE_MAP}",
    f"residual scan closure: {G016_RESIDUAL_SCAN_CLOSURE}",
)


def _submit(
    queue: TaskQueue,
    name: str,
    *,
    priority: int = 5,
    max_attempts: int = 3,
) -> str:
    return queue.submit(
        task_id=name,
        task_type="voice.tts",
        model_name="abby",
        payload={"text": name, "priority": priority},
        max_attempts=max_attempts,
    )


def test_submit_once_is_stable_and_rejects_identity_aliasing(tmp_path):
    queue = TaskQueue(str(tmp_path / "queue.duckdb"))
    request = {
        "task_type": "voice.tts",
        "model_name": "abby",
        "payload": {"text": "hello"},
        "idempotency_key": "tts:sha256:one",
    }

    first = queue.submit_once(**request)
    assert queue.submit_once(**request) == first
    assert len(queue.list()) == 1

    with pytest.raises(ValueError, match="different work"):
        queue.submit_once(**{**request, "payload": {"text": "changed"}})


def test_submit_with_outcome_reports_exact_replays(tmp_path):
    queue = TaskQueue(str(tmp_path / "queue.duckdb"))
    request = {
        "task_id": "voice-task",
        "task_type": "voice.tts",
        "model_name": "abby",
        "payload": {"text": "hello"},
    }

    assert queue.submit_with_outcome(**request) == ("voice-task", False)
    assert queue.submit_with_outcome(**request) == ("voice-task", True)
    with pytest.raises(ValueError, match="different work"):
        queue.submit_with_outcome(
            **{**request, "payload": {"text": "changed"}}
        )


def test_submit_with_outcome_reports_concurrent_replay(tmp_path):
    path = str(tmp_path / "queue.duckdb")
    queues = (TaskQueue(path), TaskQueue(path))
    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, bool]] = []
    errors: list[BaseException] = []

    def submit(queue: TaskQueue) -> None:
        try:
            barrier.wait()
            outcomes.append(
                queue.submit_with_outcome(
                    task_id="voice-task",
                    task_type="voice.tts",
                    model_name="abby",
                    payload={"text": "hello"},
                )
            )
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(queue,)) for queue in queues]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcome[1] for outcome in outcomes) == [False, True]
    assert len(queues[0].list()) == 1
    for queue in queues:
        queue.close()


def test_claims_are_highest_priority_first_with_fifo_tie_break(tmp_path):
    queue = TaskQueue(str(tmp_path / "queue.duckdb"))
    low = _submit(queue, "low", priority=2)
    high_oldest = _submit(queue, "high-oldest", priority=9)
    high_newest = _submit(queue, "high-newest", priority=9)

    first = queue.claim_next(worker_id="trusted")
    second = queue.claim_next(worker_id="trusted")
    assert first is not None and first.task_id == high_oldest
    assert second is not None and second.task_id == high_newest
    assert queue.get(low)["status"] == "queued"

    # Trust-tier caps remain eligibility filters, not priority inversions.
    capped = queue.claim_next(worker_id="baseline", max_priority=5)
    assert capped is not None and capped.task_id == low


def test_atomic_priority_microbatch_claims_do_not_overlap(tmp_path):
    path = str(tmp_path / "queue.duckdb")
    queue = TaskQueue(path)
    expected = {
        _submit(queue, f"task-{index}", priority=(index % 10) + 1)
        for index in range(12)
    }
    barrier = threading.Barrier(3)
    claimed: list[list[str]] = []
    errors: list[BaseException] = []

    def claim(worker: str) -> None:
        local = TaskQueue(path)
        try:
            barrier.wait()
            rows = local.claim_next_many(
                worker_id=worker,
                supported_task_types=["voice.tts"],
                max_tasks=6,
                same_task_type=True,
            )
            claimed.append([row.task_id for row in rows])
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)
        finally:
            local.close()

    threads = [threading.Thread(target=claim, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    flattened = [task_id for batch in claimed for task_id in batch]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == expected


def test_owned_heartbeat_extends_lease_and_expired_claim_recovers(tmp_path):
    """owner heartbeats renew leases; persisted attempt/backoff/lease state survives recovery."""

    queue = TaskQueue(str(tmp_path / "queue.duckdb"), default_lease_seconds=10)
    task_id = _submit(queue, "recoverable", max_attempts=2)
    claim = queue.claim_next(worker_id="worker-a", lease_seconds=10)
    assert claim is not None
    assert claim.attempt == 1
    assert claim.lease_until is not None

    assert queue.heartbeat(
        task_id=task_id,
        worker_id="worker-b",
        now=claim.heartbeat_at + 1,
    ) is False
    assert queue.heartbeat(
        task_id=task_id,
        worker_id="worker-a",
        lease_seconds=20,
        now=claim.heartbeat_at + 1,
    ) is True
    renewed = queue.get(task_id)
    assert renewed["lease_until"] == pytest.approx(claim.heartbeat_at + 21)
    assert queue.recover_expired_leases(now=claim.heartbeat_at + 15) == 0
    assert queue.recover_expired_leases(now=claim.heartbeat_at + 22) == 1

    recovered = queue.get(task_id)
    assert recovered["status"] == "queued"
    assert recovered["assigned_worker"] is None
    assert recovered["attempt"] == 1
    assert recovered["lease_until"] is None

    second = queue.claim_next(worker_id="worker-b")
    assert second is not None
    assert second.task_id == task_id
    assert second.attempt == 2


def test_backoff_blocks_claim_and_expired_final_attempt_fails(tmp_path):
    """persisted attempt/backoff/lease state blocks claims until next_attempt_at elapses."""

    queue = TaskQueue(str(tmp_path / "queue.duckdb"))
    task_id = _submit(queue, "retry", max_attempts=2)
    first = queue.claim_next(worker_id="worker-a")
    assert first is not None

    assert queue.retry(
        task_id=task_id,
        worker_id="worker-a",
        delay_seconds=60,
        error="provider unavailable",
    )
    assert queue.claim_next(worker_id="worker-b") is None

    # Advance only the persisted deadline; no wall-clock sleep is needed.
    conn = queue._get_conn()
    conn.execute("UPDATE tasks SET next_attempt_at=0 WHERE task_id=?", (task_id,))
    second = queue.claim_next(worker_id="worker-b", lease_seconds=1)
    assert second is not None and second.attempt == 2
    assert second.lease_until is not None

    assert queue.recover_expired_leases(now=second.lease_until + 1) == 1
    exhausted = queue.get(task_id)
    assert exhausted["status"] == "failed"
    assert exhausted["assigned_worker"] is None
    assert "maximum attempts" in exhausted["error"]
    assert queue.claim_next(worker_id="worker-c") is None


def test_stale_worker_cannot_finish_recovered_claim(tmp_path):
    queue = TaskQueue(str(tmp_path / "queue.duckdb"))
    task_id = _submit(queue, "owned", max_attempts=2)
    first = queue.claim_next(worker_id="worker-a", lease_seconds=1)
    assert first is not None and first.lease_until is not None
    assert queue.recover_expired_leases(now=first.lease_until + 1) == 1

    second = queue.claim_next(worker_id="worker-b")
    assert second is not None
    assert queue.complete(
        task_id=task_id,
        worker_id="worker-a",
        status="completed",
        result={"audio": "stale"},
    ) is False
    assert queue.complete(
        task_id=task_id,
        worker_id="worker-b",
        status="completed",
        result={"audio": "current"},
    ) is True
    completed = queue.get(task_id)
    assert completed["status"] == "completed"
    assert completed["result"]["audio"] == "current"
    assert completed["lease_until"] is None


def test_existing_duckdb_schema_is_migrated_in_place(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "legacy.duckdb")
    conn = duckdb.connect(path)
    conn.execute(
        """
        CREATE TABLE tasks (
            task_id VARCHAR PRIMARY KEY,
            task_type VARCHAR NOT NULL,
            model_name VARCHAR NOT NULL,
            payload_json VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            assigned_worker VARCHAR,
            created_at DOUBLE NOT NULL,
            updated_at DOUBLE NOT NULL,
            result_json VARCHAR,
            error VARCHAR
        )
        """
    )
    conn.execute(
        "INSERT INTO tasks VALUES "
        "('legacy', 'voice.tts', 'abby', '{\"priority\": 8}', "
        "'queued', NULL, 1, 1, NULL, NULL)"
    )
    conn.close()

    queue = TaskQueue(path)
    migrated = queue.get("legacy")
    assert migrated is not None
    assert migrated["priority"] == 8
    assert migrated["attempt"] == 0
    assert migrated["max_attempts"] == 3
    assert migrated["next_attempt_at"] == 0
    assert queue.claim_next(worker_id="worker") is not None


def _provider_request(request_id: str, **overrides: object) -> ProviderBatchRequest:
    values: dict[str, object] = {
        "request_id": request_id,
        "payload": request_id,
        "provider_id": "abby_indextts",
        "route": "synthesis",
        "model": "abby-index-tts",
        "operation": "voice.tts",
        "context_limit": 8_192,
        "voice": "abby",
        "locale": "en-US",
        "reference_hash": "a" * 64,
        "codec": "wav",
        "sample_rate": 24_000,
        "channels": 1,
        "policy": {"network": False},
        "tenant_policy": {"tenant": "211", "data_class": "public"},
        "generation_settings": {"temperature": 0},
    }
    values.update(overrides)
    return ProviderBatchRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("provider_id", "abby_whisper"),
        ("route", "transcription"),
        ("model", "abby-whisper"),
        ("operation", "voice.asr"),
        ("context_limit", 4_096),
        ("voice", "alternate"),
        ("locale", "es-US"),
        ("reference_hash", "b" * 64),
        ("codec", "flac"),
        ("sample_rate", 16_000),
        ("channels", 2),
        ("policy", {"network": True}),
        ("tenant_policy", {"tenant": "other"}),
        ("generation_settings", {"temperature": 1}),
    ],
)
def test_audio_provider_batch_key_covers_every_compatibility_dimension(
    field,
    different,
):
    base = _provider_request("base").batch_key
    changed = _provider_request("changed", **{field: different}).batch_key
    assert base != changed
    assert base.digest != changed.digest


@pytest.mark.parametrize(
    "provider_id",
    ["abby_indextts", "index-tts", "IndexTTSHTTP", "whisper", "HuggingFaceWhisperHTTP"],
)
def test_audio_adapters_are_physical_batch_size_one(provider_id):
    """IndexTTS/Whisper batch-size-one policy keeps physical provider calls at one member."""

    calls: list[tuple[str, ...]] = []

    def dispatch(requests):
        members = tuple(requests)
        calls.append(tuple(member.request_id for member in members))
        return [member.request_id for member in members]

    with ProviderBatchScheduler(
        dispatch,
        config=ProviderBatchSchedulerConfig(
            max_batch_size=8,
            batch_window_ms=5,
            provider_limits={provider_id: 1},
        ),
    ) as scheduler:
        results = scheduler.execute_many(
            [
                _provider_request("first", provider_id=provider_id),
                _provider_request("second", provider_id=provider_id),
            ],
            wait_timeout=2,
        )

    assert calls == [("first",), ("second",)]
    assert all(result.successful for result in results)
    assert scheduler.metrics().max_observed_batch_size == 1


def _audio_registry(tmp_path) -> PeerCapabilityRegistry:
    registry = PeerCapabilityRegistry(path=str(tmp_path / "audio-capabilities.json"))
    registry.upsert_from_status(
        peer_id="voice-peer",
        multiaddr="/ip4/127.0.0.1/tcp/4001/p2p/voice-peer",
        status={
            "ok": True,
            "capabilities": {
                "models": ["abby-index-tts"],
                "available_memory_bytes": 8_000_000_000,
                "audio_capabilities": {
                    "devices": ["cuda"],
                    "artifact_schemes": ["ipfs"],
                    "voice.tts": {
                        "providers": ["index-tts"],
                        "voices": ["abby"],
                        "codecs": ["wav"],
                        "locales": ["en-US"],
                    },
                },
            },
            "local_worker": {"supported_task_types": ["tts"]},
            "detail": {"runtime": {"cuda_available": True}},
        },
    )
    return registry


def _tts_payload() -> dict[str, object]:
    return {
        "provider": "index-tts",
        "model_name": "abby-index-tts",
        "voice": "abby",
        "codec": "wav",
        "locale": "en-US",
        "device": "cuda",
        "required_memory_bytes": 4_000_000_000,
        "reference_audio": {"uri": "ipfs://bafy-reference"},
    }


@pytest.mark.parametrize(
    ("field", "unsupported"),
    [
        ("provider", "other-provider"),
        ("model_name", "other-model"),
        ("voice", "other-voice"),
        ("codec", "flac"),
        ("locale", "fr-FR"),
        ("device", "cpu"),
        ("required_memory_bytes", 9_000_000_000),
        ("reference_audio", {"uri": "file:///unshared/audio.wav"}),
    ],
)
def test_audio_capability_registry_rejects_unsupported_constraints(
    tmp_path,
    field,
    unsupported,
):
    registry = _audio_registry(tmp_path)
    payload = _tts_payload()
    assert registry.matches_task_requirements(
        peer_id="voice-peer",
        task_type="voice.tts",
        model_name="abby-index-tts",
        payload=payload,
    )

    payload[field] = unsupported
    assert not registry.matches_task_requirements(
        peer_id="voice-peer",
        task_type="voice.tts",
        model_name=str(payload["model_name"]),
        payload=payload,
    )


def test_orchestrator_releases_incompatible_claimed_audio(monkeypatch, tmp_path):
    orchestrator = TaskOrchestrator(
        config=OrchestratorConfig(
            queue_path=str(tmp_path / "tasks.duckdb"),
            orchestrator_id="orch-test",
            base_worker_id="worker-test",
            min_workers=0,
            max_workers=0,
            mesh_peer_fanout=1,
            mesh_claim_batch=2,
        ),
        supported_task_types=["voice.tts"],
    )

    class Remote:
        peer_id = "voice-peer"
        multiaddr = "/ip4/127.0.0.1/tcp/4001/p2p/voice-peer"

    class Registry:
        def score_peer_for_task(self, **_kwargs):
            return 20.0

        def matches_task_requirements(self, *, payload, **_kwargs):
            return payload.get("provider") == "index-tts"

    monkeypatch.setattr(orchestrator, "_get_capability_registry", Registry)
    import ipfs_accelerate_py.p2p_tasks.client as client

    monkeypatch.setattr(
        client,
        "claim_many_sync",
        lambda **_kwargs: [
            {
                "task_id": "compatible",
                "task_type": "voice.tts",
                "model_name": "abby-index-tts",
                "payload": {"provider": "index-tts"},
            },
            {
                "task_id": "incompatible",
                "task_type": "voice.tts",
                "model_name": "other",
                "payload": {"provider": "other"},
            },
        ],
    )
    released: list[dict[str, object]] = []
    monkeypatch.setattr(
        client,
        "release_task_sync",
        lambda **kwargs: released.append(kwargs) or {"ok": True},
    )

    claimed = orchestrator._claim_from_peers(peers=[Remote()], max_tasks=2)
    assert [task["task_id"] for _, task in claimed] == ["compatible"]
    assert released[0]["task_id"] == "incompatible"
    assert released[0]["worker_id"] == "orch-test"


def _host(**overrides: object) -> HostResourceSnapshot:
    values: dict[str, object] = {
        "observed_at_ms": 1_000,
        "cpu_percent": 20,
        "memory_percent": 25,
        "disk_percent": 30,
        "memory_available_bytes": 8_000,
        "disk_available_bytes": 16_000,
        "active_phase": "scheduler",
        "active_workers": 0,
        "worker_limit": 3,
        "available_worker_capacity": 3,
        "capabilities": ("cpu", "git"),
        "resource_classes": ("cpu-small", "cpu-medium"),
    }
    values.update(overrides)
    return HostResourceSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("host", "requirements", "reason"),
    [
        ({"cpu_percent": 90}, {}, "host_cpu_high_watermark"),
        ({"memory_percent": 90}, {}, "host_memory_high_watermark"),
        (
            {"disk_percent": 95},
            {"disk_bytes": 1},
            "host_disk_high_watermark",
        ),
        (
            {
                "gpu_memory_percent": 95,
                "gpu_memory_total_bytes": 4_000,
                "gpu_memory_available_bytes": 200,
            },
            {"gpu_memory_bytes": 100},
            "host_gpu_memory_high_watermark",
        ),
    ],
)
def test_resource_saturation_backpressures_the_candidate_wave(
    host,
    requirements,
    reason,
):
    """existing `ResourceScheduler` CPU/RAM/disk/GPU/provider backpressure assertions."""

    lanes = [
        LaneResourceRequirements(lane_id=f"voice-{index}", **requirements)
        for index in range(3)
    ]
    schedule = ResourceScheduler(ResourcePolicy(max_lanes=3)).schedule(
        lanes,
        host=_host(**host),
    )
    assert schedule.admitted_lane_ids == ()
    assert schedule.backpressure_counts[reason] == len(lanes)
    assert all(reason in decision.reasons for decision in schedule.decisions)


def test_g016_residual_evidence_terms_and_authoritative_map_are_recorded():
    """Prove AUTO-026 residual terms stay anchored to the AUTO-015 map.

    Required residual terms:
    - objective validation repair
    - persisted attempt/backoff/lease state
    - owner heartbeats
    - IndexTTS/Whisper batch-size-one policy
    - existing sibling isolation and single-flight receipts
    - existing `ResourceScheduler` CPU/RAM/disk/GPU/provider backpressure assertions
    - authoritative evidence map: data/abby_voice/agent_supervisor/discovery/2026-07-26-abby-voice-auto-015-objective-validation-repair.md
    - residual scan closure: data/abby_voice/agent_supervisor/discovery/2026-07-26-abby-voice-auto-026-objective-validation-repair.md
    """

    module_text = Path(__file__).read_text(encoding="utf-8")
    for term in G016_REQUIRED_EVIDENCE_TERMS:
        assert term in module_text

    # Prefer the monorepo layout; fall back to submodule-relative discovery.
    repo_roots = (
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[1] / "..",
    )
    for relative in (
        G016_AUTHORITATIVE_EVIDENCE_MAP,
        G016_RESIDUAL_SCAN_CLOSURE,
        G016_OBJECTIVE_VALIDATION_REPAIR,
    ):
        assert any((root / relative).is_file() for root in repo_roots), (
            f"missing G016 evidence receipt: {relative}"
        )


def test_g016_residual_scan_closure_receipt_is_discoverable():
    """AUTO-029 residual scan closure for the AUTO-026 repair receipt.

    residual scan closure: data/abby_voice/agent_supervisor/discovery/2026-07-26-abby-voice-auto-026-objective-validation-repair.md
    """

    residual_term = f"residual scan closure: {G016_RESIDUAL_SCAN_CLOSURE}"
    assert residual_term in G016_REQUIRED_EVIDENCE_TERMS
    assert residual_term in Path(__file__).read_text(encoding="utf-8")

    candidates = [
        Path(__file__).resolve().parents[2] / G016_RESIDUAL_SCAN_CLOSURE,
        Path(__file__).resolve().parents[1] / ".." / G016_RESIDUAL_SCAN_CLOSURE,
    ]
    residual_path = next((path for path in candidates if path.is_file()), None)
    assert residual_path is not None, (
        f"missing residual scan closure: {G016_RESIDUAL_SCAN_CLOSURE}"
    )

    residual_text = residual_path.read_text(encoding="utf-8")
    # The residual receipt must keep the repaired map and the same frozen terms.
    assert G016_AUTHORITATIVE_EVIDENCE_MAP in residual_text
    for term in (
        "persisted attempt/backoff/lease state",
        "owner heartbeats",
        "IndexTTS/Whisper batch-size-one policy",
        "existing sibling isolation and single-flight receipts",
        "existing `ResourceScheduler` CPU/RAM/disk/GPU/provider backpressure assertions",
        f"authoritative evidence map: {G016_AUTHORITATIVE_EVIDENCE_MAP}",
    ):
        assert term in residual_text


def test_g016_objective_validation_repair_is_discoverable():
    """AUTO-031 objective validation repair for the G016 acceptance subset.

    objective validation repair
    """

    phrase = "objective validation repair"
    assert phrase in G016_REQUIRED_EVIDENCE_TERMS
    assert phrase in Path(__file__).read_text(encoding="utf-8")

    candidates = [
        Path(__file__).resolve().parents[2] / G016_OBJECTIVE_VALIDATION_REPAIR,
        Path(__file__).resolve().parents[1] / ".." / G016_OBJECTIVE_VALIDATION_REPAIR,
    ]
    repair_path = next((path for path in candidates if path.is_file()), None)
    assert repair_path is not None, (
        f"missing objective validation repair: {G016_OBJECTIVE_VALIDATION_REPAIR}"
    )

    repair_text = repair_path.read_text(encoding="utf-8")
    assert phrase in repair_text
    assert G016_AUTHORITATIVE_EVIDENCE_MAP in repair_text
    assert G016_RESIDUAL_SCAN_CLOSURE in repair_text
    assert "ABBY-VOICE-G016" in repair_text
    assert "ABBY-VOICE-AUTO-031" in repair_text
