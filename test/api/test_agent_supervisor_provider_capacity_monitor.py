from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    provider_capacity_monitor as monitor_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_capacity_monitor import (
    ProviderCapacityMonitor,
    ProviderCapacityMonitorConfig,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_capacity_snapshot import (
    DUAL_REVIEW_PROVIDER_ID,
    PROVIDER_CAPACITY_BUDGET_SEMANTICS,
    load_provider_capacity_snapshot,
    synthesize_dual_review_provider_capacity,
)


def _readiness(*, ready: bool = True) -> dict[str, object]:
    return {
        "ready": ready,
        "implementation": {
            "provider": "grok_cli",
            "binary_available": ready,
            "authenticated": ready,
        },
        "review": {
            "provider": "codex_cli",
            "binary_available": ready,
            "authenticated": ready,
            "independent": True,
        },
    }


def _private_path(tmp_path: Path) -> Path:
    os.chmod(tmp_path, 0o700)
    return tmp_path / "capacity.json"


def test_one_shot_publishes_readiness_and_remaining_operator_budgets(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    config = ProviderCapacityMonitorConfig(
        snapshot_path=path,
        max_age_ms=1_000,
        interval_seconds=0.1,
        grok_max_concurrency=2,
        codex_max_concurrency=2,
        grok_request_budget=2,
        codex_request_budget=2,
        grok_token_budget=8_192,
        codex_token_budget=8_192,
        context_budget_tokens=32_768,
    )
    monitor = ProviderCapacityMonitor(
        config,
        readiness_source=_readiness,
        process_counter=lambda: {"grok_cli": 1, "codex_cli": 0},
        clock_ms=lambda: 100_000,
    )

    result = monitor.publish_once()
    capacities = {
        item.provider_id: item
        for item in load_provider_capacity_snapshot(
            path,
            max_age_ms=1_000,
            now_ms=100_000,
        )
    }

    assert result["published"] is True
    assert result["operator_bounds"]["budget_semantics"] == (
        PROVIDER_CAPACITY_BUDGET_SEMANTICS
    )
    assert result["process_count_scope"]["scope"] == (
        "current-uid-noninteractive-cli-invocation-roots"
    )
    assert capacities["grok_cli"].active_requests == 1
    assert capacities["grok_cli"].quota_remaining == 1
    assert capacities["grok_cli"].token_budget_remaining == 4_096
    assert capacities["codex_cli"].active_requests == 0
    assert capacities["codex_cli"].quota_remaining == 2
    assert capacities["codex_cli"].token_budget_remaining == 8_192
    assert all(item.healthy for item in capacities.values())


def test_fresh_unhealthy_snapshot_recovers_on_next_good_probe(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    ready = False
    clock = iter((200_000, 200_100))
    monitor = ProviderCapacityMonitor(
        ProviderCapacityMonitorConfig(
            snapshot_path=path,
            max_age_ms=1_000,
            interval_seconds=0.1,
        ),
        readiness_source=lambda: _readiness(ready=ready),
        process_counter=lambda: {"grok_cli": 0, "codex_cli": 0},
        clock_ms=clock.__next__,
    )

    first = monitor.publish_once()
    first_pair = {
        item.provider_id: item
        for item in synthesize_dual_review_provider_capacity(
            load_provider_capacity_snapshot(
                path, max_age_ms=1_000, now_ms=200_000
            ),
            max_age_ms=1_000,
            now_ms=200_000,
        )
    }[DUAL_REVIEW_PROVIDER_ID]
    assert first["ready"] is False
    assert first_pair.healthy is False

    ready = True
    second = monitor.publish_once()
    second_pair = {
        item.provider_id: item
        for item in synthesize_dual_review_provider_capacity(
            load_provider_capacity_snapshot(
                path, max_age_ms=1_000, now_ms=200_100
            ),
            max_age_ms=1_000,
            now_ms=200_100,
        )
    }[DUAL_REVIEW_PROVIDER_ID]
    assert second["ready"] is True
    assert second_pair.healthy is True


def test_daemon_mode_refreshes_before_ttl_and_stops_cleanly(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    sleeps: list[float] = []
    clock = iter((300_000, 300_100))
    monitor = ProviderCapacityMonitor(
        ProviderCapacityMonitorConfig(
            snapshot_path=path,
            max_age_ms=1_000,
            interval_seconds=0.1,
        ),
        readiness_source=_readiness,
        process_counter=lambda: {"grok_cli": 0, "codex_cli": 0},
        clock_ms=clock.__next__,
        sleep=sleeps.append,
    )

    result = monitor.run(max_cycles=2, stop_event=threading.Event())

    assert result["cycles"] == 2
    assert sleeps == [0.1]
    loaded = load_provider_capacity_snapshot(
        path,
        max_age_ms=1_000,
        now_ms=300_100,
    )
    assert {item.observed_at_ms for item in loaded} == {300_100}

    stopped_path = tmp_path / "stopped.json"
    sampled = threading.Event()
    stop_monitor = ProviderCapacityMonitor(
        ProviderCapacityMonitorConfig(
            snapshot_path=stopped_path,
            interval_seconds=1.0,
        ),
        readiness_source=lambda: (sampled.set() or _readiness()),
        process_counter=lambda: {"grok_cli": 0, "codex_cli": 0},
    )
    thread = threading.Thread(target=stop_monitor.run)
    thread.start()
    assert sampled.wait(2)
    stop_monitor.stop()
    thread.join(2)
    assert thread.is_alive() is False


def test_process_counter_matches_only_invocation_roots_and_deduplicates_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AccessDenied(Exception):
        pass

    class NoSuchProcess(Exception):
        pass

    def process(
        pid: int,
        parent_pid: int,
        command: list[str],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            pid=pid,
            info={
                "ppid": parent_pid,
                "cmdline": command,
                "uids": SimpleNamespace(effective=os.geteuid()),
            },
        )

    processes = [
        process(10, 1, ["node", "/usr/local/bin/codex", "exec", "-"]),
        process(11, 10, ["/vendor/bin/codex", "exec", "-"]),
        process(12, 1, ["codex", "resume", "session-id"]),
        process(13, 1, ["codex"]),
        process(
            20,
            1,
            [
                "grok",
                "--model",
                "grok-4.5",
                "--prompt-file",
                "/tmp/prompt",
                "--output-format",
                "plain",
            ],
        ),
        process(21, 1, ["grok"]),
    ]
    fake_psutil = SimpleNamespace(
        AccessDenied=AccessDenied,
        NoSuchProcess=NoSuchProcess,
        process_iter=lambda _fields: processes,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert monitor_module.count_active_cli_processes() == {
        "grok_cli": 1,
        "codex_cli": 1,
    }


def test_proc_fallback_rejects_partial_bounded_scan(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "1").mkdir(parents=True)
    (proc_root / "2").mkdir()

    with pytest.raises(RuntimeError, match="exceeds bounded fallback scan"):
        monitor_module._count_with_proc(  # noqa: SLF001
            maximum_processes=1,
            proc_root=proc_root,
        )


def test_auth_ready_is_distinct_from_exhausted_admission(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    monitor = ProviderCapacityMonitor(
        ProviderCapacityMonitorConfig(
            snapshot_path=path,
            max_age_ms=1_000,
            interval_seconds=0.1,
        ),
        readiness_source=_readiness,
        process_counter=lambda: {"grok_cli": 3, "codex_cli": 3},
        clock_ms=lambda: 400_000,
    )

    result = monitor.publish_once()
    capacities = load_provider_capacity_snapshot(
        path,
        max_age_ms=1_000,
        now_ms=400_000,
    )

    assert result["auth_ready"] is True
    assert result["admission_ready"] is False
    assert result["ready"] is False
    assert all(item.healthy is False for item in capacities)
    assert all(item.available_concurrency == 0 for item in capacities)


def test_cli_one_shot_uses_conservative_defaults_and_validates_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _private_path(tmp_path)
    monkeypatch.setattr(
        monitor_module,
        "production_cli_policy_readiness",
        _readiness,
    )
    monkeypatch.setattr(
        monitor_module,
        "count_active_cli_processes",
        lambda: {"grok_cli": 0, "codex_cli": 0},
    )

    assert monitor_module.main(
        [
            "--snapshot-path",
            str(path),
            "--max-age-ms",
            "1000",
            "--interval-seconds",
            "0.1",
            "--once",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["cycles"] == 1
    capacities = load_provider_capacity_snapshot(
        path,
        max_age_ms=1_000,
    )
    assert {item.max_concurrency for item in capacities} == {2}

    with pytest.raises(ValueError, match="shorter than max_age_ms"):
        ProviderCapacityMonitorConfig(
            snapshot_path=tmp_path / "invalid.json",
            max_age_ms=1_000,
            interval_seconds=1.0,
        )
