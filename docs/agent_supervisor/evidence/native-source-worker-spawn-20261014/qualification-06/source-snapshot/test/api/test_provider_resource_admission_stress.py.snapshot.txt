"""End-to-end queue admission with bounded authored stub work."""

from benchmarks.agent_supervisor.provider_resource_admission import run_benchmark


def test_external_pressure_backoff_recovery_and_cancellation_stress() -> None:
    report = run_benchmark(32)
    assert report["succeeded"] == 28
    assert report["rejected_unsafe"] == 2
    assert report["cancelled_queued"] == report["cancelled_running"] == 1
    assert report["peak_provider_callbacks"] == 2
    assert report["peak_resource_leases"] == 2
    assert report["remaining_leases"] == report["remaining_batches"] == report["remaining_queue"] == 0
    assert set(report["pressure_observations_ms"]) == {"cpu", "memory", "provider"}
    assert report["live_provider_calls"] == report["target_executions"] == report["solver_executions"] == 0
