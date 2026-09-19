"""DOEP-110 end-to-end token and compute telemetry."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    record_end_to_end_token_compute,
)


def test_measured_tokens_are_not_self_certified_zeros() -> None:
    receipt = record_end_to_end_token_compute(
        input_tokens=12, output_tokens=4, cpu_seconds=0.5, gpu_seconds=None
    )
    samples = {item["name"]: item for item in receipt["samples"]}
    assert samples["input_tokens"]["status"] == "measured"
    assert samples["input_tokens"]["value"] == 12
    assert samples["gpu_seconds"]["status"] == "unavailable"
    assert samples["gpu_seconds"]["value"] is None
    assert receipt["completion_authority"] is False
