"""SAWM-031 model serving, batching, hardware, and quantization."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.program_world_batching import (
    ProgramWorldBatchingError,
    batch_program_world_predictions,
)
from ipfs_accelerate_py.agent_supervisor.runtime.program_world_model_serving import (
    ProgramWorldServingError,
    serve_program_world_specialist,
)


def test_unadmitted_checkpoint_is_unavailable() -> None:
    result = serve_program_world_specialist({"specialist": "call_ranking"})
    assert result["served"] is False
    assert result["reason_code"] == "checkpoint_unavailable"
    assert result["completion_authority"] is False


def test_admitted_specialist_is_proposal_only() -> None:
    result = serve_program_world_specialist(
        {"specialist": "call_ranking", "checkpoint_admitted": True, "quantized": True}
    )
    assert result["served"] is True
    assert result["device"] == "cpu"
    assert result["quantized"] is True
    assert result["proposal_only"] is True


def test_privacy_and_batch_bounds() -> None:
    with pytest.raises(ProgramWorldServingError, match="privacy"):
        serve_program_world_specialist(
            {"checkpoint_admitted": True, "privacy_denied": True}
        )
    with pytest.raises(ProgramWorldBatchingError, match="max_batch"):
        batch_program_world_predictions([{"id": n} for n in range(9)], max_batch=8)
    with pytest.raises(ProgramWorldBatchingError, match="mix"):
        batch_program_world_predictions(
            [
                {"id": 1, "privacy": "a", "tenant": "t", "profile": "p", "authority": "x"},
                {"id": 2, "privacy": "b", "tenant": "t", "profile": "p", "authority": "x"},
            ]
        )


def test_caller_claimed_gpu_is_not_simulated(monkeypatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.program_world_model_serving.probe_gpu",
        lambda: False,
    )
    with pytest.raises(ProgramWorldServingError, match="gpu_unavailable"):
        serve_program_world_specialist(
            {"checkpoint_admitted": True, "gpu_available": True, "require_gpu": True}
        )
    cpu = serve_program_world_specialist({"checkpoint_admitted": True})
    assert cpu["device"] == "cpu"
