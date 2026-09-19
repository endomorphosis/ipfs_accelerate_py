"""DOEP-066 ContextPack reuse and omission benchmark."""

from __future__ import annotations

import pytest

from benchmarks.agent_supervisor.doep.context_pack import (
    ContextPackBenchmarkError,
    run_context_pack_benchmark,
    validate_context_pack_benchmark,
)


CASES = (
    {
        "case_id": "reuse",
        "required": ("policy", "proof"),
        "included": ("policy", "proof", "goal"),
        "omitted": ("analogue",),
        "prefix": "goal|policy",
        "reused_prefix": True,
    },
    {
        "case_id": "fresh",
        "required": ("policy",),
        "included": ("policy", "goal"),
        "omitted": (),
        "prefix": "goal|policy",
        "reused_prefix": False,
    },
)


def test_reuse_and_omission_rates_do_not_drop_required_spans() -> None:
    result = validate_context_pack_benchmark(run_context_pack_benchmark(CASES))
    assert result["n"] == 2
    assert result["prefix_reuse_rate"] == 0.5
    assert result["required_dropped"] == 0
    assert result["completion_authority"] is False


def test_required_omission_fails_validation() -> None:
    bad = (
        {
            "case_id": "drop",
            "required": ("proof",),
            "included": ("goal",),
            "omitted": ("proof",),
            "prefix": "goal",
            "reused_prefix": False,
        },
    )
    with pytest.raises(ContextPackBenchmarkError, match="required spans"):
        validate_context_pack_benchmark(run_context_pack_benchmark(bad))
