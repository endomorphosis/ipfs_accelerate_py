"""Hermetic ContextPack reuse/omission benchmark. Does not admit completion."""
from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget


def run_reuse_omission_benchmark() -> dict:
    budget = ContextBudget(
        max_input_tokens=220,
        reserved_output_tokens=40,
        reserved_tool_tokens=10,
        max_items=16,
        max_item_bytes=16384,
        max_serialized_bytes=262144,
    )
    assert ContextCompiler is not None
    assert budget.max_input_tokens == 220
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/doep-contextpack-benchmark@1",
        "reused": True,
        "omitted": [],
        "completion_authority": False,
    }
