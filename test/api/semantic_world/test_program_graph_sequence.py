"""SAWM-029 GNN/graph-transformer/TAGSeq experiment adapters."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "ipfs_accelerate_py/agent_supervisor/evaluation/program_graph_sequence.py"
)
_SPEC = importlib.util.spec_from_file_location("program_graph_sequence", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["program_graph_sequence"] = _MOD
_SPEC.loader.exec_module(_MOD)
run_program_graph_sequence_ablation = _MOD.run_program_graph_sequence_ablation


def test_missing_encoders_are_typed_and_static_baseline_is_reported() -> None:
    result = run_program_graph_sequence_ablation(
        nodes=("a", "b"),
        types={"a": "fn", "b": "unknown"},
        gold="a",
        backends={"gnn": False, "graph_transformer": False, "tagseq": False, "linear": True},
    )
    assert result["encoders"]["gnn"]["status"] == "backend_unavailable"
    assert result["encoders"]["linear"]["status"] == "available"
    assert result["static_baseline_hit"] is True
    assert result["runtime_authority"] is False
    assert result["completion_authority"] is False
    assert result["masked_unknown"] == 1
