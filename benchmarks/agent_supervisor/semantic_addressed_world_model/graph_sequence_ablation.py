"""SAWM-029 ablation runner over fair graph-sequence adapters."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.evaluation.program_graph_sequence import (
    run_program_graph_sequence_ablation,
)


def main() -> dict:
    return run_program_graph_sequence_ablation(
        nodes=("n0", "n1", "n2"),
        types={"n0": "fn", "n1": "fn", "n2": "unknown"},
        gold="n1",
    )


if __name__ == "__main__":
    print(main())
