"""SAWM-037 required completion gate.

Completion is refused unless the required receipt set is present. This
module does not write DuckDB.
"""

from __future__ import annotations

from typing import Any, Mapping

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_required import (
    REQUIRED_DISPATCH_RECEIPTS,
)


REQUIRED_COMPLETION_RECEIPTS: tuple[str, ...] = (
    *REQUIRED_DISPATCH_RECEIPTS,
    "proof_test_validation",
    "execution_transition",
    "post_root",
    "expected_result",
    "dogfood",
)


class ProgramWorldRequiredCompletionGate:
    def admit_required_program_world_completion(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        present = tuple(request.get("receipts") or ())
        missing = tuple(
            name for name in REQUIRED_COMPLETION_RECEIPTS if name not in present
        )
        return {
            "admitted": not missing,
            "missing": missing,
            "completion_authority": False,
            "cas_completed": False,
        }


def admit_required_program_world_completion(request: Mapping[str, Any]) -> dict[str, Any]:
    return ProgramWorldRequiredCompletionGate().admit_required_program_world_completion(
        request
    )
