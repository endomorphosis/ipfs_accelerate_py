"""DOEP-111 Codex-primed baseline harness.

The harness records a frozen baseline identity. It does not invoke Codex,
complete tasks, or treat model output as admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class BaselineHarnessError(ValueError):
    """Closed baseline-harness contract violation."""


@dataclass(frozen=True, slots=True)
class CodexPrimedBaseline:
    case_id: str
    prompt_cid: str
    frozen: bool = True


def run_codex_primed_baseline(
    cases: Sequence[Mapping[str, Any]],
    *,
    invoke_codex: bool = False,
) -> dict[str, Any]:
    if invoke_codex:
        raise BaselineHarnessError("harness must not auto-start or invoke Codex")
    if not cases:
        raise BaselineHarnessError("empty baseline")
    frozen = tuple(
        CodexPrimedBaseline(case_id=str(item["case_id"]), prompt_cid=str(item["prompt_cid"]))
        for item in cases
    )
    return {
        "n": len(frozen),
        "cases": [item.case_id for item in frozen],
        "codex_invoked": False,
        "completion_authority": False,
        "frozen": True,
    }
