"""DOEP-125/126 honest non-promotion and residual-gap reporting."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def produce_promotion_or_honest_non_promotion(
    results: Sequence[Mapping[str, Any]],
    *,
    remaining_todo: int = 24,
) -> dict[str, Any]:
    claims_completion = any(item.get("completion_authority") is True for item in results)
    promoted = remaining_todo == 0 and not claims_completion and bool(results)
    if claims_completion or remaining_todo > 0:
        promoted = False
    return {
        "schema": "doep-promotion-decision@1",
        "decision": "promoted" if promoted else "honest_non_promotion",
        "promoted": promoted,
        "completion_authority": False,
        "remaining_todo": remaining_todo,
        "reason": (
            "remaining DOEP tasks are still todo on the live extra-gate; "
            "overlay tests are not DuckDB completion"
        ),
        "n_results": len(results),
    }


def publish_residual_gap_and_marginal_return_report(
    decision: Mapping[str, Any],
    *,
    native_completed: int = 61,
) -> dict[str, Any]:
    return {
        "schema": "doep-residual-gap-and-marginal-return@1",
        "completion_authority": False,
        "promoted": bool(decision.get("promoted")),
        "native_completed": native_completed,
        "remaining_todo": int(decision.get("remaining_todo") or 0),
        "gaps": [
            "live extra-gate cannot claim remaining todos",
            "overlay tests are not DuckDB completion evidence",
            "dispatcher still stalls on native observation/settlement",
        ],
        "marginal_return": {
            "more_overlay_modules": "diminishing versus extra-gate claim path",
            "forged_cas": "forbidden",
        },
        "decision": decision.get("decision"),
    }
