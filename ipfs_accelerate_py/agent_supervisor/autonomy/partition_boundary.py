"""SPAR W4 SCC partitions and boundary contracts on the existing board owner.

A refactor wave may write inside one SCC. Crossing SCCs requires a declared
boundary contract. Partial SCC completion is forbidden. Missing partition
payload is fail-open. TypeSafe is never this owner. This is not a second board.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

CROSS_SCC_WITHOUT_CONTRACT = "cross_scc_without_contract"
PARTIAL_SCC_FORBIDDEN = "partial_scc_completion_forbidden"
PARTITION_RESPECTED = "partition_respected"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return tuple(out)
    return ()


def _wave(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("refactor_wave", "extraction_wave", "partition"):
        nested = state.get(key)
        if isinstance(nested, Mapping):
            return {**state, **nested}
    return state


def claims_partition_boundary(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    return bool(
        payload.get("refactor_wave")
        or payload.get("extraction_wave")
        or payload.get("scc_ids")
        or payload.get("write_sccs")
        or payload.get("boundary_contract_set_cid")
        or payload.get("partition_candidate_cid")
        or payload.get("partial_scc") is True
    )


def partition_boundary_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    claimed = claims_partition_boundary(payload)
    merged = _wave(payload)
    sccs = _ids(merged.get("write_sccs") or merged.get("scc_ids"))
    contract = str(
        merged.get("boundary_contract_set_cid")
        or merged.get("boundary_contract")
        or ""
    ).strip()
    members = merged.get("scc_member_count")
    done = merged.get("completed_scc_members")
    partial_flag = merged.get("partial_scc") is True
    if isinstance(members, int) and isinstance(done, int) and not isinstance(
        members, bool
    ) and not isinstance(done, bool):
        if members > 0 and done < members:
            partial_flag = True
    cross = len(sccs) > 1 and not contract
    reason = ""
    if claimed and partial_flag:
        reason = PARTIAL_SCC_FORBIDDEN
    elif claimed and cross:
        reason = CROSS_SCC_WITHOUT_CONTRACT
    elif claimed and sccs and (len(sccs) == 1 or contract):
        reason = PARTITION_RESPECTED
    blocks = bool(
        claimed and reason in {PARTIAL_SCC_FORBIDDEN, CROSS_SCC_WITHOUT_CONTRACT}
    )
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "scc_ids": sccs,
        "boundary_contract": contract,
        "respected": reason == PARTITION_RESPECTED,
        "blocks_completion": blocks,
        "reason_code": reason,
    }


def partition_boundary_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(partition_boundary_view(state)["blocks_completion"])


__all__ = [
    "CROSS_SCC_WITHOUT_CONTRACT",
    "PARTIAL_SCC_FORBIDDEN",
    "PARTITION_RESPECTED",
    "claims_partition_boundary",
    "partition_boundary_blocks_completion",
    "partition_boundary_view",
]
