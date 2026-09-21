"""Independent stage backpressure and safe preemption."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

STAGES = ("plan", "implement", "validate", "merge", "refresh")


class BackpressureError(ValueError):
    """Stage backpressure rejected an unsafe admission or preemption."""


def admit_stage(stage: str, inflight: int, limit: int) -> Mapping[str, Any]:
    if stage not in STAGES:
        raise BackpressureError(f"unknown stage {stage}")
    if type(limit) is not int or limit < 0:
        raise BackpressureError("limit must be a non-negative int")
    if inflight >= limit:
        decision = MappingProxyType({"admitted": False, "reason": "stage-backpressure", "stage": stage})
    else:
        decision = MappingProxyType({"admitted": True, "reason": "", "stage": stage})
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        mirror_work_record(
            catalog_kind="metadata",
            record_kind="stage_backpressure_admission",
            record_ref=str(stage or "stage-backpressure"),
            subject_kind="record_cid",
            subject_ref=str(stage or "stage-backpressure"),
        )
    except Exception:
        pass
    return decision


def preempt(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("has_external_effect") and not record.get("compensatable"):
        raise BackpressureError("unsafe preemption of uncompensatable effect")
    if record.get("claimed_exclusive") and not record.get("lease_releaseable"):
        raise BackpressureError("cannot preempt an unreleasable exclusive lease")
    return MappingProxyType({"preempted": True, "safe": True})
