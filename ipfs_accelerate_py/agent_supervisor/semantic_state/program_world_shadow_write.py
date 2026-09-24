"""SAWM-034 shadow_write for program-world artifacts.

Shadow records sit beside authoritative behavior. They cannot influence
planning, routing, or completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class ShadowWriteError(ValueError):
    """Closed shadow-write contract violation."""


@dataclass(frozen=True, slots=True)
class ShadowWriteReceipt:
    artifact_id: str
    kind: str
    authoritative: bool = False
    influences_planning: bool = False
    influences_routing: bool = False
    completion_authority: bool = False


@dataclass(frozen=True, slots=True)
class ShadowParityReport:
    compared: int
    mismatches: int
    completion_authority: bool = False


def _mirror_shadow_artifact(receipt: ShadowWriteReceipt) -> ShadowWriteReceipt:
    """Record a shadow artifact id. The artifact body is not stored."""

    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        record_ref = str(receipt.artifact_id or "shadow-artifact")
        mirror_work_record(
            catalog_kind="metadata",
            record_kind="shadow_artifact_record",
            record_ref=record_ref,
            subject_kind="record_cid",
            subject_ref=record_ref,
        )
    except Exception:
        pass
    return receipt


@dataclass
class ProgramWorldShadowWriter:
    _shadow: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def record_program_world_shadow_artifacts(
        self, artifact: Mapping[str, Any]
    ) -> ShadowWriteReceipt:
        artifact_id = str(artifact.get("artifact_id") or "")
        if not artifact_id:
            raise ShadowWriteError("artifact_id is required")
        if artifact.get("authoritative") is True:
            raise ShadowWriteError("shadow artifacts cannot be authoritative")
        self._shadow[artifact_id] = dict(artifact)
        return _mirror_shadow_artifact(
            ShadowWriteReceipt(
                artifact_id=artifact_id,
                kind=str(artifact.get("kind") or "projection"),
            )
        )

    def parity(self, authoritative: Mapping[str, Mapping[str, Any]]) -> ShadowParityReport:
        mismatches = 0
        for key, value in self._shadow.items():
            current = authoritative.get(key)
            if current is None:
                continue
            if current.get("cid") != value.get("cid"):
                mismatches += 1
        return ShadowParityReport(compared=len(self._shadow), mismatches=mismatches)


def record_program_world_shadow_artifacts(
    artifact: Mapping[str, Any],
    *,
    writer: ProgramWorldShadowWriter | None = None,
) -> ShadowWriteReceipt:
    return (writer or ProgramWorldShadowWriter()).record_program_world_shadow_artifacts(artifact)
