"""Project admitted change-propagation packets into exact supervisor tasks.

This is a projection boundary, not a planner.  Providers never nominate files
or plan order: every emitted task derives its predicted files, write scope,
SCC membership, and dependency edges directly from one canonical
``ChangePropagationEditPacket@1``.  Analytical steps become deterministic
non-provider tasks; only model-required steps may later reach ``llm_router``.

A small in-memory plan index makes projection idempotent: a repeated packet
or a second packet for the same plan/tree cannot mint duplicate tasks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Final

from ..analysis.change_propagation_contracts import PropagationAuthorityRoots
from ..proof.change_propagation_edit_packet import (
    CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE,
    ChangePropagationEditPacket,
    ChangePropagationEditPacketError,
    PropagationEditStep,
    PropagationEditStepKind,
)
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .objective_graph import ObjectiveFinding, ObjectiveTaskRecord


CHANGE_PROPAGATION_TASK_SOURCE_INTERFACE: Final[str] = "ChangePropagationTaskSource@1"
CHANGE_PROPAGATION_TASK_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-task-projection@1"
)
CHANGE_PROPAGATION_STEP_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-step-task@1"
)
CHANGE_PROPAGATION_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-supervisor-task@1"
)
TASK_ID_PREFIX: Final[str] = "PROP-PACKET-"
MAX_PROJECTED_PATHS: Final[int] = 1_024
MAX_STEPS: Final[int] = 512


class ChangePropagationTaskSourceError(ValueError):
    """A packet cannot safely become an implementation task set."""


class ChangePropagationTaskProjectionReason(str, Enum):
    """Closed outcomes for the packet-to-task admission boundary."""

    EMITTED = "emitted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    REJECTED = "rejected"
    ABSTAINED = "abstained"
    MALFORMED = "malformed"
    SCOPE_MISMATCH = "scope_mismatch"
    PARTIAL = "partial"
    EMPTY = "empty"


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ChangePropagationTaskSourceError(f"{name} must be a non-empty trimmed identifier")
    if any(character.isspace() for character in value) or "\x00" in value:
        raise ChangePropagationTaskSourceError(f"{name} must be an opaque identifier")
    return value


def _paths(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ChangePropagationTaskSourceError(f"{name} must be a path sequence")
    if any(not isinstance(item, str) for item in value):
        raise ChangePropagationTaskSourceError(f"{name} must contain only paths")
    result = tuple(sorted(set(value)))
    if not result or len(result) > MAX_PROJECTED_PATHS:
        raise ChangePropagationTaskSourceError(f"{name} must be a bounded non-empty path sequence")
    return result


def deterministic_change_propagation_task_id(
    packet_id: str,
    plan_id: str,
    step_id: str,
    tree_id: str,
) -> str:
    """Return the stable task alias for one admitted plan step in one tree."""

    digest = sha256(
        canonical_json_bytes(
            {
                "schema": CHANGE_PROPAGATION_TASK_SCHEMA,
                "packet_id": _identifier(packet_id, "packet_id"),
                "plan_id": _identifier(plan_id, "plan_id"),
                "step_id": _identifier(step_id, "step_id"),
                "tree_id": _identifier(tree_id, "tree_id"),
            }
        )
    ).hexdigest()
    return TASK_ID_PREFIX + digest[:24].upper()


def _packet_from(
    value: ChangePropagationEditPacket | Mapping[str, Any],
) -> ChangePropagationEditPacket:
    if isinstance(value, ChangePropagationEditPacket):
        # Reparse the canonical record so projection is independent of incidental
        # object mutation and forged subclasses.
        return ChangePropagationEditPacket.from_dict(value.to_record())
    if isinstance(value, Mapping):
        return ChangePropagationEditPacket.from_dict(value)
    raise ChangePropagationTaskSourceError(
        "packet must be ChangePropagationEditPacket@1 or its canonical record"
    )


def _step_invokes_provider(step: PropagationEditStep) -> bool:
    """Analytical (and non-model) steps never open a provider path."""

    return step.kind is PropagationEditStepKind.MODEL_REQUIRED


def _format_value_sources(packet: ChangePropagationEditPacket, step: PropagationEditStep) -> str:
    sources = step.selected_value_sources or packet.selected_value_sources
    if not sources:
        return "none (no missing-input values admitted for this step)"
    lines = []
    for item in sources:
        lines.append(
            f"- requirement {item.requirement_id}: candidate {item.candidate_id}"
            + (f" via proof {item.proof_id}" if item.proof_id else "")
        )
    return "\n".join(lines)


def _format_behavior(packet: ChangePropagationEditPacket, step: PropagationEditStep) -> str:
    ids = step.required_behavior_ids or packet.required_behavior_ids
    return ", ".join(ids) if ids else "none (behavior must not be invented)"


def _task_prompt(
    packet: ChangePropagationEditPacket,
    step: PropagationEditStep,
    task_id: str,
    *,
    invokes_provider: bool,
) -> str:
    """Render a bounded, source-free prompt from packet/step fields only."""

    write_paths = ", ".join(step.write_paths) if step.write_paths else "(none)"
    read_paths = ", ".join(step.read_paths) if step.read_paths else "(none)"
    deps = ", ".join(step.dependency_step_ids) if step.dependency_step_ids else "none"
    unsupported = (
        ", ".join(step.unsupported_limits or packet.unsupported_limits) or "none"
    )
    validation = "\n".join(f"- {command}" for command in packet.validation_commands)
    fixed_point = packet.fixed_point_obligation_ref
    fixed_point_posts = ", ".join(packet.fixed_point_postcondition_refs)
    postconditions = ", ".join(step.postcondition_refs or packet.per_edit_postcondition_refs)
    proof_refs = ", ".join(step.proof_refs or packet.proof_refs)
    provider_line = (
        "This step is MODEL-REQUIRED and may escalate only through the bounded "
        "change-propagation llm_router path with an exact writer lease."
        if invokes_provider
        else (
            "This step is ANALYTICAL (or non-model).  It must be applied by a "
            "deterministic transform and MUST NOT invoke a provider or llm_router."
        )
    )
    return (
        f"Implement proof-gated change-propagation task {task_id}.\n\n"
        "Authority binding:\n"
        f"- Packet: {packet.packet_id}\n"
        f"- Plan: {packet.plan_id} (content {packet.plan_content_id})\n"
        f"- Admission: {packet.admission_id}\n"
        f"- Step: {step.step_id} ({step.kind.value} / plan {step.plan_step_kind.value})\n"
        f"- SCC group: {step.scc_group_id or 'none'}\n"
        f"- Dependency steps: {deps}\n"
        f"- Candidate tree: {packet.roots.candidate_tree_id}\n"
        f"- Candidate overlay: {packet.roots.candidate_overlay_id}\n"
        f"- Graph/index/model/config: {packet.roots.graph_id} / {packet.roots.index_id} / "
        f"{packet.roots.model_id} / {packet.roots.config_id}\n"
        f"- Translator/toolchain/policy: {packet.roots.translator_id} / "
        f"{packet.roots.toolchain_id} / {packet.roots.policy_id}\n\n"
        f"{provider_line}\n\n"
        "Admitted value sources (exact; model cannot choose another):\n"
        f"{_format_value_sources(packet, step)}\n\n"
        f"Admitted required behavior: {_format_behavior(packet, step)}\n"
        f"Unsupported limits: {unsupported}. Do not infer, implement, or claim support outside these limits.\n\n"
        "Scope is closed to packet step authority:\n"
        f"- Declared outputs / write paths: {write_paths}\n"
        f"- Allowed read paths: {read_paths}\n"
        f"- Packet write allowlist: {', '.join(packet.permitted_write_paths)}\n"
        "- The executor must not add, modify, rename, or request any path outside the step write authority.\n\n"
        f"Per-edit postconditions: {postconditions}\n"
        f"Fixed-point obligation: {fixed_point}\n"
        f"Fixed-point postconditions: {fixed_point_posts}\n"
        f"Proof references: {proof_refs}\n"
        f"Invalidators: {', '.join(packet.invalidation_refs)}\n"
        f"Index references: {', '.join(packet.index_refs)}\n"
        f"Graph references: {', '.join(packet.graph_refs)}\n\n"
        "Run focused validation, then re-prove to the fixed-point obligation:\n"
        f"{validation}\n"
    )


@dataclass(frozen=True, slots=True)
class ChangePropagationStepTask:
    """One plan-step task projected from an admitted packet."""

    step_id: str
    task_id: str
    kind: PropagationEditStepKind
    plan_step_kind: str
    write_paths: tuple[str, ...]
    read_paths: tuple[str, ...]
    dependency_step_ids: tuple[str, ...]
    dependency_task_ids: tuple[str, ...]
    scc_group_id: str
    invokes_provider: bool
    prompt: str
    task_record: ObjectiveTaskRecord | None = None
    transform_id: str = ""
    obligation_ids: tuple[str, ...] = ()

    @property
    def emitted(self) -> bool:
        return self.task_record is not None

    @property
    def declared_outputs(self) -> tuple[str, ...]:
        return self.write_paths

    @property
    def allowed_write_paths(self) -> tuple[str, ...]:
        return self.write_paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHANGE_PROPAGATION_STEP_TASK_SCHEMA,
            "step_id": self.step_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "plan_step_kind": self.plan_step_kind,
            "write_paths": list(self.write_paths),
            "read_paths": list(self.read_paths),
            "dependency_step_ids": list(self.dependency_step_ids),
            "dependency_task_ids": list(self.dependency_task_ids),
            "scc_group_id": self.scc_group_id,
            "invokes_provider": self.invokes_provider,
            "transform_id": self.transform_id,
            "obligation_ids": list(self.obligation_ids),
            "prompt": self.prompt,
            "emitted": self.emitted,
        }


@dataclass(frozen=True, slots=True)
class ChangePropagationTaskProjection:
    """Emitted step tasks for one plan/packet, or a typed no-task result.

    ``step_tasks`` and ``task_records`` are empty for rejected, stale, malformed,
    partial, and duplicate inputs.  Consumers should use ``implementation_tasks``
    rather than treating a failed packet as a partially specified task set.
    """

    packet_id: str
    plan_id: str
    plan_content_id: str
    tree_id: str
    reason: ChangePropagationTaskProjectionReason
    step_order: tuple[str, ...]
    scc_group_ids: tuple[str, ...]
    predicted_files: tuple[str, ...]
    write_scope: tuple[str, ...]
    step_tasks: tuple[ChangePropagationStepTask, ...] = ()
    detail: str = ""

    @property
    def emitted(self) -> bool:
        return self.reason is ChangePropagationTaskProjectionReason.EMITTED

    @property
    def implementation_tasks(self) -> tuple[ObjectiveTaskRecord, ...]:
        return tuple(
            item.task_record
            for item in self.step_tasks
            if item.task_record is not None
        )

    @property
    def task_records(self) -> tuple[ObjectiveTaskRecord, ...]:
        return self.implementation_tasks

    @property
    def provider_step_tasks(self) -> tuple[ChangePropagationStepTask, ...]:
        """Steps that may later reach a provider (model-required only)."""

        return tuple(item for item in self.step_tasks if item.invokes_provider)

    @property
    def analytical_step_tasks(self) -> tuple[ChangePropagationStepTask, ...]:
        return tuple(item for item in self.step_tasks if not item.invokes_provider)

    @property
    def declared_outputs(self) -> tuple[str, ...]:
        return self.predicted_files

    @property
    def allowed_write_paths(self) -> tuple[str, ...]:
        return self.write_scope

    @property
    def projection_id(self) -> str:
        return content_identity(
            {
                "schema": CHANGE_PROPAGATION_TASK_PROJECTION_SCHEMA,
                "packet_id": self.packet_id,
                "plan_id": self.plan_id,
                "plan_content_id": self.plan_content_id,
                "tree_id": self.tree_id,
                "reason": self.reason.value,
                "step_order": list(self.step_order),
                "scc_group_ids": list(self.scc_group_ids),
                "predicted_files": list(self.predicted_files),
                "write_scope": list(self.write_scope),
                "step_task_ids": [item.task_id for item in self.step_tasks],
                "step_kinds": [item.kind.value for item in self.step_tasks],
                "invokes_provider": [item.invokes_provider for item in self.step_tasks],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHANGE_PROPAGATION_TASK_PROJECTION_SCHEMA,
            "interface": CHANGE_PROPAGATION_TASK_SOURCE_INTERFACE,
            "projection_id": self.projection_id,
            "packet_id": self.packet_id,
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "tree_id": self.tree_id,
            "reason": self.reason.value,
            "step_order": list(self.step_order),
            "scc_group_ids": list(self.scc_group_ids),
            "predicted_files": list(self.predicted_files),
            "write_scope": list(self.write_scope),
            "step_tasks": [item.to_dict() for item in self.step_tasks],
            "detail": self.detail,
            "emitted": self.emitted,
        }

    def task_for_step(self, step_id: str) -> ChangePropagationStepTask | None:
        for item in self.step_tasks:
            if item.step_id == step_id:
                return item
        return None


class ChangePropagationTaskSource:
    """Idempotently issue exact-scope supervisor tasks from admitted packets."""

    interface: Final[str] = CHANGE_PROPAGATION_TASK_SOURCE_INTERFACE

    def __init__(
        self,
        *,
        current_roots: PropagationAuthorityRoots | None = None,
        roots: PropagationAuthorityRoots | None = None,
        current_tree_id: str | None = None,
    ) -> None:
        if current_roots is not None and roots is not None and current_roots != roots:
            raise ChangePropagationTaskSourceError("current_roots and roots disagree")
        configured_roots = current_roots if current_roots is not None else roots
        if configured_roots is not None and not isinstance(
            configured_roots, PropagationAuthorityRoots
        ):
            raise ChangePropagationTaskSourceError(
                "current_roots must be PropagationAuthorityRoots"
            )
        self._lock = RLock()
        # plan_id + candidate_tree_id → projection (duplicate plans share work)
        self._by_plan_tree: dict[tuple[str, str], ChangePropagationTaskProjection] = {}
        self._current_roots = configured_roots
        self._current_tree_id = (
            _identifier(current_tree_id, "current_tree_id")
            if current_tree_id is not None
            else ""
        )

    def project(
        self,
        packet: ChangePropagationEditPacket | Mapping[str, Any],
        *,
        current_roots: PropagationAuthorityRoots | None = None,
        roots: PropagationAuthorityRoots | None = None,
        current_tree_id: str | None = None,
        provider_outputs: Sequence[str] | None = None,
    ) -> ChangePropagationTaskProjection:
        """Project one current packet into ordered step tasks.

        ``provider_outputs`` is accepted solely as a scope fence.  If supplied,
        it must exactly equal the packet write allowlist; it never supplements
        it.  ``roots`` is a compatibility spelling for ``current_roots``.
        """

        try:
            parsed = _packet_from(packet)
        except (
            ChangePropagationEditPacketError,
            ChangePropagationTaskSourceError,
            TypeError,
            ValueError,
        ) as exc:
            return ChangePropagationTaskProjection(
                "",
                "",
                "",
                "",
                ChangePropagationTaskProjectionReason.MALFORMED,
                (),
                (),
                (),
                (),
                detail=str(exc),
            )
        if current_roots is not None and roots is not None and current_roots != roots:
            raise ChangePropagationTaskSourceError("current_roots and roots disagree")
        expected_roots = (
            current_roots
            if current_roots is not None
            else roots
            if roots is not None
            else self._current_roots
        )
        if expected_roots is not None and (
            not isinstance(expected_roots, PropagationAuthorityRoots)
            or parsed.roots != expected_roots
        ):
            return ChangePropagationTaskProjection(
                parsed.packet_id,
                parsed.plan_id,
                parsed.plan_content_id,
                parsed.roots.candidate_tree_id,
                ChangePropagationTaskProjectionReason.STALE,
                (),
                (),
                (),
                (),
                detail="packet authority roots are not current",
            )
        expected_tree_id = (
            _identifier(current_tree_id, "current_tree_id")
            if current_tree_id is not None
            else self._current_tree_id
        )
        if expected_tree_id and parsed.roots.candidate_tree_id != expected_tree_id:
            return ChangePropagationTaskProjection(
                parsed.packet_id,
                parsed.plan_id,
                parsed.plan_content_id,
                parsed.roots.candidate_tree_id,
                ChangePropagationTaskProjectionReason.STALE,
                (),
                (),
                (),
                (),
                detail="packet candidate tree is not current",
            )
        if not parsed.steps or not parsed.step_order:
            return ChangePropagationTaskProjection(
                parsed.packet_id,
                parsed.plan_id,
                parsed.plan_content_id,
                parsed.roots.candidate_tree_id,
                ChangePropagationTaskProjectionReason.EMPTY,
                (),
                (),
                (),
                (),
                detail="packet has no projectable steps",
            )
        if not parsed.permitted_write_paths or not parsed.proof_refs:
            return ChangePropagationTaskProjection(
                parsed.packet_id,
                parsed.plan_id,
                parsed.plan_content_id,
                parsed.roots.candidate_tree_id,
                ChangePropagationTaskProjectionReason.PARTIAL,
                (),
                (),
                (),
                (),
                detail="packet is incomplete (write authority or proofs missing)",
            )
        if not parsed.fixed_point_obligation_ref:
            return ChangePropagationTaskProjection(
                parsed.packet_id,
                parsed.plan_id,
                parsed.plan_content_id,
                parsed.roots.candidate_tree_id,
                ChangePropagationTaskProjectionReason.PARTIAL,
                (),
                (),
                (),
                (),
                detail="packet lacks fixed-point validation authority",
            )
        if provider_outputs is not None:
            try:
                supplied = _paths(provider_outputs, "provider_outputs")
            except ChangePropagationTaskSourceError as exc:
                return ChangePropagationTaskProjection(
                    parsed.packet_id,
                    parsed.plan_id,
                    parsed.plan_content_id,
                    parsed.roots.candidate_tree_id,
                    ChangePropagationTaskProjectionReason.SCOPE_MISMATCH,
                    (),
                    (),
                    (),
                    (),
                    detail=str(exc),
                )
            if supplied != tuple(sorted(set(parsed.permitted_write_paths))):
                return ChangePropagationTaskProjection(
                    parsed.packet_id,
                    parsed.plan_id,
                    parsed.plan_content_id,
                    parsed.roots.candidate_tree_id,
                    ChangePropagationTaskProjectionReason.SCOPE_MISMATCH,
                    (),
                    (),
                    (),
                    (),
                    detail="provider outputs must exactly equal packet permitted_write_paths",
                )
        return self._emit(parsed)

    def project_packet(
        self,
        packet: ChangePropagationEditPacket | Mapping[str, Any],
        **kwargs: Any,
    ) -> ChangePropagationTaskProjection:
        """Explicit alias retained for integrations that name packet input."""

        return self.project(packet, **kwargs)

    def project_packets(
        self,
        packets: Sequence[ChangePropagationEditPacket | Mapping[str, Any]],
        *,
        current_roots: PropagationAuthorityRoots | None = None,
    ) -> tuple[ChangePropagationTaskProjection, ...]:
        """Compatibility spelling for a deterministic batch projection."""

        return self.project_many(packets, current_roots=current_roots)

    def project_many(
        self,
        packets: Sequence[ChangePropagationEditPacket | Mapping[str, Any]],
        *,
        current_roots: PropagationAuthorityRoots | None = None,
    ) -> tuple[ChangePropagationTaskProjection, ...]:
        """Project in canonical plan/packet order without duplicate emitted tasks."""

        if isinstance(packets, (str, bytes, bytearray)) or not isinstance(packets, Sequence):
            raise ChangePropagationTaskSourceError("packets must be a sequence")
        indexed = list(enumerate(packets))
        indexed.sort(key=lambda item: self._sort_key(item[1], item[0]))
        results: list[ChangePropagationTaskProjection] = []
        emitted_plans: set[tuple[str, str]] = set()
        for _, packet in indexed:
            projection = self.project(packet, current_roots=current_roots)
            if projection.emitted:
                key = (projection.plan_id, projection.tree_id)
                if key in emitted_plans:
                    # Same plan already emitted in this batch — surface as duplicate
                    # without inventing a second task set.
                    results.append(
                        ChangePropagationTaskProjection(
                            projection.packet_id,
                            projection.plan_id,
                            projection.plan_content_id,
                            projection.tree_id,
                            ChangePropagationTaskProjectionReason.DUPLICATE,
                            (),
                            (),
                            (),
                            (),
                            detail="plan/tree already projected in this batch",
                        )
                    )
                    continue
                emitted_plans.add(key)
            results.append(projection)
        return tuple(results)

    def _sort_key(self, packet: Any, ordinal: int) -> tuple[str, str, int]:
        try:
            parsed = _packet_from(packet)
            return (parsed.plan_id, parsed.packet_id, ordinal)
        except (
            ChangePropagationEditPacketError,
            ChangePropagationTaskSourceError,
            TypeError,
            ValueError,
        ):
            return ("~invalid", "~invalid", ordinal)

    def _emit(self, packet: ChangePropagationEditPacket) -> ChangePropagationTaskProjection:
        tree_id = packet.roots.candidate_tree_id
        key = (packet.plan_id, tree_id)
        with self._lock:
            existing = self._by_plan_tree.get(key)
            if existing is not None:
                if existing.packet_id == packet.packet_id:
                    return existing
                return ChangePropagationTaskProjection(
                    packet.packet_id,
                    packet.plan_id,
                    packet.plan_content_id,
                    tree_id,
                    ChangePropagationTaskProjectionReason.DUPLICATE,
                    (),
                    (),
                    (),
                    (),
                    detail="plan/tree already has projected implementation tasks",
                )

            by_step = {step.step_id: step for step in packet.steps}
            ordered_steps: list[PropagationEditStep] = []
            for step_id in packet.step_order:
                step = by_step.get(step_id)
                if step is None:
                    return ChangePropagationTaskProjection(
                        packet.packet_id,
                        packet.plan_id,
                        packet.plan_content_id,
                        tree_id,
                        ChangePropagationTaskProjectionReason.MALFORMED,
                        (),
                        (),
                        (),
                        (),
                        detail=f"step_order references missing step {step_id}",
                    )
                ordered_steps.append(step)

            task_id_by_step: dict[str, str] = {}
            for step in ordered_steps:
                task_id_by_step[step.step_id] = deterministic_change_propagation_task_id(
                    packet.packet_id,
                    packet.plan_id,
                    step.step_id,
                    tree_id,
                )

            step_tasks: list[ChangePropagationStepTask] = []
            for step in ordered_steps:
                invokes_provider = _step_invokes_provider(step)
                task_id = task_id_by_step[step.step_id]
                dependency_task_ids = tuple(
                    task_id_by_step[dep]
                    for dep in step.dependency_step_ids
                    if dep in task_id_by_step
                )
                # Task write authority is exactly the step write paths (packet
                # step authority).  Empty write sets are valid for pure
                # validation/checkpoint steps.
                write_paths = tuple(step.write_paths)
                read_paths = tuple(step.read_paths)
                prompt = _task_prompt(
                    packet, step, task_id, invokes_provider=invokes_provider
                )
                finding = ObjectiveFinding(
                    fingerprint=f"{packet.plan_content_id}:{step.step_id}",
                    goal_id="RPR-G200",
                    title=(
                        f"Propagate {step.kind.value} step {step.step_id}"
                        + (f" → {write_paths[0]}" if write_paths else "")
                    ),
                    summary=(
                        "Apply the exact proof-gated change-propagation plan step "
                        "without widening scope or inventing behavior."
                    ),
                    priority="P0",
                    track="propagation-task-gate",
                    missing_evidence=list(step.postcondition_refs)
                    + [packet.fixed_point_obligation_ref],
                    present_evidence={
                        "proof_refs": list(step.proof_refs or packet.proof_refs),
                        "value_sources": [
                            item.candidate_id
                            for item in (step.selected_value_sources or packet.selected_value_sources)
                        ],
                        "behavior_ids": list(
                            step.required_behavior_ids or packet.required_behavior_ids
                        ),
                    },
                    evidence_methods=["proof-gated-change-propagation"],
                    objective_path="change-propagation-edit-packet",
                    outputs=list(write_paths),
                    validation=" && ".join(packet.validation_commands),
                    predicted_files=list(write_paths),
                    changed_paths=list(write_paths),
                    interfaces=[
                        CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE,
                        CHANGE_PROPAGATION_TASK_SOURCE_INTERFACE,
                    ],
                    acceptance_subset=[
                        "write scope exactly equals packet step write_paths",
                        "SCC order and dependency metadata preserved",
                        "fixed-point validation succeeds",
                        "analytical steps never invoke a provider",
                    ],
                    evidence_subset=list(step.proof_refs or packet.proof_refs),
                    context_paths=list(read_paths),
                    dependencies=list(dependency_task_ids),
                    semantic_identity=(
                        f"change-propagation:{packet.plan_id}:{step.step_id}:{tree_id}"
                    ),
                    dedupe_key=(
                        f"change-propagation:{packet.plan_id}:{step.step_id}:{tree_id}"
                    ),
                    completion_authority="",
                    preconditions=list(step.precondition_refs),
                    effects=[
                        f"invokes_provider={invokes_provider}",
                        f"scc_group={step.scc_group_id or 'none'}",
                    ],
                    ast_symbols=[step.step_id, step.transform_id] if step.transform_id else [step.step_id],
                )
                task = ObjectiveTaskRecord(
                    task_id=task_id,
                    task_block=prompt,
                    finding=finding,
                    discovery_path=Path(write_paths[0]) if write_paths else Path("."),
                    depends_on=dependency_task_ids,
                )
                step_tasks.append(
                    ChangePropagationStepTask(
                        step_id=step.step_id,
                        task_id=task_id,
                        kind=step.kind,
                        plan_step_kind=step.plan_step_kind.value,
                        write_paths=write_paths,
                        read_paths=read_paths,
                        dependency_step_ids=tuple(step.dependency_step_ids),
                        dependency_task_ids=dependency_task_ids,
                        scc_group_id=step.scc_group_id,
                        invokes_provider=invokes_provider,
                        prompt=prompt,
                        task_record=task,
                        transform_id=step.transform_id,
                        obligation_ids=tuple(step.obligation_ids),
                    )
                )

            # Packet-level write scope is the closed authority union.
            write_scope = tuple(packet.permitted_write_paths)
            projection = ChangePropagationTaskProjection(
                packet_id=packet.packet_id,
                plan_id=packet.plan_id,
                plan_content_id=packet.plan_content_id,
                tree_id=tree_id,
                reason=ChangePropagationTaskProjectionReason.EMITTED,
                step_order=tuple(packet.step_order),
                scc_group_ids=tuple(packet.scc_group_ids),
                predicted_files=write_scope,
                write_scope=write_scope,
                step_tasks=tuple(step_tasks),
            )
            self._by_plan_tree[key] = projection
            return projection


def project_change_propagation_task(
    packet: ChangePropagationEditPacket | Mapping[str, Any],
    *,
    current_roots: PropagationAuthorityRoots | None = None,
    current_tree_id: str | None = None,
    provider_outputs: Sequence[str] | None = None,
) -> ChangePropagationTaskProjection:
    """Stateless convenience entry point for one deterministic projection."""

    return ChangePropagationTaskSource().project(
        packet,
        current_roots=current_roots,
        current_tree_id=current_tree_id,
        provider_outputs=provider_outputs,
    )


__all__ = [
    "CHANGE_PROPAGATION_STEP_TASK_SCHEMA",
    "CHANGE_PROPAGATION_TASK_PROJECTION_SCHEMA",
    "CHANGE_PROPAGATION_TASK_SCHEMA",
    "CHANGE_PROPAGATION_TASK_SOURCE_INTERFACE",
    "ChangePropagationStepTask",
    "ChangePropagationTaskProjection",
    "ChangePropagationTaskProjectionReason",
    "ChangePropagationTaskSource",
    "ChangePropagationTaskSourceError",
    "deterministic_change_propagation_task_id",
    "project_change_propagation_task",
]
