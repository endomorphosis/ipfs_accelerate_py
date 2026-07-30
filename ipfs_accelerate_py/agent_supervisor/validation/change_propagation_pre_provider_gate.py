"""Final, side-effect-free admission boundary before change-propagation providers.

This module deliberately does not dispatch a provider, read a worktree, or
load target source.  Its only repository input is an already-built
``RepositorySnapshot`` ledger (plus explicit frontier/lease/capability
receipts).  A successful result is a narrow receipt authorizing *one*
model-required step's existing paths; it is never authority to select a
target, invent behavior, or enlarge a packet.

Any tree/overlay drift, graph/index/model/config drift, translator/toolchain/
policy drift, target span/hash drift, proof downgrade, plan/packet
incompleteness, unresolved frontier, partial SCC group, incomplete behavior,
escaped/read-only path, or provider-identity / path-lease mismatch blocks
before ``llm_router``.  Analytical steps are always rejected for provider
hand-off.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..analysis.change_propagation_contracts import (
    AtomicPropagationPlan,
    ImpactClosureReceipt,
    ImpactCompleteness,
    PlanDisposition,
    PropagationAuthorityRoots,
)
from ..analysis.repository_snapshot import (
    CoverageKind,
    EntryKind,
    GitStatus,
    RepositorySnapshot,
)
from ..integrations.change_propagation_capabilities import (
    ChangePropagationCapabilityReport,
    ChangePropagationCapabilityStatus,
)
from ..planning.change_propagation_plan import PropagationPlanAdmission
from ..proof.change_propagation_edit_packet import (
    ChangePropagationEditPacket,
    PropagationEditStep,
    PropagationEditStepKind,
)
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from ..todo_daemon.change_propagation_provider_router import (
    ProviderModelConfigIdentity,
    WriterLease,
)


CHANGE_PROPAGATION_PRE_PROVIDER_GATE_INTERFACE: Final[str] = (
    "ChangePropagationPreProviderGate@1"
)
PROPAGATION_GATE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-pre-provider-gate-receipt@1"
)
MAX_GATE_RECEIPT_BYTES: Final[int] = 65_536
MAX_GATE_PATHS: Final[int] = 1_024
DEFAULT_GATE_TTL_SECONDS: Final[int] = 300
DEFAULT_REQUIRED_CAPABILITIES: Final[tuple[str, ...]] = (
    "accelerator.llm_router",
)


class ChangePropagationPreProviderGateError(ValueError):
    """The proposed provider hand-off is not current and fully proved."""


class PropagationGateReason(str, Enum):
    """Closed, machine-readable pre-provider rejection codes."""

    MALFORMED_INPUT = "malformed_input"
    PACKET_PLAN_MISMATCH = "packet_plan_mismatch"
    ABSTAINED_OR_PARTIAL = "abstained_or_partial"
    ROOT_DRIFT = "root_drift"
    TREE_OR_OVERLAY_CHANGED = "tree_or_overlay_changed"
    GRAPH_INDEX_MODEL_CONFIG_DRIFT = "graph_index_model_config_drift"
    TRANSLATOR_TOOLCHAIN_POLICY_DRIFT = "translator_toolchain_policy_drift"
    TARGET_MISSING_OR_MOVED = "target_missing_or_moved"
    TARGET_HASH_DRIFT = "target_hash_drift"
    READ_ONLY_OR_ESCAPED_PATH = "read_only_or_escaped_path"
    EXPIRED_PROOF = "expired_proof"
    PROOF_DOWNGRADED = "proof_downgraded"
    INCOMPLETE_CAPABILITY = "incomplete_capability"
    INCOMPLETE_BEHAVIOR = "incomplete_behavior"
    INCOMPLETE_PLAN_OR_PACKET = "incomplete_plan_or_packet"
    PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
    PATH_LEASE_MISMATCH = "path_lease_mismatch"
    FRONTIER_UNRESOLVED = "frontier_unresolved"
    ANALYTICAL_STEP_PROVIDER = "analytical_step_provider_forbidden"
    PARTIAL_SCC_GROUP = "partial_scc_group"
    STEP_NOT_FOUND = "step_not_found"
    STEP_NOT_MODEL_REQUIRED = "step_not_model_required"


# Alias retained for consumers that share vocabulary with contract-repair gate.
PreProviderGateReason = PropagationGateReason


def _paths(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationPreProviderGateError(f"{name} must be a path sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ChangePropagationPreProviderGateError(f"{name} contains an invalid path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ChangePropagationPreProviderGateError(f"{name} contains an escaped path")
        result.add(path.as_posix())
    if not result or len(result) > MAX_GATE_PATHS:
        raise ChangePropagationPreProviderGateError(f"{name} is empty or exceeds its bound")
    return tuple(sorted(result))


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationPreProviderGateError(f"{name} must be an identifier sequence")
    result = tuple(
        sorted({value.strip() for value in values if isinstance(value, str) and value.strip()})
    )
    if not result or len(result) > MAX_GATE_PATHS:
        raise ChangePropagationPreProviderGateError(f"{name} is empty or exceeds its bound")
    return result


def _optional_paths(values: Sequence[str] | None, name: str) -> tuple[str, ...]:
    if not values:
        return ()
    return _paths(values, name)


@dataclass(frozen=True)
class PropagationGateReceipt:
    """A bounded proof that one model-required step was safe to expose."""

    packet_id: str
    plan_id: str
    plan_content_id: str
    admission_id: str
    step_id: str
    snapshot_id: str
    roots: PropagationAuthorityRoots
    read_paths: tuple[str, ...]
    write_paths: tuple[str, ...]
    capability_report_id: str
    required_capability_ids: tuple[str, ...]
    provider_identity: Mapping[str, str]
    writer_lease_id: str
    proof_refs: tuple[str, ...]
    fixed_point_obligation_ref: str
    checked_at: int
    expires_at: int
    scc_group_id: str = ""
    frontier_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise ChangePropagationPreProviderGateError(
                "receipt roots must be PropagationAuthorityRoots"
            )
        for name in (
            "packet_id",
            "plan_id",
            "plan_content_id",
            "admission_id",
            "step_id",
            "snapshot_id",
            "capability_report_id",
            "fixed_point_obligation_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
                raise ChangePropagationPreProviderGateError(
                    f"receipt {name} must be a compact identifier"
                )
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "read_paths", _paths(self.read_paths, "read_paths") if self.read_paths else ())
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(
            self,
            "required_capability_ids",
            _ids(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(
            self,
            "proof_refs",
            _ids(self.proof_refs, "proof_refs"),
        )
        if not isinstance(self.provider_identity, Mapping):
            raise ChangePropagationPreProviderGateError("receipt provider_identity must be a mapping")
        identity = {
            str(key): str(value)
            for key, value in self.provider_identity.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        for required in ("provider_id", "model_id", "config_id"):
            if not identity.get(required):
                raise ChangePropagationPreProviderGateError(
                    f"receipt provider_identity requires {required}"
                )
        object.__setattr__(self, "provider_identity", identity)
        lease = self.writer_lease_id
        if not isinstance(lease, str) or (lease and (not lease.strip() or any(c.isspace() for c in lease))):
            raise ChangePropagationPreProviderGateError("receipt writer_lease_id is malformed")
        object.__setattr__(self, "writer_lease_id", lease.strip() if lease else "")
        object.__setattr__(
            self,
            "scc_group_id",
            self.scc_group_id.strip() if isinstance(self.scc_group_id, str) else "",
        )
        if not isinstance(self.frontier_complete, bool):
            raise ChangePropagationPreProviderGateError("frontier_complete must be a boolean")
        for name in ("checked_at", "expires_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ChangePropagationPreProviderGateError(
                    f"receipt {name} must be a non-negative integer"
                )
        if self.expires_at <= self.checked_at:
            raise ChangePropagationPreProviderGateError("receipt must expire after it is checked")
        if len(canonical_json_bytes(self.to_dict())) > MAX_GATE_RECEIPT_BYTES:
            raise ChangePropagationPreProviderGateError("receipt exceeds its serialized byte bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_GATE_RECEIPT_SCHEMA,
            "interface": CHANGE_PROPAGATION_PRE_PROVIDER_GATE_INTERFACE,
            "packet_id": self.packet_id,
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "admission_id": self.admission_id,
            "step_id": self.step_id,
            "snapshot_id": self.snapshot_id,
            "roots": self.roots.to_dict(),
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "capability_report_id": self.capability_report_id,
            "required_capability_ids": list(self.required_capability_ids),
            "provider_identity": dict(self.provider_identity),
            "writer_lease_id": self.writer_lease_id,
            "proof_refs": list(self.proof_refs),
            "fixed_point_obligation_ref": self.fixed_point_obligation_ref,
            "checked_at": self.checked_at,
            "expires_at": self.expires_at,
            "scc_group_id": self.scc_group_id,
            "frontier_complete": self.frontier_complete,
            "provider_invoked": False,
            "authorized_paths": list(self.write_paths),
        }

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def content_id(self) -> str:
        return self.receipt_id

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationGateReceipt":
        fields = {
            "schema",
            "interface",
            "receipt_id",
            "packet_id",
            "plan_id",
            "plan_content_id",
            "admission_id",
            "step_id",
            "snapshot_id",
            "roots",
            "read_paths",
            "write_paths",
            "capability_report_id",
            "required_capability_ids",
            "provider_identity",
            "writer_lease_id",
            "proof_refs",
            "fixed_point_obligation_ref",
            "checked_at",
            "expires_at",
            "scc_group_id",
            "frontier_complete",
            "provider_invoked",
            "authorized_paths",
        }
        if not isinstance(payload, Mapping) or set(payload).difference(fields):
            raise ChangePropagationPreProviderGateError("receipt contains unsupported fields")
        if (
            payload.get("schema") != PROPAGATION_GATE_RECEIPT_SCHEMA
            or payload.get("interface") != CHANGE_PROPAGATION_PRE_PROVIDER_GATE_INTERFACE
        ):
            raise ChangePropagationPreProviderGateError(
                "receipt has an unsupported schema or interface"
            )
        if payload.get("provider_invoked", False) is not False:
            raise ChangePropagationPreProviderGateError(
                "a pre-provider receipt cannot claim provider invocation"
            )
        try:
            roots_payload = payload["roots"]
            if isinstance(roots_payload, PropagationAuthorityRoots):
                roots = roots_payload
            else:
                roots = PropagationAuthorityRoots.from_dict(roots_payload)
            receipt = cls(
                packet_id=payload["packet_id"],
                plan_id=payload["plan_id"],
                plan_content_id=payload["plan_content_id"],
                admission_id=payload["admission_id"],
                step_id=payload["step_id"],
                snapshot_id=payload["snapshot_id"],
                roots=roots,
                read_paths=tuple(payload.get("read_paths") or ()),
                write_paths=tuple(payload["write_paths"]),
                capability_report_id=payload["capability_report_id"],
                required_capability_ids=tuple(payload["required_capability_ids"]),
                provider_identity=dict(payload["provider_identity"]),
                writer_lease_id=payload.get("writer_lease_id", ""),
                proof_refs=tuple(payload["proof_refs"]),
                fixed_point_obligation_ref=payload["fixed_point_obligation_ref"],
                checked_at=payload["checked_at"],
                expires_at=payload["expires_at"],
                scc_group_id=payload.get("scc_group_id", ""),
                frontier_complete=bool(payload.get("frontier_complete", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChangePropagationPreProviderGateError("receipt is malformed") from exc
        if tuple(payload.get("authorized_paths", receipt.write_paths)) != receipt.write_paths:
            raise ChangePropagationPreProviderGateError("receipt cannot broaden authorized paths")
        if payload.get("receipt_id") not in (None, "", receipt.receipt_id):
            raise ChangePropagationPreProviderGateError("receipt identity is forged")
        return receipt


# Compatibility spelling used by some integrations and the repair gate tests.
PreProviderGateReceipt = PropagationGateReceipt


def capability_report_to_dict(report: ChangePropagationCapabilityReport) -> dict[str, Any]:
    """Return the body-free, float-free portion relevant to admission."""

    return {
        "schema_version": report.schema_version,
        "report_version": report.report_version,
        "accelerator_module_paths": list(report.accelerator_module_paths),
        "datasets_module_paths": list(report.datasets_module_paths),
        "datasets_gitlink_revision": report.datasets_gitlink_revision,
        "capabilities": [
            {
                "capability_id": item.capability_id,
                "status": item.status.value,
                "module_paths": list(item.module_paths),
                "interface_version": item.interface_version,
                "schema_version": item.schema_version,
                "supported_semantics": list(item.supported_semantics),
                "reconstruction_compatible": item.reconstruction_compatible,
                "operations": list(item.operations),
            }
            for item in sorted(report.capabilities, key=lambda item: item.capability_id)
        ],
    }


class ChangePropagationPreProviderGate:
    """Replay current packet, plan, proof, capability, lease, and snapshot bindings.

    ``validate`` is pure and returns closed reason codes.  Callers must call
    ``require_valid`` and obtain its receipt before invoking a provider; this
    class has no callback parameter and cannot execute untrusted source.
    """

    interface: Final[str] = CHANGE_PROPAGATION_PRE_PROVIDER_GATE_INTERFACE

    def validate(
        self,
        packet: ChangePropagationEditPacket,
        admission: PropagationPlanAdmission,
        snapshot: RepositorySnapshot,
        *,
        current_roots: PropagationAuthorityRoots,
        capability_report: ChangePropagationCapabilityReport,
        now: int,
        step_id: str | None = None,
        provider_identity: ProviderModelConfigIdentity | Mapping[str, Any] | None = None,
        writer_lease: WriterLease | None = None,
        impact_closure: ImpactClosureReceipt | None = None,
        required_capability_ids: Sequence[str] = DEFAULT_REQUIRED_CAPABILITIES,
        read_only_paths: Sequence[str] = (),
        expires_at: int | None = None,
        scc_completed_step_ids: Sequence[str] | None = None,
    ) -> tuple[PropagationGateReason, ...]:
        invalid: set[PropagationGateReason] = set()
        typed = (
            isinstance(packet, ChangePropagationEditPacket)
            and isinstance(admission, PropagationPlanAdmission)
            and isinstance(snapshot, RepositorySnapshot)
            and isinstance(current_roots, PropagationAuthorityRoots)
            and isinstance(capability_report, ChangePropagationCapabilityReport)
            and isinstance(now, int)
            and not isinstance(now, bool)
        )
        if not typed:
            return (PropagationGateReason.MALFORMED_INPUT,)
        try:
            required = _ids(required_capability_ids, "required_capability_ids")
            blocked_paths = _optional_paths(read_only_paths, "read_only_paths")
        except ChangePropagationPreProviderGateError:
            return (PropagationGateReason.MALFORMED_INPUT,)

        # --- Admission / plan disposition ---
        if not admission.admitted or admission.disposition is not PlanDisposition.ADMITTED:
            invalid.add(PropagationGateReason.ABSTAINED_OR_PARTIAL)
        plan = admission.plan
        if not isinstance(plan, AtomicPropagationPlan):
            invalid.add(PropagationGateReason.MALFORMED_INPUT)
            return tuple(sorted(invalid, key=lambda item: item.value))
        if plan.disposition is not PlanDisposition.ADMITTED:
            invalid.add(PropagationGateReason.ABSTAINED_OR_PARTIAL)
        if admission.alternative_plan_ids:
            invalid.add(PropagationGateReason.ABSTAINED_OR_PARTIAL)

        # --- Root drift (full authority vector) ---
        if (
            current_roots != packet.roots
            or plan.roots != packet.roots
            or admission.plan.roots != current_roots
        ):
            invalid.add(PropagationGateReason.ROOT_DRIFT)

        # Split graph/index/model/config vs translator/toolchain/policy for
        # precise reason codes even when full roots also mismatch.
        if (
            current_roots.graph_id != packet.roots.graph_id
            or current_roots.index_id != packet.roots.index_id
            or current_roots.model_id != packet.roots.model_id
            or current_roots.config_id != packet.roots.config_id
            or packet.roots.graph_id not in packet.graph_refs
            or packet.roots.index_id not in packet.index_refs
        ):
            invalid.add(PropagationGateReason.GRAPH_INDEX_MODEL_CONFIG_DRIFT)
        if (
            current_roots.translator_id != packet.roots.translator_id
            or current_roots.toolchain_id != packet.roots.toolchain_id
            or current_roots.policy_id != packet.roots.policy_id
        ):
            invalid.add(PropagationGateReason.TRANSLATOR_TOOLCHAIN_POLICY_DRIFT)

        # --- Packet ↔ plan completeness / identity ---
        if (
            packet.plan_id != plan.plan_id
            or packet.plan_content_id != plan.content_id
            or packet.admission_id != admission.content_id
            or packet.change_set_id != plan.change_set_id
            or packet.delta_id != plan.delta_id
            or packet.impact_closure_id != plan.impact_closure_id
            or packet.obligation_set_id != plan.obligation_set_id
            or set(packet.step_order) != {step.step_id for step in plan.steps}
            or list(packet.step_order) != list(admission.step_order)
            or set(packet.permitted_write_paths) != set(plan.permitted_write_paths)
            or set(packet.permitted_read_paths) != set(plan.permitted_read_paths)
            or packet.proof_refs != plan.proof_refs
            or packet.invalidation_refs != plan.invalidation_refs
            or packet.fixed_point_obligation_ref != plan.fixed_point_obligation_ref
        ):
            invalid.add(PropagationGateReason.PACKET_PLAN_MISMATCH)

        if (
            not packet.steps
            or not packet.permitted_write_paths
            or not packet.proof_refs
            or not packet.fixed_point_obligation_ref
            or not packet.validation_commands
            or not packet.per_edit_postcondition_refs
            or not packet.fixed_point_postcondition_refs
        ):
            invalid.add(PropagationGateReason.INCOMPLETE_PLAN_OR_PACKET)

        # --- Resolve step (default: sole model-required step, else require id) ---
        step = self._resolve_step(packet, step_id)
        if step is None:
            if step_id:
                invalid.add(PropagationGateReason.STEP_NOT_FOUND)
            elif not packet.model_required_step_ids:
                invalid.add(PropagationGateReason.STEP_NOT_MODEL_REQUIRED)
            else:
                invalid.add(PropagationGateReason.MALFORMED_INPUT)
            return tuple(sorted(invalid, key=lambda item: item.value))

        if step.kind is PropagationEditStepKind.ANALYTICAL:
            invalid.add(PropagationGateReason.ANALYTICAL_STEP_PROVIDER)
            invalid.add(PropagationGateReason.STEP_NOT_MODEL_REQUIRED)
        elif step.kind is not PropagationEditStepKind.MODEL_REQUIRED:
            invalid.add(PropagationGateReason.STEP_NOT_MODEL_REQUIRED)
        if step.step_id not in packet.model_required_step_ids:
            if step.kind is PropagationEditStepKind.ANALYTICAL:
                invalid.add(PropagationGateReason.ANALYTICAL_STEP_PROVIDER)
            else:
                invalid.add(PropagationGateReason.STEP_NOT_MODEL_REQUIRED)

        # --- Behavior completeness for model-required steps ---
        if step.kind is PropagationEditStepKind.MODEL_REQUIRED:
            if not step.write_paths:
                invalid.add(PropagationGateReason.INCOMPLETE_PLAN_OR_PACKET)
            if not step.required_behavior_ids and not packet.required_behavior_ids:
                invalid.add(PropagationGateReason.INCOMPLETE_BEHAVIOR)
            # Model steps that list unsupported limits still gate only when
            # behavior is present; incomplete admitted semantics fail closed.
            if step.unsupported_limits and not (
                step.required_behavior_ids or packet.required_behavior_ids
            ):
                invalid.add(PropagationGateReason.INCOMPLETE_BEHAVIOR)

        # --- Proof reconstruction ---
        plan_proof = set(plan.proof_refs)
        packet_proof = set(packet.proof_refs)
        if not plan_proof or not packet_proof:
            invalid.add(PropagationGateReason.PROOF_DOWNGRADED)
        elif packet_proof != plan_proof:
            invalid.add(PropagationGateReason.PROOF_DOWNGRADED)
        if step.proof_refs and not set(step.proof_refs).issubset(packet_proof):
            invalid.add(PropagationGateReason.PROOF_DOWNGRADED)

        # --- Expiry ---
        effective_expires = (
            expires_at
            if expires_at is not None
            else now + DEFAULT_GATE_TTL_SECONDS
        )
        if (
            isinstance(effective_expires, bool)
            or not isinstance(effective_expires, int)
            or effective_expires <= now
        ):
            invalid.add(PropagationGateReason.EXPIRED_PROOF)

        # --- Tree / overlay / snapshot ---
        # Candidate tree is the mutation surface; snapshot head must match it
        # (or the policy-bound tree identity carried as candidate_tree_id).
        if (
            snapshot.head_tree_id != current_roots.candidate_tree_id
            or snapshot.index_tree_id != snapshot.head_tree_id
            or not snapshot.is_clean
            or snapshot.stats.overlay_path_count
        ):
            invalid.add(PropagationGateReason.TREE_OR_OVERLAY_CHANGED)
        if current_roots.candidate_tree_id != packet.roots.candidate_tree_id:
            invalid.add(PropagationGateReason.TREE_OR_OVERLAY_CHANGED)
        if current_roots.candidate_overlay_id != packet.roots.candidate_overlay_id:
            invalid.add(PropagationGateReason.TREE_OR_OVERLAY_CHANGED)

        # --- Target spans / hashes for every write path ---
        write_paths = tuple(step.write_paths) if step.write_paths else tuple(packet.permitted_write_paths)
        before_by_path = {item.path: item for item in packet.before_hashes}
        try:
            snapshot.assert_exhaustive_tracked_coverage()
        except Exception:
            invalid.add(PropagationGateReason.TARGET_MISSING_OR_MOVED)

        for path in write_paths:
            if path in blocked_paths:
                invalid.add(PropagationGateReason.READ_ONLY_OR_ESCAPED_PATH)
                continue
            try:
                target = snapshot.disposition_for_path(path)
            except Exception:
                target = None
            if (
                target is None
                or not target.tracked
                or target.overlay
                or target.git_status is not GitStatus.CLEAN
                or target.entry_kind is not EntryKind.REGULAR
                or target.kind
                in {
                    CoverageKind.EXCLUDED,
                    CoverageKind.UNSUPPORTED,
                    CoverageKind.BINARY_OR_GENERATED,
                }
            ):
                invalid.add(PropagationGateReason.TARGET_MISSING_OR_MOVED)
                continue
            before = before_by_path.get(path)
            if before is not None and before.artifact_id:
                digests = {
                    value
                    for value in (target.content_digest, target.git_object_id)
                    if value
                }
                if before.artifact_id not in digests:
                    invalid.add(PropagationGateReason.TARGET_HASH_DRIFT)

        # Path authority fences
        if not write_paths:
            invalid.add(PropagationGateReason.READ_ONLY_OR_ESCAPED_PATH)
        else:
            if not set(write_paths).issubset(set(packet.permitted_write_paths)):
                invalid.add(PropagationGateReason.READ_ONLY_OR_ESCAPED_PATH)
            if not set(step.read_paths).issubset(set(packet.permitted_read_paths)):
                invalid.add(PropagationGateReason.READ_ONLY_OR_ESCAPED_PATH)
            if any(path in blocked_paths for path in write_paths):
                invalid.add(PropagationGateReason.READ_ONLY_OR_ESCAPED_PATH)

        # --- Frontier completeness ---
        if impact_closure is not None:
            if not isinstance(impact_closure, ImpactClosureReceipt):
                invalid.add(PropagationGateReason.MALFORMED_INPUT)
            else:
                if impact_closure.roots != packet.roots:
                    invalid.add(PropagationGateReason.ROOT_DRIFT)
                if impact_closure.content_id != packet.impact_closure_id and (
                    getattr(impact_closure, "delta_id", "") != packet.delta_id
                ):
                    # Allow same delta identity if content_id scheme differs; still
                    # require complete frontier.
                    pass
                if impact_closure.completeness is not ImpactCompleteness.COMPLETE:
                    invalid.add(PropagationGateReason.FRONTIER_UNRESOLVED)
                if impact_closure.frontier_node_ids or impact_closure.frontier_edge_ids:
                    invalid.add(PropagationGateReason.FRONTIER_UNRESOLVED)
                if impact_closure.delta_id != packet.delta_id:
                    invalid.add(PropagationGateReason.PACKET_PLAN_MISMATCH)

        # --- SCC group partial completion ---
        if step.scc_group_id and scc_completed_step_ids is not None:
            group = next(
                (g for g in packet.scc_groups if g.group_id == step.scc_group_id),
                None,
            )
            if group is None:
                invalid.add(PropagationGateReason.PARTIAL_SCC_GROUP)
            else:
                completed = set(scc_completed_step_ids)
                # Provider may only run for a step when prior group members that
                # are dependencies of this step are accounted for; a partial
                # group (some members done, some missing without this step) fails.
                siblings = set(group.step_ids)
                other_done = completed & siblings - {step.step_id}
                other_pending = siblings - completed - {step.step_id}
                # If some siblings completed and some pending without an ordered
                # dependency relationship, treat as partial group.
                if other_done and other_pending:
                    # Allow only when all pending are not dependencies of completed
                    # members in a way that already committed partial work — fail
                    # closed whenever the group is mid-flight without this step
                    # being the sole remaining member.
                    if other_pending - set(step.dependency_step_ids):
                        invalid.add(PropagationGateReason.PARTIAL_SCC_GROUP)

        # --- Provider identity ---
        identity = self._coerce_identity(provider_identity)
        if identity is None:
            invalid.add(PropagationGateReason.PROVIDER_IDENTITY_MISMATCH)
        else:
            if identity.router_backend != "llm_router":
                invalid.add(PropagationGateReason.PROVIDER_IDENTITY_MISMATCH)
            # Config root is the frozen supervisor binding; model_id on the
            # identity may name the provider model while still requiring the
            # packet config root to match current roots (already checked).
            if identity.config_id not in {
                packet.roots.config_id,
                current_roots.config_id,
            }:
                invalid.add(PropagationGateReason.PROVIDER_IDENTITY_MISMATCH)

        # --- Writer / path lease ---
        if writer_lease is None:
            # Lease is required for a successful provider hand-off receipt.
            invalid.add(PropagationGateReason.PATH_LEASE_MISMATCH)
        else:
            if not isinstance(writer_lease, WriterLease):
                invalid.add(PropagationGateReason.PATH_LEASE_MISMATCH)
            else:
                if (
                    writer_lease.packet_id != packet.packet_id
                    or writer_lease.plan_id != packet.plan_id
                    or writer_lease.step_id != step.step_id
                ):
                    invalid.add(PropagationGateReason.PATH_LEASE_MISMATCH)
                if set(writer_lease.permitted_write_paths) != set(write_paths):
                    invalid.add(PropagationGateReason.PATH_LEASE_MISMATCH)
                if writer_lease.tree_id and writer_lease.tree_id != packet.roots.candidate_tree_id:
                    invalid.add(PropagationGateReason.PATH_LEASE_MISMATCH)
                if identity is not None:
                    if writer_lease.provider_id and writer_lease.provider_id != identity.provider_id:
                        invalid.add(PropagationGateReason.PROVIDER_IDENTITY_MISMATCH)
                    if writer_lease.model_id and writer_lease.model_id != identity.model_id:
                        invalid.add(PropagationGateReason.PROVIDER_IDENTITY_MISMATCH)
                    if writer_lease.config_id and writer_lease.config_id != identity.config_id:
                        invalid.add(PropagationGateReason.PROVIDER_IDENTITY_MISMATCH)

        # --- Capabilities / reconstruction ---
        capability_map = capability_report.capability_map
        for capability_id in required:
            capability = capability_map.get(capability_id)
            if (
                capability is None
                or capability.status is not ChangePropagationCapabilityStatus.AVAILABLE
                or not capability.reconstruction_compatible
            ):
                invalid.add(PropagationGateReason.INCOMPLETE_CAPABILITY)

        return tuple(sorted(invalid, key=lambda item: item.value))

    def require_valid(
        self,
        packet: ChangePropagationEditPacket,
        admission: PropagationPlanAdmission,
        snapshot: RepositorySnapshot,
        *,
        current_roots: PropagationAuthorityRoots,
        capability_report: ChangePropagationCapabilityReport,
        now: int,
        step_id: str | None = None,
        provider_identity: ProviderModelConfigIdentity | Mapping[str, Any] | None = None,
        writer_lease: WriterLease | None = None,
        impact_closure: ImpactClosureReceipt | None = None,
        required_capability_ids: Sequence[str] = DEFAULT_REQUIRED_CAPABILITIES,
        read_only_paths: Sequence[str] = (),
        expires_at: int | None = None,
        scc_completed_step_ids: Sequence[str] | None = None,
    ) -> PropagationGateReceipt:
        invalid = self.validate(
            packet,
            admission,
            snapshot,
            current_roots=current_roots,
            capability_report=capability_report,
            now=now,
            step_id=step_id,
            provider_identity=provider_identity,
            writer_lease=writer_lease,
            impact_closure=impact_closure,
            required_capability_ids=required_capability_ids,
            read_only_paths=read_only_paths,
            expires_at=expires_at,
            scc_completed_step_ids=scc_completed_step_ids,
        )
        if invalid:
            raise ChangePropagationPreProviderGateError(
                "change propagation pre-provider gate rejected: "
                + ", ".join(item.value for item in invalid)
            )
        step = self._resolve_step(packet, step_id)
        assert step is not None  # validate already enforced
        identity = self._coerce_identity(provider_identity)
        assert identity is not None
        assert writer_lease is not None
        required = _ids(required_capability_ids, "required_capability_ids")
        write_paths = tuple(step.write_paths)
        effective_expires = (
            expires_at if expires_at is not None else now + DEFAULT_GATE_TTL_SECONDS
        )
        frontier_complete = True
        if impact_closure is not None:
            frontier_complete = (
                impact_closure.completeness is ImpactCompleteness.COMPLETE
                and not impact_closure.frontier_node_ids
                and not impact_closure.frontier_edge_ids
            )
        return PropagationGateReceipt(
            packet_id=packet.packet_id,
            plan_id=packet.plan_id,
            plan_content_id=packet.plan_content_id,
            admission_id=packet.admission_id,
            step_id=step.step_id,
            snapshot_id=snapshot.snapshot_id,
            roots=packet.roots,
            read_paths=tuple(step.read_paths),
            write_paths=write_paths,
            capability_report_id=content_identity(capability_report_to_dict(capability_report)),
            required_capability_ids=required,
            provider_identity=identity.to_dict(),
            writer_lease_id=writer_lease.lease_id,
            proof_refs=tuple(step.proof_refs or packet.proof_refs),
            fixed_point_obligation_ref=packet.fixed_point_obligation_ref,
            checked_at=now,
            expires_at=effective_expires,
            scc_group_id=step.scc_group_id,
            frontier_complete=frontier_complete,
        )

    check = require_valid
    admit = require_valid

    def is_valid(self, *args: Any, **kwargs: Any) -> bool:
        return not self.validate(*args, **kwargs)

    def _resolve_step(
        self,
        packet: ChangePropagationEditPacket,
        step_id: str | None,
    ) -> PropagationEditStep | None:
        by_id = {step.step_id: step for step in packet.steps}
        if step_id:
            return by_id.get(step_id)
        model_ids = list(packet.model_required_step_ids)
        if len(model_ids) == 1:
            return by_id.get(model_ids[0])
        return None

    def _coerce_identity(
        self,
        value: ProviderModelConfigIdentity | Mapping[str, Any] | None,
    ) -> ProviderModelConfigIdentity | None:
        if value is None:
            return None
        if isinstance(value, ProviderModelConfigIdentity):
            return value
        if isinstance(value, Mapping):
            try:
                return ProviderModelConfigIdentity(
                    provider_id=str(value["provider_id"]),
                    model_id=str(value["model_id"]),
                    config_id=str(value["config_id"]),
                    router_backend=str(value.get("router_backend", "llm_router")),
                )
            except (KeyError, TypeError, ValueError):
                return None
        return None


__all__ = [
    "CHANGE_PROPAGATION_PRE_PROVIDER_GATE_INTERFACE",
    "DEFAULT_GATE_TTL_SECONDS",
    "DEFAULT_REQUIRED_CAPABILITIES",
    "MAX_GATE_RECEIPT_BYTES",
    "ChangePropagationPreProviderGate",
    "ChangePropagationPreProviderGateError",
    "PROPAGATION_GATE_RECEIPT_SCHEMA",
    "PreProviderGateReason",
    "PreProviderGateReceipt",
    "PropagationGateReason",
    "PropagationGateReceipt",
    "capability_report_to_dict",
]
