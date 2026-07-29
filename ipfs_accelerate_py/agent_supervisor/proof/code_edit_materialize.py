"""Materialize CodeEditPacket@1 into supervisor tasks (CBP-080).

Generalizes the plateau materializer pattern:

* open claims / obligations → :class:`CodeEditPacket`
* materializer emits ``validation_commands`` for focused tests, domain
  metrics, and cache-aware re-proof at the declared assurance
* cache and claim status are recorded without embedding full proof bodies
* optional PlateauCodexPacket bridge projects residual / packet handles only
  (gold IR bodies are rejected from cache keys and packet metadata)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .code_claim_contracts import ClaimStatus, CodeClaimRecord
from .code_edit_packet import (
    CODE_EDIT_PACKET_INTERFACE,
    CacheDisposition,
    CacheStatusRecord,
    ClaimStatusRecord,
    CodeEditPacket,
    CodeEditPacketError,
    NonImplementableReason,
    ProverBinding,
    build_code_edit_packet,
    compute_implementable,
)
from .code_proof_obligations import (
    CodeProofObligationCompilation,
    CompiledCodeProofItem,
    ObligationCompileStatus,
)
from .code_proof_query import (
    ClaimQueryHit,
    CodeProofQuery,
    CodeProofQueryResult,
    build_code_proof_query,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    content_identity,
)
from ..validation.validation_commands import (
    ValidationCommand,
    ValidationStage,
    ValidationVerdictKind,
    build_validation_commands,
)


CODE_EDIT_MATERIALIZE_INTERFACE: Final = "CodeEditMaterialize@1"
CODE_EDIT_MATERIALIZE_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-edit-materialize-report@1"
)
CODE_EDIT_SUPERVISOR_TASK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-edit-supervisor-task@1"
)
CODE_EDIT_MATERIALIZE_VERSION: Final = "1"

# Validation command kind markers (stable, machine-readable).
VALIDATION_KIND_TEST: Final = "test"
VALIDATION_KIND_DOMAIN_METRICS: Final = "domain_metrics"
VALIDATION_KIND_CACHE_AWARE_REPROOF: Final = "cache_aware_reproof"

DEFAULT_TEST_MODULE: Final = "test/api/test_agent_supervisor_code_edit_packet.py"
DEFAULT_DOMAIN_METRICS_MODULE: Final = (
    "test/api/test_agent_supervisor_code_proof_self_properties.py"
)

# Plateau / SRT bridge — handles only; never gold bodies.
_GOLD_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "gold_ir",
        "gold_ir_body",
        "gold_source",
        "full_ir",
        "proof_body",
        "lean_source_body",
    }
)


class CodeEditMaterializeError(CodeEditPacketError):
    """Materialization input is malformed or unsafe."""


def _sorted_unique(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        text = str(raw or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(sorted(out))


def _assurance_value(value: Any) -> AssuranceLevel:
    if isinstance(value, AssuranceLevel):
        return value
    text = str(value or AssuranceLevel.KERNEL_VERIFIED.value).strip()
    try:
        return AssuranceLevel(text)
    except ValueError as exc:
        raise CodeEditMaterializeError(
            f"invalid assurance level: {text!r}"
        ) from exc


def _reject_gold_bodies(payload: Mapping[str, Any], *, where: str) -> None:
    for key, raw in payload.items():
        key_l = str(key).lower()
        if key_l in _GOLD_BODY_MARKERS or any(m in key_l for m in _GOLD_BODY_MARKERS):
            raise CodeEditMaterializeError(
                f"{where} must not couple gold IR / proof bodies into cache keys "
                f"or packet metadata ({key})"
            )
        if isinstance(raw, Mapping):
            _reject_gold_bodies(raw, where=where)


def emit_validation_commands(
    *,
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    predicted_files: Sequence[str] = (),
    property_ids: Sequence[str] = (),
    obligation_ids: Sequence[str] = (),
    test_module: str = DEFAULT_TEST_MODULE,
    domain_metrics_module: str = DEFAULT_DOMAIN_METRICS_MODULE,
    cache_aware: bool = True,
) -> tuple[str, ...]:
    """Emit ordered shell validation commands for an implementable packet.

    Order (acceptance):

    1. focused tests
    2. domain metrics
    3. cache-aware re-proof at declared assurance
    """

    assurance = _assurance_value(required_assurance)
    files = _sorted_unique(predicted_files)
    props = _sorted_unique(property_ids)
    obligations = _sorted_unique(obligation_ids)

    test_cmd = f"python -m pytest {test_module} -q"
    if files:
        # Keep the module primary; path hints ride as comments are not shell-safe.
        # Materializer records predicted files on the task separately.
        pass

    metrics_cmd = f"python -m pytest {domain_metrics_module} -q --tb=no"

    prop_csv = ",".join(props) if props else ""
    obl_csv = ",".join(obligations) if obligations else ""
    cache_flag = "cache-aware=true" if cache_aware else "cache-aware=false"
    reproof_cmd = (
        "python -m ipfs_accelerate_py.agent_supervisor.code_proof_reproof "
        f"--assurance {assurance.value} "
        f"--{cache_flag} "
        f"--properties {prop_csv or '*'} "
        f"--obligations {obl_csv or '*'}"
    )

    return (test_cmd, metrics_cmd, reproof_cmd)


def emit_validation_command_specs(
    *,
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    predicted_files: Sequence[str] = (),
    property_ids: Sequence[str] = (),
    obligation_ids: Sequence[str] = (),
    test_module: str = DEFAULT_TEST_MODULE,
    domain_metrics_module: str = DEFAULT_DOMAIN_METRICS_MODULE,
    cache_aware: bool = True,
) -> tuple[ValidationCommand, ...]:
    """Typed :class:`ValidationCommand` specs matching :func:`emit_validation_commands`."""

    commands = emit_validation_commands(
        required_assurance=required_assurance,
        predicted_files=predicted_files,
        property_ids=property_ids,
        obligation_ids=obligation_ids,
        test_module=test_module,
        domain_metrics_module=domain_metrics_module,
        cache_aware=cache_aware,
    )
    stages = (
        ValidationStage.FOCUSED,
        ValidationStage.FOCUSED,
        ValidationStage.KERNEL,
    )
    verdicts = (
        ValidationVerdictKind.TEST,
        ValidationVerdictKind.TEST,
        ValidationVerdictKind.KERNEL,
    )
    kinds = (
        VALIDATION_KIND_TEST,
        VALIDATION_KIND_DOMAIN_METRICS,
        VALIDATION_KIND_CACHE_AWARE_REPROOF,
    )
    impact = tuple(predicted_files) if predicted_files else ()
    specs: list[ValidationCommand] = []
    for ordinal, (cmd, stage, verdict, kind) in enumerate(
        zip(commands, stages, verdicts, kinds)
    ):
        specs.append(
            ValidationCommand(
                command=cmd,
                raw_command=cmd,
                stage=stage,
                resource_cost=1 if ordinal < 2 else 4,
                impact_paths=impact,
                cacheable=True,
                ordinal=ordinal,
                validation_id=f"cbp-080:{kind}",
                verdict_kind=verdict,
                source=kind,
            )
        )
    return build_validation_commands(specs)


@dataclass(frozen=True)
class CodeEditSupervisorTask:
    """One supervisor-facing task projected from a CodeEditPacket."""

    task_id: str
    packet: CodeEditPacket
    validation_commands: tuple[str, ...]
    title: str = ""
    implementable: bool = True
    predicted_files: tuple[str, ...] = ()
    acceptance_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        if not self.task_id:
            raise CodeEditMaterializeError("task_id is required")
        if not isinstance(self.packet, CodeEditPacket):
            raise CodeEditMaterializeError("packet must be a CodeEditPacket")
        cmds = tuple(str(c).strip() for c in (self.validation_commands or ()) if str(c).strip())
        object.__setattr__(self, "validation_commands", cmds)
        object.__setattr__(self, "title", str(self.title or "").strip())
        object.__setattr__(self, "implementable", bool(self.implementable))
        object.__setattr__(
            self, "predicted_files", _sorted_unique(self.predicted_files)
        )
        object.__setattr__(
            self, "acceptance_ids", _sorted_unique(self.acceptance_ids)
        )
        object.__setattr__(self, "notes", _sorted_unique(self.notes))
        if not isinstance(self.metadata, Mapping):
            raise CodeEditMaterializeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CODE_EDIT_SUPERVISOR_TASK_SCHEMA,
            "interface": CODE_EDIT_MATERIALIZE_INTERFACE,
            "task_id": self.task_id,
            "title": self.title,
            "implementable": bool(self.implementable),
            "packet": self.packet.to_dict(include_id=True),
            "validation_commands": list(self.validation_commands),
            "predicted_files": list(self.predicted_files),
            "acceptance_ids": list(self.acceptance_ids),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if include_id:
            payload["content_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "content_id"}
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeEditSupervisorTask":
        if not isinstance(payload, Mapping):
            raise CodeEditMaterializeError("supervisor task must be an object")
        packet_payload = payload.get("packet")
        if not isinstance(packet_payload, Mapping):
            raise CodeEditMaterializeError("supervisor task requires a packet object")
        return cls(
            task_id=str(payload.get("task_id") or ""),
            packet=CodeEditPacket.from_dict(packet_payload),
            validation_commands=tuple(payload.get("validation_commands") or ()),
            title=str(payload.get("title") or ""),
            implementable=bool(payload.get("implementable", True)),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            acceptance_ids=tuple(payload.get("acceptance_ids") or ()),
            notes=tuple(payload.get("notes") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CodeEditMaterializeReport:
    """Content-addressed report for a materialization pass."""

    repository_tree_id: str
    packets: tuple[CodeEditPacket, ...]
    tasks: tuple[CodeEditSupervisorTask, ...]
    implementable_count: int = 0
    blocked_count: int = 0
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_tree_id", str(self.repository_tree_id or "").strip()
        )
        object.__setattr__(self, "packets", tuple(self.packets or ()))
        object.__setattr__(self, "tasks", tuple(self.tasks or ()))
        object.__setattr__(self, "notes", _sorted_unique(self.notes))
        if not isinstance(self.metadata, Mapping):
            raise CodeEditMaterializeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        impl = sum(1 for p in self.packets if p.implementable)
        blocked = len(self.packets) - impl
        object.__setattr__(self, "implementable_count", impl)
        object.__setattr__(self, "blocked_count", blocked)

    @property
    def report_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CODE_EDIT_MATERIALIZE_REPORT_SCHEMA,
            "interface": CODE_EDIT_MATERIALIZE_INTERFACE,
            "version": CODE_EDIT_MATERIALIZE_VERSION,
            "repository_tree_id": self.repository_tree_id,
            "packets": [p.to_dict(include_id=True) for p in self.packets],
            "tasks": [t.to_dict(include_id=True) for t in self.tasks],
            "implementable_count": int(self.implementable_count),
            "blocked_count": int(self.blocked_count),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if include_id:
            payload["report_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "report_id"}
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeEditMaterializeReport":
        if not isinstance(payload, Mapping):
            raise CodeEditMaterializeError("materialize report must be an object")
        packets = tuple(
            CodeEditPacket.from_dict(item)
            for item in (payload.get("packets") or ())
        )
        tasks = tuple(
            CodeEditSupervisorTask.from_dict(item)
            for item in (payload.get("tasks") or ())
        )
        return cls(
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            packets=packets,
            tasks=tasks,
            notes=tuple(payload.get("notes") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )


def _claim_status_from_record(claim: CodeClaimRecord) -> ClaimStatusRecord:
    return ClaimStatusRecord(
        claim_id=str(claim.claim_id),
        property_id=str(claim.property_id or ""),
        status=claim.status.value
        if isinstance(claim.status, ClaimStatus)
        else str(claim.status),
        obligation_id=str(claim.obligation_id or ""),
        evidence_ids=tuple(claim.evidence_ids or ()),
        evidence_tiers=tuple(
            str(getattr(t, "value", t)) for t in (claim.evidence_tiers or ())
        ),
        required_assurance=claim.required_assurance,
        derived_assurance=claim.derived_assurance,
    )


def _claim_status_from_hit(hit: ClaimQueryHit) -> ClaimStatusRecord:
    status = hit.status.value if isinstance(hit.status, ClaimStatus) else str(hit.status)
    obl = hit.obligation_ids[0] if hit.obligation_ids else ""
    return ClaimStatusRecord(
        claim_id=str(hit.claim_id or ""),
        property_id=str(hit.property_id or ""),
        status=status,
        obligation_id=obl,
        evidence_ids=tuple(hit.evidence_ids or ()),
        evidence_tiers=tuple(hit.evidence_tiers or ()),
        required_assurance=_assurance_value(
            (hit.provenance or {}).get("required_assurance")
            or AssuranceLevel.KERNEL_VERIFIED
        ),
        derived_assurance=_assurance_value(
            (hit.provenance or {}).get("derived_assurance")
            or AssuranceLevel.UNVERIFIED
        ),
        reason_codes=tuple(hit.reason_codes or ()),
    )


def _cache_status_from_hit(
    hit: ClaimQueryHit,
    *,
    required_assurance: AssuranceLevel,
) -> CacheStatusRecord:
    disposition = CacheDisposition.NOT_QUERIED
    reasons = list(hit.reason_codes or ())
    lookup = str((hit.provenance or {}).get("cache_lookup") or "").strip().lower()
    if lookup == "hit" or hit.receipt_id:
        disposition = CacheDisposition.HIT
    elif lookup == "miss":
        disposition = CacheDisposition.MISS
    elif lookup == "stale":
        disposition = CacheDisposition.STALE
    elif lookup == "rejected":
        disposition = CacheDisposition.REJECTED
    elif lookup == "timeout":
        disposition = CacheDisposition.TIMEOUT
    # reason codes may encode reject/timeout even without cache_lookup.
    lowered = {r.lower() for r in reasons}
    if "timeout" in lowered or "single_flight_timeout" in lowered:
        disposition = CacheDisposition.TIMEOUT
    if "reject" in lowered or "rejected" in lowered or "cache_rejected" in lowered:
        disposition = CacheDisposition.REJECTED
    return CacheStatusRecord(
        disposition=disposition,
        cache_key_id=str(hit.cache_key_id or ""),
        receipt_id=str(hit.receipt_id or ""),
        reason_codes=tuple(reasons),
        required_assurance=required_assurance,
    )


def _predicted_files_from_item(item: CompiledCodeProofItem) -> tuple[str, ...]:
    files: list[str] = []
    meta = dict(item.metadata or {})
    for key in ("predicted_files", "changed_paths", "paths"):
        raw = meta.get(key)
        if isinstance(raw, (list, tuple)):
            files.extend(str(p) for p in raw)
        elif isinstance(raw, str) and raw.strip():
            files.append(raw)
    if item.obligation is not None:
        for scope in item.obligation.ast_scope_ids or ():
            # Scopes are not always paths; only keep path-like scopes.
            text = str(scope)
            if "/" in text or text.endswith(".py"):
                files.append(text)
    return _sorted_unique(files)


def packet_from_claim(
    claim: CodeClaimRecord,
    *,
    predicted_files: Sequence[str] = (),
    acceptance_ids: Sequence[str] = (),
    invalidation_reasons: Sequence[str] = (),
    prover: ProverBinding | Mapping[str, Any] | None = None,
    cache_status: CacheStatusRecord | Mapping[str, Any] | None = None,
    task_id: str = "",
    residual_ref_ids: Sequence[str] = (),
    plateau_packet_id: str = "",
    reason_codes: Sequence[str] = (),
    force_non_implementable: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CodeEditPacket:
    """Project one :class:`CodeClaimRecord` into a CodeEditPacket."""

    if not isinstance(claim, CodeClaimRecord):
        raise CodeEditMaterializeError("claim must be a CodeClaimRecord")
    claim_rec = _claim_status_from_record(claim)
    inv = list(invalidation_reasons)
    for sel in claim.invalidation_selectors or ():
        code = getattr(sel, "reason_code", "") or ""
        if code:
            inv.append(str(code))
        kind = getattr(sel, "kind", None)
        if kind is not None:
            inv.append(str(getattr(kind, "value", kind)))
    return build_code_edit_packet(
        repository_tree_id=claim.repository_tree_id,
        claim_ids=(str(claim.claim_id),) if claim.claim_id else (),
        obligation_ids=(claim.obligation_id,) if claim.obligation_id else (),
        assumption_ids=tuple(claim.assumption_ids or ()),
        invalidation_reasons=inv,
        predicted_files=predicted_files,
        acceptance_ids=acceptance_ids,
        property_ids=(claim.property_id,) if claim.property_id else (),
        required_assurance=claim.required_assurance,
        cache_status=cache_status,
        claim_status=claim_rec,
        prover=prover,
        repository_id=claim.repository_id,
        task_id=task_id,
        residual_ref_ids=residual_ref_ids,
        plateau_packet_id=plateau_packet_id,
        claim_lifecycle=claim_rec.status,
        reason_codes=reason_codes,
        force_non_implementable=force_non_implementable,
        metadata=metadata,
    )


def packet_from_query_hit(
    hit: ClaimQueryHit,
    *,
    predicted_files: Sequence[str] = (),
    acceptance_ids: Sequence[str] = (),
    invalidation_reasons: Sequence[str] = (),
    prover: ProverBinding | Mapping[str, Any] | None = None,
    repository_id: str = "",
    task_id: str = "",
    residual_ref_ids: Sequence[str] = (),
    plateau_packet_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CodeEditPacket:
    """Project one :class:`ClaimQueryHit` into a CodeEditPacket."""

    if not isinstance(hit, ClaimQueryHit):
        raise CodeEditMaterializeError("hit must be a ClaimQueryHit")
    assurance = _assurance_value(
        (hit.provenance or {}).get("required_assurance")
        or AssuranceLevel.KERNEL_VERIFIED
    )
    claim_rec = _claim_status_from_hit(hit)
    cache_rec = _cache_status_from_hit(hit, required_assurance=assurance)
    return build_code_edit_packet(
        repository_tree_id=hit.repository_tree_id,
        claim_ids=(hit.claim_id,) if hit.claim_id else (),
        obligation_ids=tuple(hit.obligation_ids or ()),
        assumption_ids=(),
        invalidation_reasons=list(invalidation_reasons) + list(hit.reason_codes or ()),
        predicted_files=predicted_files,
        acceptance_ids=acceptance_ids,
        property_ids=(hit.property_id,) if hit.property_id else (),
        required_assurance=assurance,
        cache_status=cache_rec,
        claim_status=claim_rec,
        prover=prover,
        repository_id=repository_id,
        task_id=task_id,
        residual_ref_ids=residual_ref_ids,
        plateau_packet_id=plateau_packet_id,
        claim_lifecycle=claim_rec.status,
        reason_codes=hit.reason_codes,
        metadata=metadata,
    )


def packet_from_compiled_item(
    item: CompiledCodeProofItem,
    *,
    repository_tree_id: str = "",
    repository_id: str = "",
    predicted_files: Sequence[str] = (),
    acceptance_ids: Sequence[str] = (),
    prover: ProverBinding | Mapping[str, Any] | None = None,
    task_id: str = "",
    plateau_packet_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CodeEditPacket:
    """Project one :class:`CompiledCodeProofItem` into a CodeEditPacket."""

    if not isinstance(item, CompiledCodeProofItem):
        raise CodeEditMaterializeError("item must be a CompiledCodeProofItem")
    tree = repository_tree_id
    if not tree and item.obligation is not None:
        tree = str(item.obligation.repository_tree_id or "")
    files = tuple(predicted_files) or _predicted_files_from_item(item)
    inv = [str(s.get("reason_code") or s.get("kind") or "") for s in (item.invalidation_selectors or ()) if isinstance(s, Mapping)]
    inv = [x for x in inv if x]
    inv.extend(item.reason_codes or ())

    compile_status = (
        item.status.value
        if isinstance(item.status, ObligationCompileStatus)
        else str(item.status)
    )
    claim_rec: ClaimStatusRecord
    if item.claim is not None and isinstance(item.claim, CodeClaimRecord):
        claim_rec = _claim_status_from_record(item.claim)
        if not tree:
            tree = item.claim.repository_tree_id
        if not repository_id:
            repository_id = item.claim.repository_id
    else:
        status_key = (
            item.status
            if isinstance(item.status, ObligationCompileStatus)
            else str(item.status or "")
        )
        status_map = {
            ObligationCompileStatus.OPEN: ClaimStatus.OPEN.value,
            ObligationCompileStatus.UNSUPPORTED: ClaimStatus.UNSUPPORTED.value,
            ObligationCompileStatus.NOT_MEASURED: ClaimStatus.NOT_MEASURED.value,
            "open": ClaimStatus.OPEN.value,
            "unsupported": ClaimStatus.UNSUPPORTED.value,
            "not_measured": ClaimStatus.NOT_MEASURED.value,
        }
        claim_rec = ClaimStatusRecord(
            claim_id=item.claim_id,
            property_id=item.property_id,
            status=status_map.get(status_key, ClaimStatus.UNKNOWN.value),
            obligation_id=item.obligation_id,
            required_assurance=item.required_assurance,
            reason_codes=tuple(item.reason_codes or ()),
        )

    cache_rec = CacheStatusRecord(
        disposition=CacheDisposition.NOT_QUERIED,
        cache_key_id=str(item.cache_key_id or ""),
        required_assurance=item.required_assurance,
        reason_codes=tuple(item.reason_codes or ()),
    )

    return build_code_edit_packet(
        repository_tree_id=tree,
        claim_ids=(claim_rec.claim_id,) if claim_rec.claim_id else (),
        obligation_ids=(item.obligation_id,) if item.obligation_id else (),
        assumption_ids=tuple(item.assumption_ids or ()),
        invalidation_reasons=inv,
        predicted_files=files,
        acceptance_ids=acceptance_ids,
        property_ids=(item.property_id,) if item.property_id else (),
        required_assurance=item.required_assurance,
        cache_status=cache_rec,
        claim_status=claim_rec,
        prover=prover,
        repository_id=repository_id,
        task_id=task_id,
        residual_ref_ids=tuple(item.residual_ref_ids or ()),
        plateau_packet_id=plateau_packet_id,
        claim_lifecycle=claim_rec.status,
        compile_status=compile_status,
        reason_codes=item.reason_codes,
        metadata=metadata,
    )


def bridge_plateau_codex_packet(
    plateau: Mapping[str, Any],
    *,
    repository_tree_id: str = "",
    predicted_files: Sequence[str] = (),
    acceptance_ids: Sequence[str] = (),
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    prover: ProverBinding | Mapping[str, Any] | None = None,
) -> CodeEditPacket:
    """Optional additive bridge from a PlateauCodexPacket-like mapping.

    Only residual / packet / claim handles are projected.  Gold IR bodies and
    full proof texts are rejected so they never enter cache keys or metadata.
    """

    if not isinstance(plateau, Mapping):
        raise CodeEditMaterializeError("plateau packet must be a mapping")
    _reject_gold_bodies(plateau, where="plateau_codex_packet")

    plateau_id = str(
        plateau.get("packet_id")
        or plateau.get("content_id")
        or plateau.get("plateau_packet_id")
        or ""
    )
    tree = str(
        repository_tree_id
        or plateau.get("repository_tree_id")
        or plateau.get("source_tree_id")
        or plateau.get("tree_id")
        or ""
    )
    residual_refs = _sorted_unique(
        plateau.get("residual_ref_ids")
        or plateau.get("residual_ids")
        or plateau.get("residuals")
        or ()
    )
    claim_ids = _sorted_unique(plateau.get("claim_ids") or ())
    obligation_ids = _sorted_unique(plateau.get("obligation_ids") or ())
    assumption_ids = _sorted_unique(
        plateau.get("assumption_ids") or plateau.get("assumptions") or ()
    )
    property_ids = _sorted_unique(plateau.get("property_ids") or ())
    files = _sorted_unique(
        predicted_files or plateau.get("predicted_files") or plateau.get("paths") or ()
    )
    acceptance = _sorted_unique(
        acceptance_ids or plateau.get("acceptance_ids") or ()
    )
    inv = _sorted_unique(
        plateau.get("invalidation_reasons") or plateau.get("reason_codes") or ()
    )
    status = str(plateau.get("status") or plateau.get("claim_status") or ClaimStatus.OPEN.value)

    return build_code_edit_packet(
        repository_tree_id=tree,
        claim_ids=claim_ids,
        obligation_ids=obligation_ids,
        assumption_ids=assumption_ids,
        invalidation_reasons=inv,
        predicted_files=files,
        acceptance_ids=acceptance,
        property_ids=property_ids,
        required_assurance=required_assurance,
        claim_status=ClaimStatusRecord(
            claim_id=claim_ids[0] if claim_ids else "",
            property_id=property_ids[0] if property_ids else "",
            status=status,
            obligation_id=obligation_ids[0] if obligation_ids else "",
            required_assurance=_assurance_value(required_assurance),
        ),
        prover=prover,
        residual_ref_ids=residual_refs,
        plateau_packet_id=plateau_id,
        claim_lifecycle=status,
        metadata={
            "bridge": "PlateauCodexPacket@1",
            "plateau_handles_only": True,
            "gold_ir_excluded": True,
        },
    )


def _task_id_for_packet(packet: CodeEditPacket, *, ordinal: int) -> str:
    if packet.task_id:
        return packet.task_id
    prop = packet.property_ids[0] if packet.property_ids else f"item-{ordinal}"
    return f"cbp-edit:{prop}:{packet.packet_id[:16]}"


def materialize_supervisor_task(
    packet: CodeEditPacket,
    *,
    task_id: str = "",
    title: str = "",
    test_module: str = DEFAULT_TEST_MODULE,
    domain_metrics_module: str = DEFAULT_DOMAIN_METRICS_MODULE,
    cache_aware: bool = True,
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CodeEditSupervisorTask:
    """Turn one packet into a supervisor task with validation_commands."""

    if not isinstance(packet, CodeEditPacket):
        raise CodeEditMaterializeError("packet must be a CodeEditPacket")
    tid = task_id or _task_id_for_packet(packet, ordinal=0)
    if packet.implementable:
        commands = emit_validation_commands(
            required_assurance=packet.required_assurance,
            predicted_files=packet.predicted_files,
            property_ids=packet.property_ids,
            obligation_ids=packet.obligation_ids,
            test_module=test_module,
            domain_metrics_module=domain_metrics_module,
            cache_aware=cache_aware,
        )
    else:
        # Blocked packets still record the intended command kinds for audit,
        # but mark them non-executable via empty list? Acceptance wants
        # materializer emits validation_commands for implementable work;
        # blocked packets get empty commands to avoid accidental execution.
        commands = ()

    prop = packet.property_ids[0] if packet.property_ids else "code-edit"
    default_title = (
        f"Implement open obligation for {prop}"
        if packet.implementable
        else f"Blocked code edit for {prop}"
    )
    return CodeEditSupervisorTask(
        task_id=tid,
        packet=packet,
        validation_commands=commands,
        title=title or default_title,
        implementable=packet.implementable,
        predicted_files=packet.predicted_files,
        acceptance_ids=packet.acceptance_ids,
        notes=notes,
        metadata={
            "interface": CODE_EDIT_PACKET_INTERFACE,
            "required_assurance": (
                packet.required_assurance.value
                if isinstance(packet.required_assurance, AssuranceLevel)
                else str(packet.required_assurance)
            ),
            "non_implementable_reasons": list(packet.non_implementable_reasons),
            "semantic_authority": False,
            **dict(metadata or {}),
        },
    )


def materialize_code_edit_packets(
    *,
    claims: Sequence[CodeClaimRecord] | None = None,
    compilation: CodeProofObligationCompilation | None = None,
    query: CodeProofQuery | None = None,
    query_result: CodeProofQueryResult | None = None,
    open_only: bool = True,
    predicted_files: Sequence[str] = (),
    acceptance_ids: Sequence[str] = (),
    prover: ProverBinding | Mapping[str, Any] | None = None,
    test_module: str = DEFAULT_TEST_MODULE,
    domain_metrics_module: str = DEFAULT_DOMAIN_METRICS_MODULE,
    cache_aware: bool = True,
    plateau_packet: Mapping[str, Any] | None = None,
    repository_tree_id: str = "",
    repository_id: str = "",
    include_blocked: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> CodeEditMaterializeReport:
    """Materialize packets and supervisor tasks from claims / query / compilation.

    When ``open_only`` is true (default), only open claims/items become
    implementable candidates; other statuses produce blocked packets when
    ``include_blocked`` is true.
    """

    packets: list[CodeEditPacket] = []
    notes: list[str] = []

    if plateau_packet is not None:
        packets.append(
            bridge_plateau_codex_packet(
                plateau_packet,
                repository_tree_id=repository_tree_id,
                predicted_files=predicted_files,
                acceptance_ids=acceptance_ids,
                prover=prover,
            )
        )
        notes.append("plateau_codex_bridge")

    # Prefer explicit query_result hits.
    hits: list[ClaimQueryHit] = []
    if query_result is not None:
        hits.extend(query_result.hits)
    elif query is not None:
        if open_only:
            hits.extend(query.properties_open().hits)
        else:
            hits.extend(query.hits)
    elif claims is not None:
        built = build_code_proof_query(claims=list(claims))
        if open_only:
            hits.extend(built.properties_open().hits)
        else:
            hits.extend(built.hits)

    for hit in hits:
        status = hit.status if isinstance(hit.status, ClaimStatus) else ClaimStatus(str(hit.status))
        if open_only and status is not ClaimStatus.OPEN and not include_blocked:
            continue
        if open_only and status is not ClaimStatus.OPEN and include_blocked:
            # Still materialize blocked packets for non-open when include_blocked.
            pass
        packets.append(
            packet_from_query_hit(
                hit,
                predicted_files=predicted_files,
                acceptance_ids=acceptance_ids,
                prover=prover,
                repository_id=repository_id,
            )
        )

    if compilation is not None:
        if not isinstance(compilation, CodeProofObligationCompilation):
            raise CodeEditMaterializeError(
                "compilation must be a CodeProofObligationCompilation"
            )
        tree = repository_tree_id or compilation.repository_tree_id
        repo = repository_id or compilation.repository_id
        for item in compilation.items:
            if open_only and item.status is not ObligationCompileStatus.OPEN:
                if not include_blocked:
                    continue
            packets.append(
                packet_from_compiled_item(
                    item,
                    repository_tree_id=tree,
                    repository_id=repo,
                    predicted_files=predicted_files,
                    acceptance_ids=acceptance_ids,
                    prover=prover,
                )
            )
        if not tree and compilation.repository_tree_id:
            tree = compilation.repository_tree_id
        if not repository_tree_id:
            repository_tree_id = tree

    # Deduplicate by packet content id (last write wins, stable sort after).
    by_id: dict[str, CodeEditPacket] = {}
    for packet in packets:
        by_id[packet.packet_id] = packet
    ordered = tuple(
        sorted(
            by_id.values(),
            key=lambda p: (
                p.property_ids[0] if p.property_ids else "",
                p.packet_id,
            ),
        )
    )

    if not repository_tree_id:
        for packet in ordered:
            if packet.repository_tree_id:
                repository_tree_id = packet.repository_tree_id
                break

    tasks: list[CodeEditSupervisorTask] = []
    for idx, packet in enumerate(ordered):
        if not include_blocked and not packet.implementable:
            continue
        tasks.append(
            materialize_supervisor_task(
                packet,
                task_id=_task_id_for_packet(packet, ordinal=idx),
                test_module=test_module,
                domain_metrics_module=domain_metrics_module,
                cache_aware=cache_aware,
                metadata=metadata,
            )
        )

    notes.append("cache_miss_is_not_refutation")
    notes.append("prover_semantic_authority_false")
    notes.append("no_full_proof_bodies")

    return CodeEditMaterializeReport(
        repository_tree_id=repository_tree_id,
        packets=ordered,
        tasks=tuple(tasks),
        notes=tuple(notes),
        metadata={
            "open_only": bool(open_only),
            "include_blocked": bool(include_blocked),
            "cache_aware": bool(cache_aware),
            "interface": CODE_EDIT_MATERIALIZE_INTERFACE,
            **dict(metadata or {}),
        },
    )


# Convenience aliases matching repository materialize_* naming.
materialize_code_edit_tasks = materialize_code_edit_packets


__all__ = [
    "CODE_EDIT_MATERIALIZE_INTERFACE",
    "CODE_EDIT_MATERIALIZE_REPORT_SCHEMA",
    "CODE_EDIT_MATERIALIZE_VERSION",
    "CODE_EDIT_SUPERVISOR_TASK_SCHEMA",
    "DEFAULT_DOMAIN_METRICS_MODULE",
    "DEFAULT_TEST_MODULE",
    "VALIDATION_KIND_CACHE_AWARE_REPROOF",
    "VALIDATION_KIND_DOMAIN_METRICS",
    "VALIDATION_KIND_TEST",
    "CodeEditMaterializeError",
    "CodeEditMaterializeReport",
    "CodeEditSupervisorTask",
    "bridge_plateau_codex_packet",
    "emit_validation_command_specs",
    "emit_validation_commands",
    "materialize_code_edit_packets",
    "materialize_code_edit_tasks",
    "materialize_supervisor_task",
    "packet_from_claim",
    "packet_from_compiled_item",
    "packet_from_query_hit",
    # Re-export packet surface for callers that import materialize only.
    "CacheDisposition",
    "CacheStatusRecord",
    "ClaimStatusRecord",
    "CodeEditPacket",
    "NonImplementableReason",
    "ProverBinding",
    "build_code_edit_packet",
    "compute_implementable",
]
