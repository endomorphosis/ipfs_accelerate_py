"""DCR-033 local, deterministic proof/counterexample reconstruction boundary.

This module is deliberately not a prover.  A reviewed, injected local adapter
performs reconstruction; this boundary rechecks every binding and every
content identity before emitting a non-authoritative result.  Solver claims,
simulations, copied expectations, and missing observations never become proof
or completion authority here.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..analysis.mcp_live_observer import McpObservationEpoch, is_current_mcp_observation_epoch
from ..autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    DeterministicRepairCapabilities,
    NetworkMode,
    SolverReadiness,
)
from ..autonomous_repair.contracts import RepairAuthorityRoots
from .formal_verification_contracts import content_identity
from .mcp_contract_obligations import McpGraphContractObligation, McpObligationDisposition
from .multi_prover_router import (
    DCR032_PROVER_ROUTE_SCHEMA,
    DeterministicProverDisposition,
    DeterministicProverRoute,
)


DCR033_KERNEL_RECONSTRUCTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/dcr-033-kernel-reconstruction@1"
)
DCR033_KERNEL_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/dcr-033-kernel-reconstruction-request@1"
)


class KernelReconstructionDisposition(str, Enum):  # noqa: UP042 - Python 3.8
    RECONSTRUCTED = "reconstructed_candidate"
    REFUTED = "refuted_candidate"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    INTEGRATION_PENDING = "integration_pending"


class KernelAdapterOutcome(str, Enum):  # noqa: UP042 - Python 3.8
    PROVED = "proved"
    REFUTED = "refuted"
    UNVERIFIED = "unverified"
    SAT = "sat"
    SIMULATED = "simulated"
    EXPECTED_COPIED = "expected_copied"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class KernelReconstructionAdapterBinding:
    """Exact DCR-004 module/toolchain receipts required by an injected adapter."""

    adapter_id: str
    module_capability_id: str
    toolchain_id: str
    module_digest: str
    toolchain_digest: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.adapter_id,
                self.module_capability_id,
                self.toolchain_id,
                self.module_digest,
                self.toolchain_digest,
            )
        ):
            raise ValueError("kernel adapter binding fields must be non-empty text")
        if not self.module_digest.startswith(
            "module:sha256:"
        ) or not self.toolchain_digest.startswith("executable:sha256:"):
            raise ValueError("kernel adapter binding digests must be canonical")

    def to_dict(self) -> dict[str, str]:
        return {
            "adapter_id": self.adapter_id,
            "module_capability_id": self.module_capability_id,
            "toolchain_id": self.toolchain_id,
            "module_digest": self.module_digest,
            "toolchain_digest": self.toolchain_digest,
        }


@dataclass(frozen=True)
class KernelReconstructionRoots:
    """Exact graph, current live observation, and repair-authority roots."""

    authority_roots: RepairAuthorityRoots
    graph_cid: str
    live_transcript_cid: str

    def __post_init__(self) -> None:
        if not isinstance(self.authority_roots, RepairAuthorityRoots):
            raise ValueError("authority_roots must be RepairAuthorityRoots")
        if not all(
            isinstance(value, str) and value for value in (self.graph_cid, self.live_transcript_cid)
        ):
            raise ValueError("kernel reconstruction roots must be non-empty exact identifiers")

    @property
    def roots_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_roots": self.authority_roots.to_dict(),
            "graph_cid": self.graph_cid,
            "live_transcript_cid": self.live_transcript_cid,
        }


@dataclass(frozen=True)
class KernelReconstructionRequest:
    obligation: McpGraphContractObligation
    route: DeterministicProverRoute
    roots: KernelReconstructionRoots
    adapter_binding: KernelReconstructionAdapterBinding

    @property
    def request_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR033_KERNEL_REQUEST_SCHEMA,
            "obligation_cid": content_identity(self.obligation.to_dict()),
            "route_cid": self.route.content_id,
            "roots": self.roots.to_dict(),
            "adapter_binding": self.adapter_binding.to_dict(),
        }


@dataclass(frozen=True)
class KernelReconstructionOutput:
    """Typed raw adapter output; claims are independently recomputed below."""

    outcome: KernelAdapterOutcome
    proof_bytes: bytes = b""
    certificate_bytes: bytes = b""
    counterexample_bytes: bytes = b""
    proof_cid: str = ""
    certificate_cid: str = ""
    counterexample_cid: str = ""


class LocalDeterministicReconstructionAdapter(Protocol):
    """Reviewed injected adapter.  It has no network or model authority."""

    binding: KernelReconstructionAdapterBinding

    def reconstruct(self, request: KernelReconstructionRequest) -> KernelReconstructionOutput: ...

    def replay_counterexample(
        self, request: KernelReconstructionRequest, counterexample: bytes
    ) -> bool: ...


@dataclass(frozen=True)
class KernelReconstructionResult:
    disposition: KernelReconstructionDisposition
    reason_codes: tuple[str, ...]
    request_cid: str = ""
    proof_cid: str = ""
    certificate_cid: str = ""
    counterexample_cid: str = ""
    counterexample_bytes: bytes = b""
    roots: KernelReconstructionRoots | None = None

    @property
    def mutation_authorized(self) -> bool:
        return False

    @property
    def model_call_count(self) -> int:
        return 0

    @property
    def completion_authoritative(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": DCR033_KERNEL_RECONSTRUCTION_SCHEMA,
            "authoritative": False,
            "completion_authoritative": False,
            "mutation_authorized": False,
            "model_call_count": 0,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "request_cid": self.request_cid,
            "proof_cid": self.proof_cid,
            "certificate_cid": self.certificate_cid,
            "counterexample_cid": self.counterexample_cid,
            "counterexample_base64": base64.b64encode(self.counterexample_bytes).decode("ascii"),
            "roots": self.roots.to_dict() if self.roots else {},
        }
        return {**body, "result_cid": content_identity(body)}


def _cid(kind: str, value: bytes, request: KernelReconstructionRequest) -> str:
    return content_identity(
        {
            "schema": DCR033_KERNEL_RECONSTRUCTION_SCHEMA + "/" + kind,
            "request_cid": request.request_cid,
            "sha256": "sha256:" + hashlib.sha256(value).hexdigest(),
        }
    )


def _evidence_ok(
    evidence: Sequence[CapabilityEvidenceReceipt],
    *,
    evidence_id: str,
    kind: str,
    subject_id: str,
    digest: str,
    version: str,
) -> bool:
    return any(
        item.verifies(
            evidence_id=evidence_id,
            evidence_kind=kind,
            subject_id=subject_id,
            subject_digest=digest,
            subject_version=version,
        )
        for item in evidence
    )


def _qualified_adapter(
    adapter: Any,
    route: DeterministicProverRoute,
    capabilities: Any,
    evidence: Sequence[CapabilityEvidenceReceipt],
) -> tuple[KernelReconstructionAdapterBinding | None, tuple[str, ...]]:
    binding = getattr(adapter, "binding", None)
    if (
        not isinstance(binding, KernelReconstructionAdapterBinding)
        or not callable(getattr(adapter, "reconstruct", None))
        or not callable(getattr(adapter, "replay_counterexample", None))
    ):
        return None, ("reviewed_local_adapter_required",)
    if (
        not isinstance(capabilities, DeterministicRepairCapabilities)
        or capabilities.network_mode is not NetworkMode.OFFLINE
    ):
        return None, ("offline_capability_inventory_required",)
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or not all(isinstance(item, CapabilityEvidenceReceipt) for item in evidence)
    ):
        return None, ("typed_capability_evidence_required",)
    module = next(
        (
            item
            for item in capabilities.modules
            if item.capability_id == binding.module_capability_id
        ),
        None,
    )
    toolchain = next(
        (item for item in capabilities.toolchains if item.tool_id == binding.toolchain_id), None
    )
    if not isinstance(module, CapabilityReceipt) or not isinstance(toolchain, SolverReadiness):
        return None, ("adapter_capability_receipt_missing",)
    if (
        not module.available
        or not module.initialized
        or not module.reconstructed
        or not module.self_test_passed
        or module.network_mode is not NetworkMode.OFFLINE
        or module.content_digest != binding.module_digest
        or not toolchain.available
        or not toolchain.reconstructed
        or not toolchain.self_test_passed
        or toolchain.network_mode is not NetworkMode.OFFLINE
        or toolchain.executable_digest != binding.toolchain_digest
        or module.receipt_id not in route.capability_receipt_ids
        or toolchain.receipt_id not in route.capability_receipt_ids
    ):
        return None, ("adapter_capability_binding_invalid",)
    checks = (
        _evidence_ok(
            evidence,
            evidence_id=module.capability_id,
            kind=kind,
            subject_id=module.capability_id,
            digest=module.content_digest,
            version=module.distribution_version,
        )
        for kind in ("initialization", "reconstruction", "self_test")
    )
    if (
        not all(checks)
        or not _evidence_ok(
            evidence,
            evidence_id=toolchain.reconstruction_id,
            kind="reconstruction",
            subject_id=toolchain.tool_id,
            digest=toolchain.executable_digest,
            version=toolchain.version,
        )
        or not _evidence_ok(
            evidence,
            evidence_id=toolchain.self_test_id,
            kind="self_test",
            subject_id=toolchain.tool_id,
            digest=toolchain.executable_digest,
            version=toolchain.version,
        )
    ):
        return None, ("adapter_capability_evidence_missing_or_stale",)
    return binding, ()


def _current_inputs(
    route: Any,
    obligation: Any,
    roots: Any,
    observation: Any,
) -> tuple[KernelReconstructionRoots | None, tuple[str, ...]]:
    if not isinstance(route, DeterministicProverRoute) or not isinstance(
        obligation, McpGraphContractObligation
    ):
        return None, ("typed_dcr032_route_and_dcr031_obligation_required",)
    if not isinstance(roots, KernelReconstructionRoots) or not isinstance(
        observation, McpObservationEpoch
    ):
        return None, ("typed_roots_and_live_observation_required",)
    if (
        route.SCHEMA != DCR032_PROVER_ROUTE_SCHEMA
        or route.disposition is not DeterministicProverDisposition.ROUTED
        or route.integration_pending
        or route.model_call_count != 0
        or route.external_execution_count != 0
        or obligation.disposition is not McpObligationDisposition.OPEN
        or route.obligation_id != obligation.obligation_id
        or route.obligation_cid != content_identity(obligation.to_dict())
        or roots.graph_cid != obligation.graph_cid
        or observation.graph_cid != roots.graph_cid
        or observation.epoch_cid != roots.live_transcript_cid
        or observation.snapshot_roots.get("forest") != roots.authority_roots.repository_forest_cid
        or not is_current_mcp_observation_epoch(
            observation,
            graph_cid=observation.graph_cid,
            semantic_roots=observation.semantic_roots,
            snapshot_roots=observation.snapshot_roots,
        )
    ):
        return None, ("route_obligation_or_roots_not_current",)
    return roots, ()


def _minimize_counterexample(
    adapter: LocalDeterministicReconstructionAdapter,
    request: KernelReconstructionRequest,
    value: bytes,
) -> bytes:
    """Deterministic deletion minimization; candidates survive only by replay."""

    current = value
    width = max(1, len(current) // 2)
    while current and width:
        changed = False
        for start in range(0, len(current), width):
            candidate = current[:start] + current[start + width :]
            if candidate and adapter.replay_counterexample(request, candidate):
                current, changed = candidate, True
                break
        if not changed:
            width //= 2
    return current


def reconstruct_dcr033_kernel(
    *,
    route: DeterministicProverRoute,
    obligation: McpGraphContractObligation,
    roots: KernelReconstructionRoots,
    live_observation: McpObservationEpoch,
    adapter: LocalDeterministicReconstructionAdapter,
    capabilities: DeterministicRepairCapabilities,
    capability_evidence: Sequence[CapabilityEvidenceReceipt] = (),
) -> KernelReconstructionResult:
    """Reconstruct only through a current DCR-032 route and typed local adapter."""

    current_roots, reasons = _current_inputs(route, obligation, roots, live_observation)
    if current_roots is None:
        return KernelReconstructionResult(
            KernelReconstructionDisposition.INTEGRATION_PENDING, reasons
        )
    binding, reasons = _qualified_adapter(adapter, route, capabilities, capability_evidence)
    if binding is None:
        return KernelReconstructionResult(
            KernelReconstructionDisposition.UNAVAILABLE, reasons, roots=current_roots
        )
    request = KernelReconstructionRequest(obligation, route, current_roots, binding)
    try:
        output = adapter.reconstruct(request)
    except Exception:  # noqa: BLE001 - adapter diagnostics are not public evidence
        return KernelReconstructionResult(
            KernelReconstructionDisposition.UNAVAILABLE,
            ("local_adapter_error",),
            request.request_cid,
            roots=current_roots,
        )
    if not isinstance(output, KernelReconstructionOutput):
        return KernelReconstructionResult(
            KernelReconstructionDisposition.INVALID,
            ("typed_adapter_output_required",),
            request.request_cid,
            roots=current_roots,
        )
    if output.outcome in (KernelAdapterOutcome.ERROR, KernelAdapterOutcome.UNAVAILABLE):
        return KernelReconstructionResult(
            KernelReconstructionDisposition.UNAVAILABLE,
            ("adapter_output_unavailable",),
            request.request_cid,
            roots=current_roots,
        )
    if output.outcome not in (KernelAdapterOutcome.PROVED, KernelAdapterOutcome.REFUTED):
        return KernelReconstructionResult(
            KernelReconstructionDisposition.INVALID,
            ("unverified_sat_simulated_or_copied_output",),
            request.request_cid,
            roots=current_roots,
        )
    if output.outcome is KernelAdapterOutcome.PROVED:
        proof_cid, certificate_cid = (
            _cid("proof", output.proof_bytes, request),
            _cid("certificate", output.certificate_bytes, request),
        )
        if (
            not output.proof_bytes
            or not output.certificate_bytes
            or output.proof_cid != proof_cid
            or output.certificate_cid != certificate_cid
        ):
            return KernelReconstructionResult(
                KernelReconstructionDisposition.INVALID,
                ("proof_or_certificate_identity_invalid",),
                request.request_cid,
                roots=current_roots,
            )
        return KernelReconstructionResult(
            KernelReconstructionDisposition.RECONSTRUCTED,
            (),
            request.request_cid,
            proof_cid,
            certificate_cid,
            roots=current_roots,
        )
    counterexample_cid = _cid("counterexample", output.counterexample_bytes, request)
    if not output.counterexample_bytes or output.counterexample_cid != counterexample_cid:
        return KernelReconstructionResult(
            KernelReconstructionDisposition.INVALID,
            ("counterexample_identity_invalid",),
            request.request_cid,
            roots=current_roots,
        )
    try:
        if not adapter.replay_counterexample(request, output.counterexample_bytes):
            return KernelReconstructionResult(
                KernelReconstructionDisposition.INVALID,
                ("counterexample_not_replayable",),
                request.request_cid,
                roots=current_roots,
            )
        minimized = _minimize_counterexample(adapter, request, output.counterexample_bytes)
    except Exception:  # noqa: BLE001
        return KernelReconstructionResult(
            KernelReconstructionDisposition.UNAVAILABLE,
            ("counterexample_replay_unavailable",),
            request.request_cid,
            roots=current_roots,
        )
    return KernelReconstructionResult(
        KernelReconstructionDisposition.REFUTED,
        (),
        request.request_cid,
        counterexample_cid=_cid("counterexample", minimized, request),
        counterexample_bytes=minimized,
        roots=current_roots,
    )


__all__ = [
    "DCR033_KERNEL_RECONSTRUCTION_SCHEMA",
    "DCR033_KERNEL_REQUEST_SCHEMA",
    "KernelAdapterOutcome",
    "KernelReconstructionAdapterBinding",
    "KernelReconstructionDisposition",
    "KernelReconstructionOutput",
    "KernelReconstructionRequest",
    "KernelReconstructionResult",
    "KernelReconstructionRoots",
    "LocalDeterministicReconstructionAdapter",
    "reconstruct_dcr033_kernel",
]
