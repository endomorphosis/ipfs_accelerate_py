"""Focused DCR-033 local deterministic kernel reconstruction tests."""

from __future__ import annotations

import hashlib

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    McpObservationTranscript,
    ObservationStatus,
    RequiredMcpObservation,
    build_mcp_observation_epoch,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    DeterministicRepairCapabilities,
    NetworkMode,
    SolverReadiness,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.proof.kernel_reconstruction import (
    KernelAdapterOutcome,
    KernelReconstructionAdapterBinding,
    KernelReconstructionDisposition,
    KernelReconstructionOutput,
    KernelReconstructionRoots,
    reconstruct_dcr033_kernel,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationFragment,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    DeterministicProverBackend,
    DeterministicProverResources,
    route_dcr032_local_prover,
)


def _digest(label: str, value: str) -> str:
    return label + ":sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _obligation() -> McpGraphContractObligation:
    graph = _digest("graph", "dcr021")
    return McpGraphContractObligation(
        obligation_id=_digest("obligation", "dcr031"),
        family=McpObligationFamily.JSONRPC_BASELINE,
        fragment=McpObligationFragment.JSONRPC,
        backend=McpObligationBackend.LOGIC_IR_CANDIDATE,
        disposition=McpObligationDisposition.OPEN,
        graph_cid=graph,
        candidate_cid=_digest("candidate", "dcr030"),
        input_cids=tuple(sorted((graph, _digest("input", "dcr030")))),
    )


def _capabilities() -> tuple[
    DeterministicRepairCapabilities, tuple[CapabilityEvidenceReceipt, ...]
]:
    module = CapabilityReceipt(
        "fixture.kernel.adapter",
        CapabilityStatus.AVAILABLE,
        origin="/reviewed/adapter.py",
        distribution="fixture",
        expected_version="1.0",
        distribution_version="1.0",
        content_digest=_digest("module", "adapter"),
        initialized=True,
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    tool = SolverReadiness(
        "fixture-kernel",
        CapabilityStatus.AVAILABLE,
        executable="fixture-kernel",
        path="/reviewed/kernel",
        expected_version="1.0",
        version="1.0",
        executable_digest=_digest("executable", "kernel"),
        reconstruction_id="kernel-reconstruct",
        self_test_id="kernel-self-test",
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    evidence = tuple(
        CapabilityEvidenceReceipt(
            evidence_id=eid,
            evidence_kind=kind,
            subject_id=sid,
            subject_digest=digest,
            subject_version="1.0",
            transcript_digest=_digest("transcript", eid + kind),
            passed=True,
        )
        for eid, kind, sid, digest in (
            *(
                (module.capability_id, kind, module.capability_id, module.content_digest)
                for kind in ("initialization", "reconstruction", "self_test")
            ),
            (tool.reconstruction_id, "reconstruction", tool.tool_id, tool.executable_digest),
            (tool.self_test_id, "self_test", tool.tool_id, tool.executable_digest),
        )
    )
    return DeterministicRepairCapabilities((module,), (tool,), NetworkMode.OFFLINE), evidence


def _live(graph_cid: str, forest: str):
    required = RequiredMcpObservation(
        "fixture-service",
        "edge:one",
        "fixture",
        "fixture.op",
        "request",
        "fixture-schema",
        "fixture-profile",
        "mcp",
    )
    receipt = McpObservationTranscript(
        ObservationStatus.OBSERVED,
        None,
        "fixture-service",
        "mcp",
        "fixture.op",
        "http://127.0.0.1:8765",
        b"{}",
        b"{}",
        graph_cid,
        "runtime",
        "process",
        "template",
    )
    return build_mcp_observation_epoch(
        graph_cid=graph_cid,
        semantic_roots={"descriptor": "root:descriptor"},
        snapshot_roots={"forest": forest},
        required_observations=(required,),
        receipts=(receipt,),
    )


class _Adapter:
    def __init__(self, outcome: KernelAdapterOutcome, *, bad_identity: bool = False) -> None:
        self.binding = KernelReconstructionAdapterBinding(
            "fixture-adapter",
            "fixture.kernel.adapter",
            "fixture-kernel",
            _digest("module", "adapter"),
            _digest("executable", "kernel"),
        )
        self.outcome, self.bad_identity = outcome, bad_identity

    @staticmethod
    def _cid(kind: str, value: bytes, request) -> str:
        return content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/dcr-033-kernel-reconstruction@1/"
                + kind,
                "request_cid": request.request_cid,
                "sha256": "sha256:" + hashlib.sha256(value).hexdigest(),
            }
        )

    def reconstruct(self, request):
        if self.outcome is KernelAdapterOutcome.PROVED:
            proof, cert = b"proof", b"certificate"
            return KernelReconstructionOutput(
                self.outcome,
                proof,
                cert,
                proof_cid="bad" if self.bad_identity else self._cid("proof", proof, request),
                certificate_cid=self._cid("certificate", cert, request),
            )
        if self.outcome is KernelAdapterOutcome.REFUTED:
            value = b"xx!"
            return KernelReconstructionOutput(
                self.outcome,
                counterexample_bytes=value,
                counterexample_cid=self._cid("counterexample", value, request),
            )
        return KernelReconstructionOutput(self.outcome)

    def replay_counterexample(self, _request, counterexample: bytes) -> bool:
        return counterexample.endswith(b"!")


def _inputs():
    obligation = _obligation()
    capabilities, evidence = _capabilities()
    route = route_dcr032_local_prover(
        obligation,
        backend=DeterministicProverBackend(
            "fixture-route", "fixture.kernel.adapter", "fixture-kernel", ("jsonrpc",)
        ),
        capabilities=capabilities,
        capability_evidence=evidence,
        resources=DeterministicProverResources(7, 100, 4096),
    )
    roots = RepairAuthorityRoots(
        "repo", "forest:current", "tree:current", "policy:current", "plan:current", "packet:current"
    )
    live = _live(obligation.graph_cid, roots.repository_forest_cid)
    return (
        route,
        obligation,
        KernelReconstructionRoots(roots, obligation.graph_cid, live.epoch_cid),
        live,
        capabilities,
        evidence,
    )


def test_reconstruction_recomputes_proof_and_certificate_identities() -> None:
    route, obligation, roots, live, capabilities, evidence = _inputs()
    result = reconstruct_dcr033_kernel(
        route=route,
        obligation=obligation,
        roots=roots,
        live_observation=live,
        adapter=_Adapter(KernelAdapterOutcome.PROVED),
        capabilities=capabilities,
        capability_evidence=evidence,
    )
    assert result.disposition is KernelReconstructionDisposition.RECONSTRUCTED
    assert result.proof_cid and result.certificate_cid
    assert result.model_call_count == 0 and result.completion_authoritative is False
    assert result.to_dict()["authoritative"] is False


def test_refutation_is_replayed_minimized_and_keeps_roots() -> None:
    route, obligation, roots, live, capabilities, evidence = _inputs()
    result = reconstruct_dcr033_kernel(
        route=route,
        obligation=obligation,
        roots=roots,
        live_observation=live,
        adapter=_Adapter(KernelAdapterOutcome.REFUTED),
        capabilities=capabilities,
        capability_evidence=evidence,
    )
    assert result.disposition is KernelReconstructionDisposition.REFUTED
    assert result.counterexample_bytes == b"!"
    assert result.roots == roots


def test_sat_copied_bad_or_stale_data_never_becomes_reconstruction() -> None:
    route, obligation, roots, live, capabilities, evidence = _inputs()
    sat = reconstruct_dcr033_kernel(
        route=route,
        obligation=obligation,
        roots=roots,
        live_observation=live,
        adapter=_Adapter(KernelAdapterOutcome.SAT),
        capabilities=capabilities,
        capability_evidence=evidence,
    )
    bad = reconstruct_dcr033_kernel(
        route=route,
        obligation=obligation,
        roots=roots,
        live_observation=live,
        adapter=_Adapter(KernelAdapterOutcome.PROVED, bad_identity=True),
        capabilities=capabilities,
        capability_evidence=evidence,
    )
    stale_roots = KernelReconstructionRoots(
        roots.authority_roots, obligation.graph_cid, "stale-live"
    )
    stale = reconstruct_dcr033_kernel(
        route=route,
        obligation=obligation,
        roots=stale_roots,
        live_observation=live,
        adapter=_Adapter(KernelAdapterOutcome.PROVED),
        capabilities=capabilities,
        capability_evidence=evidence,
    )
    assert sat.disposition is KernelReconstructionDisposition.INVALID
    assert bad.disposition is KernelReconstructionDisposition.INVALID
    assert stale.disposition is KernelReconstructionDisposition.INTEGRATION_PENDING


def test_nonproof_outcomes_and_unqualified_adapter_receipts_fail_closed() -> None:
    route, obligation, roots, live, capabilities, evidence = _inputs()
    for outcome in (
        KernelAdapterOutcome.UNVERIFIED,
        KernelAdapterOutcome.SIMULATED,
        KernelAdapterOutcome.EXPECTED_COPIED,
    ):
        result = reconstruct_dcr033_kernel(
            route=route,
            obligation=obligation,
            roots=roots,
            live_observation=live,
            adapter=_Adapter(outcome),
            capabilities=capabilities,
            capability_evidence=evidence,
        )
        assert result.disposition is KernelReconstructionDisposition.INVALID
    unavailable = reconstruct_dcr033_kernel(
        route=route,
        obligation=obligation,
        roots=roots,
        live_observation=live,
        adapter=_Adapter(KernelAdapterOutcome.ERROR),
        capabilities=capabilities,
        capability_evidence=evidence,
    )
    missing_receipts = reconstruct_dcr033_kernel(
        route=route,
        obligation=obligation,
        roots=roots,
        live_observation=live,
        adapter=_Adapter(KernelAdapterOutcome.PROVED),
        capabilities=capabilities,
        capability_evidence=(),
    )
    assert unavailable.disposition is KernelReconstructionDisposition.UNAVAILABLE
    assert missing_receipts.disposition is KernelReconstructionDisposition.UNAVAILABLE
