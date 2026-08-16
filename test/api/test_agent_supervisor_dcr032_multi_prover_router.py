"""DCR-032 deterministic local-prover router contract tests.

These tests deliberately supply receipts only.  They never import, spawn, or
execute a prover.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    DeterministicRepairCapabilities,
    NetworkMode,
    SolverReadiness,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationFragment,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    DCR032_INTEGRATION_PENDING_REASON,
    DeterministicProverBackend,
    DeterministicProverDisposition,
    DeterministicProverResources,
    route_dcr032_local_prover,
)


def _digest(label: str, value: str) -> str:
    return f"{label}:sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _obligation(
    *, fragment: McpObligationFragment = McpObligationFragment.JSONRPC
) -> McpGraphContractObligation:
    return McpGraphContractObligation(
        obligation_id=_digest("obligation", "dcr031"),
        family=McpObligationFamily.JSONRPC_BASELINE,
        fragment=fragment,
        backend=McpObligationBackend.LOGIC_IR_CANDIDATE,
        disposition=McpObligationDisposition.OPEN,
        graph_cid=_digest("graph", "dcr021"),
        candidate_cid=_digest("source", "dcr030"),
        input_cids=tuple(sorted((_digest("graph", "dcr021"), _digest("input", "dcr030")))),
    )


def _backend(*, provider_kind: str = "local_offline") -> DeterministicProverBackend:
    return DeterministicProverBackend(
        backend_id="local-qfuf",
        module_capability_id="fixture.logic.qfuf",
        toolchain_id="fixture-solver",
        supported_fragments=("jsonrpc",),
        provider_kind=provider_kind,
    )


def _capabilities() -> tuple[
    DeterministicRepairCapabilities, tuple[CapabilityEvidenceReceipt, ...]
]:
    module_digest = _digest("module", "fixture-logic")
    tool_digest = _digest("executable", "fixture-solver")
    module = CapabilityReceipt(
        capability_id="fixture.logic.qfuf",
        status=CapabilityStatus.AVAILABLE,
        origin="/reviewed/qfuf.py",
        distribution="fixture-logic",
        expected_version="1.0",
        distribution_version="1.0",
        content_digest=module_digest,
        symbols=("QfUf",),
        initialized=True,
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    toolchain = SolverReadiness(
        tool_id="fixture-solver",
        status=CapabilityStatus.AVAILABLE,
        executable="fixture-solver",
        path="/reviewed/fixture-solver",
        expected_version="1.0",
        version="1.0",
        executable_digest=tool_digest,
        self_test_id="fixture-solver-self-test",
        self_test_passed=True,
        reconstruction_id="fixture-solver-reconstruction",
        reconstructed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    inventory = DeterministicRepairCapabilities(
        modules=(module,), toolchains=(toolchain,), network_mode=NetworkMode.OFFLINE
    )
    evidence = tuple(
        CapabilityEvidenceReceipt(
            evidence_id=evidence_id,
            evidence_kind=kind,
            subject_id=subject_id,
            subject_digest=digest,
            subject_version="1.0",
            transcript_digest=_digest("transcript", f"{evidence_id}:{kind}"),
            passed=True,
        )
        for evidence_id, kind, subject_id, digest in (
            *((module.capability_id, kind, module.capability_id, module.content_digest)
              for kind in ("initialization", "reconstruction", "self_test")),
            (
                toolchain.reconstruction_id,
                "reconstruction",
                toolchain.tool_id,
                toolchain.executable_digest,
            ),
            (toolchain.self_test_id, "self_test", toolchain.tool_id, toolchain.executable_digest),
        )
    )
    return inventory, evidence


def _route(**changes: object):
    inventory, evidence = _capabilities()
    values: dict[str, object] = {
        "obligation": _obligation(),
        "backend": _backend(),
        "capabilities": inventory,
        "capability_evidence": evidence,
        "resources": DeterministicProverResources(
            seed=7, max_steps=100, max_memory_bytes=4096
        ),
    }
    values.update(changes)
    return route_dcr032_local_prover(**values)  # type: ignore[arg-type]


def test_current_dcr031_route_is_canonical_but_never_proof_authority() -> None:
    route = _route()

    assert route.disposition is DeterministicProverDisposition.ROUTED
    assert route.integration_pending is False
    assert route.execution_permitted is True
    assert route.proof_authorized is False
    assert route.proof_authority_call_count == 0
    assert route.model_call_count == 0
    assert route.external_execution_count == 0
    assert route.content_id == _route().content_id


def test_noncurrent_or_unsupported_logic_and_remote_model_do_not_authorize() -> None:
    unsupported_fragment = _route(
        obligation=_obligation(fragment=McpObligationFragment.UNSUPPORTED)
    )
    synthetic_mapping = _route(obligation={"logic_fragment": "jsonrpc"})
    remote_route = _route(backend=_backend(provider_kind="remote_model"))

    assert unsupported_fragment.disposition is DeterministicProverDisposition.DEFER_CAPABILITY
    assert synthetic_mapping.disposition is DeterministicProverDisposition.DEFER_CAPABILITY
    assert DCR032_INTEGRATION_PENDING_REASON in synthetic_mapping.reason_codes
    assert remote_route.disposition is DeterministicProverDisposition.UNSUPPORTED
    for route in (unsupported_fragment, synthetic_mapping, remote_route):
        assert route.proof_authorized is False
        assert route.model_call_count == 0
        assert route.external_execution_count == 0


def test_importability_flags_and_missing_exact_receipts_are_unavailable() -> None:
    inventory, evidence = _capabilities()
    importable_only = replace(
        inventory.modules[0],
        initialized=False,
        reconstructed=False,
        self_test_passed=False,
    )
    unavailable_inventory = replace(inventory, modules=(importable_only,))
    missing_receipt_route = _route(
        capabilities=unavailable_inventory, capability_evidence=()
    )
    stub_inventory = replace(
        inventory,
        modules=(replace(inventory.modules[0], reason_codes=("stub_todo_or_simulated_source",)),),
    )
    stub_route = _route(capabilities=stub_inventory, capability_evidence=evidence)

    for route in (missing_receipt_route, stub_route):
        assert route.disposition is DeterministicProverDisposition.UNAVAILABLE
        assert route.proof_authorized is False
        assert route.proof_authority_call_count == 0
        assert route.external_execution_count == 0


def test_sat_without_reconstruction_is_deferred_and_never_proof_authority() -> None:
    route = _route(reported_outcome="sat", proof_reconstruction_receipt_id="")

    assert route.disposition is DeterministicProverDisposition.DEFER_CAPABILITY
    assert "sat_without_required_reconstruction" in route.reason_codes
    assert route.proof_authorized is False
    assert route.proof_authority_call_count == 0
    assert route.model_call_count == 0
    assert route.external_execution_count == 0
