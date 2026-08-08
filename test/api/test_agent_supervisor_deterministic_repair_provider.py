"""DCR-000 tests for the non-prompt deterministic runtime adapter."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityReceipt,
    CapabilityStatus,
    NetworkMode,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AuthorityStage,
    DeterministicRepairDisposition,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.no_llm_policy import (
    DeterministicRepairAuthorityPolicy,
)
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_repair_provider import (
    DETERMINISTIC_REPAIR_PROVIDER_INTERFACE,
    DeterministicRepairProvider,
    DeterministicRepairRequest,
    DeterministicRepairStateMachineBinding,
    DeterministicRepairTransition,
)


def roots() -> RepairAuthorityRoots:
    return RepairAuthorityRoots(
        repository_id="repository:fixture",
        repository_forest_cid="forest:fixture",
        git_tree_id="tree:fixture",
        policy_root="policy:fixture",
        rpr_plan_cid="plan:fixture",
        rpr_packet_cid="packet:fixture",
    )


def observed() -> RepairEvidenceEnvelope:
    return RepairEvidenceEnvelope(
        repair_id="repair:fixture",
        disposition=DeterministicRepairDisposition.REFUTED_REPAIRABLE,
        authority_stage=AuthorityStage.OBSERVED,
        authority_roots=roots(),
        observation_cid="observation:fixture",
    )


def capability() -> CapabilityReceipt:
    return CapabilityReceipt(
        capability_id="ipfs_datasets_py.logic.fixture_machine",
        status=CapabilityStatus.AVAILABLE,
        origin="/fixture/ipfs_datasets_py/logic/fixture_machine.py",
        distribution="ipfs-datasets-py",
        expected_version="1",
        distribution_version="1",
        content_digest="module:sha256:fixture",
        symbols=("FixtureMachine",),
        initialized=True,
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )


class DerivedMachine:
    def __init__(self) -> None:
        self.calls = 0

    def advance(self, request: DeterministicRepairRequest) -> DeterministicRepairTransition:
        self.calls += 1
        state = RepairEvidenceEnvelope(
            repair_id=request.state.repair_id,
            disposition=DeterministicRepairDisposition.REFUTED_REPAIRABLE,
            authority_stage=AuthorityStage.DERIVED,
            authority_roots=request.state.authority_roots,
            observation_cid=request.state.observation_cid,
            previous_authority_stage=request.state.authority_stage,
            previous_envelope_cid=request.state.content_id,
            derivation_cid="derivation:fixture",
        )
        return DeterministicRepairTransition(
            disposition=state.disposition,
            state=state,
            reason_code="derived",
        )


def provider(machine: object | None = None, **kwargs: object) -> DeterministicRepairProvider:
    machine = machine or DerivedMachine()
    receipt = capability()
    binding = DeterministicRepairStateMachineBinding(
        machine_id="local-repair-machine",
        pin=receipt.receipt_id,
        machine=machine,  # type: ignore[arg-type]
        capability=receipt,
        **kwargs,
    )
    return DeterministicRepairProvider(
        binding,
        authority_policy=DeterministicRepairAuthorityPolicy(
            local_logic_pins=frozenset({receipt.receipt_id})
        ),
    )


def test_admitted_typed_machine_binds_input_output_cids_without_completion_authority() -> None:
    machine = DerivedMachine()
    result = provider(machine).execute(DeterministicRepairRequest("request:1", observed()))
    assert DETERMINISTIC_REPAIR_PROVIDER_INTERFACE == "DeterministicRepairProvider@1"
    assert result.disposition is DeterministicRepairDisposition.REFUTED_REPAIRABLE
    assert result.invoked is True and machine.calls == 1
    assert result.input_evidence_cid == observed().content_id
    assert result.output_evidence_cid
    assert result.to_dict()["completion_authoritative"] is False
    assert result.to_dict()["fallback_authorized"] is False


@pytest.mark.parametrize("raw_input", ("write this patch", {"prompt": "fix it"}, lambda: None))
def test_prompts_strings_and_raw_inputs_reject_before_machine_invocation(raw_input: object) -> None:
    machine = DerivedMachine()
    result = provider(machine).execute(raw_input)
    assert result.disposition is DeterministicRepairDisposition.REJECTED
    assert result.reason_code == "invalid-request"
    assert machine.calls == 0


def test_callable_and_forbidden_machine_routes_are_rejected_at_binding_boundary() -> None:
    with pytest.raises(ValueError, match="non-callable object"):
        DeterministicRepairStateMachineBinding("local", "logic:1", lambda request: None, capability())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forbidden route"):
        DeterministicRepairStateMachineBinding("model-fallback", "logic:1", DerivedMachine(), capability())
    with pytest.raises(ValueError, match="deterministic local logic only"):
        DeterministicRepairStateMachineBinding(
            "local", capability().receipt_id, DerivedMachine(), capability(), route="prover_subprocess"  # type: ignore[arg-type]
        )


def test_policy_denial_and_nonzero_counters_reject_before_invocation() -> None:
    machine = DerivedMachine()
    receipt = capability()
    binding = DeterministicRepairStateMachineBinding("local", receipt.receipt_id, machine, receipt)
    denied = DeterministicRepairProvider(
        binding,
        authority_policy=DeterministicRepairAuthorityPolicy(),
    ).execute(DeterministicRepairRequest("request:denied", observed()))
    assert denied.reason_code == "execution-route-denied"
    assert machine.calls == 0

    with pytest.raises(ValueError, match="exactly zero"):
        DeterministicRepairRequest("request:counter", observed(), model_call_count=1)
    assert machine.calls == 0


def test_missing_or_forged_capability_attestation_denies_malicious_machine_before_invocation() -> None:
    class MaliciousMachine:
        def __init__(self) -> None:
            self.calls = 0

        def advance(self, request: DeterministicRepairRequest) -> DeterministicRepairTransition:
            self.calls += 1
            raise AssertionError("must not be invoked")

    machine = MaliciousMachine()
    receipt = capability()
    with pytest.raises(ValueError, match="exact capability receipt_id"):
        DeterministicRepairStateMachineBinding("local", "forged:pin", machine, receipt)
    with pytest.raises(ValueError, match="CapabilityReceipt"):
        DeterministicRepairStateMachineBinding("local", receipt.receipt_id, machine, None)  # type: ignore[arg-type]
    assert machine.calls == 0


def test_invalid_output_and_typed_terminal_results_never_fallback() -> None:
    class BadOutput:
        def advance(self, request: DeterministicRepairRequest) -> object:
            return {"provider": "fallback"}

    rejected = provider(BadOutput()).execute(DeterministicRepairRequest("request:bad", observed()))
    assert rejected.reason_code == "unknown-state-machine-output"

    class Deferred:
        def advance(self, request: DeterministicRepairRequest) -> DeterministicRepairTransition:
            return DeterministicRepairTransition(
                disposition=DeterministicRepairDisposition.DEFER_CAPABILITY,
                reason_code="capability-unavailable",
            )

    deferred = provider(Deferred()).execute(DeterministicRepairRequest("request:defer", observed()))
    assert deferred.disposition is DeterministicRepairDisposition.DEFER_CAPABILITY
    assert deferred.invoked is True
    assert not deferred.to_dict()["fallback_authorized"]
