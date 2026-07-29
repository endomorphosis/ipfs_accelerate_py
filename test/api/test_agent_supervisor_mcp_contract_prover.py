"""SCA-061 tests for fail-closed MCP contract proof routing."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractParityClaim,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    DEFAULT_MCP_CONTRACT_CATALOG,
    ContractSourceKind,
    McpClaimFamily,
    admit_source,
    build_contract_from_sources,
    make_source_record,
    register_contract,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    ProofVerdict,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    compile_contract_claim,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_prover import (
    ContractProofOutcome,
    ContractProofRoute,
    LocalCheckResult,
    McpContractProofResult,
    McpContractProver,
)


def _obligation(
    family: McpClaimFamily = McpClaimFamily.ARGUMENTS_PRESERVED,
    *,
    state: ParityState = ParityState.SATISFIED,
):
    source = make_source_record(
        kind=ContractSourceKind.JSON_SCHEMA,
        subject="repo.inspect",
        source_version="1.2.0",
        schema_version="2020-12",
        path="schemas/repo-inspect.json",
        payload_fingerprint="sha256:repo-inspect-v1",
    )
    catalog = admit_source(DEFAULT_MCP_CONTRACT_CATALOG, source)
    contract, contradictions = build_contract_from_sources(
        claim_family=family,
        subject="repo.inspect",
        sources=(source,),
        tool_name="repo.inspect",
    )
    catalog = register_contract(
        catalog,
        contract,
        contradictions=contradictions,
    )
    reason = (
        "schema_keyword_unsupported"
        if state is ParityState.UNSUPPORTED
        else "parity_satisfied"
    )
    # SCA-060's compiler supports graph/temporal catalog families even though
    # the SCA-051 parity-record constructor has a narrower source vocabulary.
    source_family = (
        McpClaimFamily.ARGUMENTS_PRESERVED
        if family
        in {
            McpClaimFamily.DECLARED_TOOL_EXISTS,
            McpClaimFamily.INVOCATION_REACHABLE,
            McpClaimFamily.SNAPSHOT_FRESHNESS,
            McpClaimFamily.NO_DYNAMIC_AUTHORITY,
        }
        else family
    )
    claim = ContractParityClaim(
        family=source_family,
        state=state,
        operation_id="repo.inspect",
        premise_ids=("premise:schema", "premise:route"),
        reason_codes=(reason,),
    )
    if source_family is not family:
        object.__setattr__(claim, "family", family)
    return compile_contract_claim(
        claim,
        catalog=catalog,
        contract=contract.contract_id,
        repository_id="repository:fixture",
        snapshot_id="tree:fixture",
        scope_ids=("scope:descriptor", "scope:handler"),
        assumption_ids=("assumption:closed-registry",),
        toolchain_id="toolchain:python-3.12",
        policy_id="policy:mcp-v1",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )


def _capability(*operations: ProofProviderOperation) -> ProofProviderCapability:
    return ProofProviderCapability(
        provider_id="fixture-provider",
        provider_version="1.0",
        operations=(ProofProviderOperation.CAPABILITY, *operations),
        isolation=(ProofProviderIsolation.IN_PROCESS,),
    )


class _ForgingProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def capability(self, payload):
        self.calls.append("capability")
        return _capability(ProofProviderOperation.PROVE).to_dict()

    def prove(self, payload, **kwargs):
        self.calls.append("prove")
        return {
            "outcome": "proved",
            "assurance": "kernel_verified",
            "proof_receipt": {
                "verdict": "proved",
                "authoritative_assurance": "kernel_verified",
            },
        }


def test_outcomes_are_closed_and_mutually_distinct() -> None:
    assert {item.value for item in ContractProofOutcome} == {
        "proved",
        "refuted",
        "unsupported",
        "inconclusive",
        "timed_out",
    }


def test_every_supported_fragment_selects_its_reviewed_route() -> None:
    expected = {
        McpClaimFamily.DECLARED_TOOL_EXISTS: ContractProofRoute.LOCAL_GRAPH,
        McpClaimFamily.ARGUMENTS_PRESERVED: ContractProofRoute.LOCAL_SCHEMA,
        McpClaimFamily.TRANSPORT_PARITY: ContractProofRoute.SMT,
        McpClaimFamily.POLICY_BEFORE_EFFECT: ContractProofRoute.CEC,
        McpClaimFamily.SNAPSHOT_FRESHNESS: ContractProofRoute.TDFOL,
    }
    prover = McpContractProver(provider_getter=lambda _provider_id: None)
    assert {
        family: prover.route(_obligation(family))
        for family in expected
    } == expected


def test_local_graph_checks_required_edges_without_optional_provider() -> None:
    obligation = _obligation(McpClaimFamily.DECLARED_TOOL_EXISTS)
    facts = {
        "premise_results": {
            "premise:schema": True,
            "premise:route": True,
        },
        "required_edges": (("descriptor", "handler"),),
        "observed_edges": (("descriptor", "handler"),),
    }
    result = McpContractProver(
        provider_getter=lambda _provider_id: (
            _ for _ in ()
        ).throw(AssertionError("local graph check resolved a provider"))
    ).prove(obligation, facts=facts)

    assert result.outcome is ContractProofOutcome.PROVED
    assert result.route is ContractProofRoute.LOCAL_GRAPH
    assert result.receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_local_schema_proves_with_independently_derived_receipt() -> None:
    obligation = _obligation()
    result = McpContractProver().prove(
        obligation,
        facts={
            "premise_results": {
                "premise:schema": True,
                "premise:route": True,
            },
            "schema_valid": True,
        },
    )

    assert result.outcome is ContractProofOutcome.PROVED
    assert result.route is ContractProofRoute.LOCAL_SCHEMA
    assert result.receipt.authoritative_verdict is ProofVerdict.PROVED
    assert result.receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.receipt.satisfies(obligation.required_assurance)
    assert result.fallback_used is True
    assert result.counterexample is None

    restored = McpContractProofResult.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()


def test_local_failure_has_only_compact_failed_premises_and_edges() -> None:
    obligation = _obligation()
    check = LocalCheckResult(
        ContractProofOutcome.REFUTED,
        failed_premise_ids=("premise:schema",),
        failed_edges=(("descriptor", "handler"),),
        reason_codes=("missing_contract_edge",),
    )
    result = McpContractProver(
        local_schema_checker=lambda _obligation, _facts: check
    ).prove(obligation)

    assert result.outcome is ContractProofOutcome.REFUTED
    assert result.receipt.authoritative_verdict is ProofVerdict.DISPROVED
    assert result.counterexample is not None
    assert result.counterexample.byte_size < 16 * 1024
    assert result.counterexample.assumption_ids == ("premise:schema",)
    assert result.counterexample.payload == {
        "contradiction": {
            "failed_edges": [["descriptor", "handler"]],
        },
        "premises": ["premise:schema"],
    }
    encoded = result.counterexample.to_json()
    assert "provider_output" not in encoded
    assert '"contains_source":false' in encoded


def test_unsupported_fragment_and_missing_provider_fail_closed_deterministically() -> None:
    unsupported = _obligation(state=ParityState.UNSUPPORTED)
    unsupported_result = McpContractProver().prove(unsupported)
    assert unsupported_result.outcome is ContractProofOutcome.UNSUPPORTED
    assert unsupported_result.route is ContractProofRoute.NONE
    assert unsupported_result.reason_codes == ("schema_keyword_unsupported",)

    relation = _obligation(McpClaimFamily.TRANSPORT_PARITY)
    prover = McpContractProver(provider_getter=lambda _provider_id: None)
    first = prover.prove(relation)
    second = prover.prove(relation)
    assert first.outcome is ContractProofOutcome.UNSUPPORTED
    assert first.route is ContractProofRoute.SMT
    assert first.to_dict() == second.to_dict()
    assert first.reason_codes == ("provider_unavailable",)


def test_operation_capability_is_probed_before_provider_dispatch() -> None:
    events: list[str] = []

    class UnsupportedProvider:
        def capability(self, payload):
            events.append("capability")
            return _capability().to_dict()

        def prove(self, payload, **kwargs):  # pragma: no cover - must stay closed
            events.append("prove")
            raise AssertionError("unsupported operation was dispatched")

    result = McpContractProver(
        providers={ContractProofRoute.CEC: UnsupportedProvider()}
    ).prove(_obligation(McpClaimFamily.POLICY_BEFORE_EFFECT))

    assert result.outcome is ContractProofOutcome.UNSUPPORTED
    assert result.reason_codes == ("provider_operation_unsupported",)
    assert events == ["capability"]


def test_timeout_is_not_collapsed_into_inconclusive_or_unsupported() -> None:
    class TimeoutProvider:
        def capability(self, payload):
            return _capability(ProofProviderOperation.PROVE).to_dict()

        def prove(self, payload, **kwargs):
            raise TimeoutError("bounded fixture timeout")

    result = McpContractProver(
        providers={ContractProofRoute.SMT: TimeoutProvider()}
    ).prove(_obligation(McpClaimFamily.TRANSPORT_PARITY))

    assert result.outcome is ContractProofOutcome.TIMED_OUT
    assert result.outcome is not ContractProofOutcome.INCONCLUSIVE
    assert result.outcome is not ContractProofOutcome.UNSUPPORTED
    assert result.reason_codes == ("provider_timed_out",)


def test_forged_provider_assurance_and_receipt_are_rejected() -> None:
    provider = _ForgingProvider()
    result = McpContractProver(
        providers={ContractProofRoute.SMT: provider}
    ).prove(_obligation(McpClaimFamily.TRANSPORT_PARITY))

    assert provider.calls == ["capability", "prove"]
    assert result.outcome is ContractProofOutcome.INCONCLUSIVE
    assert result.reason_codes == ("provider_assurance_rejected",)
    assert result.receipt.provider_claimed_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.receipt.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert result.receipt.authoritative_verdict is ProofVerdict.INCONCLUSIVE
    assert not result.receipt.satisfies(AssuranceLevel.SOLVER_CHECKED)
    assert result.counterexample is None


def test_kernel_candidate_is_separately_probed_and_remains_non_authoritative() -> None:
    calls: list[str] = []

    class SolverProvider:
        def capability(self, payload):
            calls.append("solver:capability")
            return _capability(ProofProviderOperation.PROVE).to_dict()

        def prove(self, payload, **kwargs):
            calls.append("solver:prove")
            return {"outcome": "candidate", "artifact_id": "candidate:fixture"}

    class KernelProvider:
        def capability(self, payload):
            calls.append("kernel:capability")
            return _capability(ProofProviderOperation.VERIFY).to_dict()

        def verify(self, payload, **kwargs):
            calls.append("kernel:verify")
            return {
                "outcome": "proved",
                "assurance": "kernel_verified",
                "artifact_id": "kernel:fixture",
            }

    result = McpContractProver(
        providers={
            ContractProofRoute.SMT: SolverProvider(),
            ContractProofRoute.KERNEL: KernelProvider(),
        }
    ).prove(_obligation(McpClaimFamily.TRANSPORT_PARITY))

    assert calls == [
        "solver:capability",
        "solver:prove",
        "kernel:capability",
        "kernel:verify",
    ]
    assert result.outcome is ContractProofOutcome.INCONCLUSIVE
    assert result.reason_codes == (
        "provider_candidate_requires_independent_validation",
    )
    assert result.receipt.authoritative_assurance is AssuranceLevel.UNVERIFIED


def test_provider_loader_is_lazy_and_no_provider_is_used_for_local_checks() -> None:
    calls: list[str] = []

    def forbidden_loader():
        calls.append("loaded")
        raise AssertionError("provider loader was invoked on a local route")

    result = McpContractProver(
        providers={ContractProofRoute.SMT: forbidden_loader}
    ).prove(
        _obligation(),
        facts={
            "premise_results": {
                "premise:schema": True,
                "premise:route": True,
            },
            "schema_valid": True,
        },
    )

    assert result.outcome is ContractProofOutcome.PROVED
    assert calls == []
