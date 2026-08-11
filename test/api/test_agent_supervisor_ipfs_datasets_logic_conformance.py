"""SCA-214 real-module conformance for exact datasets logic/prover backends.

These tests exercise actual ``ipfs_datasets_py.logic`` IR/TDFOL/CEC/SMT/Hammer
signatures through the supervisor facades.  Fixture-only success is rejected:
capability labels alone cannot register a backend, solver output remains a
candidate until trusted reconstruction, and unavailable providers are
unsupported rather than silent local success.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

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
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    DATASETS_LOGIC_BACKEND_SPECS,
    DATASETS_LOGIC_CANDIDATE_ASSURANCE,
    LOGIC_IR_INTERFACE,
    DatasetsLogicBackendError,
    DatasetsLogicBackendKind,
    DatasetsLogicBackendProbe,
    DatasetsLogicBackendProvider,
    DatasetsLogicBackendRegistry,
    DatasetsLogicSymbolReceipt,
    build_datasets_logic_backend_registry,
    call_logic_ir_identity,
    create_datasets_logic_backend_provider,
    probe_all_datasets_logic_backends,
    probe_datasets_logic_backend,
    select_premises_retaining_identities,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderCapability,
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
    McpContractProver,
    create_mcp_contract_prover_with_datasets_logic_backends,
    datasets_logic_backends_are_registered,
)


pytest.importorskip("ipfs_datasets_py")


def _obligation(family: McpClaimFamily):
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
    catalog = register_contract(catalog, contract, contradictions=contradictions)
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
        state=ParityState.SATISFIED,
        operation_id="repo.inspect",
        premise_ids=("premise:schema", "premise:route"),
        reason_codes=("parity_satisfied",),
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


def _real_corpus_manifest():
    from ipfs_datasets_py.logic.hammers.corpus import CorpusManifest, CorpusSource
    from ipfs_datasets_py.logic.hammers.models import ITPKind

    manifest = CorpusManifest(
        manifest_id="conformance-corpus",
        created_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
    )
    manifest.register_source(
        CorpusSource(
            corpus_id="reviewed-corpus",
            name="Reviewed Corpus",
            source_itp=ITPKind.LEAN,
            version_ref="git:deadbeef",
            license_id="Apache-2.0",
        )
    )
    for theorem_id, statement in (
        ("premise:relation", "Ready may transition only to running."),
        ("premise:state", "The current state is ready."),
        ("premise:extra", "Unrelated algebraic identity."),
    ):
        manifest.add_theorem(
            theorem_id=theorem_id,
            corpus_id="reviewed-corpus",
            statement=statement,
        )
    return manifest


def test_exact_backend_specs_cover_ir_tdfol_cec_smt_and_hammer() -> None:
    kinds = {item for item in DatasetsLogicBackendKind}
    assert kinds == {
        DatasetsLogicBackendKind.IR,
        DatasetsLogicBackendKind.TDFOL,
        DatasetsLogicBackendKind.CEC,
        DatasetsLogicBackendKind.SMT,
        DatasetsLogicBackendKind.HAMMER,
    }
    assert set(DATASETS_LOGIC_BACKEND_SPECS) == kinds
    assert LOGIC_IR_INTERFACE == "LogicIR@1"
    cec_symbols = DATASETS_LOGIC_BACKEND_SPECS[
        DatasetsLogicBackendKind.CEC
    ].symbols
    assert {(item.module, item.name) for item in cec_symbols} == {
        ("ipfs_datasets_py.logic.CEC.native", "parse_dcec_string"),
        ("ipfs_datasets_py.logic.CEC.native", "Formula"),
        ("ipfs_datasets_py.logic.CEC.native", "DCECContainer"),
        ("ipfs_datasets_py.logic.CEC.native", "DCECStatement"),
    }


def test_real_module_probes_call_actual_ir_tdfol_cec_smt_hammer_signatures() -> None:
    probes = probe_all_datasets_logic_backends()
    by_kind = {probe.kind: probe for probe in probes}
    assert set(by_kind) == set(DatasetsLogicBackendKind)

    for kind, probe in by_kind.items():
        assert probe.available, (
            f"{kind.value} backend must be available for real-module conformance: "
            f"{probe.unavailable_reason} ({probe.reason_code})"
        )
        assert probe.symbol_receipts
        assert all(item.available for item in probe.symbol_receipts)
        assert all(item.signature for item in probe.symbol_receipts)
        assert probe.capability_revision.startswith("datasets-logic-capability:sha256:")
        # Capability labels alone are insufficient: receipts bind exact modules.
        assert all(
            item.module.startswith("ipfs_datasets_py.logic.")
            for item in probe.symbol_receipts
        )

    # IR identity surface is actually invoked, not merely imported.
    identity = call_logic_ir_identity(
        {"obligation_id": "obl:1", "premise_ids": ["premise:a"]},
        domain="conformance",
        schema_version="logic-ir/v1",
    )
    assert identity["logic_ir_interface"] == LOGIC_IR_INTERFACE
    assert identity["digest"]
    assert identity["cid"]
    assert identity["candidate"] is True
    assert identity["authoritative_assurance"] == DATASETS_LOGIC_CANDIDATE_ASSURANCE

    # TDFOL / CEC / SMT / Hammer facades invoke real signatures through prove().
    call_log: list[tuple[str, dict]] = []

    def hook(operation: str, details: dict) -> None:
        call_log.append((operation, dict(details)))

    registry, _probes = build_datasets_logic_backend_registry(invocation_hook=hook)
    assert {item.kind for item in registry.registrations} == set(
        DatasetsLogicBackendKind
    )

    ir = registry.require(DatasetsLogicBackendKind.IR).provider
    ir_result = ir.prove(
        {
            "obligation_id": "obl:ir",
            "premise_ids": ["premise:schema"],
            "snapshot_id": "tree:fixture",
        }
    )
    assert ir_result["candidate"] is True
    assert ir_result["authoritative_assurance"] == DATASETS_LOGIC_CANDIDATE_ASSURANCE
    assert ir_result["proof_success"] is False

    tdfol = registry.require(DatasetsLogicBackendKind.TDFOL).provider
    tdfol_result = tdfol.prove(
        {
            "obligation_id": "obl:tdfol",
            "action": "policy_before_effect",
            "timeout_ms": 50,
        }
    )
    assert tdfol_result["candidate"] is True
    assert tdfol_result["proof_success"] is False

    cec = registry.require(DatasetsLogicBackendKind.CEC).provider
    cec_result = cec.prove(
        {
            "obligation_id": "obl:cec",
            "statement": "Obligatory(agent, keep_policy)",
        }
    )
    assert cec_result["candidate"] is True
    assert cec_result["proof_success"] is False

    hammer = registry.require(DatasetsLogicBackendKind.HAMMER).provider
    hammer_result = hammer.prove({"obligation_id": "obl:hammer"})
    assert hammer_result["candidate"] is True
    assert hammer_result["proof_success"] is False

    # SMT may lack the native solver package; either a candidate or typed
    # unsupported/unavailable is acceptable, never silent local success.
    smt = registry.require(DatasetsLogicBackendKind.SMT).provider
    smt_result = smt.prove({"obligation_id": "obl:smt", "formula": "True"})
    if hasattr(smt_result, "ok"):
        assert smt_result.ok is False
        assert smt_result.error.code.value in {"unsupported", "unavailable"}
        assert smt_result.error.details.get("proof_success") is False
    else:
        assert smt_result["candidate"] is True
        assert smt_result["proof_success"] is False

    operations = {operation for operation, _details in call_log}
    assert "compute_identity" in operations or "capability" in operations
    assert "tdfol_create_obligation" in operations or "tdfol_prove" in operations
    assert "cec_add_statement" in operations
    assert "hammer_bound" in operations or "hammer_select_premises" in operations


def test_premise_selection_retains_corpus_and_goal_identities() -> None:
    manifest = _real_corpus_manifest()
    goal_id = "goal:reviewed-transition"
    selection = select_premises_retaining_identities(
        corpus_manifest=manifest,
        goal_statement="Ready may transition only when the current state is ready.",
        goal_theorem_id=goal_id,
        top_k=2,
        corpus_revision=manifest.revision,
    )

    assert selection["corpus_revision"] == manifest.revision
    assert selection["goal_theorem_id"] == goal_id
    assert selection["candidate"] is True
    assert selection["selected_premise_ids"]
    assert set(selection["selected_premise_ids"]).issubset(
        {"premise:relation", "premise:state", "premise:extra"}
    )

    with pytest.raises(DatasetsLogicBackendError, match="corpus revision"):
        select_premises_retaining_identities(
            corpus_manifest=manifest,
            goal_statement="Ready may transition only when the current state is ready.",
            goal_theorem_id=goal_id,
            top_k=2,
            corpus_revision="corpus:forged-revision",
        )


def test_only_registered_capability_probed_backends_run() -> None:
    # Capability labels / forged probes cannot register.
    registry = DatasetsLogicBackendRegistry()
    forged = DatasetsLogicBackendProbe(
        kind=DatasetsLogicBackendKind.SMT,
        provider_id="smt",
        available=True,
        reconstruction_compatible=True,
        mcp_route="smt",
        package_version="forged",
        capability_revision="forged-revision",
        symbol_receipts=(
            DatasetsLogicSymbolReceipt(
                module="not.a.real.module",
                name="fake",
                qualname="not.a.real.module.fake",
                available=False,
                signature="",
                reason_code="symbol_missing",
            ),
        ),
    )
    with pytest.raises(
        DatasetsLogicBackendError,
        match="capability labels alone|signature_probe_required|unavailable",
    ):
        # Available=True with failed receipts must still refuse registration.
        registry.register(forged)

    # Unavailable probes never register.
    unavailable = probe_datasets_logic_backend(
        DatasetsLogicBackendKind.SMT,
        importer=lambda name: (_ for _ in ()).throw(
            ModuleNotFoundError(name)
        ),
    )
    assert unavailable.available is False
    with pytest.raises(DatasetsLogicBackendError, match="unavailable"):
        registry.register(unavailable)

    # Empty prover has no registered datasets backends.
    empty = McpContractProver(provider_getter=lambda _provider_id: None)
    assert not datasets_logic_backends_are_registered(
        empty, ContractProofRoute.SMT, ContractProofRoute.CEC
    )
    relation = _obligation(McpClaimFamily.TRANSPORT_PARITY)
    result = empty.prove(relation)
    assert result.outcome is ContractProofOutcome.UNSUPPORTED
    assert result.reason_codes == ("provider_unavailable",)

    # Registered capability-probed backends are the only ones that run.
    call_log: list[str] = []

    def hook(operation: str, _details: dict) -> None:
        call_log.append(operation)

    prover, built = create_mcp_contract_prover_with_datasets_logic_backends(
        invocation_hook=hook,
        kinds=(
            DatasetsLogicBackendKind.SMT,
            DatasetsLogicBackendKind.CEC,
            DatasetsLogicBackendKind.TDFOL,
        ),
    )
    assert built.registrations
    assert datasets_logic_backends_are_registered(
        prover,
        ContractProofRoute.SMT,
        ContractProofRoute.CEC,
        ContractProofRoute.TDFOL,
    )

    # An unregistered provider id must not be resolved through the global registry.
    assert prover._resolve_provider(ContractProofRoute.KERNEL)[0] is None


def test_solver_output_remains_candidate_until_trusted_reconstruction() -> None:
    registry, _probes = build_datasets_logic_backend_registry(
        kinds=(DatasetsLogicBackendKind.CEC, DatasetsLogicBackendKind.TDFOL)
    )
    provider = registry.require(DatasetsLogicBackendKind.CEC).provider
    raw = provider.prove(
        {
            "obligation_id": "obl:candidate",
            "statement": "Obligatory(agent, keep_policy)",
            "premise_ids": ["premise:schema", "premise:route"],
        }
    )
    assert raw["candidate"] is True
    assert raw["proof_success"] is False
    assert raw["kernel_checked"] is False
    assert raw["authoritative_assurance"] == DATASETS_LOGIC_CANDIDATE_ASSURANCE
    assert raw["reconstruction_required"] is True

    reconstruct = provider.reconstruct({"obligation_id": "obl:candidate"})
    assert reconstruct.ok is False
    assert reconstruct.error.details["reason_code"] == "trusted_reconstruction_required"

    # Without a trusted receipt validator, positive provider output stays
    # non-authoritative even after capability-probed dispatch.
    prover, _registry = create_mcp_contract_prover_with_datasets_logic_backends(
        kinds=(DatasetsLogicBackendKind.CEC,),
    )
    result = prover.prove(_obligation(McpClaimFamily.POLICY_BEFORE_EFFECT))
    assert result.route is ContractProofRoute.CEC
    assert result.outcome is not ContractProofOutcome.PROVED
    assert result.receipt.authoritative_assurance is not AssuranceLevel.KERNEL_VERIFIED
    assert result.receipt.authoritative_assurance in {
        AssuranceLevel.UNVERIFIED,
        AssuranceLevel.CANDIDATE,
        AssuranceLevel.SOLVER_CHECKED,
    }

    # A trusted validator is the only promotion path; forged provider assurance
    # fields inside the candidate payload are ignored.
    trusted_seen: list[Mapping[str, object]] = []

    def trusted_validator(obligation, raw_result):
        trusted_seen.append(dict(raw_result))
        return None

    prover_with_validator, _ = create_mcp_contract_prover_with_datasets_logic_backends(
        kinds=(DatasetsLogicBackendKind.CEC,),
        trusted_receipt_validator=trusted_validator,
    )
    validated = prover_with_validator.prove(
        _obligation(McpClaimFamily.POLICY_BEFORE_EFFECT)
    )
    assert trusted_seen, "trusted reconstruction must observe solver candidates"
    assert validated.outcome is not ContractProofOutcome.PROVED
    assert validated.receipt.authoritative_assurance is not AssuranceLevel.KERNEL_VERIFIED


def test_cec_native_wiring_parses_typed_formula_before_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Formula:
        pass

    class DCECStatement:
        pass

    formula = Formula()
    statement = DCECStatement()

    def parse_dcec_string(expression: str):
        events.append(("parse", expression))
        return formula

    class DCECContainer:
        def __init__(self) -> None:
            events.append("initialize")

        def add_statement(self, value, *, label=None):
            events.append(("add", value, label))
            return statement

    native_module = SimpleNamespace(
        parse_dcec_string=parse_dcec_string,
        Formula=Formula,
        DCECContainer=DCECContainer,
        DCECStatement=DCECStatement,
    )
    provider = create_datasets_logic_backend_provider(
        DatasetsLogicBackendKind.CEC
    )
    monkeypatch.setattr(
        provider,
        "_importer",
        lambda name: (
            native_module
            if name == "ipfs_datasets_py.logic.CEC.native"
            else pytest.fail(f"unexpected CEC import: {name}")
        ),
    )

    result = provider.prove(
        {
            "obligation_id": "obl:native-cec",
            "statement": "Obligatory(agent, keep_policy)",
        }
    )

    assert result["candidate"] is True
    assert result["proof_success"] is False
    assert result["kernel_checked"] is False
    assert result["authoritative_assurance"] == DATASETS_LOGIC_CANDIDATE_ASSURANCE
    assert result["invocation"] == {
        "symbol": (
            "ipfs_datasets_py.logic.CEC.native.DCECContainer.add_statement"
        ),
        "statement_valid": True,
        "formula_type": "Formula",
        "statement_type": "DCECStatement",
    }
    assert events == [
        ("parse", "Obligatory(agent, keep_policy)"),
        "initialize",
        ("add", formula, "mcp-obligation"),
    ]
    assert [entry["operation"] for entry in provider.call_log] == [
        "cec_parse_statement",
        "cec_add_statement",
    ]


@pytest.mark.parametrize(
    ("failure_stage", "reason_code", "failure_code"),
    (
        ("parse", "cec_statement_parse_failed", "unsupported"),
        ("typed_formula", "cec_typed_formula_required", "unsupported"),
        ("initialize", "cec_runtime_unavailable", "unavailable"),
        ("add", "cec_statement_add_failed", "unsupported"),
    ),
)
def test_cec_native_wiring_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    reason_code: str,
    failure_code: str,
) -> None:
    class Formula:
        pass

    class DCECStatement:
        pass

    def parse_dcec_string(_expression: str):
        if failure_stage == "parse":
            raise ValueError("invalid DCEC expression")
        if failure_stage == "typed_formula":
            return object()
        return Formula()

    class DCECContainer:
        def __init__(self) -> None:
            if failure_stage == "initialize":
                raise RuntimeError("container unavailable")

        def add_statement(self, _formula, *, label=None):
            assert label == "mcp-obligation"
            if failure_stage == "add":
                raise RuntimeError("statement rejected")
            return DCECStatement()

    native_module = SimpleNamespace(
        parse_dcec_string=parse_dcec_string,
        Formula=Formula,
        DCECContainer=DCECContainer,
        DCECStatement=DCECStatement,
    )
    provider = create_datasets_logic_backend_provider(
        DatasetsLogicBackendKind.CEC
    )
    monkeypatch.setattr(
        provider,
        "_importer",
        lambda name: (
            native_module
            if name == "ipfs_datasets_py.logic.CEC.native"
            else pytest.fail(f"unexpected CEC import: {name}")
        ),
    )

    response = provider.prove(
        {
            "obligation_id": "obl:native-cec-failure",
            "statement": "Obligatory(agent, keep_policy)",
        }
    )

    assert response.ok is False
    assert response.error.code.value == failure_code
    assert response.error.details["reason_code"] == reason_code
    assert response.error.details["proof_success"] is False
    assert response.error.details.get("candidate", False) is False


def test_unavailable_providers_are_unsupported_not_local_success() -> None:
    def broken_importer(name: str):
        raise ModuleNotFoundError(name)

    probes = probe_all_datasets_logic_backends(importer=broken_importer)
    assert probes
    assert all(not probe.available for probe in probes)
    assert all(probe.reason_code for probe in probes)

    with pytest.raises(DatasetsLogicBackendError):
        create_datasets_logic_backend_provider(
            DatasetsLogicBackendKind.TDFOL,
            importer=broken_importer,
        )

    prover, registry = create_mcp_contract_prover_with_datasets_logic_backends(
        importer=broken_importer,
        kinds=(
            DatasetsLogicBackendKind.SMT,
            DatasetsLogicBackendKind.CEC,
            DatasetsLogicBackendKind.TDFOL,
        ),
    )
    assert registry.registrations == ()
    for family, route in (
        (McpClaimFamily.TRANSPORT_PARITY, ContractProofRoute.SMT),
        (McpClaimFamily.POLICY_BEFORE_EFFECT, ContractProofRoute.CEC),
        (McpClaimFamily.SNAPSHOT_FRESHNESS, ContractProofRoute.TDFOL),
    ):
        result = prover.prove(_obligation(family))
        assert result.route is route
        assert result.outcome is ContractProofOutcome.UNSUPPORTED
        assert "provider_unavailable" in result.reason_codes
        assert result.receipt.verdict is ProofVerdict.UNSUPPORTED
        # Local checkers must not silently absorb remote-route obligations.
        assert result.fallback_used is True


def test_fixture_only_provider_cannot_satisfy_exact_module_gate() -> None:
    class FixtureOnlyProvider:
        provider_id = "smt"
        provider_version = "fixture"

        def capability(self, payload=None):
            return {
                "capability": ProofProviderCapability(
                    provider_id="smt",
                    provider_version="fixture",
                    operations=(
                        ProofProviderOperation.CAPABILITY,
                        ProofProviderOperation.PROVE,
                    ),
                ).to_dict()
            }

        def prove(self, payload, **kwargs):
            return {
                "outcome": "proved",
                "assurance": "kernel_verified",
                "proof_success": True,
            }

    # The exact-module registry refuses fixture providers without probes.
    registry = DatasetsLogicBackendRegistry()
    with pytest.raises(DatasetsLogicBackendError):
        registry.register(
            SimpleNamespace(  # type: ignore[arg-type]
                available=True,
                kind=DatasetsLogicBackendKind.SMT,
                provider_id="smt",
                symbol_receipts=(),
                capability_revision="fixture",
            )
        )

    # Even if a fixture is injected into the MCP prover, forged kernel assurance
    # is rejected and never becomes authoritative without trusted reconstruction.
    result = McpContractProver(
        providers={ContractProofRoute.SMT: FixtureOnlyProvider()}
    ).prove(_obligation(McpClaimFamily.TRANSPORT_PARITY))
    assert result.outcome is ContractProofOutcome.INCONCLUSIVE
    assert result.reason_codes == ("provider_assurance_rejected",)
    assert result.receipt.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert result.receipt.provider_claimed_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_registered_backend_provider_capability_matches_probe() -> None:
    provider = create_datasets_logic_backend_provider(DatasetsLogicBackendKind.IR)
    capability_payload = provider.capability()
    capability = ProofProviderCapability.from_dict(capability_payload["capability"])
    assert capability.provider_id == "logic-ir"
    assert capability.supports(ProofProviderOperation.PROVE)
    assert capability.metadata["logic_ir_interface"] == LOGIC_IR_INTERFACE
    assert capability.metadata["candidate_authoritative"] is False
    assert capability.metadata["available"] is True
    assert provider.probe.to_dict()["schema"].endswith("datasets-logic-probe@1")
