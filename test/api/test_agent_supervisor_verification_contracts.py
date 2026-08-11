from __future__ import annotations

import inspect
import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofVerdict,
    ResourceBudget,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ProofReceipt as FormalProofReceipt,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestExecutionKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    PROOF_OBLIGATION_NOT_APPLICABLE_CID,
    CacheReuseDecision,
    CacheReuseDisposition,
    CounterexampleReceipt,
    DirectExecutionObservation,
    ModelRoute,
    ModelRouteDecision,
    ProofReceipt,
    StaticAnalysisReceipt,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationBoundsError,
    VerificationBundle,
    VerificationCommitment,
    VerificationContractError,
    VerificationIdentityCompiler,
    VerificationIdentityError,
    VerificationPlan,
    VerificationReceiptKey,
    VerificationReceiptKind,
    VerificationSummary,
    aggregate_terminal_status,
)

TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
SEMANTIC_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1"
ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
OBLIGATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/verification-proof-obligation@1"
)


def _structured_cid(schema: str, value: object) -> str:
    return content_identity({"schema": schema, "value": value})


def _artifact(label: str) -> str:
    return content_identity({"artifact": label, "schema": "fixture-artifact@1"})


def _compiler_kwargs(
    kind: VerificationReceiptKind = VerificationReceiptKind.TYPE_CHECK,
) -> dict[str, object]:
    tree = {
        "base_git_tree": "0123456789abcdef",
        "patched_overlay": {"src/example.py": "sha256:source-v2"},
    }
    semantic = {
        "symbols": ["example.calculate@2"],
        "edge_root": "sha256:semantic-edges",
    }
    environment = {
        "network_policy": "deny_all",
        "sandbox_schema": "hermetic-sandbox@1",
        "filesystem": "read-only-source+private-artifacts",
        "executable": "/usr/bin/python3.12",
        "platform": "linux-x86_64",
        "toolchain": "locked-python@1",
    }
    proof_obligation = None
    if kind is VerificationReceiptKind.PROOF:
        proof_obligation = {
            "normalized_obligation": "not (x >= 0 implies result >= 0)",
            "translation_scheme": "python-contract-to-smtlib2@1",
            "negation_scheme": "countermodel-negation@1",
            "translator_version": "1.2.0",
        }
    return {
        "observed_repository_tree": tree,
        "claimed_repository_tree_cid": _structured_cid(TREE_SCHEMA, tree),
        "patch_base_tree_id": "git-tree:base",
        "repository_state_tree_id": "git-tree:base",
        "invalidation_plan_tree_id": "git-tree:base",
        "context_pack_tree_id": "git-tree:base",
        "observed_semantic_state": semantic,
        "repository_state_semantic_root_cid": _structured_cid(
            SEMANTIC_SCHEMA, semantic
        ),
        "invalidation_plan_semantic_root_cid": _structured_cid(
            SEMANTIC_SCHEMA, semantic
        ),
        "context_pack_semantic_root_cid": _structured_cid(
            SEMANTIC_SCHEMA, semantic
        ),
        "affected_symbol_versions": (
            {
                "symbol": "example.calculate",
                "version": 2,
                "source_cid": _artifact("source-v2"),
            },
        ),
        "observed_environment": environment,
        "claimed_environment_cid": _structured_cid(ENVIRONMENT_SCHEMA, environment),
        "dependency_lock_bytes": b"package==1.2.3 --hash=sha256:abcd\n",
        "selector_argv": (
            "/usr/bin/python3.12",
            "-m",
            "mypy" if kind is VerificationReceiptKind.TYPE_CHECK else "pytest",
            "src/example.py",
        ),
        "proof_obligation": proof_obligation,
        "tool_name": "z3" if kind is VerificationReceiptKind.PROOF else "mypy",
        "tool_version": "4.13.3" if kind is VerificationReceiptKind.PROOF else "1.18.2",
        "configuration_bytes": b"[tool]\nstrict = true\n",
        "fixture_data_bytes": (b"fixture-one\n", b"fixture-two\n"),
        "network_policy": "deny_all",
        "receipt_schema_version": 1,
        "receipt_kind": kind,
        "adapter_schema": (
            "z3-verification-adapter@1"
            if kind is VerificationReceiptKind.PROOF
            else "mypy-verification-adapter@1"
        ),
    }


def _key(
    kind: VerificationReceiptKind = VerificationReceiptKind.TYPE_CHECK,
    **changes: object,
) -> VerificationReceiptKey:
    values = _compiler_kwargs(kind)
    values.update(changes)
    return VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def _observation(
    key: VerificationReceiptKey,
    status: TerminalStatus = TerminalStatus.PASSED,
    *,
    label: str = "run",
) -> DirectExecutionObservation:
    return DirectExecutionObservation(
        receipt_key_cid=key.key_id,
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        terminal_status=status,
        command_argv=("/usr/bin/python3.12", "-m", key.tool_name),
        duration_ms=125,
        exit_code=0 if status is TerminalStatus.PASSED else 1,
        stdout_artifact_cid=_artifact(f"{label}-stdout"),
        stderr_artifact_cid=_artifact(f"{label}-stderr"),
        artifact_cids=(_artifact(f"{label}-report"),),
        reason_codes=(f"{label}_observed",),
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=30_000,
        cpu_time_ms=20_000,
        memory_bytes=512 * 1024 * 1024,
        disk_bytes=64 * 1024 * 1024,
        max_processes=4,
        max_premises=32,
        max_output_bytes=1_000_000,
        model_token_limit=4_096,
        provider_quota=1,
        network_allowed=False,
    )


def _formal_receipt(
    key: VerificationReceiptKey,
    evidence: tuple[ProofEvidence, ...],
    *,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> FormalProofReceipt:
    return FormalProofReceipt(
        obligation_id=key.proof_obligation_cid,
        plan_id=_artifact("formal-plan"),
        attempt_id="attempt:one",
        repository_id="repository:fixture",
        repository_tree_id=key.repository_tree_cid,
        ast_scope_ids=("scope:example.calculate",),
        premise_ids=("premise:contract",),
        translator_id="translator:python-smt@1",
        solver_id="solver:z3@4.13.3",
        kernel_id="kernel:reviewed-z3-result@1",
        toolchain_id="toolchain:locked@1",
        theorem_registry_id="registry:fixture@1",
        policy_id="policy:proof@1",
        resource_budget=_budget(),
        verdict=verdict,
        evidence=evidence,
        provider_id="provider:fixture",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        freshness=freshness,
    )


def _solver_evidence(
    key: VerificationReceiptKey,
    *,
    accepted: bool = True,
    simulated: bool = False,
) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.ACCEPTED if accepted else EvidenceVerdict.REJECTED,
        artifact_id=_artifact("solver-result"),
        subject_id=key.proof_obligation_cid,
        verifier_id="solver:z3@4.13.3",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=simulated,
        metadata={"counterexample_verified": not accepted},
    )


def _route(*, human: bool = False) -> ModelRouteDecision:
    route = ModelRoute.HUMAN_REVIEW_REQUIRED if human else ModelRoute.SMALL_LOCAL_MODEL
    return ModelRouteDecision(
        route=route,
        considered_routes=(ModelRoute.SMALL_LOCAL_MODEL, route)
        if human
        else (route,),
        decisive_reason_codes=(
            "unresolved_authority" if human else "localized_exact_counterexample"
        ,),
        required_capabilities=("bounded_context",),
        context_token_estimate=2_048,
        policy_cid=_artifact("route-policy"),
    )


def test_terminal_status_vocabulary_is_exact_and_closed() -> None:
    expected = {
        "passed",
        "failed",
        "proved",
        "disproved",
        "unknown",
        "timeout",
        "unavailable",
        "not_modeled",
        "stale",
        "invalid",
        "cancelled",
        "simulated",
    }
    assert {item.value for item in TerminalStatus} == expected
    assert all(item.terminal for item in TerminalStatus)
    assert {item for item in TerminalStatus if item.successful} == {
        TerminalStatus.PASSED,
        TerminalStatus.PROVED,
    }
    with pytest.raises(ValueError):
        TerminalStatus("PASSED")
    with pytest.raises(ValueError):
        TerminalStatus("timed_out")


def test_compiler_binds_exact_observed_target_separately_from_equal_base_roots() -> None:
    values = _compiler_kwargs()
    key = VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    assert values["patch_base_tree_id"] == values["repository_state_tree_id"]
    assert values["patch_base_tree_id"] == values["invalidation_plan_tree_id"]
    assert values["patch_base_tree_id"] == values["context_pack_tree_id"]
    assert key.repository_tree_cid == values["claimed_repository_tree_cid"]
    assert key.repository_tree_cid != values["patch_base_tree_id"]
    assert key.proof_obligation_cid == PROOF_OBLIGATION_NOT_APPLICABLE_CID
    assert VerificationReceiptKey.from_dict(key.to_record()) == key


@pytest.mark.parametrize(
    "field",
    (
        "repository_state_tree_id",
        "invalidation_plan_tree_id",
        "context_pack_tree_id",
    ),
)
def test_compiler_rejects_any_base_root_mismatch(field: str) -> None:
    values = _compiler_kwargs()
    values[field] = "git-tree:different"
    with pytest.raises(VerificationIdentityError, match="base trees disagree"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    (
        "repository_state_semantic_root_cid",
        "invalidation_plan_semantic_root_cid",
        "context_pack_semantic_root_cid",
    ),
)
def test_compiler_rejects_any_semantic_root_mismatch(field: str) -> None:
    values = _compiler_kwargs()
    values[field] = _artifact("wrong-semantic-root")
    with pytest.raises(VerificationIdentityError, match="semantic roots disagree"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_compiler_rejects_target_and_environment_claim_mismatch() -> None:
    values = _compiler_kwargs()
    values["claimed_repository_tree_cid"] = _artifact("wrong-tree")
    with pytest.raises(VerificationIdentityError, match="patched tree"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["claimed_environment_cid"] = _artifact("wrong-environment")
    with pytest.raises(VerificationIdentityError, match="effective environment"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_compiler_rejects_environment_network_policy_mismatch() -> None:
    values = _compiler_kwargs()
    values["network_policy"] = "loopback_only"
    with pytest.raises(VerificationIdentityError, match="network policy"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_every_key_input_mutation_changes_identity() -> None:
    base = _key()
    mutations: list[VerificationReceiptKey] = []

    tree = {"base_git_tree": "other", "patched_overlay": {}}
    mutations.append(
        _key(
            observed_repository_tree=tree,
            claimed_repository_tree_cid=_structured_cid(TREE_SCHEMA, tree),
        )
    )
    semantic = {"symbols": ["different"], "edge_root": "different"}
    semantic_cid = _structured_cid(SEMANTIC_SCHEMA, semantic)
    mutations.append(
        _key(
            observed_semantic_state=semantic,
            repository_state_semantic_root_cid=semantic_cid,
            invalidation_plan_semantic_root_cid=semantic_cid,
            context_pack_semantic_root_cid=semantic_cid,
        )
    )
    mutations.append(
        _key(
            affected_symbol_versions=(
                {"symbol": "other", "version": 9, "source_cid": _artifact("other")},
            )
        )
    )
    environment = {
        **_compiler_kwargs()["observed_environment"],  # type: ignore[dict-item]
        "platform": "linux-aarch64",
    }
    mutations.append(
        _key(
            observed_environment=environment,
            claimed_environment_cid=_structured_cid(ENVIRONMENT_SCHEMA, environment),
        )
    )
    mutations.extend(
        (
            _key(dependency_lock_bytes=b"different lock"),
            _key(selector_argv=("/usr/bin/python3.12", "-m", "mypy", "other.py")),
            _key(tool_name="pyright"),
            _key(tool_version="1.19.0"),
            _key(configuration_bytes=b"different config"),
            _key(fixture_data_bytes=(b"different fixture",)),
            _key(
                observed_environment={
                    **_compiler_kwargs()["observed_environment"],  # type: ignore[dict-item]
                    "network_policy": "loopback_only",
                },
                claimed_environment_cid=_structured_cid(
                    ENVIRONMENT_SCHEMA,
                    {
                        **_compiler_kwargs()["observed_environment"],  # type: ignore[dict-item]
                        "network_policy": "loopback_only",
                    },
                ),
                network_policy="loopback_only",
            ),
            _key(receipt_schema_version=2),
            _key(adapter_schema="mypy-verification-adapter@2"),
        )
    )

    assert len({item.key_id for item in mutations}) == len(mutations)
    assert all(item.key_id != base.key_id for item in mutations)

    proof_a = _key(VerificationReceiptKind.PROOF)
    values = _compiler_kwargs(VerificationReceiptKind.PROOF)
    obligation = dict(values["proof_obligation"])  # type: ignore[arg-type]
    obligation["negation_scheme"] = "alternate-negation@2"
    proof_b = _key(VerificationReceiptKind.PROOF, proof_obligation=obligation)
    assert proof_b.proof_obligation_cid != proof_a.proof_obligation_cid
    assert proof_b.key_id != proof_a.key_id


def test_key_canonicalizes_set_like_inputs_but_preserves_selector_order() -> None:
    values = _compiler_kwargs()
    symbols = values["affected_symbol_versions"]
    values["affected_symbol_versions"] = tuple(reversed(symbols))  # type: ignore[arg-type]
    fixtures = values["fixture_data_bytes"]
    values["fixture_data_bytes"] = tuple(reversed(fixtures))  # type: ignore[arg-type]
    assert VerificationIdentityCompiler().compile_key(**values).key_id == _key().key_id  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["selector_argv"] = tuple(reversed(values["selector_argv"]))  # type: ignore[arg-type]
    assert VerificationIdentityCompiler().compile_key(**values).key_id != _key().key_id  # type: ignore[arg-type]


def test_proof_applicability_and_translation_are_fail_closed() -> None:
    values = _compiler_kwargs(VerificationReceiptKind.PROOF)
    values["proof_obligation"] = None
    with pytest.raises(VerificationIdentityError, match="normalized obligation"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs(VerificationReceiptKind.PROOF)
    values["proof_obligation"] = {"normalized_obligation": "x"}
    with pytest.raises(VerificationIdentityError, match="translation bindings"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["proof_obligation"] = {
        "normalized_obligation": "x",
        "translation_scheme": "x@1",
        "negation_scheme": "x@1",
        "translator_version": "1",
    }
    with pytest.raises(VerificationIdentityError, match="non-proof"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_wrong_schema_interface_version_unknown_field_and_forged_id_reject() -> None:
    key = _key()
    for field, value in (
        ("schema", "wrong@1"),
        ("interface", "Wrong@1"),
        ("contract_version", 2),
        ("contract_version", True),
    ):
        payload = key.to_record()
        payload[field] = value
        with pytest.raises(VerificationContractError):
            VerificationReceiptKey.from_dict(payload)

    payload = key.to_record()
    payload["unexpected"] = "field"
    with pytest.raises(VerificationContractError, match="unsupported fields"):
        VerificationReceiptKey.from_dict(payload)

    payload = key.to_record()
    payload["key_id"] = _artifact("forged")
    with pytest.raises(VerificationIdentityError, match="does not match"):
        VerificationReceiptKey.from_dict(payload)


@pytest.mark.parametrize(
    "bad_value",
    (
        {"ratio": 0.5},
        {"secret": "do-not-hash"},
        {"nested": {"private_witness": "proof"}},
        {"authorization": "Bearer token"},
    ),
)
def test_identity_inputs_reject_floats_secrets_and_witnesses(
    bad_value: dict[str, object],
) -> None:
    values = _compiler_kwargs()
    values["observed_semantic_state"] = bad_value
    values["repository_state_semantic_root_cid"] = _artifact("claim")
    values["invalidation_plan_semantic_root_cid"] = _artifact("claim")
    values["context_pack_semantic_root_cid"] = _artifact("claim")
    with pytest.raises(VerificationContractError):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_oversized_identity_bytes_and_text_reject() -> None:
    values = _compiler_kwargs()
    values["dependency_lock_bytes"] = b"x" * (16 * 1_048_576 + 1)
    with pytest.raises(VerificationBoundsError):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    with pytest.raises(VerificationBoundsError):
        replace(_key(), tool_version="x" * 9_000)


@pytest.mark.parametrize(
    "status",
    (
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.SIMULATED,
        TerminalStatus.CANCELLED,
        TerminalStatus.INVALID,
    ),
)
def test_direct_receipts_preserve_nonaccepting_terminal_status(
    status: TerminalStatus,
) -> None:
    key = _key()
    receipt = TypeCheckReceipt(key=key, execution=_observation(key, status))
    assert receipt.status is status
    assert receipt.terminal_success is False
    assert TypeCheckReceipt.from_dict(receipt.to_record()) == receipt


def test_direct_execution_must_bind_key_tree_environment_and_argv() -> None:
    key = _key()
    other = _key(tool_version="other")
    with pytest.raises(VerificationIdentityError, match="receipt key"):
        TypeCheckReceipt(key=key, execution=_observation(other))

    observation = _observation(key)
    with pytest.raises(VerificationIdentityError, match="environment"):
        TypeCheckReceipt(
            key=key,
            execution=replace(observation, environment_cid=_artifact("other-env")),
        )

    with pytest.raises(VerificationContractError, match="command_argv"):
        replace(observation, command_argv=())


def test_nonproof_receipts_reject_proof_statuses() -> None:
    key = _key()
    with pytest.raises(VerificationContractError, match="non-proof"):
        TypeCheckReceipt(
            key=key,
            execution=_observation(key, TerminalStatus.PROVED),
        )


def test_test_pass_projection_comes_from_full_existing_receipt() -> None:
    key = _key(VerificationReceiptKind.TEST, tool_name="pytest", adapter_schema="pytest-adapter@1")
    source_key = TestExecutionKey(
        locator_cid=_artifact("test-locator"),
        # A repository forest identifies a wider checkout projection and is
        # deliberately distinct from the exact Git tree executed here.
        repository_forest_cid=_artifact("repository-forest"),
        git_tree_id=key.repository_tree_cid,
        fixture_cids=key.fixture_data_cids,
        pytest_version=key.tool_version,
        command_semantics_cid=key.selector_cid,
        config_cid=key.configuration_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        environment_cid=key.environment_cid,
    )
    existing = TestPassReceipt(
        execution_key_cid=source_key.execution_key_id,
        locator_cid=source_key.locator_cid,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        admitted=True,
    )
    receipt = TestReceipt(
        key=key,
        execution=_observation(key),
        test_pass_receipt=existing,
        test_execution_key=source_key,
    )
    assert receipt.status is TerminalStatus.PASSED
    assert receipt.terminal_success
    assert TestReceipt.from_dict(receipt.to_record()) == receipt

    forged = receipt.to_record()
    forged["status"] = TerminalStatus.SIMULATED.value
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        TestReceipt.from_dict(forged)

    not_admitted = replace(existing, admitted=False)
    rejected = TestReceipt(
        key=key,
        execution=_observation(key),
        test_pass_receipt=not_admitted,
        test_execution_key=source_key,
    )
    assert rejected.status is TerminalStatus.INVALID
    assert not rejected.terminal_success

    mismatched_key = replace(source_key, environment_cid=_artifact("wrong-env"))
    mismatched_receipt = replace(
        existing, execution_key_cid=mismatched_key.execution_key_id
    )
    with pytest.raises(VerificationIdentityError, match="does not match"):
        TestReceipt(
            key=key,
            execution=_observation(key),
            test_pass_receipt=mismatched_receipt,
            test_execution_key=mismatched_key,
        )

    mismatched_tree_key = replace(source_key, git_tree_id=_artifact("wrong-tree"))
    mismatched_tree_receipt = replace(
        existing, execution_key_cid=mismatched_tree_key.execution_key_id
    )
    with pytest.raises(VerificationIdentityError, match="does not match"):
        TestReceipt(
            key=key,
            execution=_observation(key),
            test_pass_receipt=mismatched_tree_receipt,
            test_execution_key=mismatched_tree_key,
        )


def test_receipt_success_is_not_an_independent_constructor_field() -> None:
    for receipt_type in (
        StaticAnalysisReceipt,
        TypeCheckReceipt,
        TestReceipt,
        ProofReceipt,
    ):
        assert "status" not in inspect.signature(receipt_type).parameters
        assert "passed" not in inspect.signature(receipt_type).parameters
        assert "proved" not in inspect.signature(receipt_type).parameters


def test_formal_proof_success_uses_existing_authoritative_assurance() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    formal = _formal_receipt(key, (_solver_evidence(key),))
    assert formal.authoritative_assurance is AssuranceLevel.SOLVER_CHECKED
    receipt = ProofReceipt(
        key=key,
        execution=_observation(key, TerminalStatus.PROVED),
        formal_proof_receipt=formal,
    )
    assert receipt.status is TerminalStatus.PROVED
    assert receipt.terminal_success
    assert ProofReceipt.from_dict(receipt.to_record()) == receipt


def test_provider_claim_simulation_staleness_and_counterexample_project_safely() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    provider_only = ProofEvidence(
        kind=EvidenceKind.SMT_CANDIDATE,
        authority=EvidenceAuthority.PROVIDER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=_artifact("provider-candidate"),
        subject_id=key.proof_obligation_cid,
        verifier_id="provider:untrusted",
        freshness=EvidenceFreshness.CURRENT,
        independent=False,
    )
    claimed = ProofReceipt(
        key=key,
        execution=_observation(key, TerminalStatus.UNKNOWN),
        formal_proof_receipt=_formal_receipt(key, (provider_only,)),
    )
    assert claimed.status is TerminalStatus.UNKNOWN
    assert not claimed.terminal_success

    simulated = ProofReceipt(
        key=key,
        execution=_observation(key, TerminalStatus.SIMULATED),
        formal_proof_receipt=_formal_receipt(
            key, (_solver_evidence(key, simulated=True),)
        ),
    )
    assert simulated.status is TerminalStatus.SIMULATED
    assert not simulated.terminal_success

    stale = ProofReceipt(
        key=key,
        execution=_observation(key, TerminalStatus.STALE),
        formal_proof_receipt=_formal_receipt(
            key,
            (_solver_evidence(key),),
            freshness=EvidenceFreshness.STALE,
        ),
    )
    assert stale.status is TerminalStatus.STALE

    disproved = ProofReceipt(
        key=key,
        execution=_observation(key, TerminalStatus.DISPROVED),
        formal_proof_receipt=_formal_receipt(
            key,
            (_solver_evidence(key, accepted=False),),
            verdict=ProofVerdict.DISPROVED,
        ),
    )
    assert disproved.status is TerminalStatus.DISPROVED
    assert not disproved.terminal_success


def test_proof_direct_observation_cannot_mint_proved_or_failed() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    for status in (
        TerminalStatus.PROVED,
        TerminalStatus.DISPROVED,
        TerminalStatus.PASSED,
        TerminalStatus.FAILED,
    ):
        with pytest.raises(VerificationContractError, match="conclusive proof"):
            ProofReceipt(key=key, execution=_observation(key, status))


def test_cache_decision_reuses_only_successful_receipt_statuses() -> None:
    key = _key()
    receipt = TypeCheckReceipt(key, _observation(key))
    decision = CacheReuseDecision(
        key_cid=key.key_id,
        disposition=CacheReuseDisposition.REUSED,
        reason_codes=("exact_current_production_receipt",),
        receipt_cid=receipt.receipt_id,
        candidate_status=receipt.status,
    )
    assert decision.reusable
    assert CacheReuseDecision.from_dict(decision.to_record()) == decision

    for status in (
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.SIMULATED,
        TerminalStatus.STALE,
        TerminalStatus.UNKNOWN,
    ):
        with pytest.raises(VerificationContractError, match="successful terminal"):
            replace(decision, candidate_status=status)

    payload = decision.to_record()
    payload["reusable"] = 1
    with pytest.raises(VerificationContractError, match="boolean"):
        CacheReuseDecision.from_dict(payload)


def test_model_route_is_provider_neutral_and_boolean_projection_is_strict() -> None:
    decision = _route()
    assert decision.route is ModelRoute.SMALL_LOCAL_MODEL
    assert ModelRouteDecision.from_dict(decision.to_record()) == decision
    parameters = inspect.signature(ModelRouteDecision).parameters
    assert not {"provider", "vendor", "model_id"} & set(parameters)

    payload = decision.to_record()
    payload["provider"] = "vendor-specific"
    with pytest.raises(VerificationContractError, match="unsupported fields"):
        ModelRouteDecision.from_dict(payload)

    payload = decision.to_record()
    payload["requires_human_review"] = 0
    with pytest.raises(VerificationContractError, match="boolean"):
        ModelRouteDecision.from_dict(payload)


def _plan(key: VerificationReceiptKey) -> VerificationPlan:
    return VerificationPlan(
        repository_tree_cid=key.repository_tree_cid,
        semantic_state_root_cid=key.semantic_state_root_cid,
        environment_cid=key.environment_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        required_receipt_keys=(key,),
        cache_reuse_decisions=(
            CacheReuseDecision(
                key_cid=key.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            ),
        ),
        affected_tests=(),
        fallback_tests=(),
        required_static_checks=(),
        required_type_checks=("src/example.py",),
        affected_proof_obligation_cids=(),
        full_suite_required=False,
        full_suite_reason_codes=(),
        human_review_required=False,
        human_review_reason_codes=(),
        expected_cpu_millis=1_000,
        expected_memory_bytes=256 * 1024 * 1024,
        expected_processes=1,
        expected_proof_slots=0,
        expected_artifact_bytes=1_000_000,
        step_timeouts_ms={"type-check": 30_000},
        max_execution_time_ms=60_000,
        dependency_dag={"type-check": ()},
        acceptance_criteria=("all required current checks pass",),
        policy_cid=_artifact("verification-policy"),
    )


def test_verification_plan_round_trip_order_and_fail_closed_dag() -> None:
    key = _key()
    plan = _plan(key)
    assert plan.execution_order == ("type-check",)
    assert VerificationPlan.from_dict(plan.to_record()) == plan

    with pytest.raises(VerificationContractError, match="cycle"):
        replace(
            plan,
            dependency_dag={"a": ("b",), "b": ("a",)},
            step_timeouts_ms={"a": 1, "b": 1},
        )
    with pytest.raises(VerificationContractError, match="reason_codes"):
        replace(plan, full_suite_required=True, full_suite_reason_codes=())
    with pytest.raises(VerificationContractError, match="boolean"):
        replace(plan, human_review_required=1)  # type: ignore[arg-type]


def test_counterexample_is_compact_typed_and_round_trips() -> None:
    key = _key()
    failed = TypeCheckReceipt(key, _observation(key, TerminalStatus.FAILED))
    counterexample = CounterexampleReceipt(
        failed_key_cid=key.key_id,
        failed_receipt_cid=failed.receipt_id,
        failed_selector="src/example.py",
        failure_identity_cid=_artifact("failure-identity"),
        relevant_symbol_version_cids=key.affected_symbol_version_cids,
        minimized_traceback=("example.py:12: expected str, observed int",),
        relevant_assertion="result must be a string",
        relevant_input={"state": "present", "value": {"argument_type": "int"}},
        expected_output={"state": "present", "value": "str"},
        observed_output={"state": "present", "value": "int"},
        source_spans=(
            {
                "path": "src/example.py",
                "start_line": 10,
                "end_line": 13,
                "artifact_cid": _artifact("source-span"),
                "symbol": "example.calculate",
            },
        ),
        environment_cid=key.environment_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        reproduction_argv=("/usr/bin/python3.12", "-m", "mypy", "src/example.py"),
        artifact_cids=(_artifact("bounded-diagnostic"),),
        minimized=True,
        reason_codes=("deterministic_slice_preserved_failure",),
    )
    assert CounterexampleReceipt.from_dict(counterexample.to_record()) == counterexample
    assert len(counterexample.canonical_bytes()) < 262_144

    with pytest.raises(VerificationContractError, match="private or witness"):
        replace(
            counterexample,
            relevant_input={"state": "present", "value": {"secret": "token"}},
        )
    with pytest.raises(VerificationBoundsError):
        replace(counterexample, minimized_traceback=("x" * 3_000,))


def _bundle(
    key: VerificationReceiptKey,
    receipt: TypeCheckReceipt,
) -> VerificationBundle:
    return VerificationBundle(
        plan_cid=_plan(key).plan_id,
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        required_check_key_cids=(key.key_id,),
        receipts=(receipt,),
        reused_receipt_cids=(),
        executed_receipt_cids=(receipt.receipt_id,),
        counterexamples=(),
        unresolved_requirement_ids=(),
        mandatory_fallback_pending=False,
        human_review_required=False,
        policy_cid=_artifact("verification-policy"),
    )


def test_bundle_binds_objects_ids_status_cardinality_tree_and_environment() -> None:
    key = _key()
    receipt = TypeCheckReceipt(key, _observation(key))
    bundle = _bundle(key, receipt)
    assert bundle.structurally_complete
    assert VerificationBundle.from_dict(bundle.to_record()) == bundle

    failed = TypeCheckReceipt(key, _observation(key, TerminalStatus.FAILED, label="failed"))
    assert not _bundle(key, failed).structurally_complete

    other_key = _key(tool_version="other")
    other_receipt = TypeCheckReceipt(other_key, _observation(other_key))
    with pytest.raises(VerificationIdentityError, match="required check set"):
        replace(bundle, receipts=(other_receipt,), executed_receipt_cids=(other_receipt.receipt_id,))

    with pytest.raises(VerificationContractError, match="one result per key"):
        replace(
            bundle,
            receipts=(receipt, failed),
            executed_receipt_cids=(receipt.receipt_id, failed.receipt_id),
        )

    env_values = _compiler_kwargs()
    environment = {
        **env_values["observed_environment"],  # type: ignore[dict-item]
        "platform": "linux-aarch64",
    }
    mixed_key = _key(
        observed_environment=environment,
        claimed_environment_cid=_structured_cid(ENVIRONMENT_SCHEMA, environment),
    )
    mixed_receipt = TypeCheckReceipt(mixed_key, _observation(mixed_key))
    with pytest.raises(VerificationIdentityError, match="mixed"):
        VerificationBundle(
            plan_cid=bundle.plan_cid,
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            required_check_key_cids=(key.key_id, mixed_key.key_id),
            receipts=(receipt, mixed_receipt),
            reused_receipt_cids=(),
            executed_receipt_cids=(receipt.receipt_id, mixed_receipt.receipt_id),
            counterexamples=(),
            unresolved_requirement_ids=(),
            mandatory_fallback_pending=False,
            human_review_required=False,
            policy_cid=bundle.policy_cid,
        )


def test_missing_bundle_key_must_be_explicitly_unresolved() -> None:
    key = _key()
    with pytest.raises(VerificationContractError, match="explicit unresolved"):
        VerificationBundle(
            plan_cid=_artifact("plan"),
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            required_check_key_cids=(key.key_id,),
            receipts=(),
            reused_receipt_cids=(),
            executed_receipt_cids=(),
            counterexamples=(),
            unresolved_requirement_ids=(),
            mandatory_fallback_pending=False,
            human_review_required=False,
            policy_cid=_artifact("policy"),
        )


def test_summary_round_trip_and_route_flag_consistency() -> None:
    key = _key()
    summary = VerificationSummary(
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        changed_symbol_version_cids=key.affected_symbol_version_cids,
        dependency_cone_symbols=("example.calculate",),
        selected_tests=("test/test_example.py::test_calculate",),
        reused_check_key_cids=(),
        executed_check_key_cids=(key.key_id,),
        failure_receipt_cids=(),
        counterexample_cids=(),
        unresolved_obligation_cids=(),
        full_suite_pending=False,
        human_review_required=False,
        verification_wall_time_ms=125,
        reused_time_saved_ms=0,
        counterexample_context_tokens=0,
        aggregate_terminal_status=TerminalStatus.PASSED,
        model_route_decision=_route(),
        policy_cid=_artifact("summary-policy"),
    )
    assert VerificationSummary.from_dict(summary.to_record()) == summary
    with pytest.raises(VerificationContractError, match="human-review"):
        replace(summary, human_review_required=True)


def _leaf(
    key: VerificationReceiptKey,
    receipt_cid: str,
    status: TerminalStatus,
) -> dict[str, str]:
    return {
        "key_cid": key.key_id,
        "receipt_cid": receipt_cid,
        "receipt_kind": key.receipt_kind.value,
        "status": status.value,
    }


def test_all_terminal_statuses_round_trip_in_commitment_leaves() -> None:
    for status in TerminalStatus:
        kind = (
            VerificationReceiptKind.PROOF
            if status in {TerminalStatus.PROVED, TerminalStatus.DISPROVED}
            else VerificationReceiptKind.TYPE_CHECK
        )
        key = _key(kind)
        commitment = VerificationCommitment(
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            required_check_key_cids=(key.key_id,),
            admitted_leaves=(_leaf(key, _artifact(f"receipt-{status.value}"), status),),
            public_statement={"requirement": "verify current patch"},
            unresolved_obligation_count=0,
        )
        assert VerificationCommitment.from_dict(commitment.to_record()) == commitment
        assert commitment.aggregate_terminal_status is status


def test_commitment_is_deterministic_sensitive_and_exact_membership_bound() -> None:
    first_key = _key()
    second_key = _key(tool_version="other")
    first_receipt = _artifact("first-receipt")
    second_receipt = _artifact("second-receipt")
    values = {
        "repository_tree_cid": first_key.repository_tree_cid,
        "environment_cid": first_key.environment_cid,
        "required_check_key_cids": (first_key.key_id, second_key.key_id),
        "admitted_leaves": (
            _leaf(first_key, first_receipt, TerminalStatus.PASSED),
            _leaf(second_key, second_receipt, TerminalStatus.PASSED),
        ),
        "public_statement": {"requirement": "all exact checks pass"},
        "unresolved_obligation_count": 0,
    }
    forward = VerificationCommitment(**values)
    reverse = VerificationCommitment(
        **{
            **values,
            "required_check_key_cids": tuple(
                reversed(values["required_check_key_cids"])
            ),
            "admitted_leaves": tuple(reversed(values["admitted_leaves"])),
        }
    )
    assert forward.merkle_root == reverse.merkle_root
    assert forward.commitment_id == reverse.commitment_id
    assert forward.aggregate_terminal_status is TerminalStatus.PASSED
    assert VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF is False

    changed = VerificationCommitment(
        **{
            **values,
            "admitted_leaves": (
                _leaf(first_key, _artifact("changed-receipt"), TerminalStatus.PASSED),
                _leaf(second_key, second_receipt, TerminalStatus.PASSED),
            ),
        }
    )
    assert changed.merkle_root != forward.merkle_root
    assert changed.commitment_id != forward.commitment_id

    foreign_key = _key(tool_version="foreign")
    with pytest.raises(VerificationIdentityError, match="exact required check set"):
        replace(
            forward,
            admitted_leaves=(
                _leaf(first_key, first_receipt, TerminalStatus.PASSED),
                _leaf(foreign_key, second_receipt, TerminalStatus.PASSED),
            ),
        )


def test_commitment_cannot_claim_pass_for_empty_or_unresolved_membership() -> None:
    key = _key()
    with pytest.raises(VerificationContractError, match="not be empty"):
        VerificationCommitment(
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            required_check_key_cids=(),
            admitted_leaves=(),
            public_statement={"requirement": "nothing"},
            unresolved_obligation_count=0,
        )

    commitment = VerificationCommitment(
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        required_check_key_cids=(key.key_id,),
        admitted_leaves=(_leaf(key, _artifact("receipt"), TerminalStatus.PASSED),),
        public_statement={"requirement": "current check"},
        unresolved_obligation_count=1,
    )
    assert commitment.aggregate_terminal_status is TerminalStatus.UNKNOWN
    forged = commitment.to_record()
    forged["aggregate_terminal_status"] = TerminalStatus.PASSED.value
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        VerificationCommitment.from_dict(forged)


def test_aggregate_status_precedence_is_fail_closed() -> None:
    assert aggregate_terminal_status((TerminalStatus.PASSED, TerminalStatus.PROVED)) is TerminalStatus.PASSED
    assert aggregate_terminal_status((TerminalStatus.PROVED,)) is TerminalStatus.PROVED
    assert aggregate_terminal_status((TerminalStatus.PASSED, TerminalStatus.TIMEOUT)) is TerminalStatus.TIMEOUT
    assert aggregate_terminal_status((TerminalStatus.FAILED, TerminalStatus.INVALID)) is TerminalStatus.INVALID
    assert aggregate_terminal_status((), unresolved_obligation_count=0) is TerminalStatus.UNKNOWN


def test_canonical_json_round_trip_is_deterministic_and_nested_values_are_frozen() -> None:
    decision = ModelRouteDecision(
        route=ModelRoute.SMALL_LOCAL_MODEL,
        considered_routes=(ModelRoute.SMALL_LOCAL_MODEL,),
        decisive_reason_codes=("localized_exact_counterexample",),
        required_capabilities=("exact_contracts", "bounded_context"),
        context_token_estimate=1_000,
        policy_cid=_artifact("policy"),
    )
    payload = decision.to_dict()
    reordered = {key: payload[key] for key in reversed(tuple(payload))}
    assert ModelRouteDecision.from_dict(reordered).content_id == decision.content_id
    assert json.loads(decision.to_json())["route"] == "small_local_model"

    key = _key()
    commitment = VerificationCommitment(
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        required_check_key_cids=(key.key_id,),
        admitted_leaves=(_leaf(key, _artifact("receipt"), TerminalStatus.PASSED),),
        public_statement={"nested": {"value": "immutable"}},
        unresolved_obligation_count=0,
    )
    with pytest.raises(TypeError):
        commitment.public_statement["new"] = "mutation"  # type: ignore[index]


def test_package_import_surface_is_complete() -> None:
    from ipfs_accelerate_py.agent_supervisor import verification

    for name in (
        "StaticAnalysisReceipt",
        "TypeCheckReceipt",
        "TestReceipt",
        "ProofReceipt",
        "CounterexampleReceipt",
        "VerificationBundle",
        "VerificationSummary",
        "CacheReuseDecision",
        "ModelRouteDecision",
        "VerificationPlan",
        "VerificationCommitment",
        "VerificationReceiptKey",
        "VerificationIdentityCompiler",
    ):
        assert name in verification.__all__
        assert getattr(verification, name) is not None


def test_raw_identity_helper_uses_real_cid_not_pseudo_hash() -> None:
    key = _key()
    assert key.dependency_lock_cid == cid_for_bytes(
        _compiler_kwargs()["dependency_lock_bytes"]  # type: ignore[arg-type]
    )
    assert key.dependency_lock_cid.startswith("bafkrei")
