from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.invariant_mining import (
    ASSURANCE_CAMPAIGN_API_INTERFACE_PIN,
    REQUIRED_VACUITY_CLASSES,
    AdversarialFixture,
    AssuranceApiAdapter,
    AssuranceApiStatus,
    InvariantMiner,
    InvariantMiningError,
    InvariantValidator,
    NonVacuityValidator,
    VacuityClass,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.specification_mining import (
    CandidateStatus,
    EvidenceTier,
    PropertyKind,
    SourceKind,
    SourceProvenance,
    SpecificationCandidate,
)


def bindings(**overrides: str) -> ArtifactBindings:
    payload = {
        "repository_id": "repo-main",
        "repository_commit": "abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G000",
        "task_id": "PCPC-014",
        "contract_revision": "procedure-contracts-v1",
        "policy_revision": "authority-policy-v1",
        "environment_id": "python312-linux-lock1",
    }
    payload.update(overrides)
    return ArtifactBindings(**payload)


def provenance(
    *,
    source_kind: SourceKind = SourceKind.TYPE,
    evidence_tier: EvidenceTier | None = None,
    provenance_cid: str = "prov-a",
    artifact_cid: str = "art-a",
) -> SourceProvenance:
    return SourceProvenance(
        source_kind=source_kind,
        evidence_tier=evidence_tier or {
            SourceKind.TYPE: EvidenceTier.TYPE_DECLARATION,
            SourceKind.TEST: EvidenceTier.TEST_OBSERVATION,
            SourceKind.MUTANT: EvidenceTier.MUTANT_OBSERVATION,
            SourceKind.RUNTIME_CHECK: EvidenceTier.RUNTIME_OBSERVATION,
        }.get(source_kind, EvidenceTier.NOMINATION),
        provenance_cid=provenance_cid,
        artifact_cid=artifact_cid,
    )


def candidate(
    *,
    property_kind: PropertyKind = PropertyKind.INVARIANT,
    property_id: str = "invariant.scope",
    binding: str = "procedure.scope_paths",
    operator: ConditionOperator = ConditionOperator.SUBSET_OF,
    operand: object = ("ipfs_accelerate_py/agent_supervisor",),
    evidence_cid: str = "evidence-a",
    source_kind: SourceKind = SourceKind.TYPE,
    passing_test_count: int = 0,
    status: CandidateStatus = CandidateStatus.CANDIDATE,
) -> SpecificationCandidate:
    source = provenance(source_kind=source_kind)
    return SpecificationCandidate(
        property_kind=property_kind,
        property_id=property_id,
        binding=binding,
        operator=operator,
        operand=operand,
        required=True,
        status=status,
        evidence_tier=source.evidence_tier,
        evidence_cids=(evidence_cid,),
        source_provenances=(source,),
        passing_test_count=passing_test_count,
    )


def fixture(
    vacuity_class: VacuityClass,
    property_id: str,
    *,
    witness: dict[str, object] | None = None,
    evidence_cid: str | None = None,
) -> AdversarialFixture:
    slug = vacuity_class.value.replace("_", "-")
    return AdversarialFixture(
        fixture_id=f"{slug}.{property_id}",
        vacuity_class=vacuity_class,
        target_property_id=property_id,
        evidence_cid=evidence_cid or f"vacuity.{vacuity_class.value}.{property_id}",
        witness=witness or {},
    )


class RecordingCampaignApi:
    def __init__(
        self,
        *,
        findings: tuple[dict[str, str], ...] = (),
        finding_cids: tuple[str, ...] = ("aae-vacuity-receipt",),
        result_cid: str = "aae-vacuity-result",
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.findings = findings
        self.finding_cids = finding_cids
        self.result_cid = result_cid

    @property
    def interface_id(self) -> str:
        return ASSURANCE_CAMPAIGN_API_INTERFACE_PIN

    def probe_api(self, name: str) -> dict[str, object]:
        return {"command": name, "available": True, "status": "available"}

    def analyze_vacuity(self, manifest: object, repository_state: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(
            {
                "manifest": manifest,
                "repository_state": repository_state,
                "kwargs": kwargs,
            }
        )
        return {
            "result_cid": self.result_cid,
            "finding_cids": self.finding_cids,
            "findings": self.findings,
            "precise_nonclaims": ("completeness-beyond-tested-obligations",),
            "residual_properties": (),
        }


class UnavailableCampaignApi:
    @property
    def interface_id(self) -> str:
        return ASSURANCE_CAMPAIGN_API_INTERFACE_PIN

    def probe_api(self, name: str) -> dict[str, object]:
        return {
            "command": name,
            "available": False,
            "status": "typed_unavailable",
            "reason_code": "assurance_api_unavailable",
        }


def validator(api: object | None = None) -> NonVacuityValidator:
    campaign = api if api is not None else RecordingCampaignApi()
    return NonVacuityValidator(campaign_api=campaign, emitted_at_ms=11)


def test_assurance_api_adapter_binds_existing_campaign_interface() -> None:
    api = RecordingCampaignApi()
    adapter = AssuranceApiAdapter(api, emitted_at_ms=3)
    probe = adapter.probe()
    assert adapter.interface_id == ASSURANCE_CAMPAIGN_API_INTERFACE_PIN
    assert probe.available is True
    assert probe.status is AssuranceApiStatus.AVAILABLE
    assert probe.command == "analyze_vacuity"
    observation = adapter.analyze_vacuity(
        bindings=bindings(),
        fixtures=(fixture(VacuityClass.EMPTY_OUTPUT, "postcondition.tests-admitted"),),
        candidate_ids=("postcondition.tests-admitted",),
        tested_obligation_ids=("postcondition.tests-admitted",),
    )
    assert observation.status is AssuranceApiStatus.AVAILABLE
    assert "aae-vacuity-result" in observation.receipt_cids
    assert api.calls
    subjects = api.calls[0]["kwargs"]["subjects"]
    assert subjects[0]["vacuity_family"] == "test"
    assert api.calls[0]["manifest"]["completeness_claimed"] is False
    assert api.calls[0]["repository_state"]["tree_id"] == "tree-abc123"


def test_adapter_unavailable_is_typed_and_does_not_weaken_rejection() -> None:
    api = UnavailableCampaignApi()
    result = NonVacuityValidator(campaign_api=api).validate(
        (
            candidate(
                property_kind=PropertyKind.PRECONDITION,
                property_id="precondition.impossible",
                binding="bindings.tree_id",
                operator=ConditionOperator.EXISTS,
                operand=(),
                evidence_cid="empty-pre",
            ),
        ),
        bindings=bindings(),
    )
    assert result.assurance_api_status is AssuranceApiStatus.TYPED_UNAVAILABLE
    assert result.surviving == ()
    assert result.refused[0].status is CandidateStatus.REFUSED
    assert result.counterexamples[0].conflict_class == "impossible_precondition"
    assert result.completeness_claimed is False


@pytest.mark.parametrize(
    ("vacuity_class", "subject", "extra_fixture_witness"),
    [
        (
            VacuityClass.IMPOSSIBLE_PRECONDITION,
            candidate(
                property_kind=PropertyKind.PRECONDITION,
                property_id="precondition.impossible",
                binding="bindings.tree_id",
                operator=ConditionOperator.CURRENT,
                operand="tree-other",
                evidence_cid="stale-tree",
            ),
            {"contradictory": True},
        ),
        (
            VacuityClass.UNREACHABLE_BRANCH,
            candidate(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.after-dead-step",
                binding="step:dead-step",
                operator=ConditionOperator.ADMITTED,
                operand=None,
                evidence_cid="dead-branch",
            ),
            {"reachable": False},
        ),
        (
            VacuityClass.EMPTY_OUTPUT,
            candidate(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.empty-output",
                binding="local:test-result",
                operator=ConditionOperator.IN_CLOSED_SET,
                operand=(),
                evidence_cid="empty-domain",
            ),
            {"empty_domain": True},
        ),
        (
            VacuityClass.NO_INVOCATION,
            candidate(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.never-invoked",
                binding="local:test-result",
                operator=ConditionOperator.ADMITTED,
                operand=None,
                evidence_cid="no-invocation",
                source_kind=SourceKind.TEST,
                passing_test_count=1,
            ),
            {"invoked": False},
        ),
        (
            VacuityClass.MOCK_EFFECT,
            candidate(
                property_kind=PropertyKind.EFFECT,
                property_id="effect.write",
                binding="procedure.declared_effects",
                operator=ConditionOperator.EXISTS,
                operand="repository_write",
                evidence_cid="mock-effect-observation",
                source_kind=SourceKind.MUTANT,
            ),
            {"mock_substitution": True, "observed_effect": False},
        ),
        (
            VacuityClass.FIXTURE_SHORTCUT,
            candidate(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.fixture-only",
                binding="local:test-result",
                operator=ConditionOperator.ADMITTED,
                operand=None,
                evidence_cid="shortcut-fixture-only",
                source_kind=SourceKind.TEST,
                passing_test_count=1,
            ),
            {"fixture_shortcut": True},
        ),
        (
            VacuityClass.CONSTANT_RESTATEMENT,
            candidate(
                property_kind=PropertyKind.INVARIANT,
                property_id="invariant.tautology",
                binding="constant",
                operator=ConditionOperator.EQUALS,
                operand=True,
                evidence_cid="constant-true",
            ),
            {"constant_restatement": True},
        ),
        (
            VacuityClass.INVARIANT_COUNTEREXAMPLE,
            candidate(
                property_kind=PropertyKind.INVARIANT,
                property_id="invariant.scope",
                binding="procedure.scope_paths",
                operator=ConditionOperator.SUBSET_OF,
                operand=("ipfs_accelerate_py/agent_supervisor",),
                evidence_cid="scope-claim",
            ),
            {"violating_observation_cid": "escape-write"},
        ),
    ],
)
def test_adversarial_vacuity_fixtures_are_rejected(
    vacuity_class: VacuityClass,
    subject: SpecificationCandidate,
    extra_fixture_witness: dict[str, object],
) -> None:
    attack = fixture(vacuity_class, subject.property_id, witness=extra_fixture_witness)
    result = validator().validate((subject,), bindings=bindings(), fixtures=(attack,))
    assert result.surviving == ()
    assert len(result.refused) == 1
    refused = result.refused[0]
    assert refused.status is CandidateStatus.REFUSED
    assert refused.to_artifact(result.bindings).state is ArtifactState.REJECTED
    assert result.counterexamples[0].property_id == subject.property_id
    assert result.counterexamples[0].conflict_class == vacuity_class.value
    artifact = result.counterexamples[0].to_artifact(result.bindings)
    assert artifact.state is ArtifactState.REJECTED
    assert parse_procedure_artifact(artifact.to_dict()) == artifact
    assert vacuity_class.value in result.receipt.facts["vacuity_classes_tested"]
    assert result.receipt.facts["completeness_claimed"] is False


def test_structural_impossible_precondition_and_empty_output_do_not_need_fixtures() -> None:
    result = validator().validate(
        (
            candidate(
                property_kind=PropertyKind.PRECONDITION,
                property_id="precondition.empty-set",
                binding="bindings.tree_id",
                operator=ConditionOperator.IN_CLOSED_SET,
                operand=(),
                evidence_cid="empty-precondition",
            ),
            candidate(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.empty-output",
                binding="local:output",
                operator=ConditionOperator.EXISTS,
                operand=(),
                evidence_cid="empty-post",
            ),
        ),
        bindings=bindings(),
    )
    refused_ids = {item.property_id for item in result.refused}
    assert refused_ids == {"precondition.empty-set", "postcondition.empty-output"}
    classes = {item.conflict_class for item in result.counterexamples}
    assert "impossible_precondition" in classes
    assert "empty_output" in classes


def test_surviving_candidate_retains_adversarial_receipts_and_current_bindings() -> None:
    api = RecordingCampaignApi()
    healthy = candidate()
    result = NonVacuityValidator(campaign_api=api, emitted_at_ms=21).validate(
        (healthy,),
        bindings=bindings(),
        current_bindings=bindings(),
        fixtures=(),
    )
    assert len(result.surviving) == 1
    survivor = result.surviving[0]
    assert survivor.status is CandidateStatus.CANDIDATE
    assert survivor.property_id == "invariant.scope"
    assert "aae-vacuity-result" in survivor.evidence_cids
    assert "aae-vacuity-receipt" in survivor.evidence_cids
    assert result.bindings == bindings()
    assert result.bindings.tree_id == "tree-abc123"
    assert result.adversarial_receipt_cids == ("aae-vacuity-result", "aae-vacuity-receipt")
    assert result.receipt.bindings == bindings()
    assert result.receipt.facts["current_tree_id"] == "tree-abc123"
    assert result.receipt.facts["current_repository_commit"] == "abc123"
    assert result.receipt.facts["adversarial_receipt_cids"] == (
        "aae-vacuity-result",
        "aae-vacuity-receipt",
    )
    decoded = parse_procedure_artifact(result.receipt.to_dict())
    assert decoded == result.receipt
    assert decoded.state is ArtifactState.CANDIDATE


def test_survivors_cannot_claim_completeness_or_upgrade_status() -> None:
    result = validator().validate((candidate(),), bindings=bindings())
    assert result.completeness_claimed is False
    assert result.upgraded_count == 0
    assert result.receipt.state is ArtifactState.CANDIDATE
    assert result.receipt.facts["completeness_claimed"] is False
    assert result.receipt.facts["verified_count"] == 0
    assert result.receipt.facts["upgraded_count"] == 0
    assert "completeness-beyond-tested-obligations" in result.receipt.facts["precise_nonclaims"]
    assert set(result.receipt.facts["tested_obligation_ids"]) == {"invariant.scope"}
    assert tuple(result.receipt.facts["vacuity_classes_tested"]) == tuple(
        item.value for item in REQUIRED_VACUITY_CLASSES
    )
    artifact = result.surviving[0].to_artifact(result.bindings)
    assert artifact.state is ArtifactState.CANDIDATE
    assert ArtifactState.VERIFIED.value not in artifact.labels
    with pytest.raises(InvariantMiningError, match="completeness"):
        from ipfs_accelerate_py.agent_supervisor.procedure_compiler.invariant_mining import (
            NonVacuityValidationResult,
        )

        NonVacuityValidationResult(
            bindings=result.bindings,
            surviving=result.surviving,
            refused=result.refused,
            counterexamples=result.counterexamples,
            fixtures=result.fixtures,
            tested_obligation_ids=result.tested_obligation_ids,
            vacuity_classes_tested=result.vacuity_classes_tested,
            adversarial_receipt_cids=result.adversarial_receipt_cids,
            assurance_api_status=result.assurance_api_status,
            campaign=result.campaign,
            receipt=result.receipt,
            completeness_claimed=True,
        )


def test_all_required_vacuity_classes_are_exercised_together() -> None:
    subjects = {
        VacuityClass.IMPOSSIBLE_PRECONDITION: candidate(
            property_kind=PropertyKind.PRECONDITION,
            property_id="precondition.impossible",
            binding="bindings.tree_id",
            operator=ConditionOperator.EXISTS,
            operand=(),
            evidence_cid="pre-empty",
        ),
        VacuityClass.UNREACHABLE_BRANCH: candidate(
            property_kind=PropertyKind.POSTCONDITION,
            property_id="postcondition.dead",
            binding="step:dead-step",
            operator=ConditionOperator.ADMITTED,
            operand=None,
            evidence_cid="post-dead",
        ),
        VacuityClass.EMPTY_OUTPUT: candidate(
            property_kind=PropertyKind.POSTCONDITION,
            property_id="postcondition.empty",
            binding="local:output",
            operator=ConditionOperator.EQUALS,
            operand=(),
            evidence_cid="post-empty",
        ),
        VacuityClass.NO_INVOCATION: candidate(
            property_kind=PropertyKind.POSTCONDITION,
            property_id="postcondition.uninvoked",
            binding="local:output",
            operator=ConditionOperator.ADMITTED,
            operand=None,
            evidence_cid="post-uninvoked",
            source_kind=SourceKind.TEST,
            passing_test_count=1,
        ),
        VacuityClass.MOCK_EFFECT: candidate(
            property_kind=PropertyKind.EFFECT,
            property_id="effect.mock-write",
            binding="procedure.declared_effects",
            operator=ConditionOperator.EXISTS,
            operand="repository_write",
            evidence_cid="mock-effect",
            source_kind=SourceKind.MUTANT,
        ),
        VacuityClass.FIXTURE_SHORTCUT: candidate(
            property_kind=PropertyKind.POSTCONDITION,
            property_id="postcondition.shortcut",
            binding="local:output",
            operator=ConditionOperator.ADMITTED,
            operand=None,
            evidence_cid="shortcut-fixture-only",
            source_kind=SourceKind.TEST,
            passing_test_count=1,
        ),
        VacuityClass.CONSTANT_RESTATEMENT: candidate(
            property_kind=PropertyKind.INVARIANT,
            property_id="invariant.constant",
            binding="constant",
            operator=ConditionOperator.EQUALS,
            operand=True,
            evidence_cid="inv-constant",
        ),
        VacuityClass.INVARIANT_COUNTEREXAMPLE: candidate(
            property_kind=PropertyKind.INVARIANT,
            property_id="invariant.scope-escape",
            binding="procedure.scope_paths",
            operator=ConditionOperator.SUBSET_OF,
            operand=("ipfs_accelerate_py/agent_supervisor",),
            evidence_cid="inv-escape",
        ),
    }
    attacks = tuple(
        fixture(
            vacuity_class,
            subject.property_id,
            witness={
                VacuityClass.UNREACHABLE_BRANCH: {"reachable": False},
                VacuityClass.NO_INVOCATION: {"invoked": False},
                VacuityClass.MOCK_EFFECT: {"mock_substitution": True, "observed_effect": False},
                VacuityClass.FIXTURE_SHORTCUT: {"fixture_shortcut": True},
                VacuityClass.CONSTANT_RESTATEMENT: {"constant_restatement": True},
                VacuityClass.INVARIANT_COUNTEREXAMPLE: {
                    "violating_observation_cid": "out-of-scope"
                },
                VacuityClass.IMPOSSIBLE_PRECONDITION: {"contradictory": True},
                VacuityClass.EMPTY_OUTPUT: {"empty_domain": True},
            }[vacuity_class],
        )
        for vacuity_class, subject in subjects.items()
    )
    healthy = candidate()
    result = validator().validate(
        (*subjects.values(), healthy),
        bindings=bindings(),
        fixtures=attacks,
    )
    assert {item.property_id for item in result.refused} == {
        item.property_id for item in subjects.values()
    }
    assert {item.conflict_class for item in result.counterexamples} == {
        item.value for item in REQUIRED_VACUITY_CLASSES
    }
    assert {item.property_id for item in result.surviving} == {"invariant.scope"}
    assert result.receipt.facts["completeness_claimed"] is False


def test_binding_drift_is_refused_before_campaign() -> None:
    with pytest.raises(InvariantMiningError, match="current bindings"):
        validator().validate(
            (candidate(),),
            bindings=bindings(),
            current_bindings=bindings(tree_id="tree-other", repository_commit="other"),
        )


def test_invariant_validator_rejects_counterexamples_and_keeps_survivors_candidate() -> None:
    api = RecordingCampaignApi()
    healthy = candidate()
    broken = candidate(
        property_id="invariant.stale-tree",
        binding="bindings.tree_id",
        operator=ConditionOperator.CURRENT,
        operand="tree-other",
        evidence_cid="stale-invariant",
    )
    result = InvariantValidator(campaign_api=api, emitted_at_ms=9).validate(
        (healthy, broken),
        bindings=bindings(),
        fixtures=(
            fixture(
                VacuityClass.INVARIANT_COUNTEREXAMPLE,
                "invariant.stale-tree",
                witness={"violating_observation_cid": "world-delta"},
            ),
        ),
    )
    assert {item.property_id for item in result.surviving} == {"invariant.scope"}
    assert {item.property_id for item in result.refused} == {"invariant.stale-tree"}
    assert result.counterexamples[0].conflict_class == "invariant_counterexample"
    assert result.invariant_artifacts[0].state is ArtifactState.CANDIDATE
    assert result.invariant_artifacts[0].bindings == bindings()
    assert "aae-vacuity-result" in result.surviving[0].evidence_cids
    assert result.completeness_claimed is False
    assert result.receipt.state is ArtifactState.CANDIDATE
    assert result.receipt.facts["verified_count"] == 0
    assert result.receipt.facts["completeness_claimed"] is False
    assert result.non_vacuity.receipt.content_id == result.receipt.facts["non_vacuity_receipt_cid"]
    decoded = parse_procedure_artifact(result.receipt.to_dict())
    assert decoded == result.receipt


def test_invariant_validator_requires_invariant_candidates() -> None:
    with pytest.raises(InvariantMiningError, match="invariant candidate"):
        InvariantValidator(campaign_api=RecordingCampaignApi()).validate(
            (
                candidate(
                    property_kind=PropertyKind.PRECONDITION,
                    property_id="precondition.current-tree",
                    binding="bindings.tree_id",
                    operator=ConditionOperator.CURRENT,
                    operand="tree-abc123",
                ),
            ),
            bindings=bindings(),
        )


def test_invariant_miner_candidates_remain_unvalidated_until_independent_campaign() -> None:
    from ipfs_accelerate_py.agent_supervisor.procedure_compiler.specification_mining import (
        AdmittedSource,
        PropertyNomination,
    )

    source = AdmittedSource(
        bindings=bindings(),
        source_id="type.scope",
        source_kind=SourceKind.TYPE,
        evidence_tier=EvidenceTier.TYPE_DECLARATION,
        provenance_cid="prov-scope",
        artifact_cid="art-scope",
        nominations=(
            PropertyNomination(
                property_kind=PropertyKind.INVARIANT,
                property_id="invariant.scope",
                binding="procedure.scope_paths",
                operator=ConditionOperator.SUBSET_OF,
                operand=("ipfs_accelerate_py/agent_supervisor",),
                evidence_cid="scope-type",
            ),
        ),
    )
    mined = InvariantMiner().mine((source,))
    assert mined.candidates
    assert mined.upgraded_count == 0
    validated = InvariantValidator(campaign_api=RecordingCampaignApi()).validate(
        mined,
        bindings=bindings(),
    )
    assert validated.surviving
    assert validated.receipt.facts["completeness_claimed"] is False
    assert validated.assurance_api_status is AssuranceApiStatus.AVAILABLE


def test_campaign_findings_fail_closed_union_with_local_witnesses() -> None:
    api = RecordingCampaignApi(
        findings=({"property_id": "invariant.scope", "vacuity_class": "invariant_counterexample"},),
        finding_cids=("aae-finding-scope",),
        result_cid="aae-finding-result",
    )
    result = NonVacuityValidator(campaign_api=api).validate(
        (candidate(),),
        bindings=bindings(),
    )
    assert result.surviving == ()
    assert result.refused[0].property_id == "invariant.scope"
    assert "aae-finding-result" in result.refused[0].evidence_cids
