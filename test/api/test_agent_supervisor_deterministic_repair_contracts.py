"""Focused DCR-002 tests for deterministic repair evidence authority."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AuthorityStage,
    DeterministicRepairAuthorityError,
    DeterministicRepairContractError,
    DeterministicRepairDisposition,
    ForgedRepairEvidenceIdentityError,
    PostEditValidationReceipt,
    PublicationReceipt,
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
    ReproofReceipt,
    assert_deterministic_repair_transition,
    closed_deterministic_repair_dispositions,
    parse_deterministic_repair_disposition,
    repair_evidence_cid,
    repair_evidence_digest,
    verify_repair_evidence_envelope,
)


def _cid(name: str) -> str:
    return repair_evidence_cid({"fixture": name})


@pytest.fixture
def roots() -> RepairAuthorityRoots:
    return RepairAuthorityRoots(
        repository_id="repository:sha256:fixture",
        repository_forest_cid=_cid("forest"),
        git_tree_id=_cid("tree"),
        policy_root=_cid("policy"),
        rpr_plan_cid=_cid("rpr-plan"),
        rpr_packet_cid=_cid("rpr-packet"),
    )


def _envelope(
    authority_stage: AuthorityStage,
    roots: RepairAuthorityRoots,
    previous: RepairEvidenceEnvelope | None = None,
    disposition: DeterministicRepairDisposition = DeterministicRepairDisposition.REPAIRED_PENDING_VALIDATION,
    **changes: object,
) -> RepairEvidenceEnvelope:
    values: dict[str, object] = {
        "repair_id": "repair:fixture",
        "disposition": disposition,
        "authority_stage": authority_stage,
        "authority_roots": roots,
        "observation_cid": _cid("observation"),
        "previous_authority_stage": previous.authority_stage if previous else None,
        "previous_envelope_cid": previous.content_id if previous else "",
        "derivation_cid": "",
        "admission_cid": "",
        "mutation_receipt_cid": "",
        "post_edit_validation_cid": "",
        "reproof_cid": "",
        "publication_cid": "",
    }
    if authority_stage in {
        AuthorityStage.DERIVED,
        AuthorityStage.ADMITTED,
        AuthorityStage.MUTATED,
        AuthorityStage.POST_EDIT_VALIDATED,
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    }:
        values["derivation_cid"] = _cid("derivation")
    if authority_stage in {
        AuthorityStage.ADMITTED,
        AuthorityStage.MUTATED,
        AuthorityStage.POST_EDIT_VALIDATED,
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    }:
        values["admission_cid"] = _cid("admission")
    if authority_stage in {
        AuthorityStage.MUTATED,
        AuthorityStage.POST_EDIT_VALIDATED,
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    }:
        values["mutation_receipt_cid"] = _cid("mutation")
    if authority_stage in {
        AuthorityStage.POST_EDIT_VALIDATED,
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    }:
        values["post_edit_validation_cid"] = _cid("validation")
    if authority_stage in {
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    }:
        values["reproof_cid"] = _cid("reproof")
    if authority_stage is AuthorityStage.PUBLISHED:
        values["publication_cid"] = _cid("publication")
    if authority_stage in {
        AuthorityStage.ADMITTED,
        AuthorityStage.MUTATED,
        AuthorityStage.POST_EDIT_VALIDATED,
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    }:
        admission = (
            previous.admission_receipt
            if previous is not None and previous.admission_receipt is not None
            else RepairAdmissionReceipt(
                repair_id="repair:fixture",
                authority_roots=roots,
                predecessor_evidence_cid=values["previous_envelope_cid"],
                derivation_cid=values["derivation_cid"],
            )
        )
        values["admission_receipt"] = admission
        values["admission_cid"] = admission.content_id
    if authority_stage in {
        AuthorityStage.POST_EDIT_VALIDATED,
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    }:
        validation = (
            previous.post_edit_validation_receipt
            if previous is not None and previous.post_edit_validation_receipt is not None
            else PostEditValidationReceipt(
                repair_id="repair:fixture",
                authority_roots=roots,
                predecessor_evidence_cid=values["previous_envelope_cid"],
                admission_receipt_cid=values["admission_cid"],
                mutation_receipt_cid=values["mutation_receipt_cid"],
                passed=True,
            )
        )
        values["post_edit_validation_receipt"] = validation
        values["post_edit_validation_cid"] = validation.content_id
    if authority_stage in {AuthorityStage.REPROVED, AuthorityStage.PUBLISHED}:
        reproof = (
            previous.reproof_receipt
            if previous is not None and previous.reproof_receipt is not None
            else ReproofReceipt(
                repair_id="repair:fixture",
                authority_roots=roots,
                predecessor_evidence_cid=values["previous_envelope_cid"],
                admission_receipt_cid=values["admission_cid"],
                post_edit_validation_receipt_cid=(values["post_edit_validation_cid"]),
                mutation_receipt_cid=values["mutation_receipt_cid"],
                proved=True,
            )
        )
        values["reproof_receipt"] = reproof
        values["reproof_cid"] = reproof.content_id
    if authority_stage is AuthorityStage.PUBLISHED:
        publication = PublicationReceipt(
            repair_id="repair:fixture",
            authority_roots=roots,
            predecessor_evidence_cid=values["previous_envelope_cid"],
            admission_receipt_cid=values["admission_cid"],
            post_edit_validation_receipt_cid=values["post_edit_validation_cid"],
            reproof_receipt_cid=values["reproof_cid"],
            mutation_receipt_cid=values["mutation_receipt_cid"],
            published=True,
        )
        values["publication_receipt"] = publication
        values["publication_cid"] = publication.content_id
    values.update(changes)
    return RepairEvidenceEnvelope(**values)  # type: ignore[arg-type]


def _complete_chain(roots: RepairAuthorityRoots) -> list[RepairEvidenceEnvelope]:
    chain = [_envelope(AuthorityStage.OBSERVED, roots)]
    for state in (
        AuthorityStage.DERIVED,
        AuthorityStage.ADMITTED,
        AuthorityStage.MUTATED,
        AuthorityStage.POST_EDIT_VALIDATED,
        AuthorityStage.REPROVED,
        AuthorityStage.PUBLISHED,
    ):
        outcome = (
            DeterministicRepairDisposition.COMPLETED
            if state is AuthorityStage.PUBLISHED
            else DeterministicRepairDisposition.REPAIRED_PENDING_VALIDATION
        )
        chain.append(_envelope(state, roots, chain[-1], disposition=outcome))
    return chain


def test_closed_dispositions_and_unknown_or_synthetic_values_fail_closed() -> None:
    assert {item.value for item in closed_deterministic_repair_dispositions()} == {
        "proved_valid",
        "refuted_repairable",
        "repaired_pending_validation",
        "abstain_review",
        "defer_capability",
        "rejected",
        "completed",
    }
    assert (
        parse_deterministic_repair_disposition("proved_valid")
        is DeterministicRepairDisposition.PROVED_VALID
    )
    with pytest.raises(DeterministicRepairContractError, match="must be one of"):
        parse_deterministic_repair_disposition("synthetic_authorize")

    class SyntheticDisposition(str):
        pass

    with pytest.raises(DeterministicRepairContractError, match="must be one of"):
        parse_deterministic_repair_disposition(SyntheticDisposition("proved_valid"))


def test_canonical_round_trip_cid_and_digest_are_order_independent(
    roots: RepairAuthorityRoots,
) -> None:
    observed = _envelope(AuthorityStage.OBSERVED, roots)
    replayed = RepairEvidenceEnvelope.from_dict(observed.to_record())

    assert replayed == observed
    assert replayed.content_id == observed.content_id
    assert repair_evidence_digest({"b": 2, "a": 1}) == repair_evidence_digest({"a": 1, "b": 2})
    assert repair_evidence_cid({"b": 2, "a": 1}) == repair_evidence_cid({"a": 1, "b": 2})

    forged = observed.to_record()
    forged["content_id"] = _cid("forged")
    with pytest.raises(ForgedRepairEvidenceIdentityError):
        RepairEvidenceEnvelope.from_dict(forged)


def test_observation_or_derivation_cannot_authorize_mutation_or_completion(
    roots: RepairAuthorityRoots,
) -> None:
    observed = _envelope(AuthorityStage.OBSERVED, roots)
    derived = _envelope(AuthorityStage.DERIVED, roots, observed)

    assert not observed.authorizes_mutation
    assert not derived.authorizes_mutation
    with pytest.raises(DeterministicRepairAuthorityError, match="illegal"):
        assert_deterministic_repair_transition("observed", "mutated")
    with pytest.raises(DeterministicRepairAuthorityError, match="illegal"):
        assert_deterministic_repair_transition("derived", "published")
    with pytest.raises(DeterministicRepairAuthorityError, match="must be observed"):
        assert_deterministic_repair_transition(None, "published")


def test_completion_requires_post_edit_validation_reproof_and_publication(
    roots: RepairAuthorityRoots,
) -> None:
    chain = _complete_chain(roots)
    completed = chain[-1]
    assert not completed.completion_authoritative
    assert completed.completion_structurally_complete
    with pytest.raises(DeterministicRepairAuthorityError, match="prior reproved"):
        verify_repair_evidence_envelope(completed, require_completion=True)
    assert (
        verify_repair_evidence_envelope(
            completed,
            expected_authority_roots=roots,
            previous=chain[-2],
            require_completion=True,
        )
        == completed
    )

    reproved = chain[-2]
    with pytest.raises(DeterministicRepairAuthorityError, match="publication is required"):
        _envelope(
            AuthorityStage.PUBLISHED,
            roots,
            reproved,
            disposition=DeterministicRepairDisposition.COMPLETED,
            publication_cid="",
        )


def test_transition_rebinds_exact_roots_and_prior_envelope(
    roots: RepairAuthorityRoots,
) -> None:
    observed = _envelope(AuthorityStage.OBSERVED, roots)
    derived = _envelope(AuthorityStage.DERIVED, roots, observed)
    drifted_roots = RepairAuthorityRoots(
        repository_id=roots.repository_id,
        repository_forest_cid=_cid("drifted-forest"),
        git_tree_id=roots.git_tree_id,
        policy_root=roots.policy_root,
        rpr_plan_cid=roots.rpr_plan_cid,
        rpr_packet_cid=roots.rpr_packet_cid,
    )
    drifted = _envelope(AuthorityStage.ADMITTED, drifted_roots, derived)

    with pytest.raises(DeterministicRepairAuthorityError, match="roots"):
        drifted.require_advances(derived)
    with pytest.raises(DeterministicRepairAuthorityError, match="previous_envelope_cid"):
        _envelope(
            AuthorityStage.ADMITTED,
            roots,
            derived,
            previous_envelope_cid=_cid("forged-previous"),
        ).require_advances(derived)


def test_raw_strings_never_authorize_mutation_or_completion(
    roots: RepairAuthorityRoots,
) -> None:
    observed = _envelope(AuthorityStage.OBSERVED, roots)
    derived = _envelope(AuthorityStage.DERIVED, roots, observed)
    raw_admitted = RepairEvidenceEnvelope(
        repair_id="repair:fixture",
        disposition=DeterministicRepairDisposition.REPAIRED_PENDING_VALIDATION,
        authority_stage=AuthorityStage.ADMITTED,
        authority_roots=roots,
        observation_cid=_cid("observation"),
        previous_authority_stage=AuthorityStage.DERIVED,
        previous_envelope_cid=derived.content_id,
        derivation_cid=_cid("derivation"),
        admission_cid=_cid("untyped-admission"),
    )
    assert not raw_admitted.authorizes_mutation
    with pytest.raises(DeterministicRepairAuthorityError, match="typed admission"):
        verify_repair_evidence_envelope(
            raw_admitted,
            previous=derived,
            require_mutation_authority=True,
        )

    raw_completed = RepairEvidenceEnvelope(
        repair_id="repair:fixture",
        disposition=DeterministicRepairDisposition.COMPLETED,
        authority_stage=AuthorityStage.PUBLISHED,
        authority_roots=roots,
        observation_cid=_cid("observation"),
        previous_authority_stage=AuthorityStage.REPROVED,
        previous_envelope_cid=_cid("prior"),
        derivation_cid=_cid("derivation"),
        admission_cid=_cid("admission"),
        mutation_receipt_cid=_cid("mutation"),
        post_edit_validation_cid=_cid("validation"),
        reproof_cid=_cid("reproof"),
        publication_cid=_cid("publication"),
    )
    assert not raw_completed.completion_authoritative
    with pytest.raises(DeterministicRepairAuthorityError, match="prior reproved"):
        verify_repair_evidence_envelope(raw_completed, require_completion=True)


def test_typed_receipts_reject_forgery_stale_links_and_root_drift(
    roots: RepairAuthorityRoots,
) -> None:
    completed = _complete_chain(roots)[-1]
    previous = _complete_chain(roots)[-2]

    forged = completed.to_record()
    assert isinstance(forged["publication_receipt"], dict)
    forged["publication_receipt"]["content_id"] = _cid("forged-publication")
    with pytest.raises(ForgedRepairEvidenceIdentityError):
        RepairEvidenceEnvelope.from_dict(forged)

    record = completed.to_record()
    record.pop("content_id")
    assert isinstance(record["publication_receipt"], dict)
    record["publication_receipt"]["published"] = False
    with pytest.raises(DeterministicRepairAuthorityError, match="typed publication"):
        verify_repair_evidence_envelope(
            RepairEvidenceEnvelope.from_dict(record),
            previous=previous,
            require_completion=True,
        )

    stale_publication = PublicationReceipt(
        repair_id=completed.repair_id,
        authority_roots=roots,
        predecessor_evidence_cid=_cid("stale-predecessor"),
        admission_receipt_cid=completed.admission_cid,
        post_edit_validation_receipt_cid=completed.post_edit_validation_cid,
        reproof_receipt_cid=completed.reproof_cid,
        mutation_receipt_cid=completed.mutation_receipt_cid,
        published=True,
    )
    stale = _envelope(
        AuthorityStage.PUBLISHED,
        roots,
        previous,
        disposition=DeterministicRepairDisposition.COMPLETED,
        publication_receipt=stale_publication,
        publication_cid=stale_publication.content_id,
    )
    with pytest.raises(DeterministicRepairAuthorityError, match="publication receipt"):
        verify_repair_evidence_envelope(
            stale,
            previous=previous,
            require_completion=True,
        )

    drifted_roots = RepairAuthorityRoots(
        repository_id=roots.repository_id,
        repository_forest_cid=_cid("drifted-forest"),
        git_tree_id=roots.git_tree_id,
        policy_root=roots.policy_root,
        rpr_plan_cid=roots.rpr_plan_cid,
        rpr_packet_cid=roots.rpr_packet_cid,
    )
    stale_publication = PublicationReceipt(
        repair_id=completed.repair_id,
        authority_roots=drifted_roots,
        predecessor_evidence_cid=completed.previous_envelope_cid,
        admission_receipt_cid=completed.admission_cid,
        post_edit_validation_receipt_cid=completed.post_edit_validation_cid,
        reproof_receipt_cid=completed.reproof_cid,
        mutation_receipt_cid=completed.mutation_receipt_cid,
        published=True,
    )
    drifted = _envelope(
        AuthorityStage.PUBLISHED,
        roots,
        previous,
        disposition=DeterministicRepairDisposition.COMPLETED,
        publication_receipt=stale_publication,
        publication_cid=stale_publication.content_id,
    )
    with pytest.raises(DeterministicRepairAuthorityError, match="roots"):
        verify_repair_evidence_envelope(
            drifted,
            previous=previous,
            require_completion=True,
        )


def test_successors_cannot_replace_established_receipts_or_evidence(
    roots: RepairAuthorityRoots,
) -> None:
    chain = _complete_chain(roots)
    derived, admitted, mutated = chain[1:4]
    replacement_admission = RepairAdmissionReceipt(
        repair_id=admitted.repair_id,
        authority_roots=roots,
        predecessor_evidence_cid=_cid("replacement-admission-predecessor"),
        derivation_cid=admitted.derivation_cid,
    )
    replaced_mutation = _envelope(
        AuthorityStage.MUTATED,
        roots,
        admitted,
        admission_receipt=replacement_admission,
        admission_cid=replacement_admission.content_id,
    )
    with pytest.raises(DeterministicRepairAuthorityError, match="admission_cid must remain"):
        verify_repair_evidence_envelope(replaced_mutation, previous=admitted)

    reproved = chain[-2]
    replacement_reproof = ReproofReceipt(
        repair_id=reproved.repair_id,
        authority_roots=roots,
        predecessor_evidence_cid=_cid("replacement-reproof-predecessor"),
        admission_receipt_cid=reproved.admission_cid,
        post_edit_validation_receipt_cid=reproved.post_edit_validation_cid,
        mutation_receipt_cid=reproved.mutation_receipt_cid,
        proved=True,
    )
    replaced_publication = _envelope(
        AuthorityStage.PUBLISHED,
        roots,
        reproved,
        disposition=DeterministicRepairDisposition.COMPLETED,
        reproof_receipt=replacement_reproof,
        reproof_cid=replacement_reproof.content_id,
    )
    with pytest.raises(DeterministicRepairAuthorityError, match="reproof_cid must remain"):
        verify_repair_evidence_envelope(replaced_publication, previous=reproved)

    assert derived.derivation_cid == admitted.derivation_cid
    assert admitted.admission_receipt == mutated.admission_receipt
