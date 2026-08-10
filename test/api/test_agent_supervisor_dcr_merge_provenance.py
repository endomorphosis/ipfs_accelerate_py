"""Contract tests for the DCR-074 evidence-only publication boundary."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AuthorityStage,
    DeterministicRepairDisposition,
    PostEditValidationReceipt,
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
    ReproofReceipt,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.publish import (
    PublicationDisposition,
    RepairPublicationError,
    RepairPublisher,
    SubmodulePinTransition,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _cid(name: str) -> str:
    return content_identity({"fixture": name})


def _reproved() -> RepairEvidenceEnvelope:
    roots = RepairAuthorityRoots(
        repository_id="repository:fixture",
        repository_forest_cid=_cid("forest"),
        git_tree_id=_cid("tree"),
        policy_root=_cid("policy"),
        rpr_plan_cid=_cid("plan"),
        rpr_packet_cid=_cid("packet"),
    )
    observed = RepairEvidenceEnvelope(
        repair_id="repair:fixture",
        disposition=DeterministicRepairDisposition.REPAIRED_PENDING_VALIDATION,
        authority_stage=AuthorityStage.OBSERVED,
        authority_roots=roots,
        observation_cid=_cid("observation"),
    )
    derived = RepairEvidenceEnvelope(
        repair_id="repair:fixture",
        disposition=observed.disposition,
        authority_stage=AuthorityStage.DERIVED,
        authority_roots=roots,
        observation_cid=observed.observation_cid,
        previous_authority_stage=observed.authority_stage,
        previous_envelope_cid=observed.content_id,
        derivation_cid=_cid("derivation"),
    )
    admission = RepairAdmissionReceipt(
        repair_id="repair:fixture",
        authority_roots=roots,
        predecessor_evidence_cid=derived.content_id,
        derivation_cid=derived.derivation_cid,
    )
    admitted = RepairEvidenceEnvelope(
        repair_id="repair:fixture", disposition=observed.disposition,
        authority_stage=AuthorityStage.ADMITTED, authority_roots=roots,
        observation_cid=observed.observation_cid,
        previous_authority_stage=derived.authority_stage,
        previous_envelope_cid=derived.content_id, derivation_cid=derived.derivation_cid,
        admission_cid=admission.content_id, admission_receipt=admission,
    )
    mutated = RepairEvidenceEnvelope(
        repair_id="repair:fixture", disposition=observed.disposition,
        authority_stage=AuthorityStage.MUTATED, authority_roots=roots,
        observation_cid=observed.observation_cid,
        previous_authority_stage=admitted.authority_stage,
        previous_envelope_cid=admitted.content_id, derivation_cid=admitted.derivation_cid,
        admission_cid=admitted.admission_cid, admission_receipt=admission,
        mutation_receipt_cid=_cid("mutation"),
    )
    validation = PostEditValidationReceipt(
        repair_id="repair:fixture", authority_roots=roots,
        predecessor_evidence_cid=mutated.content_id,
        admission_receipt_cid=admission.content_id,
        mutation_receipt_cid=mutated.mutation_receipt_cid, passed=True,
    )
    validated = RepairEvidenceEnvelope(
        repair_id="repair:fixture", disposition=observed.disposition,
        authority_stage=AuthorityStage.POST_EDIT_VALIDATED, authority_roots=roots,
        observation_cid=observed.observation_cid,
        previous_authority_stage=mutated.authority_stage,
        previous_envelope_cid=mutated.content_id, derivation_cid=mutated.derivation_cid,
        admission_cid=admission.content_id, admission_receipt=admission,
        mutation_receipt_cid=mutated.mutation_receipt_cid,
        post_edit_validation_cid=validation.content_id,
        post_edit_validation_receipt=validation,
    )
    reproof = ReproofReceipt(
        repair_id="repair:fixture", authority_roots=roots,
        predecessor_evidence_cid=validated.content_id,
        admission_receipt_cid=admission.content_id,
        post_edit_validation_receipt_cid=validation.content_id,
        mutation_receipt_cid=mutated.mutation_receipt_cid, proved=True,
    )
    return RepairEvidenceEnvelope(
        repair_id="repair:fixture", disposition=observed.disposition,
        authority_stage=AuthorityStage.REPROVED, authority_roots=roots,
        observation_cid=observed.observation_cid,
        previous_authority_stage=validated.authority_stage,
        previous_envelope_cid=validated.content_id, derivation_cid=validated.derivation_cid,
        admission_cid=admission.content_id, admission_receipt=admission,
        mutation_receipt_cid=mutated.mutation_receipt_cid,
        post_edit_validation_cid=validation.content_id,
        post_edit_validation_receipt=validation, reproof_cid=reproof.content_id,
        reproof_receipt=reproof,
    )


def _publish(*, observed_head: str = "commit:head"):
    transition = SubmodulePinTransition(
        owner_root="external/ipfs_accelerate", predecessor_pin="pin:old",
        successor_pin="pin:new", provider_commit_id="commit:provider",
        pin_commit_id="commit:consumer",
    )
    envelope = _reproved()
    return RepairPublisher().publish(
        envelope, target_ref="refs/heads/main", expected_target_head="commit:head",
        observed_target_head=observed_head, validation_evidence_cid=envelope.post_edit_validation_cid,
        provider_commit_ids=("commit:provider",), consumer_commit_ids=("commit:consumer",),
        pin_transitions=(transition,),
    )


def test_publishes_only_a_typed_evidence_transition() -> None:
    result = _publish()
    assert result.published
    assert result.provenance.disposition is PublicationDisposition.PUBLISHED
    assert result.published_envelope is not None
    assert result.published_envelope.authority_stage is AuthorityStage.PUBLISHED


def test_changed_head_requires_replan_without_minting_receipt() -> None:
    result = _publish(observed_head="commit:changed")
    assert not result.published
    assert result.provenance.disposition is PublicationDisposition.REPLAN
    assert result.provenance.reason_codes == ("target_head_changed",)
    assert result.publication_receipt is None


def test_validation_evidence_must_be_the_current_typed_receipt() -> None:
    envelope = _reproved()
    with pytest.raises(RepairPublicationError, match="current typed validation"):
        RepairPublisher().publish(
            envelope,
            target_ref="refs/heads/main",
            expected_target_head="commit:head",
            observed_target_head="commit:head",
            validation_evidence_cid=_cid("different-validation"),
            provider_commit_ids=("commit:provider",),
            consumer_commit_ids=("commit:consumer",),
        )


def test_pin_commit_must_be_a_distinct_consumer_commit() -> None:
    with pytest.raises(RepairPublicationError, match="provider and pin commits"):
        SubmodulePinTransition(
            owner_root="external/ipfs_accelerate", predecessor_pin="pin:old",
            successor_pin="pin:new", provider_commit_id="commit:shared",
            pin_commit_id="commit:shared",
        )
