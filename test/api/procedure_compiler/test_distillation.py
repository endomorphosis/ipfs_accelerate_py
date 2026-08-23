from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    HoleType,
    ProviderClass,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.distillation import (
    BUILDER_REVISION,
    REQUIRED_PARTITIONS,
    REQUIRED_PROVENANCE_FIELDS,
    CorpusPartition,
    DistillationAdmissionError,
    DistillationCorpus,
    DistillationCorpusBuilder,
    DistillationEvaluation,
    DistillationExample,
    DistillationLabel,
    DistillationReason,
    DistillationReferenceKind,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.hole_resolution import (
    HoleCandidate,
    HoleContextReference,
    HoleRequest,
    HoleResolution,
    HoleResolutionAction,
    HoleResolutionReason,
    HoleResolutionValidator,
    HoleValidationReceipt,
)


def _bindings(**changes: object) -> ArtifactBindings:
    values: dict[str, object] = {
        "repository_id": "repo-main",
        "repository_commit": "commit-abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G030",
        "task_id": "PCPC-021",
        "contract_revision": "procedure-contracts-v1",
        "policy_revision": "authority-policy-v1",
        "environment_id": "python312-linux-lock1",
    }
    values.update(changes)
    return ArtifactBindings(**values)


def _reference(**changes: object) -> HoleContextReference:
    values: dict[str, object] = {
        "reference_id": "evidence.symbols",
        "content_id": "cid-symbols-1",
        "tree_id": "tree-abc123",
        "byte_count": 64,
        "token_count": 16,
        "required": True,
        "summary": "allowed-symbols",
    }
    values.update(changes)
    return HoleContextReference(**values)


def _request(**changes: object) -> HoleRequest:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "hole_id": "hole.select-symbol",
        "hole_type": HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
        "input_schema_ref": "schema.symbol-in",
        "output_schema_ref": "schema.symbol-out",
        "allowed_provider_classes": (
            ProviderClass.EXACT_CACHE,
            ProviderClass.REMOTE_STANDARD_MODEL,
        ),
        "context_budget_bytes": 32_768,
        "validation_observation_ids": ("observation.tests", "proof.hole-outcome"),
        "fallback_step_id": "step.fallback",
        "maximum_attempts": 2,
        "input_payload": {"allowed_values": ("pkg.mod:symbol_a", "pkg.mod:symbol_b")},
        "context_references": (_reference(),),
        "authority_requirement_ids": ("authority.execute",),
        "effect_classes": (EffectClass.OBSERVE, EffectClass.MODEL_REQUEST),
        "token_budget": 4_096,
    }
    values.update(changes)
    return HoleRequest(**values)


def _candidate(request: HoleRequest, *, selected: str, fingerprint: str) -> HoleCandidate:
    return HoleCandidate(
        bindings=request.bindings,
        request_cid=request.content_id,
        hole_id=request.hole_id,
        hole_type=request.hole_type,
        output_schema_ref=request.output_schema_ref,
        provider_class=ProviderClass.EXACT_CACHE,
        output={"schema_ref": request.output_schema_ref, "selected": selected},
        context_receipt_cid=f"cid-context.{fingerprint}",
        evidence_fingerprint=fingerprint,
        token_count=8,
    )


def _resolution(request: HoleRequest, candidate: HoleCandidate) -> HoleResolution:
    return HoleResolution(
        bindings=request.bindings,
        request_cid=request.content_id,
        hole_id=request.hole_id,
        action=HoleResolutionAction.PROPOSE,
        reason_code=HoleResolutionReason.CANDIDATE_PROPOSED,
        fallback_step_id=request.fallback_step_id,
        provider_class=ProviderClass.EXACT_CACHE.value,
        candidate_cid=candidate.content_id,
        context_receipt_cid=candidate.context_receipt_cid,
        evidence_fingerprint=candidate.evidence_fingerprint,
        attempts_used=1,
    )


def _receipt(
    request: HoleRequest,
    candidate: HoleCandidate,
    *,
    observations: tuple[str, ...] = ("observation.tests", "proof.hole-outcome"),
) -> HoleValidationReceipt:
    validator = HoleResolutionValidator(current_tree_id=request.bindings.tree_id)
    return validator.validate_candidate(
        request,
        candidate,
        current_tree_id=request.bindings.tree_id,
        observations=observations,
    )


def _bundle(
    *,
    hole_id: str,
    selected: str,
    fingerprint: str,
    accepted: bool = True,
) -> tuple[HoleRequest, HoleCandidate, HoleResolution, HoleValidationReceipt]:
    request = _request(hole_id=hole_id)
    candidate = _candidate(request, selected=selected, fingerprint=fingerprint)
    resolution = _resolution(request, candidate)
    receipt = _receipt(request, candidate)
    assert receipt.accepted is accepted
    return request, candidate, resolution, receipt


def _builder(**changes: object) -> DistillationCorpusBuilder:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "corpus_id": "corpus.hole-residual",
        "current_tree_id": "tree-abc123",
    }
    values.update(changes)
    return DistillationCorpusBuilder(**values)


def _admit(
    builder: DistillationCorpusBuilder,
    bundle: tuple[HoleRequest, HoleCandidate, HoleResolution, HoleValidationReceipt],
    *,
    partition: CorpusPartition,
    **changes: object,
) -> DistillationExample:
    request, candidate, resolution, receipt = bundle
    values: dict[str, object] = {
        "request": request,
        "candidate": candidate,
        "resolution": resolution,
        "receipt": receipt,
        "partition": partition,
        "proof_cid": "proof.hole-outcome",
        "counterexample_cids": ("cex.boundary-near-match", "cex.negative-import"),
        "family_id": "family.import-purity",
        "language": "python",
        "framework": "pytest",
    }
    values.update(changes)
    return builder.admit(**values)


def test_corpus_admits_validated_accepted_and_rejected_examples() -> None:
    builder = _builder()
    accepted = _admit(
        builder,
        _bundle(hole_id="hole.select-symbol.train", selected="pkg.mod:symbol_a", fingerprint="cid-ev-train"),
        partition=CorpusPartition.TRAINING,
        example_id="ex.train.accepted",
    )
    rejected = _admit(
        builder,
        _bundle(
            hole_id="hole.select-symbol.heldout",
            selected="pkg.mod:not-allowed",
            fingerprint="cid-ev-heldout",
            accepted=False,
        ),
        partition=CorpusPartition.HELD_OUT,
        example_id="ex.heldout.rejected",
    )

    assert accepted.label is DistillationLabel.ACCEPTED
    assert accepted.outcome is DistillationLabel.ACCEPTED
    assert rejected.label is DistillationLabel.REJECTED
    assert rejected.outcome is DistillationLabel.REJECTED
    assert accepted.state is ArtifactState.CANDIDATE
    assert rejected.can_authorize is False
    assert accepted.builder_revision == BUILDER_REVISION

    corpus = builder.build()
    assert corpus.accepted_count == 1
    assert corpus.rejected_count == 1
    assert {row.label for row in corpus.rows} == {
        DistillationLabel.ACCEPTED,
        DistillationLabel.REJECTED,
    }
    evaluation = builder.evaluation
    assert isinstance(evaluation, DistillationEvaluation)
    assert evaluation.disjoint is True
    assert evaluation.complete_provenance is True
    assert evaluation.admitted_count == 2


def test_corpus_rows_are_bounded_disjoint_and_carry_complete_provenance() -> None:
    builder = _builder()
    _admit(
        builder,
        _bundle(hole_id="hole.select-symbol.train", selected="pkg.mod:symbol_a", fingerprint="cid-ev-train"),
        partition=CorpusPartition.TRAINING,
        example_id="ex.train.accepted",
    )
    _admit(
        builder,
        _bundle(
            hole_id="hole.select-symbol.dev",
            selected="pkg.mod:symbol_b",
            fingerprint="cid-ev-dev",
        ),
        partition=CorpusPartition.DEVELOPMENT,
        example_id="ex.dev.accepted",
    )
    _admit(
        builder,
        _bundle(
            hole_id="hole.select-symbol.heldout",
            selected="pkg.mod:not-allowed",
            fingerprint="cid-ev-heldout",
            accepted=False,
        ),
        partition=CorpusPartition.HELD_OUT,
        example_id="ex.heldout.rejected",
    )
    _admit(
        builder,
        _bundle(
            hole_id="hole.select-symbol.negative",
            selected="pkg.mod:not-allowed",
            fingerprint="cid-ev-neg",
            accepted=False,
        ),
        partition=CorpusPartition.NEGATIVE,
        example_id="ex.negative.rejected",
    )

    corpus = builder.build()
    decoded = DistillationCorpus.from_dict(corpus.to_dict())
    assert decoded == corpus
    parsed = parse_procedure_artifact(corpus.to_dict())
    assert isinstance(parsed, DistillationCorpus)
    assert parsed.disjoint is True
    assert set(parsed.partition_example_cids) == set(REQUIRED_PARTITIONS)
    partition_sets = [set(parsed.partition_example_cids[name]) for name in REQUIRED_PARTITIONS]
    for index, left in enumerate(partition_sets):
        for right in partition_sets[index + 1 :]:
            assert left.isdisjoint(right)

    for row in parsed.rows:
        record = row.to_record()
        for field_name in REQUIRED_PROVENANCE_FIELDS:
            assert record[field_name]
        assert row.validation_cid
        assert row.proof_cid == "proof.hole-outcome"
        assert row.outcome in DistillationLabel
        assert row.counterexample_cids == ("cex.boundary-near-match", "cex.negative-import")
        assert "prompt" not in record
        assert "output" not in record
        assert "input_payload" not in record

    assert parsed.can_authorize is False
    assert parsed.state is ArtifactState.CANDIDATE
    with pytest.raises(FrozenInstanceError):
        parsed.disjoint = False  # type: ignore[misc]


def test_corpus_round_trip_keeps_large_bodies_as_content_references() -> None:
    builder = _builder()
    example = _admit(
        builder,
        _bundle(hole_id="hole.select-symbol.train", selected="pkg.mod:symbol_a", fingerprint="cid-ev-train"),
        partition=CorpusPartition.TRAINING,
        example_id="ex.train.accepted",
    )
    decoded = DistillationExample.from_dict(example.to_dict())
    assert decoded == example
    parsed = parse_procedure_artifact(example.to_dict())
    assert isinstance(parsed, DistillationExample)
    kinds = {item.kind for item in parsed.content_references}
    assert DistillationReferenceKind.CANDIDATE in kinds
    assert DistillationReferenceKind.VALIDATION in kinds
    assert DistillationReferenceKind.PROOF in kinds
    assert DistillationReferenceKind.COUNTEREXAMPLE in kinds
    assert all(item.content_id for item in parsed.content_references)
    assert "docstring" not in parsed.typed_features
    assert parsed.typed_features["selected"] == "pkg.mod:symbol_a"
    payload = parsed.to_dict()
    assert "prompt_body" not in payload
    assert "source_body" not in payload
    assert "model_transcript" not in payload


def test_corpus_rejects_prompt_bodies() -> None:
    builder = _builder()
    bundle = _bundle(
        hole_id="hole.select-symbol.prompt",
        selected="pkg.mod:symbol_a",
        fingerprint="cid-ev-prompt",
    )
    with pytest.raises(DistillationAdmissionError, match="prompt body") as caught:
        _admit(
            builder,
            bundle,
            partition=CorpusPartition.TRAINING,
            typed_features={"prompt_body": "fill-the-hole"},
        )
    assert caught.value.reason_code is DistillationReason.PROMPT_REJECTED
    assert builder.examples == ()


def test_corpus_rejects_stale_examples() -> None:
    builder = _builder(current_tree_id="tree-old")
    with pytest.raises(DistillationAdmissionError, match="stale") as caught:
        _admit(
            builder,
            _bundle(
                hole_id="hole.select-symbol.stale",
                selected="pkg.mod:symbol_a",
                fingerprint="cid-ev-stale",
            ),
            partition=CorpusPartition.TRAINING,
        )
    assert caught.value.reason_code is DistillationReason.STALE_EXAMPLE


def test_corpus_rejects_unverified_examples() -> None:
    builder = _builder()
    request, candidate, resolution, _receipt_ignored = _bundle(
        hole_id="hole.select-symbol.unverified",
        selected="pkg.mod:not-allowed",
        fingerprint="cid-ev-unverified",
        accepted=False,
    )
    forged = HoleValidationReceipt(
        bindings=request.bindings,
        request_cid=request.content_id,
        candidate_cid=candidate.content_id,
        hole_id=request.hole_id,
        accepted=True,
        reason_code=HoleResolutionReason.CANDIDATE_PROPOSED,
        observation_ids=("observation.tests", "proof.hole-outcome"),
    )
    with pytest.raises(DistillationAdmissionError, match="independently reproducible") as caught:
        builder.admit(
            request=request,
            candidate=candidate,
            resolution=resolution,
            receipt=forged,
            partition=CorpusPartition.TRAINING,
            proof_cid="proof.hole-outcome",
            counterexample_cids=("cex.boundary-near-match",),
            family_id="family.import-purity",
            language="python",
            framework="pytest",
        )
    assert caught.value.reason_code is DistillationReason.UNVERIFIED_EXAMPLE


def test_corpus_rejects_mislabeled_examples() -> None:
    builder = _builder()
    with pytest.raises(DistillationAdmissionError, match="does not match") as caught:
        _admit(
            builder,
            _bundle(
                hole_id="hole.select-symbol.mislabel",
                selected="pkg.mod:symbol_a",
                fingerprint="cid-ev-mislabel",
            ),
            partition=CorpusPartition.TRAINING,
            label=DistillationLabel.REJECTED,
        )
    assert caught.value.reason_code is DistillationReason.MISLABELED_EXAMPLE


def test_corpus_rejects_private_examples() -> None:
    builder = _builder()
    with pytest.raises(DistillationAdmissionError, match="private") as caught:
        _admit(
            builder,
            _bundle(
                hole_id="hole.select-symbol.private",
                selected="pkg.mod:symbol_a",
                fingerprint="cid-ev-private",
            ),
            partition=CorpusPartition.TRAINING,
            typed_features={"chain_of_thought": "hidden-reasoning"},
        )
    assert caught.value.reason_code is DistillationReason.PRIVATE_EXAMPLE


def test_corpus_detects_partition_leakage() -> None:
    builder = _builder()
    bundle = _bundle(
        hole_id="hole.select-symbol.leak",
        selected="pkg.mod:symbol_a",
        fingerprint="cid-ev-leak",
    )
    _admit(builder, bundle, partition=CorpusPartition.TRAINING, example_id="ex.train.leak")
    with pytest.raises(DistillationAdmissionError, match="overlaps another partition") as caught:
        _admit(
            builder,
            bundle,
            partition=CorpusPartition.HELD_OUT,
            example_id="ex.heldout.leak",
        )
    assert caught.value.reason_code is DistillationReason.PARTITION_LEAKAGE
    assert len(builder.examples) == 1


def test_corpus_rejects_unbound_proof_and_missing_counterexamples() -> None:
    builder = _builder()
    bundle = _bundle(
        hole_id="hole.select-symbol.proof",
        selected="pkg.mod:symbol_a",
        fingerprint="cid-ev-proof",
    )
    with pytest.raises(DistillationAdmissionError, match="proof identity") as caught:
        _admit(
            builder,
            bundle,
            partition=CorpusPartition.TRAINING,
            proof_cid="proof.unbound",
        )
    assert caught.value.reason_code is DistillationReason.MISSING_PROOF

    with pytest.raises(DistillationAdmissionError, match="counterexample"):
        _admit(
            builder,
            bundle,
            partition=CorpusPartition.TRAINING,
            counterexample_cids=(),
        )


def test_corpus_cannot_authorize_or_leave_candidate_tier() -> None:
    builder = _builder()
    example = _admit(
        builder,
        _bundle(hole_id="hole.select-symbol.train", selected="pkg.mod:symbol_a", fingerprint="cid-ev-train"),
        partition=CorpusPartition.TRAINING,
        example_id="ex.train.accepted",
    )
    payload = example.to_dict()
    payload["can_authorize"] = True
    with pytest.raises(DistillationAdmissionError, match="cannot authorize"):
        DistillationExample.from_dict(payload)
    payload = example.to_dict()
    payload["state"] = ArtifactState.PROMOTED.value
    with pytest.raises(DistillationAdmissionError, match="candidate-tier"):
        DistillationExample.from_dict(payload)
    corpus = builder.build()
    assert corpus.can_grant_authority is False
    assert corpus.can_promote is False
    assert builder.evaluation is not None
    assert builder.evaluation.can_authorize is False
    assert builder.evaluation.can_grant_authority is False
