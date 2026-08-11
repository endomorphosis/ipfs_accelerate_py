"""Adversarial conformance tests for the pinned datasets embedding provider (LPR-031)."""

from __future__ import annotations

import math

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_embedding_provider import (
    CANARY_TEXTS,
    ConstantVectorShimBackend,
    DatasetsEmbeddingCapability,
    DeterministicLocalEmbeddingBackend,
    EmbeddingCanaryDisposition,
    EmbeddingCanaryReason,
    EmbeddingCanaryReceipt,
    EmbeddingLaneStatus,
    EmbeddingProviderBindingError,
    EmbeddingProviderError,
    EmbeddingProviderStatus,
    EmbeddingRequest,
    EmbeddingResult,
    IpfsDatasetsEmbeddingProvider,
    MissingDependencySuccessShim,
    PinnedEmbeddingPolicy,
    UnpinnedRemoteEmbeddingBackend,
    create_ipfs_datasets_embedding_provider,
    create_pinned_embedding_policy,
    inspect_datasets_embedding_capability,
)


def _policy(**extra: object) -> PinnedEmbeddingPolicy:
    values: dict[str, object] = {
        "provider_id": "ipfs_datasets_py.embeddings",
        "model_artifact_id": "model:deterministic-local",
        "model_revision": "1",
        "dimensions": 8,
        "chunker_id": "chunker:symbol-span@1",
        "normalizer": "l2",
        "distance": "cosine",
        "corpus_root_id": "corpus:fixture",
        "index_root_id": "index:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "config_id": "config:fixture",
        "allow_remote": False,
    }
    values.update(extra)
    return PinnedEmbeddingPolicy(**values)  # type: ignore[arg-type]


def test_policy_pins_provider_model_revision_dimension_chunker_normalizer_distance_and_roots() -> None:
    policy = _policy()
    assert policy.provider_id
    assert policy.model_artifact_id
    assert policy.model_revision
    assert policy.dimensions == 8
    assert policy.chunker_id
    assert policy.normalizer == "l2"
    assert policy.distance == "cosine"
    assert policy.corpus_root_id
    assert policy.index_root_id
    assert policy.policy_id
    assert PinnedEmbeddingPolicy.from_dict(policy.to_record()).content_id == policy.content_id


def test_unpinned_remote_policy_is_rejected() -> None:
    with pytest.raises(EmbeddingProviderBindingError):
        _policy(allow_remote=True, remote_endpoint_id="")
    with pytest.raises(EmbeddingProviderBindingError):
        _policy(allow_remote=False, remote_endpoint_id="https://example.invalid/embed")


def test_cosine_requires_l2_normalizer() -> None:
    with pytest.raises(EmbeddingProviderError):
        _policy(normalizer="none", distance="cosine")


def test_capability_inspection_does_not_import_optional_package() -> None:
    cap = inspect_datasets_embedding_capability()
    assert isinstance(cap, DatasetsEmbeddingCapability)
    assert cap.semantic_authority is False
    assert cap.authoritative is False
    assert cap.vector_lane is EmbeddingLaneStatus.NOT_PROBED
    assert DatasetsEmbeddingCapability.from_dict(cap.to_record()).content_id == cap.content_id


def test_deterministic_local_backend_passes_canary_and_embeds() -> None:
    policy = _policy()
    backend = DeterministicLocalEmbeddingBackend(policy)
    provider = IpfsDatasetsEmbeddingProvider(policy, backend=backend, auto_canary=True)
    assert provider.vector_lane_enabled is True
    assert provider.vector_lane is EmbeddingLaneStatus.ENABLED
    assert provider.canary_receipt is not None
    assert provider.canary_receipt.disposition is EmbeddingCanaryDisposition.PASSED
    assert EmbeddingCanaryReason.OK.value in provider.canary_receipt.reasons

    result = provider.embed(["alpha", "beta"])
    assert result.status is EmbeddingProviderStatus.COMPLETED
    assert result.semantic_authority is False
    assert result.dimensions == policy.dimensions
    assert len(result.vectors) == 2
    assert result.vectors[0] != result.vectors[1]
    assert all(math.isfinite(x) for vector in result.vectors for x in vector)
    assert EmbeddingResult.from_dict(result.to_record()).content_id == result.content_id

    capability = provider.capability()
    assert capability.available is True
    assert capability.vector_lane_enabled is True
    assert capability.policy_id == policy.policy_id


def test_canary_rejects_constant_vector_shim_and_disables_only_vector_lane() -> None:
    policy = _policy()
    provider = IpfsDatasetsEmbeddingProvider(
        policy,
        backend=ConstantVectorShimBackend(policy, fill=0.0),
        auto_canary=True,
    )
    assert provider.vector_lane_enabled is False
    assert provider.vector_lane is EmbeddingLaneStatus.CANARY_FAILED
    assert provider.canary_receipt is not None
    assert provider.canary_receipt.disposition is EmbeddingCanaryDisposition.FAILED
    assert EmbeddingCanaryReason.CONSTANT_VECTOR.value in provider.canary_receipt.reasons

    result = provider.embed(["anything"])
    assert result.status is EmbeddingProviderStatus.CANARY_FAILED
    assert result.vectors == ()
    assert result.vector_lane is EmbeddingLaneStatus.CANARY_FAILED
    # Exact analysis is not blocked — only the optional lane is disabled.
    assert provider.capability().vector_lane is EmbeddingLaneStatus.CANARY_FAILED


def test_canary_rejects_missing_dependency_success_shim() -> None:
    policy = _policy()
    provider = IpfsDatasetsEmbeddingProvider(
        policy,
        backend=MissingDependencySuccessShim(),
        auto_canary=True,
    )
    assert provider.vector_lane is EmbeddingLaneStatus.CANARY_FAILED
    assert provider.canary_receipt is not None
    assert (
        EmbeddingCanaryReason.MISSING_DEPENDENCY_SHIM.value
        in provider.canary_receipt.reasons
    )


def test_canary_rejects_non_finite_vectors() -> None:
    policy = _policy(dimensions=4)

    class NaNBackend:
        kind = "nan_backend"

        def embed(self, texts):
            return [[float("nan")] * 4 for _ in texts]

    provider = IpfsDatasetsEmbeddingProvider(
        policy, backend=NaNBackend(), auto_canary=True
    )
    assert provider.vector_lane is EmbeddingLaneStatus.CANARY_FAILED
    assert provider.canary_receipt is not None
    assert EmbeddingCanaryReason.NON_FINITE.value in provider.canary_receipt.reasons


def test_canary_rejects_dimension_drift() -> None:
    policy = _policy(dimensions=8)

    class WrongDimBackend:
        kind = "wrong_dim"

        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    provider = IpfsDatasetsEmbeddingProvider(
        policy, backend=WrongDimBackend(), auto_canary=True
    )
    assert provider.vector_lane is EmbeddingLaneStatus.CANARY_FAILED
    assert provider.canary_receipt is not None
    assert EmbeddingCanaryReason.DIMENSION_DRIFT.value in provider.canary_receipt.reasons


def test_canary_rejects_config_drift() -> None:
    policy = _policy(dimensions=8)

    class DriftBackend(DeterministicLocalEmbeddingBackend):
        kind = "config_drift"

        def __init__(self, policy: PinnedEmbeddingPolicy) -> None:
            super().__init__(policy)
            self.dimensions = policy.dimensions + 1  # drifted pin

    provider = IpfsDatasetsEmbeddingProvider(
        policy, backend=DriftBackend(policy), auto_canary=True
    )
    assert provider.vector_lane is EmbeddingLaneStatus.CANARY_FAILED
    assert provider.canary_receipt is not None
    assert EmbeddingCanaryReason.CONFIG_DRIFT.value in provider.canary_receipt.reasons


def test_never_uses_unpinned_remote_embedding() -> None:
    policy = _policy()
    provider = IpfsDatasetsEmbeddingProvider(
        policy,
        backend=UnpinnedRemoteEmbeddingBackend(),
        auto_canary=True,
    )
    assert provider.vector_lane is EmbeddingLaneStatus.UNPINNED_REJECTED
    assert provider.canary_receipt is not None
    assert EmbeddingCanaryReason.UNPINNED_REMOTE.value in provider.canary_receipt.reasons
    result = provider.embed(["x"])
    assert result.status is EmbeddingProviderStatus.REJECTED
    assert result.vectors == ()


def test_request_binding_rejects_corpus_or_policy_drift() -> None:
    policy = _policy()
    provider = create_ipfs_datasets_embedding_provider(
        policy,
        backend=DeterministicLocalEmbeddingBackend(policy),
    )
    with pytest.raises(EmbeddingProviderBindingError):
        provider.embed(
            EmbeddingRequest(
                policy_id=policy.policy_id,
                texts=("hello",),
                corpus_root_id="corpus:other",
                index_root_id=policy.index_root_id,
            )
        )
    with pytest.raises(EmbeddingProviderBindingError):
        provider.embed(
            EmbeddingRequest(
                policy_id="policy:forged",
                texts=("hello",),
                corpus_root_id=policy.corpus_root_id,
                index_root_id=policy.index_root_id,
            )
        )


def test_disable_vector_lane_is_operator_visible() -> None:
    policy = _policy()
    provider = create_ipfs_datasets_embedding_provider(
        policy,
        backend=DeterministicLocalEmbeddingBackend(policy),
    )
    assert provider.vector_lane_enabled is True
    provider.disable_vector_lane(reason="operator_disabled")
    assert provider.vector_lane is EmbeddingLaneStatus.DISABLED
    result = provider.embed(["x"])
    assert result.status is EmbeddingProviderStatus.DISABLED
    assert result.vectors == ()


def test_canary_receipt_and_result_round_trip() -> None:
    policy = create_pinned_embedding_policy(dimensions=4)
    provider = IpfsDatasetsEmbeddingProvider(
        policy,
        backend=DeterministicLocalEmbeddingBackend(policy),
    )
    receipt = provider.canary_receipt
    assert receipt is not None
    assert EmbeddingCanaryReceipt.from_dict(receipt.to_record()).content_id == receipt.content_id
    assert len(CANARY_TEXTS) >= 2
