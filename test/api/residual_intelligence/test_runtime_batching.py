from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PrivacyClass,
    ResidualIntelligenceError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.packaging import (
    ExpertRuntimeManifest,
    PackagedExpert,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.runtime import (
    REASON_DUPLICATE_WEIGHTS,
    REASON_SIMULATION,
    ExpertBatch,
    ExpertResourceLease,
    ResidualInferenceRuntime,
)


def package() -> PackagedExpert:
    return PackagedExpert(
        manifest=ExpertRuntimeManifest(
            architecture_id="arch:1",
            weights_uri="cas://weights/abc",
            tokenizer_id="tok:1",
            quantization_id="q:none",
            runtime_id="rt:1",
            operators_id="ops:1",
            hardware_id="cpu-standard",
            environment_id="env:1",
            evaluation_id="eval:1",
            admission_id="admission:1",
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        )
    )


def test_batch_lease_and_deterministic_unload() -> None:
    runtime = ResidualInferenceRuntime()
    receipt = runtime.submit(
        ExpertBatch(package=package(), request_ids=("req:1", "req:2")),
        ExpertResourceLease(lease_id="lease:1", hardware_id="cpu-standard"),
    )
    assert receipt.unloaded is True
    assert receipt.to_dict()["candidate_only"] is True
    with pytest.raises(ResidualIntelligenceError, match=REASON_SIMULATION):
        ExpertBatch(package=package(), request_ids=("req:1",), simulated=True)
    loaded = ResidualInferenceRuntime(loaded_weight_uris=("cas://weights/abc",))
    with pytest.raises(ResidualIntelligenceError, match=REASON_DUPLICATE_WEIGHTS):
        loaded.submit(
            ExpertBatch(package=package(), request_ids=("req:3",)),
            ExpertResourceLease(lease_id="lease:2", hardware_id="cpu-standard"),
        )
