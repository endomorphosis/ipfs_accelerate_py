from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    PrivacyClass,
    ResidualIntelligenceError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.packaging import (
    REASON_WEIGHTS_IN_GIT,
    ExpertRuntimeManifest,
    PackagedExpert,
    QuantizationQualification,
)


def manifest(**overrides):
    payload = dict(
        architecture_id="arch:1",
        weights_uri="cas://weights/abc",
        tokenizer_id="tok:1",
        quantization_id="q:int8",
        runtime_id="rt:1",
        operators_id="ops:1",
        hardware_id="cpu-standard",
        environment_id="env:1",
        evaluation_id="eval:1",
        admission_id="admission:1",
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
    )
    payload.update(overrides)
    return ExpertRuntimeManifest(**payload)


def test_package_and_quantization_qualification() -> None:
    pkg = PackagedExpert(manifest=manifest())
    ok = QuantizationQualification(
        full_precision_metrics={"precision": 990_000},
        quantized_metrics={"precision": 980_000},
        hardware_live=True,
        operator_compatible=True,
        warm_latency_ms=5,
        cold_latency_ms=40,
        regression_ppm=10_000,
        approved_regression_bound_ppm=20_000,
    )
    assert ok.disposition() is ExpertDisposition.ACCEPT
    unavailable = QuantizationQualification(
        full_precision_metrics={"precision": 990_000},
        quantized_metrics={"precision": 980_000},
        hardware_live=False,
        operator_compatible=True,
        warm_latency_ms=5,
        cold_latency_ms=40,
        regression_ppm=0,
        approved_regression_bound_ppm=20_000,
    )
    assert unavailable.disposition() is ExpertDisposition.CAPABILITY_UNAVAILABLE
    with pytest.raises(ResidualIntelligenceError, match=REASON_WEIGHTS_IN_GIT):
        PackagedExpert(manifest=pkg.manifest, git_tracked_weights=True)
    with pytest.raises(ResidualIntelligenceError, match=REASON_WEIGHTS_IN_GIT):
        manifest(weights_uri="git://repo/weights.bin")
