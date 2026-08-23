"""Quantization and packaging qualification. Weights stay outside Git."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .contracts import (
    ExpertDisposition,
    PrivacyClass,
    ResidualIntelligenceError,
    required_text,
)

PACKAGE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-packaged-expert@1"
QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-quantization-qualification@1"
)
REASON_WEIGHTS_IN_GIT: Final = "weights_must_stay_outside_git"
REASON_PRIVACY_EXPORT: Final = "privacy_export_denied"


def _ppm_map(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError(f"{name} must be a mapping")
    result: dict[str, int] = {}
    for key, item in value.items():
        if type(item) is not int or isinstance(item, bool) or not 0 <= item <= 1_000_000:
            raise ResidualIntelligenceError(f"{name}.{key} must be ppm")
        result[str(key)] = item
    return result


@dataclass(frozen=True)
class ExpertRuntimeManifest:
    architecture_id: str
    weights_uri: str
    tokenizer_id: str
    quantization_id: str
    runtime_id: str
    operators_id: str
    hardware_id: str
    environment_id: str
    evaluation_id: str
    admission_id: str
    privacy_class: PrivacyClass
    schema: str = "ipfs_accelerate_py/agent-supervisor/residual-expert-runtime-manifest@1"

    def __post_init__(self) -> None:
        for name in (
            "architecture_id",
            "weights_uri",
            "tokenizer_id",
            "quantization_id",
            "runtime_id",
            "operators_id",
            "hardware_id",
            "environment_id",
            "evaluation_id",
            "admission_id",
        ):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        if self.weights_uri.startswith("git:") or "/.git/" in self.weights_uri:
            raise ResidualIntelligenceError(REASON_WEIGHTS_IN_GIT)
        if self.privacy_class is PrivacyClass.PROOF_WITNESS:
            raise ResidualIntelligenceError(REASON_PRIVACY_EXPORT)


@dataclass(frozen=True)
class PackagedExpert:
    manifest: ExpertRuntimeManifest
    git_tracked_weights: bool = False
    schema: str = PACKAGE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ExpertRuntimeManifest):
            raise ResidualIntelligenceError("package requires ExpertRuntimeManifest")
        if self.git_tracked_weights:
            raise ResidualIntelligenceError(REASON_WEIGHTS_IN_GIT)


@dataclass(frozen=True)
class QuantizationQualification:
    full_precision_metrics: Mapping[str, int]
    quantized_metrics: Mapping[str, int]
    hardware_live: bool
    operator_compatible: bool
    warm_latency_ms: int
    cold_latency_ms: int
    regression_ppm: int
    approved_regression_bound_ppm: int
    schema: str = QUALIFICATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "full_precision_metrics",
            _ppm_map(self.full_precision_metrics, "full_precision_metrics"),
        )
        object.__setattr__(
            self, "quantized_metrics", _ppm_map(self.quantized_metrics, "quantized_metrics")
        )
        if type(self.hardware_live) is not bool or type(self.operator_compatible) is not bool:
            raise ResidualIntelligenceError("hardware/operator flags must be boolean")
        for name in ("warm_latency_ms", "cold_latency_ms", "regression_ppm", "approved_regression_bound_ppm"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise ResidualIntelligenceError(f"{name} must be a non-negative integer")

    def disposition(self) -> ExpertDisposition:
        if not self.hardware_live or not self.operator_compatible:
            return ExpertDisposition.CAPABILITY_UNAVAILABLE
        if self.regression_ppm > self.approved_regression_bound_ppm:
            return ExpertDisposition.REJECT_INPUT
        return ExpertDisposition.ACCEPT
