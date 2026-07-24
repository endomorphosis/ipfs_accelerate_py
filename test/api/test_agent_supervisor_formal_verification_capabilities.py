from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor import (
    FORMAL_VERIFICATION_CAPABILITY_REPORT_VERSION,
    FORMAL_VERIFICATION_CAPABILITY_SCHEMA_VERSION,
    CapabilityDimension,
    CapabilityHealth,
    FormalVerificationCapabilityProbe,
    FormalVerificationProbeConfig,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_capabilities import (
    EffectiveContextLimit,
    InferenceCanaryRequest,
    InferenceCanaryResult,
    LeanstralCapability,
    _find_spec_without_import,
    discover_effective_context_limit,
)


EXPECTED_PROVIDERS = {
    "hammer",
    "tdfol",
    "external_provers",
    "lean",
    "leanstral",
    "frame_logic",
    "knowledge_graphs",
    "zkp_backends",
}


class FakeDiscovery:
    def __init__(
        self,
        *,
        modules: set[str] | None = None,
        executables: dict[str, str] | None = None,
    ) -> None:
        self.modules = modules or set()
        self.executables = executables or {}
        self.package_calls: list[str] = []
        self.executable_calls: list[str] = []

    def find_spec(self, module: str) -> object | None:
        self.package_calls.append(module)
        if module in self.modules:
            return SimpleNamespace(origin=f"/python/{module.replace('.', '/')}.py")
        return None

    def which(self, executable: str) -> str | None:
        self.executable_calls.append(executable)
        return self.executables.get(executable)

    @staticmethod
    def version(distribution: str) -> str:
        return f"{distribution}-test-version"


class MutableClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _all_modules() -> set[str]:
    return {
        "ipfs_datasets_py.logic.hammers",
        "ipfs_datasets_py.logic.hammers.learned_selector",
        "ipfs_datasets_py.logic.TDFOL",
        "ipfs_datasets_py.logic.external_provers",
        "ipfs_datasets_py.logic.external_provers.interactive.lean_prover_bridge",
        "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
        "ipfs_accelerate_py.llm_router",
        "ipfs_datasets_py.logic.modal.leanstral",
        "ipfs_datasets_py.logic.modal.codec",
        "ipfs_datasets_py.logic.flogic",
        "ipfs_datasets_py.knowledge_graphs",
        "ipfs_datasets_py.logic.zkp.backends",
        "ipfs_datasets_py.logic.zkp.circuits",
        "spacy",
        "en_core_web_sm",
        "transformers",
        "torch",
        "networkx",
        "z3",
        "cvc5",
        "symai",
        "py_ecc",
    }


def _artifact_directories(tmp_path: Path) -> tuple[Path, Path, Path]:
    leanstral = tmp_path / "leanstral-model"
    leanstral.mkdir()
    (leanstral / "config.json").write_text("{}", encoding="utf-8")
    (leanstral / "model.safetensors").write_bytes(b"weights")

    groth16 = tmp_path / "groth16"
    groth16.mkdir()
    (groth16 / "proving_key.bin").write_bytes(b"pk")
    (groth16 / "verifying_key.bin").write_bytes(b"vk")

    provekit = tmp_path / "provekit"
    provekit.mkdir()
    (provekit / "manifest.json").write_text("{}", encoding="utf-8")
    (provekit / "circuit.pkp").write_bytes(b"prover")
    (provekit / "circuit.pkv").write_bytes(b"verifier")
    return leanstral, groth16, provekit


def test_report_is_versioned_and_covers_every_formal_logic_provider(
    tmp_path: Path,
) -> None:
    leanstral, groth16, provekit = _artifact_directories(tmp_path)
    discovery = FakeDiscovery(
        modules=_all_modules(),
        executables={
            name: f"/usr/bin/{name}"
            for name in (
                "eprover",
                "vampire",
                "z3",
                "cvc5",
                "coqtop",
                "lean",
                "runergo",
                "groth16",
                "provekit-cli",
            )
        },
    )
    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_model_path=str(leanstral),
            groth16_artifacts_path=str(groth16),
            provekit_artifacts_path=str(provekit),
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        distribution_version=discovery.version,
        environ={},
    )

    report = probe.probe()
    payload = report.to_dict()

    assert report.schema_version == FORMAL_VERIFICATION_CAPABILITY_SCHEMA_VERSION
    assert report.report_version == FORMAL_VERIFICATION_CAPABILITY_REPORT_VERSION
    assert set(report.capabilities) == EXPECTED_PROVIDERS
    assert set(payload["providers"]) == EXPECTED_PROVIDERS
    assert payload["bounded"] is True
    assert report.probe_count <= probe.config.max_checks
    assert json.loads(json.dumps(payload)) == payload


def test_each_health_dimension_is_serialized_separately(tmp_path: Path) -> None:
    leanstral, groth16, provekit = _artifact_directories(tmp_path)
    discovery = FakeDiscovery(
        modules=_all_modules(),
        executables={
            "lean": "/opt/lean",
            "eprover": "/opt/eprover",
            "runergo": "/opt/runergo",
            "groth16": "/opt/groth16",
            "provekit-cli": "/opt/provekit-cli",
        },
    )
    report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_model_path=str(leanstral),
            groth16_artifacts_path=str(groth16),
            provekit_artifacts_path=str(provekit),
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    expected_dimensions = {dimension.value for dimension in CapabilityDimension}
    for provider in report.providers:
        assert set(provider.to_dict()["health"]) == expected_dimensions
        assert provider.provider_health

    assert report.provider("hammer").executable_health
    assert report.provider("external_provers").package_health
    assert report.provider("leanstral").model_health
    assert report.provider("zkp_backends").circuit_health
    assert report.provider("knowledge_graphs").optional_dependency_health


def test_missing_optional_packages_models_bindings_and_binaries_are_explicit() -> None:
    discovery = FakeDiscovery()
    report = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    tdfol = report.provider("tdfol")
    assert tdfol.status is CapabilityHealth.UNAVAILABLE
    assert "not importable" in tdfol.reason
    assert "spaCy is not installed" in tdfol.optional_dependency_health[0].reason
    assert "model weights" in tdfol.model_health[0].reason

    external = report.provider("external_provers")
    bindings = {check.name: check for check in external.package_health}
    assert bindings["Z3 Python binding"].status is CapabilityHealth.UNAVAILABLE
    assert "bindings are not installed" in bindings["Z3 Python binding"].reason
    assert bindings["CVC5 Python binding"].status is CapabilityHealth.UNAVAILABLE

    lean = report.provider("lean")
    assert lean.status is CapabilityHealth.UNAVAILABLE
    assert "executable" in lean.executable_health[0].reason

    frame = report.provider("frame_logic")
    assert "simulation mode" in frame.executable_health[0].reason

    leanstral = report.provider("leanstral")
    assert "model weights are not configured" in leanstral.model_health[0].reason


def test_core_provider_degrades_when_only_optional_nlp_health_is_missing() -> None:
    modules = {
        "ipfs_datasets_py.logic.TDFOL",
        "ipfs_datasets_py.knowledge_graphs",
    }
    discovery = FakeDiscovery(modules=modules)
    report = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert report.provider("tdfol").status is CapabilityHealth.DEGRADED
    assert report.provider("knowledge_graphs").status is CapabilityHealth.DEGRADED
    assert "optional" in report.provider("tdfol").reason
    assert "NLP/model" in report.provider("knowledge_graphs").reason


def test_python_discovery_failures_do_not_escape_or_import_providers() -> None:
    calls: list[str] = []

    def broken_find_spec(module: str) -> None:
        calls.append(module)
        raise RuntimeError("broken import hook")

    report = FormalVerificationCapabilityProbe(
        find_spec=broken_find_spec,
        which=lambda _name: None,
        environ={},
    ).probe()

    assert calls
    assert set(report.capabilities) == EXPECTED_PROVIDERS
    assert all(
        provider.status in {CapabilityHealth.DEGRADED, CapabilityHealth.UNAVAILABLE}
        for provider in report.providers
    )
    assert "failed safely" in report.provider("hammer").package_health[0].reason


def test_default_dotted_package_discovery_does_not_execute_parent_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "side_effect_provider"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('package initializer executed')\n",
        encoding="utf-8",
    )
    (package / "backend.py").write_text("CAPABLE = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("side_effect_provider", None)

    spec = _find_spec_without_import("side_effect_provider.backend")

    assert spec is not None
    assert spec.origin and spec.origin.endswith("backend.py")
    assert "side_effect_provider" not in sys.modules


def test_probe_is_cached_until_forced_or_ttl_expires() -> None:
    discovery = FakeDiscovery()
    clock = MutableClock()
    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(cache_ttl_seconds=10),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
        monotonic=clock,
        wall_clock=clock,
    )

    first = probe.probe()
    package_calls = len(discovery.package_calls)
    second = probe.probe()
    assert second is first
    assert len(discovery.package_calls) == package_calls

    forced = probe.probe(force_refresh=True)
    assert forced is not first
    assert len(discovery.package_calls) > package_calls

    calls_after_force = len(discovery.package_calls)
    clock.value += 11
    expired = probe.probe()
    assert expired is not forced
    assert len(discovery.package_calls) > calls_after_force

    probe.clear_cache()
    assert probe.probe() is not expired


def test_probe_check_budget_bounds_discovery_work() -> None:
    discovery = FakeDiscovery(modules=_all_modules())
    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(max_checks=1, timeout_seconds=5),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    )

    report = probe.probe()

    assert report.probe_count == 1
    assert len(discovery.package_calls) + len(discovery.executable_calls) == 1
    limited = [
        check
        for provider in report.providers
        for check in provider.checks
        if check.metadata.get("probe_limited")
    ]
    assert limited
    assert all("limit" in check.reason for check in limited)


def test_probe_wall_clock_budget_bounds_stalled_discovery() -> None:
    def stalled_find_spec(_module: str) -> None:
        time.sleep(0.25)

    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(timeout_seconds=0.03),
        find_spec=stalled_find_spec,
        which=lambda _name: None,
        environ={},
    )

    started = time.monotonic()
    report = probe.probe()
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert report.duration_seconds < 0.15
    assert report.provider("hammer").status is CapabilityHealth.UNAVAILABLE
    assert "timed out safely" in report.provider("hammer").package_health[0].reason


def test_capability_availability_never_becomes_proof_success(
    tmp_path: Path,
) -> None:
    leanstral, groth16, provekit = _artifact_directories(tmp_path)
    discovery = FakeDiscovery(
        modules=_all_modules(),
        executables={
            "lean": "/opt/lean",
            "eprover": "/opt/eprover",
            "runergo": "/opt/runergo",
            "groth16": "/opt/groth16",
            "provekit-cli": "/opt/provekit-cli",
        },
    )
    report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_model_path=str(leanstral),
            groth16_artifacts_path=str(groth16),
            provekit_artifacts_path=str(provekit),
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert any(provider.available for provider in report.providers)
    assert report.proof_attempted is False
    assert report.proof_success is False
    assert all(provider.proof_success is False for provider in report.providers)
    assert all(
        check.proof_attempted is False and check.proof_success is False
        for provider in report.providers
        for check in provider.checks
    )
    assert report.to_dict()["proof_success"] is False
    assert report.to_dict()["providers"]["lean"]["proof_success"] is False


def test_zkp_backend_health_requires_executable_and_circuit_artifacts(
    tmp_path: Path,
) -> None:
    discovery = FakeDiscovery(
        modules={
            "ipfs_datasets_py.logic.zkp.backends",
            "ipfs_datasets_py.logic.zkp.circuits",
        },
        executables={"groth16": "/opt/groth16"},
    )
    report = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()
    zkp = report.provider("zkp_backends")
    provider_checks = {check.name: check for check in zkp.provider_health}

    assert zkp.status is CapabilityHealth.DEGRADED
    assert provider_checks["simulated ZKP backend"].status is CapabilityHealth.DEGRADED
    assert provider_checks["Groth16 backend"].status is CapabilityHealth.UNAVAILABLE
    assert "non-cryptographic" in provider_checks["simulated ZKP backend"].reason
    assert any(
        "not configured" in check.reason for check in zkp.circuit_health
    )

    artifact_dir = tmp_path / "groth16"
    artifact_dir.mkdir()
    (artifact_dir / "proving_key.bin").write_bytes(b"pk")
    (artifact_dir / "verifying_key.bin").write_bytes(b"vk")
    ready = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(groth16_artifacts_path=str(artifact_dir)),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert ready.provider("zkp_backends").status is CapabilityHealth.AVAILABLE


def test_invalid_probe_limits_fail_fast() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        FormalVerificationProbeConfig(timeout_seconds=0)
    with pytest.raises(ValueError, match="max_checks"):
        FormalVerificationProbeConfig(max_checks=0)
    with pytest.raises(ValueError, match="cache_ttl_seconds"):
        FormalVerificationProbeConfig(cache_ttl_seconds=-1)


def test_leanstral_surfaces_are_independent_routing_capabilities() -> None:
    discovery = FakeDiscovery(
        modules={
            "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
            "ipfs_accelerate_py.llm_router",
        },
    )
    leanstral = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe().provider("leanstral")
    tagged = [
        check
        for check in leanstral.provider_health
        if "leanstral_capability" in check.metadata
    ]
    capabilities = {
        check.metadata["leanstral_capability"]: check
        for check in tagged
    }

    assert len(tagged) == len(LeanstralCapability) == 5
    assert set(capabilities) == {item.value for item in LeanstralCapability}
    assert (
        leanstral.leanstral_capability(LeanstralCapability.ROUTE_READINESS)
        is capabilities["route_readiness"]
    )
    assert capabilities["route_readiness"].status is CapabilityHealth.AVAILABLE
    assert (
        capabilities["local_model_execution"].status
        is CapabilityHealth.UNAVAILABLE
    )
    assert (
        capabilities["legal_language_preprocessing"].status
        is CapabilityHealth.UNAVAILABLE
    )
    assert capabilities["codec_availability"].status is CapabilityHealth.UNAVAILABLE
    assert (
        capabilities["kernel_verification"].status
        is CapabilityHealth.UNAVAILABLE
    )
    assert leanstral.status is CapabilityHealth.DEGRADED


def test_configured_local_model_does_not_depend_on_kernel_or_codec(
    tmp_path: Path,
) -> None:
    model, _groth16, _provekit = _artifact_directories(tmp_path)
    discovery = FakeDiscovery(
        modules={
            "transformers",
            "torch",
        },
    )
    leanstral = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(leanstral_model_path=str(model)),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe().provider("leanstral")
    capabilities = {
        check.metadata["leanstral_capability"]: check
        for check in leanstral.provider_health
        if "leanstral_capability" in check.metadata
    }

    assert (
        capabilities["local_model_execution"].status
        is CapabilityHealth.AVAILABLE
    )
    assert capabilities["codec_availability"].status is CapabilityHealth.UNAVAILABLE
    assert (
        capabilities["kernel_verification"].status
        is CapabilityHealth.UNAVAILABLE
    )


def test_effective_context_limit_uses_smallest_authority_and_reserves() -> None:
    limit = discover_effective_context_limit(
        configured_route={"limits": {"context_window_tokens": 32_768}},
        server={"capabilities": {"max_context_tokens": 24_000}},
        model={"max_position_embeddings": 16_384},
        output_reserve_tokens=1_024,
        safety_margin_tokens=2_048,
    )

    assert isinstance(limit, EffectiveContextLimit)
    assert limit.raw_context_limit_tokens == 16_384
    assert limit.effective_context_limit_tokens == 13_312
    assert limit.effective_context_tokens == 13_312
    assert limit.limiting_source == "model"
    assert limit.available
    assert json.loads(json.dumps(limit.to_dict())) == limit.to_dict()


def test_effective_context_limit_is_unknown_or_exhausted_not_unlimited() -> None:
    unknown = discover_effective_context_limit(
        output_reserve_tokens=100,
        safety_margin_tokens=20,
    )
    exhausted = discover_effective_context_limit(
        configured_route=1_000,
        output_reserve_tokens=800,
        safety_margin_tokens=200,
    )

    assert unknown.raw_context_limit_tokens is None
    assert unknown.effective_context_limit_tokens is None
    assert not unknown.available
    assert exhausted.effective_context_limit_tokens == 0
    assert not exhausted.available


def test_context_limit_is_attached_to_route_health_and_can_fail_route() -> None:
    discovery = FakeDiscovery(
        modules={
            "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
            "ipfs_accelerate_py.llm_router",
        },
    )
    leanstral = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_route={
                "route_id": "local-a",
                "context_window_tokens": 8_192,
            },
            leanstral_server={"context_length": 4_096},
            leanstral_model={"model_id": "leanstral-7b", "n_ctx": 2_048},
            leanstral_output_reserve_tokens=1_536,
            leanstral_safety_margin_tokens=512,
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe().provider("leanstral")
    route = next(
        check
        for check in leanstral.provider_health
        if check.metadata.get("leanstral_capability") == "route_readiness"
    )

    assert route.status is CapabilityHealth.UNAVAILABLE
    context = route.metadata["context_limit"]
    assert context["limiting_source"] == "model"
    assert context["effective_context_limit_tokens"] == 0
    assert context["route_id"] == "local-a"


def test_inference_canary_is_opt_in_and_never_runs_by_default() -> None:
    calls: list[InferenceCanaryRequest] = []
    probe = FormalVerificationCapabilityProbe(
        find_spec=lambda _module: None,
        which=lambda _name: None,
        inference_canary=lambda request: calls.append(request) or "OK",
        environ={},
    )

    report = probe.probe()
    canary = next(
        check
        for check in report.provider("leanstral").provider_health
        if check.name == "bounded inference canary"
    )

    assert calls == []
    assert canary.status is CapabilityHealth.DISABLED
    assert canary.proof_attempted is False
    assert canary.proof_success is False


def test_enabled_inference_canary_receives_only_bounded_health_request() -> None:
    calls: list[InferenceCanaryRequest] = []
    discovery = FakeDiscovery(
        modules={
            "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
            "ipfs_accelerate_py.llm_router",
        },
    )

    def canary(request: InferenceCanaryRequest) -> dict[str, object]:
        calls.append(request)
        return {"ok": True, "output": "OK", "output_tokens": 1}

    report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_route={"route_id": "managed-a", "context_length": 32_000},
            leanstral_model={"model_id": "leanstral-v2", "context_length": 16_000},
            run_leanstral_inference_canary=True,
            leanstral_canary_timeout_seconds=0.1,
            leanstral_canary_input_tokens=6,
            leanstral_canary_output_tokens=2,
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        inference_canary=canary,
        environ={},
    ).probe()
    check = next(
        item
        for item in report.provider("leanstral").provider_health
        if item.name == "bounded inference canary"
    )

    assert len(calls) == 1
    assert calls[0].route == "managed-a"
    assert calls[0].model == "leanstral-v2"
    assert calls[0].max_input_tokens == 6
    assert calls[0].max_output_tokens == 2
    assert calls[0].timeout_seconds == 0.1
    assert check.status is CapabilityHealth.AVAILABLE
    assert check.metadata["proof_attempted"] is False
    assert check.metadata["proof_success"] is False


def test_inference_canary_has_independent_timeout_and_response_bound() -> None:
    def stalled(_request: InferenceCanaryRequest) -> str:
        time.sleep(0.2)
        return "OK"

    timeout_report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            timeout_seconds=1,
            run_leanstral_inference_canary=True,
            leanstral_canary_timeout_seconds=0.02,
        ),
        find_spec=lambda _module: None,
        which=lambda _name: None,
        inference_canary=stalled,
        environ={},
    ).probe()
    timeout_check = next(
        item
        for item in timeout_report.provider("leanstral").provider_health
        if item.name == "bounded inference canary"
    )
    assert timeout_check.status is CapabilityHealth.UNAVAILABLE
    assert "timed out safely" in timeout_check.reason

    response_report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            run_leanstral_inference_canary=True,
            leanstral_canary_max_response_bytes=2,
        ),
        find_spec=lambda _module: None,
        which=lambda _name: None,
        inference_canary=lambda _request: "TOO LARGE",
        environ={},
    ).probe()
    response_check = next(
        item
        for item in response_report.provider("leanstral").provider_health
        if item.name == "bounded inference canary"
    )
    assert response_check.status is CapabilityHealth.UNAVAILABLE
    assert "response-size limit" in response_check.reason


def test_callback_result_cannot_evade_canary_bounds_or_claim_verification() -> None:
    with pytest.raises(ValueError, match="status"):
        InferenceCanaryResult(
            status=CapabilityHealth.VERIFIED,
            reason="not permitted",
            duration_seconds=0,
        )

    report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            run_leanstral_inference_canary=True,
            leanstral_canary_max_response_bytes=4,
        ),
        find_spec=lambda _module: None,
        which=lambda _name: None,
        inference_canary=lambda _request: InferenceCanaryResult(
            status=CapabilityHealth.AVAILABLE,
            reason="callback says healthy",
            duration_seconds=0,
            response_bytes=10,
            metadata={"proof_success": True},
        ),
        environ={},
    ).probe()
    check = next(
        item
        for item in report.provider("leanstral").provider_health
        if item.name == "bounded inference canary"
    )

    assert check.status is CapabilityHealth.UNAVAILABLE
    assert "response-size limit" in check.reason
    assert check.metadata["proof_success"] is False
    assert check.proof_success is False


@pytest.mark.parametrize(
    ("callback_result", "token_name"),
    [
        (
            {
                "ok": True,
                "output": "OK",
                "input_tokens": 7,
                "output_tokens": 1,
            },
            "input_tokens",
        ),
        (
            {
                "ok": True,
                "output": "OK",
                "input_tokens": 6,
                "output_tokens": 3,
            },
            "output_tokens",
        ),
        (
            InferenceCanaryResult(
                status=CapabilityHealth.AVAILABLE,
                reason="callback says healthy",
                duration_seconds=0,
                response_bytes=2,
                metadata={"input_tokens": 7, "output_tokens": 1},
            ),
            "input_tokens",
        ),
        (
            InferenceCanaryResult(
                status=CapabilityHealth.AVAILABLE,
                reason="callback says healthy",
                duration_seconds=0,
                response_bytes=2,
                metadata={"input_tokens": 6, "output_tokens": 3},
            ),
            "output_tokens",
        ),
    ],
)
def test_canary_reported_token_use_cannot_exceed_request_bounds(
    callback_result, token_name
) -> None:
    report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            run_leanstral_inference_canary=True,
            leanstral_canary_input_tokens=6,
            leanstral_canary_output_tokens=2,
        ),
        find_spec=lambda _module: None,
        which=lambda _name: None,
        inference_canary=lambda _request: callback_result,
        environ={},
    ).probe()
    check = next(
        item
        for item in report.provider("leanstral").provider_health
        if item.name == "bounded inference canary"
    )

    assert check.status is CapabilityHealth.UNAVAILABLE
    assert f"exceeded its {token_name} limit" in check.reason
    expected_tokens = (
        callback_result[token_name]
        if isinstance(callback_result, dict)
        else callback_result.metadata[token_name]
    )
    assert check.metadata["metadata"][token_name] == expected_tokens
    assert check.metadata["proof_success"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"configured_route": 0}, "positive integer"),
        ({"server": {"context_length": -1}}, "positive integer"),
        ({"model": "unknown"}, "object or integer"),
        ({"output_reserve_tokens": -1}, "non-negative integer"),
        ({"safety_margin_tokens": True}, "non-negative integer"),
    ],
)
def test_invalid_context_limit_metadata_fails_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        discover_effective_context_limit(**kwargs)
