from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.hyperproperty_verification import (
    AutoHyperAdapter,
    BoundedSelfCompositionChecker,
    ConformanceStatus,
    DEFAULT_HYPERPROPERTY_MODELS,
    EngineCapability,
    EngineCapabilityStatus,
    EngineConformanceReceipt,
    HyperLTLAdapter,
    HyperpropertyEngine,
    HyperpropertyEvidenceKind,
    HyperpropertyKind,
    HyperpropertyModel,
    HyperpropertyValidationError,
    HyperpropertyVerdict,
    HyperpropertyVerifier,
    HyperpropertyVerificationResult,
    Hypertrace,
    MCHyperAdapter,
    ObservationPolicy,
    bounded_self_composition,
    model_for,
    probe_hyperproperty_engines,
)
from ipfs_accelerate_py.agent_supervisor.prover_matrix_registry import (
    CommandRequest,
    CommandResult,
)


EXPECTED_KINDS = {
    HyperpropertyKind.PROMPT_ISOLATION,
    HyperpropertyKind.WORKTREE_ISOLATION,
    HyperpropertyKind.LOG_REDACTION,
    HyperpropertyKind.PROVIDER_ROUTING,
    HyperpropertyKind.ZKP_WITNESS_NONINTERFERENCE,
    HyperpropertyKind.CROSS_TASK_CACHE_SEPARATION,
}


def _model() -> HyperpropertyModel:
    return HyperpropertyModel(
        model_id="test.noninterference",
        version="1",
        kind=HyperpropertyKind.PROMPT_ISOLATION,
        observation_policy=ObservationPolicy(
            policy_id="test.public-output",
            version="1",
            low_input_fields=("request.operation",),
            observation_fields=("output.digest",),
            subject_fields=("task_id",),
            description="Only the reviewed output digest is observable.",
        ),
        hyperltl_formula=(
            "forall pi1. forall pi2. "
            "G(low_equal(pi1, pi2) -> output_equal(pi1, pi2))"
        ),
        description="Private lane state cannot affect public output.",
    )


def _trace(
    trace_id: str,
    *,
    private: str,
    output: str,
    operation: str = "prove",
    task_id: str = "REF-286",
    extra: object | None = None,
) -> Hypertrace:
    observations: dict[str, object] = {"output": {"digest": output}}
    if extra is not None:
        observations["unapproved"] = extra
    return Hypertrace(
        trace_id=trace_id,
        task_id=task_id,
        lane_id="lane-subject",
        worktree_id="tree-subject",
        public_inputs={"request": {"operation": operation}},
        private_inputs={"foreign_lane_secret": private},
        observations=observations,
    )


def _set_path(target: dict[str, object], path: str, value: object) -> None:
    components = path.split(".")
    current = target
    for component in components[:-1]:
        existing = current.get(component)
        child = existing if isinstance(existing, dict) else {}
        current[component] = child
        current = child
    current[components[-1]] = value


def _default_pair(
    model: HyperpropertyModel, *, changed_observation: bool = False
) -> tuple[Hypertrace, Hypertrace]:
    public_inputs: dict[str, object] = {}
    for field in model.observation_policy.low_input_fields:
        _set_path(public_inputs, field, f"low:{field}")
    left_observations: dict[str, object] = {}
    right_observations: dict[str, object] = {}
    for index, field in enumerate(model.observation_policy.observation_fields):
        _set_path(left_observations, field, f"public:{field}")
        _set_path(
            right_observations,
            field,
            "changed-public-value"
            if changed_observation and index == 0
            else f"public:{field}",
        )
    return (
        Hypertrace(
            trace_id="left",
            task_id="task",
            lane_id="lane",
            worktree_id="tree",
            public_inputs=public_inputs,
            private_inputs={"secret": "left-secret"},
            observations=left_observations,
        ),
        Hypertrace(
            trace_id="right",
            task_id="task",
            lane_id="lane",
            worktree_id="tree",
            public_inputs=public_inputs,
            private_inputs={"secret": "right-secret"},
            observations=right_observations,
        ),
    )


def test_reviewed_models_cover_every_required_cross_lane_property() -> None:
    assert len(DEFAULT_HYPERPROPERTY_MODELS) == 6
    assert {model.kind for model in DEFAULT_HYPERPROPERTY_MODELS} == EXPECTED_KINDS
    assert len({model.content_id for model in DEFAULT_HYPERPROPERTY_MODELS}) == 6

    for model in DEFAULT_HYPERPROPERTY_MODELS:
        assert model_for(model.kind) is model
        assert model.hyperltl_formula.lower().count("forall") == 2
        assert model.observation_policy.observation_fields
        assert model.observation_policy_id == model.observation_policy.content_id
        payload = model.to_record()
        assert json.loads(json.dumps(payload)) == payload
        assert HyperpropertyModel.from_dict(payload) == model


def test_exact_observation_policy_is_part_of_model_and_result_identity() -> None:
    model = _model()
    changed_policy = ObservationPolicy(
        policy_id=model.observation_policy.policy_id,
        version=model.observation_policy.version,
        low_input_fields=model.observation_policy.low_input_fields,
        observation_fields=("output.digest", "output.routing"),
        subject_fields=model.observation_policy.subject_fields,
    )
    changed_model = HyperpropertyModel(
        model_id=model.model_id,
        version=model.version,
        kind=model.kind,
        observation_policy=changed_policy,
        hyperltl_formula=model.hyperltl_formula,
        description=model.description,
    )
    assert changed_policy.content_id != model.observation_policy_id
    assert changed_model.content_id != model.content_id

    payload = model.to_record()
    payload["observation_policy_id"] = changed_policy.content_id
    with pytest.raises(HyperpropertyValidationError, match="does not match"):
        HyperpropertyModel.from_dict(payload)


def test_private_trace_values_have_no_serialization_or_repr_channel() -> None:
    secret = "PRIVATE-WITNESS-REF-286"
    trace = _trace("one", private=secret, output="same")
    serialized = json.dumps(trace.to_dict())

    assert secret not in repr(trace)
    assert secret not in serialized
    assert "foreign_lane_secret" not in serialized
    assert trace.to_dict()["private_inputs_redacted"] is True


@pytest.mark.parametrize("model", DEFAULT_HYPERPROPERTY_MODELS)
def test_every_model_has_bounded_passing_self_composition_evidence(
    model: HyperpropertyModel,
) -> None:
    result = bounded_self_composition(model, _default_pair(model))

    assert result.verdict is HyperpropertyVerdict.HOLDS
    assert result.evidence_kind is HyperpropertyEvidenceKind.BOUNDED_SELF_COMPOSITION
    assert result.authoritative is False
    assert result.bounded is True
    assert result.observation_policy_id == model.observation_policy_id
    assert result.counterexample is None


@pytest.mark.parametrize("model", DEFAULT_HYPERPROPERTY_MODELS)
def test_every_model_detects_an_approved_observation_change(
    model: HyperpropertyModel,
) -> None:
    result = bounded_self_composition(
        model, _default_pair(model, changed_observation=True)
    )
    assert result.verdict is HyperpropertyVerdict.VIOLATED
    assert result.counterexample is not None
    assert result.counterexample.observation_policy_id == model.observation_policy_id


def test_counterexample_is_minimal_redacted_value_free_and_policy_bound() -> None:
    left_secret = "lane-A-secret-value"
    right_secret = "lane-B-witness-value"
    left = _trace("one", private=left_secret, output="public-A")
    right = _trace("two", private=right_secret, output="public-B")

    result = bounded_self_composition(_model(), (left, right))
    counterexample = result.counterexample
    assert result.verdict is HyperpropertyVerdict.VIOLATED
    assert counterexample is not None
    assert counterexample.redacted is True
    assert counterexample.minimized is True
    assert len(counterexample.trace_refs) == 2
    assert len(set(counterexample.trace_refs)) == 2
    assert [item.field for item in counterexample.differences] == ["output.digest"]

    payload = counterexample.to_record()
    serialized = json.dumps(payload)
    for forbidden in (
        left_secret,
        right_secret,
        "foreign_lane_secret",
        "public-A",
        "public-B",
    ):
        assert forbidden not in serialized
    assert payload["contains_private_inputs"] is False
    assert counterexample.from_dict(payload) == counterexample
    result_payload = result.to_record()
    assert HyperpropertyVerificationResult.from_dict(result_payload) == result


def test_unapproved_observations_cannot_expand_counterexample_policy() -> None:
    secret = "must-not-enter-counterexample"
    left = _trace("one", private="A", output="same", extra=secret)
    right = _trace("two", private="B", output="same", extra="different")

    result = bounded_self_composition(_model(), (left, right))
    assert result.verdict is HyperpropertyVerdict.HOLDS
    assert secret not in result.to_json()


def test_noncomparable_traces_and_exhausted_bounds_are_inconclusive() -> None:
    no_pair = bounded_self_composition(
        _model(),
        (
            _trace("one", private="A", output="same", operation="prove"),
            _trace("two", private="B", output="same", operation="verify"),
        ),
    )
    assert no_pair.verdict is HyperpropertyVerdict.INCONCLUSIVE

    traces = tuple(
        _trace(str(index), private=str(index), output="same")
        for index in range(4)
    )
    bounded = BoundedSelfCompositionChecker(max_traces=2, max_pairs=1).check(
        _model(), traces
    )
    assert bounded.verdict is HyperpropertyVerdict.INCONCLUSIVE
    assert bounded.authoritative is False
    assert bounded.explored_pairs == 1


class FakeEngineRuntime:
    def __init__(
        self,
        executable: Path | None,
        *,
        conformance: CommandResult | None = None,
    ) -> None:
        self.executable = executable
        self.conformance = conformance or CommandResult(
            returncode=0, stdout="PROPERTY HOLDS; SAT; VERIFIED; TRUE\n"
        )
        self.requests: list[CommandRequest] = []

    def which(self, _name: str) -> str | None:
        return str(self.executable) if self.executable else None

    def run(self, request: CommandRequest) -> CommandResult:
        self.requests.append(request)
        if "--version" in request.command:
            return CommandResult(returncode=0, stdout="HyperEngine 1.2.3\n")
        return self.conformance


def _executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    "adapter_type",
    (HyperLTLAdapter, AutoHyperAdapter, MCHyperAdapter),
)
def test_each_adapter_is_unavailable_when_executable_is_absent(adapter_type) -> None:
    capability = adapter_type(which=lambda _name: None).probe()
    assert capability.status is EngineCapabilityStatus.UNAVAILABLE
    assert capability.available is False
    assert capability.conformance_passed is False
    assert capability.conformance_receipt is None


def test_discovery_and_version_are_still_unavailable_without_fixture(
    tmp_path: Path,
) -> None:
    runtime = FakeEngineRuntime(_executable(tmp_path, "autohyper"))
    capability = AutoHyperAdapter(
        which=runtime.which, command_runner=runtime.run
    ).probe(run_conformance=False)

    assert capability.status is EngineCapabilityStatus.UNAVAILABLE
    assert capability.executable_version == "HyperEngine 1.2.3"
    assert not capability.available
    assert len(runtime.requests) == 1


@pytest.mark.parametrize(
    ("adapter_type", "name"),
    (
        (HyperLTLAdapter, "hyperltl"),
        (AutoHyperAdapter, "autohyper"),
        (MCHyperAdapter, "mchyper"),
    ),
)
def test_only_executed_passing_conformance_promotes_adapter(
    tmp_path: Path, adapter_type, name: str
) -> None:
    runtime = FakeEngineRuntime(_executable(tmp_path, name))
    capability = adapter_type(
        which=runtime.which,
        command_runner=runtime.run,
        monotonic=lambda: 1.0,
    ).probe()

    assert capability.status is EngineCapabilityStatus.CONFORMANT
    assert capability.available is True
    assert capability.conformance_passed is True
    assert capability.conformance_receipt is not None
    assert capability.conformance_receipt.fixture_identity == (
        adapter_type.fixture.content_id
    )
    assert capability.conformance_receipt.executable_identity.startswith("sha256:")
    assert capability.conformance_receipt.command_identity.startswith("b")
    assert capability.to_dict()["discovery_is_conformance"] is False
    assert EngineCapability.from_dict(capability.to_record()) == capability
    assert EngineConformanceReceipt.from_dict(
        capability.conformance_receipt.to_record()
    ) == capability.conformance_receipt
    assert len(runtime.requests) == 2
    fixture_request = runtime.requests[-1]
    assert fixture_request.timeout_seconds > 0
    assert fixture_request.max_output_bytes > 0


def test_failed_or_timed_out_fixture_remains_unavailable(tmp_path: Path) -> None:
    executable = _executable(tmp_path, "mchyper")
    for result in (
        CommandResult(returncode=1, stdout="PROPERTY VIOLATED"),
        CommandResult(returncode=None, timed_out=True),
        CommandResult(returncode=0, stdout="ambiguous output"),
    ):
        runtime = FakeEngineRuntime(executable, conformance=result)
        capability = MCHyperAdapter(
            which=runtime.which, command_runner=runtime.run
        ).probe()
        assert capability.status is EngineCapabilityStatus.UNAVAILABLE
        assert capability.available is False
        assert capability.conformance_receipt is not None
        assert not capability.conformance_receipt.passed


def test_contracts_reject_forged_passing_conformance() -> None:
    with pytest.raises(HyperpropertyValidationError, match="execution success"):
        EngineConformanceReceipt(
            engine=HyperpropertyEngine.HYPERLTL,
            status=ConformanceStatus.PASSED,
            fixture_id="fixture",
            fixture_identity="fixture-id",
            executable_path="/tool",
            executable_identity="sha256:tool",
            executable_version="1",
            command_identity="command-id",
            returncode=1,
            timed_out=False,
            output_sha256="sha256:output",
            marker_matched=True,
            duration_ms=1,
            reason="claimed pass",
        )

    with pytest.raises(HyperpropertyValidationError, match="passing executable"):
        EngineCapability(
            engine=HyperpropertyEngine.HYPERLTL,
            status=EngineCapabilityStatus.CONFORMANT,
            executable_path="/tool",
            executable_version="1",
            reason="claimed available",
        )


def test_probe_reports_all_three_engines_independently_unavailable() -> None:
    adapters = (
        HyperLTLAdapter(which=lambda _name: None),
        AutoHyperAdapter(which=lambda _name: None),
        MCHyperAdapter(which=lambda _name: None),
    )
    capabilities = probe_hyperproperty_engines(adapters)
    assert {item.engine for item in capabilities} == set(HyperpropertyEngine)
    assert all(not item.available for item in capabilities)


def test_verifier_uses_non_authoritative_fallback_when_engines_unavailable() -> None:
    verifier = HyperpropertyVerifier(
        (
            HyperLTLAdapter(which=lambda _name: None),
            AutoHyperAdapter(which=lambda _name: None),
            MCHyperAdapter(which=lambda _name: None),
        )
    )
    capabilities = verifier.capabilities()
    result = verifier.verify(
        _model(),
        (
            _trace("one", private="A", output="same"),
            _trace("two", private="B", output="same"),
        ),
        capabilities=capabilities,
    )
    assert result.verdict is HyperpropertyVerdict.HOLDS
    assert result.evidence_kind is HyperpropertyEvidenceKind.BOUNDED_SELF_COMPOSITION
    assert result.authoritative is False
