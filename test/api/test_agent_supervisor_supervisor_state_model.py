from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.prover_matrix_registry import (
    CommandRequest,
    CommandResult,
)
from ipfs_accelerate_py.agent_supervisor.supervisor_state_model import (
    LIVENESS_PROPERTIES,
    MODEL_CHECK_RECEIPT_SCHEMA,
    SAFETY_PROPERTIES,
    SUPERVISOR_TLA_MODEL_SCHEMA,
    SUPERVISOR_TRANSITION_SCHEMA,
    CounterexampleTrace,
    ModelCheckBounds,
    ModelCheckStatus,
    ModelCheckerExecutionConfig,
    ModelCheckerTool,
    ModelValidationError,
    SupervisorStateModelChecker,
    SupervisorStateModelGenerator,
    SupervisorTransitionSchema,
    TransitionRule,
    generate_supervisor_state_model,
    parse_counterexample_trace,
)


def _schema(
    *,
    transitions: tuple[TransitionRule, ...] | None = None,
) -> SupervisorTransitionSchema:
    return SupervisorTransitionSchema(
        tasks=("task-b", "task-a"),
        agents=("agent-2", "agent-1"),
        states=(
            "running",
            "pending",
            "accepted",
            "evidence_ready",
            "retrying",
            "completed",
            "failed",
            "cancelled",
        ),
        initial_state="pending",
        terminal_states=("failed", "completed", "cancelled"),
        dependency_satisfied_states=("completed",),
        transitions=transitions
        or (
            TransitionRule(
                "Merge",
                ("evidence_ready",),
                "completed",
                requires_dependencies=True,
                requires_evidence=True,
                requires_owner=True,
                capacity_delta=-1,
                records_merge=True,
                clears_claim=True,
            ),
            TransitionRule(
                "Accept",
                ("pending",),
                "accepted",
                accepts_claim=True,
                increments_fence=True,
            ),
            TransitionRule(
                "Start",
                ("retrying", "accepted"),
                "running",
                requires_dependencies=True,
                requires_owner=True,
                capacity_delta=1,
                marks_progress=True,
            ),
            TransitionRule(
                "Evidence",
                ("running",),
                "evidence_ready",
                requires_owner=True,
                produces_evidence=True,
            ),
            TransitionRule(
                "Retry",
                ("running",),
                "retrying",
                requires_owner=True,
                capacity_delta=-1,
                increments_retry=True,
            ),
            TransitionRule(
                "Cancel",
                (
                    "retrying",
                    "pending",
                    "running",
                    "evidence_ready",
                    "accepted",
                ),
                "cancelled",
                capacity_delta=-1,
                clears_claim=True,
            ),
            TransitionRule(
                "Fail",
                ("evidence_ready", "running"),
                "failed",
                requires_owner=True,
                capacity_delta=-1,
                clears_claim=True,
            ),
        ),
        dependencies={"task-b": ("task-a",), "task-a": ()},
        required_evidence={
            "task-b": ("review", "test"),
            "task-a": ("test",),
        },
        capacity=1,
        source_identity="bafk-plan",
        metadata={"profile": "test"},
    )


def _model():
    return generate_supervisor_state_model(
        _schema(),
        bounds=ModelCheckBounds(
            max_steps=24,
            max_retries=1,
            max_fence=4,
            max_tasks=2,
            max_agents=2,
            max_states=8,
            max_transitions=7,
            max_evidence_ids=2,
        ),
        module_name="Supervisor_Test",
    )


def test_generator_is_deterministic_canonical_and_schema_driven() -> None:
    schema = _schema()
    serialized = schema.to_dict()
    shuffled = {
        **serialized,
        "tasks": list(reversed(serialized["tasks"])),
        "agents": list(reversed(serialized["agents"])),
        "states": list(reversed(serialized["states"])),
        "terminal_states": list(reversed(serialized["terminal_states"])),
        "transitions": list(reversed(serialized["transitions"])),
        "dependencies": {
            key: serialized["dependencies"][key]
            for key in reversed(serialized["dependencies"])
        },
        "required_evidence": {
            key: serialized["required_evidence"][key]
            for key in reversed(serialized["required_evidence"])
        },
    }
    shuffled.pop("schema_identity")
    bounds = ModelCheckBounds(
        max_steps=24,
        max_retries=1,
        max_fence=4,
        max_tasks=2,
        max_agents=2,
        max_states=8,
        max_transitions=7,
        max_evidence_ids=2,
    )

    first = SupervisorStateModelGenerator().generate(
        schema, bounds=bounds, module_name="Supervisor_Test"
    )
    second = SupervisorStateModelGenerator().generate(
        shuffled, bounds=bounds, module_name="Supervisor_Test"
    )

    assert first.model_text == second.model_text
    assert first.tlc_config_text == second.tlc_config_text
    assert first.apalache_config_text == second.apalache_config_text
    assert first.model_identity == second.model_identity
    assert first.artifact_identity == second.artifact_identity
    assert first.schema == SUPERVISOR_TLA_MODEL_SCHEMA
    assert schema.schema == SUPERVISOR_TRANSITION_SCHEMA
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()

    # The transition actions and state names come from the input schema.
    assert "Accept(t, a, f) ==" in first.model_text
    assert 'state[t] \\in {"accepted", "retrying"}' in first.model_text
    assert 'Tasks == {"task-a", "task-b"}' in first.model_text
    assert "AcceptClaim(t, a, f)" not in first.model_text


def test_model_contains_all_required_safety_and_bounded_liveness_properties() -> None:
    model = _model()

    for property_name in SAFETY_PROPERTIES:
        assert f"{property_name} ==" in model.model_text
        assert f"INVARIANT {property_name}" in model.tlc_config_text
    for property_name in LIVENESS_PROPERTIES:
        assert f"{property_name} ==" in model.model_text
        assert f"PROPERTY {property_name}" in model.tlc_config_text

    assert "Cardinality(claims[t]) <= 1" in model.model_text
    assert "lastMutationFence[t] = fence[t]" in model.model_text
    assert "[lastMutationFence EXCEPT ![t] = f]" in model.model_text
    assert "[lastMutationFence EXCEPT ![t] = f + 1]" in model.model_text
    assert "state[d] \\in DependencySatisfiedStates" in model.model_text
    assert "mergeCount[t] <= 1" in model.model_text
    assert "Cardinality(active) <= MaxCapacity" in model.model_text
    assert "RequiredEvidence[t] \\subseteq evidence[t]" in model.model_text
    assert "step = MaxSteps" in model.model_text
    assert model.tlc_config_text.startswith("SPECIFICATION Spec\n")
    assert model.apalache_config_text == (
        "INIT Init\nNEXT Next\nINVARIANT Safety\n"
    )
    assert model.to_dict()["bounded"] is True
    assert model.to_dict()["unbounded_proof"] is False


def test_schema_and_finite_bound_validation_fail_closed() -> None:
    with pytest.raises(ModelValidationError, match="acyclic"):
        SupervisorTransitionSchema(
            tasks=("a", "b"),
            agents=("worker",),
            states=("pending", "completed"),
            initial_state="pending",
            terminal_states=("completed",),
            dependency_satisfied_states=("completed",),
            transitions=(TransitionRule("Done", ("pending",), "completed"),),
            dependencies={"a": ("b",), "b": ("a",)},
        )

    with pytest.raises(ModelValidationError, match="leaves a terminal"):
        SupervisorTransitionSchema(
            tasks=("a",),
            agents=("worker",),
            states=("pending", "completed"),
            initial_state="pending",
            terminal_states=("completed",),
            dependency_satisfied_states=("completed",),
            transitions=(TransitionRule("Retry", ("completed",), "pending"),),
        )

    with pytest.raises(ModelValidationError, match="exceeds finite bounds"):
        SupervisorStateModelGenerator().generate(
            _schema(),
            bounds=ModelCheckBounds(max_tasks=1),
        )

    payload = _schema().to_dict()
    payload["schema_identity"] = "sha256:not-the-schema"
    with pytest.raises(ModelValidationError, match="identity does not match"):
        SupervisorTransitionSchema.from_dict(payload)

    malformed_rule = _schema().to_dict()
    malformed_rule.pop("schema_identity")
    malformed_rule["transitions"][0]["requires_evidence"] = "false"
    with pytest.raises(ModelValidationError, match="requires_evidence must be boolean"):
        SupervisorTransitionSchema.from_dict(malformed_rule)


class FakeRunner:
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.requests: list[CommandRequest] = []

    def __call__(self, request: CommandRequest) -> CommandResult:
        self.requests.append(request)
        if "--version" in request.command or "version" in request.command:
            return CommandResult(0, stdout="ModelChecker 1.2.3\n")
        return self.result


def test_tlc_receipt_records_exact_bounded_experiment_and_never_claims_proof() -> None:
    runner = FakeRunner(
        CommandResult(
            0,
            stdout="Model checking completed. No error has been found.\n",
        )
    )
    checker = SupervisorStateModelChecker(
        command_runner=runner,
        which=lambda name: "/tools/tlc" if name == "tlc" else None,
        wall_clock=lambda: 1_700_000_000,
        monotonic=iter((10.0, 10.125)).__next__,
    )

    receipt = checker.check(
        _model(),
        tool=ModelCheckerTool.TLC,
        config=ModelCheckerExecutionConfig(
            timeout_seconds=9,
            version_timeout_seconds=2,
            max_output_bytes=8192,
        ),
    )

    assert receipt.schema == MODEL_CHECK_RECEIPT_SCHEMA
    assert receipt.status is ModelCheckStatus.PASSED
    assert receipt.passed
    assert receipt.bounded is True
    assert receipt.unbounded_proof is False
    assert "proof" not in receipt.assurance_label.lower()
    assert "steps<=24" in receipt.assurance_label
    assert receipt.tool_version == "ModelChecker 1.2.3"
    assert receipt.configuration_text == receipt.model.tlc_config_text
    assert receipt.stdout == "Model checking completed. No error has been found.\n"
    assert receipt.stderr == ""
    assert receipt.command[0] == "/tools/tlc"
    assert receipt.command[1] == "-config"
    assert receipt.version_command == ("/tools/tlc", "--version")
    assert receipt.version_returncode == 0
    assert receipt.version_stdout == "ModelChecker 1.2.3\n"
    assert receipt.version_stderr == ""
    assert receipt.checked_safety_properties == tuple(sorted(SAFETY_PROPERTIES))
    assert receipt.checked_liveness_properties == tuple(
        sorted(LIVENESS_PROPERTIES)
    )
    assert receipt.duration_ms == 125
    payload = receipt.to_dict()
    assert payload["model"]["model_text"] == receipt.model.model_text
    assert payload["configuration_text"] == receipt.model.tlc_config_text
    assert payload["bounds"] == receipt.model.bounds.to_dict()
    assert payload["stdout_sha256"].startswith("sha256:")
    assert payload["receipt_id"] == receipt.receipt_id
    assert json.loads(json.dumps(payload)) == payload
    assert runner.requests[-1].timeout_seconds == 9
    assert runner.requests[-1].max_output_bytes == 8192


def test_apalache_receipt_is_explicitly_bounded_safety_not_liveness() -> None:
    runner = FakeRunner(CommandResult(0, stdout="Checker reports no error\n"))
    receipt = SupervisorStateModelChecker(
        command_runner=runner,
        which=lambda name: "/tools/apalache-mc" if name == "apalache-mc" else None,
        wall_clock=lambda: 1_700_000_000,
        monotonic=iter((2.0, 2.01)).__next__,
    ).check(_model(), tool="apalache")

    assert receipt.status is ModelCheckStatus.PASSED
    assert receipt.checked_safety_properties == tuple(sorted(SAFETY_PROPERTIES))
    assert receipt.checked_liveness_properties == ()
    assert receipt.configuration_text == receipt.model.apalache_config_text
    assert "--length=24" in receipt.command
    assert "--inv=Safety" in receipt.command
    assert "--no-deadlock" in receipt.command
    assert receipt.version_command == ("/tools/apalache-mc", "version")
    assert "proof" not in receipt.reason.lower()
    assert receipt.to_dict()["unbounded_proof"] is False


def test_counterexample_output_and_trace_are_retained_exactly() -> None:
    output = """Error: Invariant UniqueAcceptance is violated.
The behavior up to this point is:
State 1: <Initial predicate>
/\\ state = [task-a |-> "pending"]
/\\ step = 0
State 2: <Accept line 42>
/\\ state = [task-a |-> "accepted"]
/\\ step = 1
"""
    runner = FakeRunner(CommandResult(12, stdout=output, stderr="TLC failed\n"))
    receipt = SupervisorStateModelChecker(
        command_runner=runner,
        which=lambda _name: "/tools/tlc",
        wall_clock=lambda: 1_700_000_000,
        monotonic=iter((3.0, 3.2)).__next__,
    ).check(_model(), tool="tlc")

    assert receipt.status is ModelCheckStatus.COUNTEREXAMPLE
    assert receipt.stdout == output
    assert receipt.stderr == "TLC failed\n"
    assert isinstance(receipt.counterexample, CounterexampleTrace)
    assert receipt.counterexample.raw == output + "\nTLC failed\n"
    assert [state.index for state in receipt.counterexample.states] == [1, 2]
    assert receipt.counterexample.states[0].assignments["step"] == "0"
    assert receipt.counterexample.states[1].assignments["step"] == "1"
    payload = receipt.to_dict()
    assert payload["counterexample"]["states"][1]["label"] == "Accept line 42"
    assert payload["counterexample"]["raw"] == output + "\nTLC failed\n"


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (CommandResult(None, timed_out=True), ModelCheckStatus.TIMED_OUT),
        (CommandResult(None, error="runner crashed"), ModelCheckStatus.ERROR),
        (CommandResult(0, stdout="ambiguous output"), ModelCheckStatus.UNKNOWN),
        (CommandResult(2, stderr="bad invocation"), ModelCheckStatus.ERROR),
    ],
)
def test_timeout_error_unknown_and_nonzero_are_distinct_fail_closed_outcomes(
    result: CommandResult,
    status: ModelCheckStatus,
) -> None:
    receipt = SupervisorStateModelChecker(
        command_runner=FakeRunner(result),
        which=lambda _name: "/tools/tlc",
        monotonic=iter((1.0, 1.1)).__next__,
    ).check(_model())

    assert receipt.status is status
    assert not receipt.passed
    assert receipt.unbounded_proof is False


def test_missing_checker_is_unavailable_and_does_not_claim_properties_checked() -> None:
    receipt = SupervisorStateModelChecker(
        command_runner=lambda _request: pytest.fail("runner must not be called"),
        which=lambda _name: None,
        wall_clock=lambda: 1_700_000_000,
        monotonic=iter((5.0, 5.0)).__next__,
    ).check(_model(), tool="apalache")

    assert receipt.status is ModelCheckStatus.UNAVAILABLE
    assert receipt.command == ()
    assert receipt.version_command == ()
    assert receipt.executable == ""
    assert receipt.checked_safety_properties == ()
    assert receipt.checked_liveness_properties == ()
    assert receipt.model.model_text == _model().model_text
    assert receipt.configuration_text == receipt.model.apalache_config_text


def test_trace_parser_preserves_unstructured_counterexample_text() -> None:
    trace = parse_counterexample_trace("counterexample without state blocks")
    assert trace.states == ()
    assert trace.raw == "counterexample without state blocks"


def test_explicit_executable_and_truncation_are_recorded() -> None:
    runner = FakeRunner(
        CommandResult(
            0,
            stdout="Checker reports no error",
            output_truncated=True,
        )
    )
    receipt = SupervisorStateModelChecker(
        command_runner=runner,
        which=lambda _name: None,
        monotonic=iter((7.0, 7.01)).__next__,
    ).check(
        _model(),
        tool="apalache",
        executable=Path("/opt/apalache"),
    )

    assert receipt.status is ModelCheckStatus.UNKNOWN
    assert receipt.executable == "/opt/apalache"
    assert receipt.output_truncated is True
