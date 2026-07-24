from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import (
    ProtocolVerifier as ExportedProtocolVerifier,
)
from ipfs_accelerate_py.agent_supervisor.protocol_verification import (
    ATTESTATION_PROTOCOL_MODEL,
    CORE_PROTOCOL_MODEL,
    DEFAULT_PROTOCOL_MODELS,
    PROVERIF_CONFORMANCE_FIXTURE,
    TAMARIN_CONFORMANCE_FIXTURE,
    ConformanceStatus,
    ProtocolAttackCounterexample,
    ProtocolConformanceReceipt,
    ProtocolLaneResult,
    ProtocolModel,
    ProtocolProperty,
    ProtocolQueryKind,
    ProtocolSuiteResult,
    ProtocolTool,
    ProtocolToolCapability,
    ProtocolToolchainReceipt,
    ProtocolValidationError,
    ProtocolVerdict,
    ProtocolVerifier,
    ProVerifAdapter,
    TamarinAdapter,
    ToolCapabilityStatus,
    canonicalize_attack_trace,
    default_protocol_models,
    probe_protocol_tools,
    protocol_model_for,
)
from ipfs_accelerate_py.agent_supervisor.prover_matrix_registry import (
    CommandRequest,
    CommandResult,
)


EXPECTED_PROPERTIES = set(ProtocolProperty)


def test_versioned_models_cover_the_complete_protocol_and_document_queries() -> None:
    assert default_protocol_models() == DEFAULT_PROTOCOL_MODELS
    assert len(DEFAULT_PROTOCOL_MODELS) == 2
    assert {
        prop for model in DEFAULT_PROTOCOL_MODELS for prop in model.properties
    } == EXPECTED_PROPERTIES
    assert CORE_PROTOCOL_MODEL.optional_attestation is False
    assert ATTESTATION_PROTOCOL_MODEL.optional_attestation is True

    for model in DEFAULT_PROTOCOL_MODELS:
        assert model.version == "1"
        assert protocol_model_for(model.model_id) is model
        assert set(query.kind for query in model.queries) == set(ProtocolQueryKind)
        assert len({query.content_id for query in model.queries}) == len(model.queries)
        for query in model.queries:
            assert query.abstraction
            assert query.assumptions
            assert query.excluded_behaviors
            assert query.tamarin_name in model.tamarin_source
            assert query.proverif_label in model.proverif_source
        assert ProtocolModel.from_dict(model.to_record()) == model
        assert json.loads(model.to_json()) == model.to_dict()
    assert ExportedProtocolVerifier is ProtocolVerifier


def test_model_identity_binds_sources_query_abstractions_and_version() -> None:
    payload = CORE_PROTOCOL_MODEL.to_record()
    payload["tamarin_source"] += "\n"
    with pytest.raises(ProtocolValidationError, match="source"):
        ProtocolModel.from_dict(payload)

    changed = dict(CORE_PROTOCOL_MODEL.to_dict())
    changed["content_id"] = CORE_PROTOCOL_MODEL.content_id
    changed["version"] = "2"
    with pytest.raises(ProtocolValidationError, match="identity"):
        ProtocolModel.from_dict(changed)


@pytest.mark.parametrize(
    "fixture",
    (TAMARIN_CONFORMANCE_FIXTURE, PROVERIF_CONFORMANCE_FIXTURE),
)
def test_each_end_to_end_fixture_has_all_query_classes_and_known_attack(
    fixture,
) -> None:
    payload = fixture.to_dict()
    assert set(fixture.query_kinds) == set(ProtocolQueryKind)
    assert len(fixture.safe_markers) == 4
    assert fixture.attack_marker
    assert payload["end_to_end"] is True
    assert payload["contains_known_attack"] is True
    assert fixture.source_identity.startswith("sha256:")
    assert type(fixture).from_dict(fixture.to_record()) == fixture


class FakeRuntime:
    def __init__(
        self,
        executable: Path | None,
        *,
        fixture_output: str = "",
        model_output: str = "",
        fixture_result: CommandResult | None = None,
        model_result: CommandResult | None = None,
    ) -> None:
        self.executable = executable
        self.fixture_output = fixture_output
        self.model_output = model_output
        self.fixture_result = fixture_result
        self.model_result = model_result
        self.requests: list[CommandRequest] = []

    def which(self, _name: str) -> str | None:
        return str(self.executable) if self.executable else None

    def run(self, request: CommandRequest) -> CommandResult:
        self.requests.append(request)
        if "--version" in request.command or "-help" in request.command:
            return CommandResult(returncode=0, stdout="SymbolicTool 1.2.3\n")
        is_fixture = any("fixture." in item for item in request.command)
        if is_fixture:
            return self.fixture_result or CommandResult(
                returncode=0, stdout=self.fixture_output
            )
        return self.model_result or CommandResult(
            returncode=0, stdout=self.model_output
        )


def _executable(tmp_path: Path, name: str) -> Path:
    result = tmp_path / name
    result.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    result.chmod(0o755)
    return result


def _fixture_output(tool: ProtocolTool) -> str:
    fixture = (
        TAMARIN_CONFORMANCE_FIXTURE
        if tool is ProtocolTool.TAMARIN
        else PROVERIF_CONFORMANCE_FIXTURE
    )
    return "\n".join((*fixture.safe_markers, fixture.attack_marker)) + "\n"


def _verified_output(model: ProtocolModel, tool: ProtocolTool) -> str:
    if tool is ProtocolTool.TAMARIN:
        return "\n".join(
            f"{query.tamarin_name}: verified" for query in model.queries
        )
    return "\n".join(
        f"RESULT {query.proverif_label} is true." for query in model.queries
    )


@pytest.mark.parametrize("adapter_type", (TamarinAdapter, ProVerifAdapter))
def test_missing_executable_is_unavailable(adapter_type) -> None:
    capability = adapter_type(which=lambda _name: None).probe()
    assert capability.status is ToolCapabilityStatus.UNAVAILABLE
    assert not capability.available
    assert capability.conformance_receipt is None


@pytest.mark.parametrize(
    ("adapter_type", "name"),
    ((TamarinAdapter, "tamarin-prover"), (ProVerifAdapter, "proverif")),
)
def test_executable_presence_or_install_success_cannot_replace_fixture(
    tmp_path: Path, adapter_type, name: str
) -> None:
    runtime = FakeRuntime(_executable(tmp_path, name))
    capability = adapter_type(
        which=runtime.which, command_runner=runtime.run
    ).probe(run_conformance=False)

    assert capability.status is ToolCapabilityStatus.UNAVAILABLE
    assert not capability.available
    assert capability.executable_version == "SymbolicTool 1.2.3"
    assert "fixture was not run" in capability.reason
    assert capability.to_dict()["installer_success_is_conformance"] is False
    assert len(runtime.requests) == 1


@pytest.mark.parametrize(
    ("adapter_type", "tool", "name"),
    (
        (TamarinAdapter, ProtocolTool.TAMARIN, "tamarin-prover"),
        (ProVerifAdapter, ProtocolTool.PROVERIF, "proverif"),
    ),
)
def test_only_complete_end_to_end_fixture_promotes_a_lane(
    tmp_path: Path, adapter_type, tool: ProtocolTool, name: str
) -> None:
    runtime = FakeRuntime(
        _executable(tmp_path, name), fixture_output=_fixture_output(tool)
    )
    capability = adapter_type(
        which=runtime.which,
        command_runner=runtime.run,
        monotonic=lambda: 1.0,
    ).probe()

    assert capability.status is ToolCapabilityStatus.CONFORMANT
    assert capability.available
    assert capability.conformance_passed
    receipt = capability.conformance_receipt
    assert receipt is not None
    assert receipt.status is ConformanceStatus.PASSED
    assert len(receipt.safe_markers_matched) == 4
    assert receipt.attack_marker_matched
    assert receipt.fixture_identity == adapter_type.fixture.content_id
    assert receipt.executable_identity.startswith("sha256:")
    assert receipt.command_identity.startswith("b")
    assert ProtocolConformanceReceipt.from_dict(receipt.to_record()) == receipt
    assert ProtocolToolCapability.from_dict(capability.to_record()) == capability


@pytest.mark.parametrize(
    ("adapter_type", "tool", "name"),
    (
        (TamarinAdapter, ProtocolTool.TAMARIN, "tamarin-prover"),
        (ProVerifAdapter, ProtocolTool.PROVERIF, "proverif"),
    ),
)
def test_fixture_must_both_prove_safe_queries_and_detect_attack(
    tmp_path: Path, adapter_type, tool: ProtocolTool, name: str
) -> None:
    full = _fixture_output(tool).splitlines()
    for incomplete in (
        "\n".join(full[:-1]),
        "\n".join(full[1:]),
        "installation succeeded",
    ):
        runtime = FakeRuntime(
            _executable(tmp_path, name), fixture_output=incomplete
        )
        capability = adapter_type(
            which=runtime.which, command_runner=runtime.run
        ).probe()
        assert capability.status is ToolCapabilityStatus.NONCONFORMANT
        assert not capability.available
        assert capability.conformance_receipt is not None
        assert not capability.conformance_receipt.passed


def test_contract_rejects_forged_passing_fixture_receipt() -> None:
    command = ("/tool", "--prove", "/fixture")
    with pytest.raises(ProtocolValidationError, match="complete end-to-end"):
        ProtocolConformanceReceipt(
            tool=ProtocolTool.TAMARIN,
            status=ConformanceStatus.PASSED,
            fixture_id="fixture",
            fixture_identity="sha256:" + "1" * 64,
            fixture_source_identity="sha256:" + "2" * 64,
            executable_path="/tool",
            executable_identity="sha256:" + "3" * 64,
            executable_version="1",
            command=command,
            command_identity=__import__(
                "ipfs_accelerate_py.agent_supervisor.formal_verification_contracts",
                fromlist=["content_identity"],
            ).content_identity({"command": command}),
            returncode=0,
            timed_out=False,
            output_sha256="sha256:" + "4" * 64,
            output_truncated=False,
            safe_markers_matched=("only one marker",),
            attack_marker_matched=False,
            duration_ms=1,
            reason="forged",
        )


def _conformant_adapter(tmp_path: Path, adapter_type, model: ProtocolModel):
    tool = adapter_type.tool
    runtime = FakeRuntime(
        _executable(tmp_path, tool.value),
        fixture_output=_fixture_output(tool),
        model_output=_verified_output(model, tool),
    )
    adapter = adapter_type(which=runtime.which, command_runner=runtime.run)
    capability = adapter.probe()
    assert capability.available
    return adapter, capability, runtime


@pytest.mark.parametrize("model", DEFAULT_PROTOCOL_MODELS)
@pytest.mark.parametrize("adapter_type", (TamarinAdapter, ProVerifAdapter))
def test_passing_model_run_has_exact_authoritative_receipts(
    tmp_path: Path, model: ProtocolModel, adapter_type
) -> None:
    adapter, capability, runtime = _conformant_adapter(
        tmp_path, adapter_type, model
    )
    result = adapter.verify(model, capability)

    assert result.verdict is ProtocolVerdict.VERIFIED
    assert result.authoritative
    assert len(result.query_results) == len(model.queries)
    assert all(
        item.verdict is ProtocolVerdict.VERIFIED
        for item in result.query_results
    )
    receipt = result.toolchain_receipt
    assert receipt is not None
    assert receipt.model_identity == model.content_id
    assert receipt.model_source_identity == model.source_identity_for(adapter.tool)
    assert receipt.query_set_identity == model.query_set_identity
    assert receipt.capability_identity == capability.content_id
    assert (
        receipt.conformance_receipt_identity
        == capability.conformance_receipt.content_id
    )
    assert receipt.executable_identity == capability.executable_identity
    assert receipt.stdout_sha256.startswith("sha256:")
    assert ProtocolToolchainReceipt.from_dict(receipt.to_record()) == receipt
    assert ProtocolLaneResult.from_dict(result.to_record()) == result
    assert len(runtime.requests) == 3


@pytest.mark.parametrize("adapter_type", (TamarinAdapter, ProVerifAdapter))
def test_attack_becomes_canonical_redacted_counterexample(
    tmp_path: Path, adapter_type
) -> None:
    query = CORE_PROTOCOL_MODEL.queries[1]
    tool = adapter_type.tool
    if tool is ProtocolTool.TAMARIN:
        output = "\n".join(
            (
                f"{CORE_PROTOCOL_MODEL.queries[0].tamarin_name}: verified",
                f"{query.tamarin_name}: falsified - attack found",
                (
                    f"TRACE query={query.query_id} step=7 "
                    "event=ClaimAccepted actor=intruder "
                    "message=TOP-SECRET-SIGNED-CLAIM"
                ),
            )
        )
    else:
        lines = [
            f"RESULT {item.proverif_label} is true."
            for item in CORE_PROTOCOL_MODEL.queries
        ]
        lines[1] = f"RESULT {query.proverif_label} is false."
        lines.append(
            f"TRACE query={query.query_id} step=7 "
            "event=ClaimAccepted actor=intruder "
            "message=TOP-SECRET-SIGNED-CLAIM"
        )
        output = "\n".join(lines)
    runtime = FakeRuntime(
        _executable(tmp_path, tool.value),
        fixture_output=_fixture_output(tool),
        model_output=output,
    )
    adapter = adapter_type(which=runtime.which, command_runner=runtime.run)
    capability = adapter.probe()
    result = adapter.verify(CORE_PROTOCOL_MODEL, capability)

    assert result.verdict is ProtocolVerdict.VIOLATED
    attacked = next(
        item for item in result.query_results if item.query_id == query.query_id
    )
    assert attacked.verdict is ProtocolVerdict.VIOLATED
    counterexample = attacked.counterexample
    assert counterexample is not None
    assert counterexample.minimized and counterexample.redacted
    assert len(counterexample.steps) == 1
    assert counterexample.steps[0].index == 0
    assert counterexample.steps[0].message_ref.startswith("sha256:")
    serialized = counterexample.to_json()
    assert "TOP-SECRET-SIGNED-CLAIM" not in serialized
    assert "contains_raw_transcript" in serialized
    assert ProtocolAttackCounterexample.from_dict(
        counterexample.to_record()
    ) == counterexample


def test_canonical_counterexample_is_deterministic_and_abstraction_bound() -> None:
    query = CORE_PROTOCOL_MODEL.queries[3]
    output = "\n".join(
        (
            f"TRACE query={query.query_id} step=9 event=Mutation "
            "actor=old_claimant message=lease-token-0",
            f"TRACE query={query.query_id} step=3 event=Grant "
            "actor=supervisor message=lease-token-0",
            f"TRACE query={query.query_id} step=9 event=Mutation "
            "actor=old_claimant message=lease-token-0",
        )
    )
    left = canonicalize_attack_trace(
        CORE_PROTOCOL_MODEL, ProtocolTool.TAMARIN, query, output
    )
    right = canonicalize_attack_trace(
        CORE_PROTOCOL_MODEL, ProtocolTool.TAMARIN, query, output
    )

    assert left == right
    assert left.content_id == right.content_id
    assert [step.index for step in left.steps] == [0, 1]
    assert left.query_identity == query.content_id
    assert left.abstraction_identity.startswith("b")
    assert output not in left.to_json()


def test_unavailable_capability_cannot_produce_model_receipt() -> None:
    adapter = TamarinAdapter(which=lambda _name: None)
    capability = adapter.probe()
    result = adapter.verify(CORE_PROTOCOL_MODEL, capability)

    assert result.verdict is ProtocolVerdict.UNAVAILABLE
    assert result.authoritative is False
    assert result.query_results == ()
    assert result.toolchain_receipt is None


def test_toolchain_drift_invalidates_conformant_capability(tmp_path: Path) -> None:
    adapter, capability, runtime = _conformant_adapter(
        tmp_path, TamarinAdapter, CORE_PROTOCOL_MODEL
    )
    assert runtime.executable is not None
    runtime.executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")

    result = adapter.verify(CORE_PROTOCOL_MODEL, capability)
    assert result.verdict is ProtocolVerdict.UNAVAILABLE
    assert not result.authoritative
    assert "drifted" in result.reason


def test_suite_requires_both_authoritative_lanes(tmp_path: Path) -> None:
    tamarin, tamarin_cap, _ = _conformant_adapter(
        tmp_path, TamarinAdapter, CORE_PROTOCOL_MODEL
    )
    proverif = ProVerifAdapter(which=lambda _name: None)
    proverif_cap = proverif.probe()
    verifier = ProtocolVerifier((tamarin, proverif))

    partial = verifier.verify(
        CORE_PROTOCOL_MODEL, capabilities=(tamarin_cap, proverif_cap)
    )
    assert partial.complete is False
    assert partial.verdict is ProtocolVerdict.UNAVAILABLE
    assert len(partial.lane_results) == 2
    assert sum(item.authoritative for item in partial.lane_results) == 1
    assert ProtocolSuiteResult.from_dict(partial.to_record()) == partial


def test_probe_reports_both_lanes_independently() -> None:
    capabilities = probe_protocol_tools(
        (
            TamarinAdapter(which=lambda _name: None),
            ProVerifAdapter(which=lambda _name: None),
        )
    )
    assert {item.tool for item in capabilities} == set(ProtocolTool)
    assert all(not item.available for item in capabilities)
