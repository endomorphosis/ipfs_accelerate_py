"""Focused DCR-047 deterministic code-generation roundtrip tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityStatus,
    NetworkMode,
    SolverReadiness,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.codegen_repairs import (
    CODEGEN_ACTIVATION_STATUS,
    CodegenRoundtripDisposition,
    CodegenRoundtripRequest,
    GeneratedOutputObservation,
    GeneratorRunObservation,
    validate_codegen_roundtrip,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)


def _digest(label: str, value: bytes) -> str:
    prefix = "sha256" if label == "sha256" else f"{label}:sha256"
    return f"{prefix}:{hashlib.sha256(value).hexdigest()}"


def _descriptor() -> OperatorDescriptor:
    return OperatorDescriptor.from_mapping(
        {
            "operator_id": "codegen.sync.swissknife",
            "kind": "replace_exact_bytes",
            "input_schema": {
                "type": "object",
                "required": ["path", "before", "authority"],
                "properties": {"path": "path", "before": "sha256", "authority": "cid"},
                "additional_properties": False,
            },
            "owner_root": "swissknife",
            "write_scope": ["src/generated.ts"],
            "before_predicates": ["authority_source_current"],
            "after_predicates": ["semantic_roundtrip_equal"],
            "applicability_proofs": ["repeat_generation_identical"],
            "preview": {
                "kind": "metadata_only",
                "fields": ["path", "before", "authority"],
            },
            "inverse": {
                "kind": "restore_exact_before_bytes",
                "binding": "before",
            },
            "validation_commands": [["python3", "-m", "pytest", "-q"]],
        }
    )


def _evidence(*, kind: str, evidence_id: str, executable_digest: str) -> CapabilityEvidenceReceipt:
    return CapabilityEvidenceReceipt(
        evidence_id=evidence_id,
        evidence_kind=kind,
        subject_id="fixture-codegen",
        subject_digest=executable_digest,
        subject_version="1.0",
        transcript_digest=_digest("transcript", f"{kind}-ok".encode()),
        passed=True,
        network_mode=NetworkMode.OFFLINE,
    )


def _request() -> CodegenRoundtripRequest:
    descriptor = _descriptor()
    registry = OperatorRegistry(
        (descriptor,),
        reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id},
    )
    executable_digest = _digest("executable", b"fixture deterministic generator")
    toolchain = SolverReadiness(
        tool_id="fixture-codegen",
        status=CapabilityStatus.AVAILABLE,
        executable="fixture-codegen",
        path="/reviewed/bin/fixture-codegen",
        expected_version="1.0",
        version="1.0",
        executable_digest=executable_digest,
        self_test_id="self-test:fixture-codegen",
        self_test_passed=True,
        reconstruction_id="reconstruction:fixture-codegen",
        reconstructed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    output = GeneratedOutputObservation(
        owner_root="swissknife",
        relative_path="src/generated.ts",
        authority_source_cid="authority-source-cid",
        before_bytes=b"export const version = 0;\n",
        before_digest=_digest("sha256", b"export const version = 0;\n"),
        generated_bytes=b"export const version = 1;\n",
        generated_digest=_digest("sha256", b"export const version = 1;\n"),
        semantic_cid="semantic-contract-cid",
        decoded_semantic_cid="semantic-contract-cid",
    )
    first = GeneratorRunObservation(
        run_ordinal=1,
        generator_id=toolchain.tool_id,
        executable_digest=toolchain.executable_digest,
        generator_version=toolchain.version,
        argv=("fixture-codegen", "--input", "contracts/schema.json"),
        authority_source_cid="authority-source-cid",
        forest_cid="forest-cid",
        transcript_digest=_digest("transcript", b"run-one"),
        outputs=(output,),
    )
    second = replace(
        first,
        run_ordinal=2,
        transcript_digest=_digest("transcript", b"run-two"),
    )
    report = registry.report()
    return CodegenRoundtripRequest(
        descriptor=descriptor,
        registry=registry,
        pinned_registry_cid=report["registry_cid"],
        toolchain=toolchain,
        tool_evidence=(
            _evidence(
                kind="self_test",
                evidence_id=toolchain.self_test_id,
                executable_digest=executable_digest,
            ),
            _evidence(
                kind="reconstruction",
                evidence_id=toolchain.reconstruction_id,
                executable_digest=executable_digest,
            ),
        ),
        authority_source_cid="authority-source-cid",
        forest_cid="forest-cid",
        owner_root="swissknife",
        generated_paths=("src/generated.ts",),
        first_run=first,
        second_run=second,
    )


def test_repeatable_roundtrip_is_bound_but_never_authoritative() -> None:
    result = validate_codegen_roundtrip(_request())

    assert result.disposition is CodegenRoundtripDisposition.INTEGRATION_PENDING
    assert result.reason_codes == (CODEGEN_ACTIVATION_STATUS,)
    assert result.request_cid and len(result.run_cids) == 2
    assert len(result.output_cids) == len(result.inverse_cids) == 1
    assert result.execution_authorized is False
    assert result.completion_authorized is False
    assert result.model_call_count == result.provider_call_count == result.network_call_count == 0


def test_nondeterministic_second_generation_abstains() -> None:
    request = _request()
    changed = replace(
        request.second_run.outputs[0],
        generated_bytes=b"export const version = 2;\n",
        generated_digest=_digest("sha256", b"export const version = 2;\n"),
    )
    result = validate_codegen_roundtrip(
        replace(request, second_run=replace(request.second_run, outputs=(changed,)))
    )

    assert result.disposition is CodegenRoundtripDisposition.ABSTAINED
    assert result.reason_codes == ("repeat_generation_is_not_byte_identical",)
    assert not result.output_cids and not result.inverse_cids


def test_hand_owned_stale_tool_and_raw_request_reject() -> None:
    request = _request()
    hand_owned = replace(request.first_run.outputs[0], hand_owned=True)
    hand_owned_second = replace(request.second_run.outputs[0], hand_owned=True)
    rejected_hand_owned = validate_codegen_roundtrip(
        replace(
            request,
            first_run=replace(request.first_run, outputs=(hand_owned,)),
            second_run=replace(request.second_run, outputs=(hand_owned_second,)),
        )
    )
    rejected_tool = validate_codegen_roundtrip(
        replace(
            request,
            toolchain=replace(request.toolchain, status=CapabilityStatus.UNAVAILABLE),
        )
    )
    rejected_raw = validate_codegen_roundtrip({"prompt": "regenerate everything"})

    assert rejected_hand_owned.disposition is CodegenRoundtripDisposition.REJECTED
    assert rejected_hand_owned.reason_codes == (
        "authority_source_or_semantic_roundtrip_is_invalid",
    )
    assert rejected_tool.disposition is CodegenRoundtripDisposition.REJECTED
    assert rejected_raw.disposition is CodegenRoundtripDisposition.REJECTED
