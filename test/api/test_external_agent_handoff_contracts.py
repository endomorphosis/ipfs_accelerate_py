"""Deterministic tests for EAAEF-010 transport-neutral handoff contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    ABSOLUTE_MAX_EVENTS,
    AGENT_CHECKPOINT_INTERFACE,
    AGENT_CONTEXT_ARTIFACT_INTERFACE,
    APPROVAL_EVENT_INTERFACE,
    CONVERSATION_EVENT_INTERFACE,
    CONTRACT_VERSION,
    ENCRYPTED_EXPORT_REFERENCE_INTERFACE,
    EXTERNAL_AGENT_HANDOFF_REQUEST_INTERFACE,
    EXTERNAL_AGENT_HANDOFF_REQUEST_SCHEMA,
    EXTERNAL_AGENT_SESSION_INTERFACE,
    HANDOFF_ADMISSION_RECEIPT_INTERFACE,
    HANDOFF_CONTRACT_FAMILY,
    HANDOFF_CONTRACT_VERSION,
    HANDOFF_NORMALIZATION_REPORT_INTERFACE,
    PATCH_EVENT_INTERFACE,
    SCHEMA_VERSION,
    TOOL_INVOCATION_EVENT_INTERFACE,
    TOOL_RESULT_EVENT_INTERFACE,
    AdmissionVerdict,
    AgentCheckpoint,
    AgentContextArtifact,
    ApprovalDecision,
    ApprovalEvent,
    ApprovalKind,
    ContextArtifactKind,
    ConversationEvent,
    ConversationRole,
    DisclosureClass,
    EncryptedExportReference,
    EventKind,
    ExternalAgentHandoffRequest,
    ExternalAgentSession,
    HandoffAdmissionReceipt,
    HandoffBounds,
    HandoffBoundsError,
    HandoffContractError,
    HandoffIdentityError,
    HandoffMode,
    HandoffNormalizationReport,
    HandoffProvenance,
    HandoffTrustError,
    HandoffVersionError,
    PatchEvent,
    PatchKind,
    RetentionClass,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
    TrustClass,
    canonical_handoff_json_bytes,
    decode_handoff_contract,
    decode_handoff_event,
    normalized_stream_identity,
    validate_event_sequence,
)


FIXED_MS = 1_700_000_000_000
SHA_A = "sha256:" + ("a" * 64)
SHA_B = "sha256:" + ("b" * 64)
SHA_C = "sha256:" + ("c" * 64)
SHA_D = "sha256:" + ("d" * 64)
SHA_E = "sha256:" + ("e" * 64)
SHA_F = "sha256:" + ("f" * 64)
DIGEST_A = "1" * 64

STACK_COMPATIBILITY = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "stack_compatibility_manifest.json"
)


def _provenance(
    *,
    trust_class: TrustClass = TrustClass.IMPORTED_EXPORTABLE,
    source_family: SourceFamily = SourceFamily.CODEX,
) -> HandoffProvenance:
    return HandoffProvenance(
        source_family=source_family,
        source_export_version="codex-export-1",
        adapter_id="codex@1",
        captured_at_ms=FIXED_MS,
        trust_class=trust_class,
        exportable=trust_class is not TrustClass.REJECTED,
    )


def _export_ref() -> EncryptedExportReference:
    return EncryptedExportReference(
        ciphertext_cid=SHA_A,
        digest_sha256=DIGEST_A,
        byte_count=2048,
        key_envelope_cid=SHA_B,
    )


def _conversation(sequence: int = 0, **changes: object) -> ConversationEvent:
    values: dict[str, object] = {
        "sequence": sequence,
        "role": ConversationRole.USER,
        "text": "continue from the exported session",
        "reasoning_summary": "explicit exportable summary",
        "provenance": _provenance(),
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return ConversationEvent(**values)


def _invocation(sequence: int = 1, **changes: object) -> ToolInvocationEvent:
    values: dict[str, object] = {
        "sequence": sequence,
        "tool_name": "apply_patch",
        "arguments": {"path": "src/example.py"},
        "provenance": _provenance(),
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return ToolInvocationEvent(**values)


def _tool_result(sequence: int = 2, **changes: object) -> ToolResultEvent:
    values: dict[str, object] = {
        "sequence": sequence,
        "tool_name": "apply_patch",
        "invocation_event_id": SHA_C,
        "result_content_id": SHA_D,
        "result_excerpt": "ok",
        "claimed_success": True,
        "provenance": _provenance(),
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return ToolResultEvent(**values)


def _patch(sequence: int = 3, **changes: object) -> PatchEvent:
    values: dict[str, object] = {
        "sequence": sequence,
        "patch_kind": PatchKind.UNIFIED_DIFF,
        "patch_content_id": SHA_E,
        "paths": ("src/example.py",),
        "claimed_applied": True,
        "provenance": _provenance(),
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return PatchEvent(**values)


def _approval(sequence: int = 4, **changes: object) -> ApprovalEvent:
    values: dict[str, object] = {
        "sequence": sequence,
        "approval_kind": ApprovalKind.IMPORTED_CLAIM,
        "decision": ApprovalDecision.APPROVE,
        "subject_content_id": SHA_E,
        "provenance": _provenance(),
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return ApprovalEvent(**values)


def _session(events: tuple[object, ...] | None = None, **changes: object) -> ExternalAgentSession:
    if events is None:
        events = (_conversation(), _invocation(), _tool_result(), _patch(), _approval())
    event_ids = validate_event_sequence(events)  # type: ignore[arg-type]
    values: dict[str, object] = {
        "source_family": SourceFamily.CODEX,
        "raw_export_id": SHA_A,
        "event_content_ids": event_ids,
        "provenance": _provenance(),
        "objective_id": "objective:handoff",
        "context_id": "context:pack",
        "repository_id": "repo:example",
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return ExternalAgentSession(**values)


def _request(session: ExternalAgentSession | None = None, **changes: object) -> ExternalAgentHandoffRequest:
    session = session or _session()
    values: dict[str, object] = {
        "source_family": SourceFamily.CODEX,
        "source_export_version": "codex-export-1",
        "raw_export_ref": _export_ref(),
        "session_id": session.session_id,
        "caller_principal_id": "principal:operator",
        "idempotency_key": "idempotency:handoff-1",
        "mode": HandoffMode.PREVIEW,
        "objective_id": "objective:handoff",
        "context_id": "context:pack",
        "repository_id": "repo:example",
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return ExternalAgentHandoffRequest(**values)


def test_frozen_contract_family_matches_stack_compatibility_manifest() -> None:
    assert HANDOFF_CONTRACT_VERSION == 1
    assert CONTRACT_VERSION == 1
    assert SCHEMA_VERSION == 1
    expected = {
        "request": EXTERNAL_AGENT_HANDOFF_REQUEST_INTERFACE,
        "session": EXTERNAL_AGENT_SESSION_INTERFACE,
        "conversation_event": CONVERSATION_EVENT_INTERFACE,
        "tool_invocation_event": TOOL_INVOCATION_EVENT_INTERFACE,
        "tool_result_event": TOOL_RESULT_EVENT_INTERFACE,
        "patch_event": PATCH_EVENT_INTERFACE,
        "approval_event": APPROVAL_EVENT_INTERFACE,
        "checkpoint": AGENT_CHECKPOINT_INTERFACE,
        "context_artifact": AGENT_CONTEXT_ARTIFACT_INTERFACE,
        "normalization_report": HANDOFF_NORMALIZATION_REPORT_INTERFACE,
        "admission_receipt": HANDOFF_ADMISSION_RECEIPT_INTERFACE,
    }
    assert dict(HANDOFF_CONTRACT_FAMILY) == expected
    assert STACK_COMPATIBILITY.is_file()
    manifest = json.loads(STACK_COMPATIBILITY.read_text(encoding="utf-8"))
    frozen = dict(manifest["frozen_contracts"]["handoff"])
    frozen.pop("status", None)
    assert frozen == expected


def test_nested_contracts_round_trip_and_preserve_content_identity() -> None:
    session = _session()
    request = _request(session)
    checkpoint = AgentCheckpoint(
        session_id=session.session_id,
        sequence=0,
        event_content_ids=session.event_content_ids[:2],
        normalized_stream_id="",
        provenance=_provenance(),
        repository_id="repo:example",
        tree_id="tree:abc",
        created_at_ms=FIXED_MS,
    )
    artifact = AgentContextArtifact(
        kind=ContextArtifactKind.OBJECTIVE,
        artifact_content_id=SHA_F,
        provenance=_provenance(),
        summary="typed objective capsule",
    )
    report = HandoffNormalizationReport(
        request_id=request.request_id,
        session_id=session.session_id,
        source_family=SourceFamily.CODEX,
        raw_export_id=SHA_A,
        accepted_event_ids=session.event_content_ids,
        imported_success_claims_untrusted=1,
        created_at_ms=FIXED_MS,
    )
    receipt = HandoffAdmissionReceipt(
        request_id=request.request_id,
        session_id=session.session_id,
        verdict=AdmissionVerdict.PREVIEW_ONLY,
        trust_class=TrustClass.IMPORTED_EXPORTABLE,
        raw_export_id=session.raw_export_id,
        normalized_stream_id=session.normalized_stream_id,
        reason_code="imported_history_untrusted",
        policy_id="policy:handoff@1",
        objective_id="objective:handoff",
        context_id="context:pack",
        repository_id="repo:example",
        created_at_ms=FIXED_MS,
    )
    values = (
        HandoffBounds(),
        _provenance(),
        _export_ref(),
        _conversation(),
        _invocation(),
        _tool_result(),
        _patch(),
        _approval(),
        checkpoint,
        artifact,
        session,
        request,
        report,
        receipt,
    )
    for value in values:
        restored = type(value).from_json(value.to_json())
        assert restored == value
        assert restored.content_id == value.content_id
        assert decode_handoff_contract(json.loads(value.to_json())) == value
        assert json.loads(value.to_json())["contract_version"] == 1
        assert json.loads(value.to_json())["schema"].endswith("@1")
        assert json.loads(value.to_json())["interface"].endswith("@1")


def test_serialization_and_identity_are_deterministic_across_input_order() -> None:
    first = _invocation(arguments={"b": 2, "a": 1})
    second = _invocation(arguments={"a": 1, "b": 2})
    assert first.to_json() == second.to_json()
    assert first.content_id == second.content_id
    assert first.to_json() == json.dumps(
        json.loads(first.to_json()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert canonical_handoff_json_bytes(first) == first.canonical_bytes()


def test_records_are_frozen() -> None:
    event = _conversation()
    with pytest.raises(FrozenInstanceError):
        event.text = "mutated"  # type: ignore[misc]
    request = _request()
    with pytest.raises(FrozenInstanceError):
        request.mode = HandoffMode.ATTACH  # type: ignore[misc]


def test_unknown_schema_and_version_are_rejected() -> None:
    payload = _conversation().to_dict()
    payload["schema"] = "ipfs_accelerate_py/agent-supervisor/conversation-event@2"
    with pytest.raises(HandoffVersionError):
        ConversationEvent.from_dict(payload)
    payload = _request().to_dict()
    payload["interface"] = "ExternalAgentHandoffRequest@2"
    with pytest.raises(HandoffVersionError):
        ExternalAgentHandoffRequest.from_dict(payload)
    payload = _session().to_dict()
    payload["contract_version"] = 2
    with pytest.raises(HandoffVersionError):
        ExternalAgentSession.from_dict(payload)
    with pytest.raises(HandoffVersionError):
        decode_handoff_contract({"schema": "UnknownRecord@1", "contract_version": 1})


def test_unknown_top_level_fields_are_rejected() -> None:
    payload = _conversation().to_dict()
    payload["extra"] = "nope"
    with pytest.raises(HandoffContractError, match="unsupported fields"):
        ConversationEvent.from_dict(payload)


def test_forged_content_identity_is_rejected() -> None:
    payload = _conversation().to_dict()
    payload["content_id"] = SHA_A
    with pytest.raises(HandoffIdentityError):
        ConversationEvent.from_dict(payload)
    payload = _request().to_dict()
    payload["request_id"] = SHA_A
    with pytest.raises(HandoffIdentityError):
        ExternalAgentHandoffRequest.from_dict(payload)


def test_hidden_chain_of_thought_is_rejected() -> None:
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        _conversation(residual_fields={"thinking": "secret scratchpad"})
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        _conversation(residual_fields={"chain_of_thought": "hidden"})
    payload = _conversation().to_dict()
    payload["residual_fields"] = {"hidden_reasoning": "no"}
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        ConversationEvent.from_dict(payload)


def test_residual_unknown_fields_are_bounded() -> None:
    event = _conversation(residual_fields={"client_tag": "visible"})
    assert event.residual_fields["client_tag"] == "visible"
    restored = ConversationEvent.from_json(event.to_json())
    assert restored.residual_fields == event.residual_fields
    too_many = {f"field_{index}": "x" for index in range(HandoffBounds().max_unknown_fields + 1)}
    with pytest.raises(HandoffBoundsError):
        _conversation(residual_fields=too_many)
    nested: dict[str, object] = {"leaf": "too deep"}
    for _ in range(HandoffBounds().max_depth + 1):
        nested = {"child": nested}
    with pytest.raises(HandoffBoundsError):
        _conversation(residual_fields=nested)
    with pytest.raises(HandoffContractError, match="floats"):
        _conversation(residual_fields={"score": 1.5})


def test_imported_tool_calls_are_never_executed_or_trusted() -> None:
    invocation = _invocation()
    assert invocation.executed is False
    assert invocation.kind is EventKind.TOOL_INVOCATION
    with pytest.raises(HandoffTrustError):
        _invocation(executed=True)
    payload = invocation.to_dict()
    payload["executed"] = True
    with pytest.raises(HandoffTrustError):
        ToolInvocationEvent.from_dict(payload)

    result = _tool_result()
    assert result.claimed_success is True
    assert result.trusted_success is False
    with pytest.raises(HandoffTrustError):
        _tool_result(trusted_success=True)

    patch = _patch()
    assert patch.claimed_applied is True
    assert patch.applied is False
    with pytest.raises(HandoffTrustError):
        _patch(applied=True)

    approval = _approval()
    assert approval.grants_effects is False
    with pytest.raises(HandoffTrustError):
        _approval(grants_effects=True)


def test_event_sequence_must_be_strictly_increasing() -> None:
    events = (_conversation(0), _invocation(1), _tool_result(2))
    assert validate_event_sequence(events) == tuple(item.event_id for item in events)
    with pytest.raises(HandoffContractError, match="strictly increasing"):
        validate_event_sequence((_conversation(1), _invocation(1)))
    decoded = decode_handoff_event(_patch().to_dict())
    assert isinstance(decoded, PatchEvent)
    assert decoded.kind is EventKind.PATCH


def test_raw_normalized_objective_context_and_repository_identities_are_distinct() -> None:
    session = _session()
    assert session.raw_export_id != session.normalized_stream_id
    assert session.normalized_stream_id == normalized_stream_identity(session.event_content_ids)
    with pytest.raises(HandoffIdentityError, match="distinct"):
        _session(raw_export_id=normalized_stream_identity(()), event_content_ids=())
    with pytest.raises(HandoffIdentityError, match="distinct"):
        _session(objective_id="repo:example")


def test_checkpoint_binds_restart_safe_prefix_identity() -> None:
    session = _session()
    checkpoint = AgentCheckpoint(
        session_id=session.session_id,
        sequence=0,
        event_content_ids=session.event_content_ids[:3],
        normalized_stream_id="",
        provenance=_provenance(),
        created_at_ms=FIXED_MS,
    )
    assert checkpoint.restart_safe is True
    assert checkpoint.normalized_stream_id == normalized_stream_identity(
        session.event_content_ids[:3]
    )
    with pytest.raises(HandoffContractError, match="restart-safe"):
        AgentCheckpoint(
            session_id=session.session_id,
            sequence=0,
            event_content_ids=(),
            normalized_stream_id="",
            provenance=_provenance(),
            restart_safe=False,
        )
    with pytest.raises(HandoffIdentityError):
        AgentCheckpoint(
            session_id=session.session_id,
            sequence=0,
            event_content_ids=session.event_content_ids[:1],
            normalized_stream_id=SHA_A,
            provenance=_provenance(),
        )


def test_context_artifact_excludes_bodies_and_encrypted_raw_disclosure() -> None:
    artifact = AgentContextArtifact(
        kind=ContextArtifactKind.CAPSULE,
        artifact_content_id=SHA_F,
        provenance=_provenance(),
        summary="reference only",
    )
    assert "body" not in artifact.to_dict()
    assert artifact.disclosure_class is DisclosureClass.PUBLIC_PROJECTION
    with pytest.raises(HandoffContractError, match="encrypted raw"):
        AgentContextArtifact(
            kind=ContextArtifactKind.CAPSULE,
            artifact_content_id=SHA_F,
            provenance=_provenance(),
            disclosure_class=DisclosureClass.ENCRYPTED_RAW,
        )
    payload = artifact.to_dict()
    payload["transcript_body"] = "nope"
    with pytest.raises(HandoffContractError, match="unsupported fields"):
        AgentContextArtifact.from_dict(payload)


def test_handoff_request_rejects_provider_selection_and_transcript_bodies() -> None:
    request = _request()
    assert request.schema == EXTERNAL_AGENT_HANDOFF_REQUEST_SCHEMA
    assert request.mode is HandoffMode.PREVIEW
    payload = request.to_dict()
    payload["provider_id"] = "grok"
    with pytest.raises(HandoffTrustError, match="cannot select a provider"):
        ExternalAgentHandoffRequest.from_dict(payload)
    with pytest.raises(HandoffContractError, match="public projection"):
        _request(disclosure_class=DisclosureClass.ENCRYPTED_RAW)


def test_normalization_report_keeps_raw_and_stream_identities_apart() -> None:
    session = _session()
    request = _request(session)
    report = HandoffNormalizationReport(
        request_id=request.request_id,
        session_id=session.session_id,
        source_family=SourceFamily.CODEX,
        raw_export_id=session.raw_export_id,
        accepted_event_ids=session.event_content_ids,
        unknown_fields_retained=1,
        hidden_chain_of_thought_rejected=2,
        imported_success_claims_untrusted=1,
        created_at_ms=FIXED_MS,
    )
    assert report.imported_invocations_not_executed is True
    assert report.raw_export_id != report.normalized_stream_id
    assert "transcript" not in report.to_dict()
    payload = report.to_dict()
    payload["transcript_body"] = "exported chat"
    with pytest.raises(HandoffContractError, match="transcript"):
        HandoffNormalizationReport.from_dict(payload)
    with pytest.raises(HandoffTrustError):
        HandoffNormalizationReport(
            request_id=request.request_id,
            session_id=session.session_id,
            source_family=SourceFamily.CODEX,
            raw_export_id=session.raw_export_id,
            accepted_event_ids=session.event_content_ids,
            imported_invocations_not_executed=False,
        )


def test_admission_receipt_completion_requires_reverified_or_admitted_trust() -> None:
    session = _session()
    request = _request(session)
    imported = HandoffAdmissionReceipt(
        request_id=request.request_id,
        session_id=session.session_id,
        verdict=AdmissionVerdict.PREVIEW_ONLY,
        trust_class=TrustClass.IMPORTED_UNVERIFIED,
        raw_export_id=session.raw_export_id,
        normalized_stream_id=session.normalized_stream_id,
        reason_code="imported_unverified",
        policy_id="policy:handoff@1",
        created_at_ms=FIXED_MS,
    )
    assert imported.completion_eligible is False
    with pytest.raises(HandoffTrustError, match="satisfy completion"):
        HandoffAdmissionReceipt(
            request_id=request.request_id,
            session_id=session.session_id,
            verdict=AdmissionVerdict.ADMITTED,
            trust_class=TrustClass.IMPORTED_EXPORTABLE,
            raw_export_id=session.raw_export_id,
            normalized_stream_id=session.normalized_stream_id,
            reason_code="claimed_complete",
            policy_id="policy:handoff@1",
            completion_eligible=True,
        )
    admitted = HandoffAdmissionReceipt(
        request_id=request.request_id,
        session_id=session.session_id,
        verdict=AdmissionVerdict.ADMITTED,
        trust_class=TrustClass.INDEPENDENTLY_ADMITTED,
        raw_export_id=session.raw_export_id,
        normalized_stream_id=session.normalized_stream_id,
        reason_code="independently_admitted",
        policy_id="policy:handoff@1",
        created_at_ms=FIXED_MS,
    )
    assert admitted.completion_eligible is True
    with pytest.raises(HandoffTrustError):
        HandoffAdmissionReceipt(
            request_id=request.request_id,
            session_id=session.session_id,
            verdict=AdmissionVerdict.ADMITTED,
            trust_class=TrustClass.QUARANTINED,
            raw_export_id=session.raw_export_id,
            normalized_stream_id=session.normalized_stream_id,
            reason_code="quarantined",
            policy_id="policy:handoff@1",
        )


def test_bounds_reject_absolute_and_relative_overflow() -> None:
    with pytest.raises(HandoffBoundsError):
        HandoffBounds(max_events=ABSOLUTE_MAX_EVENTS + 1)
    with pytest.raises(HandoffBoundsError):
        HandoffBounds(max_text_bytes=8_000, max_record_bytes=4_000)
    tight = HandoffBounds(
        max_events=2,
        max_text_bytes=8,
        max_record_bytes=32_768,
        max_serialized_bytes=65_536,
        max_unknown_field_bytes=2_048,
    )
    with pytest.raises(HandoffBoundsError):
        _conversation(sequence=0, bounds=tight, text="abcdefghijk")
    with pytest.raises(HandoffContractError, match="repository-relative"):
        _patch(paths=("../secret.py",))
    with pytest.raises(HandoffContractError, match="private material"):
        _invocation(arguments={"api_key": "sk-test"})


def test_malformed_json_and_non_objects_fail_closed() -> None:
    with pytest.raises(HandoffContractError, match="malformed"):
        ConversationEvent.from_json("{")
    with pytest.raises(HandoffContractError, match="object"):
        ConversationEvent.from_json("[]")
    with pytest.raises(HandoffContractError):
        decode_handoff_event("not-an-object")  # type: ignore[arg-type]


def test_human_approval_with_binding_may_grant_effects() -> None:
    event = _approval(
        approval_kind=ApprovalKind.HUMAN,
        provenance=_provenance(trust_class=TrustClass.LOCALLY_REVERIFIED),
        authority_binding_id="binding:operator",
        grants_effects=True,
    )
    assert event.grants_effects is True
    restored = ApprovalEvent.from_json(event.to_json())
    assert restored.grants_effects is True
    with pytest.raises(HandoffTrustError, match="authority binding"):
        _approval(
            approval_kind=ApprovalKind.HUMAN,
            provenance=_provenance(trust_class=TrustClass.LOCALLY_REVERIFIED),
            grants_effects=True,
        )


def test_raw_export_reference_identity_is_distinct_from_ciphertext_and_stream() -> None:
    export_ref = _export_ref()
    session = _session(raw_export_id=export_ref.content_id)
    request = _request(session, raw_export_ref=export_ref)
    assert request.raw_export_id == export_ref.content_id
    assert export_ref.ciphertext_cid == SHA_A
    assert request.raw_export_id != export_ref.ciphertext_cid
    assert request.raw_export_id != session.normalized_stream_id
    assert session.raw_export_id == export_ref.content_id
    assert export_ref.disclosure_class is DisclosureClass.ENCRYPTED_RAW
    assert export_ref.retention_class is RetentionClass.SESSION
    with pytest.raises(HandoffIdentityError, match="distinct"):
        EncryptedExportReference(
            ciphertext_cid=SHA_A,
            digest_sha256="a" * 64,
            byte_count=8,
            key_envelope_cid=SHA_B,
        )


def test_event_and_session_identities_must_be_content_addressed() -> None:
    with pytest.raises(HandoffContractError, match="sha256 or CIDv1"):
        _session(event_content_ids=("event:not-addressed",))
    with pytest.raises(HandoffContractError, match="sha256 or CIDv1"):
        _session(checkpoint_ids=("checkpoint-1",))
    session = _session()
    restored = ExternalAgentSession.from_dict(
        {k: v for k, v in session.to_dict().items() if k != "schema"}
    )
    assert restored.session_id == session.session_id
    assert restored.session_id.startswith("b")


def test_session_source_family_must_match_provenance() -> None:
    with pytest.raises(HandoffContractError, match="source_family"):
        _session(source_family=SourceFamily.CLAUDE_CODE)
    for family in SourceFamily:
        event = _conversation(
            provenance=_provenance(source_family=family, trust_class=TrustClass.IMPORTED_EXPORTABLE)
        )
        session = ExternalAgentSession(
            source_family=family,
            raw_export_id=SHA_A,
            event_content_ids=(event.event_id,),
            provenance=_provenance(source_family=family),
            created_at_ms=FIXED_MS,
        )
        assert session.source_family is family
        assert decode_handoff_contract(session.to_dict()) == session


def test_decode_accepts_interface_without_schema_and_rejects_unknown_kind() -> None:
    payload = _conversation().to_dict()
    payload.pop("schema")
    restored = decode_handoff_contract(payload)
    assert isinstance(restored, ConversationEvent)
    assert restored.event_id == _conversation().event_id
    with pytest.raises(HandoffVersionError):
        decode_handoff_event({"kind": "hidden_thought", "contract_version": 1})
    payload = _session().to_dict()
    payload["schema_version"] = 2
    with pytest.raises(HandoffVersionError):
        ExternalAgentSession.from_dict(payload)


def test_locally_reverified_admitted_receipt_may_satisfy_completion() -> None:
    session = _session()
    request = _request(session)
    receipt = HandoffAdmissionReceipt(
        request_id=request.request_id,
        session_id=session.session_id,
        verdict=AdmissionVerdict.ADMITTED,
        trust_class=TrustClass.LOCALLY_REVERIFIED,
        raw_export_id=session.raw_export_id,
        normalized_stream_id=session.normalized_stream_id,
        reason_code="locally_reverified",
        policy_id="policy:handoff@1",
        created_at_ms=FIXED_MS,
    )
    assert receipt.completion_eligible is True
    assert TrustClass.IMPORTED_EXPORTABLE.may_satisfy_completion is False
    assert AdmissionVerdict.PREVIEW_ONLY.admits_session is True
    assert AdmissionVerdict.REJECTED.admits_session is False
    preview = _request(session, mode=HandoffMode.ATTACH)
    assert preview.mode is HandoffMode.ATTACH
    continued = _request(session, mode=HandoffMode.CONTINUE)
    assert continued.mode is HandoffMode.CONTINUE
    imported = _request(session, mode=HandoffMode.IMPORT_ONLY)
    assert imported.mode is HandoffMode.IMPORT_ONLY
    assert imported.to_dict()["mode"] == "import_only"
    with pytest.raises(HandoffTrustError, match="satisfy completion"):
        HandoffAdmissionReceipt(
            request_id=request.request_id,
            session_id=session.session_id,
            verdict=AdmissionVerdict.PREVIEW_ONLY,
            trust_class=TrustClass.LOCALLY_REVERIFIED,
            raw_export_id=session.raw_export_id,
            normalized_stream_id=session.normalized_stream_id,
            reason_code="preview_is_not_completion",
            policy_id="policy:handoff@1",
            completion_eligible=True,
        )


def test_required_fields_paths_and_types_fail_closed() -> None:
    with pytest.raises(HandoffContractError, match="required"):
        _conversation(text="", reasoning_summary="")
    with pytest.raises(HandoffContractError, match="required"):
        _invocation(tool_name="")
    with pytest.raises(HandoffContractError, match="NUL"):
        _conversation(text="ok\x00hidden")
    with pytest.raises(HandoffContractError, match="repository-relative"):
        _patch(paths=("C:/Windows/system32",))
    with pytest.raises(HandoffContractError, match="non-negative integer"):
        _conversation(sequence=True)  # type: ignore[arg-type]
    with pytest.raises(HandoffContractError, match="boolean"):
        _tool_result(claimed_success="yes")  # type: ignore[arg-type]
    with pytest.raises(HandoffContractError, match="duplicate"):
        _session(event_content_ids=(SHA_A, SHA_A))


def test_record_byte_bound_is_enforced_from_handoff_bounds() -> None:
    tight = HandoffBounds(
        max_events=8,
        max_text_bytes=64,
        max_record_bytes=256,
        max_serialized_bytes=512,
        max_unknown_field_bytes=64,
    )
    with pytest.raises(HandoffBoundsError, match="max_record_bytes"):
        _conversation(bounds=tight)


def test_session_and_argument_bounds_reject_overflow_and_floats() -> None:
    tight = HandoffBounds(max_events=1, max_record_bytes=65_536, max_serialized_bytes=262_144)
    with pytest.raises(HandoffBoundsError):
        _session(
            event_content_ids=(SHA_A, SHA_B),
            bounds=tight,
            objective_id="",
            context_id="",
            repository_id="",
        )
    with pytest.raises(HandoffContractError, match="floats"):
        _invocation(arguments={"score": 0.5})
    payload = _conversation().to_dict()
    payload["kind"] = EventKind.PATCH.value
    with pytest.raises(HandoffContractError, match="kind"):
        ConversationEvent.from_dict(payload)
    payload = _conversation().to_dict()
    payload.pop("schema")
    payload.pop("interface")
    restored = decode_handoff_event(payload)
    assert isinstance(restored, ConversationEvent)
    with pytest.raises(HandoffVersionError):
        decode_handoff_contract(payload)


def test_public_records_are_references_not_export_bodies() -> None:
    request = _request()
    session = _session()
    for payload in (request.to_dict(), session.to_dict()):
        encoded = json.dumps(payload)
        assert "transcript" not in encoded
        assert "raw_bytes" not in payload
        assert "raw_export" not in payload
    export_ref = _export_ref()
    with pytest.raises(HandoffContractError, match="encrypted_raw"):
        EncryptedExportReference(
            ciphertext_cid=SHA_A,
            digest_sha256=DIGEST_A,
            byte_count=8,
            key_envelope_cid=SHA_B,
            disclosure_class=DisclosureClass.PUBLIC_PROJECTION,
        )
    rejected = HandoffProvenance(
        source_family=SourceFamily.GENERIC_JSON,
        source_export_version="generic-json-1",
        trust_class=TrustClass.REJECTED,
        exportable=False,
        captured_at_ms=FIXED_MS,
    )
    assert rejected.exportable is False
    for kind in ContextArtifactKind:
        artifact = AgentContextArtifact(
            kind=kind,
            artifact_content_id=SHA_F,
            provenance=_provenance(),
            summary="typed context capsule",
        )
        assert artifact.kind is kind
        assert decode_handoff_contract(artifact.to_dict()) == artifact
    assert export_ref.content_id != export_ref.ciphertext_cid
