"""Actual pinned native serializers on authored, unreviewed declarations.

The root runs these under a sealed P-first native lease. Deterministic native
compilers produce real fixture artifacts; no model/prover/owner verification
or production source trust is represented by these conformance tests.
"""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import native_formalization_profile as profile
from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
from ipfs_datasets_py.logic.intent_ir.formalize.compiler import IntentFormalizationCompiler
from ipfs_datasets_py.logic.intent_ir.schema import (
    ControlEdgeKind, IntentAction, IntentControlEdge, IntentIRDocument, IntentKind,
    IntentModality, IntentStatement, NodeGrounding, ReviewStatus, SourceRef, SourceSpan,
    StatementKind,
)
from ipfs_datasets_py.logic.security_ir.formalization_adapter import SecurityIRFormalizationAdapter
from ipfs_datasets_py.logic.security_ir.model import (
    Policy, PolicyEffect, Principal, Resource, SecurityClaim, SecurityExtension,
    SecurityIR, SecuritySource, StateMachine, StateTransition, ThreatAssumption,
)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def intent_case(opaque=False):
    source_text = "Authored fixture: publish the generated result if ready; retry on failure."
    ref = SourceRef("source:fixture", "urn:fixture:native-intent", "fixture", "fixture-v1",
        sha(source_text.encode()), review_status=ReviewStatus.UNREVIEWED,
        span=SourceSpan(0, len(source_text)))
    statements = (
        IntentStatement("goal", StatementKind.GOAL, IntentModality.INTENDED,
            "Publish the generated result.", (ref.ref_id,), "" if opaque else "publish",
            ("result", "agent"), confidence=0.75),
        IntentStatement("guard", StatementKind.GUARD, IntentModality.REQUIRED,
            "The result is ready.", (ref.ref_id,), "ready", ("result",)),
        IntentStatement("effect", StatementKind.EFFECT, IntentModality.ASSERTED,
            "The result is published.", (ref.ref_id,), "published", ("result",)),
        IntentStatement("verification", StatementKind.VERIFICATION, IntentModality.REQUIRED,
            "Observe publication.", (ref.ref_id,), "observed", ("result",),
            grounding=NodeGrounding.INFERRED),
    )
    document = IntentIRDocument("intent:fixture", "Unreviewed native fixture", IntentKind.PROCEDURE,
        (ref,), statements,
        (IntentAction("action:publish", "agent", "publish", ("result",), (ref.ref_id,),
             precondition_ids=("guard",), effect_ids=("effect",), verification_ids=("verification",)),
         IntentAction("action:retry", "agent", "retry", ("result",), (ref.ref_id,))),
        (IntentControlEdge("edge:failure", "action:publish", "action:retry", ControlEdgeKind.ON_FAILURE,
             guard_statement_id="guard", source_ref_ids=(ref.ref_id,)),
         IntentControlEdge("edge:retry", "action:retry", "action:retry", ControlEdgeKind.RETRY,
             source_ref_ids=(ref.ref_id,))),
        entry_action_ids=("action:publish",), terminal_action_ids=("action:retry",))
    artifact = IntentFormalizationCompiler().compile_document(document)
    artifact = replace(artifact, metadata={**artifact.metadata.to_dict(),
        "observation": {"body": "retained body", "text": "retained text",
                        "payload": {"confidence": 0.75}}})
    return document, artifact


def security_case(missing_source=False, extension=False):
    body = b"Authored unreviewed security fixture; declaration only."
    source = SecuritySource("source:security-fixture", "urn:fixture:native-security",
        revision="" if missing_source else "fixture-v1",
        content_sha256="" if missing_source else sha(body), review_status="unreviewed",
        attributes={"text": "retained original source descriptor"})
    assumption = ThreatAssumption("assumption:runtime", "Declared runtime premise.", (source.source_id,))
    policy = Policy("policy:publish", "Declared publish policy", PolicyEffect.ALLOW,
        principal_ids=("principal:agent",), resource_ids=("resource:result",), source_ids=(source.source_id,),
        attributes={"body": "native policy context", "weight": 0.25})
    machine = StateMachine("machine:publish", ("pending", "published"), "pending",
        (StateTransition("pending", "published", "publish", "ready", "record_publish",
                         attributes={"payload": {"text": "native transition context"}}),),
        source_ids=(source.source_id,))
    claim = SecurityClaim("claim:publish", "Only declared result bytes are published.", "publishing",
        assumption_ids=(assumption.assumption_id,), policy_ids=(policy.policy_id,), source_ids=(source.source_id,))
    declaration = SecurityIR("security:fixture", sources=(source,),
        principals=(Principal("principal:agent", source_ids=(source.source_id,)),),
        resources=(Resource("resource:result", owner_principal_ids=("principal:agent",), source_ids=(source.source_id,)),),
        assumptions=(assumption,), policies=(policy,), state_machines=(machine,), claims=(claim,),
        extensions=(SecurityExtension("extension:unmapped", "fixture-vocabulary", "1",
                    {"body": "retain optional semantics"}, required=False, source_ids=(source.source_id,)),) if extension else ())
    return declaration, SecurityIRFormalizationAdapter().adapt_artifact(declaration)


@pytest.fixture(params=("intent", "security"))
def native_case(request):
    return intent_case() if request.param == "intent" else security_case()


def test_complete_actual_native_payloads_and_separate_byte_identities_roundtrip(native_case):
    declaration, artifact = native_case
    original_declaration, original_artifact = declaration.to_dict(), artifact.to_dict()
    result = profile.capture_native_formalization_profile(declaration, artifact)
    wire = result.to_dict()
    assert wire["native_declaration"] == original_declaration
    assert wire["native_formalization"] == original_artifact
    assert result.native_formalization_json_bytes == artifact.to_json().encode()
    assert result.native_formalization_identity_preimage == artifact.canonical_bytes()
    assert result.native_formalization_identity_preimage != result.native_formalization_json_bytes
    assert wire["byte_bindings"]["native_formalization_identity"] == artifact.identity.to_dict()
    expected = canonical_intent_ir_bytes(declaration) if type(declaration) is IntentIRDocument else declaration.canonical_bytes()
    assert result.native_declaration_bytes == expected
    if type(declaration) is SecurityIR:
        assert result.native_declaration_identity_preimage == declaration.identity.canonical_bytes
        assert result.native_declaration_identity_preimage != result.native_declaration_bytes
    restored = profile.decode_native_formalization_profile(result.transport_bytes, expected_sha256=result.transport_sha256)
    assert restored.transport_bytes == result.transport_bytes
    assert restored.to_dict() == wire
    assert declaration.to_dict() == original_declaration and artifact.to_dict() == original_artifact
    assert result.status == "unsupported_for_operational_admission"
    assert wire["authority"]["owner_verification_required"] is True
    assert all(value is False for key, value in wire["authority"].items() if key != "owner_verification_required")
    assert all(Path(pin["path"]).resolve().is_relative_to(profile.NATIVE_ROOT) for pin in wire["serializer_source_pins"].values())
    assert all(sha(Path(pin["path"]).read_bytes()) == pin["sha256"] for pin in wire["serializer_source_pins"].values())
    with pytest.raises(FrozenInstanceError):
        result._transport_bytes = b"changed"
    wire["native_formalization"]["metadata"]["changed"] = True
    assert "changed" not in result.to_dict()["native_formalization"]["metadata"]


def test_intent_confidence_text_control_effect_and_inferred_fields_are_not_compacted():
    declaration, artifact = intent_case()
    wire = profile.NativeFormalizationProfile(declaration, artifact).to_dict()
    statements = {row["statement_id"]: row for row in wire["native_declaration"]["statements"]}
    assert statements["goal"]["confidence"] == 0.75
    assert statements["goal"]["arguments"] == ["result", "agent"]
    assert statements["verification"]["grounding"] == "inferred"
    assert {edge["kind"] for edge in wire["native_declaration"]["control_edges"]} == {"on_failure", "retry"}
    assert wire["native_formalization"]["metadata"]["observation"] == {
        "body": "retained body", "text": "retained text", "payload": {"confidence": 0.75}}
    legacy = [gap for gap in wire["unsupported_frontier"] if gap["code"] == "legacy_formula_expression"]
    assert len(legacy) == len(artifact.formulas)
    assert any(row.expression["kind"] == "typed_fact" and not row.is_typed for row in artifact.formulas)
    assert any(gap["code"] == "independent_action_effect_binding_required" for gap in wire["unsupported_frontier"])


def test_opaque_predicate_and_exact_native_diagnostic_paths_remain_unsupported():
    declaration, artifact = intent_case(opaque=True)
    wire = profile.NativeFormalizationProfile(declaration, artifact).to_dict()
    assert any(gap["code"] == "native_predicate_missing" for gap in wire["unsupported_frontier"])
    assert any(gap["code"] == "opaque_formula" for gap in wire["unsupported_frontier"])
    for gap in wire["unsupported_frontier"]:
        if gap["code"] == "native_compiler_diagnostic":
            index = int(gap["path"].rsplit("/", 1)[1])
            native = wire["native_formalization"]["diagnostics"]["diagnostics"][index]
            assert native["code"] == gap["native_code"]
            assert native["location"]["field_path"] == gap["native_field_path"]


def test_security_source_descriptor_optional_extension_and_initial_state_do_not_gain_authority():
    declaration, artifact = security_case(missing_source=True, extension=True)
    wire = profile.NativeFormalizationProfile(declaration, artifact).to_dict()
    assert wire["native_declaration"]["sources"][0]["content_sha256"] == ""
    assert wire["native_declaration"]["state_machines"][0]["initial_state"] == "pending"
    assert "current_state" not in wire["native_declaration"]["state_machines"][0]
    assert wire["native_declaration"]["extensions"][0]["payload"]["body"] == "retain optional semantics"
    codes = {gap["code"] for gap in wire["unsupported_frontier"]}
    assert {"original_source_digest_missing", "original_source_revision_missing",
            "source_descriptor_is_not_original_content", "native_extension_semantics_not_lowered",
            "native_policy_is_not_operational_grant", "independent_runtime_state_binding_required"} <= codes


@pytest.mark.parametrize("change", (
    lambda wire: wire["authority"].update(admitted=True),
    lambda wire: wire["byte_bindings"]["native_formalization_identity"].update(digest="sha256:" + "0" * 64),
    lambda wire: wire["native_formalization"].update(declaration_digest="sha256:" + "0" * 64),
    lambda wire: wire["native_declaration"].pop("tags"),
    lambda wire: wire["native_formalization"]["metadata"].update(extra_unbound_semantics="changed"),
    lambda wire: wire.update(unrecognized_profile_field=True),
))
def test_profile_tampering_missing_fields_and_repaired_identities_refuse(change):
    result = profile.NativeFormalizationProfile(*intent_case())
    wire = result.to_dict()
    change(wire)
    with pytest.raises(profile.NativeFormalizationProfileError):
        profile.decode_native_formalization_profile(encoded(wire))


@pytest.mark.parametrize("body", (b'{"schema":1,"schema":2}', b'{"nested":{"x":1,"x":2}}',
    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b"x" * (profile.MAX_BYTES + 1)),
    ids=("duplicate-top-level", "duplicate-nested", "nan", "infinity", "overflow", "oversize"))
def test_duplicate_nonfinite_and_oversize_json_refuses_before_native_reconstruction(body):
    with pytest.raises(profile.NativeFormalizationProfileError):
        profile.decode_native_formalization_profile(body)


def test_exact_concrete_native_objects_and_transport_hash_are_required():
    declaration, artifact = intent_case()
    with pytest.raises(profile.NativeFormalizationProfileError, match="actual pinned"):
        profile.NativeFormalizationProfile(declaration.to_dict(), artifact)
    with pytest.raises(profile.NativeFormalizationProfileError, match="actual pinned"):
        profile.NativeFormalizationProfile(declaration, artifact.to_dict())
    other_declaration = replace(declaration, document_id="intent:other-fixture")
    other_artifact = IntentFormalizationCompiler().compile_document(other_declaration)
    # Both artifacts pass their own native validation. The profile must reject
    # joining the original declaration to the other declaration's artifact.
    with pytest.raises(profile.NativeFormalizationProfileError, match="join mismatch"):
        profile.NativeFormalizationProfile(declaration, other_artifact)
    result = profile.NativeFormalizationProfile(declaration, artifact)
    with pytest.raises(profile.NativeFormalizationProfileError, match="transport identity"):
        profile.decode_native_formalization_profile(result.transport_bytes, expected_sha256="0" * 64)
    with pytest.raises(profile.NativeFormalizationProfileError, match="encoding"):
        profile.decode_native_formalization_profile(json.dumps(result.to_dict(), indent=2))


def test_preloaded_foreign_datasets_origin_refuses_before_any_serializer(monkeypatch):
    # Only an origin-contract control; this test never imports the HACC checkout.
    foreign = SimpleNamespace(__file__="/home/barberb/HACC/complaint-generator/ipfs_datasets_py/foreign.py")
    monkeypatch.setitem(sys.modules, "ipfs_datasets_py.foreign_profile_origin", foreign)
    with pytest.raises(profile.NativeFormalizationProfileError, match="pinned native tree"):
        profile.NativeFormalizationProfile({}, {})


def test_capture_source_guard_starts_before_first_serializer_and_checks_finally(monkeypatch):
    declaration, artifact = intent_case()
    original_pins, original_dict = profile._source_pins, IntentIRDocument.to_dict
    events = []

    def observed_dict(self):
        events.append("serialize")
        return original_dict(self)

    def observed_pins():
        events.append("pins")
        result = original_pins()
        if events.count("pins") > 1:
            first = next(iter(result))
            result[first] = {**result[first], "sha256": "0" * 64}
        return result

    # An observer-contract control, not edits to P or evidence of real drift.
    monkeypatch.setattr(IntentIRDocument, "to_dict", observed_dict)
    monkeypatch.setattr(profile, "_source_pins", observed_pins)
    with pytest.raises(profile.NativeFormalizationProfileError, match="source closure changed"):
        profile.NativeFormalizationProfile(declaration, artifact)
    assert events[0] == "pins" and events[-1] == "pins"
    assert "serialize" in events[1:-1]


def test_native_oversize_payload_is_refused_without_truncation():
    declaration, artifact = intent_case()
    oversized = replace(artifact, metadata={"body": "x" * profile.MAX_BYTES})
    with pytest.raises(profile.NativeFormalizationProfileError, match="2 MiB"):
        profile.NativeFormalizationProfile(declaration, oversized)
    assert oversized.metadata["body"] == "x" * profile.MAX_BYTES


def test_legacy_json_cannot_be_relabelled_as_typed_expression():
    result = profile.NativeFormalizationProfile(*intent_case())
    wire = result.to_dict()
    wire["native_formalization"]["formulas"][0]["expression"] = {
        "interface": "TypedExpression@1", "schema_version": "typed-expression/v1",
        "body": "not a validated native AST"}
    with pytest.raises(profile.NativeFormalizationProfileError):
        profile.decode_native_formalization_profile(encoded(wire))
