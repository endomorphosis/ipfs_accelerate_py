"""A detached residual focuses evidence without shrinking the admitted task."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import ResidualLlmPacket
from ipfs_accelerate_py.agent_supervisor.runtime.repository_residual_context import (
    PreparedRepositoryResidualContext, RepositoryResidualContextError,
    prepare_repository_residual_context, replay_repository_residual_context,
    validate_repository_residual_context,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.residual_provider_invocation import (
    PathLease, ResidualProviderInvocation,
)


EXPECTED = dict(expected_task_id="task:residual", repository_id="repository:arithmetic",
    tree_id="sha256:current-source", policy_id="policy:bounded-task",
    policy_revision="sha256:current-policy", write_paths=("calc.py", "tests/test_calc.py"),
    validation_commands=("python -m pytest tests/test_calc.py", "python -m compileall calc.py"),
    acceptance_ids=("acceptance:compatible-callers", "acceptance:bounded-result"))


def packet(**changes):
    data = dict(task_id=EXPECTED["expected_task_id"], repository_id=EXPECTED["repository_id"],
        tree_id=EXPECTED["tree_id"], policy_id=EXPECTED["policy_id"],
        policy_revision=EXPECTED["policy_revision"], write_paths=EXPECTED["write_paths"],
        validation_commands=EXPECTED["validation_commands"], acceptance_ids=EXPECTED["acceptance_ids"],
        obligation_ids=("obligation:repair-overflow", "obligation:retain-callers"),
        counterexample_capsule={"schema": "authored-counterexample@1",
            "target_ids": ["symbol:calc.add"], "counterexamples": [{"counterexample_id": "cex:overflow",
                "summary": "An overflowing input fails the bounded result requirement.",
                "evidence_refs": ["sha256:retained-validation"]}], "redacted": True},
        forest_id="forest:current-task", codex_packet_ref="sha256:repair-candidate",
        transition_ref="sha256:failure-transition", producer_id="authored-residual-producer@1",
        authority_roots={"repository_id": EXPECTED["repository_id"], "tree_id": EXPECTED["tree_id"],
            "unadmitted_external_reference": "sha256:inert-root"})
    data.update(changes)
    # Direct existing contract construction avoids the seal helper's optional
    # metadata mirror. These tests need no database or provider owner.
    return ResidualLlmPacket(**data)


def prepare(residual=None, **changes):
    expected = dict(EXPECTED, **changes)
    return prepare_repository_residual_context(residual_packet=packet() if residual is None else residual,
                                               **expected)


def test_existing_renderer_is_reused_with_every_sealed_field_retained(monkeypatch):
    residual = packet()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider callback must never run")

    monkeypatch.setattr(ResidualProviderInvocation, "invoke", forbidden)
    binding = prepare(residual)
    existing = ResidualProviderInvocation(require_path_lease=True).prepare(residual,
        path_lease=PathLease(permitted_write_paths=EXPECTED["write_paths"]), base_env={})
    assert binding.prompt == existing.prompt_body
    assert binding.context == json.loads(existing.prompt_body)
    assert binding.packet == residual.to_dict()
    assert binding.context_id == existing.content_id
    assert binding.packet_id == residual.packet_id
    for field in ("forest_id", "codex_packet_ref", "transition_ref", "authority_roots",
                  "counterexample_capsule", "obligation_ids", "acceptance_ids", "limits"):
        assert binding.context[field] == binding.packet[field]
    assert binding.packet["producer_id"] == "authored-residual-producer@1"
    assert binding.context["provider_instructions"]
    assert binding.context["model_constraints"]["nomination_only"] is True


def test_receipt_is_content_free_scoped_and_does_not_claim_native_usage_or_authority():
    binding = prepare()
    receipt = binding.receipt
    assert receipt["context_sha256"] == hashlib.sha256(binding.prompt.encode()).hexdigest()
    assert receipt["context_bytes"] == len(binding.prompt.encode())
    assert receipt["packet_id"] == binding.packet_id
    assert receipt["residual_obligation_count"] == 2
    assert receipt["write_path_count"] == 2
    assert receipt["validation_count"] == 2
    assert receipt["acceptance_count"] == 2
    assert receipt["delivery_mode"] == "self_contained"
    assert receipt["nomination_only"] is True
    assert receipt["token_estimate"] == (receipt["context_bytes"] + 2) // 3
    assert receipt["provider_input_tokens"] is None
    assert receipt["token_estimate_scope"] == (
        "sealed_residual_context_only_excludes_full_task_prompt_and_wrappers")
    for field in ("provider_retained_parent", "provider_invoked", "proof_authority",
                  "semantic_authority", "write_authority", "execution_authority", "completion_authority",
                  "omission_authority", "source_semantics_verified", "world_proof_verified"):
        assert receipt[field] is False
    serialized = json.dumps(receipt)
    for text in ("calc.py", "repair-overflow", "overflowing input", "inert-root", "compileall"):
        assert text not in serialized


@pytest.mark.parametrize("field,value", [
    ("expected_task_id", "task:other"), ("repository_id", "repository:other"),
    ("tree_id", "sha256:other-source"), ("policy_id", "policy:other"),
    ("policy_revision", "sha256:other-policy"),
    ("write_paths", ("calc.py",)), ("write_paths", ("calc.py", "tests/test_calc.py", "extra.py")),
    ("write_paths", ("tests/test_calc.py", "calc.py")),
    ("validation_commands", ("python -m pytest tests/test_calc.py",)),
    ("validation_commands", ("python -m compileall calc.py", "python -m pytest tests/test_calc.py")),
    ("acceptance_ids", ("acceptance:compatible-callers",)),
    ("acceptance_ids", ("acceptance:compatible-callers", "acceptance:bounded-result", "acceptance:new")),
])
def test_native_identity_or_population_mismatch_refuses(field, value):
    with pytest.raises(RepositoryResidualContextError, match="mismatch"):
        prepare(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("expected_task_id", " task:residual"), ("repository_id", ""),
    ("policy_id", ""), ("policy_revision", ""), ("policy_id", None),
    ("write_paths", "calc.py"), ("write_paths", ()),
    ("write_paths", ("calc.py", "calc.py")), ("validation_commands", ()),
    ("acceptance_ids", (False,)),
])
def test_invalid_native_expectations_are_not_defaulted_or_normalized(field, value):
    with pytest.raises(RepositoryResidualContextError):
        prepare(**{field: value})


def test_empty_packet_policy_is_not_upgraded_from_native_expectations():
    with pytest.raises(RepositoryResidualContextError, match="policy_id_mismatch"):
        prepare(packet(policy_id=""))


def test_every_export_is_detached_and_original_nested_mutation_cannot_change_binding():
    residual = packet()
    binding = prepare(residual)
    original = binding.record
    residual.counterexample_capsule["target_ids"].append("symbol:new")
    binding.packet["counterexample_capsule"]["target_ids"].append("symbol:copy")
    binding.context["authority_roots"]["unexpected"] = "sha256:copy"
    binding.receipt["execution_authority"] = True
    binding.record["context"]["provider_instructions"].clear()
    assert binding.record == original
    assert validate_repository_residual_context(binding, **EXPECTED) == binding.receipt
    with pytest.raises(RepositoryResidualContextError, match="packet_invalid"):
        prepare(residual)
    with pytest.raises(FrozenInstanceError):
        binding._context_json = "{}"


@pytest.mark.parametrize("field,value", [
    ("semantic_authority", True), ("write_authority", True), ("completion_authority", True),
    ("nomination_only", False), ("packet_id", "sha256:forged"), ("task_id", " task:residual"),
])
def test_forged_typed_attributes_cannot_hide_behind_hardcoded_serialization(field, value):
    residual = packet()
    object.__setattr__(residual, field, value)
    with pytest.raises(RepositoryResidualContextError):
        prepare(residual)


def test_full_canonical_mapping_and_record_are_supported():
    residual = packet()
    assert prepare(residual.to_dict()).record == prepare(residual).record
    assert prepare(residual.to_record()).record == prepare(residual).record


@pytest.mark.parametrize("change", ["missing_policy", "unknown", "missing_id", "wrong_schema",
    "wrong_content_id", "null_content_id", "numeric_false", "unknown_limit", "unknown_body"])
def test_noncanonical_mappings_cannot_silently_drop_fields(change):
    value = packet().to_dict()
    if change == "missing_policy":
        del value["policy_id"]
    elif change == "unknown":
        value["unreviewed_field"] = "inert but unrecognized"
    elif change == "missing_id":
        del value["packet_id"]
    elif change == "wrong_schema":
        value["schema"] = "unrecognized@1"
    elif change == "wrong_content_id":
        value["content_id"] = "sha256:wrong"
    elif change == "null_content_id":
        value["content_id"] = None
    elif change == "numeric_false":
        value["write_authority"] = 0
    elif change == "unknown_limit":
        value["limits"]["unreviewed_limit"] = 12
    else:
        value["full_source"] = "must not be ignored"
    with pytest.raises(RepositoryResidualContextError):
        prepare(value)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object(), b"bytes"])
def test_non_literal_capsule_values_refuse_before_contract_coercion(invalid):
    value = packet().to_dict()
    value["counterexample_capsule"]["unreviewed_value"] = invalid
    with pytest.raises(RepositoryResidualContextError):
        prepare(value)


def test_cycles_and_excessive_nested_structures_are_bounded():
    value = packet().to_dict()
    value["counterexample_capsule"]["cycle"] = value
    with pytest.raises(RepositoryResidualContextError, match="structure_bound"):
        prepare(value)
    value = packet().to_dict()
    value["counterexample_capsule"]["many"] = ["small"] * 257
    with pytest.raises(RepositoryResidualContextError, match="sequence_bound"):
        prepare(value)


def test_existing_preparer_rejects_prompt_over_its_bound_even_if_packet_allows_more():
    residual = packet(counterexample_capsule={"target_ids": ["symbol:calc.add"],
        "observations": ["bounded candidate detail " + str(index) + "x" * 480 for index in range(70)]},
        max_bytes=65_536, max_tokens=22_000, max_capsule_bytes=60_000)
    with pytest.raises(RepositoryResidualContextError, match="context_preparation_refused"):
        prepare(residual)


def test_ambient_environment_does_not_change_detached_identity(monkeypatch):
    first = prepare()
    monkeypatch.setenv("PATH", "/a/different/path")
    monkeypatch.setenv("OPENAI_API_KEY", "must-never-be-read")
    assert prepare().record == first.record


def test_replay_reconstructs_without_restoring_native_owner_or_provider_authority():
    binding = prepare()
    replayed = replay_repository_residual_context(binding.record, **EXPECTED)
    assert replayed.record == binding.record
    assert replayed.receipt["source_owner_freshness"] == "not_revalidated_by_this_binding"
    assert replayed.receipt["provider_invoked"] is False
    assert replayed.receipt["execution_authority"] is False


@pytest.mark.parametrize("change", ["schema", "unknown", "packet", "context", "receipt", "count", "authority"])
def test_replay_recomputes_complete_record_and_refuses_any_tampering(change):
    record = prepare().record
    if change == "schema":
        record["schema"] = "wrong@1"
    elif change == "unknown":
        record["unknown"] = False
    elif change == "packet":
        record["packet"]["producer_id"] = "forged-producer@1"
    elif change == "context":
        record["context"]["provider_instructions"].append("Expand the task's authority.")
    elif change == "receipt":
        record["receipt"]["context_sha256"] = "0" * 64
    elif change == "count":
        record["receipt"]["residual_obligation_count"] = True
    else:
        record["receipt"]["execution_authority"] = True
    with pytest.raises(RepositoryResidualContextError):
        replay_repository_residual_context(record, **EXPECTED)


@pytest.mark.parametrize("field,value", [
    ("_context_json", "{}"), ("_receipt_json", "{}"), ("_packet_json", "invalid"),
    ("_context_json", "x" * 131_073), ("_context_json", None),
])
def test_validation_does_not_trust_frozen_or_directly_constructed_storage(field, value):
    binding = replace(prepare(), **{field: value})
    with pytest.raises(RepositoryResidualContextError):
        validate_repository_residual_context(binding, **EXPECTED)


def test_private_prompt_bytes_must_match_the_recomputed_pin_even_if_json_decodes_identically():
    original = prepare()
    for prompt in (original.prompt + "\n", original.prompt.replace(
            '"stage":"residual_llm_authorized"',
            '"stage":"untrusted-extra","stage":"residual_llm_authorized"')):
        assert json.loads(prompt) == original.context
        forged = replace(original, _context_json=prompt)
        with pytest.raises(RepositoryResidualContextError, match="storage_not_canonical"):
            validate_repository_residual_context(forged, **EXPECTED)


def test_errors_do_not_include_untrusted_input_or_underlying_exception_bodies():
    secret = "untrusted-private-body-marker"
    value = packet().to_dict()
    value["counterexample_capsule"]["full_source"] = secret
    with pytest.raises(RepositoryResidualContextError) as captured:
        prepare(value)
    assert secret not in str(captured.value)
    assert "full_source" not in str(captured.value)


def test_a_duck_typed_binding_has_no_privilege():
    class Impostor:
        record = prepare().record

    with pytest.raises(RepositoryResidualContextError, match="binding_type_invalid"):
        validate_repository_residual_context(Impostor(), **EXPECTED)
    assert PreparedRepositoryResidualContext is not Impostor
