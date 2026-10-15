"""Historical whole-dialogue input replay, with one real producer fixture.

The callbacks and token counts are authored offline observations. These tests
establish input reconstruction and count labeling, never provider usage,
task success, remote tokenizer compatibility, or actual token savings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

import benchmarks.agent_supervisor.container_coding.semantic_decision_input_audit as audit
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue as dialogue
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_worker as worker
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter,
    render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
import test.api.test_semantic_router_integration as integration_fixtures
import test.api.test_semantic_router_translation as translation_fixtures


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _counter(*, fallback=False):
    def count(text):
        if fallback:
            return 100
        if "\nSupervisor verified capsule deliveries:\n" in text:
            return 1200
        return 100 if dialogue.ROUND_SCHEMA in text else 1000

    return PromptTokenCounter(
        provider="codex_cli", model="pinned", tokenizer_id="authored-audit-counter",
        tokenizer_revision="fixture@1", special_token_policy="authored literal count",
        count_tokens=count, count_kind="estimate", compatibility_basis="authored_fixture_only",
    )


def _provider_receipt(case, call, *, failed=False):
    """Author bound wrapper/native receipts without executing any provider."""
    rendered, advisory = render_model_prompt(prompt=call.prompt, purpose="coding", workspace=case.workspace)
    native = {
        "schema": "router-implementation-invocation@1", "invocation_id": call.call_id,
        "provider": call.provider, "model": call.model, "purpose": "coding",
        "workspace": str(case.workspace), "status": "failed" if failed else "provider_returned",
        "prompt_sha256": _sha(call.prompt), "prompt_bytes": len(call.prompt.encode()),
        "native_prompt_sha256": _sha(call.prompt), "native_prompt_bytes": len(call.prompt.encode()),
        "router_prompt_sha256": _sha(call.prompt), "router_prompt_bytes": len(call.prompt.encode()),
        "model_prompt_sha256": _sha(rendered), "model_prompt_bytes": len(rendered.encode()),
        "workspace_advisory_sha256": _sha(advisory), "workspace_advisory_bytes": len(advisory.encode()),
        "provider_called": True, "completion_authority": False, "usage": None,
    }
    return {
        "schema": "router-semantic-decision-provider@1", "call_id": call.call_id,
        "session_id": call.session_id, "generation": call.generation,
        "binding_sha256": call.binding_sha256, "provider": call.provider, "model": call.model,
        "prompt_sha256": _sha(call.prompt), "prompt_bytes": len(call.prompt.encode()),
        "model_prompt_sha256": _sha(rendered), "model_prompt_bytes": len(rendered.encode()),
        "workspace": str(case.workspace), "source_repository": str(case.repository),
        "provider_called": True, "status": "failed" if failed else "provider_returned",
        "native_invocation": native, "completion_authority": False,
    }


def _dialogue_receipt(case, *, fallback=False, prepared_only=False, second_failure=False):
    calls = []

    def invoke(call):
        calls.append(call)
        child = _provider_receipt(case, call, failed=second_failure and len(calls) == 2)
        if second_failure and len(calls) == 2:
            raise dialogue.SemanticDecisionProviderFailure("AUTHORED-SENSITIVE-FAILURE-DETAIL", receipt=child)
        if fallback:
            response = "Literal native answer: café and spaces.  \r\n"
        else:
            response = {"session_id": call.session_id, "generation": call.generation,
                        "binding_sha256": call.binding_sha256}
            response.update(
                {"schema": dialogue.REQUEST_SCHEMA, "capsule_ids": [case.ids[0]]}
                if len(calls) == 1 else
                {"schema": dialogue.FINAL_SCHEMA, "text": "Literal final café response.  \r\n"}
            )
            response = _encoded(response)
        return dialogue.SemanticDecisionReply(response, child)

    try:
        result = dialogue.run_semantic_decision_dialogue(
            native_prompt=case.prompt, repository=case.repository, workspace=case.workspace,
            policy=case.policy, provider="codex_cli", model="pinned", invoke=invoke,
            count_tokens=_counter(fallback=fallback), max_calls=3,
            max_total_input_tokens=150 if prepared_only else None,
        )
        receipt = result.receipt
    except dialogue.SemanticDecisionDialogueError as error:
        if not (prepared_only or second_failure):
            raise
        receipt = error.receipt
    receipt["worker_launch"] = copy.deepcopy(case.launch)
    return receipt, tuple(calls)


@pytest.fixture(scope="module")
def audit_case(tmp_path_factory):
    """Create the producer/worktree once; no imported fixture is recollected."""
    temporary = tmp_path_factory.mktemp("decision-input-audit")
    native = translation_fixtures.native.__wrapped__(temporary)
    patch = pytest.MonkeyPatch()
    try:
        repository, workspace, prompt, source = integration_fixtures.allocated.__wrapped__(native, patch)
    finally:
        patch.undo()
    repository.chmod(0o755)
    (repository / ".runtime").chmod(0o755)
    workspace.chmod(0o755)
    artifact = repository / ".runtime/semantic/worker-context.json"
    parent = json.loads(artifact.read_text())
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    # Native task requirements must survive exact replay, including UTF-8 byte
    # counts, CRLF and significant trailing spaces outside the JSON capsule.
    literal = "\r\nRequired literal café λ instruction: preserve two spaces.  \r\n"
    prompt += literal
    policy = SemanticDecisionPolicy(required_capsule_ids=ids[1:], optional_capsule_ids=ids[:1])
    descriptor = worker.build_decision_launch_policy(native_prompt=prompt, policy=policy, repository=repository)
    policy_artifact = temporary / "owner-selection.json"
    policy_artifact.write_bytes(_encoded(descriptor).encode())
    policy_artifact.chmod(0o444)
    launch = {
        "launch_policy_artifact": str(policy_artifact), "launch_policy_sha256": _sha(policy_artifact.read_bytes()),
        "original_native_prompt_sha256": _sha(prompt), "original_native_prompt_bytes": len(prompt.encode()),
        "required_suffix_sha256": _sha(b""), "required_suffix_bytes": 0,
    }
    case = SimpleNamespace(
        repository=repository, workspace=workspace, workspace_root=temporary,
        prompt=prompt, source=source, parent=parent, artifact=artifact, policy=policy,
        policy_artifact=policy_artifact, descriptor=descriptor, ids=ids, literal=literal, launch=launch,
    )
    case.expanded, case.expanded_calls = _dialogue_receipt(case)
    case.fallback, case.fallback_calls = _dialogue_receipt(case, fallback=True)
    case.prepared, case.prepared_calls = _dialogue_receipt(case, prepared_only=True)
    case.failed, case.failed_calls = _dialogue_receipt(case, second_failure=True)
    assert len(case.expanded_calls) == len(case.failed_calls) == 2
    assert len(case.fallback_calls) == len(case.prepared_calls) == 1
    return case


def _audit(case, receipt=None, **changes):
    kwargs = dict(native_prompt=case.prompt, dialogue=case.expanded if receipt is None else receipt,
                  repository=case.repository, workspace_root=case.workspace_root)
    kwargs.update(changes)
    return audit.audit_semantic_decision_dialogue(**kwargs)


def _assert_observation_only(result):
    assert result["schema"] == "terminal-semantic-decision-dialogue-input-audit@1"
    assert result["provider_calls"] == 0
    assert result["freshness_revalidated"] is False
    assert result["completion_authority"] is False
    assert result["token_savings_established"] is False
    assert result["raw_prompts_exported"] is False
    assert "AUTHORED-SENSITIVE-FAILURE-DETAIL" not in _encoded(result)


def _assert_rounds_verified(result, count):
    _assert_observation_only(result)
    assert result["status"] == "verified"
    assert result["original_native_input_verified"] is True
    assert result["all_recorded_round_inputs_verified"] is True
    assert result["recorded_round_count"] == result["verified_round_count"] == count
    assert len(result["attempts"]) == count
    assert all(row["status"] == "verified" for row in result["attempts"])


def test_real_two_round_dialogue_reconstructs_every_rendered_input(audit_case):
    before = _encoded(audit_case.expanded)
    result = _audit(audit_case)
    _assert_rounds_verified(result, 2)
    assert _encoded(audit_case.expanded) == before
    for observed, retained, call in zip(result["attempts"], audit_case.expanded["attempts"], audit_case.expanded_calls):
        assert observed["call_id_sha256"] == _sha(call.call_id)
        assert observed["generation"] == call.generation
        for field in ("prompt_sha256", "prompt_bytes", "model_prompt_sha256", "model_prompt_bytes"):
            assert observed[field] == retained[field]
        assert observed["native_invocation_id_sha256"] == _sha(call.call_id)
        assert observed["provider_dispatch_reported"] is True
        assert observed["native_receipt_input_binding_validated"] is True
    comparison = result["recorded_local_input_comparison"]
    assert comparison["native_content_tokens"] == 1000
    assert comparison["all_round_content_token_sum"] == 1300
    assert comparison["difference_from_single_native_input"] == 300
    assert comparison["count_kind"] == "estimate"
    assert comparison["tokenizer_compatibility_basis"] == "authored_fixture_only"
    assert comparison["historical_counts_recomputed"] is False
    assert comparison["provider_usage"] is False


def test_recorded_local_tokenizer_counts_keep_their_label_without_recounting(audit_case):
    # This is an authored historical labeling variant. No tokenizer is loaded
    # and these values do not establish compatibility with a remote model.
    receipt = copy.deepcopy(audit_case.expanded)
    receipt["token_count_kind"] = "tokenizer"
    receipt["tokenizer_compatibility_basis"] = "caller_declaration"
    receipt["total_input_budget_basis"] = "local_tokenizer_content_tokens"
    for attempt in receipt["attempts"]:
        attempt["content_token_count_kind"] = "tokenizer"
    result = _audit(audit_case, receipt)
    _assert_rounds_verified(result, 2)
    comparison = result["recorded_local_input_comparison"]
    assert comparison["count_kind"] == "tokenizer"
    assert comparison["tokenizer_compatibility_basis"] == "caller_declaration"
    assert comparison["all_round_content_token_sum"] == 1300
    assert comparison["historical_counts_recomputed"] is False
    assert comparison["provider_usage"] is False


def test_literal_unicode_crlf_required_suffix_is_replayed_without_export(audit_case):
    assert all(audit_case.literal in call.prompt for call in audit_case.expanded_calls)
    result = _audit(audit_case)
    _assert_rounds_verified(result, 2)
    assert audit_case.literal not in _encoded(result)
    assert "Required literal" not in _encoded(result)
    assert any(row["prompt_bytes"] != len(call.prompt)
               for row, call in zip(result["attempts"], audit_case.expanded_calls))
    altered = audit_case.prompt.replace("café λ", "cafe lambda")
    refused = _audit(audit_case, native_prompt=altered)
    _assert_observation_only(refused)
    assert refused["status"] == "unverified"


def test_native_fallback_audits_actual_native_input_without_dialogue_protocol(audit_case):
    assert audit_case.fallback["selected_representation"] == "native"
    assert audit_case.fallback_calls[0].prompt == audit_case.prompt
    assert audit_case.fallback["expansions"] == []
    result = _audit(audit_case, audit_case.fallback)
    _assert_rounds_verified(result, 1)
    comparison = result["recorded_local_input_comparison"]
    assert comparison["native_content_tokens"] == comparison["all_round_content_token_sum"] == 100
    assert comparison["difference_from_single_native_input"] == 0


def test_post_execution_source_edits_do_not_invalidate_historical_inputs(audit_case):
    source_path = audit_case.repository / "mod.py"
    original = source_path.read_bytes()
    try:
        source_path.write_bytes(original + b"\n# Authored completed candidate edit.\n")
        result = _audit(audit_case)
        _assert_rounds_verified(result, 2)
    finally:
        source_path.write_bytes(original)


def test_removed_allocated_worktree_is_only_a_recorded_input_identity(audit_case):
    moved = audit_case.workspace.with_name("retained-worktree-bytes")
    audit_case.workspace.rename(moved)
    try:
        assert not audit_case.workspace.exists()
        result = _audit(audit_case)
        _assert_rounds_verified(result, 2)
    finally:
        moved.rename(audit_case.workspace)


def test_retained_producer_blocks_remain_authenticated_after_source_edits(audit_case):
    source_path = audit_case.repository / "mod.py"
    block = audit_case.artifact.parent / "blocks" / audit_case.parent["manifest"]["mod.py"]["source_cid"]
    original_source, original_block = source_path.read_bytes(), block.read_bytes()
    try:
        source_path.write_bytes(original_source + b"\n# Historical freshness is deliberately not required.\n")
        block.write_bytes(original_block + b"\n# Forged retained producer block.\n")
        result = _audit(audit_case)
        _assert_observation_only(result)
        assert result["status"] == "unverified"
        assert result["all_recorded_round_inputs_verified"] is False
    finally:
        block.write_bytes(original_block)
        source_path.write_bytes(original_source)


def test_prepared_expansion_that_never_reached_an_attempt_is_not_charged_as_delivery(audit_case):
    receipt = audit_case.prepared
    assert receipt["status"] == "failed" and receipt["failure_reason"] == "total_input_budget_exceeded"
    assert len(receipt["expansions"]) == 1
    assert receipt["expansions"][0]["included_in_invocation_attempts"] == []
    result = _audit(audit_case, receipt)
    _assert_rounds_verified(result, 1)
    assert result["recorded_local_input_comparison"]["all_round_content_token_sum"] == 100


def test_second_provider_failure_retains_and_verifies_both_attempt_inputs(audit_case):
    receipt = audit_case.failed
    assert receipt["status"] == "failed" and receipt["attempts"][1]["status"] == "provider_failed"
    assert receipt["expansions"][0]["included_in_invocation_attempts"] == [receipt["attempts"][1]["call_id"]]
    result = _audit(audit_case, receipt)
    _assert_rounds_verified(result, 2)
    assert result["attempts"][1]["native_receipt_input_binding_validated"] is True
    assert result["recorded_local_input_comparison"]["all_round_content_token_sum"] == 1300


@pytest.mark.parametrize("alteration", ["policy_digest", "parent_digest", "packet_digest", "original_digest", "suffix_digest"])
def test_foreign_policy_parent_packet_or_launch_binding_refuses(audit_case, alteration):
    receipt = copy.deepcopy(audit_case.expanded)
    if alteration == "policy_digest":
        receipt["policy_sha256"] = "0" * 64
    elif alteration == "parent_digest":
        receipt["parent_sha256"] = "0" * 64
    elif alteration == "packet_digest":
        receipt["packet_prompt_sha256"] = "0" * 64
    elif alteration == "original_digest":
        receipt["worker_launch"]["original_native_prompt_sha256"] = "0" * 64
    else:
        receipt["worker_launch"]["required_suffix_sha256"] = "0" * 64
    result = _audit(audit_case, receipt)
    _assert_observation_only(result)
    assert result["status"] == "unverified"
    assert result["recorded_local_input_comparison"] is None


@pytest.mark.parametrize("alteration", ["raw_digest", "schema", "partition", "duplicate_field"])
def test_pinned_descriptor_bytes_and_closed_policy_schema_remain_required(audit_case, alteration):
    path = audit_case.policy_artifact
    original = path.read_bytes()
    receipt = copy.deepcopy(audit_case.expanded)
    descriptor = copy.deepcopy(audit_case.descriptor)
    if alteration == "raw_digest":
        replacement = original + b" "
    elif alteration == "schema":
        descriptor["schema"] = "foreign-policy@1"
        replacement = _encoded(descriptor).encode()
    elif alteration == "partition":
        descriptor["policy"]["optional_capsule_ids"] = []
        replacement = _encoded(descriptor).encode()
    else:
        replacement = b'{"schema":"shadow",' + original[1:]
    try:
        path.chmod(0o644)
        path.write_bytes(replacement)
        path.chmod(0o444)
        if alteration != "raw_digest":
            receipt["worker_launch"]["launch_policy_sha256"] = _sha(replacement)
        result = _audit(audit_case, receipt)
        _assert_observation_only(result)
        assert result["status"] == "unverified"
    finally:
        path.chmod(0o644)
        path.write_bytes(original)
        path.chmod(0o444)


@pytest.mark.parametrize("alteration", ["foreign_call", "duplicate_call", "missing_call", "generation", "descriptor_digest", "descriptor_bytes", "capsule"])
def test_inconsistent_expansion_delivery_schedule_cannot_verify(audit_case, alteration):
    receipt = copy.deepcopy(audit_case.expanded)
    expansion = receipt["expansions"][0]
    if alteration == "foreign_call":
        expansion["included_in_invocation_attempts"] = ["foreign-call"]
    elif alteration == "duplicate_call":
        expansion["included_in_invocation_attempts"] *= 2
    elif alteration == "missing_call":
        expansion["included_in_invocation_attempts"] = []
    elif alteration == "generation":
        expansion["after_generation"] = 1
    elif alteration == "descriptor_digest":
        expansion["descriptor_sha256"] = "0" * 64
    elif alteration == "descriptor_bytes":
        expansion["descriptor_bytes"] += 1
    else:
        expansion["capsule_ids"] = [audit_case.ids[1]]
    result = _audit(audit_case, receipt)
    _assert_observation_only(result)
    assert result["status"] == "unverified"
    assert result["all_recorded_round_inputs_verified"] is False


def test_one_changed_round_digest_preserves_only_the_verified_round_subtotal(audit_case):
    receipt = copy.deepcopy(audit_case.expanded)
    receipt["attempts"][1]["model_prompt_sha256"] = "0" * 64
    result = _audit(audit_case, receipt)
    _assert_observation_only(result)
    assert result["status"] == "partial"
    assert result["verified_round_count"] == 1 and result["recorded_round_count"] == 2
    assert result["attempts"][0]["status"] == "verified"
    assert result["attempts"][1]["status"] == "unverified"
    assert result["all_recorded_round_inputs_verified"] is False
    assert result["recorded_local_input_comparison"] is None


@pytest.mark.parametrize("alteration", ["sum", "native_bool", "attempt_bool", "mixed_kind", "negative", "basis_empty", "binding_model", "max_input_exceeded", "max_total_exceeded"])
def test_bad_recorded_count_algebra_does_not_destroy_byte_replay_or_claim_savings(audit_case, alteration):
    receipt = copy.deepcopy(audit_case.expanded)
    if alteration == "sum":
        receipt["local_input_content_token_sum"] += 1
    elif alteration == "native_bool":
        receipt["native_content_tokens"] = True
    elif alteration == "attempt_bool":
        receipt["attempts"][0]["content_tokens"] = True
    elif alteration == "mixed_kind":
        receipt["attempts"][1]["content_token_count_kind"] = "tokenizer"
    elif alteration == "negative":
        receipt["attempts"][1]["content_tokens"] = -1
    elif alteration == "basis_empty":
        receipt["tokenizer_compatibility_basis"] = ""
    elif alteration == "binding_model":
        receipt["tokenizer_binding"]["model"] = "foreign-model"
    elif alteration == "max_input_exceeded":
        receipt["max_input_tokens"] = 1100
    else:
        receipt["max_total_input_tokens"] = 1250
    result = _audit(audit_case, receipt)
    _assert_rounds_verified(result, 2)
    assert result["recorded_local_input_comparison"] is None


def test_foreign_native_provider_receipt_cannot_claim_native_input_binding(audit_case):
    receipt = copy.deepcopy(audit_case.expanded)
    receipt["attempts"][0]["provider_receipt"]["native_invocation"]["model_prompt_sha256"] = "0" * 64
    result = _audit(audit_case, receipt)
    _assert_rounds_verified(result, 2)
    assert result["attempts"][0]["native_receipt_input_binding_validated"] is False
    assert result["attempts"][1]["native_receipt_input_binding_validated"] is True
    assert "native_invocation_id_sha256" not in result["attempts"][0]


@pytest.mark.parametrize("alteration", ["schema", "missing_launch", "duplicate_call", "generation_bool", "too_many_rounds", "foreign_workspace"])
def test_malformed_dialogue_identity_or_round_population_abstains(audit_case, alteration):
    receipt = copy.deepcopy(audit_case.expanded)
    if alteration == "schema":
        receipt["schema"] = "unregistered-dialogue@1"
    elif alteration == "missing_launch":
        del receipt["worker_launch"]
    elif alteration == "duplicate_call":
        receipt["attempts"][1]["call_id"] = receipt["attempts"][0]["call_id"]
    elif alteration == "generation_bool":
        receipt["attempts"][0]["generation"] = False
    elif alteration == "too_many_rounds":
        receipt["attempts"] *= 3
        receipt["invocation_attempt_count"] = len(receipt["attempts"])
    else:
        receipt["workspace"] = str(audit_case.workspace_root.parent / "foreign-allocation")
    result = _audit(audit_case, receipt)
    _assert_observation_only(result)
    assert result["status"] == "unverified"
    assert result["recorded_local_input_comparison"] is None

