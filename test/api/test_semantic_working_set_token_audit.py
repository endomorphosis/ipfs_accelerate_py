"""Historical bytes and scoped recount of real native working-set inputs.

Native capsules, registered socket and original router run normally. Boundary
admission, provider replies and token counts are explicit authored fixtures.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import _session
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set_dispatch as active
from test.api.test_semantic_working_set_token_dispatch import (  # noqa: F401
    dispatch, evidence_service, allocated, native, provider, token_counter, options,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_signed_working_set_closing import working, worker  # noqa: F401


WINNERS = [("raw_native", (100, 700, 500)), ("semantic_full", (1000, 100, 500)),
           ("working_set", (1000, 700, 100))]


def capture(dispatch, counts, **changes):
    service, (_, calls), args = dispatch
    counter, _ = token_counter(counts=counts, **changes)
    _, receipt = runner.run(**options(args, counter))
    receipt["phase"] = "coding"
    return service, calls[-1][0], receipt, counter


def replay(service, receipt, counter=None):
    parsed = audit._receipt(receipt, service.workspace.parent)
    return audit._model_projection(rendered=service.prompt, receipt=parsed,
        repository=service.repository, token_counter=counter)


@pytest.mark.parametrize("winner,counts", WINNERS)
@pytest.mark.parametrize("kind", ["tokenizer", "estimate"])
def test_reconstructs_all_three_winners_with_explicit_counter(dispatch, winner, counts, kind):
    service, delivered, receipt, counter = capture(dispatch, counts, count_kind=kind)
    model, _, checks = replay(service, receipt, counter)
    assert model == delivered and all(checks.values()), checks
    assert checks["token_working_set_token_winner_recounted"] is True
    assert receipt["semantic_working_set_dispatch"]["selection"]["historical_replay"] is False
    assert receipt["semantic_working_set_dispatch"]["selected_representation"] == winner
    assert receipt["semantic_working_set_dispatch"]["selection"]["recommendation_only"] is True
    assert receipt["semantic_working_set_dispatch"]["selection"]["dispatch_usable"] is False


@pytest.mark.parametrize("winner,counts", WINNERS)
def test_default_replay_verifies_bytes_without_promoting_recorded_token_winner(dispatch, winner, counts):
    service, delivered, receipt, counter = capture(dispatch, counts)
    model, _, checks = replay(service, receipt)
    assert model == delivered and checks["model_prompt_sha256"] is True
    assert checks["token_working_set_declared_selection"] is True
    assert checks["token_working_set_candidate_inputs"] is True
    assert checks["token_working_set_token_winner_recounted"] is None
    assert not all(checks.values())
    wrong = replace(counter, tokenizer_revision="different-revision")
    assert replay(service, receipt, wrong)[2]["token_working_set_token_winner_recounted"] is None


@pytest.mark.parametrize("winner,counts", WINNERS)
def test_closed_source_session_joins_evaluated_raw_or_selected_semantic_lineage(dispatch, winner, counts):
    service, _, receipt, _ = capture(dispatch, counts)
    service.close()
    raw = service.audit_path.read_bytes()
    observed, _, _ = _session(raw, receipt, {"task_alias": service.descriptor["task_id"]},
        expected_semantic_working_set=active.PROFILE)
    assert observed["semantic_working_set_representation"] == winner
    assert observed["source_guidance_delivered"] is (winner != "raw_native")
    assert observed["evaluated_source_lineage_only"] is (winner == "raw_native")
    assert observed["counts"]["delivered_source_bytes"] == 0
    assert observed["semantic_evidence_counts"]["delivered_pages"] == (1 if winner == "working_set" else 0)
    altered = deepcopy(receipt)
    source = (altered["semantic_working_set_evaluated"]["source_view_retrieval"] if winner == "raw_native"
              else altered["source_view_retrieval"])
    source["descriptor_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _session(raw, altered, {"task_alias": service.descriptor["task_id"]},
            expected_semantic_working_set=active.PROFILE)
    with pytest.raises(ValueError):
        _session(b"{}", receipt, {"task_alias": service.descriptor["task_id"]},
            expected_semantic_working_set=active.PROFILE)


@pytest.mark.parametrize("change", ["profile", "selected_hash", "selected_bytes", "provider", "descriptor", "missing_lineage",
    "active_translation", "active_source", "preflight", "metadata", "recommendation_as_dispatch"])
def test_raw_reader_refuses_aliases_foreign_bindings_and_active_delivery_claims(dispatch, change):
    service, _, receipt, _ = capture(dispatch, WINNERS[0][1])
    altered = deepcopy(receipt)
    dispatch_receipt = altered["semantic_working_set_dispatch"]
    if change == "profile": dispatch_receipt["requested_profile"] = "capsule-references@1"
    elif change == "selected_hash": dispatch_receipt["selected_model_prompt_sha256"] = "0" * 64
    elif change == "selected_bytes": dispatch_receipt["selected_router_prompt_bytes"] = True
    elif change == "provider": altered["provider"] = "grok_cli"
    elif change == "descriptor": altered["semantic_working_set_evaluated"]["source_view_retrieval"]["session_id"] = "f" * 32
    elif change == "missing_lineage": altered.pop("semantic_working_set_evaluated")
    elif change == "active_translation": altered["semantic_translation"] = altered["semantic_working_set_evaluated"]["semantic_translation"]
    elif change == "active_source": altered["source_view_retrieval"] = altered["semantic_working_set_evaluated"]["source_view_retrieval"]
    elif change == "preflight": altered["semantic_working_set_preflight"] = {}
    elif change == "metadata": altered["semantic_metadata_view"] = {}
    else: altered["semantic_working_set_dispatch"] = dispatch_receipt["selection"]
    with pytest.raises((ValueError, KeyError, TypeError)):
        audit._receipt(altered, service.workspace.parent)


def test_independent_reconstruction_detects_unselected_raw_candidate_hash_tampering(dispatch):
    service, _, receipt, counter = capture(dispatch, WINNERS[1][1])
    altered = deepcopy(receipt)
    altered["semantic_working_set_dispatch"]["selection"]["candidates"]["raw_native"]["model_prompt_sha256"] = "0" * 64
    _, _, checks = replay(service, altered, counter)
    assert checks["token_working_set_candidate_inputs"] is False and not all(checks.values())


def test_counter_mutation_and_recount_disagreement_do_not_qualify_tokens(dispatch):
    service, _, receipt, counter = capture(dispatch, WINNERS[1][1])
    original = counter.count_tokens
    def mutate(text):
        object.__setattr__(counter, "tokenizer_revision", "changed-inside-counter")
        return original(text)
    object.__setattr__(counter, "count_tokens", mutate)
    assert replay(service, receipt, counter)[2]["token_working_set_token_winner_recounted"] is False
    stable, _ = token_counter(counts=WINNERS[1][1])
    calls = []
    original = stable.count_tokens
    def drift(text):
        calls.append(text)
        return original(text) + (text in calls[:-1])
    object.__setattr__(stable, "count_tokens", drift)
    assert replay(service, receipt, stable)[2]["token_working_set_token_winner_recounted"] is False


def test_existing_automatic_resolver_recounts_only_matching_full_provenance(dispatch, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer as tokenizer
    service, (_, calls), args = dispatch
    # Use the real offline resolver. Its locally available tokenizer or labelled
    # fallback remains the recorded quality; no asset or model is downloaded.
    _, receipt = runner.run(**(args | dict(semantic_working_set=active.PROFILE)))
    receipt["phase"] = "coding"
    assert receipt["prompt_tokenizer_source"]["offline"] is True
    model, _, checks = replay(service, receipt)
    assert model == calls[-1][0] and all(checks.values()), checks
    original = tokenizer.resolve_prompt_tokenizer
    def different_provenance(**kwargs):
        counter, provenance = original(**kwargs)
        return counter, provenance | {"reason": "authored-different-provenance"}
    monkeypatch.setattr(tokenizer, "resolve_prompt_tokenizer", different_provenance)
    assert replay(service, receipt)[2]["token_working_set_token_winner_recounted"] is None


@pytest.mark.parametrize("winner,counts", WINNERS)
def test_selected_reply_contract_and_native_schema_are_reconstructed(dispatch, winner, counts):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    from test.api.test_semantic_working_set_token_dispatch import canonical, digest
    service, (state, calls), args = dispatch
    counter, _ = token_counter(counts=counts)
    state["reply"] = '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
    def schema_observed():
        literal = canonical(calls[-1][1]["codex_output_schema"])
        set_last_cli_observation("codex_cli", {**get_last_cli_observation("codex_cli"),
            "codex_output_schema_sha256": digest(literal), "codex_output_schema_bytes": len(literal.encode())})
    state["after_generate"] = schema_observed
    _, receipt = runner.run(**options(args, counter), coding_reply_mode="ordinary-completion@1")
    receipt["phase"] = "coding"
    assert receipt["coding_reply_contract"]["response_validated"] is True
    model, _, checks = replay(service, receipt, counter)
    assert model == calls[-1][0] and all(checks.values()), checks
    altered = deepcopy(receipt)
    altered["semantic_working_set_dispatch"]["selected_reply_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        audit._receipt(altered, service.workspace.parent)


def test_signed_raw_scope_retains_literal_instruction_and_historical_reconstruction(worker, working, provider):
    from benchmarks.agent_supervisor.container_coding.test_terminal_signed_working_set_closing import invoke
    counter, _ = token_counter(counts=WINNERS[0][1])
    _, receipt = invoke(worker, **(working.options | dict(semantic_working_set=active.PROFILE,
        prompt_token_counter=counter)))
    receipt["phase"] = "coding"
    assert receipt["semantic_translation"] is None
    assert receipt["public_instruction"]["schema"] == "supervisor-public-instruction-inclusion@5"
    assert receipt["terminal_semantic_scope_dispatch_rechecked"] is True
    parsed = audit._receipt(receipt, worker.workspace.parent)
    model, _, checks = audit._model_projection(rendered=worker.prompt, receipt=parsed,
        repository=worker.case.root, token_counter=counter)
    assert sha256(model.encode()).hexdigest() == receipt["model_prompt_sha256"]
    assert all(checks.values()), checks
