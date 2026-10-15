"""Lossless provider projection through indexed public+Doctor continuation.

Reuse the signed native public-profile fixture and its two real router children.
Only provider output, host privilege identity, and public-check execution are
authored. Planning, indexed evidence, original requirement capture, continuation
admission, worker preflight, retention, and historical input replay stay native.
No paid model, hidden verifier, or live provider material is accessed.
"""
import copy
import hashlib
import json

from benchmarks.agent_supervisor.container_coding import terminal_candidate_input_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_prompt_projection as projection
from test.api import test_public_validation_continuation as continuation
from test.api.test_public_validation_continuation import successful_first  # noqa: F401


PROVIDER_ASSERTIONS = r'''
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_prompt_projection as projection
native_generate=generate
projected_inputs=[]
def generate(model_prompt,**kwargs):
    assert second
    original=json.loads(prompt)
    view=projection.project_candidate_prompt(prompt)
    presented=json.loads(view.provider_prompt)
    assert model_prompt.count(view.provider_prompt)==1 and prompt not in model_prompt
    assert projection.restore_candidate_prompt(view.provider_prompt,view.receipt)==prompt
    assert presented['original_candidate']=={key:value for key,value in original.items() if key!='native_prompt'}
    native_context,end=json.JSONDecoder().raw_decode(original['native_prompt'])
    assert presented['native_context']==native_context
    assert presented['native_suffix']==original['native_prompt'][end:]
    assert native_context['evidence']
    assert native_context['acceptance'] and native_context['scope'] and native_context['authority']
    assert original['original_public_instruction_block']
    assert original['doctor_advisory']==config['doctor_advisory']
    assert original['public_validation']['feedback']['status']=='failed'
    legacy_input=model_prompt.replace(view.provider_prompt,prompt,1)
    assert len(model_prompt.encode())<len(legacy_input.encode())
    projected_inputs.append(dict(view=view,legacy=legacy_input,model=model_prompt))
    return native_generate(model_prompt,**kwargs)
'''

RECEIPT_ASSERTIONS = r'''
assert len(projected_inputs)==1
observed=projected_inputs[0]
selected=projection.validate_selection_receipt(receipt['candidate_prompt_projection'])
assert selected['selected_mode']==projection.MODE
assert selected['view_receipt']==observed['view'].receipt
assert selected['native_complete_sha256']==hashlib.sha256(observed['legacy'].encode()).hexdigest()
assert selected['selected_complete_sha256']==hashlib.sha256(observed['model'].encode()).hexdigest()
assert selected['selected_complete_sha256']==receipt['model_prompt_sha256']
assert selected['selected_complete_bytes']==receipt['model_prompt_bytes']
assert selected['candidate_complete_bytes']<selected['native_complete_bytes']
assert receipt['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
assert receipt['candidate_round_dispatch']['provider_input_sha256']==receipt['prompt_sha256']
assert receipt['semantic_translation'] is None
assert selected['proof_authority'] is selected['execution_authority'] is selected['completion_authority'] is False
assert selected['provider_token_usage_verified'] is selected['complete_run_token_savings_claimed'] is False
'''


def _projected_child():
    child = continuation.PUBLIC_CHILD
    provider_marker = "llm_router.generate_text=generate"
    receipt_marker = "print(json.dumps(dict(status='second_returned'"
    assert child.count(provider_marker) == child.count(receipt_marker) == 1
    return child.replace(provider_marker, PROVIDER_ASSERTIONS + "\n" + provider_marker).replace(
        receipt_marker, RECEIPT_ASSERTIONS + "\n" + receipt_marker)


def test_indexed_public_doctor_projection_preserves_requirements_and_replays_after_cleanup(
        successful_first, monkeypatch, record_property):
    # Fixture setup has already completed the first native router child. Only
    # its separately admitted second child receives these stronger assertions.
    monkeypatch.setattr(continuation, "PUBLIC_CHILD", _projected_child())
    replayed = []
    native_replay = audit.replay

    def replay(**kwargs):
        result = native_replay(**kwargs)
        replayed.append((dict(kwargs), result))
        return result

    monkeypatch.setattr(audit, "replay", replay)
    continuation.test_real_exit_zero_public_failure_admits_one_separate_router_child(
        successful_first, monkeypatch)
    assert len(replayed) == 1
    request, (model, advisory, checks) = replayed[0]
    receipt = request["receipt"]
    selected = projection.validate_selection_receipt(receipt["candidate_prompt_projection"])
    assert all(checks.values()), checks
    assert selected["selected_mode"] == projection.MODE
    assert selected["selected_complete_sha256"] == hashlib.sha256(model.encode()).hexdigest()
    assert selected["selected_complete_bytes"] < selected["native_complete_bytes"]
    assert successful_first.native.planning["provider_calls"] == 0
    assert successful_first.native.dispatch["provider_calls"] == 0
    assert successful_first.launch._state.reserved_attempts == {1, 2}

    # The reused lifecycle test has closed the candidate broker and destroyed
    # its live handoff. Historical replay must use retained original bytes,
    # even after a later edit to the allocated candidate source.
    target = successful_first.native.workspace / "eigen.py"
    target.write_bytes(target.read_bytes() + b"\n# later unrelated candidate edit\n")
    restored, restored_advisory, restored_checks = native_replay(**request)
    assert restored == model and restored_advisory == advisory
    assert all(restored_checks.values()), restored_checks

    # Rehashing a projection observation cannot alter the retained owner input
    # or authorize a stronger interpretation of the candidate representation.
    for mutation in ("authority", "selection", "digest", "original_digest", "drop_projection"):
        changed = copy.deepcopy(receipt)
        view = changed["candidate_prompt_projection"]
        if mutation == "authority":
            view["proof_authority"] = True
        elif mutation == "selection":
            view["selected_mode"] = "legacy"
        elif mutation == "digest":
            view["selected_complete_sha256"] = "f" * 64
        elif mutation == "original_digest":
            view["view_receipt"]["original_input_sha256"] = "e" * 64
            canonical = json.dumps(view["view_receipt"], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False)
            view["view_receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        else:
            del changed["candidate_prompt_projection"]
        try:
            _, _, changed_checks = native_replay(**dict(request, receipt=changed))
        except ValueError:
            continue
        assert not all(changed_checks.values()), mutation
    record_property("candidate_projection_native", json.dumps(dict(
        schema="authored-native-candidate-projection-qualification@1",
        selected_mode=selected["selected_mode"],
        native_complete_bytes=selected["native_complete_bytes"],
        candidate_complete_bytes=selected["candidate_complete_bytes"],
        selected_complete_bytes=selected["selected_complete_bytes"],
        representation_bytes_removed=selected["native_complete_bytes"] - selected["selected_complete_bytes"],
        original_requirements_exactly_restored=True,
        retained_input_audit_verified=all(checks.values()),
        retained_input_audit_verified_after_cleanup_and_source_edit=all(restored_checks.values()),
        forged_receipt_variants_refused=5, actual_router_children=2,
        live_provider_calls=0, public_check_execution_authored=True,
        provider_token_usage_verified=False, complete_run_token_savings_claimed=False,
        raw_content_exported=False), sort_keys=True))
