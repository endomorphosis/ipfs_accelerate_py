"""Every-epoch replay on one authored synthetic scratch-head unit fixture.

This tiny corpus and its one 16-epoch fit are unit evidence only. They contain
no official data and provide no benchmark score or convergence qualification.
"""
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_corpus as corpus
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_ranker_training as head
from benchmarks.agent_supervisor.container_coding import terminal_codebase_ranker_trace_authentication as api
from benchmarks.agent_supervisor.container_coding.terminal_codebase_intent_training_join import _digest
from test.api.test_terminal_codebase_intent_corpus import make_inputs


@pytest.fixture(scope="module")
def authenticated_fixture():
    originals = make_inputs()
    frozen = corpus.build_terminal_intent_relevance_corpus(**originals)
    fitted = head.train_terminal_codebase_intent_ranker(
        corpus_receipt=frozen, original_inputs=originals, epochs=16)
    arguments = _arguments(originals, frozen, fitted)
    authentication = api.authenticate_terminal_ranker_training_trace(fitted, **arguments)
    return originals, frozen, fitted, authentication


def _arguments(originals, frozen, fitted):
    return {"corpus_receipt": frozen, "original_inputs": originals,
        "expected_checkpoint_sha256": _digest(fitted["checkpoint"]),
        "expected_training_receipt_sha256": fitted["training_receipt"]["receipt_sha256"],
        "expected_ranker_result_sha256": _digest(fitted)}


def _reseal_ranker(fitted):
    training = fitted["training_receipt"]
    training["receipt_sha256"] = _digest({key: value for key, value in training.items()
        if key != "receipt_sha256"})
    fitted["result_sha256"] = _digest({key: value for key, value in fitted.items()
        if key != "result_sha256"})


def _old_validate(fitted, originals, frozen):
    return head.validate_terminal_codebase_intent_ranker(fitted,
        corpus_receipt=frozen, original_inputs=originals,
        expected_checkpoint_sha256=_digest(fitted["checkpoint"]),
        expected_training_receipt_sha256=fitted["training_receipt"]["receipt_sha256"])


def _forbid_gradient(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid inputs or an insufficient budget must precede gradient evaluation")
    monkeypatch.setattr(api, "_replay_observation", forbidden)
    monkeypatch.setattr(head, "_objective", forbidden)


class _PinSubclass(str):
    pass


class _BudgetSubclass(int):
    pass


@pytest.mark.parametrize("pin", ["expected_checkpoint_sha256",
    "expected_training_receipt_sha256", "expected_ranker_result_sha256"])
@pytest.mark.parametrize("value", [None, False, 0, "0" * 63, "A" * 64,
    _PinSubclass("0" * 64)])
def test_malformed_external_pins_refused_before_gradient(
        authenticated_fixture, monkeypatch, pin, value):
    originals, frozen, fitted, _ = authenticated_fixture
    arguments = _arguments(originals, frozen, fitted)
    arguments[pin] = value
    _forbid_gradient(monkeypatch)
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(fitted, **arguments)


@pytest.mark.parametrize("pin", ["expected_checkpoint_sha256",
    "expected_training_receipt_sha256", "expected_ranker_result_sha256"])
def test_foreign_external_pins_refused_before_gradient(authenticated_fixture, monkeypatch, pin):
    originals, frozen, fitted, _ = authenticated_fixture
    arguments = _arguments(originals, frozen, fitted)
    arguments[pin] = "0" * 64
    _forbid_gradient(monkeypatch)
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(fitted, **arguments)


@pytest.mark.parametrize("value", [None, True, [], {}])
def test_malformed_ranker_receipts_refused_before_gradient(
        authenticated_fixture, monkeypatch, value):
    originals, frozen, fitted, _ = authenticated_fixture
    arguments = _arguments(originals, frozen, fitted)
    arguments["expected_ranker_result_sha256"] = _digest(value)
    _forbid_gradient(monkeypatch)
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(value, **arguments)


@pytest.mark.parametrize("field", ["epochs", "epoch", "weight", "metric"])
def test_boolean_numeric_fields_refused_before_gradient(authenticated_fixture, monkeypatch, field):
    originals, frozen, fitted, _ = authenticated_fixture
    malformed = deepcopy(fitted)
    if field == "epochs":
        malformed["training_receipt"]["epochs"] = True
    elif field == "epoch":
        malformed["training_receipt"]["trace"][8]["epoch"] = True
    elif field == "weight":
        checkpoint = malformed["checkpoint"]
        checkpoint["weights"][0] = True
        checkpoint["weights_sha256"] = _digest(checkpoint["weights"])
        malformed["training_receipt"]["checkpoint_sha256"] = _digest(checkpoint)
    else:
        malformed["training_receipt"]["trace"][8]["gradient_norm"] = True
    _reseal_ranker(malformed)
    _forbid_gradient(monkeypatch)
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(malformed,
            **_arguments(originals, frozen, malformed))


@pytest.mark.parametrize("budget", [None, True, False, 0, -1, 1.0, 1_000_001,
    _BudgetSubclass(8080)])
def test_invalid_replay_budgets_refused_before_gradient(authenticated_fixture, monkeypatch, budget):
    originals, frozen, fitted, _ = authenticated_fixture
    _forbid_gradient(monkeypatch)
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(fitted,
            **_arguments(originals, frozen, fitted), max_coordinate_operations=budget)


def test_one_operation_short_budget_refused_before_gradient(authenticated_fixture, monkeypatch):
    originals, frozen, fitted, _ = authenticated_fixture
    assert len(fitted["training_receipt"]["train_pairs"]) == 1
    _forbid_gradient(monkeypatch)
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(fitted,
            **_arguments(originals, frozen, fitted), max_coordinate_operations=8079)


def test_genuine_receipt_replays_every_epoch_with_exact_call_counts(authenticated_fixture, monkeypatch):
    originals, frozen, fitted, authentication = authenticated_fixture
    calls = {"objective": 0, "replay": 0, "prepare": 0}
    objective = head._objective
    replay_observation = api._replay_observation
    prepare = head._prepare

    def observe(*args, **kwargs):
        calls["objective"] += 1
        return objective(*args, **kwargs)

    def replay(*args, **kwargs):
        calls["replay"] += 1
        return replay_observation(*args, **kwargs)

    def prepare_inputs(*args, **kwargs):
        calls["prepare"] += 1
        return prepare(*args, **kwargs)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("trace authentication must not perform a new training fit")

    monkeypatch.setattr(head, "_objective", observe)
    monkeypatch.setattr(api, "_replay_observation", replay)
    monkeypatch.setattr(head, "_prepare", prepare_inputs)
    monkeypatch.setattr(head, "train_terminal_codebase_intent_ranker", forbidden_fit)
    validated = api.validate_terminal_ranker_training_trace_authentication(authentication,
        receipt=fitted, **_arguments(originals, frozen, fitted))
    assert validated == authentication
    assert calls == {"objective": 19, "replay": 17, "prepare": 2}
    assert validated["endpoint_validation_gradient_evaluations"] == 2
    assert validated["replay_gradient_evaluations"] == 17
    assert validated["gradient_evaluations"] == 19
    assert validated["native_preparation_calls"] == 2
    assert validated["passive_validator_calls"] == 1
    assert validated["optimizer_replay_updates"] == 16
    assert validated["new_training_fit_calls"] == 0
    assert validated["checkpoint_activation_calls"] == 0
    assert validated["replay_coordinate_operations"] == 8080
    assert validated["checked_trace_rows"] == validated["checked_weight_state_digests"] == 17


def test_exact_replay_coordinate_budget_is_sufficient(authenticated_fixture):
    originals, frozen, fitted, _ = authenticated_fixture
    authenticated = api.authenticate_terminal_ranker_training_trace(fitted,
        **_arguments(originals, frozen, fitted), max_coordinate_operations=8080)
    assert authenticated["replay_coordinate_operations"] == 8080


def test_genuine_claims_bind_full_trace_states_and_preserve_unknowns(authenticated_fixture):
    _, frozen, fitted, authentication = authenticated_fixture
    trace = fitted["training_receipt"]["trace"]
    states = authentication["authenticated_states"]
    assert authentication["schema"] == api.SCHEMA
    assert authentication["status"] == "authenticated_finite_native_training_trace"
    assert authentication["corpus_sha256"] == frozen["corpus_sha256"]
    assert len(states) == 17
    assert all(set(state) == {"epoch", "weights_sha256", "gradient_sha256", "finite_state_sha256"}
        for state in states)
    assert [state["epoch"] for state in states] == list(range(17))
    assert [state["weights_sha256"] for state in states] == [row["weights_sha256"] for row in trace]
    assert authentication["initial_weights_sha256"] == _digest([0.0] * 80)
    assert authentication["final_weights_sha256"] == fitted["checkpoint"]["weights_sha256"]
    assert authentication["authenticated_states_sha256"] == _digest(states)
    assert authentication["authenticated_trace_sha256"] == _digest(trace)
    assert authentication["ranker_result_sha256"] == _digest(fitted)
    assert authentication["native_ranker_result_self_sha256"] == fitted["result_sha256"]
    representation = authentication["finite_state_digest_representation"]
    assert representation["stored_fields"] == ["epoch", "weights_sha256", "gradient_sha256",
        "finite_state_sha256"]
    assert representation["finite_state_digest_commits"] == ["epoch", "weights", "gradient", "observation"]
    assert representation["all_epoch_states_retained"] is True
    profile = authentication["native_update_profile"]
    assert profile["python_implementation"] == "cpython"
    assert profile["float_radix"] == 2 and profile["float_mantissa_bits"] == 53
    assert profile["summation"] == "CPython math.fsum"
    assert profile["nonlinear_functions"] == ["math.exp", "math.log1p", "math.sqrt"]
    assert authentication["native_update_profile_sha256"] == _digest(profile)
    assert all(authentication[field] is False for field in head._AUTHORITY)
    assert all(authentication[field] is False for field in (
        "independent_pin_origin_authenticated", "historical_training_provenance_authenticated_here",
        "historical_execution_origin_authenticated", "native_update_error_bound_proved",
        "binary64_error_bound_proved", "real_logistic_descent_proved", "asymptotic_convergence_proved",
        "global_optimizer_convergence_proved", "autoencoder_convergence_proved",
        "generalized_ranking_gain_qualified", "complete_prompt_interpretation_qualified"))
    assert all(profile[field] is False for field in (
        "platform_libm_identity_independently_pinned", "binary64_error_bound_proved",
        "math_fsum_error_bound_proved", "libm_error_bound_proved",
        "cross_platform_bitwise_reproducibility_proved"))
    assert authentication["full_task_satisfaction"] == "unknown"
    assert authentication["planning_handoff"] == "abstained"
    assert authentication["authentication_sha256"] == _digest({key: value
        for key, value in authentication.items() if key != "authentication_sha256"})


def test_forged_ranker_outer_selfdigest_refused_even_with_matching_full_object_pin(authenticated_fixture):
    originals, frozen, fitted, _ = authenticated_fixture
    forged = deepcopy(fitted)
    forged["result_sha256"] = "0" * 64
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(forged,
            **_arguments(originals, frozen, forged))


def test_resealed_negative_zero_endpoint_refused_by_exact_native_replay(authenticated_fixture):
    originals, frozen, fitted, _ = authenticated_fixture
    forged = deepcopy(fitted)
    assert forged["training_receipt"]["trace"][0]["L2_penalty"] == 0.0
    forged["training_receipt"]["trace"][0]["L2_penalty"] = -0.0
    _reseal_ranker(forged)
    # Python numeric equality treats both zeros alike; exact JSON digests do
    # not. The companion must bind the actual native representation.
    assert _old_validate(forged, originals, frozen) == forged
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(forged,
            **_arguments(originals, frozen, forged))


@pytest.mark.parametrize("field", ["objective", "pair_logistic_loss", "L2_penalty",
    "gradient_norm", "weights_sha256"])
def test_resealed_interior_history_accepted_by_endpoint_validator_is_refused(
        authenticated_fixture, field):
    originals, frozen, fitted, _ = authenticated_fixture
    forged = deepcopy(fitted)
    interior = forged["training_receipt"]["trace"][8]
    if field == "weights_sha256":
        interior[field] = "0" * 64
    else:
        interior[field] += 0.125
    _reseal_ranker(forged)
    # Supplying the forged pins deliberately removes pin mismatch as an
    # explanation: the old endpoint replay accepts this false history.
    assert _old_validate(forged, originals, frozen) == forged
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(forged,
            **_arguments(originals, frozen, forged))


def _reseal_authentication(receipt):
    receipt["authentication_sha256"] = _digest({key: value for key, value in receipt.items()
        if key != "authentication_sha256"})


def _alter_claim(value):
    if type(value) is bool:
        return not value
    if type(value) in (int, float):
        return value + 1
    raise AssertionError("test mutation must name a supported claim")


def _validate_authentication(receipt, fixture):
    originals, frozen, fitted, _ = fixture
    return api.validate_terminal_ranker_training_trace_authentication(receipt,
        receipt=fitted, **_arguments(originals, frozen, fitted))


@pytest.mark.parametrize("field", ["float_mantissa_bits", "libm_error_bound_proved"])
def test_resealed_native_profile_claim_is_replayed(authenticated_fixture, field):
    forged = deepcopy(authenticated_fixture[3])
    profile = forged["native_update_profile"]
    profile[field] = _alter_claim(profile[field])
    forged["native_update_profile_sha256"] = _digest(profile)
    _reseal_authentication(forged)
    with pytest.raises(api.RankerTraceAuthenticationError):
        _validate_authentication(forged, authenticated_fixture)


@pytest.mark.parametrize("field", ["gradient_evaluations", "optimizer_replay_updates"])
def test_resealed_operation_count_is_replayed(authenticated_fixture, field):
    forged = deepcopy(authenticated_fixture[3])
    forged[field] += 1
    _reseal_authentication(forged)
    with pytest.raises(api.RankerTraceAuthenticationError):
        _validate_authentication(forged, authenticated_fixture)


@pytest.mark.parametrize("field", ["hard_max_coordinate_operations", "coordinate_operations"])
def test_resealed_budget_count_is_replayed(authenticated_fixture, field):
    forged = deepcopy(authenticated_fixture[3])
    budget = forged["replay_budget"]
    budget[field] = _alter_claim(budget[field])
    _reseal_authentication(forged)
    with pytest.raises(api.RankerTraceAuthenticationError):
        _validate_authentication(forged, authenticated_fixture)


@pytest.mark.parametrize("field", ["epoch", "weights_sha256", "gradient_sha256", "finite_state_sha256"])
def test_every_resealed_intermediate_state_binding_is_replayed(authenticated_fixture, field):
    forged = deepcopy(authenticated_fixture[3])
    state = forged["authenticated_states"][8]
    state[field] = state[field] + 1 if field == "epoch" else "0" * 64
    forged["authenticated_states_sha256"] = _digest(forged["authenticated_states"])
    _reseal_authentication(forged)
    with pytest.raises(api.RankerTraceAuthenticationError):
        _validate_authentication(forged, authenticated_fixture)


@pytest.mark.parametrize("mutation", ["reverse", "truncate"])
def test_resealed_state_chain_order_and_completeness_are_replayed(authenticated_fixture, mutation):
    forged = deepcopy(authenticated_fixture[3])
    if mutation == "reverse":
        forged["authenticated_states"].reverse()
    else:
        forged["authenticated_states"].pop()
    forged["authenticated_states_sha256"] = _digest(forged["authenticated_states"])
    _reseal_authentication(forged)
    with pytest.raises(api.RankerTraceAuthenticationError):
        _validate_authentication(forged, authenticated_fixture)


@pytest.mark.parametrize("field", ["proof_authority", "global_optimizer_convergence_proved"])
def test_resealed_authority_or_convergence_promotion_is_refused(authenticated_fixture, field):
    forged = deepcopy(authenticated_fixture[3])
    forged[field] = True
    _reseal_authentication(forged)
    with pytest.raises(api.RankerTraceAuthenticationError):
        _validate_authentication(forged, authenticated_fixture)


def test_companion_outer_selfdigest_tampering_refused_before_gradient(authenticated_fixture, monkeypatch):
    forged = deepcopy(authenticated_fixture[3])
    forged["authentication_sha256"] = "0" * 64
    _forbid_gradient(monkeypatch)
    with pytest.raises(api.RankerTraceAuthenticationError):
        _validate_authentication(forged, authenticated_fixture)


def test_resealed_forged_checkpoint_with_matching_endpoint_is_refused(authenticated_fixture):
    originals, frozen, fitted, _ = authenticated_fixture
    forged = deepcopy(fitted)
    checkpoint = forged["checkpoint"]
    checkpoint["weights"][0] += 0.125
    checkpoint["weights_sha256"] = _digest(checkpoint["weights"])
    training = forged["training_receipt"]
    training["checkpoint_sha256"] = _digest(checkpoint)
    prepared = head._prepare(frozen, originals)
    observation, _ = head._objective(checkpoint["weights"], prepared[4])
    training["trace"][-1] = {"epoch": 16, **observation,
        "weights_sha256": checkpoint["weights_sha256"]}
    _reseal_ranker(forged)
    forged = head._output(prepared, checkpoint, training)
    assert _old_validate(forged, originals, frozen) == forged
    with pytest.raises(api.RankerTraceAuthenticationError):
        api.authenticate_terminal_ranker_training_trace(forged,
            **_arguments(originals, frozen, forged))
