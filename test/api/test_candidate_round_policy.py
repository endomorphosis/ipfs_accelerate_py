"""An explicit owner choice never changes the legacy worker's argument grammar."""

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.candidate_round_policy import (
    split_owner_candidate_rounds, validate_candidate_rounds,
)
from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv


@pytest.mark.parametrize("value", [None, True, False, 0, 3, -1, 2.0, "2"])
def test_only_exact_bounded_round_choice(value):
    with pytest.raises(ValueError):
        validate_candidate_rounds(value)


@pytest.mark.parametrize("argv", [
    ["--candidate-rounds"], ["--candidate-rounds", "0"],
    ["--candidate-rounds", "02"], ["--candidate-rounds="],
    ["--candidate-rounds=2", "--candidate-rounds", "2"],
    ["--candidate-rounds", "1", "--candidate-rounds=2"],
    ["--candidate-rounds", "--model", "x"], [False], ("--candidate-rounds", "2"),
])
def test_ambiguous_or_incomplete_round_control_refuses(argv):
    with pytest.raises(ValueError):
        split_owner_candidate_rounds(argv)


@pytest.mark.parametrize("selection,expected", [([], 1), (["--candidate-rounds", "1"], 1),
    (["--candidate-rounds", "2"], 2), (["--candidate-rounds=2"], 2)])
def test_round_control_does_not_reach_original_worker(selection, expected):
    original = ["--model", "authored", *selection, "--timeout", "90"]
    before = list(original)
    rounds, worker = split_owner_candidate_rounds(original)
    assert (rounds, worker) == (expected, ["--model", "authored", "--timeout", "90"])
    assert original == before


def _argv(**options):
    return implementation_argv(router=Path("/owner/router-worker"), model="gpt-6.1-sol",
        reasoning="high", timeout=90, semantic_repository=options.pop("semantic_repository", Path("/canonical")),
        doctor=options.pop("doctor", None), source_view_profile=options.pop("source_view_profile", "python-comments@1"),
        **options)


def test_signed_implementation_selects_two_rounds_only_explicitly():
    legacy = _argv()
    assert "--candidate-rounds" not in legacy
    selected = _argv(candidate_rounds=2)
    rounds, stripped = split_owner_candidate_rounds(selected)
    assert rounds == 2 and stripped == legacy


@pytest.mark.parametrize("options", [dict(semantic_repository=None), dict(source_view_profile="disabled"),
    dict(doctor={"route": "model_router", "status": "residual"}),
    dict(doctor={"route": "doctor_candidate", "status": "candidate_ready", "provider_calls": 0})])
def test_candidate_round_cannot_relax_other_dispatch_contracts(options):
    with pytest.raises(ValueError, match="ordinary semantic source coding"):
        _argv(candidate_rounds=2, **options)
