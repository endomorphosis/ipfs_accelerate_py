"""Controlled tests for the mini semantic fixture package."""

from pkg.mod import fn, helper


def test_fn() -> None:
    assert fn(1) == 2


def test_helper() -> None:
    assert helper(3) == 6


def test_deliberately_fails() -> None:
    # Seeded failure used only when the evaluation recipe expects it.
    # Default mini-repo keeps this xfail-like skip via env override in tests;
    # when run plain it fails so full-suite oracle can observe it.
    assert False, "seeded deliberate failure"
