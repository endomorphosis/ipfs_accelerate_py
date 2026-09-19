"""DOEP-046 wrapper: model-based and temporal invariant tests exist and pass."""

from __future__ import annotations

from test.api.doep import test_control_plane_model as model_tests


def test_control_plane_model_module_exports_invariants() -> None:
    assert callable(model_tests.test_generation_never_decreases)
    assert callable(model_tests.test_live_owner_cannot_mint_successor)
    assert callable(model_tests.test_completed_cannot_be_rewritten_as_success)
    assert callable(model_tests.test_exclusive_owner_loss_then_same_generation_restart)


def test_generation_never_decreases() -> None:
    model_tests.test_generation_never_decreases()


def test_live_owner_cannot_mint_successor() -> None:
    model_tests.test_live_owner_cannot_mint_successor()


def test_completed_cannot_be_rewritten_as_success() -> None:
    model_tests.test_completed_cannot_be_rewritten_as_success()


def test_exclusive_owner_loss_then_same_generation_restart() -> None:
    model_tests.test_exclusive_owner_loss_then_same_generation_restart()
