"""DOEP-104 wrapper for cross-supervisor isolation tests."""

from __future__ import annotations

from test.api.doep import test_cross_supervisor_isolation as isolation


def test_cross_supervisor_isolation_module_exists() -> None:
    assert callable(isolation.test_sawm_owner_cannot_write_doep_or_spar)
    assert callable(isolation.test_doep_owner_cannot_write_sawm)


def test_sawm_owner_cannot_write_doep_or_spar() -> None:
    isolation.test_sawm_owner_cannot_write_doep_or_spar()


def test_doep_owner_cannot_write_sawm() -> None:
    isolation.test_doep_owner_cannot_write_sawm()
