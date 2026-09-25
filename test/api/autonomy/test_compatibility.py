from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.compatibility import (
    COMPATIBILITY_RESPECTED,
    INIT_ORDER_INCOMPATIBLE,
    PICKLE_FALLBACK_NOT_IDENTITY,
    PLUGIN_INCOMPATIBLE,
    SILENT_PUBLIC_INCOMPATIBILITY,
    compatibility_view,
)


def test_missing_compatibility_payload_does_not_block() -> None:
    view = compatibility_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False


def test_pickle_repr_str_cannot_be_identity() -> None:
    for key in ("pickle_fallback", "repr_fallback", "str_fallback"):
        view = compatibility_view({key: True})
        assert view["blocks_completion"] is True
        assert view["reason_code"] == PICKLE_FALLBACK_NOT_IDENTITY
        assert view["completes_task"] is False


def test_plugin_and_init_order_breaks_cannot_complete() -> None:
    plugin = compatibility_view({"plugin_incompatible": True})
    assert plugin["reason_code"] == PLUGIN_INCOMPATIBLE
    assert plugin["blocks_completion"] is True
    init = compatibility_view({"initialization_order_cycle": True})
    assert init["reason_code"] == INIT_ORDER_INCOMPATIBLE
    silent = compatibility_view({"silent_public_incompatibility": True})
    assert silent["reason_code"] == SILENT_PUBLIC_INCOMPATIBILITY
    assert silent["accepted_as_authority"] is False


def test_respected_compatibility_does_not_complete() -> None:
    view = compatibility_view({"compatibility": {"compatible": True}})
    assert view["respected"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["reason_code"] == COMPATIBILITY_RESPECTED
