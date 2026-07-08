#!/usr/bin/env python3
"""UNI-215 alert import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.alert_tools import (
    evaluate_alert_rules,
    list_alert_rules,
    remove_alert_rule,
    send_discord_message,
)
from ipfs_accelerate_py.mcp_server.tools.alert_tools import native_alert_tools


def test_alert_package_exports_supported_native_functions() -> None:
    assert send_discord_message is native_alert_tools.send_discord_message
    assert evaluate_alert_rules is native_alert_tools.evaluate_alert_rules
    assert list_alert_rules is native_alert_tools.list_alert_rules
    assert remove_alert_rule is native_alert_tools.remove_alert_rule