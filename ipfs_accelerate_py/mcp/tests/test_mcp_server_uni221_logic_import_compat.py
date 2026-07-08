#!/usr/bin/env python3
"""UNI-221 logic import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.logic_tools import (
    cec_analyze_formula,
    cec_check_theorem,
    cec_formula_complexity,
    cec_parse,
    cec_prove,
    cec_validate_formula,
    logic_capabilities,
    logic_health,
    tdfol_convert,
    tdfol_kb_add_axiom,
    tdfol_kb_add_theorem,
    tdfol_kb_export,
    tdfol_kb_query,
    tdfol_parse,
    tdfol_prove,
)
from ipfs_accelerate_py.mcp_server.tools.logic_tools import native_logic_tools


def test_logic_package_exports_supported_native_functions() -> None:
    assert logic_capabilities is native_logic_tools.logic_capabilities
    assert logic_health is native_logic_tools.logic_health
    assert tdfol_parse is native_logic_tools.tdfol_parse
    assert tdfol_convert is native_logic_tools.tdfol_convert
    assert tdfol_prove is native_logic_tools.tdfol_prove
    assert tdfol_kb_add_axiom is native_logic_tools.tdfol_kb_add_axiom
    assert tdfol_kb_add_theorem is native_logic_tools.tdfol_kb_add_theorem
    assert tdfol_kb_query is native_logic_tools.tdfol_kb_query
    assert tdfol_kb_export is native_logic_tools.tdfol_kb_export
    assert cec_prove is native_logic_tools.cec_prove
    assert cec_check_theorem is native_logic_tools.cec_check_theorem
    assert cec_parse is native_logic_tools.cec_parse
    assert cec_validate_formula is native_logic_tools.cec_validate_formula
    assert cec_analyze_formula is native_logic_tools.cec_analyze_formula
    assert cec_formula_complexity is native_logic_tools.cec_formula_complexity