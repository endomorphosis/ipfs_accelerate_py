"""Logic-tools category for unified mcp_server."""

from .native_logic_tools import (
	cec_analyze_formula,
	cec_check_theorem,
	cec_formula_complexity,
	cec_parse,
	cec_prove,
	cec_validate_formula,
	logic_capabilities,
	logic_health,
	register_native_logic_tools,
	tdfol_convert,
	tdfol_kb_add_axiom,
	tdfol_kb_add_theorem,
	tdfol_kb_export,
	tdfol_kb_query,
	tdfol_parse,
	tdfol_prove,
)

__all__ = [
	"logic_capabilities",
	"logic_health",
	"tdfol_parse",
	"tdfol_convert",
	"tdfol_prove",
	"tdfol_kb_add_axiom",
	"tdfol_kb_add_theorem",
	"tdfol_kb_query",
	"tdfol_kb_export",
	"cec_prove",
	"cec_check_theorem",
	"cec_parse",
	"cec_validate_formula",
	"cec_analyze_formula",
	"cec_formula_complexity",
	"register_native_logic_tools",
]
