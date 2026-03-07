"""CLI category for unified mcp_server."""

from .native_cli_tools import (
	discover_biomolecules_rag_cli,
	discover_enzyme_inhibitors_cli,
	discover_protein_binders_cli,
	execute_command,
	register_native_cli_tools,
	scrape_clinical_trials_cli,
	scrape_pubmed_cli,
)

__all__ = [
	"execute_command",
	"scrape_pubmed_cli",
	"scrape_clinical_trials_cli",
	"discover_protein_binders_cli",
	"discover_enzyme_inhibitors_cli",
	"discover_biomolecules_rag_cli",
	"register_native_cli_tools",
]
