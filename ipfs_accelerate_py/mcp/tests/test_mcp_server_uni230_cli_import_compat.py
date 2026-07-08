#!/usr/bin/env python3
"""UNI-230 CLI import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.cli import (
    discover_biomolecules_rag_cli,
    discover_enzyme_inhibitors_cli,
    discover_protein_binders_cli,
    execute_command,
    scrape_clinical_trials_cli,
    scrape_pubmed_cli,
)
from ipfs_accelerate_py.mcp_server.tools.cli import native_cli_tools


def test_cli_package_exports_supported_native_functions() -> None:
    assert execute_command is native_cli_tools.execute_command
    assert scrape_pubmed_cli is native_cli_tools.scrape_pubmed_cli
    assert scrape_clinical_trials_cli is native_cli_tools.scrape_clinical_trials_cli
    assert discover_protein_binders_cli is native_cli_tools.discover_protein_binders_cli
    assert discover_enzyme_inhibitors_cli is native_cli_tools.discover_enzyme_inhibitors_cli
    assert discover_biomolecules_rag_cli is native_cli_tools.discover_biomolecules_rag_cli