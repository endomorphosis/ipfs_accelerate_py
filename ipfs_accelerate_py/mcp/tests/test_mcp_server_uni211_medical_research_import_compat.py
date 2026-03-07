#!/usr/bin/env python3
"""UNI-211 medical research import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.medical_research_scrapers import (
    scrape_clinical_trials,
    scrape_pubmed_medical_research,
)
from ipfs_accelerate_py.mcp_server.tools.medical_research_scrapers import native_medical_research_scrapers


def test_medical_research_package_exports_supported_native_functions() -> None:
    assert scrape_pubmed_medical_research is native_medical_research_scrapers.scrape_pubmed_medical_research
    assert scrape_clinical_trials is native_medical_research_scrapers.scrape_clinical_trials