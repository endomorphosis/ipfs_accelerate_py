"""Native medical-research-scrapers category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_medical_research_scrapers_api() -> Dict[str, Any]:
    """Resolve source medical-research APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.medical_research_scrapers.medical_research_mcp_tools import (  # type: ignore
            scrape_clinical_trials as _scrape_clinical_trials,
            scrape_pubmed_medical_research as _scrape_pubmed_medical_research,
        )

        if not callable(_scrape_pubmed_medical_research) or not callable(_scrape_clinical_trials):
            raise TypeError("Source medical research exports are not callable")

        return {
            "scrape_pubmed_medical_research": _scrape_pubmed_medical_research,
            "scrape_clinical_trials": _scrape_clinical_trials,
        }
    except Exception:
        logger.warning(
            "Source medical_research_scrapers import unavailable, using fallback medical functions"
        )

        def _scrape_pubmed_fallback(
            query: str,
            max_results: int = 100,
            email: Optional[str] = None,
            research_type: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (email, research_type)
            return {
                "status": "success",
                "query": query,
                "articles": [],
                "total_count": 0,
                "max_results": max_results,
                "fallback": True,
            }

        def _scrape_trials_fallback(
            query: str,
            condition: Optional[str] = None,
            intervention: Optional[str] = None,
            max_results: int = 50,
        ) -> Dict[str, Any]:
            _ = (condition, intervention)
            return {
                "status": "success",
                "query": query,
                "trials": [],
                "total_count": 0,
                "max_results": max_results,
                "fallback": True,
            }

        return {
            "scrape_pubmed_medical_research": _scrape_pubmed_fallback,
            "scrape_clinical_trials": _scrape_trials_fallback,
        }


_API = _load_medical_research_scrapers_api()


async def scrape_pubmed_medical_research(
    query: str,
    max_results: int = 100,
    email: Optional[str] = None,
    research_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Scrape PubMed medical research datasets for a query."""
    result = _API["scrape_pubmed_medical_research"](
        query=query,
        max_results=max_results,
        email=email,
        research_type=research_type,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def scrape_clinical_trials(
    query: str,
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    max_results: int = 50,
) -> Dict[str, Any]:
    """Scrape ClinicalTrials.gov datasets for conditions/interventions."""
    result = _API["scrape_clinical_trials"](
        query=query,
        condition=condition,
        intervention=intervention,
        max_results=max_results,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_medical_research_scrapers(manager: Any) -> None:
    """Register native medical-research-scrapers category tools in unified manager."""
    manager.register_tool(
        category="medical_research_scrapers",
        name="scrape_pubmed_medical_research",
        func=scrape_pubmed_medical_research,
        description="Scrape PubMed medical research literature.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "email": {"type": ["string", "null"]},
                "research_type": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "medical-research-scrapers"],
    )

    manager.register_tool(
        category="medical_research_scrapers",
        name="scrape_clinical_trials",
        func=scrape_clinical_trials,
        description="Scrape clinical trial records for a search query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "condition": {"type": ["string", "null"]},
                "intervention": {"type": ["string", "null"]},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "medical-research-scrapers"],
    )
