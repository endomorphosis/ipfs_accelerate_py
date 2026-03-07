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


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        if "status" not in envelope:
            if envelope.get("error") or envelope.get("success") is False:
                envelope["status"] = "error"
            else:
                envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def scrape_pubmed_medical_research(
    query: str,
    max_results: int = 100,
    email: Optional[str] = None,
    research_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Scrape PubMed medical research datasets for a query."""
    if not isinstance(query, str) or not query.strip():
        return _error_result("query must be a non-empty string", query=query)
    if not isinstance(max_results, int) or max_results < 1:
        return _error_result("max_results must be an integer >= 1", max_results=max_results)
    if email is not None and (not isinstance(email, str) or not email.strip()):
        return _error_result("email must be null or a non-empty string", email=email)
    if research_type is not None and (not isinstance(research_type, str) or not research_type.strip()):
        return _error_result(
            "research_type must be null or a non-empty string",
            research_type=research_type,
        )

    clean_query = query.strip()
    clean_email = email.strip() if isinstance(email, str) else None
    clean_research_type = research_type.strip() if isinstance(research_type, str) else None

    try:
        result = _API["scrape_pubmed_medical_research"](
            query=clean_query,
            max_results=max_results,
            email=clean_email,
            research_type=clean_research_type,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("query", clean_query)
        envelope.setdefault("max_results", max_results)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("articles", [])
            envelope.setdefault("total_count", len(envelope.get("articles") or []))
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            query=clean_query,
            max_results=max_results,
        )


async def scrape_clinical_trials(
    query: str,
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    max_results: int = 50,
) -> Dict[str, Any]:
    """Scrape ClinicalTrials.gov datasets for conditions/interventions."""
    if not isinstance(query, str) or not query.strip():
        return _error_result("query must be a non-empty string", query=query)
    if condition is not None and (not isinstance(condition, str) or not condition.strip()):
        return _error_result("condition must be null or a non-empty string", condition=condition)
    if intervention is not None and (not isinstance(intervention, str) or not intervention.strip()):
        return _error_result(
            "intervention must be null or a non-empty string",
            intervention=intervention,
        )
    if not isinstance(max_results, int) or max_results < 1:
        return _error_result("max_results must be an integer >= 1", max_results=max_results)

    clean_query = query.strip()
    clean_condition = condition.strip() if isinstance(condition, str) else None
    clean_intervention = intervention.strip() if isinstance(intervention, str) else None

    try:
        result = _API["scrape_clinical_trials"](
            query=clean_query,
            condition=clean_condition,
            intervention=clean_intervention,
            max_results=max_results,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("query", clean_query)
        envelope.setdefault("max_results", max_results)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("trials", [])
            envelope.setdefault("total_count", len(envelope.get("trials") or []))
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            query=clean_query,
            max_results=max_results,
        )


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
                "query": {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1, "default": 100},
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
                "query": {"type": "string", "minLength": 1},
                "condition": {"type": ["string", "null"]},
                "intervention": {"type": ["string", "null"]},
                "max_results": {"type": "integer", "minimum": 1, "default": 50},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "medical-research-scrapers"],
    )
