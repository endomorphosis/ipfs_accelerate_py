"""Native legal-dataset-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_legal_dataset_tools_api() -> Dict[str, Any]:
    """Resolve source legal-dataset-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.legal_dataset_tools import (  # type: ignore
            list_state_jurisdictions as _list_state_jurisdictions,
            scrape_state_laws as _scrape_state_laws,
        )

        try:
            from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.legal_dataset_tools.enhanced_query_expander import (  # type: ignore
                expand_legal_query as _expand_legal_query,
                get_legal_relationships as _get_legal_relationships,
                get_legal_synonyms as _get_legal_synonyms,
            )
        except Exception:
            _expand_legal_query = None
            _get_legal_synonyms = None
            _get_legal_relationships = None

        return {
            "list_state_jurisdictions": _list_state_jurisdictions,
            "scrape_state_laws": _scrape_state_laws,
            "expand_legal_query": _expand_legal_query,
            "get_legal_synonyms": _get_legal_synonyms,
            "get_legal_relationships": _get_legal_relationships,
        }
    except Exception:
        logger.warning(
            "Source legal_dataset_tools import unavailable, using fallback legal-dataset functions"
        )

        async def _list_states_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "states": {"CA": "California", "NY": "New York", "TX": "Texas"},
                "count": 3,
                "note": "fallback",
            }

        async def _scrape_state_laws_fallback(
            states: Optional[List[str]] = None,
            legal_areas: Optional[List[str]] = None,
            output_format: str = "json",
            include_metadata: bool = True,
            rate_limit_delay: float = 2.0,
            max_statutes: Optional[int] = None,
            use_state_specific_scrapers: bool = True,
            output_dir: Optional[str] = None,
            write_jsonld: bool = True,
            strict_full_text: bool = False,
            min_full_text_chars: int = 300,
            hydrate_statute_text: bool = True,
        ) -> Dict[str, Any]:
            _ = (
                legal_areas,
                output_format,
                include_metadata,
                rate_limit_delay,
                max_statutes,
                use_state_specific_scrapers,
                output_dir,
                write_jsonld,
                strict_full_text,
                min_full_text_chars,
                hydrate_statute_text,
            )
            return {
                "status": "success",
                "data": [],
                "metadata": {
                    "selected_states": states or ["all"],
                    "fallback": True,
                },
                "output_format": "json",
            }

        async def _expand_legal_query_fallback(
            query: str,
            strategy: str = "balanced",
            max_expansions: int = 10,
            include_synonyms: bool = True,
            include_related: bool = True,
            include_acronyms: bool = True,
            domains: Optional[List[str]] = None,
            min_confidence: float = 0.5,
        ) -> Dict[str, Any]:
            base_query = str(query or "").strip()
            normalized_domains = list(domains or [])
            expansions = [base_query]
            if include_synonyms:
                expansions.append(f"{base_query} statute")
            if include_related:
                expansions.append(f"{base_query} compliance")
            if include_acronyms:
                expansions.append(f"{base_query} regulation")
            unique_expansions = [item for idx, item in enumerate(expansions) if item and item not in expansions[:idx]]
            return {
                "status": "success",
                "original_query": base_query,
                "expanded_queries": unique_expansions[:max_expansions],
                "terms_expanded": base_query.split(),
                "expansion_metadata": {
                    "fallback": True,
                    "domains": normalized_domains,
                    "min_confidence": float(min_confidence),
                },
                "strategy_used": strategy,
                "total_expansions": min(len(unique_expansions), max_expansions),
                "expansion_config": {
                    "include_synonyms": include_synonyms,
                    "include_related": include_related,
                    "include_acronyms": include_acronyms,
                },
                "mcp_tool": "expand_legal_query",
            }

        async def _get_legal_synonyms_fallback(
            term: Optional[str] = None,
            category: Optional[str] = None,
        ) -> Dict[str, Any]:
            synonym_map = {
                "regulation": ["rule", "directive", "requirement"],
                "compliance": ["adherence", "conformity", "observance"],
                "statute": ["law", "code", "act"],
            }
            if term is not None:
                normalized_term = str(term).strip().lower()
                return {
                    "status": "success",
                    "term": normalized_term,
                    "synonyms": synonym_map.get(normalized_term, []),
                    "count": len(synonym_map.get(normalized_term, [])),
                }
            return {
                "status": "success",
                "synonyms": synonym_map,
                "total_terms": len(synonym_map),
                "category": category,
                "message": f"Retrieved {len(synonym_map)} legal terms",
            }

        async def _get_legal_relationships_fallback(
            term: Optional[str] = None,
            relationship_type: Optional[str] = None,
        ) -> Dict[str, Any]:
            relationships = {
                "regulation": {
                    "hierarchical": ["statute", "agency guidance"],
                    "procedural": ["notice", "comment"],
                    "domain": ["environmental", "administrative"],
                }
            }
            if term is not None:
                normalized_term = str(term).strip().lower()
                term_relationships = relationships.get(normalized_term, {})
                if relationship_type:
                    term_relationships = {
                        str(relationship_type): term_relationships.get(str(relationship_type), [])
                    }
                return {
                    "status": "success",
                    "term": normalized_term,
                    "relationships": term_relationships,
                    "relationship_type": relationship_type,
                }
            return {
                "status": "success",
                "relationships": relationships,
                "total_terms": len(relationships),
                "relationship_type": relationship_type,
            }

        return {
            "list_state_jurisdictions": _list_states_fallback,
            "scrape_state_laws": _scrape_state_laws_fallback,
            "expand_legal_query": _expand_legal_query_fallback,
            "get_legal_synonyms": _get_legal_synonyms_fallback,
            "get_legal_relationships": _get_legal_relationships_fallback,
        }


_API = _load_legal_dataset_tools_api()


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


async def list_state_jurisdictions() -> Dict[str, Any]:
    """List supported state jurisdictions for legal data scraping."""
    try:
        result = _API["list_state_jurisdictions"]()
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("states", {})
            envelope.setdefault("count", len(envelope.get("states") or {}))
        return envelope
    except Exception as exc:
        return _error_result(str(exc))


async def scrape_state_laws(
    states: Optional[List[str]] = None,
    legal_areas: Optional[List[str]] = None,
    output_format: str = "json",
    include_metadata: bool = True,
    rate_limit_delay: float = 2.0,
    max_statutes: Optional[int] = None,
    use_state_specific_scrapers: bool = True,
    output_dir: Optional[str] = None,
    write_jsonld: bool = True,
    strict_full_text: bool = False,
    min_full_text_chars: int = 300,
    hydrate_statute_text: bool = True,
) -> Dict[str, Any]:
    """Scrape state legal datasets for configured jurisdictions."""
    if states is not None and (
        not isinstance(states, list)
        or not all(isinstance(state, str) and state.strip() for state in states)
    ):
        return _error_result("states must be null or a list of non-empty strings", states=states)
    if legal_areas is not None and (
        not isinstance(legal_areas, list)
        or not all(isinstance(area, str) and area.strip() for area in legal_areas)
    ):
        return _error_result(
            "legal_areas must be null or a list of non-empty strings",
            legal_areas=legal_areas,
        )
    if not isinstance(output_format, str) or output_format.strip().lower() not in {
        "json",
        "csv",
        "parquet",
    }:
        return _error_result(
            "output_format must be one of: json, csv, parquet",
            output_format=output_format,
        )
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean", include_metadata=include_metadata)
    if not isinstance(rate_limit_delay, (int, float)) or rate_limit_delay < 0:
        return _error_result("rate_limit_delay must be a number >= 0", rate_limit_delay=rate_limit_delay)
    if max_statutes is not None and (not isinstance(max_statutes, int) or max_statutes < 1):
        return _error_result("max_statutes must be null or an integer >= 1", max_statutes=max_statutes)
    if not isinstance(use_state_specific_scrapers, bool):
        return _error_result(
            "use_state_specific_scrapers must be a boolean",
            use_state_specific_scrapers=use_state_specific_scrapers,
        )
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        return _error_result("output_dir must be null or a non-empty string", output_dir=output_dir)
    if not isinstance(write_jsonld, bool):
        return _error_result("write_jsonld must be a boolean", write_jsonld=write_jsonld)
    if not isinstance(strict_full_text, bool):
        return _error_result("strict_full_text must be a boolean", strict_full_text=strict_full_text)
    if not isinstance(min_full_text_chars, int) or min_full_text_chars < 1:
        return _error_result(
            "min_full_text_chars must be an integer >= 1",
            min_full_text_chars=min_full_text_chars,
        )
    if not isinstance(hydrate_statute_text, bool):
        return _error_result(
            "hydrate_statute_text must be a boolean",
            hydrate_statute_text=hydrate_statute_text,
        )

    clean_states = [state.strip().upper() for state in states] if states is not None else None
    clean_legal_areas = [area.strip() for area in legal_areas] if legal_areas is not None else None
    clean_output_format = output_format.strip().lower()
    clean_output_dir = output_dir.strip() if isinstance(output_dir, str) else None

    try:
        result = _API["scrape_state_laws"](
            states=clean_states,
            legal_areas=clean_legal_areas,
            output_format=clean_output_format,
            include_metadata=include_metadata,
            rate_limit_delay=float(rate_limit_delay),
            max_statutes=max_statutes,
            use_state_specific_scrapers=use_state_specific_scrapers,
            output_dir=clean_output_dir,
            write_jsonld=write_jsonld,
            strict_full_text=strict_full_text,
            min_full_text_chars=min_full_text_chars,
            hydrate_statute_text=hydrate_statute_text,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("output_format", clean_output_format)
        if clean_states is not None:
            envelope.setdefault("states", clean_states)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("data", [])
            envelope.setdefault(
                "metadata",
                {
                    "selected_states": clean_states or ["all"],
                    "legal_areas": clean_legal_areas or [],
                    "include_metadata": include_metadata,
                },
            )
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            states=clean_states,
            output_format=clean_output_format,
        )


async def expand_legal_query(
    query: str,
    strategy: str = "balanced",
    max_expansions: int = 10,
    include_synonyms: bool = True,
    include_related: bool = True,
    include_acronyms: bool = True,
    domains: Optional[List[str]] = None,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    """Expand legal queries with source-compatible validation and fallback behavior."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error_result("query must be a non-empty string", query=query)

    normalized_strategy = str(strategy or "").strip().lower()
    valid_strategies = {"conservative", "balanced", "aggressive"}
    if normalized_strategy not in valid_strategies:
        return _error_result(
            "strategy must be one of: conservative, balanced, aggressive",
            strategy=strategy,
        )
    if not isinstance(max_expansions, int) or not 1 <= max_expansions <= 50:
        return _error_result(
            "max_expansions must be an integer between 1 and 50",
            max_expansions=max_expansions,
        )
    if not isinstance(include_synonyms, bool):
        return _error_result("include_synonyms must be a boolean", include_synonyms=include_synonyms)
    if not isinstance(include_related, bool):
        return _error_result("include_related must be a boolean", include_related=include_related)
    if not isinstance(include_acronyms, bool):
        return _error_result("include_acronyms must be a boolean", include_acronyms=include_acronyms)
    if domains is not None and (
        not isinstance(domains, list)
        or not all(isinstance(domain, str) and domain.strip() for domain in domains)
    ):
        return _error_result("domains must be null or a list of non-empty strings", domains=domains)
    if not isinstance(min_confidence, (int, float)) or not 0.0 <= float(min_confidence) <= 1.0:
        return _error_result(
            "min_confidence must be a number between 0.0 and 1.0",
            min_confidence=min_confidence,
        )

    normalized_domains = [domain.strip().lower() for domain in domains] if domains is not None else None
    valid_domains = {"administrative", "criminal", "civil", "environmental", "labor"}
    if normalized_domains is not None:
        invalid_domains = [domain for domain in normalized_domains if domain not in valid_domains]
        if invalid_domains:
            return _error_result(
                "domains must be drawn from: administrative, criminal, civil, environmental, labor",
                domains=normalized_domains,
            )

    delegate = _API.get("expand_legal_query")
    if delegate is None:
        return _error_result("expand_legal_query handler unavailable")

    try:
        result = delegate(
            query=normalized_query,
            strategy=normalized_strategy,
            max_expansions=max_expansions,
            include_synonyms=include_synonyms,
            include_related=include_related,
            include_acronyms=include_acronyms,
            domains=normalized_domains,
            min_confidence=float(min_confidence),
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("original_query", normalized_query)
        envelope.setdefault("strategy_used", normalized_strategy)
        envelope.setdefault("expanded_queries", [])
        envelope.setdefault("expansion_metadata", {})
        envelope.setdefault("total_expansions", len(envelope.get("expanded_queries") or []))
        if normalized_domains is not None:
            envelope.setdefault("domains", normalized_domains)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), query=normalized_query, strategy=normalized_strategy)


async def get_legal_synonyms(
    term: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Return legal synonym metadata with deterministic validation envelopes."""
    normalized_term = str(term).strip().lower() if term is not None else None
    if term is not None and not normalized_term:
        return _error_result("term must be null or a non-empty string", term=term)
    normalized_category = str(category).strip().lower() if category is not None else None
    if category is not None and not normalized_category:
        return _error_result("category must be null or a non-empty string", category=category)

    delegate = _API.get("get_legal_synonyms")
    if delegate is None:
        return _error_result("get_legal_synonyms handler unavailable")

    try:
        result = delegate(term=normalized_term, category=normalized_category)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if normalized_term is not None:
            envelope.setdefault("term", normalized_term)
        else:
            envelope.setdefault("category", normalized_category)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            if normalized_term is not None:
                envelope.setdefault("synonyms", [])
                envelope.setdefault("count", len(envelope.get("synonyms") or []))
        return envelope
    except Exception as exc:
        return _error_result(str(exc), term=normalized_term, category=normalized_category)


async def get_legal_relationships(
    term: Optional[str] = None,
    relationship_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Return legal relationship metadata with deterministic validation envelopes."""
    normalized_term = str(term).strip().lower() if term is not None else None
    if term is not None and not normalized_term:
        return _error_result("term must be null or a non-empty string", term=term)
    normalized_relationship_type = (
        str(relationship_type).strip().lower() if relationship_type is not None else None
    )
    valid_relationship_types = {"hierarchical", "procedural", "domain"}
    if relationship_type is not None and normalized_relationship_type not in valid_relationship_types:
        return _error_result(
            "relationship_type must be null or one of: hierarchical, procedural, domain",
            relationship_type=relationship_type,
        )

    delegate = _API.get("get_legal_relationships")
    if delegate is None:
        return _error_result("get_legal_relationships handler unavailable")

    try:
        result = delegate(term=normalized_term, relationship_type=normalized_relationship_type)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if normalized_term is not None:
            envelope.setdefault("term", normalized_term)
        envelope.setdefault("relationship_type", normalized_relationship_type)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("relationships", {})
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            term=normalized_term,
            relationship_type=normalized_relationship_type,
        )


def register_native_legal_dataset_tools(manager: Any) -> None:
    """Register native legal-dataset-tools category tools in unified manager."""
    manager.register_tool(
        category="legal_dataset_tools",
        name="list_state_jurisdictions",
        func=list_state_jurisdictions,
        description="List available US state jurisdictions for legal scraping.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "legal-dataset-tools"],
    )

    manager.register_tool(
        category="legal_dataset_tools",
        name="scrape_state_laws",
        func=scrape_state_laws,
        description="Scrape state laws and statutes datasets.",
        input_schema={
            "type": "object",
            "properties": {
                "states": {"type": ["array", "null"], "items": {"type": "string"}},
                "legal_areas": {"type": ["array", "null"], "items": {"type": "string"}},
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv", "parquet"],
                    "default": "json",
                },
                "include_metadata": {"type": "boolean", "default": True},
                "rate_limit_delay": {"type": "number", "minimum": 0, "default": 2.0},
                "max_statutes": {"type": ["integer", "null"], "minimum": 1},
                "use_state_specific_scrapers": {"type": "boolean", "default": True},
                "output_dir": {"type": ["string", "null"]},
                "write_jsonld": {"type": "boolean", "default": True},
                "strict_full_text": {"type": "boolean", "default": False},
                "min_full_text_chars": {"type": "integer", "minimum": 1, "default": 300},
                "hydrate_statute_text": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "legal-dataset-tools"],
    )

    manager.register_tool(
        category="legal_dataset_tools",
        name="expand_legal_query",
        func=expand_legal_query,
        description="Expand legal queries with source-compatible synonym and relationship strategies.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "strategy": {
                    "type": "string",
                    "enum": ["conservative", "balanced", "aggressive"],
                    "default": "balanced",
                },
                "max_expansions": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "include_synonyms": {"type": "boolean", "default": True},
                "include_related": {"type": "boolean", "default": True},
                "include_acronyms": {"type": "boolean", "default": True},
                "domains": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "legal-dataset-tools"],
    )

    manager.register_tool(
        category="legal_dataset_tools",
        name="get_legal_synonyms",
        func=get_legal_synonyms,
        description="Get legal synonyms for a term or return available synonym mappings.",
        input_schema={
            "type": "object",
            "properties": {
                "term": {"type": ["string", "null"]},
                "category": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "legal-dataset-tools"],
    )

    manager.register_tool(
        category="legal_dataset_tools",
        name="get_legal_relationships",
        func=get_legal_relationships,
        description="Get legal term relationships across hierarchical, procedural, and domain views.",
        input_schema={
            "type": "object",
            "properties": {
                "term": {"type": ["string", "null"]},
                "relationship_type": {
                    "type": ["string", "null"],
                    "enum": ["hierarchical", "procedural", "domain", None],
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "legal-dataset-tools"],
    )
