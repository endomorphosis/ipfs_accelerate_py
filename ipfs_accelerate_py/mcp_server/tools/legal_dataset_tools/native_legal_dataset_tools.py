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

        return {
            "list_state_jurisdictions": _list_state_jurisdictions,
            "scrape_state_laws": _scrape_state_laws,
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

        return {
            "list_state_jurisdictions": _list_states_fallback,
            "scrape_state_laws": _scrape_state_laws_fallback,
        }


_API = _load_legal_dataset_tools_api()


async def list_state_jurisdictions() -> Dict[str, Any]:
    """List supported state jurisdictions for legal data scraping."""
    result = _API["list_state_jurisdictions"]()
    if hasattr(result, "__await__"):
        return await result
    return result


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
    result = _API["scrape_state_laws"](
        states=states,
        legal_areas=legal_areas,
        output_format=output_format,
        include_metadata=include_metadata,
        rate_limit_delay=rate_limit_delay,
        max_statutes=max_statutes,
        use_state_specific_scrapers=use_state_specific_scrapers,
        output_dir=output_dir,
        write_jsonld=write_jsonld,
        strict_full_text=strict_full_text,
        min_full_text_chars=min_full_text_chars,
        hydrate_statute_text=hydrate_statute_text,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
                "output_format": {"type": "string"},
                "include_metadata": {"type": "boolean"},
                "rate_limit_delay": {"type": "number"},
                "max_statutes": {"type": ["integer", "null"]},
                "use_state_specific_scrapers": {"type": "boolean"},
                "output_dir": {"type": ["string", "null"]},
                "write_jsonld": {"type": "boolean"},
                "strict_full_text": {"type": "boolean"},
                "min_full_text_chars": {"type": "integer"},
                "hydrate_statute_text": {"type": "boolean"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "legal-dataset-tools"],
    )
