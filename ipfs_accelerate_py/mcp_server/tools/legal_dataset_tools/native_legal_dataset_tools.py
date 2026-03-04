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


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    return {"result": payload}


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
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            states=clean_states,
            output_format=clean_output_format,
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
