"""Native CLI category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PUBMED_RESEARCH_TYPES = {
    "clinical_trial",
    "meta_analysis",
    "review",
    "research_article",
}
_TRIAL_PHASES = {"Phase 1", "Phase 2", "Phase 3", "Phase 4"}
_INTERACTION_TYPES = {"binding", "inhibition", "activation"}
_DISCOVERY_TYPES = {"binders", "inhibitors", "pathway"}
_OUTPUT_FORMATS = {"json", "table"}


def _load_cli_tools_api() -> Dict[str, Any]:
    """Resolve source CLI APIs with compatibility fallback."""
    api: Dict[str, Any] = {}

    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.cli.execute_command import (  # type: ignore
            execute_command as _execute_command,
        )

        api["execute_command"] = _execute_command
    except Exception:
        logger.warning("Source cli execute_command import unavailable, using fallback cli function")

        async def _execute_command_fallback(
            command: str,
            args: Optional[List[str]] = None,
            timeout_seconds: int = 60,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "command": command,
                "args": args or [],
                "timeout_seconds": timeout_seconds,
                "message": "fallback",
            }

        api["execute_command"] = _execute_command_fallback

    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.medical_research_scrapers.medical_research_mcp_tools import (  # type: ignore
            discover_biomolecules_rag as _discover_biomolecules_rag,
            discover_enzyme_inhibitors as _discover_enzyme_inhibitors,
            discover_protein_binders as _discover_protein_binders,
            scrape_clinical_trials as _scrape_clinical_trials,
            scrape_pubmed_medical_research as _scrape_pubmed_medical_research,
        )

        api.update(
            {
                "scrape_pubmed_cli": _scrape_pubmed_medical_research,
                "scrape_clinical_trials_cli": _scrape_clinical_trials,
                "discover_protein_binders_cli": _discover_protein_binders,
                "discover_enzyme_inhibitors_cli": _discover_enzyme_inhibitors,
                "discover_biomolecules_rag_cli": _discover_biomolecules_rag,
            }
        )
    except Exception:
        logger.warning("Source medical research CLI imports unavailable, using fallback cli medical functions")

        def _scrape_pubmed_fallback(
            query: str,
            max_results: int = 100,
            email: Optional[str] = None,
            research_type: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (email, research_type)
            return {
                "status": "success",
                "success": True,
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
                "success": True,
                "query": query,
                "trials": [],
                "total_count": 0,
                "max_results": max_results,
                "fallback": True,
            }

        def _discover_protein_binders_fallback(
            target_protein: str,
            interaction_type: Optional[str] = None,
            min_confidence: float = 0.5,
            max_results: int = 50,
        ) -> Dict[str, Any]:
            _ = interaction_type
            return {
                "status": "success",
                "success": True,
                "target_protein": target_protein,
                "min_confidence": min_confidence,
                "max_results": max_results,
                "candidates": [],
                "fallback": True,
            }

        def _discover_enzyme_inhibitors_fallback(
            target_enzyme: str,
            enzyme_class: Optional[str] = None,
            min_confidence: float = 0.5,
            max_results: int = 50,
        ) -> Dict[str, Any]:
            _ = enzyme_class
            return {
                "status": "success",
                "success": True,
                "target_enzyme": target_enzyme,
                "min_confidence": min_confidence,
                "max_results": max_results,
                "candidates": [],
                "fallback": True,
            }

        def _discover_biomolecules_rag_fallback(
            target: str,
            discovery_type: str,
            max_results: int = 50,
            min_confidence: float = 0.5,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "success": True,
                "target": target,
                "discovery_type": discovery_type,
                "max_results": max_results,
                "min_confidence": min_confidence,
                "candidates": [],
                "fallback": True,
            }

        api.update(
            {
                "scrape_pubmed_cli": _scrape_pubmed_fallback,
                "scrape_clinical_trials_cli": _scrape_trials_fallback,
                "discover_protein_binders_cli": _discover_protein_binders_fallback,
                "discover_enzyme_inhibitors_cli": _discover_enzyme_inhibitors_fallback,
                "discover_biomolecules_rag_cli": _discover_biomolecules_rag_fallback,
            }
        )

    return api


_API = _load_cli_tools_api()


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


async def _await_maybe(result: Any) -> Any:
    if hasattr(result, "__await__"):
        return await result
    return result


def _require_string(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return _error_result(f"{field} must be a non-empty string", **{field: value})
    return None


def _optional_string(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return _require_string(value, field)


def _validate_positive_int(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, int) or value < 1:
        return _error_result(f"{field} must be an integer >= 1", **{field: value})
    return None


def _validate_confidence(value: Any, field: str = "min_confidence") -> Optional[Dict[str, Any]]:
    if not isinstance(value, (int, float)):
        return _error_result(f"{field} must be a number between 0 and 1", **{field: value})
    numeric = float(value)
    if numeric < 0 or numeric > 1:
        return _error_result(f"{field} must be between 0 and 1", **{field: value})
    return None


def _validate_choice(value: Any, field: str, allowed: set[str], optional: bool = False) -> Optional[Dict[str, Any]]:
    if value is None and optional:
        return None
    invalid = _require_string(value, field)
    if invalid:
        return invalid
    cleaned = str(value).strip()
    if cleaned not in allowed:
        return _error_result(
            f"{field} must be one of: {', '.join(sorted(allowed))}",
            **{field: value},
        )
    return None


async def execute_command(
    command: str,
    args: Optional[List[str]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Execute a command via the MCP CLI facade."""
    if not isinstance(command, str) or not command.strip():
        return _error_result("command must be a non-empty string", command=command)
    if args is not None and (
        not isinstance(args, list)
        or not all(isinstance(arg, str) and arg.strip() for arg in args)
    ):
        return _error_result("args must be null or a list of non-empty strings", args=args)
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        return _error_result(
            "timeout_seconds must be an integer >= 1",
            timeout_seconds=timeout_seconds,
        )

    clean_command = command.strip()
    clean_args = [arg.strip() for arg in args] if args is not None else []

    try:
        result = await _await_maybe(
            _API["execute_command"](
            command=clean_command,
            args=clean_args,
            timeout_seconds=timeout_seconds,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("command", clean_command)
        envelope.setdefault("args", clean_args)
        envelope.setdefault("timeout_seconds", timeout_seconds)
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            command=clean_command,
            args=clean_args,
            timeout_seconds=timeout_seconds,
        )


async def scrape_pubmed_cli(
    query: str,
    max_results: int = 100,
    email: Optional[str] = None,
    research_type: Optional[str] = None,
    output: Optional[str] = None,
    format: str = "json",
) -> Dict[str, Any]:
    """Run the structured PubMed CLI workflow through the underlying MCP tool."""
    invalid = _require_string(query, "query")
    if invalid:
        return invalid
    invalid = _validate_positive_int(max_results, "max_results")
    if invalid:
        return invalid
    invalid = _optional_string(email, "email")
    if invalid:
        return invalid
    invalid = _validate_choice(research_type, "research_type", _PUBMED_RESEARCH_TYPES, optional=True)
    if invalid:
        return invalid
    invalid = _optional_string(output, "output")
    if invalid:
        return invalid
    invalid = _validate_choice(format, "format", _OUTPUT_FORMATS)
    if invalid:
        return invalid

    clean_query = query.strip()
    clean_email = email.strip() if isinstance(email, str) else None
    clean_research_type = research_type.strip() if isinstance(research_type, str) else None
    clean_output = output.strip() if isinstance(output, str) else None
    clean_format = format.strip()

    try:
        result = await _await_maybe(
            _API["scrape_pubmed_cli"](
                query=clean_query,
                max_results=max_results,
                email=clean_email,
                research_type=clean_research_type,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("query", clean_query)
        envelope.setdefault("max_results", max_results)
        envelope.setdefault("format", clean_format)
        envelope.setdefault("output", clean_output)
        envelope.setdefault("articles", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), query=clean_query, max_results=max_results)


async def scrape_clinical_trials_cli(
    query: Optional[str] = None,
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    phase: Optional[str] = None,
    max_results: int = 50,
    output: Optional[str] = None,
    format: str = "json",
) -> Dict[str, Any]:
    """Run the structured ClinicalTrials CLI workflow through the underlying MCP tool."""
    if query is None and condition is None:
        return _error_result("query or condition is required", query=query, condition=condition)
    invalid = _optional_string(query, "query")
    if invalid:
        return invalid
    invalid = _optional_string(condition, "condition")
    if invalid:
        return invalid
    invalid = _optional_string(intervention, "intervention")
    if invalid:
        return invalid
    invalid = _validate_choice(phase, "phase", _TRIAL_PHASES, optional=True)
    if invalid:
        return invalid
    invalid = _validate_positive_int(max_results, "max_results")
    if invalid:
        return invalid
    invalid = _optional_string(output, "output")
    if invalid:
        return invalid
    invalid = _validate_choice(format, "format", _OUTPUT_FORMATS)
    if invalid:
        return invalid

    clean_query = query.strip() if isinstance(query, str) else None
    clean_condition = condition.strip() if isinstance(condition, str) else None
    clean_intervention = intervention.strip() if isinstance(intervention, str) else None
    clean_phase = phase.strip() if isinstance(phase, str) else None
    clean_output = output.strip() if isinstance(output, str) else None
    clean_format = format.strip()
    effective_query = clean_query or clean_condition

    try:
        result = await _await_maybe(
            _API["scrape_clinical_trials_cli"](
                query=effective_query,
                condition=clean_condition,
                intervention=clean_intervention,
                max_results=max_results,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("query", effective_query)
        envelope.setdefault("phase", clean_phase)
        envelope.setdefault("max_results", max_results)
        envelope.setdefault("format", clean_format)
        envelope.setdefault("output", clean_output)
        envelope.setdefault("trials", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), query=effective_query, max_results=max_results)


async def discover_protein_binders_cli(
    target: str,
    interaction: Optional[str] = None,
    min_confidence: float = 0.5,
    max_results: int = 50,
    output: Optional[str] = None,
    format: str = "json",
) -> Dict[str, Any]:
    """Run the structured protein-binder CLI workflow through biomolecule discovery."""
    invalid = _require_string(target, "target")
    if invalid:
        return invalid
    invalid = _validate_choice(interaction, "interaction", _INTERACTION_TYPES, optional=True)
    if invalid:
        return invalid
    invalid = _validate_confidence(min_confidence)
    if invalid:
        return invalid
    invalid = _validate_positive_int(max_results, "max_results")
    if invalid:
        return invalid
    invalid = _optional_string(output, "output")
    if invalid:
        return invalid
    invalid = _validate_choice(format, "format", _OUTPUT_FORMATS)
    if invalid:
        return invalid

    clean_target = target.strip()
    clean_interaction = interaction.strip() if isinstance(interaction, str) else None
    clean_output = output.strip() if isinstance(output, str) else None
    clean_format = format.strip()
    confidence = float(min_confidence)

    try:
        result = await _await_maybe(
            _API["discover_protein_binders_cli"](
                target_protein=clean_target,
                interaction_type=clean_interaction,
                min_confidence=confidence,
                max_results=max_results,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("target", clean_target)
        envelope.setdefault("interaction", clean_interaction)
        envelope.setdefault("min_confidence", confidence)
        envelope.setdefault("max_results", max_results)
        envelope.setdefault("format", clean_format)
        envelope.setdefault("output", clean_output)
        envelope.setdefault("candidates", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), target=clean_target, max_results=max_results)


async def discover_enzyme_inhibitors_cli(
    target: str,
    enzyme_class: Optional[str] = None,
    min_confidence: float = 0.5,
    max_results: int = 50,
    output: Optional[str] = None,
    format: str = "json",
) -> Dict[str, Any]:
    """Run the structured enzyme-inhibitor CLI workflow through biomolecule discovery."""
    invalid = _require_string(target, "target")
    if invalid:
        return invalid
    invalid = _optional_string(enzyme_class, "enzyme_class")
    if invalid:
        return invalid
    invalid = _validate_confidence(min_confidence)
    if invalid:
        return invalid
    invalid = _validate_positive_int(max_results, "max_results")
    if invalid:
        return invalid
    invalid = _optional_string(output, "output")
    if invalid:
        return invalid
    invalid = _validate_choice(format, "format", _OUTPUT_FORMATS)
    if invalid:
        return invalid

    clean_target = target.strip()
    clean_enzyme_class = enzyme_class.strip() if isinstance(enzyme_class, str) else None
    clean_output = output.strip() if isinstance(output, str) else None
    clean_format = format.strip()
    confidence = float(min_confidence)

    try:
        result = await _await_maybe(
            _API["discover_enzyme_inhibitors_cli"](
                target_enzyme=clean_target,
                enzyme_class=clean_enzyme_class,
                min_confidence=confidence,
                max_results=max_results,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("target", clean_target)
        envelope.setdefault("enzyme_class", clean_enzyme_class)
        envelope.setdefault("min_confidence", confidence)
        envelope.setdefault("max_results", max_results)
        envelope.setdefault("format", clean_format)
        envelope.setdefault("output", clean_output)
        envelope.setdefault("candidates", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), target=clean_target, max_results=max_results)


async def discover_biomolecules_rag_cli(
    target: str,
    type: str,
    min_confidence: float = 0.5,
    max_results: int = 50,
    output: Optional[str] = None,
    format: str = "json",
) -> Dict[str, Any]:
    """Run the high-level biomolecule-discovery CLI workflow through the MCP tool."""
    invalid = _require_string(target, "target")
    if invalid:
        return invalid
    invalid = _validate_choice(type, "type", _DISCOVERY_TYPES)
    if invalid:
        return invalid
    invalid = _validate_confidence(min_confidence)
    if invalid:
        return invalid
    invalid = _validate_positive_int(max_results, "max_results")
    if invalid:
        return invalid
    invalid = _optional_string(output, "output")
    if invalid:
        return invalid
    invalid = _validate_choice(format, "format", _OUTPUT_FORMATS)
    if invalid:
        return invalid

    clean_target = target.strip()
    clean_type = type.strip()
    clean_output = output.strip() if isinstance(output, str) else None
    clean_format = format.strip()
    confidence = float(min_confidence)

    try:
        result = await _await_maybe(
            _API["discover_biomolecules_rag_cli"](
                target=clean_target,
                discovery_type=clean_type,
                max_results=max_results,
                min_confidence=confidence,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("target", clean_target)
        envelope.setdefault("type", clean_type)
        envelope.setdefault("min_confidence", confidence)
        envelope.setdefault("max_results", max_results)
        envelope.setdefault("format", clean_format)
        envelope.setdefault("output", clean_output)
        envelope.setdefault("candidates", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), target=clean_target, type=clean_type)


def register_native_cli_tools(manager: Any) -> None:
    """Register native CLI category tools in unified manager."""
    manager.register_tool(
        category="cli",
        name="execute_command",
        func=execute_command,
        description="Execute a command through the MCP CLI interface.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "args": {"type": ["array", "null"], "items": {"type": "string"}},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 60},
            },
            "required": ["command"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )

    manager.register_tool(
        category="cli",
        name="scrape_pubmed_cli",
        func=scrape_pubmed_cli,
        description="Structured CLI-compatible PubMed scraping workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1, "default": 100},
                "email": {"type": ["string", "null"]},
                "research_type": {
                    "type": ["string", "null"],
                    "enum": ["clinical_trial", "meta_analysis", "research_article", "review", None],
                },
                "output": {"type": ["string", "null"]},
                "format": {"type": "string", "enum": ["json", "table"], "default": "json"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )

    manager.register_tool(
        category="cli",
        name="scrape_clinical_trials_cli",
        func=scrape_clinical_trials_cli,
        description="Structured CLI-compatible ClinicalTrials scraping workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "condition": {"type": ["string", "null"]},
                "intervention": {"type": ["string", "null"]},
                "phase": {"type": ["string", "null"], "enum": ["Phase 1", "Phase 2", "Phase 3", "Phase 4", None]},
                "max_results": {"type": "integer", "minimum": 1, "default": 50},
                "output": {"type": ["string", "null"]},
                "format": {"type": "string", "enum": ["json", "table"], "default": "json"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )

    manager.register_tool(
        category="cli",
        name="discover_protein_binders_cli",
        func=discover_protein_binders_cli,
        description="Structured CLI-compatible protein binder discovery workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "interaction": {"type": ["string", "null"], "enum": ["activation", "binding", "inhibition", None]},
                "min_confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "max_results": {"type": "integer", "minimum": 1, "default": 50},
                "output": {"type": ["string", "null"]},
                "format": {"type": "string", "enum": ["json", "table"], "default": "json"},
            },
            "required": ["target"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )

    manager.register_tool(
        category="cli",
        name="discover_enzyme_inhibitors_cli",
        func=discover_enzyme_inhibitors_cli,
        description="Structured CLI-compatible enzyme inhibitor discovery workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "enzyme_class": {"type": ["string", "null"]},
                "min_confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "max_results": {"type": "integer", "minimum": 1, "default": 50},
                "output": {"type": ["string", "null"]},
                "format": {"type": "string", "enum": ["json", "table"], "default": "json"},
            },
            "required": ["target"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )

    manager.register_tool(
        category="cli",
        name="discover_biomolecules_rag_cli",
        func=discover_biomolecules_rag_cli,
        description="Structured CLI-compatible unified biomolecule discovery workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": ["binders", "inhibitors", "pathway"]},
                "min_confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "max_results": {"type": "integer", "minimum": 1, "default": 50},
                "output": {"type": ["string", "null"]},
                "format": {"type": "string", "enum": ["json", "table"], "default": "json"},
            },
            "required": ["target", "type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )
