"""Native software-engineering-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PRIORITY_LEVELS = {"emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"}
_K8S_SEVERITIES = {"ERROR", "WARN", "INFO", "DEBUG", "FATAL", "CRITICAL"}


def _load_software_engineering_tools_api() -> Dict[str, Any]:
    """Resolve source software-engineering APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.processors.development.auto_healing_engine import (  # type: ignore
            coordinate_auto_healing as _coordinate_auto_healing,
            monitor_healing_effectiveness as _monitor_healing_effectiveness,
        )
        from ipfs_datasets_py.ipfs_datasets_py.processors.development.error_pattern_engine import (  # type: ignore
            detect_error_patterns as _detect_error_patterns,
            suggest_fixes as _suggest_fixes,
        )
        from ipfs_datasets_py.ipfs_datasets_py.processors.development.github_actions_engine import (  # type: ignore
            analyze_github_actions as _analyze_github_actions,
            parse_workflow_logs as _parse_workflow_logs,
        )
        from ipfs_datasets_py.ipfs_datasets_py.processors.development.kubernetes_log_engine import (  # type: ignore
            analyze_pod_health as _analyze_pod_health,
            parse_kubernetes_logs as _parse_kubernetes_logs,
        )
        from ipfs_datasets_py.ipfs_datasets_py.processors.development.systemd_log_engine import (  # type: ignore
            analyze_service_health as _analyze_service_health,
            parse_systemd_logs as _parse_systemd_logs,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.software_engineering_tools.github_repository_scraper import (  # type: ignore
            scrape_repository as _scrape_repository,
            search_repositories as _search_repositories,
        )

        return {
            "scrape_repository": _scrape_repository,
            "search_repositories": _search_repositories,
            "analyze_github_actions": _analyze_github_actions,
            "parse_workflow_logs": _parse_workflow_logs,
            "parse_systemd_logs": _parse_systemd_logs,
            "analyze_service_health": _analyze_service_health,
            "parse_kubernetes_logs": _parse_kubernetes_logs,
            "analyze_pod_health": _analyze_pod_health,
            "detect_error_patterns": _detect_error_patterns,
            "suggest_fixes": _suggest_fixes,
            "coordinate_auto_healing": _coordinate_auto_healing,
            "monitor_healing_effectiveness": _monitor_healing_effectiveness,
        }
    except Exception:
        logger.warning(
            "Source software_engineering_tools import unavailable, using fallback software-engineering functions"
        )

        async def _scrape_repository_fallback(
            repository_url: str,
            include_prs: bool = True,
            include_issues: bool = True,
            include_workflows: bool = True,
            include_commits: bool = True,
            max_items: int = 100,
            github_token: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (include_prs, include_issues, include_workflows, include_commits, max_items, github_token)
            return {"status": "success", "repository_url": repository_url, "data": {}, "fallback": True}

        async def _search_repositories_fallback(
            query: str,
            max_results: int = 3,
            github_token: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = github_token
            return {
                "status": "success",
                "query": query,
                "results": [],
                "count": 0,
                "max_results": max_results,
                "fallback": True,
            }

        def _analyze_github_actions_fallback(
            repository_url: str,
            workflow_id: Optional[str] = None,
            max_runs: int = 50,
            github_token: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (workflow_id, github_token)
            return {
                "success": True,
                "repository_url": repository_url,
                "workflows": [],
                "success_rate": 0.0,
                "average_duration": 0.0,
                "resource_usage": {"total_workflow_runs": 0},
                "max_runs": max_runs,
                "fallback": True,
            }

        def _parse_workflow_logs_fallback(
            log_content: str,
            detect_errors: bool = True,
            extract_patterns: bool = True,
        ) -> Dict[str, Any]:
            _ = log_content
            return {
                "success": True,
                "errors": [],
                "warnings": [],
                "patterns": [],
                "statistics": {"total_lines": 0, "error_lines": 0, "warning_lines": 0},
                "detect_errors": detect_errors,
                "extract_patterns": extract_patterns,
                "fallback": True,
            }

        def _parse_systemd_logs_fallback(
            log_content: str,
            service_filter: Optional[str] = None,
            priority_filter: Optional[str] = None,
            max_entries: int = 1000,
        ) -> Dict[str, Any]:
            _ = (log_content, service_filter, priority_filter)
            return {
                "success": True,
                "entries": [],
                "statistics": {"total_entries": 0, "by_priority": {}, "by_service": {}, "by_hour": {}},
                "errors": [],
                "recommendations": [],
                "max_entries": max_entries,
                "fallback": True,
            }

        def _analyze_service_health_fallback(log_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
            _ = log_data
            return {
                "success": True,
                "service": service_name,
                "health_score": 100.0,
                "health_status": "healthy",
                "metrics": {"total_entries": 0, "error_entries": 0, "error_rate": 0.0},
                "recent_errors": [],
                "fallback": True,
            }

        def _parse_kubernetes_logs_fallback(
            log_content: str,
            namespace_filter: Optional[str] = None,
            pod_filter: Optional[str] = None,
            severity_filter: Optional[str] = None,
            max_entries: int = 1000,
        ) -> Dict[str, Any]:
            _ = (log_content, namespace_filter, pod_filter, severity_filter)
            return {
                "success": True,
                "entries": [],
                "statistics": {"total_entries": 0, "by_severity": {}, "by_namespace": {}, "by_pod": {}, "by_container": {}},
                "errors": [],
                "recommendations": [],
                "max_entries": max_entries,
                "fallback": True,
            }

        def _analyze_pod_health_fallback(log_data: Dict[str, Any], pod_name: str) -> Dict[str, Any]:
            _ = log_data
            return {
                "success": True,
                "pod": pod_name,
                "health_score": 100.0,
                "health_status": "healthy",
                "metrics": {"total_entries": 0, "error_entries": 0, "error_rate": 0.0},
                "issues": [],
                "recent_errors": [],
                "fallback": True,
            }

        def _detect_error_patterns_fallback(
            error_logs: List[str],
            pattern_library: Optional[Dict[str, str]] = None,
            min_occurrences: int = 2,
        ) -> Dict[str, Any]:
            _ = (pattern_library, min_occurrences)
            return {
                "success": True,
                "patterns": [],
                "most_common": [],
                "recommendations": [],
                "total_logs": len(error_logs),
                "fallback": True,
            }

        def _suggest_fixes_fallback(error_pattern: str) -> Dict[str, Any]:
            return {"success": True, "pattern": error_pattern, "fixes": [], "fallback": True}

        def _coordinate_auto_healing_fallback(
            error_report: Dict[str, Any],
            healing_strategies: Optional[List[Dict[str, Any]]] = None,
            dry_run: bool = True,
        ) -> Dict[str, Any]:
            _ = (error_report, healing_strategies)
            return {
                "success": True,
                "healing_actions": [],
                "executed": not dry_run,
                "results": [],
                "recommendations": [],
                "fallback": True,
            }

        def _monitor_healing_effectiveness_fallback(healing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "success": True,
                "overall_success_rate": 0.0,
                "total_healing_actions": len(healing_history),
                "successful_actions": 0,
                "action_statistics": {},
                "recommendations": [],
                "fallback": True,
            }

        return {
            "scrape_repository": _scrape_repository_fallback,
            "search_repositories": _search_repositories_fallback,
            "analyze_github_actions": _analyze_github_actions_fallback,
            "parse_workflow_logs": _parse_workflow_logs_fallback,
            "parse_systemd_logs": _parse_systemd_logs_fallback,
            "analyze_service_health": _analyze_service_health_fallback,
            "parse_kubernetes_logs": _parse_kubernetes_logs_fallback,
            "analyze_pod_health": _analyze_pod_health_fallback,
            "detect_error_patterns": _detect_error_patterns_fallback,
            "suggest_fixes": _suggest_fixes_fallback,
            "coordinate_auto_healing": _coordinate_auto_healing_fallback,
            "monitor_healing_effectiveness": _monitor_healing_effectiveness_fallback,
        }


_API = _load_software_engineering_tools_api()


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
    envelope: Dict[str, Any] = {"status": "error", "success": False, "error": message}
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
    if value is not None and (not isinstance(value, str) or not value.strip()):
        return _error_result(f"{field} must be null or a non-empty string", **{field: value})
    return None


def _validate_bool(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, bool):
        return _error_result(f"{field} must be a boolean", **{field: value})
    return None


def _validate_non_negative_int(value: Any, field: str, minimum: int = 1) -> Optional[Dict[str, Any]]:
    if not isinstance(value, int) or value < minimum:
        return _error_result(f"{field} must be an integer >= {minimum}", **{field: value})
    return None


def _validate_repo_url(repository_url: Any) -> Optional[Dict[str, Any]]:
    base_error = _require_string(repository_url, "repository_url")
    if base_error is not None:
        return base_error
    clean_repository_url = repository_url.strip()
    if not (
        clean_repository_url.startswith("https://github.com/")
        or clean_repository_url.startswith("http://github.com/")
    ):
        return _error_result(
            "repository_url must start with https://github.com/ or http://github.com/",
            repository_url=clean_repository_url,
        )
    return None


def _validate_string_list(value: Any, field: str, *, minimum: int = 1) -> Optional[Dict[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) < minimum
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        return _error_result(
            f"{field} must be a list of at least {minimum} non-empty strings",
            **{field: value},
        )
    return None


def _validate_object_list(value: Any, field: str, *, minimum: int = 0) -> Optional[Dict[str, Any]]:
    if not isinstance(value, list) or len(value) < minimum or any(not isinstance(item, dict) for item in value):
        qualifier = f"at least {minimum} " if minimum else ""
        return _error_result(f"{field} must be a list of {qualifier}objects", **{field: value})
    return None


async def scrape_repository(
    repository_url: str,
    include_prs: bool = True,
    include_issues: bool = True,
    include_workflows: bool = True,
    include_commits: bool = True,
    max_items: int = 100,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    validation = _validate_repo_url(repository_url)
    if validation is not None:
        return validation
    for field, value in {
        "include_prs": include_prs,
        "include_issues": include_issues,
        "include_workflows": include_workflows,
        "include_commits": include_commits,
    }.items():
        validation = _validate_bool(value, field)
        if validation is not None:
            return validation
    validation = _validate_non_negative_int(max_items, "max_items")
    if validation is not None:
        return validation
    validation = _optional_string(github_token, "github_token")
    if validation is not None:
        return validation

    clean_repository_url = repository_url.strip()
    clean_github_token = github_token.strip() if isinstance(github_token, str) else None
    try:
        result = await _await_maybe(
            _API["scrape_repository"](
                repository_url=clean_repository_url,
                include_prs=include_prs,
                include_issues=include_issues,
                include_workflows=include_workflows,
                include_commits=include_commits,
                max_items=max_items,
                github_token=clean_github_token,
            )
        )
    except Exception as exc:
        return _error_result(str(exc), repository_url=clean_repository_url)

    envelope = _normalize_payload(result)
    envelope.setdefault("repository_url", clean_repository_url)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("data", {})
    return envelope


async def search_repositories(query: str, max_results: int = 3, github_token: Optional[str] = None) -> Dict[str, Any]:
    validation = _require_string(query, "query")
    if validation is not None:
        return validation
    validation = _validate_non_negative_int(max_results, "max_results")
    if validation is not None:
        return validation
    validation = _optional_string(github_token, "github_token")
    if validation is not None:
        return validation

    clean_query = query.strip()
    clean_github_token = github_token.strip() if isinstance(github_token, str) else None
    try:
        result = await _await_maybe(
            _API["search_repositories"](
                query=clean_query,
                max_results=max_results,
                github_token=clean_github_token,
            )
        )
    except Exception as exc:
        return _error_result(str(exc), query=clean_query, max_results=max_results)

    envelope = _normalize_payload(result)
    envelope.setdefault("query", clean_query)
    envelope.setdefault("max_results", max_results)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("results", [])
        envelope.setdefault("count", len(envelope.get("results") or []))
    return envelope


async def analyze_github_actions(
    repository_url: str,
    workflow_id: Optional[str] = None,
    max_runs: int = 50,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    validation = _validate_repo_url(repository_url)
    if validation is not None:
        return validation
    validation = _optional_string(workflow_id, "workflow_id")
    if validation is not None:
        return validation
    validation = _validate_non_negative_int(max_runs, "max_runs")
    if validation is not None:
        return validation
    validation = _optional_string(github_token, "github_token")
    if validation is not None:
        return validation

    clean_repository_url = repository_url.strip()
    try:
        result = await _await_maybe(
            _API["analyze_github_actions"](
                repository_url=clean_repository_url,
                workflow_id=workflow_id.strip() if isinstance(workflow_id, str) else None,
                max_runs=max_runs,
                github_token=github_token.strip() if isinstance(github_token, str) else None,
            )
        )
    except Exception as exc:
        return _error_result(str(exc), repository_url=clean_repository_url)

    envelope = _normalize_payload(result)
    envelope.setdefault("repository_url", clean_repository_url)
    envelope.setdefault("max_runs", max_runs)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("workflows", [])
        envelope.setdefault("resource_usage", {})
    return envelope


async def parse_workflow_logs(log_content: str, detect_errors: bool = True, extract_patterns: bool = True) -> Dict[str, Any]:
    validation = _require_string(log_content, "log_content")
    if validation is not None:
        return validation
    for field, value in {"detect_errors": detect_errors, "extract_patterns": extract_patterns}.items():
        validation = _validate_bool(value, field)
        if validation is not None:
            return validation

    clean_log_content = log_content.strip()
    try:
        result = await _await_maybe(
            _API["parse_workflow_logs"](
                log_content=clean_log_content,
                detect_errors=detect_errors,
                extract_patterns=extract_patterns,
            )
        )
    except Exception as exc:
        return _error_result(str(exc))

    envelope = _normalize_payload(result)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("errors", [])
        envelope.setdefault("warnings", [])
        envelope.setdefault("patterns", [])
        envelope.setdefault("statistics", {"total_lines": 0, "error_lines": 0, "warning_lines": 0})
    return envelope


async def parse_systemd_logs(
    log_content: str,
    service_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    max_entries: int = 1000,
) -> Dict[str, Any]:
    validation = _require_string(log_content, "log_content")
    if validation is not None:
        return validation
    validation = _optional_string(service_filter, "service_filter")
    if validation is not None:
        return validation
    if priority_filter is not None:
        if not isinstance(priority_filter, str) or priority_filter.strip().lower() not in _PRIORITY_LEVELS:
            return _error_result(
                "priority_filter must be null or one of: alert, crit, debug, emerg, err, info, notice, warning",
                priority_filter=priority_filter,
            )
    validation = _validate_non_negative_int(max_entries, "max_entries")
    if validation is not None:
        return validation

    try:
        result = await _await_maybe(
            _API["parse_systemd_logs"](
                log_content=log_content.strip(),
                service_filter=service_filter.strip() if isinstance(service_filter, str) else None,
                priority_filter=priority_filter.strip().lower() if isinstance(priority_filter, str) else None,
                max_entries=max_entries,
            )
        )
    except Exception as exc:
        return _error_result(str(exc))

    envelope = _normalize_payload(result)
    envelope.setdefault("max_entries", max_entries)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("entries", [])
        envelope.setdefault("statistics", {"total_entries": 0, "by_priority": {}, "by_service": {}, "by_hour": {}})
        envelope.setdefault("errors", [])
        envelope.setdefault("recommendations", [])
    return envelope


async def analyze_service_health(log_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
    if not isinstance(log_data, dict) or not log_data:
        return _error_result("log_data must be a non-empty object", log_data=log_data)
    validation = _require_string(service_name, "service_name")
    if validation is not None:
        return validation

    clean_service_name = service_name.strip()
    try:
        result = await _await_maybe(_API["analyze_service_health"](log_data=log_data, service_name=clean_service_name))
    except Exception as exc:
        return _error_result(str(exc), service_name=clean_service_name)

    envelope = _normalize_payload(result)
    envelope.setdefault("service", clean_service_name)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("health_score", 100.0)
        envelope.setdefault("health_status", "healthy")
        envelope.setdefault("metrics", {"total_entries": 0, "error_entries": 0, "error_rate": 0.0})
        envelope.setdefault("recent_errors", [])
    return envelope


async def parse_kubernetes_logs(
    log_content: str,
    namespace_filter: Optional[str] = None,
    pod_filter: Optional[str] = None,
    severity_filter: Optional[str] = None,
    max_entries: int = 1000,
) -> Dict[str, Any]:
    validation = _require_string(log_content, "log_content")
    if validation is not None:
        return validation
    for field, value in {"namespace_filter": namespace_filter, "pod_filter": pod_filter}.items():
        validation = _optional_string(value, field)
        if validation is not None:
            return validation
    if severity_filter is not None:
        if not isinstance(severity_filter, str) or severity_filter.strip().upper() not in _K8S_SEVERITIES:
            return _error_result(
                "severity_filter must be null or one of: CRITICAL, DEBUG, ERROR, FATAL, INFO, WARN",
                severity_filter=severity_filter,
            )
    validation = _validate_non_negative_int(max_entries, "max_entries")
    if validation is not None:
        return validation

    try:
        result = await _await_maybe(
            _API["parse_kubernetes_logs"](
                log_content=log_content.strip(),
                namespace_filter=namespace_filter.strip() if isinstance(namespace_filter, str) else None,
                pod_filter=pod_filter.strip() if isinstance(pod_filter, str) else None,
                severity_filter=severity_filter.strip().upper() if isinstance(severity_filter, str) else None,
                max_entries=max_entries,
            )
        )
    except Exception as exc:
        return _error_result(str(exc))

    envelope = _normalize_payload(result)
    envelope.setdefault("max_entries", max_entries)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("entries", [])
        envelope.setdefault("statistics", {"total_entries": 0, "by_severity": {}, "by_namespace": {}, "by_pod": {}, "by_container": {}})
        envelope.setdefault("errors", [])
        envelope.setdefault("recommendations", [])
    return envelope


async def analyze_pod_health(log_data: Dict[str, Any], pod_name: str) -> Dict[str, Any]:
    if not isinstance(log_data, dict) or not log_data:
        return _error_result("log_data must be a non-empty object", log_data=log_data)
    validation = _require_string(pod_name, "pod_name")
    if validation is not None:
        return validation

    clean_pod_name = pod_name.strip()
    try:
        result = await _await_maybe(_API["analyze_pod_health"](log_data=log_data, pod_name=clean_pod_name))
    except Exception as exc:
        return _error_result(str(exc), pod_name=clean_pod_name)

    envelope = _normalize_payload(result)
    envelope.setdefault("pod", clean_pod_name)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("health_score", 100.0)
        envelope.setdefault("health_status", "healthy")
        envelope.setdefault("metrics", {"total_entries": 0, "error_entries": 0, "error_rate": 0.0})
        envelope.setdefault("issues", [])
        envelope.setdefault("recent_errors", [])
    return envelope


async def detect_error_patterns(
    error_logs: List[str],
    pattern_library: Optional[Dict[str, str]] = None,
    min_occurrences: int = 2,
) -> Dict[str, Any]:
    validation = _validate_string_list(error_logs, "error_logs", minimum=1)
    if validation is not None:
        return validation
    if pattern_library is not None:
        if not isinstance(pattern_library, dict) or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in pattern_library.items()
        ):
            return _error_result("pattern_library must be an object of non-empty string keys and values", pattern_library=pattern_library)
    validation = _validate_non_negative_int(min_occurrences, "min_occurrences")
    if validation is not None:
        return validation

    try:
        result = await _await_maybe(
            _API["detect_error_patterns"](
                error_logs=[item.strip() for item in error_logs],
                pattern_library=pattern_library,
                min_occurrences=min_occurrences,
            )
        )
    except Exception as exc:
        return _error_result(str(exc))

    envelope = _normalize_payload(result)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("patterns", [])
        envelope.setdefault("most_common", [])
        envelope.setdefault("recommendations", [])
    return envelope


async def suggest_fixes(error_pattern: str) -> Dict[str, Any]:
    validation = _require_string(error_pattern, "error_pattern")
    if validation is not None:
        return validation

    clean_error_pattern = error_pattern.strip()
    try:
        result = await _await_maybe(_API["suggest_fixes"](error_pattern=clean_error_pattern))
    except Exception as exc:
        return _error_result(str(exc), pattern=clean_error_pattern)

    envelope = _normalize_payload(result)
    envelope.setdefault("pattern", clean_error_pattern)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("fixes", [])
    return envelope


async def coordinate_auto_healing(
    error_report: Dict[str, Any],
    healing_strategies: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    if not isinstance(error_report, dict) or not error_report:
        return _error_result("error_report must be a non-empty object", error_report=error_report)
    if healing_strategies is not None:
        validation = _validate_object_list(healing_strategies, "healing_strategies")
        if validation is not None:
            return validation
    validation = _validate_bool(dry_run, "dry_run")
    if validation is not None:
        return validation

    try:
        result = await _await_maybe(
            _API["coordinate_auto_healing"](
                error_report=error_report,
                healing_strategies=healing_strategies,
                dry_run=dry_run,
            )
        )
    except Exception as exc:
        return _error_result(str(exc))

    envelope = _normalize_payload(result)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("healing_actions", [])
        envelope.setdefault("executed", not dry_run)
        envelope.setdefault("results", [])
        envelope.setdefault("recommendations", [])
    return envelope


async def monitor_healing_effectiveness(healing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    validation = _validate_object_list(healing_history, "healing_history", minimum=1)
    if validation is not None:
        return validation

    try:
        result = await _await_maybe(_API["monitor_healing_effectiveness"](healing_history=healing_history))
    except Exception as exc:
        return _error_result(str(exc))

    envelope = _normalize_payload(result)
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
        envelope.setdefault("overall_success_rate", 0.0)
        envelope.setdefault("total_healing_actions", len(healing_history))
        envelope.setdefault("successful_actions", 0)
        envelope.setdefault("action_statistics", {})
        envelope.setdefault("recommendations", [])
    return envelope


def register_native_software_engineering_tools(manager: Any) -> None:
    """Register native software-engineering-tools category tools in unified manager."""
    registrations = [
        {
            "name": "scrape_repository",
            "func": scrape_repository,
            "description": "Scrape GitHub repository metadata and engineering activity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repository_url": {"type": "string", "minLength": 1},
                    "include_prs": {"type": "boolean", "default": True},
                    "include_issues": {"type": "boolean", "default": True},
                    "include_workflows": {"type": "boolean", "default": True},
                    "include_commits": {"type": "boolean", "default": True},
                    "max_items": {"type": "integer", "minimum": 1, "default": 100},
                    "github_token": {"type": ["string", "null"]},
                },
                "required": ["repository_url"],
            },
        },
        {
            "name": "search_repositories",
            "func": search_repositories,
            "description": "Search GitHub repositories by query for engineering workflows.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "max_results": {"type": "integer", "minimum": 1, "default": 3},
                    "github_token": {"type": ["string", "null"]},
                },
                "required": ["query"],
            },
        },
        {
            "name": "analyze_github_actions",
            "func": analyze_github_actions,
            "description": "Analyze GitHub Actions workflow history and health.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repository_url": {"type": "string", "minLength": 1},
                    "workflow_id": {"type": ["string", "null"]},
                    "max_runs": {"type": "integer", "minimum": 1, "default": 50},
                    "github_token": {"type": ["string", "null"]},
                },
                "required": ["repository_url"],
            },
        },
        {
            "name": "parse_workflow_logs",
            "func": parse_workflow_logs,
            "description": "Parse GitHub Actions workflow logs and extract insights.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_content": {"type": "string", "minLength": 1},
                    "detect_errors": {"type": "boolean", "default": True},
                    "extract_patterns": {"type": "boolean", "default": True},
                },
                "required": ["log_content"],
            },
        },
        {
            "name": "parse_systemd_logs",
            "func": parse_systemd_logs,
            "description": "Parse systemd journal logs into structured entries.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_content": {"type": "string", "minLength": 1},
                    "service_filter": {"type": ["string", "null"]},
                    "priority_filter": {"type": ["string", "null"], "enum": ["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug", None]},
                    "max_entries": {"type": "integer", "minimum": 1, "default": 1000},
                },
                "required": ["log_content"],
            },
        },
        {
            "name": "analyze_service_health",
            "func": analyze_service_health,
            "description": "Analyze service health from parsed systemd logs.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_data": {"type": "object", "minProperties": 1},
                    "service_name": {"type": "string", "minLength": 1},
                },
                "required": ["log_data", "service_name"],
            },
        },
        {
            "name": "parse_kubernetes_logs",
            "func": parse_kubernetes_logs,
            "description": "Parse Kubernetes logs into structured entries and recommendations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_content": {"type": "string", "minLength": 1},
                    "namespace_filter": {"type": ["string", "null"]},
                    "pod_filter": {"type": ["string", "null"]},
                    "severity_filter": {"type": ["string", "null"], "enum": ["ERROR", "WARN", "INFO", "DEBUG", "FATAL", "CRITICAL", None]},
                    "max_entries": {"type": "integer", "minimum": 1, "default": 1000},
                },
                "required": ["log_content"],
            },
        },
        {
            "name": "analyze_pod_health",
            "func": analyze_pod_health,
            "description": "Analyze Kubernetes pod health from parsed logs.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_data": {"type": "object", "minProperties": 1},
                    "pod_name": {"type": "string", "minLength": 1},
                },
                "required": ["log_data", "pod_name"],
            },
        },
        {
            "name": "detect_error_patterns",
            "func": detect_error_patterns,
            "description": "Detect common error patterns across engineering logs.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "error_logs": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                    "pattern_library": {"type": ["object", "null"]},
                    "min_occurrences": {"type": "integer", "minimum": 1, "default": 2},
                },
                "required": ["error_logs"],
            },
        },
        {
            "name": "suggest_fixes",
            "func": suggest_fixes,
            "description": "Suggest fixes for a detected engineering error pattern.",
            "input_schema": {
                "type": "object",
                "properties": {"error_pattern": {"type": "string", "minLength": 1}},
                "required": ["error_pattern"],
            },
        },
        {
            "name": "coordinate_auto_healing",
            "func": coordinate_auto_healing,
            "description": "Coordinate auto-healing actions from a detected error report.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "error_report": {"type": "object", "minProperties": 1},
                    "healing_strategies": {"type": ["array", "null"], "items": {"type": "object"}},
                    "dry_run": {"type": "boolean", "default": True},
                },
                "required": ["error_report"],
            },
        },
        {
            "name": "monitor_healing_effectiveness",
            "func": monitor_healing_effectiveness,
            "description": "Monitor success rates for auto-healing actions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "healing_history": {"type": "array", "minItems": 1, "items": {"type": "object"}},
                },
                "required": ["healing_history"],
            },
        },
    ]

    for registration in registrations:
        manager.register_tool(
            category="software_engineering_tools",
            name=registration["name"],
            func=registration["func"],
            description=registration["description"],
            input_schema=registration["input_schema"],
            runtime="fastapi",
            tags=["native", "mcpp", "software-engineering-tools"],
        )
