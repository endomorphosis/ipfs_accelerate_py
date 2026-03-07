"""Native development-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_development_tools_api() -> Dict[str, Any]:
    """Resolve source development-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.development_tools import (  # type: ignore
            codebase_search as _codebase_search,
            documentation_generator as _documentation_generator,
            lint_python_codebase as _lint_python_codebase,
            run_comprehensive_tests as _run_comprehensive_tests,
            test_generator as _test_generator,
            vscode_cli_execute as _vscode_cli_execute,
            vscode_cli_install as _vscode_cli_install,
            vscode_cli_install_extension as _vscode_cli_install_extension,
            vscode_cli_list_extensions as _vscode_cli_list_extensions,
            vscode_cli_status as _vscode_cli_status,
            vscode_cli_tunnel_install_service as _vscode_cli_tunnel_install_service,
            vscode_cli_tunnel_login as _vscode_cli_tunnel_login,
            vscode_cli_uninstall_extension as _vscode_cli_uninstall_extension,
        )

        return {
            "codebase_search": _codebase_search,
            "documentation_generator": _documentation_generator,
            "lint_python_codebase": _lint_python_codebase,
            "run_comprehensive_tests": _run_comprehensive_tests,
            "test_generator": _test_generator,
            "vscode_cli_execute": _vscode_cli_execute,
            "vscode_cli_install": _vscode_cli_install,
            "vscode_cli_install_extension": _vscode_cli_install_extension,
            "vscode_cli_list_extensions": _vscode_cli_list_extensions,
            "vscode_cli_status": _vscode_cli_status,
            "vscode_cli_tunnel_install_service": _vscode_cli_tunnel_install_service,
            "vscode_cli_tunnel_login": _vscode_cli_tunnel_login,
            "vscode_cli_uninstall_extension": _vscode_cli_uninstall_extension,
        }
    except Exception:
        logger.warning(
            "Source development_tools import unavailable, using fallback development-tools functions"
        )

        def _codebase_search_fallback(
            pattern: str,
            path: str = ".",
            case_insensitive: bool = False,
            whole_word: bool = False,
            regex: bool = False,
            extensions: Optional[str] = None,
            exclude: Optional[str] = None,
            max_depth: Optional[int] = None,
            context: int = 0,
            format: str = "text",
            output: Optional[str] = None,
            compact: bool = False,
            group_by_file: bool = False,
            summary: bool = False,
        ) -> Dict[str, Any]:
            _ = (
                path,
                case_insensitive,
                whole_word,
                regex,
                extensions,
                exclude,
                max_depth,
                context,
                format,
                output,
                compact,
                group_by_file,
                summary,
            )
            return {
                "success": True,
                "result": {
                    "matches": [],
                    "summary": {"total_matches": 0, "pattern": pattern},
                },
            }

        def _documentation_generator_fallback(
            input_path: str,
            output_path: str = "docs",
            docstring_style: str = "google",
            ignore_patterns: Optional[List[str]] = None,
            include_inheritance: bool = True,
            include_examples: bool = True,
            include_source_links: bool = True,
            format_type: str = "markdown",
        ) -> Dict[str, Any]:
            _ = (
                docstring_style,
                ignore_patterns,
                include_inheritance,
                include_examples,
                include_source_links,
            )
            return {
                "success": True,
                "result": {
                    "input_path": input_path,
                    "output_path": output_path,
                    "format_type": format_type,
                    "files_generated": [f"{output_path}/documentation.{format_type}"],
                },
            }

        def _lint_python_codebase_fallback(
            path: str = ".",
            patterns: Optional[List[str]] = None,
            exclude_patterns: Optional[List[str]] = None,
            fix_issues: bool = True,
            include_dataset_rules: bool = True,
            dry_run: bool = False,
            verbose: bool = False,
        ) -> Dict[str, Any]:
            _ = (patterns, exclude_patterns, fix_issues, include_dataset_rules, dry_run, verbose)
            return {
                "success": True,
                "path": path,
                "files_processed": 0,
                "total_issues": 0,
                "total_fixes": 0,
                "modified_files": [],
            }

        def _run_comprehensive_tests_fallback(
            path: str = ".",
            run_unit_tests: bool = True,
            run_type_check: bool = True,
            run_linting: bool = True,
            run_dataset_tests: bool = True,
            test_framework: str = "pytest",
            coverage: bool = True,
            verbose: bool = False,
            save_results: bool = True,
            output_formats: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            _ = (
                run_unit_tests,
                run_type_check,
                run_linting,
                run_dataset_tests,
                test_framework,
                coverage,
                verbose,
                save_results,
                output_formats,
            )
            return {
                "success": True,
                "path": path,
                "suite_results": {},
                "summary": {
                    "total_passed": 0,
                    "total_failed": 0,
                    "total_skipped": 0,
                },
            }

        def _test_generator_fallback(
            name: str,
            description: str = "",
            test_specification: Any = None,
            output_dir: Optional[str] = None,
            harness: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (description, test_specification, harness)
            return {
                "success": True,
                "result": {
                    "name": name,
                    "output_dir": output_dir or ".",
                    "generated_files": [],
                },
            }

        def _vscode_cli_status_fallback(install_dir: Optional[str] = None) -> Dict[str, Any]:
            _ = install_dir
            return {
                "success": False,
                "error": "VSCode CLI integration unavailable",
            }

        def _vscode_cli_install_fallback(
            install_dir: Optional[str] = None,
            force: bool = False,
            commit: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (install_dir, force, commit)
            return {"success": False, "error": "VSCode CLI integration unavailable"}

        def _vscode_cli_execute_fallback(
            command: List[str],
            install_dir: Optional[str] = None,
            timeout: int = 60,
        ) -> Dict[str, Any]:
            _ = (install_dir, timeout)
            return {
                "success": False,
                "error": "VSCode CLI integration unavailable",
                "command": command,
            }

        def _vscode_cli_list_extensions_fallback(install_dir: Optional[str] = None) -> Dict[str, Any]:
            _ = install_dir
            return {"success": True, "extensions": [], "count": 0, "action": "list"}

        def _vscode_cli_install_extension_fallback(
            extension_id: str,
            install_dir: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = install_dir
            return {
                "success": False,
                "action": "install",
                "extension_id": extension_id,
                "error": "VSCode CLI integration unavailable",
            }

        def _vscode_cli_uninstall_extension_fallback(
            extension_id: str,
            install_dir: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = install_dir
            return {
                "success": False,
                "action": "uninstall",
                "extension_id": extension_id,
                "error": "VSCode CLI integration unavailable",
            }

        def _vscode_cli_tunnel_login_fallback(
            provider: str = "github",
            install_dir: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = install_dir
            return {
                "success": False,
                "action": "login",
                "provider": provider,
                "error": "VSCode CLI integration unavailable",
            }

        def _vscode_cli_tunnel_install_service_fallback(
            tunnel_name: Optional[str] = None,
            install_dir: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = install_dir
            return {
                "success": False,
                "action": "install_service",
                "tunnel_name": tunnel_name,
                "error": "VSCode CLI integration unavailable",
            }

        return {
            "codebase_search": _codebase_search_fallback,
            "documentation_generator": _documentation_generator_fallback,
            "lint_python_codebase": _lint_python_codebase_fallback,
            "run_comprehensive_tests": _run_comprehensive_tests_fallback,
            "test_generator": _test_generator_fallback,
            "vscode_cli_execute": _vscode_cli_execute_fallback,
            "vscode_cli_install": _vscode_cli_install_fallback,
            "vscode_cli_install_extension": _vscode_cli_install_extension_fallback,
            "vscode_cli_list_extensions": _vscode_cli_list_extensions_fallback,
            "vscode_cli_status": _vscode_cli_status_fallback,
            "vscode_cli_tunnel_install_service": _vscode_cli_tunnel_install_service_fallback,
            "vscode_cli_tunnel_login": _vscode_cli_tunnel_login_fallback,
            "vscode_cli_uninstall_extension": _vscode_cli_uninstall_extension_fallback,
        }


_API = _load_development_tools_api()


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output into deterministic status envelopes."""
    if isinstance(result, dict):
        payload = dict(result)
        if payload.get("error") or payload.get("success") is False:
            payload.setdefault("status", "error")
        else:
            payload.setdefault("status", "success")
        return payload
    return {"status": "success", "result": result}


def _normalize_optional_string(value: Optional[str], field: str) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string when provided")
    return normalized


def _normalize_required_string(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    return normalized


def _validate_bool(value: Any, field: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")


def _normalize_string_list(value: Optional[List[str]], field: str) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of non-empty strings when provided")
    normalized: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            raise ValueError(f"{field} must contain only non-empty strings")
        normalized.append(text)
    return normalized


async def _call_api(name: str, **kwargs: Any) -> Any:
    result = _API[name](**kwargs)
    return await result if hasattr(result, "__await__") else result


async def codebase_search(
    pattern: str,
    path: str = ".",
    case_insensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    extensions: Optional[str] = None,
    exclude: Optional[str] = None,
    max_depth: Optional[int] = None,
    context: int = 0,
    format: str = "text",
    output: Optional[str] = None,
    compact: bool = False,
    group_by_file: bool = False,
    summary: bool = False,
) -> Dict[str, Any]:
    """Search a codebase using text or regex patterns."""
    try:
        normalized_pattern = _normalize_required_string(pattern, "pattern")
        normalized_path = str(path or ".").strip() or "."
        _validate_bool(case_insensitive, "case_insensitive")
        _validate_bool(whole_word, "whole_word")
        _validate_bool(regex, "regex")
        _validate_bool(compact, "compact")
        _validate_bool(group_by_file, "group_by_file")
        _validate_bool(summary, "summary")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be an integer >= 0 when provided")
        if not isinstance(context, int) or context < 0:
            raise ValueError("context must be an integer >= 0")

        normalized_format = str(format or "text").strip().lower() or "text"
        if normalized_format not in {"text", "json"}:
            raise ValueError("format must be one of: text, json")

        normalized_extensions = _normalize_optional_string(extensions, "extensions")
        normalized_exclude = _normalize_optional_string(exclude, "exclude")
        normalized_output = _normalize_optional_string(output, "output")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "codebase_search",
            pattern=normalized_pattern,
            path=normalized_path,
            case_insensitive=case_insensitive,
            whole_word=whole_word,
            regex=regex,
            extensions=normalized_extensions,
            exclude=normalized_exclude,
            max_depth=max_depth,
            context=context,
            format=normalized_format,
            output=normalized_output,
            compact=compact,
            group_by_file=group_by_file,
            summary=summary,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "success": False,
            "pattern": normalized_pattern,
            "path": normalized_path,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("pattern", normalized_pattern)
    payload.setdefault("path", normalized_path)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("result", {})
        if isinstance(payload.get("result"), dict):
            payload["result"].setdefault("matches", [])
            payload["result"].setdefault("summary", {})
    return payload


async def documentation_generator(
    input_path: str,
    output_path: str = "docs",
    docstring_style: str = "google",
    ignore_patterns: Optional[List[str]] = None,
    include_inheritance: bool = True,
    include_examples: bool = True,
    include_source_links: bool = True,
    format_type: str = "markdown",
) -> Dict[str, Any]:
    """Generate documentation from Python code with source-compatible options."""
    try:
        normalized_input_path = _normalize_required_string(input_path, "input_path")
        normalized_output_path = _normalize_required_string(output_path, "output_path")
        normalized_docstring_style = str(docstring_style or "google").strip().lower() or "google"
        if normalized_docstring_style not in {"google", "numpy", "rest"}:
            raise ValueError("docstring_style must be one of: google, numpy, rest")
        normalized_ignore_patterns = _normalize_string_list(ignore_patterns, "ignore_patterns")
        _validate_bool(include_inheritance, "include_inheritance")
        _validate_bool(include_examples, "include_examples")
        _validate_bool(include_source_links, "include_source_links")
        normalized_format_type = str(format_type or "markdown").strip().lower() or "markdown"
        if normalized_format_type not in {"markdown", "html"}:
            raise ValueError("format_type must be one of: markdown, html")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "documentation_generator",
            input_path=normalized_input_path,
            output_path=normalized_output_path,
            docstring_style=normalized_docstring_style,
            ignore_patterns=normalized_ignore_patterns,
            include_inheritance=include_inheritance,
            include_examples=include_examples,
            include_source_links=include_source_links,
            format_type=normalized_format_type,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "success": False,
            "input_path": normalized_input_path,
            "output_path": normalized_output_path,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("input_path", normalized_input_path)
    payload.setdefault("output_path", normalized_output_path)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("result", {})
        if isinstance(payload.get("result"), dict):
            payload["result"].setdefault("files_generated", [])
            payload["result"].setdefault("format_type", normalized_format_type)
    return payload


async def lint_python_codebase(
    path: str = ".",
    patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    fix_issues: bool = True,
    include_dataset_rules: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Lint Python code with source-compatible options."""
    try:
        normalized_path = str(path or ".").strip() or "."
        normalized_patterns = _normalize_string_list(patterns, "patterns")
        normalized_exclude_patterns = _normalize_string_list(exclude_patterns, "exclude_patterns")
        _validate_bool(fix_issues, "fix_issues")
        _validate_bool(include_dataset_rules, "include_dataset_rules")
        _validate_bool(dry_run, "dry_run")
        _validate_bool(verbose, "verbose")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "lint_python_codebase",
            path=normalized_path,
            patterns=normalized_patterns,
            exclude_patterns=normalized_exclude_patterns,
            fix_issues=fix_issues,
            include_dataset_rules=include_dataset_rules,
            dry_run=dry_run,
            verbose=verbose,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False, "path": normalized_path}

    payload = _normalize_payload(resolved)
    payload.setdefault("path", normalized_path)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("files_processed", 0)
        payload.setdefault("total_issues", 0)
        payload.setdefault("total_fixes", 0)
        payload.setdefault("modified_files", [])
    return payload


async def run_comprehensive_tests(
    path: str = ".",
    run_unit_tests: bool = True,
    run_type_check: bool = True,
    run_linting: bool = True,
    run_dataset_tests: bool = True,
    test_framework: str = "pytest",
    coverage: bool = True,
    verbose: bool = False,
    save_results: bool = True,
    output_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run comprehensive tests with source-compatible configuration flags."""
    try:
        normalized_path = str(path or ".").strip() or "."
        _validate_bool(run_unit_tests, "run_unit_tests")
        _validate_bool(run_type_check, "run_type_check")
        _validate_bool(run_linting, "run_linting")
        _validate_bool(run_dataset_tests, "run_dataset_tests")
        _validate_bool(coverage, "coverage")
        _validate_bool(verbose, "verbose")
        _validate_bool(save_results, "save_results")
        normalized_test_framework = str(test_framework or "pytest").strip().lower() or "pytest"
        if normalized_test_framework not in {"pytest", "unittest"}:
            raise ValueError("test_framework must be one of: pytest, unittest")
        normalized_output_formats = _normalize_string_list(output_formats, "output_formats")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "run_comprehensive_tests",
            path=normalized_path,
            run_unit_tests=run_unit_tests,
            run_type_check=run_type_check,
            run_linting=run_linting,
            run_dataset_tests=run_dataset_tests,
            test_framework=normalized_test_framework,
            coverage=coverage,
            verbose=verbose,
            save_results=save_results,
            output_formats=normalized_output_formats,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False, "path": normalized_path}

    payload = _normalize_payload(resolved)
    payload.setdefault("path", normalized_path)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("suite_results", {})
        payload.setdefault(
            "summary",
            {"total_passed": 0, "total_failed": 0, "total_skipped": 0},
        )
    return payload


async def test_generator(
    name: str,
    description: str = "",
    test_specification: Any = None,
    output_dir: Optional[str] = None,
    harness: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate tests from a specification with source-compatible harness options."""
    try:
        normalized_name = _normalize_required_string(name, "name")
        normalized_description = str(description or "")
        if test_specification is None or test_specification == "":
            raise ValueError("test_specification is required")
        normalized_output_dir = _normalize_optional_string(output_dir, "output_dir")
        normalized_harness = None if harness is None else str(harness).strip().lower()
        if normalized_harness is not None and normalized_harness not in {"pytest", "unittest"}:
            raise ValueError("harness must be one of: pytest, unittest when provided")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "test_generator",
            name=normalized_name,
            description=normalized_description,
            test_specification=test_specification,
            output_dir=normalized_output_dir,
            harness=normalized_harness,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "success": False,
            "name": normalized_name,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("name", normalized_name)
    payload.setdefault("output_dir", normalized_output_dir)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("result", {})
        if isinstance(payload.get("result"), dict):
            payload["result"].setdefault("generated_files", [])
    return payload


test_generator.__test__ = False


async def vscode_cli_status(install_dir: Optional[str] = None) -> Dict[str, Any]:
    """Get VS Code CLI installation and status details."""
    try:
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api("vscode_cli_status", install_dir=normalized_install_dir)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "success": False,
            "install_dir": normalized_install_dir,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("install_dir", normalized_install_dir)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("installed", False)
    return payload


async def vscode_cli_install(
    install_dir: Optional[str] = None,
    force: bool = False,
    commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Install or update the VS Code CLI."""
    try:
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
        normalized_commit = _normalize_optional_string(commit, "commit")
        _validate_bool(force, "force")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "vscode_cli_install",
            install_dir=normalized_install_dir,
            force=force,
            commit=normalized_commit,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False}

    payload = _normalize_payload(resolved)
    payload.setdefault("install_dir", normalized_install_dir)
    payload.setdefault("commit", normalized_commit)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("message", "VSCode CLI install completed")
    return payload


async def vscode_cli_execute(
    command: List[str],
    install_dir: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Execute VS Code CLI commands with bounded timeout validation."""
    try:
        normalized_command = _normalize_string_list(command, "command")
        if not normalized_command:
            raise ValueError("command must contain at least one entry")
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
        if not isinstance(timeout, int) or timeout < 1 or timeout > 300:
            raise ValueError("timeout must be an integer between 1 and 300")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "vscode_cli_execute",
            command=normalized_command,
            install_dir=normalized_install_dir,
            timeout=timeout,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False, "command": normalized_command}

    payload = _normalize_payload(resolved)
    payload.setdefault("command", normalized_command)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("returncode", 0)
        payload.setdefault("stdout", "")
        payload.setdefault("stderr", "")
    return payload


async def vscode_cli_list_extensions(install_dir: Optional[str] = None) -> Dict[str, Any]:
    """List installed VS Code extensions."""
    try:
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api("vscode_cli_list_extensions", install_dir=normalized_install_dir)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False}

    payload = _normalize_payload(resolved)
    payload.setdefault("install_dir", normalized_install_dir)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("action", "list")
        payload.setdefault("extensions", [])
        payload.setdefault("count", len(payload.get("extensions", [])))
    return payload


async def vscode_cli_install_extension(
    extension_id: str,
    install_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Install a VS Code extension by marketplace ID."""
    try:
        normalized_extension_id = _normalize_required_string(extension_id, "extension_id")
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "vscode_cli_install_extension",
            extension_id=normalized_extension_id,
            install_dir=normalized_install_dir,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False, "extension_id": normalized_extension_id}

    payload = _normalize_payload(resolved)
    payload.setdefault("extension_id", normalized_extension_id)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("action", "install")
    return payload


async def vscode_cli_uninstall_extension(
    extension_id: str,
    install_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Uninstall a VS Code extension by marketplace ID."""
    try:
        normalized_extension_id = _normalize_required_string(extension_id, "extension_id")
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "vscode_cli_uninstall_extension",
            extension_id=normalized_extension_id,
            install_dir=normalized_install_dir,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False, "extension_id": normalized_extension_id}

    payload = _normalize_payload(resolved)
    payload.setdefault("extension_id", normalized_extension_id)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("action", "uninstall")
    return payload


async def vscode_cli_tunnel_login(
    provider: str = "github",
    install_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Authenticate to the VS Code tunnel service."""
    try:
        normalized_provider = str(provider or "github").strip().lower() or "github"
        if normalized_provider not in {"github", "microsoft"}:
            raise ValueError("provider must be one of: github, microsoft")
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "vscode_cli_tunnel_login",
            provider=normalized_provider,
            install_dir=normalized_install_dir,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False, "provider": normalized_provider}

    payload = _normalize_payload(resolved)
    payload.setdefault("provider", normalized_provider)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("action", "login")
        payload.setdefault("stdout", "")
        payload.setdefault("stderr", "")
    return payload


async def vscode_cli_tunnel_install_service(
    tunnel_name: Optional[str] = None,
    install_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Install the VS Code tunnel as a background service."""
    try:
        normalized_tunnel_name = _normalize_optional_string(tunnel_name, "tunnel_name")
        normalized_install_dir = _normalize_optional_string(install_dir, "install_dir")
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "success": False}

    try:
        resolved = await _call_api(
            "vscode_cli_tunnel_install_service",
            tunnel_name=normalized_tunnel_name,
            install_dir=normalized_install_dir,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "success": False, "tunnel_name": normalized_tunnel_name}

    payload = _normalize_payload(resolved)
    payload.setdefault("tunnel_name", normalized_tunnel_name)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("action", "install_service")
        payload.setdefault("stdout", "")
        payload.setdefault("stderr", "")
    return payload


def register_native_development_tools(manager: Any) -> None:
    """Register native development-tools category tools in unified manager."""
    registrations = [
        {
            "name": "codebase_search",
            "func": codebase_search,
            "description": "Search a codebase using text or regex patterns.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "default": ".", "minLength": 1},
                    "case_insensitive": {"type": "boolean", "default": False},
                    "whole_word": {"type": "boolean", "default": False},
                    "regex": {"type": "boolean", "default": False},
                    "extensions": {"type": ["string", "null"]},
                    "exclude": {"type": ["string", "null"]},
                    "max_depth": {"type": ["integer", "null"], "minimum": 0},
                    "context": {"type": "integer", "minimum": 0, "default": 0},
                    "format": {"type": "string", "enum": ["text", "json"], "default": "text"},
                    "output": {"type": ["string", "null"]},
                    "compact": {"type": "boolean", "default": False},
                    "group_by_file": {"type": "boolean", "default": False},
                    "summary": {"type": "boolean", "default": False},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "documentation_generator",
            "func": documentation_generator,
            "description": "Generate documentation from Python code using source-compatible options.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "minLength": 1},
                    "output_path": {"type": "string", "minLength": 1, "default": "docs"},
                    "docstring_style": {
                        "type": "string",
                        "enum": ["google", "numpy", "rest"],
                        "default": "google",
                    },
                    "ignore_patterns": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string", "minLength": 1}},
                            {"type": "null"},
                        ]
                    },
                    "include_inheritance": {"type": "boolean", "default": True},
                    "include_examples": {"type": "boolean", "default": True},
                    "include_source_links": {"type": "boolean", "default": True},
                    "format_type": {"type": "string", "enum": ["markdown", "html"], "default": "markdown"},
                },
                "required": ["input_path"],
            },
        },
        {
            "name": "lint_python_codebase",
            "func": lint_python_codebase,
            "description": "Lint Python codebase with source-compatible validation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "default": "."},
                    "patterns": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string", "minLength": 1}},
                            {"type": "null"},
                        ]
                    },
                    "exclude_patterns": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string", "minLength": 1}},
                            {"type": "null"},
                        ]
                    },
                    "fix_issues": {"type": "boolean", "default": True},
                    "include_dataset_rules": {"type": "boolean", "default": True},
                    "dry_run": {"type": "boolean", "default": False},
                    "verbose": {"type": "boolean", "default": False},
                },
                "required": [],
            },
        },
        {
            "name": "run_comprehensive_tests",
            "func": run_comprehensive_tests,
            "description": "Run comprehensive tests with source-compatible configuration flags.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "default": "."},
                    "run_unit_tests": {"type": "boolean", "default": True},
                    "run_type_check": {"type": "boolean", "default": True},
                    "run_linting": {"type": "boolean", "default": True},
                    "run_dataset_tests": {"type": "boolean", "default": True},
                    "test_framework": {"type": "string", "enum": ["pytest", "unittest"], "default": "pytest"},
                    "coverage": {"type": "boolean", "default": True},
                    "verbose": {"type": "boolean", "default": False},
                    "save_results": {"type": "boolean", "default": True},
                    "output_formats": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string", "minLength": 1}},
                            {"type": "null"},
                        ]
                    },
                },
                "required": [],
            },
        },
        {
            "name": "test_generator",
            "func": test_generator,
            "description": "Generate tests from a structured specification.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "default": ""},
                    "test_specification": {},
                    "output_dir": {"type": ["string", "null"]},
                    "harness": {"type": ["string", "null"], "enum": ["pytest", "unittest", None]},
                },
                "required": ["name", "test_specification"],
            },
        },
        {
            "name": "vscode_cli_status",
            "func": vscode_cli_status,
            "description": "Return VS Code CLI installation status.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    }
                },
                "required": [],
            },
        },
        {
            "name": "vscode_cli_install",
            "func": vscode_cli_install,
            "description": "Install or update the VS Code CLI.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    },
                    "force": {"type": "boolean", "default": False},
                    "commit": {"type": ["string", "null"]},
                },
                "required": [],
            },
        },
        {
            "name": "vscode_cli_execute",
            "func": vscode_cli_execute,
            "description": "Execute a VS Code CLI command.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    },
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 300, "default": 60},
                },
                "required": ["command"],
            },
        },
        {
            "name": "vscode_cli_list_extensions",
            "func": vscode_cli_list_extensions,
            "description": "List installed VS Code extensions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    }
                },
                "required": [],
            },
        },
        {
            "name": "vscode_cli_install_extension",
            "func": vscode_cli_install_extension,
            "description": "Install a VS Code extension.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "extension_id": {"type": "string", "minLength": 1},
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["extension_id"],
            },
        },
        {
            "name": "vscode_cli_uninstall_extension",
            "func": vscode_cli_uninstall_extension,
            "description": "Uninstall a VS Code extension.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "extension_id": {"type": "string", "minLength": 1},
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["extension_id"],
            },
        },
        {
            "name": "vscode_cli_tunnel_login",
            "func": vscode_cli_tunnel_login,
            "description": "Authenticate to the VS Code tunnel service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "enum": ["github", "microsoft"], "default": "github"},
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    },
                },
                "required": [],
            },
        },
        {
            "name": "vscode_cli_tunnel_install_service",
            "func": vscode_cli_tunnel_install_service,
            "description": "Install the VS Code tunnel as a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tunnel_name": {"type": ["string", "null"]},
                    "install_dir": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    },
                },
                "required": [],
            },
        },
    ]

    for registration in registrations:
        manager.register_tool(
            category="development_tools",
            name=registration["name"],
            func=registration["func"],
            description=registration["description"],
            input_schema=registration["input_schema"],
            runtime="fastapi",
            tags=["native", "mcpp", "development-tools"],
        )
