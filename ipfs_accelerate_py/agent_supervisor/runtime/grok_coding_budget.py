"""Explicit native coding-turn bounds; no continuation or completion authority."""
from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_GROK_CODING_MAX_TURNS = 128
SOURCE_PATHS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_coding_budget.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/llm_router.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
)


def validate_grok_coding_max_turns(value, *, provider=None, purpose="coding"):
    if type(value) is not int or not 1 <= value <= DEFAULT_GROK_CODING_MAX_TURNS:
        raise ValueError("Grok coding maximum turns must be an integer from 1 through 128")
    if value != DEFAULT_GROK_CODING_MAX_TURNS and (
            purpose != "coding" or provider is not None and provider != "grok_cli"):
        raise ValueError("nondefault coding turns require a Grok coding invocation")
    return value


def grok_coding_selection(value):
    value = validate_grok_coding_max_turns(value)
    return {} if value == DEFAULT_GROK_CODING_MAX_TURNS else {"grok_coding_max_turns": value}


def grok_coding_runtime_matches(report, value):
    """Observe actual coding policies, preserving unavailable evidence."""
    validate_grok_coding_max_turns(value)
    if type(report) is not dict or type(report.get("provider_invocations")) is not list:
        return None
    rows = []
    for row in report["provider_invocations"]:
        if (type(row) is not dict or row.get("purpose") not in ("planning", "coding")
                or row.get("phase") not in (None, row["purpose"])):
            return False
        if row["purpose"] == "coding":
            rows.append(row)
    if not rows:
        return None
    if report.get("unreceipted_provider_attempt"):
        return False
    return all(row.get("provider") == "grok_cli" and row.get("purpose") == "coding"
        and type(row.get("provider_invocation_policy")) is dict
        and type(row["provider_invocation_policy"].get("max_turns")) is int
        and row["provider_invocation_policy"]["max_turns"] == value for row in rows)


def grok_coding_not_applicable_provider_free(report, value):
    """Recognize explicit nonapplication, never infer it from missing usage."""
    validate_grok_coding_max_turns(value)
    if (type(report) is not dict or "unreceipted_provider_attempt" in report
            or type(report.get("grok_coding_max_turns")) is not int
            or report["grok_coding_max_turns"] != value
            or report.get("implementation_route") not in ("doctor_candidate", "doctor_contract_candidate")
            or type(report.get("provider_invocations")) is not list):
        return False
    application = report.get("coding_control_application")
    if (type(application) is not dict or set(application) != {
            "schema", "status", "implementation_route", "requested", "model_coding_dispatched",
            "token_savings_established", "completion_authority"}
            or application["schema"] != "terminal-coding-control-application@1"
            or application["status"] != "not_applicable_provider_free"
            or application["implementation_route"] != report["implementation_route"]
            or type(application["requested"]) is not dict
            or type(application["requested"].get("grok_coding_max_turns")) is not int
            or application["requested"]["grok_coding_max_turns"] != value
            or any(application[key] is not False for key in (
                "model_coding_dispatched", "token_savings_established", "completion_authority"))):
        return False
    return all(type(row) is dict and row.get("purpose") == "planning"
        and row.get("phase") in (None, "planning") for row in report["provider_invocations"])


def require_grok_coding_archive(manifest, value):
    if validate_grok_coding_max_turns(value) == DEFAULT_GROK_CODING_MAX_TURNS:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("Grok coding turn bound requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("Grok coding turn bound requires a newly qualified runtime archive")
