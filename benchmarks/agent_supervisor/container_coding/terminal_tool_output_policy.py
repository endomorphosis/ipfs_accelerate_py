"""Frozen opt-in tool output previews for the full native Codex coding route."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

DEFAULT_TOOL_OUTPUT_POLICY = "legacy"
ARTIFACT_PREVIEW_POLICY = "artifact-preview@1"
TOOL_OUTPUT_POLICIES = (DEFAULT_TOOL_OUTPUT_POLICY, ARTIFACT_PREVIEW_POLICY)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_tool_output_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/tool_output_policy.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/tool_output_artifact.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/codex_usage_receipt.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/codex_interaction_telemetry.py",
    "ipfs_accelerate_py/cli_runtime/cli_metadata.py",
    "ipfs_accelerate_py/llm_router.py",
)


def validate_tool_output_policy(value: str, *, arm: str | None = None,
                                provider: str | None = None) -> str:
    if type(value) is not str or value not in TOOL_OUTPUT_POLICIES:
        raise ValueError("explicit supported tool output policy required")
    if value != DEFAULT_TOOL_OUTPUT_POLICY:
        if arm is not None and arm != "full":
            raise ValueError("tool output policy requires the full indexed arm")
        if provider is not None and provider != "codex_cli":
            raise ValueError("tool output policy requires the Codex router route")
    return value


def tool_output_selection(value: str) -> dict:
    """Omit the legacy default so historical config/control bytes stay stable."""
    return {} if validate_tool_output_policy(value) == DEFAULT_TOOL_OUTPUT_POLICY else {
        "tool_output_policy": value,
    }


def require_tool_output_archive(manifest: dict, value: str) -> None:
    if validate_tool_output_policy(value) == DEFAULT_TOOL_OUTPUT_POLICY:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("tool output policy requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("tool output policy requires a newly qualified runtime archive")


def tool_output_dispatch_observation(report, requested_policy: str) -> dict:
    """Join advisory receipts to retained exact-input audits, without tool replay."""
    from ipfs_accelerate_py.agent_supervisor.runtime.tool_output_policy import validate_binding

    validate_tool_output_policy(requested_policy)
    report = report if type(report) is dict else {}
    declared_matches = report.get("tool_output_policy") == requested_policy
    problems = [] if declared_matches else ["requested_policy_report_mismatch"]
    if report.get("unreceipted_provider_attempt"):
        problems.append("unreceipted_provider_attempt")
    population = report.get("provider_invocations")
    if type(population) is not list or len(population) > 4096:
        problems.append("provider_invocation_population_unavailable_or_unbounded")
        population = []
    if any(type(row) is not dict or type(row.get("phase")) is not str
           or row["phase"] not in {"planning", "coding"} for row in population):
        problems.append("malformed_provider_invocation")
    coding = [row for row in population if type(row) is dict and row.get("phase") == "coding"]
    planning_unchanged = all("tool_output_policy" not in row for row in population
                             if type(row) is dict and row.get("phase") == "planning")
    if not planning_unchanged:
        problems.append("planning_tool_output_policy_present")
    audit = report.get("context_input_audit")
    audit = audit if type(audit) is dict else {}
    audit_rows, matches = audit.get("invocations"), audit.get("matches")
    audit_rows = audit_rows if type(audit_rows) is list and len(audit_rows) <= 16 else []
    matches = matches if type(matches) is list and len(matches) <= 512 else []
    valid_id = lambda value: type(value) is str and 0 < len(value) <= 4096
    identities = [row.get("invocation_id") for row in coding]
    distinct = all(valid_id(value) for value in identities) and len(set(identities)) == len(identities)
    audit_ids = [row.get("invocation_id") for row in audit_rows if type(row) is dict]
    audit_population = (distinct and len(audit_rows) == len(coding) == len(audit_ids)
        and all(valid_id(value) for value in audit_ids) and set(audit_ids) == set(identities)
        and len(set(audit_ids)) == len(audit_ids)
        and audit.get("status") == "verified_all_observed_model_inputs"
        and audit.get("all_observed_coding_inputs_verified") is True
        and type(audit.get("coding_receipts_seen")) is int and audit["coding_receipts_seen"] == len(coding)
        and type(audit.get("usable_coding_receipts")) is int and audit["usable_coding_receipts"] == len(coding)
        and audit.get("read_errors") == [])
    if not coding:
        problems.append("no_observed_coding_receipts")
    elif not distinct:
        problems.append("coding_invocation_identities_missing_or_duplicate")
    observations = []
    for row in coding:
        identity, binding = row.get("invocation_id"), row.get("tool_output_policy")
        valid_binding = False
        try:
            workspace = row.get("workspace")
            if type(workspace) is not str or not 0 < len(workspace) <= 4096 or not Path(workspace).is_absolute():
                raise ValueError("invalid coding workspace")
            validate_binding(binding, workspace=Path(workspace))
            valid_binding = (binding["requested_policy"] == requested_policy
                             and row.get("provider") == "codex_cli" and row.get("purpose") == "coding"
                             and type(row.get("model_prompt_sha256")) is str
                             and re.fullmatch(r"[0-9a-f]{64}", row["model_prompt_sha256"]) is not None
                             and type(row.get("model_prompt_bytes")) is int
                             and 0 < row["model_prompt_bytes"] <= 256_000)
        except (ValueError, TypeError, KeyError):
            pass
        per_audit = [item for item in audit_rows if type(item) is dict and item.get("invocation_id") == identity]
        per_matches = [item for item in matches if type(item) is dict and item.get("invocation_id") == identity]
        def verified_match(item):
            checks = item.get("model_input_checks")
            return (item.get("model_input_verified") is True
                and item.get("model_prompt_sha256") == row.get("model_prompt_sha256")
                and type(item.get("model_prompt_bytes")) is int
                and item.get("model_prompt_bytes") == row.get("model_prompt_bytes")
                and type(checks) is dict and checks.get("tool_output_helper_source") is True
                and checks.get("model_prompt_sha256") is True and checks.get("model_prompt_bytes") is True
                and checks.get("workspace_advisory_sha256") is True and checks.get("workspace_advisory_bytes") is True
                and all(value is True for value in checks.values()))
        verified = (valid_binding and audit_population and len(per_audit) == 1
            and per_audit[0].get("model_input_verified") is True
            and per_audit[0].get("status") == "verified_exact_model_input"
            and bool(per_matches) and all(verified_match(item) for item in per_matches))
        observations.append({"invocation_id": identity if valid_id(identity) else None,
            "policy_binding_valid": valid_binding,
            "retained_exact_input_audit_verified": bool(verified)})
    verified = bool(observations) and not problems and all(
        row["retained_exact_input_audit_verified"] for row in observations)
    return {"schema": "terminal-tool-output-dispatch-observation@1", "requested_policy": requested_policy,
        "requested_policy_matches_report": declared_matches, "planning_policy_absent": planning_unchanged,
        "coding_receipts_seen": len(coding), "invocations": observations,
        "all_coding_advisories_verified_by_retained_audit": verified,
        "effective_advisory_observed": verified, "helper_adoption_established": False,
        "status": "verified_advisory" if verified else "unverified", "problems": problems,
        "source_audit_rerun": False, "provider_usage_modified": False, "token_savings_established": False}
