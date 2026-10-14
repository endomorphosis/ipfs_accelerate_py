"""Frozen opt-in query references for the full native Codex coding route."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

DEFAULT_QUERY_REFERENCE_POLICY = "legacy"
REQUIRED_INSTRUCTION_REF_POLICY = "required-instruction-ref@1"
QUERY_REFERENCE_POLICIES = (DEFAULT_QUERY_REFERENCE_POLICY, REQUIRED_INSTRUCTION_REF_POLICY)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_query_reference_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/query_reference_policy.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_query_reference.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_input_view.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/codex_usage_receipt.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/codex_interaction_telemetry.py",
    "ipfs_accelerate_py/cli_runtime/cli_metadata.py",
    "ipfs_accelerate_py/llm_router.py",
)


def validate_query_reference_policy(value: str, *, arm: str | None = None,
                                provider: str | None = None,
                                semantic_metadata_view: str = "legacy",
                                source_input_policy: str = "legacy") -> str:
    if type(value) is not str or value not in QUERY_REFERENCE_POLICIES:
        raise ValueError("explicit supported query reference policy required")
    if value != DEFAULT_QUERY_REFERENCE_POLICY:
        if arm is not None and arm != "full":
            raise ValueError("query reference policy requires the full indexed arm")
        if provider is not None and provider != "codex_cli":
            raise ValueError("query reference policy requires the Codex router route")
    if value != DEFAULT_QUERY_REFERENCE_POLICY and (semantic_metadata_view != "legacy" or source_input_policy != "legacy"):
        raise ValueError("query reference v1 excludes metadata and source input selection")
    return value


def query_reference_selection(value: str) -> dict:
    """Omit the legacy default so historical config/control bytes stay stable."""
    return {} if validate_query_reference_policy(value) == DEFAULT_QUERY_REFERENCE_POLICY else {
        "query_reference_policy": value,
    }


def require_query_reference_archive(manifest: dict, value: str) -> None:
    if validate_query_reference_policy(value) == DEFAULT_QUERY_REFERENCE_POLICY:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("query reference policy requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("query reference policy requires a newly qualified runtime archive")


def query_reference_dispatch_observation(report, requested_policy: str) -> dict:
    """Join source inclusion receipts to retained exact-input audits, without tool replay."""
    from ipfs_accelerate_py.agent_supervisor.runtime.query_reference_policy import validate_bound_receipt
    from .terminal_context_audit import SCHEMA as AUDIT_SCHEMA, query_reference_audit_binding_sha256

    validate_query_reference_policy(requested_policy)
    report = report if type(report) is dict else {}
    declared_matches = report.get("query_reference_policy") == requested_policy
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
    planning_unchanged = all(not any(key in row for key in (
        "query_reference_policy", "query_reference_view", "query_reference_selected", "query_reference_dispatched",
        "query_reference_provider_attempted")) for row in population
                             if type(row) is dict and row.get("phase") == "planning")
    if not planning_unchanged:
        problems.append("planning_query_reference_policy_present")
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
        and audit.get("schema") == AUDIT_SCHEMA
        and all(type(item) is dict and valid_id(item.get("invocation_id"))
                and item["invocation_id"] in identities for item in matches)
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
        identity, binding = row.get("invocation_id"), row.get("query_reference_view")
        valid_binding = False
        audit_binding_sha256 = None
        try:
            workspace = row.get("workspace")
            if (type(workspace) is not str or not 0 < len(workspace.encode()) <= 4096
                    or not Path(workspace).is_absolute() or ".." in Path(workspace).parts
                    or str(Path(workspace)) != workspace):
                raise ValueError("invalid coding workspace")
            validate_bound_receipt(row.get("query_reference_policy"), binding,
                semantic_metadata_view="legacy" if row.get("semantic_metadata_view") is None else "common-bindings@1",
                source_input_policy=row.get("source_input_policy", "legacy"))
            audit_binding_sha256 = query_reference_audit_binding_sha256(row)
            valid_binding = (row.get("schema") == "router-implementation-invocation@1"
                             and row.get("query_reference_policy") == requested_policy
                             and type(row.get("query_reference_selected")) is bool
                             and row.get("query_reference_selected") is (row.get("semantic_translation") is not None)
                             and row.get("query_reference_dispatched") is row.get("query_reference_selected")
                             and row.get("query_reference_provider_attempted") is True
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
                and item.get("query_reference_audit_binding_sha256") == audit_binding_sha256
                and type(audit_binding_sha256) is str
                and item.get("model_prompt_sha256") == row.get("model_prompt_sha256")
                and type(item.get("model_prompt_bytes")) is int
                and item.get("model_prompt_bytes") == row.get("model_prompt_bytes")
                and type(checks) is dict and checks.get("query_reference_projection") is True and checks.get("query_reference_selection") is True
                and checks.get("query_reference_dispatch") is True
                and checks.get("query_reference_provider_attempted") is True
                and checks.get("model_prompt_sha256") is True and checks.get("model_prompt_bytes") is True
                and checks.get("workspace_advisory_sha256") is True and checks.get("workspace_advisory_bytes") is True
                and all(value is True for value in checks.values()))
        verified = (valid_binding and audit_population and len(per_audit) == 1
            and per_audit[0].get("model_input_verified") is True
            and per_audit[0].get("status") == "verified_exact_model_input"
            and bool(per_matches) and all(verified_match(item) for item in per_matches))
        observations.append({"invocation_id": identity if valid_id(identity) else None,
            "policy_binding_valid": valid_binding,
            "reference_selected": row.get("query_reference_selected") is True,
            "reference_dispatched": row.get("query_reference_dispatched") is True,
            "provider_attempted": row.get("query_reference_provider_attempted") is True,
            "retained_exact_input_audit_verified": bool(verified)})
    verified = bool(observations) and not problems and all(
        row["retained_exact_input_audit_verified"] for row in observations)
    selected = sum(row["retained_exact_input_audit_verified"] and row["reference_selected"] for row in observations)
    fallback = sum(row["retained_exact_input_audit_verified"] and not row["reference_selected"] for row in observations)
    effective = verified and selected == len(observations)
    status = ("verified_input" if effective else "verified_native_fallback" if verified and selected == 0
              else "verified_mixed_selection" if verified else "unverified")
    return {"schema": "terminal-query-reference-dispatch-observation@1", "requested_policy": requested_policy,
        "requested_policy_matches_report": declared_matches, "planning_policy_absent": planning_unchanged,
        "coding_receipts_seen": len(coding), "invocations": observations,
        "all_coding_inputs_verified_by_retained_audit": verified,
        "effective_input_observed": effective, "helper_adoption_established": False, "model_attention_established": False,
        "selected_reference_inputs": selected, "native_fallback_inputs": fallback,
        "unverified_inputs": len(observations) - selected - fallback,
        "status": status, "problems": problems,
        "source_audit_rerun": False, "provider_usage_modified": False, "token_savings_established": False}
