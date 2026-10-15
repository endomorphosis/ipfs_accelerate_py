"""Frozen opt-in Doctor coding attempts, distinct from proposal-token budgets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.candidate_round_policy import (
    DOCTOR_CODING_POLICY, DOCTOR_CODING_POLICIES, PUBLIC_VALIDATION_POLICY,
    validate_candidate_rounds, validate_doctor_coding_policy,
)

SOURCE_PATHS = (
    *("benchmarks/agent_supervisor/container_coding/" + name + ".py" for name in (
        "terminal_doctor_coding_policy", "benchmark_controls", "full_supervisor_benchmark",
        "full_supervisor_harbor_agent", "terminal_container_supervisor", "terminal_doctor_dispatch",
        "container_worker_deployment", "terminal_context_audit", "terminal_candidate_input_audit",
        "terminal_candidate_attempt_accounting", "terminal_candidate_round_observation", "terminal_source_view_observation",
        "terminal_semantic_working_set_policy", "terminal_source_view_policy", "benchmark_provider_profile",
        "terminal_task_profile", "terminal_indexed_preparation")),
    *("ipfs_accelerate_py/agent_supervisor/runtime/" + name + ".py" for name in (
        "candidate_round_policy", "doctor_candidate_context", "doctor_residual_context",
        "doctor_residual_budget", "owner_candidate_dispatch", "owner_candidate_dispatch_audit",
        "owner_candidate_round_prompt", "owner_candidate_round_context", "owner_candidate_round_checkpoint",
        "candidate_dispatch_context", "candidate_doctor_dispatch", "candidate_round_monitor", "candidate_round_lifecycle",
        "worker_source_broker", "worker_candidate_round_broker", "worker_candidate_round_client",
        "worker_candidate_artifacts", "worker_candidate_context", "worker_candidate_selection",
        "router_implementation_runner", "candidate_reply_contract", "grok_coding_budget",
        "candidate_public_validation", "terminal_task_profile", "candidate_execution",
        "local_planning_admission", "terminal_source_partition")),
)


def _public_validation_task_profile(value):
    """Read only the explicit bounded public declaration before setup work."""
    from .terminal_task_profile import (
        PUBLIC_ACCEPTANCE_SCHEMA, PUBLIC_VALIDATION_PROFILE, validate_task_profile,
    )
    if isinstance(value, Path):
        value = value.absolute()
        if (value.is_symlink() or not value.is_file()
                or value.resolve(strict=True) != value):
            raise ValueError("bounded public-validation task profile required")
        with value.open("rb") as stream:
            raw = stream.read(65537)
        if not raw or len(raw) > 65536:
            raise ValueError("bounded public-validation task profile required")
        def unique(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate public-validation task profile field")
                result[key] = item
            return result
        def nonfinite(_value):
            raise ValueError("nonfinite public-validation task profile value")
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    if (type(value) is not dict or value.get("schema") != PUBLIC_ACCEPTANCE_SCHEMA
            or value.get("public_validation_profile") != PUBLIC_VALIDATION_PROFILE):
        raise ValueError("public-validation Doctor policy requires its signed @4 largest-eigenval-public@1 task profile")
    return validate_task_profile(value)


def validate_doctor_coding_selection(candidate_rounds=1, doctor_coding_policy=None, *,
        arm=None, source_view_profile="disabled", source_edit_mode="disabled",
        numerical_formalization_profile=None, task_profile=None):
    rounds = validate_candidate_rounds(candidate_rounds)
    if rounds == 2 and arm == "no-index":
        raise ValueError("candidate rounds=2 requires an admitted semantic source broker; no-index is unsupported")
    policy = validate_doctor_coding_policy(doctor_coding_policy, candidate_rounds=rounds)
    if rounds == 2:
        if policy not in DOCTOR_CODING_POLICIES:
            raise ValueError("candidate rounds=2 requires explicit Doctor constraints and coding-attempt policy")
        if source_edit_mode != "disabled":
            raise ValueError("Doctor coding continuation does not admit mapped source edits")
        if numerical_formalization_profile is not None:
            raise ValueError("Doctor coding continuation does not admit the numerical formalization broker")
        if arm != "full" or source_view_profile == "disabled":
            raise ValueError("Doctor coding continuation requires the full indexed semantic source broker")
        from .terminal_source_view_policy import validate_source_view_profile
        validate_source_view_profile(source_view_profile, arm=arm)
        if policy == PUBLIC_VALIDATION_POLICY:
            _public_validation_task_profile(task_profile)
    return rounds


def doctor_coding_selection(candidate_rounds=1, doctor_coding_policy=None):
    validate_candidate_rounds(candidate_rounds)
    validate_doctor_coding_policy(doctor_coding_policy, candidate_rounds=candidate_rounds)
    if candidate_rounds == 2 and doctor_coding_policy not in DOCTOR_CODING_POLICIES:
        raise ValueError("two-attempt benchmark selection requires its explicit Doctor policy")
    return {} if candidate_rounds == 1 else {
        "candidate_rounds": candidate_rounds, "doctor_coding_policy": doctor_coding_policy}


def doctor_coding_observation(candidate_rounds=1, doctor_coding_policy=None):
    selected = doctor_coding_selection(candidate_rounds, doctor_coding_policy)
    if not selected:
        return selected
    return {**selected, "doctor_coding_policy_observation": {
        "schema": "terminal-doctor-coding-policy-observation@1", "policy": doctor_coding_policy,
        "max_coding_attempts": 2, "original_deadline_preserved": True,
        "provider_tokens_and_cost_observation_only": True,
        "native_token_cap_enforced": False, "provider_cost_cap_enforced": False,
        "residual_proposal_budget_unchanged": True, "completion_authority": False}}


def doctor_coding_runtime_declared_matches(report, candidate_rounds, doctor_coding_policy):
    """Compare declarations exactly; this does not prove execution enforcement."""
    if type(report) is not dict:
        return False
    expected = doctor_coding_observation(candidate_rounds, doctor_coding_policy)
    observed = {key: report.get(key) for key in expected}
    return json.dumps(observed, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False)


def require_doctor_coding_archive(manifest, candidate_rounds=1, doctor_coding_policy=None):
    if not doctor_coding_selection(candidate_rounds, doctor_coding_policy):
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("Doctor coding continuation requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("Doctor coding continuation requires a newly qualified runtime archive")
