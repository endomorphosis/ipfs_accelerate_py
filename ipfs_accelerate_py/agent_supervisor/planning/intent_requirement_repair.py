"""Pure bounded repair nominations from source-bound public observations.

An observation is a historical projection, never a lease or execution grant.
Repair packets nominate the existing signed write scope; unchanged task and
intent records remain the only input to any later independent admission.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

from ..proof.formal_verification_contracts import content_identity
from .residual_llm_packet import ResidualLlmPacket, ResidualLlmPacketError

SCHEMA = "intent-requirement-repair-proposal@1"
MAX_TASKS = 16
MAX_REQUIREMENTS = 256
MAX_PROPOSAL_BYTES = 262_144
MAX_OBSERVATION_BYTES = 8_000_000
_FALSE_FIELDS = ("semantic_alignment_verified", "source_semantics_verified",
    "proof_authority", "execution_authority", "completion_authority", "canonical_state_mutated")
_STATES = {"unobserved", "failed", "missing_outputs", "stale", "public_checks_passed", "not_measured"}


class IntentRequirementRepairError(ValueError):
    """The observation cannot support a bounded non-authoritative nomination."""


def _checked_observation(observation):
    if not isinstance(observation, Mapping):
        raise IntentRequirementRepairError("requirement observation object required")
    try:
        raw = json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) > MAX_OBSERVATION_BYTES:
            raise ValueError("observation exceeds byte bound")
        value = json.loads(raw)
        cid = value.pop("observation_cid")
        if (value["schema"] != "intent-requirement-observation@1"
                or cid != content_identity(value)
                or any(value.get(key) is not False for key in _FALSE_FIELDS)
                or value.get("official_reward") is not None or value.get("provider_calls") != 0
                or value.get("measurement_scope") != "public_validation_and_output_presence"
                or value.get("native_population_complete") is not True):
            raise ValueError("observation binding or authority flags differ")
        revision = {key: value[key] for key in (
            "manifest_cid", "planning_receipt_cid", "graph_cid", "contract_cid", "ledger_sha256")}
        if value["intent_revision_cid"] != content_identity(revision):
            raise ValueError("immutable intent revision differs")
        tasks, requirements = value["tasks"], value["requirements"]
        if not (isinstance(tasks, list) and 1 <= len(tasks) <= MAX_TASKS
                and isinstance(requirements, list) and len(requirements) <= MAX_REQUIREMENTS):
            raise ValueError("observation population exceeds bounds")
        by_task = {task["task_cid"]: task for task in tasks}
        by_req = {row["requirement_id"]: row for row in requirements}
        if len(by_task) != len(tasks) or len(by_req) != len(requirements):
            raise ValueError("duplicate observation identity")
        for task in tasks:
            dependencies = task["dependency_task_cids"]
            if (not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies))
                    or not set(dependencies) <= set(by_task) or task["task_cid"] in dependencies
                    or not task["outputs"] or not task["validations"]):
                raise ValueError("task structure differs from bounded projection")
        visiting, visited = set(), set()
        def visit(cid):
            if cid in visiting:
                raise ValueError("cyclic observed dependency graph")
            if cid in visited:
                return
            visiting.add(cid)
            for prior in by_task[cid]["dependency_task_cids"]:
                visit(prior)
            visiting.remove(cid)
            visited.add(cid)
        for task_cid in by_task:
            visit(task_cid)
        for row in requirements:
            if (row["measurement_status"] not in _STATES
                    or row.get("source_semantics_verified") is not False
                    or row.get("source_semantic_status") != "unresolved"
                    or not set(row["task_cids"]) <= set(by_task)):
                raise ValueError("requirement measurement differs")
        residual = sorted(row["requirement_id"] for row in requirements
            if row["measurement_status"] not in {"public_checks_passed", "not_measured"})
        if value["residual_requirement_ids"] != residual:
            raise ValueError("residual population differs")
        value["observation_cid"] = cid
        return value, by_task, by_req
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise IntentRequirementRepairError("intact bounded public requirement observation required") from exc


def build_intent_requirement_repair_proposal(observation: Mapping) -> dict:
    """Nominate checks or repairs; never dispatch, admit, or mutate a task."""
    value, tasks, requirements = _checked_observation(observation)
    residual = value["residual_requirement_ids"]
    direct = set()
    for key in residual:
        row = requirements[key]
        direct.update(ref["task_cid"] for field in (
            "failed_validation_refs", "stale_validation_refs", "unobserved_validation_refs")
            for ref in row[field])
        direct.update(cid for cid in row["task_cids"] if any(
            output["path"] in row["missing_output_paths"] and not output["present"]
            for output in tasks[cid]["outputs"]))
    affected = set(direct)
    while True:
        expanded = affected | {cid for cid, task in tasks.items()
            if set(task["dependency_task_cids"]) & affected}
        if expanded == affected:
            break
        affected = expanded
    nominations = []
    for cid in sorted(affected):
        task = tasks[cid]
        linked = sorted(key for key, row in requirements.items() if cid in row["task_cids"])
        actual_failed = any(check["status"] == "failed" for check in task["validations"])
        missing = sorted(output["path"] for output in task["outputs"] if not output["present"])
        # Declared creations are normally absent before the first observation.
        # Nominate edits only once an actual check failed or all checks were
        # observed and a required output is still absent.
        observed = all(check["status"] in {"passed", "failed"} for check in task["validations"])
        repair = actual_failed or bool(missing and observed)
        work_kind = "repair" if repair else "validation"
        signed_checks = [{key: check[key] for key in (
            "validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}
            for check in task["validations"]]
        evidence = sorted({check["evidence_digest"] for check in task["validations"]
            if check.get("evidence_digest")})
        packet, packet_error = None, None
        write_paths = sorted({output["path"] for output in task["outputs"]}) if repair else []
        if repair:
            try:
                packet = ResidualLlmPacket(
                    task_id=task["task_cid"], repository_id=value["repository_id"],
                    tree_id=value["current_source_tree_id"], write_paths=tuple(write_paths),
                    obligation_ids=tuple(linked),
                    counterexample_capsule={"target_ids": linked, "counterexample_ids": evidence,
                        "observation_ref": value["observation_cid"], "missing_output_paths": missing},
                    validation_commands=tuple(json.dumps(check, sort_keys=True, separators=(",", ":"))
                        for check in signed_checks), policy_id=value["policy_id"],
                    policy_revision=value["intent_revision_cid"],
                    acceptance_ids=tuple(row["criterion_key"] for row in task["acceptance"]),
                    authority_roots={"intent_revision": value["intent_revision_cid"],
                        "observation": value["observation_cid"], "task_contract": task["contract_cid"]},
                ).to_dict()
            except ResidualLlmPacketError as exc:
                # Never truncate a command/path/obligation to make it fit.
                packet_error = exc.reason_code
        nominations.append({"task_cid": cid, "task_key": task["task_key"],
            "task_revision": task["revision"], "task_contract_cid": task["contract_cid"],
            "work_kind": work_kind, "requirement_ids": linked, "write_paths": write_paths,
            "validation_checks": signed_checks, "missing_output_paths": missing,
            "dependency_task_cids": task["dependency_task_cids"],
            "depends_on_affected_task": bool(set(task["dependency_task_cids"]) & affected),
            "dependency_only_nomination": cid not in direct,
            "residual_packet": packet, "packet_unavailable_reason": packet_error,
            "nomination_only": True, "write_authority": False, "execution_authority": False,
            "completion_authority": False, "source_semantics_verified": False})
    result = {"schema": SCHEMA, "observation_cid": value["observation_cid"],
        "intent_revision_cid": value["intent_revision_cid"], "graph_cid": value["graph_cid"],
        "current_source_tree_id": value["current_source_tree_id"], "residual_requirement_ids": residual,
        "direct_task_cids": sorted(direct), "affected_task_cids": sorted(affected),
        "unmeasured_requirement_ids": value["unmeasured_requirement_ids"], "nominations": nominations,
        "disposition": "review_nomination" if affected else "no_measured_residual",
        "requires_fresh_observation_before_admission": True, "intent_revision_preserved": True,
        "nomination_only": True, "provider_calls": 0, "official_reward": None,
        "write_authority": False, **{key: False for key in _FALSE_FIELDS}}
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > MAX_PROPOSAL_BYTES:
        raise IntentRequirementRepairError("repair proposal exceeds byte bound")
    return {**result, "proposal_cid": content_identity(result)}
