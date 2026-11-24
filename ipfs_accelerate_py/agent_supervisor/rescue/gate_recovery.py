"""Durable semantic evidence for a repeatedly withheld deterministic gate.

This ledger diagnoses missing work; it never changes a task's status or a
gate's decision. Its repair packet is consumed by the fenced repair dispatcher.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, Mapping

GATE_REPEAT_LIMIT = 3
GATE_RETRY_BASE_MS = 60_000
GATE_RETRY_MAX_MS = 900_000
_SOURCE_SUFFIXES = {".py", ".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx"}


def _is_source_path(path: Path) -> bool:
    if path.suffix.casefold() in _SOURCE_SUFFIXES:
        return True
    # Reviewed descriptor/schema JSON is executable configuration. Generated
    # reports and runtime receipts must never reset a source refusal budget.
    return (path.suffix.casefold() == ".json"
            and not {"test-results", "data", "evidence"}.intersection(path.parts)
            and bool({"src", "config", "configs", "schemas"}.intersection(path.parts)))
_VOLATILE_KEYS = {
    "generated_at", "generated_at_ms", "generated_at_utc", "updated_at",
    "updated_at_ms", "created_at", "created_at_ms", "timestamp", "timestamp_ms",
    "heartbeat_at_ms", "last_seen", "last_seen_ms", "checked_at", "checked_at_ms",
    "duration_ms", "duration_seconds", "elapsed_ms", "elapsed_seconds", "pid",
    "log_path", "log_file", "stdout", "stderr",
    "logs", "log", "log_bytes", "log_size", "log_size_bytes", "mtime", "mtime_ns",
    "last_checked_at", "last_checked_at_ms", "observed_at", "observed_at_ms",
}
_VOLATILE_NORMALIZED = {re.sub(r"[^a-z0-9]", "", key.casefold()) for key in _VOLATILE_KEYS}
_FIELDS = (
    "status", "depends_on", "deterministic_command", "deterministic_repair",
    "validation", "outputs", "predicted_files", "acceptance", "conflict_policy",
)
_TRACE_KEYS = {
    "correlationid", "traceid", "requestid", "responseid", "runid", "receiptid",
    "receiptcid", "eventdagcid", "eventcid", "evidencesources", "inputfingerprint",
    "reportpath", "capturedat", "completedat", "capturestartedat", "capturefinishedat",
}


def _diagnostic(value: Any) -> Any:
    """Compare structured failure outcomes, never use this as proof validation.

    A report with coded gaps has already evaluated its receipt identities.
    New run/CID/correlation links do not resolve an unchanged coded obligation.
    Unknown JSON keeps its full non-clock meaning so arbitrary input evidence
    is not silently weakened.
    """

    stable = _stable(value)
    if not (isinstance(stable, dict) and isinstance(stable.get("schema"), str)
            and isinstance(stable.get("decision"), str) and isinstance(stable.get("gaps"), list)
            and all(isinstance(gap, dict) and isinstance(gap.get("code"), str) for gap in stable["gaps"])):
        return stable

    def project(item: Any, field: str = "") -> Any:
        if isinstance(item, dict):
            return {key: project(child, key) for key, child in item.items()
                    if re.sub(r"[^a-z0-9]", "", str(key).casefold()) not in _TRACE_KEYS}
        if isinstance(item, list):
            result = [project(child) for child in item]
            return sorted(result, key=_json) if field in {"gaps", "reasons"} else result
        return item

    return project(stable)


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in value.items()
                if re.sub(r"[^a-z0-9]", "", str(key).casefold()) not in _VOLATILE_NORMALIZED}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_PREREQUISITE_STEPS = {
    "MISSING_OWNER_POINTER": ("discover_registered_preparer", "verify_preparer_identity"),
    "MISSING_RUNTIME_WITNESS": ("locate_approved_runtime_witness", "verify_runtime_identity"),
    "MISSING_ARTIFACT_POINTER": ("resolve_retained_artifact_reference", "verify_artifact_identity"),
    "INPUT_SERIALIZATION_ERROR": ("reconstruct_from_verified_bytes", "verify_exact_reference_transport"),
    "LOSSLESS_CACHE_UNAVAILABLE": ("resolve_retained_artifact_reference", "verify_artifact_identity"),
}
_EXTERNAL_PREREQUISITES = {
    "MISSING_EXECUTION_AUTHORITY": "A current execution admission from its existing authority.",
    "MISSING_REVIEW": "An independent review from its existing review owner.",
    "OPERATOR_HOLD": "An explicit release from the owner of the hold.",
    "LOOKUP_BUDGET_EXHAUSTED": "A separately selected scope; preserve the spent budget.",
    "MISSING_LOOKUP_SCOPE": "An existing bounded discovery scope and registered handler.",
    "INVALID_PREREQUISITE_DIAGNOSTICS": "A complete bounded typed prerequisite declaration; no partial plan was admitted.",
}
_PREREQUISITE_ALIASES = {
    "MISSING_APPROVED_HISTORICAL_RUNTIME_WITNESS": "MISSING_RUNTIME_WITNESS",
    "MISSING_APPROVED_RUNTIME_WITNESS": "MISSING_RUNTIME_WITNESS",
}
MAX_PREREQUISITES = 16
_PREREQUISITE_DIAGNOSTIC_FIELDS = {"prerequisite_requirements", "prerequisite_diagnostics_sha256"}


def _declared_prerequisites(raw: str) -> list[Any]:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate prerequisite field")
            value[key] = item
        return value

    def nonfinite(_value):
        raise ValueError("nonfinite prerequisite field")

    if len(raw.encode("utf-8")) > 32768:
        raise ValueError("prerequisite declaration exceeds bound")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, list):
        raise ValueError("prerequisite declaration must be a list")
    if len(value) > 64:
        raise ValueError("prerequisite diagnostics exceed bound")
    return value


def _prerequisite(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not isinstance(value.get("code"), str):
        raise ValueError("typed prerequisite code required")
    if {"command", "argv", "shell", "script", "prompt", "instructions"}.intersection(value):
        raise ValueError("diagnostics cannot supply a recovery command")
    code = value["code"].upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", code):
        raise ValueError("invalid prerequisite code")
    code = _PREREQUISITE_ALIASES.get(code, code)
    subject = value.get("subject", code)
    if (not isinstance(subject, str) or not subject.strip()
            or len(subject.encode("utf-8")) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in subject)):
        raise ValueError("bounded prerequisite subject required")
    result = {"code": code, "subject": subject}
    if "expected_sha256" in value:
        digest = value["expected_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("exact prerequisite digest required")
        result["expected_sha256"] = digest
    # Clocks, prose, commands and self-asserted approvals are not work or authority.
    return result


def _prerequisite_task_fields(task_id: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Keep declared requirements as bounded data, including rejected-input identity."""

    result = dict(fields)
    declared = fields.get("prerequisite_requirements")
    if declared is not None and declared != "":
        try:
            normalized = sorted({_json(_prerequisite(item)) for item in _declared_prerequisites(declared)})
            result["prerequisite_requirements"] = "[" + ",".join(normalized) + "]"
        except (ValueError, TypeError, AttributeError, RecursionError):
            result["prerequisite_requirements"] = _json([dict(
                code="INVALID_PREREQUISITE_DIAGNOSTICS", subject=task_id)])
            # Identify rejected Python text losslessly even when it has no
            # valid UTF-8 encoding. This never admits those bytes as an artifact.
            result["prerequisite_diagnostics_sha256"] = hashlib.sha256(
                str(declared).encode("utf-8", errors="surrogatepass")).hexdigest()
    return result



def _read_budget_scope(value: Any, task_id: str) -> dict[str, str]:
    keys = {"parent_task_id", "parent_context_id", "budget_scope_id", "source_epoch", "selection_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("missing_or_invalid_selected_scope")
    result = dict(value)
    for item in result.values():
        if (type(item) is not str or not item or item != item.strip()
                or len(item.encode("utf-8")) > 512
                or any(ord(char) < 32 or ord(char) == 127 for char in item)):
            raise ValueError("invalid_scope_identity")
    if result["parent_task_id"] != task_id:
        raise ValueError("different_parent_task")
    if not re.fullmatch(r"[0-9a-f]{64}", result["selection_sha256"]):
        raise ValueError("invalid_selection_identity")
    return result


def _read_budget_artifact_ref(value: Any) -> dict[str, Any]:
    # Structural validation is NOT artifact authentication. The dispatcher must
    # use the existing reader on the selected bytes before consuming this data.
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256", "full9"}:
        raise ValueError("missing_artifact_reference")
    path, size, digest, full9 = (value[key] for key in ("path", "bytes", "sha256", "full9"))
    if (type(path) is not str or not Path(path).is_absolute() or ".." in Path(path).parts
            or "\0" in path or len(path.encode("utf-8")) > 8192
            or type(size) is not int or size < 1
            or type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("invalid_artifact_reference")
    keys = {"st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"}
    if not isinstance(full9, Mapping) or set(full9) != keys:
        raise ValueError("missing_artifact_full9")
    for item in full9.values():
        if type(item) is not str or not item.isascii() or not item.isdecimal() or str(int(item)) != item:
            raise ValueError("invalid_artifact_full9")
    if (full9["st_size"] != str(size) or full9["st_nlink"] != "1"
            or int(full9["st_mode"]) & 0o170000 != 0o100000):
        raise ValueError("artifact_reference_not_regular")
    return {"path": path, "bytes": size, "sha256": digest, "full9": dict(full9)}


def _runtime_read_budget_projection(
    task_id: str, observation: Any, selected_scope: Any,
) -> dict[str, Any]:
    """Observe one task's byte costs; never reserve, authenticate or admit work.

    The expected selection comes separately from the trusted current caller.
    Runtime counters stay outside task/source semantic fingerprint inputs.
    Both actual Source-ledger caps apply. Python integer comparisons do not
    wrap a large cost into a false fit, and no cap/spend is changed here.
    """
    blocked = {"code": "INVALID_READ_BUDGET_CONTEXT", "subject": task_id,
               "required_input": "Complete current task/revision/Source/budget scope, exact costs, both byte ledgers and authenticated evidence from their existing owner."}
    try:
        scope = _read_budget_scope(selected_scope, task_id)
        required_keys = {"scope", "ledger", "required_source_bytes", "authenticated_evidence"}
        if (not isinstance(observation, Mapping) or not required_keys.issubset(observation)
                or set(observation) - required_keys - {"cache_candidates"}
                or len(_json(dict(observation)).encode("utf-8")) > 262144):
            raise ValueError("missing_or_invalid_read_budget_observation")
        if _read_budget_scope(observation["scope"], task_id) != scope:
            raise ValueError("selected_scope_mismatch")
        ledger = observation["ledger"]
        keys = {"global_cap", "used", "captures", "internal_cap", "internal_used"}
        if (not isinstance(ledger, Mapping) or set(ledger) != keys
                or any(type(value) is not int or value < 0 for value in ledger.values())
                or ledger["used"] > ledger["global_cap"]
                or ledger["internal_used"] > ledger["internal_cap"]):
            raise ValueError("invalid_source_byte_ledger")
        costs = observation["required_source_bytes"]
        if (not isinstance(costs, Mapping) or set(costs) != {"global", "internal"}
                or any(type(value) is not int or value < 0 for value in costs.values())
                or costs["internal"] > costs["global"]):
            raise ValueError("missing_or_invalid_source_cost")
        evidence = observation["authenticated_evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != {"artifact_refs", "source_provenance"}:
            raise ValueError("missing_authentication_evidence")
        refs = evidence["artifact_refs"]
        if not isinstance(refs, list) or len(refs) > 256:
            raise ValueError("invalid_authentication_evidence")
        references = [_read_budget_artifact_ref(item) for item in refs]
        provenance = _read_budget_artifact_ref(evidence["source_provenance"])
        candidates = observation.get("cache_candidates", [])
        if not isinstance(candidates, list) or len(candidates) > MAX_PREREQUISITES:
            raise ValueError("invalid_cache_candidates")
        projected = []
        for candidate in candidates:
            if (not isinstance(candidate, Mapping) or set(candidate) != {
                    "operation", "required_fresh_Source_bytes", "scope", "artifact_ref", "source_provenance"}
                    or candidate["operation"] != "read_authenticated_private_artifact"
                    or type(candidate["required_fresh_Source_bytes"]) is not int
                    or candidate["required_fresh_Source_bytes"] != 0
                    or _read_budget_scope(candidate["scope"], task_id) != scope):
                raise ValueError("cache_candidate_scope_or_cost_mismatch")
            artifact = _read_budget_artifact_ref(candidate["artifact_ref"])
            if artifact not in references or _read_budget_artifact_ref(candidate["source_provenance"]) != provenance:
                raise ValueError("cache_candidate_evidence_mismatch")
            projected.append({"operation": candidate["operation"], "required_fresh_Source_bytes": 0,
                "scope": dict(scope), "artifact_ref": artifact, "source_provenance": provenance,
                "authentication_required": True, "admission_granted": False,
                "execution_authority": False, "completion_authority": False, "work_complete": False})
        remaining = {"global": ledger["global_cap"] - ledger["used"],
                     "internal": ledger["internal_cap"] - ledger["internal_used"]}
        insufficient = [key for key in ("global", "internal") if costs[key] > remaining[key]]
        requirement = None if not insufficient else {
            "code": "INSUFFICIENT_READ_BUDGET", "subject": task_id,
            "required_input": "A separately selected admissible read budget from its existing owner, or an authenticated zero-fresh-Source input path; retain every prior charge."}
        return {"requirement": requirement, "cache_candidates": projected,
                "diagnostics": {"status": "insufficient" if insufficient else "within_observed_budget",
                    "reason": "insufficient_" + "_and_".join(insufficient) if insufficient else "cost_fits_snapshot_only",
                    "scope": scope, "ledger": dict(ledger), "required_source_bytes": dict(costs),
                    "remaining_source_bytes": remaining, "authentication_required": True,
                    "admission_granted": False, "execution_authority": False,
                    "completion_authority": False, "work_complete": False, "budget_reset_allowed": False}}
    except (ValueError, TypeError, AttributeError, UnicodeError, RecursionError, OverflowError):
        return {"requirement": blocked, "cache_candidates": [],
                "diagnostics": {"status": "unknown", "reason": "read_budget_context_refused",
                    "authentication_required": True, "admission_granted": False,
                    "execution_authority": False, "completion_authority": False,
                    "work_complete": False, "budget_reset_allowed": False}}


def _attach_runtime_read_budget(plan: dict[str, Any], observation: dict[str, Any] | None) -> dict[str, Any]:
    if observation is None:
        return plan
    plan["read_budget"] = observation["diagnostics"]
    plan["cache_candidates"] = observation["cache_candidates"]
    if observation["requirement"] is not None:
        plan["external_requirements"].append(dict(observation["requirement"]))
    if plan["external_requirements"]:
        plan.update(status="external_dependency", steps=[], recheck_after=[])
    return plan

def prerequisite_recovery_plan(
    task_id: str, evidence: Mapping[str, Any], *,
    runtime_read_budget: Mapping[str, Any] | None = None,
    selected_scope: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Propose metadata work for typed missing inputs; never admit or run it.

    Existing fenced task admission must bind each proposal to a selected scope,
    its remaining budget and registered handler. Verification still requires
    actual independent evidence; neither a proposal nor a child exit satisfies
    the parent gate. Children deliberately do not depend on the blocked parent.
    """

    plan: dict[str, Any] = dict(
        schema="ipfs_accelerate_py/agent-supervisor/prerequisite-recovery-plan@1",
        parent_task_id=task_id, status="recovery_required", requirements=[], steps=[], recheck_after=[],
        external_requirements=[], work_complete=False, completion_authority=False,
        execution_authority=False, admission_granted=False, operator_hold_removal_allowed=False,
        budget_reset_allowed=False,
    )
    read_budget = (None if runtime_read_budget is None else
                   _runtime_read_budget_projection(task_id, runtime_read_budget, selected_scope))
    has_typed_input = False
    try:
        values = []
        fields = (evidence.get("tasks") or {}).get(task_id) or {}
        declared = fields.get("prerequisite_requirements")
        if declared is not None and declared != "":
            has_typed_input = True
            values.extend(_declared_prerequisites(declared))
        artifacts = evidence.get("artifacts") or {}
        if not isinstance(artifacts, Mapping):
            raise ValueError("prerequisite artifacts must be a mapping")
        typed_artifacts = 0
        for artifact in artifacts.values():
            if not isinstance(artifact, dict):
                continue
            if artifact.get("passed") is True or str(artifact.get("decision", "")).upper() == "GO":
                continue
            gaps = artifact.get("gaps", [])
            typed = "prerequisite" in str(artifact.get("schema", "")).lower()
            if typed:
                has_typed_input = True
                typed_artifacts += 1
                if typed_artifacts > 256:
                    raise ValueError("prerequisite diagnostics exceed bound")
            if not isinstance(gaps, list):
                if typed:
                    raise ValueError("typed prerequisite gaps must be a list")
                continue
            if len(gaps) > 64:
                if typed:
                    raise ValueError("prerequisite diagnostics exceed bound")
                # Ordinary diagnostics do not opt in through their size. Their
                # legacy packet/threshold behavior belongs to the original gate.
                continue
            matched = False
            for gap in gaps:
                if isinstance(gap, dict):
                    code = gap.get("code")
                    if not isinstance(code, str):
                        if typed:
                            raise ValueError("typed prerequisite code required")
                        continue
                    code = _PREREQUISITE_ALIASES.get(code.upper(), code.upper())
                    if typed or code in _PREREQUISITE_STEPS or code in _EXTERNAL_PREREQUISITES:
                        has_typed_input = True
                        matched = True
                        values.append(gap)
                elif typed:
                    raise ValueError("typed prerequisite object required")
                if len(values) > 64:
                    raise ValueError("prerequisite diagnostics exceed bound")
            if matched and not typed:
                typed_artifacts += 1
                if typed_artifacts > 256:
                    raise ValueError("prerequisite diagnostics exceed bound")
        requirements = {_json(item): item for item in map(_prerequisite, values)}
        if len(requirements) > MAX_PREREQUISITES:
            raise ValueError("prerequisite count exceeds bound")
    except (ValueError, TypeError, AttributeError, RecursionError):
        if not has_typed_input:
            if read_budget is None or read_budget["requirement"] is None:
                return None
            return _attach_runtime_read_budget(plan, read_budget)
        plan.update(status="external_dependency", external_requirements=[dict(
            code="INVALID_PREREQUISITE_DIAGNOSTICS", subject=task_id,
            required_input="A complete bounded typed prerequisite declaration; no partial plan was admitted.")])
        return _attach_runtime_read_budget(plan, read_budget)
    if not requirements:
        if read_budget is None or read_budget["requirement"] is None:
            # No budget prerequisite is NOT authentication, selection or admission.
            return None
        return _attach_runtime_read_budget(plan, read_budget)
    plan["requirements"] = [item for _, item in sorted(requirements.items())]
    prefix = task_id.partition("-")[0]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", prefix):
        prefix = "REC"
    for key, requirement in sorted(requirements.items()):
        code = requirement["code"]
        if code not in _PREREQUISITE_STEPS:
            plan["external_requirements"].append({**requirement, "required_input":
                _EXTERNAL_PREREQUISITES.get(code, "An existing registered resolver for this typed requirement.")})
            continue
        identity = hashlib.sha256(_json({"parent": task_id, **requirement}).encode("utf-8")).hexdigest()
        previous = []
        for index, kind in enumerate(_PREREQUISITE_STEPS[code], 1):
            identifier = f"{prefix}-PR-{identity}-{index}"
            plan["steps"].append(dict(
                task_id=identifier, kind=kind, depends_on=previous, requirement=dict(requirement),
                purpose="control_metadata_analysis" if index == 1 else "metadata_validation",
                requires_admission=True, scope_policy="inherit_selected_unit_only"))
            previous = [identifier]
        plan["recheck_after"].extend(previous)
    if plan["external_requirements"]:
        # A partial plan cannot authorize work around a global hold, unknown
        # policy or spent scope. Other independently selected units may proceed.
        plan.update(status="external_dependency", steps=[], recheck_after=[])
    return _attach_runtime_read_budget(plan, read_budget)


def evidence_file_fingerprint(path: Path) -> str:
    """Fingerprint file meaning without treating regenerated JSON clocks as work."""
    try:
        body = path.read_bytes()
    except OSError:
        return "absent"
    if path.suffix.casefold() == ".json":
        try:
            body = _json(_diagnostic(json.loads(body))).encode("utf-8")
        except (ValueError, UnicodeError):
            pass
    return hashlib.sha256(body).hexdigest()


def _artifact_summary(value: Any) -> dict[str, Any]:
    """Keep gate diagnostics without copying arbitrary artifact rows or bodies."""

    result: dict[str, Any] = {"semantic_sha256": hashlib.sha256(_json(value).encode("utf-8")).hexdigest()}
    if not isinstance(value, dict):
        return result
    for key in ("schema", "decision", "passed", "missing", "sha256"):
        item = value.get(key)
        if isinstance(item, (str, bool, int, float)):
            result[key] = item[:240] if isinstance(item, str) else item
    reasons = value.get("reasons")
    if isinstance(reasons, list):
        result["reason_count"] = len(reasons)
        result["reason_summaries"] = [reason[:240] for reason in reasons[:20] if isinstance(reason, str)]
    for key, item in value.items():
        if (key.endswith("_count") or key.endswith("_counts")) and isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = item
        elif key.endswith("_counts") and isinstance(item, dict):
            result[key] = {str(name): count for name, count in list(item.items())[:50]
                           if isinstance(count, (int, float)) and not isinstance(count, bool)}
    return result


def dependency_ids(fields: Mapping[str, str]) -> list[str]:
    return [part for part in re.split(r"[,\s]+", str(fields.get("depends_on") or ""))
            if part and part.casefold() not in {"none", "n/a", "-"}]


def dependency_closure(task_id: str, blocks: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Return direct dependencies before ancestors, tolerating a malformed cycle."""

    pending = dependency_ids(blocks.get(task_id) or {})
    result: list[str] = []
    seen = {task_id}
    while pending and len(result) < 256:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        pending.extend(dependency_ids(blocks.get(current) or {}))
    return result


def _referenced_paths(fields: Mapping[str, str], repo_root: Path) -> list[Path]:
    values: list[str] = []
    for key in ("outputs", "predicted_files"):
        values.extend(str(fields.get(key) or "").split(","))
    for key in ("deterministic_command", "deterministic_repair", "validation"):
        command = str(fields.get(key) or "")
        try:
            args = shlex.split(command)
        except ValueError:
            args = []
        values.extend(arg for arg in args[1:] if not any(c.isspace() for c in arg))
        # Python -c validators often name their evidence inside a string literal.
        for arg in args[1:]:
            values.extend(match[1] for match in re.findall(r"(['\"])([^'\"\n]+)\1", arg))
        if "-c" in args and Path(args[0]).name.startswith("python"):
            try:
                code = ast.parse(args[args.index("-c") + 1])
            except (IndexError, SyntaxError):
                pass
            else:
                values.extend(node.value for node in ast.walk(code)
                              if isinstance(node, ast.Constant) and isinstance(node.value, str))
    root = repo_root.resolve()
    result: list[Path] = []
    for value in values:
        value = value.strip()
        if not value or value.startswith("-") or any(c in value for c in "\n\x00"):
            continue
        candidate = Path(value)
        if not candidate.suffix:
            continue
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            candidate = candidate.resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        if candidate not in result:
            result.append(candidate)
    return result


def gate_evidence(
    task_id: str,
    blocks: Mapping[str, Mapping[str, str]],
    *,
    repo_root: Path,
    todo_path: Path,
    log_path: Path,
) -> tuple[str, dict[str, Any]]:
    """Hash decisions and source content, excluding clocks, logs, and board prose."""

    root = repo_root.resolve()
    dependencies = dependency_closure(task_id, blocks)
    tasks = {key: {field: str((blocks.get(key) or {}).get(field) or "")
                   for field in _FIELDS} for key in [task_id, *dependencies]}
    # Opt-in only: legacy task fingerprints and refusal ledgers stay unchanged.
    for key, fields in list(tasks.items()):
        declared = (blocks.get(key) or {}).get("prerequisite_requirements")
        if declared is not None and declared != "":
            tasks[key] = _prerequisite_task_fields(key, {**fields, "prerequisite_requirements": declared})
    artifacts: dict[str, Any] = {}
    scope: set[str] = set()
    excluded = {todo_path.resolve(), log_path.resolve()}
    for key in tasks:
        for path in _referenced_paths(blocks.get(key) or {}, root):
            if path in excluded or path.suffix.casefold() == ".log":
                continue
            relative = path.relative_to(root).as_posix()
            if key != task_id and _is_source_path(Path(relative)):
                scope.add(relative)
            if relative in artifacts:
                continue
            try:
                body = path.read_bytes()
            except OSError:
                artifacts[relative] = {"missing": True}
                continue
            if path.suffix.casefold() == ".json":
                try:
                    artifacts[relative] = _diagnostic(json.loads(body))
                    continue
                except (ValueError, UnicodeError):
                    pass
            artifacts[relative] = {"sha256": hashlib.sha256(body).hexdigest()}
    dependents = [{"task_id": name, "status": fields.get("status"),
                   "deterministic_command": fields.get("deterministic_command"), "validation": fields.get("validation")}
                  for name, fields in blocks.items() if task_id in dependency_ids(fields)
                  and str(fields.get("status") or "").casefold() in {"pending", "ready", "in_progress", "in progress"}]
    evidence = {"task_id": task_id, "tasks": tasks, "artifacts": artifacts,
                "dependencies": dependencies, "pending_dependents": dependents, "allowed_source_paths": sorted(scope)}
    return hashlib.sha256(_json(evidence).encode("utf-8")).hexdigest(), evidence


def gate_source_contract(evidence: Mapping[str, Any], command_contract: str) -> str:
    """Bind refusal limits to declared code and dependency contracts, not runs."""

    sources = {path: value for path, value in (evidence.get("artifacts") or {}).items()
               if _is_source_path(Path(path))}
    tasks = evidence.get("tasks")
    if isinstance(tasks, Mapping):
        tasks = {task: {key: value for key, value in fields.items()
                        if key not in _PREREQUISITE_DIAGNOSTIC_FIELDS}
                 if isinstance(fields, Mapping) else fields for task, fields in tasks.items()}
    payload = {"tasks": tasks, "sources": sources, "command_contract": command_contract}
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _contract_warnings(task_id: str, evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (evidence.get("tasks") or {}).get(task_id) or {}
    try:
        command = shlex.split(str(fields.get("deterministic_command") or ""))
        validation = shlex.split(str(fields.get("validation") or ""))
    except ValueError:
        return []
    warnings: list[dict[str, Any]] = []
    if any(arg in {"--fresh", "--no-cache", "--force-refresh"} or arg == "--fresh=true" for arg in command):
        warnings.append({"code": "forced_fresh_producer", "action": "Repair source or dependencies before requesting another unchanged capture."})
    transports = {"http", "https", "libp2p", "stdio", "websocket", "grpc"}
    produced = sorted(set(command) & transports)
    checked = sorted(set(validation) & transports)
    if len(produced) == 1 and not checked:
        warnings.append({"code": "single_transport_producer_validation_scope_unspecified",
                         "producer_transports": produced, "validation_transports": checked,
                         "action": "Check whether validation requires other transports. Separate capture completion from full acceptance and wire independent producers before the acceptance gate."})
    return warnings


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class GateRecovery:
    """A restart-safe ledger, separate from hard command-failure budgets."""

    def __init__(self, attempt_path: Path):
        self.path = attempt_path.with_name(f"{attempt_path.stem}.gates.json")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        self.records: dict[str, Any] = value if isinstance(value, dict) else {}

    def current(self, task_id: str, fingerprint: str) -> dict[str, Any]:
        record = self.records.get(task_id)
        return dict(record) if isinstance(record, dict) and record.get("fingerprint") == fingerprint else {}

    def store(self, task_id: str, record: Mapping[str, Any]) -> None:
        self.records[task_id] = dict(record)
        _write(self.path, self.records)

    def clear(self, task_id: str) -> None:
        if self.records.pop(task_id, None) is not None:
            _write(self.path, self.records)

    def observe(self, task_id: str, fingerprint: str, *, now_ms: int, source_contract: str = "") -> dict[str, Any]:
        previous = self.records.get(task_id) or {}
        record = self.current(task_id, fingerprint)
        same_source = bool(source_contract and isinstance(previous, dict) and previous.get("source_contract") == source_contract)
        if source_contract and not same_source:
            record = {}
        if same_source:
            for name in ("dependencies_revalidated", "dependency_validations", "dependency_validation_fingerprint", "repair_attempted", "deterministic_repair"):
                if name in previous:
                    record[name] = previous[name]
        count = max(0, int(record.get("unchanged_probes") or 0)) + 1
        record.update(fingerprint=fingerprint, unchanged_probes=count,
                      first_seen_ms=int(record.get("first_seen_ms") or now_ms), last_seen_ms=now_ms)
        if source_contract:
            refusals = int(previous.get("source_refusals") or 0) + 1 if same_source else 1
            record.update(source_contract=source_contract, source_refusals=refusals,
                          source_first_seen_ms=int(previous.get("source_first_seen_ms") or now_ms) if same_source else now_ms)
        limit_count = max(count, int(record.get("source_refusals") or 0))
        delay = 0 if limit_count < GATE_REPEAT_LIMIT else min(
            GATE_RETRY_MAX_MS, GATE_RETRY_BASE_MS * 2 ** min(limit_count - GATE_REPEAT_LIMIT, 8))
        record["next_retry_at_ms"] = now_ms + delay
        self.store(task_id, record)
        return record

    def packet_path(self, task_id: str) -> Path:
        """Keep each task's durable handoff separate without renaming legacy IDs."""

        safe = re.sub(r"[^A-Za-z0-9_-]", "-", task_id)
        if not safe or safe != task_id or len(safe) > 80:
            # Generated prerequisite IDs can differ only beyond the old
            # filename cutoff. Bind the suffix to the complete original ID.
            digest = hashlib.sha256(task_id.encode("utf-8", errors="surrogatepass")).hexdigest()
            safe = (safe[:48] or "task") + "." + digest
        return self.path.with_name(f"gate-recovery-{safe}.json")

    def packet(
        self, task_id: str, record: Mapping[str, Any], evidence: Mapping[str, Any], *,
        runtime_read_budget: Mapping[str, Any] | None = None,
        selected_scope: Mapping[str, str] | None = None,
    ) -> Path:
        path = self.packet_path(task_id)
        validations = record.get("dependency_validations") or []
        packet = {
            "schema": "ipfs_accelerate_py/agent-supervisor/gate-recovery@1",
            "reason": "unchanged_gate_withheld", "task_id": task_id,
            "gate_evidence_fingerprint": record.get("fingerprint"),
            "unchanged_probes": record.get("unchanged_probes"),
            "source_contract": record.get("source_contract"), "source_refusals": record.get("source_refusals"),
            "last_probe": record.get("last_probe", "producer"),
            "next_retry_at_ms": record.get("next_retry_at_ms"),
            **{key: value for key, value in evidence.items() if key != "artifacts"},
            "artifacts": {path: _artifact_summary(value) for path, value in (evidence.get("artifacts") or {}).items()},
            "body_free": True, "withheld_stage": record.get("withheld_stage", "command"),
            "contract_warnings": _contract_warnings(task_id, evidence),
            "dependency_validations": validations,
            # Retain spent probes and old failures without presenting them as
            # validation of a different evidence fingerprint. Legacy origins
            # are unknown until an actually reserved probe records its input.
            "dependency_validation_evidence_fingerprint": record.get("dependency_validation_fingerprint"),
            "dependency_validations_current": bool(validations
                and record.get("dependency_validation_fingerprint")
                and record.get("dependency_validation_fingerprint") == record.get("fingerprint")),
            "dependency_validation_budget_spent": bool(record.get("dependencies_revalidated")),
            "deterministic_repair": record.get("deterministic_repair"),
            "required_actions": [
                "Inspect failed or presence-only upstream validation and implement missing behavior.",
                "Use the upstream source scope; preserve the gate and operator holds.",
                "Regenerate evidence and run the original gate before completing any task.",
                "A new trace, correlation, receipt, or run identifier alone is not progress. Repair the source/dependency contract instead of admitting another identical producer.",
            ],
            "work_complete": False, "completion_authority": False,
        }
        if isinstance(evidence.get("tasks"), Mapping):
            packet["tasks"] = {key: _prerequisite_task_fields(key, fields)
                               for key, fields in evidence["tasks"].items()}
        if runtime_read_budget is None:
            prerequisite_plan = prerequisite_recovery_plan(task_id, evidence)
        else:
            prerequisite_plan = prerequisite_recovery_plan(
                task_id, evidence, runtime_read_budget=runtime_read_budget, selected_scope=selected_scope)
        if prerequisite_plan is not None:
            packet["prerequisite_recovery"] = prerequisite_plan
            packet["required_actions"].append(
                "Route typed prerequisite proposals through existing fenced task admission and registered metadata handlers. "
                "Continue a selected finite unit from verified lossless cache within its original cumulative budget; "
                "do not request per-file approval or promote a proposal into runtime authority.")
        _write(path, packet)
        return path


def admit_gate_repair_task(
    todo_text: str,
    *,
    repo_root: Path,
    task_id: str,
    repair_task_id: str,
    command: str,
    validation: str,
    predicted_files: list[str],
    acceptance: str,
    depends_on: list[str] | None = None,
    resource_class: str = "cpu-small",
    implementation_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Append an explicit repair task and add it as a gate dependency.

    The caller supplies a concrete repair contract. Original task statuses and
    acceptance remain intact; the normal lane must execute and validate repair.
    Repeating admission is idempotent, and dependency cycles are refused.
    """

    from .lane_resume import parse_deterministic_command
    from .supervisor_watchdog import parse_todo_task_fields

    prefix = task_id.partition("-")[0] + "-"
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9-]+", repair_task_id) or not repair_task_id.startswith(prefix):
        raise ValueError("repair task identity must use the gate's board namespace")
    blocks = dict(parse_todo_task_fields(todo_text, task_header_prefix=prefix))
    if task_id not in blocks or task_id == repair_task_id:
        raise ValueError("a distinct existing gate task is required")
    argv = parse_deterministic_command(command, repo_root=repo_root)
    validator = parse_deterministic_command(validation, repo_root=repo_root)
    if not argv or not validator or (Path(validator[0]).name == "test" and len(validator) > 1
                                     and validator[1] in {"-f", "-e", "-d", "-s", "-r", "-w", "-x", "-L", "-h"}):
        raise ValueError("repair requires a deterministic command and behavioral validation")
    if not predicted_files or not acceptance.strip():
        raise ValueError("repair source scope and acceptance are required")
    if resource_class not in {"cpu-small", "cpu-browser"}:
        raise ValueError("repair resource class must be cpu-small or cpu-browser")
    timeout_limit = 50400 if resource_class == "cpu-browser" else 1800
    if not 1 <= implementation_timeout_seconds <= timeout_limit:
        raise ValueError("repair timeout exceeds its resource bound")
    for relative in predicted_files:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative.strip() or relative.endswith(".todo.md"):
            raise ValueError("repair scope must stay inside the repo and exclude task boards")
        try:
            (repo_root / path).resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError("repair scope escapes the repo") from exc
    dependencies = list(dict.fromkeys(depends_on or []))
    if any(dependency not in blocks or dependency == task_id or task_id in dependency_closure(dependency, blocks)
           for dependency in dependencies):
        raise ValueError("repair dependencies must exist and must not depend on the gate")
    existing = blocks.get(repair_task_id)
    if not existing and str(blocks[task_id].get("status") or "").casefold() in {"pending", "ready", "in_progress", "in progress"}:
        target = blocks[task_id]
        producer = parse_deterministic_command(target.get("deterministic_command") or "", repo_root=repo_root)
        target_validation = parse_deterministic_command(target.get("validation") or "", repo_root=repo_root)
        def without_bypass(arguments: list[str] | None) -> list[str]:
            return [arg for arg in (arguments or []) if arg not in {"--fresh", "--fresh=true", "--no-cache", "--force-refresh"}]
        prior_scope = {part.strip() for part in str(target.get("predicted_files") or "").split(",") if part.strip()}
        if (producer and without_bypass(producer) == without_bypass(argv) and target_validation == validator
                and set(dependencies).issubset(dependency_ids(target)) and set(predicted_files).issubset(prior_scope)):
            raise ValueError("recapture-only repair repeats a pending producer and validator; repair its source or dependency contract first")
    if existing and (
        existing.get("deterministic_command") != command or existing.get("validation") != validation
        or existing.get("acceptance") != acceptance
        or [part.strip() for part in str(existing.get("predicted_files") or "").split(",")] != predicted_files
        or set(dependency_ids(existing)) != set(dependencies)
        or existing.get("resource_class", "cpu-small") != resource_class
        or int(existing.get("implementation_timeout_seconds") or 1800) != implementation_timeout_seconds
    ):
        raise ValueError("existing repair task has a different contract")
    if existing and task_id in dependency_closure(repair_task_id, blocks):
        raise ValueError("existing repair task depends on the gate")
    for value in [command, validation, acceptance, *predicted_files, *dependencies]:
        if "\n" in value or "\r" in value:
            raise ValueError("repair metadata must use single-line fields")
    # Rewire only the gate's dependency field. No status or evidence is edited.
    lines = todo_text.splitlines(keepends=True)
    in_gate = False
    found_dependencies = False
    updated: list[str] = []
    original_dependencies = dependency_ids(blocks[task_id])
    gate_dependencies = list(dict.fromkeys([*original_dependencies, repair_task_id]))
    for line in lines:
        if line.startswith("## "):
            if in_gate and not found_dependencies:
                updated.append("- Depends on: " + ", ".join(gate_dependencies) + "\n\n")
            in_gate = line[3:].strip().split(" ", 1)[0] == task_id
        if in_gate and re.match(r"- Depends on:\s*", line):
            updated.append("- Depends on: " + ", ".join(gate_dependencies) + "\n")
            found_dependencies = True
        else:
            updated.append(line)
    if in_gate and not found_dependencies:
        updated.append("\n- Depends on: " + ", ".join(gate_dependencies) + "\n")
    text = "".join(updated)
    if not existing:
        text = text.rstrip() + "\n\n" + "\n".join([
            f"## {repair_task_id} Repair upstream evidence for {task_id}",
            "", "- Status: pending", "- Priority: P0", "- Track: gate-recovery",
            "- Depends on: " + ", ".join(dependencies), f"- Recovery for: {task_id}",
            f"- Deterministic command: {command}", f"- Validation: {validation}",
            f"- Resource class: {resource_class}",
            f"- Implementation timeout seconds: {implementation_timeout_seconds}",
            "- Predicted files: " + ", ".join(predicted_files),
            "- Conflict policy: Preserve operator holds, original acceptance, and gate decisions.",
            f"- Acceptance: {acceptance}", "",
        ])
    return {"admitted": not bool(existing), "reason": "duplicate_refill" if existing else "gate_repair_admitted",
            "derived_task_id": repair_task_id, "todo_text": text,
            "work_complete": False, "completion_authority": False}
