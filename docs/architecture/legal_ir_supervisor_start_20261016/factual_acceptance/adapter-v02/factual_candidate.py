"""Bounded source correspondence for a Legal IR engineering candidate.

The pure predicate checks source-derived facts. It does not authenticate a
native owner, provider execution, task claim or repository observation. The
completion-facing CLI deliberately remains closed (exit 3) until a separately
reviewed supported native acceptance owner is bound. No file, candidate field,
environment variable or caller callback can open that gate in this revision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Mapping

SCHEMA = "legal-ir-source-factual-candidate@1"
EVIDENCE_SCHEMA = "legal-ir-source-factual-correspondence@1"
TASK = "LIR-9001"
PHASE = "bootstrap_LIR-9001_only"
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_AGGREGATE_BYTES = 12 * 1024 * 1024
MAX_NODES = 100_000
MAX_DEPTH = 32
MAX_LIST = 4096
SOURCE_PATHS = {
    "authoring_tasks": "authoring/tasks.json",
    "execution_projection": "tasks.execution.json",
    "execution_board": "tasks.todo.md",
    "source_copy_manifest": "source-copy-manifest.json",
    "current_source_bindings": "CURRENT_SOURCE_BINDINGS.json",
    "scheduler_config": "scheduler_config.json",
}
ROUTE = {
    "primary_provider_id": "grok_cli", "primary_model_id": "grok-4.7",
    "fallback_provider_id": "codex", "fallback_model_id": "gpt-6.1-sol",
    "fallback_trigger": "primary_quota_exhausted", "fallback_reasoning_effort": "high",
    "max_concurrency": 1, "primary_executable": "/home/barberb/.local/bin/grok",
}
NO_AUTHORITY = {
    "completion_authority": False, "execution_authority": False,
    "legal_meaning_review_authority": False, "training_authority": False,
    "gold_admission_authority": False, "proof_authority": False,
    "task_status_mutation_authority": False, "publication_authority": False,
}
UNRESOLVED_SCOPE = [
    "current_other_agent_runtime_state_unobserved",
    "repository_metadata_authentication_requires_native_owner",
    "statutory_semantics_unreviewed", "legal_graph_fidelity_unreviewed",
    "training_eligibility_unadmitted", "lean_build_unqualified",
]
NEXT_PHASE = {
    "proposed_task_id": "LIR-9002", "activated": False,
    "requires_independent_acceptance_and_fresh_execution_revision": True,
}
FINDINGS = {
    "preserve_task_dag": ("authoring_tasks", "/tasks", "capabilities_not_bound"),
    "preserve_goal_hierarchy": ("authoring_tasks", "/goal_hierarchy", "capabilities_not_bound"),
    "preserve_legacy_crosswalk": ("authoring_tasks", "/legacy_parent_crosswalk", "related_scope_not_equivalence"),
    "retain_source_snapshot_limits": ("current_source_bindings", "/limitations", "current_other_owner_state_unobserved"),
    "preserve_frozen_source_inventory": ("source_copy_manifest", "/files", "source_context_only"),
    "related_lac_is_not_equivalence": ("authoring_tasks", "/legacy_parent_crosswalk", "related_scope_not_equivalence"),
    "preserve_unavailable_repository": ("current_source_bindings", "/repositories", "repository_metadata_unavailable"),
}
REQUIRED_FINDINGS = frozenset({"preserve_task_dag", "retain_source_snapshot_limits"})
REPORT_FIELDS = frozenset({
    "schema", "task_id", "phase", "candidate", "orchestration_namespace",
    "source_commitments", "facts", "findings", "unresolved_scope", "authority",
    "next_phase", "independent_acceptance_required",
})


class FactualRefusal(ValueError):
    pass


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise FactualRefusal(reason)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def same(actual: object, expected: object, reason: str) -> None:
    # JSON types remain distinct: False is not accepted as numeric zero and
    # integer counts are not accepted as floating point lookalikes.
    need(canonical(actual) == canonical(expected), reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pin(raw: bytes) -> dict:
    return {"bytes": len(raw), "sha256": sha(raw)}


def strict_json(raw: bytes) -> object:
    need(type(raw) is bytes and len(raw) <= MAX_DOCUMENT_BYTES, "document_byte_bound")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def bad_constant(value):
        raise FactualRefusal("nonfinite_json_number")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise FactualRefusal("invalid_or_excessive_json") from error
    nodes = 0
    def visit(item, depth):
        nonlocal nodes
        nodes += 1
        need(nodes <= MAX_NODES and depth <= MAX_DEPTH, "json_population_bound")
        if type(item) is dict:
            need(len(item) <= MAX_LIST, "json_object_bound")
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) is list:
            need(len(item) <= MAX_LIST, "json_list_bound")
            for child in item:
                visit(child, depth + 1)
    visit(value, 0)
    return value


def _safe_relative(name: object) -> str:
    need(type(name) is str and 0 < len(name) <= 4096 and "\\" not in name and "\x00" not in name, "source_name_invalid")
    path = PurePosixPath(name)
    need(not path.is_absolute() and path.parts and all(part not in {".", ".."} for part in path.parts), "source_name_escapes")
    need(path.as_posix() == name, "source_name_not_canonical")
    return name


def _validate_graph(tasks: list, goals: list, crosswalk: list) -> None:
    need(type(tasks) is list and len(tasks) == 64, "task_population_differs")
    need(type(goals) is list and len(goals) == 34, "goal_population_differs")
    need(type(crosswalk) is list and len(crosswalk) == 24, "legacy_parent_population_differs")
    ids = [row.get("task_id") for row in tasks if type(row) is dict]
    need(len(ids) == 64 and all(type(tid) is str and re.fullmatch(r"LIR-[0-9]{4}", tid) for tid in ids)
         and len(set(ids)) == 64 and TASK in ids, "task_identity_population_differs")
    goal_ids = [row.get("goal_id") for row in goals if type(row) is dict]
    need(len(goal_ids) == 34 and all(type(gid) is str and re.fullmatch(r"LIR-G[0-9]{3}", gid) for gid in goal_ids)
         and len(set(goal_ids)) == 34, "goal_identity_population_differs")
    by_task = {row["task_id"]: row for row in tasks}
    edges = 0
    for row in tasks:
        dependencies = row.get("depends_on")
        need(type(dependencies) is list and all(type(dep) is str for dep in dependencies)
             and len(dependencies) == len(set(dependencies)), "dependency_shape_differs")
        need(all(type(dep) is str and dep in by_task and dep != row["task_id"] for dep in dependencies), "unresolved_dependency")
        need(type(row.get("goal_id")) is str and row["goal_id"] in goal_ids, "task_goal_unresolved")
        need(row.get("completion") == "manual", "task_completion_policy_changed")
        need(row.get("execution_authorized_by_this_file") is False and row.get("gold_or_teacher_admission_granted") is False, "authoring_invents_authority")
        edges += len(dependencies)
    need(edges == 115, "dependency_count_differs")
    done = set()
    pending = set(ids)
    while pending:
        ready = {tid for tid in pending if all(dep in done for dep in by_task[tid]["depends_on"])}
        need(bool(ready), "task_dependency_cycle")
        done.update(ready)
        pending.difference_update(ready)
    by_goal = {row["goal_id"]: row for row in goals}
    need(sum(row.get("parent") is None for row in goals) == 1, "goal_root_population_differs")
    for row in goals:
        seen = set()
        current = row["goal_id"]
        while current is not None:
            need(type(current) is str and current in by_goal and current not in seen, "goal_parent_cycle_or_unresolved")
            seen.add(current)
            current = by_goal[current].get("parent")
    parent_ids = [row.get("id") for row in crosswalk if type(row) is dict]
    need(len(parent_ids) == 24 and all(type(pid) is str for pid in parent_ids)
         and len(set(parent_ids)) == 24, "legacy_parent_identity_differs")
    for row in crosswalk:
        need(type(row.get("subtasks")) is list and all(type(tid) is str and tid in by_task for tid in row["subtasks"]), "legacy_subtask_unresolved")
        need(row.get("terminal_task") in row["subtasks"] and row.get("refinement_goal") in by_goal, "legacy_terminal_or_goal_unresolved")
        need(all(by_task[tid].get("legacy_parent_task") == row["id"] for tid in row["subtasks"]), "legacy_subtask_parent_mismatch")


def check_legacy_board(raw: bytes, projected: list, authoring_namespace: str) -> None:
    """Join legacy human-readable task facts without inventing canonical CIDs."""
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise FactualRefusal("legacy_board_encoding_invalid") from error
    headings = list(re.finditer(r"(?m)^## (LIR-[0-9]{4}) ([^\r\n]+)$", text))
    same([match.group(1) for match in headings], [row["task_id"] for row in projected], "legacy_board_task_population_differs")
    for index, (heading, row) in enumerate(zip(headings, projected)):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[heading.end():end]
        fields = {}
        for match in re.finditer(r"(?m)^- ([^:\r\n]+):([^\r\n]*)$", block):
            key, value = match.group(1), match.group(2).strip()
            need(key not in fields, "legacy_board_duplicate_field")
            fields[key] = value
        expected = {
            "Status": row["status"], "Completion": "manual",
            "Depends on": ", ".join(row["depends_on"]), "Goal id": row["goal_id"],
            "Legacy parent task": row["legacy_parent_task"] or "none; separate engineering/control subgoal",
            "Related LAC tasks": ", ".join(row["related_lac_tasks"]),
            "Owning repository": row["repository"], "Board namespace": authoring_namespace,
            "Bootstrap phase": PHASE,
        }
        for key, value in expected.items():
            same(fields.get(key), value, "legacy_board_field_differs:" + key)


def derive_expected(source_documents: Mapping[str, bytes], frozen_files: Mapping[str, bytes]) -> dict:
    """Derive facts from supplied frozen bytes; supplied bytes are not authority."""
    need(set(source_documents) == set(SOURCE_PATHS), "source_role_population_differs")
    need(all(type(raw) is bytes and len(raw) <= MAX_DOCUMENT_BYTES for raw in source_documents.values()), "source_byte_bound")
    documents = {role: strict_json(raw) for role, raw in source_documents.items() if role != "execution_board"}
    authoring = documents["authoring_tasks"]
    execution = documents["execution_projection"]
    manifest = documents["source_copy_manifest"]
    bindings = documents["current_source_bindings"]
    config = documents["scheduler_config"]
    need(all(type(value) is dict for value in documents.values()), "source_document_shape_differs")
    need(authoring.get("schema") == "legal-ir-supervisor-refinement/v1", "authoring_schema_differs")
    tasks, goals, crosswalk = authoring.get("tasks"), authoring.get("goal_hierarchy"), authoring.get("legacy_parent_crosswalk")
    _validate_graph(tasks, goals, crosswalk)
    need(execution.get("schema") == "legal-ir-supervisor-bootstrap-projection/v1", "execution_schema_differs")
    namespace = config.get("board_namespace")
    need(type(namespace) is str and 0 < len(namespace) <= 512, "orchestration_namespace_invalid")
    same(execution.get("board_namespace"), namespace, "execution_namespace_differs")
    same(config.get("provider"), ROUTE, "provider_tuple_differs")
    need(config.get("source_binding", {}).get("bootstrap_task_source") == "legacy-markdown", "task_source_route_differs")
    policy = config.get("bootstrap_execution_contract", {})
    same(policy.get("allowed_task_ids"), [TASK], "bootstrap_frontier_broadened")
    need(policy.get("other_tasks_phase_blocked") is True, "future_phase_unblocked")
    projected = execution.get("tasks")
    need(type(projected) is list and len(projected) == 64, "execution_task_population_differs")
    need(all(type(row) is dict for row in projected), "execution_task_shape_differs")
    same([row.get("task_id") for row in projected], [row["task_id"] for row in tasks], "execution_task_order_differs")
    task_facts = []
    for source, row in zip(tasks, projected):
        need(type(row) is dict, "execution_task_shape_differs")
        for key in ("task_id", "goal_id", "legacy_parent_task", "related_lac_tasks", "depends_on", "repository", "completion"):
            same(row.get(key), source.get(key), "execution_preserved_field_differs:" + key)
        need(row.get("status") == ("todo" if row["task_id"] == TASK else "blocked"), "execution_frontier_differs")
        need(row.get("original_status") == source.get("status") and row.get("bootstrap_phase") == PHASE, "execution_original_or_phase_differs")
        task_facts.append({
            "task_id": source["task_id"], "goal_id": source["goal_id"],
            "legacy_parent_task": source["legacy_parent_task"], "related_lac_tasks": source["related_lac_tasks"],
            "depends_on": source["depends_on"], "repository": source["repository"], "completion": source["completion"],
            "authored_status": source["status"], "execution_status": row["status"],
        })
    same(execution.get("goal_hierarchy"), goals, "execution_goals_changed")
    same(execution.get("legacy_parent_crosswalk"), crosswalk, "execution_parent_crosswalk_changed")
    check_legacy_board(source_documents["execution_board"], projected, authoring["board_namespace"])
    need(manifest.get("schema") == "legal-ir-supervisor-source-copy-manifest/v1", "source_manifest_schema_differs")
    inventory = manifest.get("files")
    need(type(inventory) is list and len(inventory) == 18, "source_inventory_population_differs")
    names = [_safe_relative(row.get("path")) for row in inventory if type(row) is dict]
    need(len(names) == 18 and len(set(names)) == 18 and set(frozen_files) == set(names), "frozen_file_population_differs")
    total = sum(len(raw) for raw in source_documents.values())
    for row in inventory:
        name = row["path"]
        need(name.startswith(("authoring/", "context/datasets/docs/autoencoders/")), "frozen_file_role_escapes")
        raw = frozen_files[name]
        need(type(raw) is bytes and len(raw) <= MAX_DOCUMENT_BYTES, "frozen_file_byte_bound")
        total += len(raw)
        same(pin(raw), {"bytes": row.get("bytes"), "sha256": row.get("sha256")}, "frozen_file_pin_differs:" + name)
    need(total <= MAX_AGGREGATE_BYTES, "aggregate_source_byte_bound")
    need(frozen_files.get("authoring/tasks.json") == source_documents["authoring_tasks"], "authoring_role_alias_differs")
    need(bindings.get("schema") == "legal-ir-supervisor-current-source-bindings/v1", "source_bindings_schema_differs")
    need(set(bindings) == {"schema", "status", "observation", "repositories", "limitations", "execution_authority", "independent_acceptance_required"}, "source_bindings_fields_differ")
    need(bindings["execution_authority"] is False and bindings["independent_acceptance_required"] is True, "metadata_promotes_authority")
    repositories = bindings["repositories"]
    need(type(repositories) is list and len(repositories) == 2, "repository_population_differs")
    need(all(type(row) is dict for row in repositories), "repository_shape_differs")
    same([row.get("repository") for row in repositories], ["ipfs_accelerate_py", "ipfs_datasets_py"], "repository_identity_population_differs")
    for row in repositories:
        need(set(row) == {"repository", "status", "checkout_path", "head", "tree", "branch", "source_files"}, "repository_fields_differ")
        need(type(row["source_files"]) is list and len(row["source_files"]) <= 256, "repository_pin_population_bound")
        if row["status"] == "unavailable":
            need(all(row[key] is None for key in ("checkout_path", "head", "tree", "branch")) and row["source_files"] == [], "unavailable_repository_invented")
        else:
            need(row["status"] == "observed_metadata", "repository_status_invalid")
            need(all(type(row[key]) is str and re.fullmatch(r"[0-9a-f]{40}", row[key]) for key in ("head", "tree")), "repository_object_id_invalid")
            need(type(row["branch"]) is str and 0 < len(row["branch"]) <= 512, "repository_branch_invalid")
            need(type(row["checkout_path"]) is str and Path(row["checkout_path"]).is_absolute(), "repository_checkout_invalid")
        selected = []
        for item in row["source_files"]:
            need(type(item) is dict and set(item) == {"path", "bytes", "sha256"}, "selected_source_pin_fields_differ")
            selected.append(_safe_relative(item["path"]))
            need(type(item["bytes"]) is int and 0 <= item["bytes"] <= 16 * 1024 * 1024, "selected_source_pin_size_invalid")
            need(type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "selected_source_pin_digest_invalid")
        need(len(selected) == len(set(selected)), "selected_source_pin_duplicate")
    need(type(bindings["limitations"]) is list and 0 < len(bindings["limitations"]) <= 64, "source_limitations_required")
    return {
        "orchestration_namespace": namespace,
        "source_commitments": {role: pin(raw) for role, raw in source_documents.items()},
        "facts": {
            "population": {"tasks": 64, "goals": 34, "dependency_edges": 115, "legacy_parents": 24, "frozen_files": 18},
            "canonical_authoring_namespace": authoring["board_namespace"], "tasks": task_facts,
            "goals": goals, "legacy_parent_crosswalk": crosswalk, "source_inventory": inventory,
            "repository_observations": repositories,
            "observation_snapshot": {key: bindings[key] for key in ("status", "observation", "limitations")},
            "eligible_tasks": [TASK], "phase_blocked_tasks": 63,
        },
        "documents": documents,
    }


def _finding(code: str, expected: dict) -> dict:
    role, pointer, gap = FINDINGS[code]
    return {"finding_code": code, "evidence_role": role, "json_pointer": pointer,
            "source_pin": expected["source_commitments"][role], "gap_code": gap}


def candidate_template(source_documents: Mapping[str, bytes], frozen_files: Mapping[str, bytes]) -> dict:
    expected = derive_expected(source_documents, frozen_files)
    return {
        "schema": SCHEMA, "task_id": TASK, "phase": PHASE, "candidate": True,
        "orchestration_namespace": expected["orchestration_namespace"],
        "source_commitments": copy.deepcopy(expected["source_commitments"]), "facts": copy.deepcopy(expected["facts"]),
        "findings": [_finding(code, expected) for code in sorted(REQUIRED_FINDINGS)],
        "unresolved_scope": list(UNRESOLVED_SCOPE), "authority": dict(NO_AUTHORITY),
        "next_phase": dict(NEXT_PHASE), "independent_acceptance_required": True,
    }


def check_facts(report_raw: bytes, source_documents: Mapping[str, bytes], frozen_files: Mapping[str, bytes]) -> dict:
    expected = derive_expected(source_documents, frozen_files)
    report = strict_json(report_raw)
    need(type(report) is dict and set(report) == REPORT_FIELDS, "candidate_fields_differ")
    need(report["schema"] == SCHEMA and report["task_id"] == TASK and report["phase"] == PHASE, "candidate_identity_differs")
    need(report["candidate"] is True and report["independent_acceptance_required"] is True, "candidate_promotes_acceptance")
    for field in ("orchestration_namespace", "source_commitments", "facts"):
        same(report[field], expected[field], "candidate_" + field + "_differs")
    same(report["authority"], NO_AUTHORITY, "candidate_promotes_authority")
    same(report["unresolved_scope"], UNRESOLVED_SCOPE, "candidate_omits_unresolved_scope")
    same(report["next_phase"], NEXT_PHASE, "candidate_activates_next_phase")
    findings = report["findings"]
    need(type(findings) is list and len(REQUIRED_FINDINGS) <= len(findings) <= len(FINDINGS), "finding_population_differs")
    seen = set()
    for row in findings:
        need(type(row) is dict and type(row.get("finding_code")) is str and row["finding_code"] in FINDINGS, "unsupported_finding")
        code = row["finding_code"]
        need(code not in seen, "duplicate_finding")
        seen.add(code)
        same(row, _finding(code, expected), "finding_reference_or_scope_differs")
        if code == "preserve_unavailable_repository":
            need(any(row["status"] == "unavailable" for row in expected["facts"]["repository_observations"]), "unavailable_repository_claim_unsupported")
        if code == "related_lac_is_not_equivalence":
            need(all(row.get("related_lac_equivalence") == "unreviewed_related_scope_only" for row in expected["facts"]["legacy_parent_crosswalk"]), "lac_equivalence_claim_unsupported")
    need(REQUIRED_FINDINGS <= seen, "required_finding_missing")
    return {
        "schema": EVIDENCE_SCHEMA, "valid": True, "task_id": TASK, "phase": PHASE,
        "candidate": pin(report_raw), "source_commitments": expected["source_commitments"],
        "facts_sha256": sha(canonical(expected["facts"])), "population": expected["facts"]["population"],
        "closed_findings_checked": len(findings), "scope": "Frozen engineering-source correspondence only; supplied metadata is not authenticated runtime observation.",
        "current_repository_observations_authenticated": False, "runtime_effects_verified": False,
        "native_validation_owner_bound": False, "completion_authority": False,
        "legal_meaning_review_authority": False, "gold_admission_authority": False,
        "training_authority": False, "proof_authority": False, "next_phase_activated": False,
    }


def read_source_file(path: Path) -> bytes:
    path = Path(path)
    need(path.is_absolute(), "absolute_source_path_required")
    for component in [*reversed(path.parents), path]:
        need(not component.is_symlink(), "source_symlink_refused")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= MAX_DOCUMENT_BYTES, "bounded_regular_source_required")
        chunks = []
        remaining = MAX_DOCUMENT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        current = path.lstat()
        witness = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        need(witness(before) == witness(after) == witness(current) and len(raw) == before.st_size and len(raw) <= MAX_DOCUMENT_BYTES, "source_changed_during_read")
        return raw
    finally:
        os.close(descriptor)


def load_source_bundle(source_root: Path) -> tuple[dict, dict]:
    source_root = Path(source_root)
    need(source_root.is_absolute(), "absolute_source_root_required")
    documents = {role: read_source_file(source_root / name) for role, name in SOURCE_PATHS.items()}
    manifest = strict_json(documents["source_copy_manifest"])
    need(type(manifest) is dict and type(manifest.get("files")) is list and len(manifest["files"]) == 18, "manifest_population_differs")
    frozen = {}
    for row in manifest["files"]:
        need(type(row) is dict, "manifest_row_invalid")
        name = _safe_relative(row.get("path"))
        need(name.startswith(("authoring/", "context/datasets/docs/autoencoders/")), "manifest_source_role_escapes")
        need(name not in frozen, "manifest_source_duplicate")
        frozen[name] = read_source_file(source_root / name)
    return documents, frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-report", type=Path, help="Completion-facing gate; remains closed without a supported native binding.")
    modes.add_argument("--check-facts", type=Path, help="Advisory predicate only; must never replace task completion validation.")
    modes.add_argument("--emit-candidate-template", action="store_true", help="Emit unaccepted source facts; not a provider execution or review.")
    args = parser.parse_args(argv)
    try:
        sources, frozen = load_source_bundle(args.source_root)
        if args.emit_candidate_template:
            print(canonical(candidate_template(sources, frozen)).decode())
            return 0
        selected = args.check_report or args.check_facts
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        evidence = check_facts(read_source_file(selected), sources, frozen)
        if args.check_report:
            print(json.dumps({"valid": False, "reason": "independent_native_validation_owner_binding_not_bound", "factual_correspondence": evidence, "completion_authority": False}, sort_keys=True))
            return 3
        print(json.dumps(evidence, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        reason = str(error) if isinstance(error, FactualRefusal) else type(error).__name__
        print(json.dumps({"valid": False, "reason": reason, "completion_authority": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
