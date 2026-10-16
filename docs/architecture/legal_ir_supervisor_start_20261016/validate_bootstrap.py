#!/usr/bin/env python3
"""Validate only the staged source-only LegalIR bootstrap documents/report.

No product imports, provider calls, Git calls, database access or job execution.
This checker does not confer launch, reviewer, training or semantic authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
PREFIX = "docs/architecture/legal_ir_supervisor_start_20261016"
PHASE = "bootstrap_LIR-9001_only"
RESULT = "data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json"
ANCESTOR_PLACEHOLDER = "FILL_CURRENT_OWNED_ACCELERATOR_SOURCE_ANCESTOR"
CONFIG_CANONICAL_SHA256 = "644775d505130dd808cc126c4f6c2e127890ace8d838e5ed697549a79bed0540"
SOURCE_MANIFEST_CANONICAL_SHA256 = "43e3f581519506751b613ebe68ccb2af1345304de1bdbf545cc70bc018acf9ad"
AUTHORING_CANONICAL_SHA256 = "c71f9e4b9c28fea8d0344fa11efb7cccdbcc504b75562c04772eaf7cd84e9ff9"
BLOCKER = "Bootstrap phase permits only LIR-9001; preserve this task and dependencies pending a separately reviewed phase transition."
OBJECTIVE = "Read only the frozen source/configuration and engineering metadata; reconcile all 64 tasks, 34 goals and 24 LIR parents with related LAC scope. Record exact evidence and reuse/adoption recommendations without importing product code, executing tests, opening corpus/model/prediction bodies, modifying code, adopting commits or changing task/database/resource authority."
DENIED = ["corpus_bodies", "model_checkpoint_bodies", "vectors_or_latents", "prediction_or_typed_payload_bodies", "credentials", "live_databases", "resource_ledgers", "reviewer_or_gold_authority"]
ACCEPTANCE = [
    "Preserve all 64 task IDs, 34 goals, 24 LIR parent IDs and their exact original dependency edges; report LAC links only as related scope.",
    "Bind the frozen source-copy inventory and actual current owner-supplied repository metadata; retain collisions, unavailable authorities and unsupported evidence explicitly.",
    "Produce only the source-only reconciliation report; do not adopt code, infer parent completion, create reviews or change task/database/resource state.",
    "Completion remains manual until independent report acceptance; successful board validation alone cannot complete reconciliation.",
]
VALIDATION = ["python3 -B " + PREFIX + "/validate_bootstrap.py --check-report " + RESULT]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def raw(path, maximum=8 * 1024 * 1024):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "regular nonsymlink file required: " + str(path))
    need(path.stat().st_size <= maximum, "file exceeds bound: " + str(path))
    value = path.read_bytes()
    need(len(value) <= maximum, "file grew beyond bound")
    return value


def load(path):
    return json.loads(raw(path).decode("utf-8"), object_pairs_hook=unique)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def expected_projection(authoring):
    result = copy.deepcopy(authoring)
    for task in result["tasks"]:
        task["original_status"] = task["status"]
        task["original_blocked_reason"] = task["blocked_reason"]
        task["bootstrap_phase"] = PHASE
        if task["task_id"] == "LIR-9001":
            task["status"] = "todo"
            task["objective"] = OBJECTIVE
            task["validation"] = list(VALIDATION)
            task["inputs"]["references"] = [
                PREFIX + "/authoring/tasks.json",
                PREFIX + "/authoring/source_baseline.json",
                PREFIX + "/source-copy-manifest.json",
                PREFIX + "/README.md",
                PREFIX + "/CURRENT_SOURCE_BINDINGS.json",
            ]
            task["inputs"]["denied_roles"] = list(DENIED)
            task["acceptance_criteria"] = list(ACCEPTANCE)
            task["acceptance"] = "; ".join(ACCEPTANCE)
            task["blocked_reason"] = ""
        else:
            task["status"] = "blocked"
            task["blocked_reason"] = BLOCKER + (
                " Original blocker: " + task["original_blocked_reason"]
                if task["original_blocked_reason"] else ""
            )
    result.update(
        schema="legal-ir-supervisor-bootstrap-projection/v1",
        status="prepared_bootstrap_not_launched",
        source_authoring_schema="legal-ir-supervisor-refinement/v1",
        bootstrap_phase=PHASE,
        bootstrap_allowed_task_ids=["LIR-9001"],
        bootstrap_install_prefix=PREFIX,
        source_authoring_transport_sha256=sha(raw(ROOT / "authoring/tasks.json")),
    )
    return result


def board_markdown(projection):
    """Preserve every original block and its extra fields, changing only phase fields."""
    original = raw(ROOT / "authoring/tasks.todo.md").decode("utf-8")
    parts = re.split(r"(?m)^(## LIR-\d{4} .*)$", original)
    need(len(parts) == 1 + 2 * 64, "original board task population differs")
    lines = [
        "# LegalIR source-only bootstrap execution board", "",
        "All 64 authored tasks and all dependencies remain present. Only LIR-9001",
        "is todo in this phase; the other 63 tasks are explicitly blocked. Original",
        "statuses and blockers remain visible. Completion is manual; this projection",
        "does not grant runtime, reviewer, training, graph or Lean authority.", "",
    ]
    by_id = {task["task_id"]: task for task in projection["tasks"]}
    order = []
    for index in range(1, len(parts), 2):
        header, body = parts[index:index + 2]
        tid = header.split()[1]
        order.append(tid)
        task = by_id[tid]
        replacements = {"Status": task["status"], "Blocked reason": task["blocked_reason"]}
        if tid == "LIR-9001":
            replacements.update({
                "Objective": task["objective"],
                "Inputs": ", ".join(task["inputs"]["references"]),
                "Denied input roles": ", ".join(DENIED),
                "Validation": "; ".join(VALIDATION),
                "Acceptance": task["acceptance"],
            })
        block = []
        for line in body.strip().splitlines():
            if line.startswith("- ") and ":" in line:
                key = line[2:].split(":", 1)[0]
                if key in replacements:
                    line = ("- " + key + ": " + replacements[key]).rstrip()
            block.append(line)
            if line.startswith("- Status:"):
                block.extend([
                    "- Original status: " + task["original_status"],
                    ("- Original blocked reason: " + task["original_blocked_reason"]).rstrip(),
                    "- Bootstrap phase: " + PHASE,
                ])
        lines.extend([header, "", *block, ""])
    need(order == [task["task_id"] for task in projection["tasks"]], "board order differs")
    return "\n".join(lines)


def check_current_source_bindings():
    """Check bounded owner metadata shape without opening referenced files."""
    bindings = json.loads(raw(ROOT / "CURRENT_SOURCE_BINDINGS.json", 256 * 1024).decode("utf-8"), object_pairs_hook=unique)
    need(type(bindings) is dict and set(bindings) == {"schema", "status", "observation", "repositories", "limitations", "execution_authority", "independent_acceptance_required"}, "current-source binding fields differ")
    need(bindings["schema"] == "legal-ir-supervisor-current-source-bindings/v1", "current-source schema differs")
    need(bindings["execution_authority"] is False and bindings["independent_acceptance_required"] is True, "metadata invents authority")
    observation = bindings["observation"]
    need(type(observation) is dict and set(observation) == {"origin", "observed_at_utc", "evidence_path"}, "current-source observation fields differ")
    rows = bindings["repositories"]
    need(type(rows) is list and [row.get("repository") for row in rows if type(row) is dict] == ["ipfs_accelerate_py", "ipfs_datasets_py"], "current-source repository population differs")
    observed = 0
    for row in rows:
        need(type(row) is dict and set(row) == {"repository", "status", "checkout_path", "head", "tree", "branch", "source_files"}, "current-source repository fields differ")
        need(row["status"] in {"unavailable", "observed_metadata"}, "unsupported current-source authority")
        files = row["source_files"]
        need(type(files) is list and len(files) <= 256, "bounded selected source-file metadata required")
        if row["status"] == "unavailable":
            need(all(row[key] is None for key in ["checkout_path", "head", "tree", "branch"]) and files == [], "unavailable current-source observations invented")
            continue
        observed += 1
        checkout = row["checkout_path"]
        need(type(checkout) is str and 1 <= len(checkout) <= 4096 and Path(checkout).is_absolute() and ".." not in Path(checkout).parts, "bounded absolute observed checkout path required")
        need(all(type(row[key]) is str and re.fullmatch(r"[0-9a-f]{40}", row[key]) for key in ["head", "tree"]), "observed source head/tree required")
        need(type(row["branch"]) is str and 1 <= len(row["branch"]) <= 512 and not any(ch.isspace() for ch in row["branch"]), "observed branch or detached marker required")
        names = []
        for item in files:
            need(type(item) is dict and set(item) == {"path", "bytes", "sha256"}, "selected source-file metadata fields differ")
            name = item["path"]
            need(type(name) is str and 1 <= len(name) <= 4096 and not Path(name).is_absolute() and ".." not in Path(name).parts, "relative selected source-file path required")
            need(type(item["bytes"]) is int and 0 <= item["bytes"] <= 16 * 1024 * 1024 and type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "selected source-file size/hash required")
            names.append(name)
        need(len(names) == len(set(names)), "duplicate selected source-file metadata")
    expected_status = "unavailable" if observed == 0 else "observed_metadata" if observed == 2 else "partially_observed"
    need(bindings["status"] == expected_status, "current-source aggregate status differs")
    if observed:
        need(all(type(observation[key]) is str and 1 <= len(observation[key]) <= 4096 for key in observation), "actual observation origin/time/evidence reference required")
        need(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|\+00:00)", observation["observed_at_utc"]), "UTC observation timestamp required")
    else:
        need(all(value is None for value in observation.values()), "unavailable observation identity/time invented")
    need(type(bindings["limitations"]) is list and 1 <= len(bindings["limitations"]) <= 64 and all(type(item) is str and 1 <= len(item) <= 4096 for item in bindings["limitations"]), "explicit bounded observation limitations required")
    return observed


def check_all():
    authoring = load(ROOT / "authoring/tasks.json")
    projection = load(ROOT / "tasks.execution.json")
    manifest = load(ROOT / "source-copy-manifest.json")
    config = load(ROOT / "scheduler_config.json")
    current_observations = check_current_source_bindings()
    need(canonical_sha(authoring) == AUTHORING_CANONICAL_SHA256, "authoring source changed")
    need(canonical_sha(manifest) == SOURCE_MANIFEST_CANONICAL_SHA256, "source inventory changed")
    config_without_ancestor = copy.deepcopy(config)
    config_without_ancestor["source_binding"]["accelerator_required_ancestor"] = ANCESTOR_PLACEHOLDER
    need(canonical_sha(config_without_ancestor) == CONFIG_CANONICAL_SHA256, "scheduler template changed outside the ancestor binding")
    need(authoring["schema"] == "legal-ir-supervisor-refinement/v1", "unexpected authoring schema")
    need(len(authoring["tasks"]) == 64 and len(authoring["goal_hierarchy"]) == 34 and len(authoring["legacy_parent_crosswalk"]) == 24, "source population differs")
    need(projection == expected_projection(authoring), "execution JSON is not the exact permitted phase projection")
    need(raw(ROOT / "tasks.todo.md").decode("utf-8") == board_markdown(projection), "execution Markdown differs")
    need(raw(ROOT / "goals.objectives.md") == raw(ROOT / "authoring/goals.objectives.md"), "goals changed")
    tasks = {task["task_id"]: task for task in projection["tasks"]}
    need(len(tasks) == 64, "duplicate task IDs")
    need([task["task_id"] for task in projection["tasks"] if task["status"] == "todo"] == ["LIR-9001"], "bootstrap dispatch population differs")
    need(all(task["completion"] == "manual" for task in tasks.values()), "automatic completion introduced")
    need(all(not task["execution_authorized_by_this_file"] and not task["gold_or_teacher_admission_granted"] for task in tasks.values()), "invented task authority")
    for tid, task in tasks.items():
        need(all(dep in tasks and dep != tid for dep in task["depends_on"]), "invalid dependency")
        need(task["inputs"]["required_dependency_results"] == task["depends_on"], "input dependency population differs")
        if tid != "LIR-9001":
            need(task["status"] == "blocked" and task["blocked_reason"].startswith(BLOCKER), "future task not phase blocked")
    need(manifest["schema"] == "legal-ir-supervisor-source-copy-manifest/v1" and len(manifest["files"]) == 18, "source-copy inventory differs")
    names = []
    for item in manifest["files"]:
        need(set(item) == {"path", "origin_repository", "origin_path", "bytes", "sha256"}, "source-copy fields differ")
        path = Path(item["path"])
        need(not path.is_absolute() and ".." not in path.parts, "unsafe source-copy path")
        value = raw(ROOT / path)
        need(len(value) == item["bytes"] and sha(value) == item["sha256"], "source-copy pin differs: " + str(path))
        names.append(item["path"])
    need(len(set(names)) == 18 and all(name.startswith(("authoring/", "context/datasets/docs/autoencoders/")) for name in names), "source-copy names differ")
    need(config["schema"] == "ipfs_accelerate_py.agent_supervisor.legal_ir_bootstrap.scheduler_config@1", "wrong scheduler schema")
    need(config["taskboard_path"] == PREFIX + "/tasks.todo.md" and config["objectives_path"] == PREFIX + "/goals.objectives.md", "scheduler paths differ")
    need(config["plan_path"] == PREFIX + "/README.md" and config["validator_path"] == PREFIX + "/validate_bootstrap.py", "scheduler verifier paths differ")
    need(config["task_prefix"] == "## LIR-" and config["board_namespace"] == authoring["board_namespace"], "scheduler task identity differs")
    need(config["max_lanes"] == 1 and config["lanes"] == [{"index": 0, "name": "legal-ir-bootstrap", "strict_shard_remainder": 0}], "bootstrap lanes differ")
    need(config["provider"] == {"primary_provider_id": "grok_cli", "primary_model_id": "grok-4.7", "fallback_provider_id": "codex", "fallback_model_id": "gpt-6.1-sol", "fallback_trigger": "primary_quota_exhausted", "fallback_reasoning_effort": "high", "max_concurrency": 1, "primary_executable": "/home/barberb/.local/bin/grok"}, "provider policy differs")
    need(config["merge_target_branch"] == config["source_binding"]["accelerator_required_branch"] == "agent/legal-ir-supervisor-start-20261016", "branch binding differs")
    need(config["source_binding"]["bootstrap_task_source"] == "legacy-markdown", "task-source route differs")
    ancestor = config["source_binding"]["accelerator_required_ancestor"]
    need(ancestor == ANCESTOR_PLACEHOLDER or re.fullmatch(r"[0-9a-f]{40}", ancestor), "invalid ancestor binding")
    need(all(config[name] is False for name in ["objective_refill_enabled", "codebase_refill_enabled", "objective_goal_refinement_enabled"]), "refill enabled")
    need(config["max_task_attempts"] == 1 and config["max_restarts"] == 1, "unbounded first-task retries")
    policy = config["bootstrap_execution_contract"]
    need(policy["allowed_task_ids"] == ["LIR-9001"] and policy["other_tasks_phase_blocked"] is True, "bootstrap scope differs")
    need(policy["native_grant_supplied"] is False and policy["launch_started"] is False and policy["code_or_test_edits_allowed"] is False, "invented activation grant")
    need(policy["writable_result_paths"] == [RESULT] and policy["finite_controller_duration_seconds"] == 900, "bootstrap output/duration differs")
    return {"valid": True, "scope": "source-only bootstrap documents; not native launch qualification or task completion", "tasks": 64, "goals": 34, "legacy_parents": 24, "todo": ["LIR-9001"], "blocked": 63, "ancestor_bound": ancestor != ANCESTOR_PLACEHOLDER, "owner_supplied_repository_observations": current_observations, "current_source_metadata_factually_authenticated": False, "product_or_provider_execution": False}


def check_report(path):
    summary = check_all()
    report = load(path)
    expected_fields = {"schema", "task_id", "phase", "board_namespace", "source_copy_manifest_sha256", "authoring_tasks_sha256", "execution_board_sha256", "current_source_bindings_sha256", "task_ids", "goal_ids", "legacy_parent_crosswalk", "repository_observations", "lac_relationships", "collisions", "reuse_recommendations", "unresolved_authorities", "next_phase", "effects", "independent_acceptance_required"}
    need(type(report) is dict and set(report) == expected_fields, "report fields differ")
    need(report["schema"] == "legal-ir-supervisor-handoff-reconciliation/v1" and report["task_id"] == "LIR-9001" and report["phase"] == PHASE, "report identity differs")
    authoring = load(ROOT / "authoring/tasks.json")
    need(report["board_namespace"] == authoring["board_namespace"], "report namespace differs")
    for key, name in [("source_copy_manifest_sha256", "source-copy-manifest.json"), ("authoring_tasks_sha256", "authoring/tasks.json"), ("execution_board_sha256", "tasks.todo.md"), ("current_source_bindings_sha256", "CURRENT_SOURCE_BINDINGS.json")]:
        need(report[key] == sha(raw(ROOT / name)), "report source pin differs: " + key)
    need(report["task_ids"] == [row["task_id"] for row in authoring["tasks"]], "report task population differs")
    need(report["goal_ids"] == [row["goal_id"] for row in authoring["goal_hierarchy"]], "report goal population differs")
    need(report["legacy_parent_crosswalk"] == authoring["legacy_parent_crosswalk"], "report dependency crosswalk differs")
    observations = report["repository_observations"]
    need(type(observations) is list and 1 <= len(observations) <= 4, "bounded actual repository metadata required")
    for item in observations:
        need(type(item) is dict and set(item) == {"repository", "source", "head", "tree", "branch", "status", "limitations"}, "repository observation fields differ")
        need(all(type(item[k]) is str and item[k].strip() for k in ["repository", "source", "status"]), "repository observation lacks origin")
        need(item["status"] in {"observed_metadata", "unavailable"}, "unsupported observation authority")
        if item["status"] == "observed_metadata":
            need(all(type(item[k]) is str and re.fullmatch(r"[0-9a-f]{40}", item[k]) for k in ["head", "tree"]), "observed repository identities required")
        else:
            need(item["head"] is None and item["tree"] is None and item["branch"] is None, "unavailable repository observation invented")
        need(type(item["limitations"]) is list and all(type(x) is str for x in item["limitations"]), "observation limitations required")
    for field in ["lac_relationships", "collisions", "reuse_recommendations", "unresolved_authorities"]:
        need(type(report[field]) is list and len(report[field]) <= 256 and all(type(x) is dict for x in report[field]), "bounded structured findings required: " + field)
    need(report["next_phase"] == {"proposed_task_id": "LIR-9002", "activated": False, "requires_independent_acceptance_and_fresh_execution_revision": True}, "report activated next phase")
    effect_names = {"product_imports", "tests_or_native_jobs", "corpus_model_prediction_payload_reads", "code_adoptions_or_edits", "database_mutations", "ledger_mutations", "reviews_or_gold_created", "task_state_mutations", "phase_activation"}
    need(type(report["effects"]) is dict and set(report["effects"]) == effect_names and all(value is False for value in report["effects"].values()), "report claims forbidden effects")
    need(report["independent_acceptance_required"] is True, "report self-admits completion")
    summary.update(scope="bounded source-only report shape and frozen-document correspondence; independent factual acceptance still required", report_checked=True, completion_authority=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--check-all", action="store_true")
    options.add_argument("--check-report", type=Path)
    options.add_argument("--render", action="store_true", help="Deliberately write the exact task Markdown projection only.")
    args = parser.parse_args()
    try:
        if args.render:
            projection = expected_projection(load(ROOT / "authoring/tasks.json"))
            need(load(ROOT / "tasks.execution.json") == projection, "projection differs before rendering")
            (ROOT / "tasks.todo.md").write_text(board_markdown(projection), encoding="utf-8")
            result = {"valid": True, "rendered": "tasks.todo.md", "product_or_provider_execution": False}
        else:
            result = check_report(args.check_report) if args.check_report else check_all()
        print(json.dumps(result, sort_keys=True))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)], "product_or_provider_execution": False}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
