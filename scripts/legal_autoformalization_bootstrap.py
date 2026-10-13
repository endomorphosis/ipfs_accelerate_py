#!/usr/bin/env python3
"""Bounded, read-only evidence jobs for the LegalIR supervisor bootstrap.

Only the requested task's data directory is written. Completion of this job
certifies inventory integrity, never legal fidelity, model promotion, or full
acceptance of the corresponding implementation-plan task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("LAC-001", "LAC-002", "LAC-003")
SCHEMA = "legal-autoformalization-bootstrap-evidence/v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_DISCOVERIES = 512
AUDIT_NAMES = ("retained-run-index.json", "historical-and-feature-runs.json",
               "recent-training-receipts.json")
NUMERIC_KEYS = {"actual_new_updates", "actual_updates", "source_optimizer_steps",
                "optimizer_steps", "completed_updates", "updates_completed",
                "optimizer_accepted_epochs", "cycles", "training_count",
                "sample_count", "validation_sample_count"}
IDENTITY_KEYS = {"initial_checkpoint_sha256", "initial_model_state_sha256",
                 "final_model_state_sha256", "source_model_state_sha256",
                 "checkpoint_sha256", "checkpoint_path", "state_path",
                 "producer", "producer_sha256", "recipe", "recipe_sha256",
                 "latent_enabled", "configured_dimension", "dimension",
                 "fresh_Adam", "inherited_optimizer_moments",
                 "optimizer_history_persisted", "projection_optimizer_scope",
                 "qualified", "admitted", "default_promotion", "deprecated",
                 "status", "execution_mode", "execution_path", "finished_at"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_identity(path, role="immutable"):
    path = Path(path).resolve()
    stat_before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
            stat_after.st_size, stat_after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path}")
    return {"path": str(path), "bytes": stat_after.st_size,
            "sha256": h.hexdigest(), "role": role}


def load_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON read cap exceeded: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pointer_get(value, pointer):
    for segment in pointer.strip("/").split("/") if pointer else ():
        segment = segment.replace("~1", "/").replace("~0", "~")
        value = value[int(segment)] if isinstance(value, list) else value[segment]
    return value


def field_evidence(value, keys, prefix="", limit=128):
    found = []
    stack = [(prefix, value)]
    while stack and len(found) < limit:
        pointer, node = stack.pop()
        if isinstance(node, dict):
            for key, child in sorted(node.items(), reverse=True):
                child_pointer = pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")
                if key in keys and (child is None or isinstance(child, (str, int, float, bool))):
                    found.append({"pointer": child_pointer, "key": key, "value": child})
                if isinstance(child, dict):
                    stack.append((child_pointer, child))
                # Metrics and tensor arrays are intentionally not traversed.
    return sorted(found, key=lambda row: row["pointer"])


def git(repo, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, timeout=20, env=env, check=False)
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed in {repo}: {result.stderr[:240]}")
    if len(result.stdout) > 2 * 1024 * 1024:
        raise ValueError(f"Git output cap exceeded in {repo}")
    return result.stdout.strip()


def add_input(result, path, role="immutable", expected_sha256=None):
    identity = file_identity(path, role)
    existing = next((x for x in result["input_artifacts"] if x["path"] == identity["path"]), None)
    if existing is None:
        result["input_artifacts"].append(identity)
    elif existing["sha256"] != identity["sha256"]:
        raise ValueError(f"Input moved during collection: {identity['path']}")
    if expected_sha256 and identity["sha256"] != expected_sha256:
        result["gaps"].append({"kind": "declared_hash_mismatch", "path": identity["path"],
                               "declared_sha256": expected_sha256,
                               "observed_sha256": identity["sha256"]})
    return identity


def read_input(result, path, expected_sha256=None):
    identity = add_input(result, path, expected_sha256=expected_sha256)
    value = load_json(path)
    if file_identity(path)["sha256"] != identity["sha256"]:
        raise ValueError(f"Input moved while parsing: {path}")
    return value


def common_result(task, sources_path, sources):
    workspace = Path(sources["workspace_root"]).resolve()
    plan = workspace / "implementation_plan/legal_autoformalization_convergence_20261013/tasks.json"
    result = {
        "schema": SCHEMA, "campaign_id": sources["campaign_id"], "task_id": task,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bootstrap evidence reconciliation",
        "original_task_acceptance": "partial",
        "legal_semantic_acceptance": False,
        "coverage": {"scope": "explicitly enumerated retained inputs", "complete": False,
                     "limits": {"json_read_cap_bytes": MAX_JSON_BYTES,
                                "new_discovery_cap": MAX_DISCOVERIES,
                                "large_models_loaded": False,
                                "network_reobservation_performed": False}},
        "input_artifacts": [], "observations": {}, "gaps": [],
        "side_effects": {"training_updates": 0, "model_inference": False,
                         "published": False, "source_mutations": [], "aliases_created": 0},
    }
    result["source_manifest_sha256"] = add_input(result, sources_path)["sha256"]
    result["bootstrap_source_sha256"] = add_input(result, __file__)["sha256"]
    result["plan_sha256"] = add_input(result, plan)["sha256"]
    if result["plan_sha256"] != sources["original_plan_sha256"]:
        raise ValueError("Original authoring plan hash differs from launch pin")
    if sources.get("bootstrap_source_sha256") and result["bootstrap_source_sha256"] != sources["bootstrap_source_sha256"]:
        raise ValueError("Bootstrap script differs from launch pin")
    return result


def collect_repositories(result, sources):
    workspace = Path(sources["workspace_root"])
    repos = [("workspace", workspace),
             ("accelerator_canonical_observation", workspace / "external/ipfs_accelerate"),
             ("datasets_stale_checkout", workspace / "external/ipfs_datasets"),
             ("datasets_current_worktree", Path(sources["datasets_root"])),
             ("accelerator_isolated_integration", Path(sources["accelerator_root"]))]
    if ROOT.resolve() != Path(sources["accelerator_root"]).resolve():
        repos.append(("executing_worker_checkout", ROOT))
    owners = {
        "accelerator": ["ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
                        "ipfs_accelerate_py/agent_supervisor/objectives/objective_graph.py"],
        "datasets": ["ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_runtime_registry.py",
                     "ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_span_split_features.py",
                     "ipfs_datasets_py/logic/autoformal/entity_cache.py"],
        "workspace": [".gitmodules", "implementation_plan/legal_autoformalization_convergence_20261013/tasks.json"],
    }
    rows = []
    for role, path in repos:
        row = {"role": role, "path": str(path.resolve()), "source_files": []}
        try:
            row.update(head=git(path, "rev-parse", "HEAD"),
                       tree=git(path, "rev-parse", "HEAD^{tree}"),
                       git_common_dir=git(path, "rev-parse", "--git-common-dir"),
                       latest_commit_utc=git(path, "log", "-1", "--format=%cI"))
            # Do not expose file contents or environment; only status descriptors.
            status = git(path, "status", "--porcelain=v1", "--untracked-files=no")
            row["dirty_status_sha256"] = hashlib.sha256(status.encode()).hexdigest()
            row["dirty_tracked_entries"] = len(status.splitlines())
            row["worktrees"] = git(path, "worktree", "list", "--porcelain").splitlines()
            row["submodule_links"] = git(path, "ls-tree", "HEAD").splitlines()
            family = "datasets" if role.startswith("datasets") else ("workspace" if role == "workspace" else "accelerator")
            for rel in owners[family]:
                file_path = path / rel
                if file_path.is_file():
                    row["source_files"].append(add_input(result, file_path, "mutable_observation"))
                else:
                    result["gaps"].append({"kind": "missing_owner_file", "role": role, "path": str(file_path)})
            row["head_after"] = git(path, "rev-parse", "HEAD")
            row["tree_after"] = git(path, "rev-parse", "HEAD^{tree}")
            if row["head_after"] != row["head"] or row["tree_after"] != row["tree"]:
                result["gaps"].append({"kind": "repository_moved_during_observation", "role": role})
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            row["error"] = str(exc)
            result["gaps"].append({"kind": "repository_observation_unavailable", "role": role, "detail": str(exc)})
        rows.append(row)
    result["observations"] = {"repositories": rows,
        "authority_selection": {"datasets_role": "datasets_current_worktree",
            "accelerator_role": "accelerator_isolated_integration",
            "stale_checkout_is_not_current_authority": True,
            "integration_branch_created_by": "orchestrator; worker inventory only"}}
    result["gaps"].append({"kind": "live_campaign_lease_authority_not_inventoried",
                           "detail": "No leases, foreign state stores or provider credentials opened."})


def audit_rows(path, audit):
    name = path.name
    if name == "historical-and-feature-runs.json":
        review = audit.get("june_review", {})
        for index, row in enumerate(review.get("runs", [])):
            yield f"/june_review/runs/{index}", "historical_june", row, review.get("report"), review.get("sha256")
        for key, lineage in (("september_daemons", "september_daemon"), ("feature_workers", "september_feature_worker")):
            for index, row in enumerate(audit.get(key, [])):
                yield f"/{key}/{index}", lineage, row, row.get("report"), row.get("report_sha256")
    else:
        lineage = "october_retained_formula" if name == "retained-run-index.json" else "recent_source_span"
        for index, row in enumerate(audit.get("runs", [])):
            yield f"/runs/{index}", lineage, row, row.get("report"), row.get("report_sha256", row.get("sha256"))


def run_record(audit_path, pointer, lineage, row, report_path, expected_sha):
    name = row.get("run_id") or row.get("cell") or row.get("arm") or row.get("job_id") or Path(report_path or "unknown").name
    # Shared June report contains distinct run records; report hash alone is not a run ID.
    key = {"report": str(Path(report_path).resolve()) if report_path else None,
           "reported_sha256": expected_sha, "run": name, "lineage": lineage}
    evidence = field_evidence(row, NUMERIC_KEYS | IDENTITY_KEYS)
    identities = {key: row.get(key) for key in ("states", "training_report_reference",
        "base_checkpoint", "candidate", "candidate_state_identity", "state_path") if key in row}
    missing = [key for key in ("producer", "recipe", "exposure") if not any(
        item["key"] == key for item in evidence)]
    return {"run_id": digest(key), "label": str(name), "lineage": lineage,
            "source": {"path": str(audit_path.resolve()), "pointer": pointer,
                       "record_sha256": digest(row)},
            "report": {"path": key["report"], "declared_sha256": expected_sha},
            "evidence": evidence, "declared_checkpoint_identities": identities,
            "missing_identity_fields": missing,
            "qualification": "retained historical qualification; no promotion by this job"}


def discover_newer(sources, cutoff):
    workspace = Path(sources["workspace_root"])
    candidates = []
    # Shallow bounded discovery, not a recursively complete disk history census.
    for directory in (workspace / "artifacts", workspace / "external/ipfs_datasets/workspace/test-logs"):
        if not directory.is_dir():
            continue
        children = sorted(directory.iterdir(), key=lambda path: path.name)
        for child in children:
            if len(candidates) >= MAX_DISCOVERIES:
                return candidates, True
            lowered = child.name.lower()
            if not any(word in lowered for word in ("legal", "decoder", "autoencoder", "span", "formal", "af029")):
                continue
            if child.stat().st_mtime < cutoff:
                continue
            candidates.append({"path": str(child.resolve()), "mtime_ns": child.stat().st_mtime_ns,
                               "disposition": "discovery_only; directory mtime is not proof of a training run"})
    return candidates, False


def collect_runs(result, sources):
    directory = Path(sources["workspace_root"]) / "artifacts/legal-autoencoder-training-history-20261013"
    runs = {}
    duplicates = []
    report_cache = {}
    source_count = 0
    audits = []
    for name in AUDIT_NAMES:
        path = directory / name
        audit = read_input(result, path)
        audits.append(str(path.resolve()))
        for pointer, lineage, row, report_path, expected in audit_rows(path, audit):
            source_count += 1
            record = run_record(path, pointer, lineage, row, report_path, expected)
            rid = record["run_id"]
            if rid in runs:
                duplicates.append({"run_id": rid, "additional_source": record["source"]})
                continue
            if report_path:
                actual_path = Path(report_path)
                if actual_path.exists():
                    try:
                        if str(actual_path) not in report_cache:
                            identity = add_input(result, actual_path, expected_sha256=expected)
                            parsed = load_json(actual_path) if actual_path.suffix == ".json" else None
                            report_cache[str(actual_path)] = (identity, parsed)
                        identity, parsed = report_cache[str(actual_path)]
                        record["report"].update(observed_sha256=identity["sha256"], bytes=identity["bytes"])
                        if parsed is not None:
                            record["report_evidence"] = field_evidence(parsed, NUMERIC_KEYS | IDENTITY_KEYS)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        result["gaps"].append({"kind": "report_read_refused", "path": str(actual_path), "detail": str(exc)})
                else:
                    result["gaps"].append({"kind": "missing_report", "path": str(actual_path), "run_id": rid})
            runs[rid] = record
    newer, capped = discover_newer(sources, (directory / AUDIT_NAMES[-1]).stat().st_mtime)
    result["observations"] = {"audit_inputs": audits, "source_record_count": source_count,
        "deduplicated_run_count": len(runs), "runs": list(runs.values()),
        "duplicate_records": duplicates, "newer_directory_discoveries": newer,
        "discovery_capped": capped,
        "lineage_edges": [], "lineage_edges_status": "checkpoint identities retained; ancestry not inferred from names"}
    result["gaps"].append({"kind": "lineage_identity_gaps",
        "records_with_missing_identity_fields": sum(bool(x["missing_identity_fields"]) for x in runs.values()),
        "detail": "Producer/recipe/exposure/optimizer/checkpoint qualification needs full task follow-up; no model state loaded."})
    if newer:
        result["gaps"].append({"kind": "newer_evidence_requires_triage", "count": len(newer)})


def collect_caches(result, sources):
    workspace = Path(sources["workspace_root"])
    datasets = Path(sources["datasets_root"])
    summary_path = workspace / "artifacts/uscode-span-rollout-20261011/campaign-summary.json"
    recovery_path = datasets / "docs/autoencoders/evidence/uscode-identity-recovery-20261011-01/index.json"
    baseline_path = workspace / "implementation_plan/legal_autoformalization_convergence_20261013/evidence_baseline.json"
    summary = read_input(result, summary_path)
    recovery = read_input(result, recovery_path)
    baseline = read_input(result, baseline_path)
    pin = recovery["parent_coverage"]["source_summary_pin"]
    if Path(pin["path"]).resolve() != summary_path.resolve() or pin["sha256"] != file_identity(summary_path)["sha256"]:
        raise ValueError("Identity recovery parent summary binding mismatch")
    counts = summary["final_accounting"]
    fields = [
        ("archived_distinct_ids", summary["archived_span_ids"], summary_path, "/archived_span_ids", True),
        ("matched_archived_ids", summary["matched_archived_span_ids"], summary_path, "/matched_archived_span_ids", True),
        ("unresolved_archived_ids", summary["unresolved_archived_span_ids"], summary_path, "/unresolved_archived_span_ids", True),
        ("regenerated_source_occurrences", counts["source_occurrence_count"], summary_path, "/final_accounting/source_occurrence_count", False),
        ("retained_target_occurrences", counts["target_occurrence_count"], summary_path, "/final_accounting/target_occurrence_count", True),
        ("outside_target_aggregate_occurrences", counts["outside_target_occurrence_count"], summary_path, "/final_accounting/outside_target_occurrence_count", False),
    ]
    result["observations"] = {
        "populations": [{"name": name, "count": count, "source": str(path.resolve()),
                         "pointer": pointer, "complete_identities_retained": retained,
                         "identity_scope": "pinned historical source/archive generation"}
                        for name, count, path, pointer, retained in fields],
        "source_scan_complete": summary["complete_source_scan"],
        "complete_all_source_occurrence_inventory": summary["complete_all_source_occurrence_inventory"],
        "final_cursor": summary["final_cursor"],
        "identity_recovery": {"stage": recovery["stage"], "scope": recovery["scope"],
                              "aliases_promoted": recovery.get("aliases_promoted"),
                              "parent_coverage": recovery["parent_coverage"]},
        "datasets": [{"repository": row["repository"], "pinned_revision": row["observed_revision"],
                      "current_revision": None, "observation_kind": "historical pin, not live Hub reobservation"}
                     for row in baseline["datasets"]],
        "window_accounting": "cumulative snapshots; inherited prefixes never summed",
    }
    result["gaps"].extend([
        {"kind": "unresolved_archive_identity", "count": summary["unresolved_archived_span_ids"]},
        {"kind": "outside_target_full_identity_inventory_missing", "count": counts["outside_target_occurrence_count"]},
        {"kind": "identity_recovery_stage", "stage": recovery["stage"]},
        {"kind": "live_hub_revision_not_reobserved", "detail": "Bounded offline bootstrap; later optional public metadata check required."},
    ])


def output_path(task, requested):
    allowed = (ROOT / "data/legal_autoformalization_convergence" / task).resolve()
    path = Path(requested) if requested else allowed / "result.json"
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(allowed):
        raise ValueError(f"Output must remain inside {allowed}")
    return path


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".result-", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def validate(result, task, sources_path, sources):
    need(result.get("schema") == SCHEMA, "Wrong result schema")
    need(result.get("task_id") == task, "Wrong task identity")
    need(result.get("campaign_id") == sources["campaign_id"], "Wrong campaign")
    need(result.get("scope") == "bootstrap evidence reconciliation", "Wrong result scope")
    need(result.get("original_task_acceptance") == "partial", "Bootstrap cannot confer original full acceptance")
    need(result.get("legal_semantic_acceptance") is False, "Bootstrap cannot confer legal meaning")
    need(result.get("side_effects") == {"training_updates": 0, "model_inference": False,
        "published": False, "source_mutations": [], "aliases_created": 0}, "Unexpected declared side effects")
    need(result.get("coverage", {}).get("complete") is False, "Unbounded completeness claim")
    need(result.get("plan_sha256") == sources["original_plan_sha256"], "Plan pin mismatch")
    need(result.get("source_manifest_sha256") == file_identity(sources_path)["sha256"], "Source manifest moved")
    need(result.get("bootstrap_source_sha256") == file_identity(__file__)["sha256"], "Collector code moved")
    inputs = {}
    warnings = []
    for row in result["input_artifacts"]:
        need(row["path"] not in inputs, "Duplicate input identity")
        inputs[row["path"]] = row
        try:
            actual = file_identity(row["path"])
            if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
                if row.get("role") == "mutable_observation":
                    warnings.append({"kind": "mutable_source_moved", "path": row["path"]})
                else:
                    raise ValueError(f"Immutable input digest mismatch: {row['path']}")
        except OSError:
            if row.get("role") == "mutable_observation":
                warnings.append({"kind": "mutable_source_unavailable", "path": row["path"]})
            else:
                raise
    for required in (str(Path(sources_path).resolve()), str(Path(__file__).resolve()),
                     str((Path(sources["workspace_root"]) / "implementation_plan/legal_autoformalization_convergence_20261013/tasks.json").resolve())):
        need(required in inputs, f"Required primary input omitted: {required}")
    observations = result["observations"]
    if task == "LAC-001":
        rows = observations["repositories"]
        roles = {row["role"] for row in rows}
        need({"workspace", "accelerator_canonical_observation", "datasets_stale_checkout",
              "datasets_current_worktree", "accelerator_isolated_integration"}.issubset(roles), "Required repository omitted")
        for row in rows:
            if "error" in row:
                need(any(gap.get("role") == row["role"] for gap in result["gaps"]), "Repository error lacks gap")
                continue
            need(len(row["head"]) == 40 and len(row["tree"]) == 40, "Invalid Git identity")
            for source in row["source_files"]:
                need(source["path"] in inputs and inputs[source["path"]]["sha256"] == source["sha256"], "Owner source binding absent")
            try:
                if git(row["path"], "rev-parse", "HEAD") != row["head_after"]:
                    warnings.append({"kind": "repository_head_moved_after_observation", "role": row["role"]})
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                warnings.append({"kind": "repository_recheck_unavailable", "detail": str(exc)})
    elif task == "LAC-002":
        audit_dir = Path(sources["workspace_root"]) / "artifacts/legal-autoencoder-training-history-20261013"
        expected = {}
        source_count = 0
        for name in AUDIT_NAMES:
            path = audit_dir / name
            need(str(path.resolve()) in inputs, "Required audit index missing")
            for pointer, lineage, row, report_path, sha in audit_rows(path, load_json(path)):
                source_count += 1
                record = run_record(path, pointer, lineage, row, report_path, sha)
                expected.setdefault(record["run_id"], record)
        runs = observations["runs"]
        need(len({row["run_id"] for row in runs}) == len(runs), "Duplicate run rows")
        need({row["run_id"] for row in runs} == set(expected), "Audit run coverage differs")
        need(observations["source_record_count"] == source_count, "Source-row denominator differs")
        need(observations["deduplicated_run_count"] == len(expected), "Run denominator differs")
        need(len(observations["duplicate_records"]) == source_count - len(expected), "Duplicate denominator differs")
        report_json_cache = {}
        for run in runs:
            original = expected[run["run_id"]]
            for key in ("label", "lineage", "source", "evidence", "declared_checkpoint_identities", "missing_identity_fields"):
                need(run[key] == original[key], f"Run source binding differs: {run['label']} {key}")
            report = run["report"]
            need(report["path"] == original["report"]["path"], "Report path differs")
            need(report["declared_sha256"] == original["report"]["declared_sha256"], "Declared report pin differs")
            if report.get("observed_sha256"):
                need(report["path"] in inputs, "Report input missing")
                need(report["observed_sha256"] == inputs[report["path"]]["sha256"], "Report hash binding differs")
            if "report_evidence" in run:
                report_path = report["path"]
                if report_path not in report_json_cache:
                    report_json_cache[report_path] = load_json(report_path)
                for evidence in run["report_evidence"]:
                    value = pointer_get(report_json_cache[report_path], evidence["pointer"])
                    need(value == evidence["value"] and type(value) is type(evidence["value"]), "Numeric/identity report evidence differs")
        labels = " ".join(str(row.get("label", "")) for row in runs).lower()
        need("temporal" in labels and "margin" in labels, "Recent temporal experiment missing")
    else:
        summary_path = Path(sources["workspace_root"]) / "artifacts/uscode-span-rollout-20261011/campaign-summary.json"
        recovery_path = Path(sources["datasets_root"]) / "docs/autoencoders/evidence/uscode-identity-recovery-20261011-01/index.json"
        need(str(summary_path.resolve()) in inputs and str(recovery_path.resolve()) in inputs, "Required cache metadata missing")
        counts = {}
        for population in observations["populations"]:
            need(population["source"] in inputs, "Population source missing")
            count = population["count"]
            need(type(count) is int and count >= 0, "Invalid population count")
            need(pointer_get(load_json(population["source"]), population["pointer"]) == count, "Population count differs from machine source")
            counts[population["name"]] = count
        need(counts["archived_distinct_ids"] == counts["matched_archived_ids"] + counts["unresolved_archived_ids"], "Archive accounting does not balance")
        need(counts["regenerated_source_occurrences"] == counts["retained_target_occurrences"] + counts["outside_target_aggregate_occurrences"], "Source accounting does not balance")
        summary = load_json(summary_path)
        recovery = load_json(recovery_path)
        need(observations["source_scan_complete"] == summary["complete_source_scan"], "Source scan claim differs")
        need(observations["complete_all_source_occurrence_inventory"] == summary["complete_all_source_occurrence_inventory"], "Identity inventory claim differs")
        need(observations["final_cursor"] == summary["final_cursor"], "Cursor differs")
        need(observations["identity_recovery"]["parent_coverage"] == recovery["parent_coverage"], "Recovery denominator differs")
        need(observations["identity_recovery"]["stage"] == recovery["stage"], "Recovery stage differs")
        need(any(gap["kind"] == "outside_target_full_identity_inventory_missing" for gap in result["gaps"]), "Identity gap omitted")
    return {"valid": True, "status": "validated_inventory_integrity", "task_id": task,
            "original_task_acceptance": "partial", "input_count": len(inputs),
            "authority_sealed": False,
            "validation_scope": "immutable input and inventory integrity; mutable observations remain time bounded",
            "current_mutable_observations_match": not warnings,
            "gap_count": len(result["gaps"]), "mutable_observation_warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "validate"))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--sources", default="config/legal_autoformalization_bootstrap_sources.json")
    parser.add_argument("--output", help="Task-owned result path; defaults to data/.../<task>/result.json")
    args = parser.parse_args()
    sources_path = Path(args.sources)
    if not sources_path.is_absolute():
        sources_path = ROOT / sources_path
    sources = load_json(sources_path)
    for key in ("campaign_id", "workspace_root", "datasets_root", "accelerator_root", "original_plan_sha256"):
        need(isinstance(sources.get(key), str) and sources[key], f"Missing source setting: {key}")
    path = output_path(args.task, args.output)
    if args.command == "collect":
        result = common_result(args.task, sources_path, sources)
        {"LAC-001": collect_repositories, "LAC-002": collect_runs,
         "LAC-003": collect_caches}[args.task](result, sources)
        atomic_write(path, result)
        print(json.dumps({"status": "evidence_collected", "task_id": args.task,
                          "output": str(path), "sha256": file_identity(path)["sha256"],
                          "gaps": len(result["gaps"]), "original_task_acceptance": "partial"}))
    else:
        result = load_json(path)
        print(json.dumps(validate(result, args.task, sources_path, sources), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"valid": False, "status": "refused", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
