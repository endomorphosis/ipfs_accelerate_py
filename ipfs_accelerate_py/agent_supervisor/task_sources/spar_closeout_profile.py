"""SPAR v1 native requirement observation, never semantic acceptance authority.

The exclusive owner's launcher supplies bootstrap-verified contracts. This
adapter checks those contracts against the same transaction as task receipts.
It deliberately does not promote nominated reports or unverified CID records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .control_plane_contracts import content_identity

SCHEMA = "ipfs_accelerate_py/agent-supervisor/spar-closeout-profile@1"
OBSERVATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/spar-closeout-requirements@1"
POLICY_FIELDS = frozenset(
    {
        "all_task_dependencies_terminal_required",
        "goal_completion_contracts_required",
        "current_tree_required",
        "active_mutating_claims_empty_required",
        "merge_queue_settled_required",
        "blocking_obligations_empty_required",
        "required_receipts_and_seals_verify",
        "non_success_terminals_never_report_success",
        "ducklake_outage_cannot_block_core_completion",
        "final_report_required",
        "required_mode_roots_and_receipts_required",
        "safety_floors_noncompensable",
        "self_hosted_capstone_required",
    }
)
REPORTS = (
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json",
    "benchmarks/agent_supervisor/semantic_refactoring/capstone_report.json",
    "benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json",
)
RECEIPT_COLUMNS = (
    "receipt_cid",
    "task_cid",
    "goal_cid",
    "attempt_id",
    "claim_cid",
    "fencing_token",
    "completed_at",
    "validation_run_id",
    "evidence_digest",
    "body",
)


def _sha(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def observe_source(
    repository_root: str, nested_repositories: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bounded fresh local source observations, not accepted-root receipts."""
    deadline = time.monotonic() + 4.0
    root = Path(repository_root)

    def git(path: Path, *args: str) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("source observation deadline exceeded")
        result = subprocess.run(
            ["git", *args],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
            timeout=remaining,
        )
        return result.stdout.strip()

    try:
        head = git(root, "rev-parse", "HEAD")
        tree = git(root, "rev-parse", "HEAD^{tree}")
        clean = not git(root, "status", "--porcelain=v1", "--untracked-files=all")
        nested = []
        for spec in nested_repositories:
            relative = spec["path"]
            path = root / relative
            if path.resolve().is_relative_to(root.resolve()) is False:
                raise ValueError("nested source path escapes repository")
            revision = git(path, "rev-parse", "HEAD")
            nested_tree = git(path, "rev-parse", "HEAD^{tree}")
            clean = clean and not git(
                path, "status", "--porcelain=v1", "--untracked-files=all"
            )
            gitlink = git(root, "ls-tree", head, "--", relative).split()
            if len(gitlink) < 3 or gitlink[:3] != ["160000", "commit", revision]:
                raise ValueError("nested source differs from parent gitlink")
            nested.append(
                {
                    "repository": spec["repository"],
                    "path": relative,
                    "head": revision,
                    "tree": nested_tree,
                    "planning_revision": spec["planning_revision"],
                    "planning_revision_is_ancestor": True,
                    "access": "read_only_contract_audit",
                }
            )
            git(
                path, "merge-base", "--is-ancestor", spec["planning_revision"], revision
            )
        forest = {
            "source_head": head,
            "nested_repositories": nested,
            "cross_repository_writes": False,
        }
        forest["source_forest_root"] = _sha(forest)
        reports = []
        for relative in REPORTS:
            path = root / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 262144:
                reports.append({"path": relative, "available": False})
                continue
            raw = path.read_bytes()
            report = json.loads(raw)
            if not isinstance(report, dict):
                raise ValueError("report is not an object")
            reports.append(
                {
                    "path": relative,
                    "available": True,
                    "content_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                    "nomination_only": report.get("nomination_only"),
                    "final_root_accepted": report.get("final_root_accepted"),
                    "can_authorize_completion": report.get("can_authorize_completion"),
                    "authority_roots": (
                        report.get("authority_roots")
                        if isinstance(report.get("authority_roots"), dict)
                        else None
                    ),
                    "typed_terminals": sorted(
                        {
                            str(g.get("expected_terminal"))
                            for g in report.get("gates", [])
                            if isinstance(g, dict) and g.get("expected_terminal")
                        }
                    ),
                }
            )
        if git(root, "rev-parse", "HEAD") != head:
            raise ValueError("source changed during observation")
        return {
            "available": True,
            "clean": clean,
            "repository_tree_id": tree,
            "source_forest": forest,
            "reports": reports,
            "semantic_acceptance_authority": False,
        }
    except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as exc:
        return {
            "available": False,
            "reason": "current_source_observation_unavailable",
            "error_class": type(exc).__name__,
            "semantic_acceptance_authority": False,
        }


class SparCloseoutProfile:
    """Closed owner-local profile; no callback, SQL, or path is admitted by RPC."""

    def __init__(self, profile: Mapping[str, Any], *, repository_root: str):
        self._profile = copy.deepcopy(dict(profile))
        self._repository_root = repository_root
        p = self._profile
        policy = p.get("completion_policy", {})
        if (
            set(p)
            != {
                "schema",
                "board_namespace",
                "bootstrap_receipt_id",
                "plan_root_cid",
                "repository_tree_id",
                "source_identities",
                "completion_policy",
                "goals",
                "tasks",
                "goal_edges",
                "nested_repositories",
            }
            or p.get("schema") != SCHEMA
            or p.get("board_namespace")
            != "semantic-preserving-autonomous-remodularization-v1"
            or set(policy) != POLICY_FIELDS | {"terminal_task_id"}
            or any(policy.get(key) is not True for key in POLICY_FIELDS)
            or policy.get("terminal_task_id") != "SPAR-050"
            or len(p.get("goals", [])) != 32
            or len(p.get("tasks", [])) != 51
            or len({g["goal_cid"] for g in p["goals"]}) != 32
            or len({t["task_cid"] for t in p["tasks"]}) != 51
        ):
            raise ValueError("SPAR closeout profile is not the closed v1 contract")
        self.profile_cid = content_identity(p)

    def assert_scope(self, binding: Mapping[str, Any]) -> None:
        p = self._profile
        if any(
            binding.get(key) != p[key]
            for key in ("board_namespace", "plan_root_cid", "repository_tree_id")
        ) or sorted(binding["task_cids"]) != sorted(t["task_cid"] for t in p["tasks"]):
            raise ValueError("SPAR closeout profile differs from native status scope")

    def evaluate(
        self, facts: Mapping[str, Any], snapshot: Mapping[str, Any]
    ) -> dict[str, Any]:
        from .intent_repository import IntentRepository

        p = self._profile
        relations = facts["relations"]
        blockers = []
        for name, relation in relations.items():
            if not relation["available"] or relation["truncated"]:
                blockers.append(f"native_relation_unavailable_or_truncated:{name}")
            if (
                name not in {"tasks", "goals", "goal_edges", "task_dependencies"}
                and relation["rows"]
            ):
                blockers.append(f"native_unsettled_rows:{name}")
        native_goals = {g["goal_cid"]: g for g in relations["goals"]["rows"]}
        native_tasks = {t["task_cid"]: t for t in relations["tasks"]["rows"]}
        if set(native_goals) != {g["goal_cid"] for g in p["goals"]}:
            blockers.append("sealed_goal_population_changed")
        if set(native_tasks) != {t["task_cid"] for t in p["tasks"]}:
            blockers.append("sealed_task_population_changed")
        expected_edges = {
            (e["parent_goal_cid"], e["child_goal_cid"], e["edge_kind"])
            for e in p["goal_edges"]
        }
        actual_edges = {
            (e["parent_goal_cid"], e["child_goal_cid"], e["edge_kind"])
            for e in relations["goal_edges"]["rows"]
        }
        if actual_edges != expected_edges:
            blockers.append("sealed_goal_edges_changed")
        expected_dependencies = {
            (t["task_cid"], d) for t in p["tasks"] for d in t["dependencies"]
        }
        actual_dependencies = {
            (t["task_cid"], t["dependency_task_cid"])
            for t in relations["task_dependencies"]["rows"]
        }
        if actual_dependencies != expected_dependencies:
            blockers.append("sealed_task_dependencies_changed")
        receipts = [
            tuple(
                json.dumps(r[key]) if key == "body" else r.get(key)
                for key in RECEIPT_COLUMNS
            )
            for r in snapshot["completion_projection"]["completion_receipts"]
        ]
        task_evidence = []
        for sealed in p["tasks"]:
            task = native_tasks.get(sealed["task_cid"])
            if task is None:
                continue
            body = json.loads(task["body_json"])
            if (
                task["task_alias"] != sealed["task_alias"]
                or task["goal_cid"] != sealed["goal_cid"]
                or body.get("title") != sealed["title"]
                or any(
                    body.get(key) != value
                    for key, value in sealed["contract_fields"].items()
                )
            ):
                blockers.append(f"sealed_task_contract_changed:{sealed['task_alias']}")
            receipt, reasons = IntentRepository._current_task_completion_binding(
                {**task, "body": body}, receipts
            )
            task_evidence.append(
                {
                    "task_alias": sealed["task_alias"],
                    "task_cid": task["task_cid"],
                    "revision": task["revision"],
                    "receipt": receipt,
                    "blockers": reasons,
                }
            )
            blockers.extend(f"{reason}:{sealed['task_alias']}" for reason in reasons)
        goal_requirements = []
        for sealed in p["goals"]:
            goal = native_goals.get(sealed["goal_cid"])
            reasons = ["independently_accepted_current_root_receipt_required"]
            if goal is None or (
                goal["goal_alias"] != sealed["goal_alias"]
                or goal["title"] != sealed["title"]
                or json.loads(goal["body_json"]).get("body") != sealed["body"]
                or str(goal.get("parent_goal_cid") or "") != sealed["parent_goal_cid"]
            ):
                reasons.append("sealed_goal_contract_changed")
                blockers.append(f"sealed_goal_contract_changed:{sealed['goal_alias']}")
            goal_requirements.append(
                {
                    "goal_alias": sealed["goal_alias"],
                    "goal_cid": sealed["goal_cid"],
                    "native_revision": goal.get("revision") if goal else None,
                    "native_status": goal.get("status") if goal else None,
                    "contract_cid": content_identity(sealed),
                    "completion_contract": sealed["body"]["completion_contract"],
                    "task_cids": [
                        t["task_cid"]
                        for t in p["tasks"]
                        if t["goal_cid"] == sealed["goal_cid"]
                    ],
                    "accepted": False,
                    "blockers": reasons,
                }
            )
        source = observe_source(self._repository_root, p["nested_repositories"])
        if not source["available"]:
            blockers.append("current_source_observation_unavailable")
        else:
            if not source["clean"]:
                blockers.append("current_source_dirty")
            for report in source["reports"]:
                if not report["available"]:
                    blockers.append(f"required_report_unavailable:{report['path']}")
                elif (
                    report.get("nomination_only") is True
                    or report.get("can_authorize_completion") is not True
                ):
                    blockers.append(
                        f"report_is_not_acceptance_authority:{report['path']}"
                    )
            final = source["reports"][0]
            if (final.get("authority_roots") or {}).get(
                "repository_forest_cid"
            ) != source["source_forest"]["source_forest_root"]:
                blockers.append("final_report_current_source_forest_mismatch")
        blockers.extend(
            [
                "datasets_independent_accepted_root_producer_and_admission_required",
                "kit_source_forest_cas_receipt_producer_and_admission_required",
                "required_mode_roots_safety_floors_capstone_fixed_point_acceptance_required",
                "spar_native_goal_cas_settlement_adapter_required",
                "runtime_lane_and_merge_queue_settlement_receipt_required",
            ]
        )
        result = {
            "schema": OBSERVATION_SCHEMA,
            "profile_cid": self.profile_cid,
            "bootstrap_receipt_id": p["bootstrap_receipt_id"],
            "completion_snapshot_cid": snapshot["snapshot_cid"],
            "contracts_compared": True,
            "goal_contracts_accepted": False,
            "completion_authority": False,
            "task_evidence": task_evidence,
            "goal_requirements": goal_requirements,
            "source_observation": source,
            "blockers": sorted(set(blockers)),
        }
        return {**result, "observation_cid": content_identity(result)}
