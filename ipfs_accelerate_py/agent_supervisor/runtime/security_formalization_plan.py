"""Consume datasets formalization evidence through the native plan compiler."""
from __future__ import annotations

from ..proof.formal_verification_contracts import content_identity


def compile_security_formalization_plan(*, report: dict) -> dict:
    """Consume a replay-validated report; a candidate is never completion.

    The advisor performs full source/model replay before calling this adapter.
    This additional check prevents accidental use of a stale modified report.
    """
    from ..planning.formal_plan_compiler import (
        FORMAL_PLAN_INPUT_SCHEMA, CompilationStatus, compile_formal_plan,
    )
    if report.get("schema") != "security-formalization-pipeline@1":
        raise ValueError("datasets formalization pipeline report required")
    from ipfs_datasets_py.logic.ir_core.identity import canonical_identity
    identity = canonical_identity({key: value for key, value in report.items() if key != "report_cid"},
        domain="security-ir/formalization-pipeline", schema_version=report["schema"]).cid
    if (report.get("report_cid") != identity or report.get("proof_authority") is not False
            or report.get("completion_authority") is not False or report.get("executes_source") is not False):
        raise ValueError("current non-authoritative formalization report required")
    tree = content_identity({"sources": report["source_hashes"]})
    root = report["report_cid"]
    def artifact_id(value):
        # Neural scores are evidence bytes, not propositions in the native
        # proof-contract vocabulary (which deliberately excludes floats).
        return canonical_identity(value, domain="security-ir/formalization-evidence",
                                  schema_version="security-formalization-evidence@1").cid
    models = []
    for row in report["function_results"]:
        if row["learned"].get("status") == "accepted" or row.get("learned_header"):
            models.append({"model_id": artifact_id({"learned": row["learned"],
                "header": row.get("learned_header")}), "path": row["path"],
                "symbol": row["symbol"], "producer": "learned_constrained_decoder",
                "source_binding": row["source_binding"]})
    for header in report["header_models"]:
        if not header.get("modeled_symbols"):
            continue
        # The full native header artifact remains an input, including all
        # declarations, assumptions and explicit unsupported cases.
        models.append({"model_id": artifact_id(header), "path": header.get("source_path", ""),
            "symbol": "reviewed-header-contracts", "producer": "deterministic_header_model"})
    goal = "SECURITY-SOURCE-MODELS"
    subgoal = "SECURITY-VERIFY-MODELS"
    tasks, ast_records = [], []
    for index, model in enumerate(models):
        task_id = "SEC-VERIFY-" + str(index + 1)
        scope = "symbol:cid:" + model["model_id"]
        tasks.append({"task_id": task_id, "task_cid": content_identity({"input": root, "model": model}),
            "goal_id": goal, "subgoal_id": subgoal, "actor_id": "agent:security-prover",
            "depends_on": [], "resource_needs": ["cpu", "prover"],
            "acceptance_criteria": ["Bind solver evidence to the exact model and its explicit assumptions"],
            "changed_ast_scopes": [scope], "metadata": {"input_root_cid": root, "model": model,
                "candidate_only": True, "completion_authority": False}})
        ast_records.append({"symbol_cid": scope, "tree_cid": tree, "symbol": model["symbol"]})
    # A single bounded frontier task retains the complete report as its input;
    # unsuccessful functions cannot vanish through ranking or plan truncation.
    frontier = "SEC-RESOLVE-FRONTIERS"
    scope = "symbol:cid:" + content_identity({"report": root, "kind": "coverage"})
    tasks.append({"task_id": frontier, "task_cid": content_identity({"report": root, "kind": "frontier"}),
        "goal_id": goal, "subgoal_id": subgoal, "actor_id": "agent:security-modeler",
        "depends_on": [], "resource_needs": ["cpu"],
        "acceptance_criteria": ["Account for every unsupported function and unresolved semantic assumption"],
        "changed_ast_scopes": [scope], "metadata": {"input_root_cid": root,
            "coverage": report["summary"], "candidate_only": True, "completion_authority": False}})
    ast_records.append({"symbol_cid": scope, "tree_cid": tree, "symbol": "formalization-coverage"})
    source = {"schema": FORMAL_PLAN_INPUT_SCHEMA, "repository_tree_id": tree,
        "objectives": [{"goal_id": goal, "goal_cid": content_identity({"goal": goal, "input": root}),
            "owner_actor_id": "agent:security-modeler",
            "acceptance_criteria": ["Source-bound candidates with explicit proof evidence and coverage"],
            "subgoals": [{"subgoal_id": subgoal, "subgoal_cid": content_identity({"subgoal": subgoal, "input": root}),
                "goal_id": goal, "parent_id": goal,
                "acceptance_criteria": ["Validate every candidate and retain all frontiers"]}]}],
        "tasks": tasks, "ast": ast_records,
        "proof_policy": {"policy_cid": "policy:security-formalization-candidates@1",
            "minimum_code_assurance": "candidate", "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:exact-source-replay"]},
        "evidence": [{"evidence_cid": root, "kind": "artifact",
            "metadata": {"input_root_cid": root, "source_hashes": report["source_hashes"],
                         "candidate_only": True}}]}
    result = compile_formal_plan(source)
    if result.status is not CompilationStatus.COMPILED:
        raise ValueError("source-bound security formalization plan did not compile: " + str(result.to_dict()))
    return {"schema": "supervisor-security-formalization-plan@1", "input_root_cid": root,
        "source": source, "compilation": result.to_dict(), "model_task_count": len(models),
        "frontier_task_count": 1, "execution_started": False,
        "provider_calls": 0, "completion_authority": False, "proof_authority": False}
