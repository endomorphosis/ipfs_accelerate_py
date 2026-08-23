"""Compile IR learning campaigns into formal plans and admission projections.

The planner is a deterministic adapter.  It never asks a model to infer
formulas from prose, never grants lease while ``RESULT(task)`` identities are
unresolved, and never attaches proof results to a candidate projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..objectives.ir_learning_campaign_contracts import (
    ACCEPTED_CAMPAIGN_SCHEMAS,
    IRLearningCampaign,
    CampaignBoardTask,
    CampaignDependencyProjection,
    CampaignWorkGraphRole,
)
from ..proof.formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    content_identity,
)
from .formal_plan_compiler import (
    FORMAL_PLAN_INPUT_SCHEMA,
    PlanAdmissionProjection,
    PlanCompilationResult,
    compile_formal_plan,
    project_formal_plan_for_admission,
)


_RESOURCE_CLASSES = {
    "RP-CPU-S": ("cpu",),
    "RP-CPU-M": ("cpu",),
    "RP-IO-PINNED": ("cpu", "io"),
    "RP-PROVER": ("cpu", "prover"),
    "RP-GPU": ("cpu", "gpu"),
    "RP-MIXED": ("cpu", "gpu", "prover", "io"),
}


def _is_campaign_envelope(value: Mapping[str, Any]) -> bool:
    """True only for a full campaign record, not an already-expanded binding."""

    if not isinstance(value, Mapping):
        return False
    if value.get("task_revisions") and not value.get("tasks"):
        return False
    schema = value.get("schema")
    if schema in ACCEPTED_CAMPAIGN_SCHEMAS and value.get("tasks"):
        return True
    return bool(value.get("campaign_id") and value.get("tasks") and value.get("roles"))


def _extract_campaign_payloads(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    campaigns: list[Mapping[str, Any]] = []
    for key in ("campaign", "ir_learning_campaign", "IRLearningCampaign"):
        value = payload.get(key)
        if isinstance(value, Mapping) and _is_campaign_envelope(value):
            campaigns.append(value)
    raw = payload.get("campaigns")
    if isinstance(raw, Mapping) and _is_campaign_envelope(raw):
        campaigns.append(raw)
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, Mapping) and _is_campaign_envelope(item):
                campaigns.append(item)
    if _is_campaign_envelope(payload) and payload.get("schema") in ACCEPTED_CAMPAIGN_SCHEMAS:
        campaigns.append(payload)
    unique: dict[str, Mapping[str, Any]] = {}
    for item in campaigns:
        unique[content_identity(dict(item))] = item
    return tuple(unique[key] for key in sorted(unique))


def parse_ir_learning_campaign(
    value: IRLearningCampaign | Mapping[str, Any],
) -> IRLearningCampaign:
    if isinstance(value, IRLearningCampaign):
        return value
    if not isinstance(value, Mapping):
        raise ContractValidationError("campaign must be an object")
    return IRLearningCampaign.from_dict(value)


def campaign_task_to_formal_record(
    task: CampaignBoardTask,
    *,
    known_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Project one board task into reviewed formal-plan compiler input."""

    known = known_task_ids or set()
    intra = [dep for dep in task.depends_on if dep in known]
    external = [dep for dep in task.depends_on if dep not in known]
    return {
        "task_id": task.task_id,
        "task_cid": task.content_id,
        "goal_id": task.parent_goal,
        "subgoal_id": task.subgoal,
        "actor_id": "agent:%s" % task.work_graph_role.value,
        "depends_on": intra,
        "resource_needs": list(_RESOURCE_CLASSES[task.resource_profile.value]),
        "acceptance_criteria": [task.acceptance_criteria],
        "validation_commands": [task.validation],
        "changed_ast_scopes": ["symbol:cid:%s" % task.work_graph_role.value],
        "metadata": {
            "campaign_schema": "IRLearningCampaign@1",
            "work_graph_role": task.work_graph_role.value,
            "external_depends_on": external,
            "board_fields": {
                name: (
                    [str(item) for item in getattr(task, name)]
                    if isinstance(getattr(task, name), tuple)
                    else getattr(getattr(task, name), "value", getattr(task, name))
                )
                for name in (
                    "status",
                    "completion",
                    "is_schedulable",
                    "priority",
                    "track",
                    "owning_repository",
                    "owned_paths",
                    "base_source_revisions",
                    "source_dataset_revisions",
                    "data_split_identity",
                    "compiler_identity",
                    "decompiler_identity",
                    "model_checkpoint_identity",
                    "objective",
                    "resource_profile",
                    "expected_inputs",
                    "expected_outputs",
                    "allowed_effects",
                    "prohibited_effects",
                    "required_proof_or_evaluation_evidence",
                    "lease_and_checkpoint_policy",
                    "rollback_procedure",
                    "result_identity",
                    "outputs",
                    "bundle",
                    "parallel_lane",
                    "predicted_files",
                    "conflict_policy",
                )
            },
        },
    }


def campaign_to_formal_input(campaign: IRLearningCampaign) -> dict[str, Any]:
    """Expand one campaign into reviewed compiler records."""

    goals: dict[str, dict[str, Any]] = {}
    subgoals_by_goal: dict[str, dict[str, dict[str, Any]]] = {}
    for task in campaign.tasks:
        goal_id = task.parent_goal
        goals.setdefault(
            goal_id,
            {
                "goal_id": goal_id,
                "goal_cid": "goal:cid:%s" % goal_id,
                "owner_actor_id": campaign.owner_actor_id,
                "acceptance_criteria": [task.acceptance_criteria],
                "subgoals": [],
            },
        )
        subgoal_id = task.subgoal
        bucket = subgoals_by_goal.setdefault(goal_id, {})
        bucket.setdefault(
            subgoal_id,
            {
                "subgoal_id": subgoal_id,
                "subgoal_cid": "subgoal:cid:%s" % subgoal_id,
                "goal_id": goal_id,
                "parent_id": goal_id,
                "acceptance_criteria": [task.acceptance_criteria],
            },
        )
    objectives = []
    for goal_id in sorted(goals):
        record = dict(goals[goal_id])
        record["subgoals"] = [
            bucket
            for _, bucket in sorted(subgoals_by_goal.get(goal_id, {}).items())
        ]
        objectives.append(record)

    ast_records = [
        {
            "symbol_cid": "symbol:cid:%s" % role.value,
            "tree_cid": campaign.repository_tree_id,
            "symbol": role.value,
        }
        for role in CampaignWorkGraphRole
    ]
    known_task_ids = {item.task_id for item in campaign.tasks}
    tasks = [
        campaign_task_to_formal_record(item, known_task_ids=known_task_ids)
        for item in campaign.tasks
    ]
    revisions = {item.task_id: item for item in campaign.task_revisions}
    for record in tasks:
        revision = revisions[str(record["task_id"])]
        metadata = dict(record["metadata"])
        metadata["task_revision"] = revision.task_revision
        metadata["lease_eligible"] = revision.lease_eligible
        metadata["unresolved_dependency_outputs"] = list(
            revision.unresolved_dependency_outputs
        )
        metadata["dependency_output_bindings"] = dict(revision.dependency_output_bindings)
        record["metadata"] = metadata
        if revision.lease_eligible:
            record["lease"] = {
                "lease_cid": "lease:cid:%s" % record["task_id"],
                "holder_id": record["actor_id"],
                "fencing_token": 1,
            }

    projection = campaign.project_dependencies()
    return {
        "schema": FORMAL_PLAN_INPUT_SCHEMA,
        "repository_tree_id": campaign.repository_tree_id,
        "objectives": objectives,
        "tasks": tasks,
        "ast": ast_records,
        "proof_policy": {
            "policy_cid": "policy:cid:%s" % campaign.campaign_id,
            "minimum_code_assurance": AssuranceLevel.CANDIDATE.value,
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:campaign-validation"],
            "required_evidence": [
                {
                    "kind": "plan_check",
                    "subject_ids": [item.task_id for item in campaign.tasks],
                    "source_scope_ids": [
                        "symbol:cid:%s" % item.work_graph_role.value for item in campaign.tasks
                    ],
                }
            ],
        },
        "evidence": [
            {
                "evidence_cid": "evidence:cid:campaign-input-root",
                "kind": "artifact",
                "metadata": {"input_root_cid": campaign.input_root_cid},
            }
        ],
        "ir_learning_campaign_binding": {
            "schema": "IRLearningCampaignBinding@1",
            "campaign_schema": "IRLearningCampaign@1",
            "campaign_id": campaign.campaign_id,
            "campaign_revision": campaign.campaign_revision,
            "input_root_cid": campaign.input_root_cid,
            "projection_id": projection.projection_id,
            "roles": [item.role.value for item in campaign.roles],
            "task_revisions": [item.to_record() for item in campaign.task_revisions],
            "lease_eligible_task_ids": list(campaign.lease_eligible_task_ids),
            "blocked_task_ids": list(campaign.blocked_task_ids),
            "unresolved_result_ids": list(projection.unresolved_result_ids),
        },
    }


def expand_campaign_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Merge declared campaigns into formal-plan compiler input.

    Sources without a campaign envelope are returned unchanged.
    """

    campaigns = _extract_campaign_payloads(payload)
    if not campaigns:
        return dict(payload)
    merged = dict(payload)
    objectives = list(payload.get("objectives") or payload.get("objective_records") or ())
    tasks = list(payload.get("tasks") or payload.get("taskboard") or payload.get("task_records") or ())
    ast_records = list(payload.get("ast") or payload.get("ast_records") or ())
    policies = list(payload.get("policies") or payload.get("proof_policy") or ())
    if isinstance(payload.get("proof_policy"), Mapping):
        policies = [payload["proof_policy"], *policies]
    evidence = list(payload.get("evidence") or payload.get("evidence_records") or ())
    campaign_bindings: list[dict[str, Any]] = []
    tree_id = str(payload.get("repository_tree_id") or "").strip()
    for raw in campaigns:
        campaign = parse_ir_learning_campaign(raw)
        expanded = campaign_to_formal_input(campaign)
        tree_id = tree_id or campaign.repository_tree_id
        objectives.extend(expanded["objectives"])
        tasks.extend(expanded["tasks"])
        ast_records.extend(expanded["ast"])
        policies.append(expanded["proof_policy"])
        evidence.extend(expanded["evidence"])
        campaign_bindings.append(expanded["ir_learning_campaign_binding"])
    merged["objectives"] = objectives
    merged["tasks"] = tasks
    merged["ast"] = ast_records
    merged["policies"] = policies
    merged["evidence"] = evidence
    if tree_id:
        merged["repository_tree_id"] = tree_id
    if len(campaign_bindings) == 1:
        merged["ir_learning_campaign_binding"] = campaign_bindings[0]
    else:
        merged["ir_learning_campaign_binding"] = {
            "schema": "IRLearningCampaignBinding@1",
            "campaigns": campaign_bindings,
        }
    return merged


def compile_ir_learning_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
) -> PlanCompilationResult:
    """Compile one campaign through the reviewed formal-plan compiler."""

    record = parse_ir_learning_campaign(campaign)
    return compile_formal_plan(campaign_to_formal_input(record))


def project_campaign_for_admission(
    campaign: IRLearningCampaign | Mapping[str, Any],
) -> tuple[CampaignDependencyProjection, PlanAdmissionProjection | None]:
    """Return campaign and formal-plan admission projections together."""

    record = parse_ir_learning_campaign(campaign)
    campaign_projection = record.project_dependencies()
    compiled = compile_ir_learning_campaign(record)
    formal = None
    if compiled.plan is not None:
        formal = project_formal_plan_for_admission(
            compiled.plan,
            generated_formula_ids=compiled.generated_formula_ids,
        )
    return campaign_projection, formal


__all__ = (
    "campaign_task_to_formal_record",
    "campaign_to_formal_input",
    "compile_ir_learning_campaign",
    "expand_campaign_source",
    "parse_ir_learning_campaign",
    "project_campaign_for_admission",
)
