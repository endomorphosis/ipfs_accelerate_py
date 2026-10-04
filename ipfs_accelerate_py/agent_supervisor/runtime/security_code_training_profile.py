"""Supervisor campaign adapter for a datasets-owned training profile."""
import json
from ipfs_datasets_py.logic.formalization.autoencoder.security.security_code_training_profile import (
    SCHEMA, build_security_code_training_profile, validate_security_code_training_profile,
)

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_security_code_learning_campaign(*, profile: dict, repository_tree_id: str):
    """Compile-able native goals/tasks; unresolved admission prevents all leases.

    Independent projection tasks have separate output paths and may later run
    in parallel. Training depends on every projection's evidence/frontier
    result. No task is completed or executed merely by constructing this DAG.
    """
    from ..objectives.ir_learning_campaign_contracts import (
        CampaignBoardTask, CampaignWorkGraphRole, IRLearningCampaign, default_campaign_roles,
    )
    from ..proof.formal_verification_contracts import content_identity
    profile = validate_security_code_training_profile(profile)
    if type(repository_tree_id) is not str or not repository_tree_id.strip():
        raise ValueError("caller-bound repository tree identity required")
    profile_id = profile["profile_cid"]
    rows = [
        ("SEC-CORPUS", "Verify pinned Publicus code corpus", "corpus", ("SOURCE-ADMISSION",),
         "Verify native source CIDs and exact body hashes before deriving any training inputs."),
        ("SEC-INITIALIZER", "Bind published LegalIR shared weights", "lineage", ("SOURCE-ADMISSION",),
         "Verify published manifest, unchanged source state and exact compatible lexical transfer."),
        ("SEC-SPLITS", "Validate independent repository-family splits", "split", ("SEC-CORPUS",),
         "Exclude benchmark families and reject source/body overlap before fitting vocabulary or model parameters."),
        ("SEC-SOURCE-CONTEXT", "Resolve complete source context where required", "compiler", ("SEC-SPLITS",),
         "Distinguish original fragments from complete compilation units; resolve missing files only at exact source revisions, retaining the fragment-to-file relation and new body identities."),
        ("SEC-FORMALIZE", "Construct source-bound native code models", "compiler", ("SEC-SOURCE-CONTEXT",),
         "Use supported deterministic source adapters with explicit semantic assumptions and specifications; retain unsupported frontiers and never manufacture contracts or formulas from CWE labels."),
    ]
    for projection in profile["code_logic"]["projections"]:
        kind = projection["kind"]
        dependencies = ("SEC-FORMALIZE", "SEC-PROJECT-PROGRAM") if projection["requires_program_binding"] else ("SEC-FORMALIZE",)
        rows.append(("SEC-PROJECT-" + kind.upper(), "Project native " + kind + " declarations",
            "compiler", dependencies,
            "Require exact source-bound %s evidence for %s/%s; preserve unsupported cases and translation losses."
            % (projection["typed_owner"], projection["family"], projection["profile"])))
    projection_ids = tuple(row[0] for row in rows if row[0].startswith("SEC-PROJECT-"))
    rows.extend([
        ("SEC-FIT", "Fit isolated security heads on the training split", "training_run",
         ("SEC-INITIALIZER", *projection_ids),
         "Select admitted source/target pairs; preserve LegalIR; fit only train rows and account for each target head separately."),
        ("SEC-EVALUATE", "Evaluate held-out security and logic candidates", "evaluation", ("SEC-FIT",),
         "Use validation for model selection and sealed test rows for held-out metrics, with coverage, calibration and abstention."),
        ("SEC-RELEASE", "Qualify the candidate checkpoint for publication", "publication", ("SEC-EVALUATE",),
         "Require inference parity, native proof evidence where claimed, provenance review, and exact publication approval."),
    ])
    tasks = []
    sources = _json({"corpus": profile["corpus"]["pin"], "legal_parent": profile["legal_parent"]})
    for task_id, title, role, dependencies, objective in rows:
        relative = "security-training/" + task_id.lower() + "/result.json"
        proof = ("Exact source and native artifact validation; classification and modeling declarations grant no proof authority. "
                 "A solver/model-checker/kernel receipt must bind every claimed proved property.")
        tasks.append(CampaignBoardTask(task_id=task_id, title=title, status="todo", completion="none",
            is_schedulable=False, priority="P1", track="security", parent_goal="SECURITY-CODE-LEARNING",
            subgoal=("SECURITY-CODE-PROJECTIONS" if role == "compiler" else "SECURITY-CODE-" + role.upper()),
            owning_repository="ipfs_datasets_py" if role == "compiler" else "ipfs_accelerate_py",
            owned_paths=(relative,), base_source_revisions=repository_tree_id, source_dataset_revisions=sources,
            data_split_identity=content_identity(profile["corpus"]),
            compiler_identity=profile["code_logic"]["profile_cid"],
            decompiler_identity=profile["code_logic"]["bridge"],
            model_checkpoint_identity="sha256:" + profile["legal_parent"]["state_sha256"],
            objective=objective, depends_on=dependencies,
            resource_profile="RP-GPU" if role == "training_run" else "RP-CPU-M",
            expected_inputs=profile_id + "; " + ", ".join("RESULT(" + dep + ")" for dep in dependencies)
                + ("; exact ProgramIR at security-training/sec-project-program/result.json"
                   if task_id == "SEC-PROJECT-CONTRACT" else ""),
            expected_outputs=relative, allowed_effects="Write only the admitted task's isolated security output namespace.",
            prohibited_effects="Modify LegalIR; mix held-out data into fitting; infer proved formulas from labels; publish without exact approval.",
            acceptance_criteria=objective, required_proof_or_evaluation_evidence=proof,
            lease_and_checkpoint_policy="Resolve exact dependency outputs and obtain native admission before a fenced lease.",
            rollback_procedure="Discard candidate outputs; preserve source checkpoints and previously qualified releases.",
            result_identity="RESULT(" + task_id + ")", outputs=(relative,),
            # A profile unit test cannot validate a fitted head or held-out
            # evaluation. Until a stage-specific evidence runner is admitted,
            # this draft must fail closed even if somebody invokes it directly.
            validation="python -c 'raise SystemExit(\"Draft stage requires an admitted artifact validator\")'",
            bundle="security-code-learning", parallel_lane=task_id.lower(), predicted_files=(relative,),
            conflict_policy="Serialize overlapping writes; independent projection outputs may run in parallel.",
            work_graph_role=CampaignWorkGraphRole(role), metadata={"profile_cid": profile_id,
                "draft_only": True, "stage_executor_configured": False}))
    return IRLearningCampaign(campaign_id="security-code-" + profile_id,
        input_root_cid=profile_id, repository_tree_id=repository_tree_id,
        roles=default_campaign_roles(), tasks=tuple(tasks),
        metadata={"profile_cid": profile_id, "draft_only": True,
                  "source_admission_pending": True, "provider_calls": 0, "training_steps": 0})


def compile_security_code_learning_plan(*, profile: dict, repository_tree_id: str) -> dict:
    """Use the native formal compiler without opening an owner or model provider."""
    from ..planning.ir_learning_campaign_planner import campaign_to_formal_input
    from ..planning.formal_plan_compiler import compile_formal_plan, CompilationStatus
    campaign = build_security_code_learning_campaign(profile=profile, repository_tree_id=repository_tree_id)
    result = compile_formal_plan(campaign_to_formal_input(campaign))
    if result.status is not CompilationStatus.COMPILED:
        raise ValueError("security code learning campaign did not compile")
    if campaign.lease_eligible_task_ids:
        raise ValueError("draft security training campaign unexpectedly granted lease eligibility")
    return {"schema": "security-code-learning-plan@1", "profile_cid": profile["profile_cid"],
        "campaign": campaign.to_record(), "compilation": result.to_dict(),
        "lease_eligible_task_ids": [], "execution_started": False, "provider_calls": 0,
        "training_steps": 0, "proof_authority": False, "publication_authority": False}
