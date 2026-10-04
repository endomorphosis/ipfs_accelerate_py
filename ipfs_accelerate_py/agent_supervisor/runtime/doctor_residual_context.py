"""Bind real Doctor residuals to the existing admitted task's router context.

Residual work proposals remain candidates. This adapter neither installs a
derived task source nor changes the native task, plan, goal, or write scope.
The signed implementation command pins the public artifact digest. The worker
rechecks that artifact against its native prompt and current source before use.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import stat

from ..context.planner_doctor_context import (
    PlannerDoctorContextRequest, ResidualLlmBudget, compile_planner_doctor_context,
)
from ..objectives.doctor_plan_refill import DoctorPlanResidual
from ..proof.formal_verification_contracts import content_identity
from . import local_planning_admission as local
from .doctor_candidate_runner import _directory, _git, _read, _unique
from .doctor_task_workflow import PreparedDoctorTaskRepair, _assert_task_current, _residual_evidence

SCHEMA = "supervisor-doctor-residual-context@1"
MAX_BYTES = 262_144
FIELDS = frozenset({"schema", "repository", "repository_cid", "base_commit", "task_cid", "task_id",
    "task_revision", "manifest_cid", "contract_cid", "source_manifest", "plan_cid", "goal_cid",
    "refill_receipt_id", "residuals", "capsule", "outputs", "validation_commands", "proposal_counts",
    "provider_calls", "derived_runtime_admitted", "completion_authority", "publication_authority", "context_cid"})


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prepare_doctor_residual_context(*, prepared: PreparedDoctorTaskRepair, result: dict) -> dict:
    """Compile a native capsule while the independently admitted owner is open."""
    if type(prepared) is not PreparedDoctorTaskRepair or result.get("status") != "residual":
        raise ValueError("actual residual Doctor preparation required")
    _assert_task_current(prepared)
    verified = local.verify_local_benchmark_admission(prepared.admission, initial=True)
    manifest = verified["manifest"]
    task = prepared.task
    contract = task["body"][local.CONTRACT_KEY]
    spec = contract["payload"]["task_spec"]
    refill = result.get("plan_refill", {})
    if (any(result.get(key) != prepared.report.get(key) for key in
            ("task_cid", "task_id", "task_revision", "plan_cid", "goal_cid", "manifest_cid", "evidence_id"))
            or refill.get("plan_root") != task["plan_cid"]
            or refill.get("plan_revision") != prepared.plan_context.plan_revision
            or refill.get("next_memory") != prepared.refill.memory.to_dict()
            or any(refill.get(key) is not False for key in
                   ("completion_authority", "mutation_authority", "derived_runtime_admitted", "seed_board_edit"))):
        raise ValueError("Doctor residual differs from current native task or candidate-only refill")
    rows = refill.get("residuals")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError("bounded real Doctor residuals required")
    residuals = tuple(DoctorPlanResidual.from_dict(row) for row in rows)
    residual_root, evidence_refs = _residual_evidence(prepared)
    outputs = [row["path"] for row in spec["outputs"]]
    validations = [shlex.join(row["argv"]) for row in spec["validations"]]
    for original, residual in zip(rows, residuals):
        if (residual.to_dict() != original or residual.residual_id != residual.identity_key
                or residual.parent_task_cid != task["task_cid"] or residual.parent_goal_cid != task["goal_cid"]
                or residual.plan_id != task["plan_cid"]
                or residual.root_id != residual_root or residual.evidence_refs != evidence_refs
                or not set(residual.predicted_files + residual.context_paths) <= set(outputs)
                or tuple(validations) != residual.validation_commands):
            raise ValueError("Doctor residual escaped exact admitted task bindings")
    graph_task = next(row for row in verified["graph"].tasks if row.task_cid == task["task_cid"])
    capsule = compile_planner_doctor_context(PlannerDoctorContextRequest(
        repository_id=manifest["repository_cid"], tree_id=local._tree(manifest["sources"]),
        task_id=task["task_alias"], objective_id=task["goal_cid"],
        intent_summary=graph_task.objective,
        acceptance_ids=tuple(content_identity(row) for row in spec["acceptance"]),
        security_roots=tuple(graph_task.policy_roots),
        open_obligation_ids=tuple(row.residual_id for row in residuals),
        allowed_paths=tuple(outputs), allowed_effects=tuple(sorted({row["effect"] for row in spec["outputs"]})),
        validation_commands=tuple(validations),
        critique_id=refill["receipt_id"], critique_decision="repair_required", deterministic_closure=False,
        # Advisory for one existing coding call; no additional proposal call is
        # authorized, and no rejected/repairable plan record is invented.
        residual_budget=ResidualLlmBudget(max_calls=1, max_tokens=4096, max_rounds=1, max_cost_units=1),
    ))
    payload = {"schema": SCHEMA, "repository": str(prepared.runtime.checkout_root),
        "repository_cid": manifest["repository_cid"], "base_commit": manifest["baseline_commit"],
        "task_cid": task["task_cid"], "task_id": task["task_alias"], "task_revision": task["revision"],
        "manifest_cid": verified["receipt"]["manifest_cid"], "contract_cid": content_identity(contract),
        "source_manifest": manifest["sources"], "plan_cid": task["plan_cid"], "goal_cid": task["goal_cid"],
        "refill_receipt_id": refill["receipt_id"], "residuals": rows, "capsule": capsule.to_dict(),
        "outputs": spec["outputs"], "validation_commands": validations,
        "proposal_counts": {"successors": len(refill.get("successors", [])),
                            "work_proposals": len(refill.get("work_proposals", []))},
        "provider_calls": 0, "derived_runtime_admitted": False,
        "completion_authority": False, "publication_authority": False}
    payload["context_cid"] = content_identity(payload)
    raw = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    if len(raw) > MAX_BYTES:
        raise ValueError("Doctor residual artifact exceeds bound")
    _assert_task_current(prepared)
    parent = prepared.runtime.checkout_root
    for name in (".runtime", "doctor-residuals"):
        parent = parent / name
        try:
            parent.mkdir(mode=0o755)
        except FileExistsError:
            pass
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022 or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError("residual context requires an owner-controlled worker-readable directory")
    artifact = parent / (_sha(raw) + ".json")
    with artifact.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())
    return {"artifact": str(artifact), "sha256": _sha(raw), "task_cid": task["task_cid"],
            "context_cid": payload["context_cid"], "native_capsule_id": capsule.capsule_id,
            "provider_calls": 0, "completion_authority": False, "derived_runtime_admitted": False}


def load_doctor_residual_advisory(*, artifact: Path, expected_sha256: str, repository: Path,
                                 task_cid: str, prompt: str, workspace: Path | None = None,
                                 require_current_source: bool = True) -> tuple[str, dict]:
    """Verify an operational advisory or replay its immutable historical input.

    Historical mode verifies baseline Git bytes and cannot establish freshness
    for a new dispatch. The router always uses the default current-source mode.
    """
    root, artifact = Path(repository).absolute(), Path(artifact).absolute()
    if (root.resolve(strict=True) != root or not artifact.is_relative_to(root / ".runtime/doctor-residuals")
            or artifact.parent != root / ".runtime/doctor-residuals"):
        raise ValueError("residual artifact must belong to the exact canonical repository")
    descriptor = _directory(artifact.parent)
    try:
        raw, info = _read(descriptor, artifact.name)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_BYTES or _sha(raw) != expected_sha256 or stat.S_IMODE(info.st_mode) & 0o222:
        raise ValueError("residual context digest, size or read-only artifact differs")
    payload = json.loads(raw, object_pairs_hook=_unique)
    if (not isinstance(payload, dict) or set(payload) != FIELDS or payload["schema"] != SCHEMA
            or payload["repository"] != str(root) or payload["task_cid"] != task_cid
            or content_identity({key: value for key, value in payload.items() if key != "context_cid"}) != payload["context_cid"]
            or payload["provider_calls"] != 0
            or any(payload[key] is not False for key in
                   ("derived_runtime_admitted", "completion_authority", "publication_authority"))):
        raise ValueError("residual context identity or candidate-only authority differs")
    wire, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    if not isinstance(wire, dict) or wire.get("objective_id") != payload["task_id"]:
        raise ValueError("residual context belongs to another native task")
    sources = payload["source_manifest"]
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 256:
        raise ValueError("bounded residual source inventory required")
    baseline = payload["base_commit"]
    if _git(root, "rev-parse", baseline + "^{commit}").decode().strip() != baseline:
        raise ValueError("residual baseline commit is unavailable")
    if require_current_source:
        if (_git(root, "rev-parse", "HEAD").decode().strip() != baseline
                or local._sources(root, sorted(sources), max_files=256) != sources):
            raise ValueError("residual source is stale")
    else:
        for name, binding in sources.items():
            local._path(name)
            if _sha(_git(root, "show", baseline + ":" + name)) != binding["sha256"]:
                raise ValueError("historical residual baseline source differs")
    if workspace is not None:
        workspace = Path(workspace).absolute()
        if (workspace.resolve(strict=True) != workspace or workspace == root
                or _git(workspace, "rev-parse", "--path-format=absolute", "--git-common-dir")
                    != _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
                or _git(workspace, "rev-parse", "HEAD").decode().strip() != baseline):
            raise ValueError("residual context requires the matching allocated worktree")
        if require_current_source and local._sources(workspace, sorted(sources), max_files=256) != sources:
            raise ValueError("allocated residual source is stale")
    capsule = payload["capsule"]
    paths = [row["path"] for row in payload["outputs"]]
    if (capsule.get("task_id") != payload["task_id"] or set(capsule.get("allowed_paths", [])) != set(paths)
            or capsule.get("validation_commands") != payload["validation_commands"]
            or capsule.get("repairable_record_ids") or capsule.get("rejected_proposal_record_ids")
            or any(capsule.get(key) is not False for key in
                   ("completion_authority", "proof_authority", "write_authority", "semantic_authority"))):
        raise ValueError("native residual capsule differs from admitted candidate-only scope")
    residuals = [DoctorPlanResidual.from_dict(row) for row in payload["residuals"]]
    if (not residuals or capsule["open_obligation_ids"] != [row.residual_id for row in residuals]
            or any(row.parent_task_cid != task_cid or row.parent_goal_cid != payload["goal_cid"]
                   or row.plan_id != payload["plan_cid"]
                   or not set(row.context_paths + row.predicted_files) <= set(paths) for row in residuals)):
        raise ValueError("residual obligations differ from the existing admitted task")
    advisory_payload = {"schema": "doctor-residual-coding-advisory@1", "context_cid": payload["context_cid"],
        "task_cid": task_cid, "native_capsule_id": capsule["capsule_id"],
        "allowed_outputs": payload["outputs"], "validation_commands": payload["validation_commands"],
        "acceptance_ids": capsule["acceptance_ids"], "security_roots": capsule["security_roots"],
        "open_obligation_ids": capsule["open_obligation_ids"],
        "residuals": [{"residual_id": row.residual_id, "kind": row.kind.value,
            "reason_codes": list(row.reason_codes), "required_capability": row.required_capability,
            "context_paths": list(row.context_paths), "attempted_strategies": list(row.attempted_strategies),
            "evidence_refs": list(row.evidence_refs)} for row in residuals],
        "proposed_work": payload["proposal_counts"], "derived_runtime_admitted": False,
        "completion_authority": False, "proof_authority": False, "scope_expansion_authority": False}
    advisory = ("\n\nVerified Doctor residual context for this existing coding task:\n"
        "Use these remaining obligations to guide work within the original declared outputs and validation. "
        "Unavailable screened analysis, an unsupported Doctor operator or a missing prover does not establish a source defect. "
        "Successor tasks and capability work are proposals awaiting independent admission; do not execute "
        "them, broaden write scope, install tools, or claim proof or completion from this advisory. "
        "Continue the original coding task; this data does not request another planner call or change its response format.\n"
        + json.dumps(advisory_payload, sort_keys=True, separators=(",", ":")))
    if len(advisory.encode()) > 32_768:
        raise ValueError("Doctor residual advisory exceeds byte bound")
    receipt = {"schema": "doctor-residual-router-context@1", "artifact": str(artifact),
        "artifact_sha256": expected_sha256, "task_cid": task_cid, "task_id": payload["task_id"],
        "context_cid": payload["context_cid"], "native_capsule_id": capsule["capsule_id"],
        "advisory_sha256": _sha(advisory.encode()), "advisory_bytes": len(advisory.encode()),
        "source_freshness_verified": require_current_source, "historical_replay": not require_current_source,
        "candidate_only": True, "completion_authority": False, "publication_authority": False,
        "derived_runtime_admitted": False, "extra_provider_calls": 0}
    return advisory, receipt
