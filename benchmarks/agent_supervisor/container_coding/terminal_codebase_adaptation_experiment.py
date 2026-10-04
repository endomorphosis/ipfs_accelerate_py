"""Retained source-bound feature adaptation and finite supervisor previews.

This authored Terminal Bench-compatible prerequisite keeps measured fitting,
bounded runtime facts, mathematical operational-model theorems and native
capacity forecasts separate. It never launches a repair worker or claims an
official benchmark reward, source equivalence, or optimizer convergence proof.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .terminal_codebase_finite_experiment import INTENT, _capture, _git, _pin, _repo, _source, _write
from .terminal_codebase_finite_service_experiment import _authority_materials, _catalog, _open, _request, _scheduler
from .terminal_codebase_finite_index import digest, wire

SCHEMA = "terminal-codebase-adaptation-experiment@1"
_MODULE = "benchmarks.agent_supervisor.container_coding.terminal_codebase_adaptation_experiment"
_TRAINING_PATHS = ("calc.py", "known_variant.py", "tune.py", "canary.py")
_SOURCE_PATHS = (*_TRAINING_PATHS, "consumer.py")
_CONSUMER_SOURCE = b"from calc import increment\n\ndef consume(n: int) -> int:\n    return increment(n)\n"
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "behavior_authority", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority", "production_admitted",
    "worker_launched", "omission_authority", "signed_evidence_admitted", "generalization_verified",
    "convergence_proved", "training_convergence_proved", "public_terminal_bench_task_satisfied",
    "whole_program_semantics_verified", "decoder_384d_qualified")}
_CRITERIA = {"schema": "terminal-codebase-adaptation-prespecified-criteria@1",
    "configuration": {"epochs": 16, "learning_rate": .01, "seed": 1729},
    "selection": "Native fixed-tuning candidate selection; canary never selects candidates.",
    "acceptance": ["Root and child attempted exactly sixteen training epochs.",
        "Every observed tuning and epoch loss is finite; selected tuning objective does not increase.",
        "Child preserves the exact parent feature basis, model contract, optimizer configuration and immutable tuning/canary/replay targets.",
        "Finite clauses remain complete and planning selects only the residual reviewed operation.",
        "Current native operational models pass Lean, with requested goals proved/refuted only in the mathematical model.",
        "Held native capacity supports the nonempty forecast and is released before returning proposals.",
        "Cold current-head, model lineage, inference, retained artifacts and metadata reconstruction agree exactly."],
    "evaluation_scope": "Four selected source files in a five-file authored repository; source-cohort transductive reconstruction diagnostics.",
    "canary_scope": "Fixed, repeated post-selection diagnostic, not unseen evaluation.",
    "claims": dict(_FALSE)}


def _owner(index, repository, head, scheduler):
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    return RepositoryPlanPreviewOwner(index=index, repository=repository, expected_head=head,
        scheduler=scheduler, timeout_seconds=90, memory_mb=1024)


def _context(path):
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import FrozenCodebaseFeatureContext
    return FrozenCodebaseFeatureContext(Path(path), (Path(path) / "context.json").read_bytes())


def _idle(scheduler):
    state = scheduler.snapshot()
    if state["active_lease_count"] or state["waiting_request_count"]:
        raise ValueError("adaptation experiment leaked a native resource reservation")
    return {"active_lease_count": 0, "waiting_request_count": 0}


def _phase(phases, label, invoke):
    """Retain the observed cost of an actual call, including a refusal."""
    start, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    status, error = "completed", None
    try:
        return invoke()
    except BaseException as caught:
        status, error = "failed", repr(caught)
        raise
    finally:
        phases.append({"phase": label, "started_at": start, "status": status,
            "wall_seconds": time.monotonic() - clock, "error": error})


def _owned_file_pins(*directories):
    pins = []
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir() or directory.resolve() != directory:
            raise ValueError("retained native artifact directory changed")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("retained native artifact became a symlink")
            if path.is_file():
                pins.append(_pin(path))
    return pins


def _artifact_bytes(pins):
    rows = []
    for pin in pins:
        path = Path(pin["path"])
        if pin["bytes"] > 64 * 1024**2 or _pin(path) != pin:
            raise ValueError("retained native artifact identity or byte bound changed")
        raw = path.read_bytes()
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("captured native artifact bytes differ from retained identity")
        rows.append({"schema": "terminal-codebase-adaptation-retained-artifact@1", "artifact": pin,
            "encoding": "base64_exact_observed_bytes", "content_base64": base64.b64encode(raw).decode("ascii")})
        if _pin(path) != pin:
            raise ValueError("retained native artifact changed while capturing complete bytes")
    return rows


def _capture_with_consumer_kg(index, repository, head, scheduler):
    """Retain actual import/call edges and the unproved consumer disposition."""
    from ipfs_datasets_py.logic.software_verification.source_adapters import adapt_source_to_software_verification
    values = _capture(index, repository, head, scheduler)
    if (set(row["path"] for row in values["sources"]) != set(_SOURCE_PATHS)
            or len(values["ast"]) != len(_SOURCE_PATHS)
            or not {"imports", "calls"}.issubset({row["relation"] for row in values["kg"]})):
        raise ValueError("complete native five-file capture lacks actual consumer import/call KG edges")
    manifest = index.load(head.manifest_cid)
    entry = next(row for row in manifest.snapshot.entries if row.path == "consumer.py")
    source = index.artifacts.get_bytes(entry.source_cid)
    if source != _CONSUMER_SOURCE or (repository / "consumer.py").read_bytes() != source:
        raise ValueError("consumer metadata source differs from the captured authored fixture")
    native = adapt_source_to_software_verification(source.decode("utf-8"), path="consumer.py",
        language="python", revision="snapshot:" + head.snapshot_cid, preserve_type_annotations=True)
    values["compiled_logic"] = [{"schema": "terminal-codebase-consumer-native-projection@1",
        "head": head.to_dict(), "source_path": "consumer.py", "source_cid": entry.source_cid,
        "native_source_adapter": native.to_dict(), "training_selected": False,
        "solvers_executed": False, "formal_proof_requested": False, **_FALSE}]
    return values


def _registry_inventory(registry):
    """Observe native rows and immutable files; inference may change neither."""
    tables = {}
    with registry._transaction() as cx:
        names = cx.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='autoencoder_control' ORDER BY table_name").fetchall()
        for (name,) in names:
            rows = cx.execute('SELECT * FROM autoencoder_control."' + name.replace('"', '""') + '"').fetchall()
            values = sorted([list(row) for row in rows], key=repr)
            tables[name] = {"row_count": len(values), "rows_sha256": digest(values)}
    artifacts = [_pin(path) for path in sorted(registry.artifact_root.rglob("*")) if path.is_file()]
    return {"schema": "terminal-codebase-native-registry-inventory@1", "owner_generation": registry.owner_generation,
        "tables": tables, "artifacts": artifacts}


def _preview(index, repository, head, scheduler, registry, context, policy, output):
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_capacity_preview import preview_capacity_bound_finite_integer_plan
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    state = output.parent
    document, catalog = build_finite_integer_intent(INTENT), _catalog()
    authority_path, prompt_path = state / "authority-materials.json", state / "prompt.txt"
    authority = json.loads(authority_path.read_bytes())
    if prompt_path.read_bytes() != INTENT.encode():
        raise ValueError("retained complete finite prompt changed")
    request = _request(index, repository, head, document, catalog, authority)
    registry_before = _registry_inventory(registry)
    def observe_policy(bound):
        if bound != request or prompt_path.read_bytes() != INTENT.encode():
            raise ValueError("complete native request or retained prompt changed")
        live = _request(index, repository, head, document, catalog, json.loads(authority_path.read_bytes()))
        live.roots.require_current(request.roots)
        return live.roots
    result = preview_capacity_bound_finite_integer_plan(owner=_owner(index, repository, head, scheduler),
        request=request, intent_document=document, source_text=INTENT, operation_catalog=catalog,
        output=output, tool_policy=policy, policy_observer=observe_policy,
        feature_context=context, model_registry=registry)
    registry_after = _registry_inventory(registry)
    _write(state / (output.name + "-registry-inventory.json"), {"schema": "terminal-codebase-preview-no-fit-observation@1",
        "before": registry_before, "after": registry_after, "unchanged": registry_before == registry_after,
        "fitting_performed": False, "promotion_performed": False})
    if registry_before != registry_after:
        raise ValueError("preview changed native model registry rows or artifact inventory")
    return result, request


def _check_preview(result, *, facts, selected, feature):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import PlanCreateInputSnapshot, PlanCreatePreviewReceipt
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import ObligationGraph
    from ipfs_accelerate_py.agent_supervisor.planning.plan_critic import PlanCritic, PlanCritique
    receipt = PlanCreatePreviewReceipt.from_dict(result["preview"])
    snapshot = PlanCreateInputSnapshot.from_dict(result["input_snapshot"])
    graph = ObligationGraph.from_dict(result["obligation_graph"])
    critique = PlanCritique.from_dict(result["critique"])
    candidate = result["candidate_plan"]
    replay = PlanCritic().critique(candidate, obligation_graph=graph, evidence=result["critic_evidence"],
        required_goal_ids=graph.root_obligation_ids, expected_effects=candidate["expected_effect_ids"])
    stages = {row.stage.value: row for row in receipt.stage_results}
    if (result["result_cid"] != cid_for_structured({k: v for k, v in result.items() if k != "result_cid"})
            or len(graph.root_obligation_ids) != 2 or len(graph.facts) != facts
            or result["current_facts_count"] != facts or result["selected_task_ids"] != selected
            or [row["task_id"] for row in candidate["tasks"]] != selected
            or receipt.input_snapshot_cid != snapshot.snapshot_cid or receipt.admitted
            or receipt.verdict.value != "review_only" or not receipt.read_only or receipt.wrote_effects
            or not critique.accepted or critique.truncated or critique != replay
            or result["feature_context"] != feature.material_binding
            or result["planning_model_calls"] != 0 or result["training_steps_during_preview"] != 0
            or any(result.get(key) is not False for key in (
                "source_semantics_verified", "proof_authority", "execution_authority",
                "production_admitted", "completion_authority", "worker_launched", "convergence_proved"))
            or any(not stages[name].passed for name in ("scan", "query", "evidence", "obligation", "candidate", "critique", "parallel_plan"))):
        raise ValueError("native capacity preview/critic/complete roots/feature authority differs")
    model = result["operational_model"]
    if (not model["kernel_checked_model"] or not model["source_identity_proved"]
            or model["requested_model_theorem_proved"] is bool(selected)
            or model["status"] != ("model_refuted" if selected else "model_proved")
            or model["scope"] != "mathematical_integer_operational_model"
            or model["source_semantics_verified"] or model["proof_authority"]):
        raise ValueError("operational model theorem incorrectly joined to finite runtime authority")
    if selected:
        if result["execution_plan"]["admitted"] is not True or result["capacity_observation"] is None:
            raise ValueError("real held capacity did not admit the nonempty forecast")
    elif (result["execution_plan"]["admitted"] is not False
            or result["execution_plan"]["status"] != "no_execution_requested" or candidate["effects"]):
        raise ValueError("finite-complete plan invented executable work")
    if not result["reservation_released_on_return"]:
        raise ValueError("capacity preview did not release its native reservation")


def _measure(context):
    retained = context.retained_artifacts
    report, state = retained["checkpoint"]["report"], retained["checkpoint"]["state"]
    before, after = report["before"]["objective"], report["after"]["objective"]
    losses = [before, after]
    for row in report["epochs"]:
        losses.extend(row[key] for key in ("train_objective", "tuning_objective", "gradient_norm") if key in row)
    if (report["attempted_epochs"] != _CRITERIA["configuration"]["epochs"]
            or any(type(value) not in {float, int} or not math.isfinite(value) for value in losses)
            or after > before or any(report["configuration"][key] != wanted
                for key, wanted in _CRITERIA["configuration"].items())):
        raise ValueError("prewritten finite/nonincrease tuning criterion failed")
    return {"schema": "terminal-codebase-adaptation-measurement@1", "context_cid": context.cid,
        "version_id": context.version_id, "parent_version_id": context.material_binding["parent_version_id"],
        "attempted_epochs": report["attempted_epochs"], "selected_total_epochs": state["completed_epochs"],
        "before_tuning_objective": before, "after_tuning_objective": after, "tuning_nonincrease": True,
        "all_recorded_losses_finite": True, "optimizer_steps": [row["step"] for row in state["adam"]],
        "actual_configuration": report["configuration"], "metrics_scope": _CRITERIA["evaluation_scope"],
        "canary_monitoring": report["codebase_provenance"]["canary_monitoring"], **_FALSE}


def _continuation(parent, child):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_projection_features import digest as native_digest
    left, right = parent.retained_artifacts["checkpoint"], child.retained_artifacts["checkpoint"]
    a, b = left["report"]["codebase_provenance"], right["report"]["codebase_provenance"]
    if (left["contract"] != right["contract"] or left["feature_space"] != right["feature_space"]
            or left["state"]["optimizer_config"] != right["state"]["optimizer_config"]
            or right["report"]["base_state_sha256"] != native_digest(left["state"])
            or child.material_binding["parent_version_id"] != parent.version_id
            or b["continuation"] != "exact_frozen_basis_adam_resume"
            or any(a[key] != b[key] for key in ("tuning_targets", "canary_targets", "replay_targets"))
            or child.retained_artifacts["lineage"][1]["checkpoint"] != left):
        raise ValueError("native child failed exact basis/optimizer/fixed evaluation continuation")
    return {"schema": "terminal-codebase-adaptation-continuation-check@1", "parent_version_id": parent.version_id,
        "child_version_id": child.version_id, "feature_basis_preserved": True, "model_contract_preserved": True,
        "adam_parent_state_bound": True, "optimizer_configuration_preserved": True,
        "fixed_tuning_canary_replay_preserved": True, "complete_ancestry_retained": True, **_FALSE}


def _cold_validate(request_path):
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import verify_current_context
    request = json.loads(Path(request_path).read_bytes())
    phases = []
    output, repository = Path(request["output"]), Path(request["repository"])
    head = CodebaseHead.from_dict(request["head"])
    scheduler = _scheduler(output / "cold-resource-admission.json")
    cx, index = _open(output)
    registry = AutoencoderRegistry(output / "train.duckdb", output / "model-artifacts")
    try:
        for row in request["retained_pins"]:
            if _pin(row["path"]) != row:
                raise ValueError("retained adaptation artifact changed before cold replay")
        frozen = _context(output / "child-frozen-context")
        registry_before = _registry_inventory(registry)
        verified = _phase(phases, "cold_current_frozen_context", lambda:
            verify_current_context(_owner(index, repository, head, scheduler), registry, frozen))
        if verified != request["child_frozen_binding"]:
            raise ValueError("cold feature context differs from exact retained binding")
        before = frozen.retained_artifacts
        fresh, current_request = _phase(phases, "cold_frozen_preview", lambda:
            _preview(index, repository, head, scheduler, registry, frozen,
                request["tool_policy"], output / "cold-frozen-preview"))
        _check_preview(fresh, facts=2, selected=[], feature=frozen)
        if fresh["match"]["observation_cid"] == request["previous_observation_cid"]:
            raise ValueError("cold preview reused an observation instead of invoking native tools")
        if current_request.to_dict() != request["expected_request"] or frozen.retained_artifacts != before:
            raise ValueError("cold reconstruction changed request or fitted during verification")
        registry_after = _registry_inventory(registry)
        if registry_before != registry_after:
            raise ValueError("cold inference changed native registry rows or artifacts")
        _write(output / "cold-preview.json", fresh)
        _write(output / "cold-context-retained.json", before)
        captured = _phase(phases, "cold_native_metadata_capture", lambda:
            _capture_with_consumer_kg(index, repository, head, scheduler))
        _write(output / "cold-capture.json", captured)
        if any(_pin(row["path"]) != row for row in request["retained_pins"]):
            raise ValueError("cold validation changed retained artifacts")
        return {"schema": "terminal-codebase-adaptation-cold-validation@1", "status": "completed",
            "head": head.to_dict(), "model_version_id": frozen.version_id,
            "verified_context": verified, "fresh_preview": _pin(output / "cold-preview.json"),
            "fresh_request": current_request.to_dict(), "fresh_current_facts_count": 2,
            "fresh_selected_task_ids": [], "fresh_observation_executed": True,
            "fitting_during_cold_validation": False, "retained_model_unchanged": True,
            "native_registry_reopened": True, "native_source_owner_reopened": True,
            "registry_before": registry_before, "registry_after": registry_after,
            "phase_timings": phases,
            **_idle(scheduler), **_FALSE}
    except BaseException as error:
        _write(output / "cold-failure.json", {"schema": "terminal-codebase-adaptation-cold-failure@1",
            "error": repr(error), "phase_timings": phases,
            **{key: scheduler.snapshot()[key] for key in ("active_lease_count", "waiting_request_count")}})
        raise
    finally:
        registry.close()
        cx.close()


def run_adaptation_experiment(*, output: Path, python_executable: Path, lean_executable: Path):
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    from ipfs_datasets_py.logic.software_contracts.codebase_source_training import CodebaseTrainingSelection
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import prepare_codebase_feature_context, verify_current_context
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata
    from .terminal_codebase_supervisor_fixture import bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.exists() or not output.parent.is_dir():
        raise ValueError("fresh canonical adaptation output directory required")
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    modules = (_MODULE,
        "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_service_experiment",
        "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_experiment",
        "benchmarks.agent_supervisor.container_coding.codebase_ir_metadata",
        "benchmarks.agent_supervisor.container_coding.terminal_codebase_supervisor_fixture",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_capacity_preview",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_capacity",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_source_custody",
        "ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_preview",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_service",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase",
        "ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service",
        "ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler",
        "ipfs_accelerate_py.agent_supervisor.planning.plan_critic",
        "ipfs_accelerate_py.agent_supervisor.planning.parallel_plan_compiler",
        "ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context",
        "ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview",
        "ipfs_datasets_py.logic.software_contracts.codebase_integer_model_lean",
        "ipfs_datasets_py.logic.software_contracts.codebase_integer_profile",
        "ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation",
        "ipfs_datasets_py.logic.software_contracts.codebase_source_training",
        "ipfs_datasets_py.logic.software_contracts.codebase_ir_targets",
        "ipfs_datasets_py.logic.software_contracts.codebase_ir",
        "ipfs_datasets_py.logic.software_verification.source_adapters",
        "ipfs_datasets_py.logic.backends.process",
        "ipfs_datasets_py.duckdb_control.autoencoder_registry",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_projection_features",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_runtime_registry",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_feature_worker",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.modal_autoencoder_cuda")
    sources = [_pin(importlib.import_module(name).__file__) for name in modules]
    output.mkdir(mode=0o700)
    cx = registry = scheduler = None
    phases = []
    try:
        _write(output / "criteria.json", _CRITERIA)  # Must precede any fit or dataset observation.
        criteria_pin = _pin(output / "criteria.json")
        _write(output / "execution-source-pins-before.json", sources)
        repository = output / "authored-repository"
        _repo(repository, {"calc.py": _source(1), "known_variant.py": b"# vocabulary fixture: offset two\n" + _source(2),
            "tune.py": b"# fixed tuning fixture\n" + _source(1),
            "canary.py": b"# fixed diagnostic fixture\n" + _source(1), "consumer.py": _CONSUMER_SOURCE})
        git_head = _git(repository, "rev-parse", "HEAD")
        source_pins_before = [_pin(repository / name) for name in _SOURCE_PATHS]
        _write(output / "repository-source-pins-before.json", source_pins_before)
        with (output / "prompt.txt").open("xb") as stream: stream.write(INTENT.encode())
        _write(output / "intent-ir.json", build_finite_integer_intent(INTENT).to_dict())
        _write(output / "operation-catalog.json", _catalog().to_dict())
        _write(output / "authority-materials.json", _authority_materials())
        policy = seal_finite_integer_tools(python_executable=python_executable, lean_executable=lean_executable)
        _write(output / "tool-policy.json", policy)
        control_pins = [criteria_pin, *[_pin(output / name) for name in ("execution-source-pins-before.json",
            "repository-source-pins-before.json", "prompt.txt", "intent-ir.json", "operation-catalog.json", "authority-materials.json", "tool-policy.json")]]
        scheduler = _scheduler(output / "resource-admission.json")
        cx, index = _open(output)
        registry = AutoencoderRegistry(output / "train.duckdb", output / "model-artifacts")
        first = _phase(phases, "initial_native_source_prepare", lambda:
            index.prepare_current(repository, repository_id="repository:authored-codebase-adaptation",
                operation_id="adaptation-initial", expected_head=None, scheduler=scheduler)).head
        owner = _owner(index, repository, first, scheduler)
        selections = tuple(CodebaseTrainingSelection(name, role, contracts=()) for name, role in (
            ("calc.py", "train"), ("known_variant.py", "train"), ("tune.py", "tune"), ("canary.py", "canary")))
        if tuple(row.path for row in selections) != _TRAINING_PATHS or any(row.path == "consumer.py" for row in selections):
            raise ValueError("metadata consumer escaped the original four training selections")
        _write(output / "training-selections.json", [item.to_dict() for item in selections])
        control_pins.append(_pin(output / "training-selections.json"))
        controls, previews, requests, contexts, measurements, records = [], [], [], [], [], {}
        def extend(values):
            for family, rows in values.items():
                target = records.setdefault(family, [])
                ordinal = len(target)
                target.extend({"schema": "terminal-codebase-adaptation-metadata-occurrence@1",
                    "occurrence": ordinal + position, "record": row} for position, row in enumerate(rows))
        extend(_phase(phases, "root_native_metadata_capture", lambda:
            _capture_with_consumer_kg(index, repository, first, scheduler)))
        off = _phase(phases, "root_model_off_prepare", lambda:
            prepare_codebase_feature_context(owner=owner, registry=registry, mode="model_off", output=output / "root-off-context"))
        root = _phase(phases, "root_training_prepare", lambda:
            prepare_codebase_feature_context(owner=owner, registry=registry, mode="train", output=output / "root-training-context",
                selections=selections, operation_id="adaptation-root", **_CRITERIA["configuration"]))
        measurements.append(_measure(root))
        root_inventory = _registry_inventory(registry)
        frozen = _phase(phases, "root_frozen_prepare", lambda:
            prepare_codebase_feature_context(owner=owner, registry=registry, mode="frozen", output=output / "root-frozen-context", version_id=root.version_id))
        if _registry_inventory(registry) != root_inventory:
            raise ValueError("frozen root preparation fitted or promoted a model")
        contexts.extend([off, root, frozen])
        for label, context in (("root-off", off), ("root-frozen", frozen)):
            result, request = _phase(phases, label + "_preview", lambda:
                _preview(index, repository, first, scheduler, registry, context, policy, output / (label + "-preview-artifacts")))
            _check_preview(result, facts=1, selected=["task:finite:offset"], feature=context)
            _write(output / (label + "-preview.json"), result)
            _write(output / (label + "-request.json"), request.to_dict())
            previews.append(result)
            requests.append(request.to_dict())
            _idle(scheduler)
        root_retained_before = root.retained_artifacts
        (repository / "calc.py").write_bytes(_source(2))  # Explicit private fixture edit; no supervisor execution.
        if _git(repository, "rev-parse", "HEAD") != git_head:
            raise ValueError("private source change moved Git HEAD")
        for label, action in (
                ("old_owner_after_same_head_edit", lambda: index.observe_current(repository, expected_head=first, scheduler=scheduler)),
                ("old_context_after_same_head_edit", lambda: verify_current_context(owner, registry, frozen))):
            try: action()
            except StaleCodebaseError as error:
                controls.append({"control": label, "rejected": True, "error_type": type(error).__name__, "error": str(error)})
            else: raise ValueError("stale authored adaptation control accepted: " + label)
        second = _phase(phases, "successor_native_source_prepare", lambda:
            index.prepare_current(repository, repository_id=first.repository_id,
                operation_id="adaptation-successor", expected_head=first, scheduler=scheduler)).head
        owner = replace(owner, expected_head=second)
        try: verify_current_context(owner, registry, frozen)
        except ValueError as error:
            controls.append({"control": "old_context_under_successor_owner", "rejected": True,
                "error_type": type(error).__name__, "error": str(error)})
        else: raise ValueError("old model context accepted against successor head")
        extend(_phase(phases, "child_native_metadata_capture", lambda:
            _capture_with_consumer_kg(index, repository, second, scheduler)))
        child_off = _phase(phases, "child_model_off_prepare", lambda:
            prepare_codebase_feature_context(owner=owner, registry=registry, mode="model_off", output=output / "child-off-context"))
        child = _phase(phases, "child_training_prepare", lambda:
            prepare_codebase_feature_context(owner=owner, registry=registry, mode="train", output=output / "child-training-context",
                selections=selections, operation_id="adaptation-child", parent_version_id=root.version_id, **_CRITERIA["configuration"]))
        measurements.append(_measure(child))
        continuation = _continuation(root, child)
        child_inventory = _registry_inventory(registry)
        child_frozen = _phase(phases, "child_frozen_prepare", lambda:
            prepare_codebase_feature_context(owner=owner, registry=registry, mode="frozen", output=output / "child-frozen-context", version_id=child.version_id))
        contexts.extend([child_off, child, child_frozen])
        if root.retained_artifacts != root_retained_before:
            raise ValueError("child continuation changed its retained parent")
        child_before_verify = child_frozen.retained_artifacts
        _phase(phases, "child_frozen_verification", lambda: verify_current_context(owner, registry, child_frozen))
        if child_frozen.retained_artifacts != child_before_verify:
            raise ValueError("frozen child verification changed selected numerical artifacts")
        if _registry_inventory(registry) != child_inventory:
            raise ValueError("frozen child preparation/verification changed native model registry")
        for label, context in (("child-off", child_off), ("child-frozen", child_frozen)):
            result, request = _phase(phases, label + "_preview", lambda:
                _preview(index, repository, second, scheduler, registry, context, policy, output / (label + "-preview-artifacts")))
            _check_preview(result, facts=2, selected=[], feature=context)
            _write(output / (label + "-preview.json"), result)
            _write(output / (label + "-request.json"), request.to_dict())
            previews.append(result)
            requests.append(request.to_dict())
            _idle(scheduler)
        if any(row["declared_task_requirement_ids"] != previews[0]["declared_task_requirement_ids"] for row in previews):
            raise ValueError("adaptation changed complete reviewed task meanings")
        source_pins_after = [_pin(repository / name) for name in _SOURCE_PATHS]
        if source_pins_before[1:] != source_pins_after[1:] or Path(source_pins_after[0]["path"]).read_bytes() != _source(2):
            raise ValueError("authored change escaped calc.py or fixed evaluation files changed")
        numerical_pins = [_pin(path) for context in contexts for path in sorted(context.output.iterdir())]
        numerical_pins.extend(_pin(registry.artifact_path(registry.get_version(version)["artifact"]))
            for version in (root.version_id, child.version_id))
        native_artifact_pins = _owned_file_pins(output / "cas", *[output / (label + "-preview-artifacts")
            for label in ("root-off", "root-frozen", "child-off", "child-frozen")])
        registry.close()
        registry = None
        cx.close()
        cx = None
        retained_pins = control_pins + source_pins_after + numerical_pins + native_artifact_pins + [_pin(output / (label + "-preview.json"))
            for label in ("root-off", "root-frozen", "child-off", "child-frozen")]
        restart = {"schema": "terminal-codebase-adaptation-cold-request@1", "output": str(output),
            "repository": str(repository), "head": second.to_dict(), "tool_policy": policy,
            "expected_request": requests[-1], "previous_observation_cid": previews[-1]["match"]["observation_cid"],
            "child_frozen_binding": child_frozen.material_binding, "retained_pins": retained_pins}
        _write(output / "cold-request.json", restart)
        command = [str(python_executable), "-m", _MODULE, "--cold-request", str(output / "cold-request.json")]
        child_process = _phase(phases, "cold_owner_registry_and_preview_process", lambda:
            subprocess.run(command, capture_output=True, text=True, timeout=180))
        _write(output / "cold-process.json", {"command": command, "returncode": child_process.returncode,
            "stdout": child_process.stdout, "stderr": child_process.stderr, "request": _pin(output / "cold-request.json")})
        if child_process.returncode != 0:
            raise ValueError("native cold adaptation validation failed: " + child_process.stderr)
        cold = json.loads((output / "cold-response.json").read_bytes())
        if cold.get("status") != "completed" or cold.get("head") != second.to_dict():
            raise ValueError("native cold response artifact differs from expected current owner")
        _write(output / "cold-validation.json", cold)
        previews.append(json.loads((output / "cold-preview.json").read_bytes()))
        requests.append(cold["fresh_request"])
        extend(json.loads((output / "cold-capture.json").read_bytes()))
        complete_native_artifact_pins = _owned_file_pins(output / "cas", output / "cold-frozen-preview",
            *[output / (label + "-preview-artifacts") for label in ("root-off", "root-frozen", "child-off", "child-frozen")])
        extend({"artifacts": _artifact_bytes(complete_native_artifact_pins + numerical_pins)})
        matches = [value["match"] for value in previews]
        numerical = [{"schema": "terminal-codebase-adaptation-numerical-context@1", "binding": context.material_binding,
            "retained": context.retained_artifacts} for context in contexts]
        extend({"intent_ir": [build_finite_integer_intent(INTENT).to_dict()],
            "contracts": [match["observation"]["contract"] for match in matches],
            "observations": [match["observation"] for match in matches],
            "current_facts": [fact for match in matches for fact in match["current_facts"]],
            "runtime_traces": [match["observation"]["trace"] for match in matches],
            "lean_certificates": [match["observation"]["lean_certificate"] for match in matches],
            "compiled_logic": [json.loads(Path(match["observation"]["artifacts"]["compiled"]["path"]).read_bytes()) for match in matches],
            "finite_matches": matches, "service_results": [*previews, cold],
            "service_input_snapshots": [value["input_snapshot"] for value in previews], "request_roots": requests,
            "obligation_graphs": [value["obligation_graph"] for value in previews],
            "candidate_portfolios": [value["portfolio"] for value in previews],
            "candidate_plans": [value["candidate_plan"] for value in previews],
            "plan_critiques": [value["critique"] for value in previews],
            "critic_evidence": [value["critic_evidence"] for value in previews],
            "execution_plans": [{"execution_plan": value["execution_plan"],
                "capacity_observation": value["capacity_observation"],
                "capacity_compilation_request": value["capacity_compilation_request"],
                "capacity_binding": value["capacity_binding"]} for value in previews],
            "operation_catalogs": [_catalog().to_dict()], "stale_controls": controls,
            "authority_materials": [{"criteria": _CRITERIA, "authority": _authority_materials(), "tool_policy": policy}],
            "source_model_proofs": [value["operational_model"] for value in previews],
            "feature_contexts": numerical,
            "training": [*numerical, *measurements, continuation, root_inventory, child_inventory,
                *[json.loads(path.read_bytes()) for path in sorted(output.glob("*-registry-inventory.json"))]],
            "vectors": [{"context_cid": item["binding"]["context_cid"], "version_id": item["binding"]["version_id"],
                "head": item["binding"]["head"], "inference": item["retained"]["inference"]}
                for item in numerical if "inference" in item["retained"]]})
        if len(records) > 31:
            raise ValueError("complete adaptation metadata exceeds 31 producer families")
        bounded = bound_terminal_codebase_metadata_records(records)
        if wire(reconstruct_terminal_codebase_metadata_records(bounded)) != wire(records):
            raise ValueError("exact complete metadata packaging changed a producer record")
        metadata = _phase(phases, "native_duckdb_ducklake_metadata_hydration", lambda:
            hydrate_codebase_ir_metadata(records=bounded, output=output / "metadata",
            source_snapshot={"schema": "terminal-codebase-adaptation-source-binding@1", "heads": [first.to_dict(), second.to_dict()],
                "source_pins_before": source_pins_before, "source_pins_after": source_pins_after,
                "criteria": criteria_pin, "complete_producer_sha256": digest(records), "same_git_head": git_head,
                "profile": "authored_source_bound_feature_adaptation_only", "fitting_configuration": _CRITERIA["configuration"]}))
        replay = _phase(phases, "cold_duckdb_ducklake_metadata_validation", lambda:
            validate_codebase_ir_metadata(output=output / "metadata", expected=metadata, fresh_process=True))
        restored = {family: [json.loads(line)["payload"] for line in
            (output / "metadata" / descriptor["relative_path"]).read_bytes().splitlines()]
            for family, descriptor in replay["exports"].items()}
        if wire(reconstruct_terminal_codebase_metadata_records(restored)) != wire(records):
            raise ValueError("native cold DuckDB/DuckLake metadata reconstruction differs")
        if any(_pin(row["path"]) != row for row in sources + control_pins + source_pins_after
                + numerical_pins + native_artifact_pins + complete_native_artifact_pins):
            raise ValueError("implementation, prewritten criteria, prompt, policy or source changed during experiment")
        _idle(scheduler)
        result = {"schema": SCHEMA, "status": "completed", "output": str(output), "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - clock,
            "fixture_scope": "authored Terminal Bench-compatible five-file repository with four selected training files",
            "training_selected_paths": list(_TRAINING_PATHS), "metadata_only_paths": ["consumer.py"],
            "native_kg_nonempty": bool(records["kg"]),
            "consumer_formal_semantics_verified": False,
            "same_git_head": git_head, "heads": [first.to_dict(), second.to_dict()],
            "criteria": _CRITERIA, "criteria_artifact": criteria_pin, "measurements": measurements,
            "continuation": continuation, "source_pins_before": source_pins_before, "source_pins_after": source_pins_after,
            "numerical_artifacts": numerical_pins,
            "native_proof_and_cas_artifacts": complete_native_artifact_pins,
            "preview_artifacts": [_pin(output / name) for name in ("root-off-preview.json", "root-frozen-preview.json", "child-off-preview.json", "child-frozen-preview.json", "cold-preview.json")],
            "training_contexts": [root.material_binding, child.material_binding], "frozen_context": child_frozen.material_binding,
            "stale_controls": controls, "cold_validation": cold, "metadata": metadata, "metadata_replay": replay,
            "complete_producer_sha256": digest(records), "complete_family_counts": {name: len(rows) for name, rows in records.items()},
            "zero_truncation": True, "execution_sources": sources, "provider_calls": 0,
            "planning_model_calls": 0, "training_steps_during_preview": 0,
            "actual_attempted_training_epochs": sum(row["attempted_epochs"] for row in measurements),
            "budget_scope": "Each preparation/preview has its own 90-second deadline; the full experiment and independent metadata restart are separate phases.",
            "latent_width": 8, "feature_column_count": len(root.retained_artifacts["checkpoint"]["feature_space"]["columns"]),
            "parameter_dtype": "float64", "registered_versions": [root.version_id, child.version_id],
            "phase_timings": phases,
            "frozen_verification_fitting": False, "official_reward": None, **_idle(scheduler), **_FALSE}
        _write(output / "metadata-result.json", metadata)
        _write(output / "metadata-replay.json", replay)
        _write(output / "execution-source-pins-after.json", [_pin(row["path"]) for row in sources])
        _write(output / "repository-source-pins-after.json", source_pins_after)
        _write(output / "phase-timings.json", phases)
        _write(output / "result.json", result)
        return result
    except BaseException as error:
        if registry is not None: registry.close()
        if cx is not None: cx.close()
        resource_counts = {}
        if scheduler is not None:
            state = scheduler.snapshot()
            resource_counts = {key: state[key] for key in ("active_lease_count", "waiting_request_count")}
        _write(output / "failure.json", {"schema": SCHEMA, "status": "failed", "error": repr(error),
            "started_at": started, "wall_seconds": time.monotonic() - clock, "phase_timings": phases, **resource_counts})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable).resolve())
    parser.add_argument("--lean", type=Path)
    parser.add_argument("--cold-request", type=Path)
    args = parser.parse_args()
    if args.cold_request:
        response = _cold_validate(args.cold_request)
        response_path = Path(json.loads(args.cold_request.read_bytes())["output"]) / "cold-response.json"
        _write(response_path, response)
        print(json.dumps({"status": response["status"], "response": _pin(response_path)}, sort_keys=True))
    else:
        if args.output is None or args.lean is None: parser.error("--output and --lean required")
        result = run_adaptation_experiment(output=args.output, python_executable=args.python, lean_executable=args.lean)
        print(json.dumps({key: result[key] for key in ("status", "output", "wall_seconds", "complete_family_counts", "actual_attempted_training_epochs")}, sort_keys=True))
