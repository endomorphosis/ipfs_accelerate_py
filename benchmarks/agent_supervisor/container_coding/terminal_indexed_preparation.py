"""Prepare the original public Bottle task for an isolated native supervisor.

``prepare`` makes no model call. It signs independently declared edits, creation,
and a public structural smoke check before any proposal. Explicit v2 operations
use the symbolic planner; provider proposals need a separately qualified worker
router callable and preserve failed usage. The official
Terminal-Bench verifier remains outside this preparation and its worker context.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

from ipfs_accelerate_py.agent_supervisor.control.profile_authority import load_local_profile
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
    RepositoryAllowlist, repository_root_cid, scan_prompt_directory_detailed,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PromptGoalPlannerConfig, _select_evidence, build_prompt_goal_provider_request,
    generate_prompt_goal_graph,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy, DirectoryScanReceipt, LocalFallbackPolicy, PromptOutputPolicy,
    PromptPlanningPolicy, PromptSource, PromptWorkflowBudget, PromptWorkflowRequest,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


MODEL = "gpt-5.6-sol"
PROVIDER = "codex_cli"
REASONING = "high"
INSTRUCTION = ".supervisor-instruction.md"
SMOKE = ".supervisor-public-smoke.py"
ARGV = ["python3", "-B", SMOKE]
PUBLIC_SMOKE = '''"""Public syntax/report-shape check; no benchmark correctness authority."""
import json
from pathlib import Path
import re
source = Path("bottle.py")
assert source.is_file() and not source.is_symlink()
compile(source.read_bytes(), "bottle.py", "exec")
report = Path("report.jsonl")
assert report.is_file() and not report.is_symlink(), "declared report is missing"
rows = [json.loads(line) for line in report.read_text().splitlines() if line.strip()]
assert rows, "report must contain at least one finding"
for row in rows:
    assert set(row) == {"file_path", "cwe_id"}
    assert row["file_path"] in {"bottle.py", "/app/bottle.py"}
    assert isinstance(row["cwe_id"], list) and row["cwe_id"]
    assert all(isinstance(cwe, str) and re.fullmatch(r"cwe-[0-9]+", cwe, re.IGNORECASE) for cwe in row["cwe_id"])
'''


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _git(repository, *args):
    return subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True).stdout


def _commit(repository, message):
    _git(repository, "-c", "user.name=Isolated benchmark", "-c",
         "user.email=benchmark@example.invalid", "commit", "-qm", message)


def _config(repository, *, timeout_seconds=90):
    return PromptGoalPlannerConfig(
        repo_root=repository, provider=PROVIDER, model=MODEL, timeout_seconds=timeout_seconds,
        max_new_tokens=4096, allow_local_fallback=False,
        max_summary_bytes=8192,
        max_provider_request_bytes=262144, allowed_validation_prefixes=(tuple(ARGV),),
    )


def _constraints(spec, text):
    return {
        "allowed_paths": spec["scope_paths"], "validation_commands": [ARGV],
        "constraint_summaries": [
            "Public instruction.md (the complete authorized task):\n" + text,
            "Produce exactly root TB-GOAL and child TB-SUBGOAL, and one task TB-CODE-TASK owned by TB-SUBGOAL. Dependencies, risks, assumptions, unresolved_questions and uncertainty_debt must be empty.",
            "All goal/task acceptance arrays must exactly equal: " + json.dumps(spec["acceptance"], sort_keys=True),
            "Task scope, outputs and validation must exactly match the independently signed declaration (omit policy_cid from proposal validation): " + json.dumps(spec, sort_keys=True),
            "Use source evidence only as descriptive context. No external domain, proof or completion authority. resource_class cpu-medium; fallback_behavior fail_closed. Fulfill the public instruction; the structural check alone is not benchmark success.",
        ],
    }


def _planning_strategy(contract):
    if contract is None:
        return "direct"
    if contract.get("schema") in {"intent-plan-requirement-contract@2", "intent-plan-requirement-contract@3"}:
        return "intent_symbolic"
    return "intent_coverage"


def _load_prepared(state):
    prepared = json.loads((state / "prepared.json").read_text())
    manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
    repository = Path(prepared["repository"])
    inputs = manifest["planning_inputs"]
    requirement_contract = (local.decode_intent_requirement_contract(manifest)
                            if manifest["schema"] == local.INTENT_MANIFEST_SCHEMA else None)
    if (prepared["request"] != inputs["request"] or prepared["scan"] != inputs["scan"]
            or len(manifest["tasks"]) != 1 or prepared["spec"] != manifest["tasks"][0]
            or prepared["query"] != (repository / INSTRUCTION).read_text()
            or prepared["constraints"] != _constraints(prepared["spec"], prepared["query"])
            or prepared["worker_inputs"] != [INSTRUCTION, SMOKE, "bottle.py"]
            or prepared.get("intent_requirement_contract") != requirement_contract
            or prepared.get("planning_strategy", _planning_strategy(requirement_contract)) != _planning_strategy(requirement_contract)
            or str(repository) != manifest["repository"]):
        raise ValueError("preparation differs from exact signed public input and task declarations")
    return prepared


def _load_intent_requirement_contract(path: Path | None, text: str) -> dict | None:
    if path is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
        MAX_INTENT_PLAN_BYTES, validate_intent_requirement_contract,
    )

    path = Path(path).absolute()
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise ValueError("intent requirement contract must be a regular canonical file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INTENT_PLAN_BYTES + 1)
    if len(raw) > MAX_INTENT_PLAN_BYTES:
        raise ValueError("intent requirement contract exceeds its byte bound")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate intent requirement contract key")
            value[key] = item
        return value
    contract = validate_intent_requirement_contract(
        json.loads(raw, object_pairs_hook=unique), source_text=text,
    )
    if contract["source_path"] != INSTRUCTION:
        raise ValueError("benchmark intent requirements must bind the original public instruction")
    return contract


def prepare(*, repository: Path, instruction: Path, state: Path,
            intent_checkpoint_descriptor: Path | None = None,
            intent_action_384_config: Path | None = None,
            intent_projection_request: Path | None = None,
            intent_projection_request_sha256: str | None = None,
            disable_intent_autoencoder: bool = False,
            enable_source_unit_autoencoder: bool = False,
            source_unit_security_decoder_descriptor: Path | None = None,
            source_unit_project_logic_families: bool = False,
            source_unit_intent_family_context: Path | None = None,
            source_unit_intent_logic_families: list[str] | None = None,
            intent_requirement_contract: Path | None = None) -> dict:
    """Capture original image bytes, then sign one explicit task before planning.

    This mutates only the disposable benchmark Git repository: its intentional
    dirty bottle.py bytes become a recorded baseline; no upstream bytes are
    restored. Public instruction/check files become immutable signed inputs.
    """
    started = time.monotonic()
    if (type(disable_intent_autoencoder) is not bool or type(enable_source_unit_autoencoder) is not bool
            or type(source_unit_project_logic_families) is not bool):
        raise ValueError("Intent preprocessing ablation requires a boolean")
    if intent_action_384_config is not None and (enable_source_unit_autoencoder or any(value is not None for value in (
            intent_checkpoint_descriptor, intent_projection_request, intent_projection_request_sha256))):
        raise ValueError("select one explicit Intent preprocessing route")
    repository = Path(repository).resolve(strict=True)
    instruction = Path(instruction).resolve(strict=True)
    state = Path(state).absolute()
    if state.is_relative_to(repository) or state.exists() or state.resolve() != state:
        raise ValueError("state must be a new non-symlink directory outside the worker repository")
    if _git(repository, "rev-parse", "--show-toplevel").decode().strip() != str(repository):
        raise ValueError("repository must be the exact existing task Git root")
    text = instruction.read_text()
    if not text.strip() or len(text.encode()) > 32768:
        raise ValueError("public instruction must contain 1 to 32768 bytes")
    requirements = _load_intent_requirement_contract(intent_requirement_contract, text)
    strategy = _planning_strategy(requirements)
    # This optional datasets-owned stage precedes any goal/task declarations.
    # Its output never replaces the raw instruction or independent domain roots.
    intent_action_selection = None
    if intent_action_384_config is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.intent_advisor_selection import prepare_intent_384_selection
        intent_advice, intent_action_selection, intent_elapsed_ns = prepare_intent_384_selection(
            instruction=text, config_path=intent_action_384_config, enabled=not disable_intent_autoencoder)
    else:
        from ipfs_accelerate_py.agent_supervisor.runtime.intent_autoencoder_advisor import prepare_intent_advice
        intent_advice = prepare_intent_advice(instruction=text,
            checkpoint_descriptor_path=intent_checkpoint_descriptor,
            projection_request_path=intent_projection_request,
            projection_request_sha256=intent_projection_request_sha256,
            enabled=not disable_intent_autoencoder and not enable_source_unit_autoencoder)
        intent_elapsed_ns = intent_advice["elapsed_ns"]
    from .terminal_source_unit_advice import prepare_source_unit_advice, _wire as source_unit_wire
    source_unit_advice = prepare_source_unit_advice(instruction=text,
        enabled=enable_source_unit_autoencoder,
        intent_descriptor_path=None if disable_intent_autoencoder else intent_checkpoint_descriptor,
        security_descriptor_path=source_unit_security_decoder_descriptor,
        project_logic_families=source_unit_project_logic_families,
        intent_family_context_path=None if disable_intent_autoencoder else source_unit_intent_family_context,
        requested_intent_families=None if disable_intent_autoencoder else source_unit_intent_logic_families)
    names = [name.decode() for name in _git(repository, "ls-files", "-z").split(b"\0") if name]
    if "bottle.py" not in names or len(names) > 252:
        raise ValueError("original Bottle source inventory is missing or exceeds the declared bound")
    for name in ("report.jsonl", INSTRUCTION, SMOKE):
        if (repository / name).exists() or (repository / name).is_symlink() or name in names:
            raise ValueError("declared create/public preparation path already exists: " + name)
    if _git(repository, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError("original input contains undeclared untracked files")
    changed = {name.decode() for name in _git(repository, "diff", "--name-only", "HEAD", "-z").split(b"\0") if name}
    if not changed <= {"bottle.py"}:
        raise ValueError("original input has an unexpected dirty tracked path")
    source_inventory = local._sources(repository, names, max_files=256)
    original_head = _git(repository, "rev-parse", "HEAD").decode().strip()
    original_diff = _git(repository, "diff", "--binary", "HEAD")
    state.mkdir(parents=True)
    intent_preplanning = {"artifact": "intent-advice.json", "artifact_sha256": None,
        "advice_sha256": intent_advice["advice_sha256"], "status": intent_advice["status"],
        "instruction_sha256": intent_advice["instruction_sha256"],
        "seconds": intent_elapsed_ns / 1_000_000_000,
        "before_goal_decomposition": True, "execution_authority": False,
        "completion_authority": False}
    if intent_action_selection is not None:
        intent_preplanning["intent_action_384_selection"] = intent_action_selection
    try:
        _write(state / "intent-advice.json", intent_advice)
        intent_preplanning["artifact_sha256"] = hashlib.sha256((state / "intent-advice.json").read_bytes()).hexdigest()
    except OSError as exc:
        intent_preplanning.update(status="fail_open_persistence_error", error_type=type(exc).__name__)
    source_unit_preplanning = {"artifact": "source-unit-advice.json", "artifact_sha256": None,
        "advice_sha256": source_unit_advice["advice_sha256"], "status": source_unit_advice["status"],
        "instruction_sha256": source_unit_advice["instruction_sha256"],
        "seconds": source_unit_advice["elapsed_ns"] / 1_000_000_000,
        "enabled": enable_source_unit_autoencoder, "before_goal_decomposition": True,
        "logic_families_enabled":source_unit_project_logic_families,
        "source_semantics_verified": False, "execution_authority": False, "completion_authority": False}
    try:
        source_unit_bytes = source_unit_wire(source_unit_advice)
        (state / "source-unit-advice.json").write_bytes(source_unit_bytes)
        source_unit_preplanning["artifact_sha256"] = hashlib.sha256(source_unit_bytes).hexdigest()
    except OSError as exc:
        source_unit_preplanning.update(status="fail_open_persistence_error", error_type=type(exc).__name__)
    (state / "original-image.diff").write_bytes(original_diff)
    _write(state / "original-image.json", {
        "head": original_head, "sources": source_inventory,
        "dirty_paths": sorted(changed), "diff_sha256": hashlib.sha256(original_diff).hexdigest(),
        "upstream_source_restored": False,
    })
    if changed:
        _git(repository, "add", "--", "bottle.py")
        _commit(repository, "Record exact original benchmark image input bytes")
    if local._sources(repository, names, max_files=256) != source_inventory:
        raise ValueError("original source bytes changed while recording their baseline")
    (repository / INSTRUCTION).write_text(text)
    (repository / SMOKE).write_text(PUBLIC_SMOKE)
    _git(repository, "add", "--", INSTRUCTION, SMOKE)
    _commit(repository, "Declare immutable public task instruction and structural smoke check")
    exclude = repository / ".git/info/exclude"
    with exclude.open("a") as stream:
        stream.write("\n.runtime/\n")
    profile_dir, lifecycle_dir = state / "profile", state / "lifecycle"
    bootstrap = Supervisor.init_local(repository=repository, consent=True,
        profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    profile = load_local_profile(repository_cid=bootstrap["repository_cid"],
        profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    policy = local.content_identity(local.LOCAL_POLICY)
    scope = [INSTRUCTION, SMOKE, "bottle.py", "report.jsonl"]
    acceptance = {
        "criterion_key": "report-shape-and-syntax",
        "criterion": "Bottle compiles and the declared report has the public requested JSONL shape; benchmark correctness remains unverified",
        "evidence_cids": [], "validation_keys": ["public-structural-smoke"],
    }
    spec = {
        "task_key": "TB-CODE-TASK", "scope_paths": scope, "dependencies": [],
        "outputs": [
            {"path": "bottle.py", "effect": "modify", "media_type": "text/x-python"},
            {"path": "report.jsonl", "effect": "create", "media_type": "text/plain"},
        ],
        "validations": [{"validation_key": "public-structural-smoke", "argv": ARGV,
            "cwd": ".", "expected_exit_codes": [0], "policy_cid": policy}],
        "acceptance": [acceptance],
    }
    domains = local.local_planning_domain_declarations(repository=repository,
        profile_dir=profile_dir, lifecycle_dir=lifecycle_dir, task_specs=[spec])
    allowlist = RepositoryAllowlist.from_roots([repository])
    budget = PromptWorkflowBudget(max_files=8, max_scan_bytes=8388608, max_file_bytes=262144,
        max_symbols=1024, max_prompt_tokens=32768, max_provider_tokens=4096,
        max_latency_ms=90000, max_goals=2, max_tasks=1, max_evidence=16,
        max_graph_depth=4, max_serialized_bytes=1048576, max_rescue_actions=1)
    request = PromptWorkflowRequest(
        prompt_source=PromptSource.inline(text, redacted_metadata={
            "summary": "Plan the public Bottle source repair and declared JSONL report; exact public instruction is supplied in constraints.",
        }),
        repository_root=str(repository), directory=str(repository),
        repository_root_cid=repository_root_cid(repository), allowlist_cid=allowlist.allowlist_cid,
        scan_policy=DirectoryScanPolicy(policy_id="terminal-public-source-scan", scanner_version="1",
            include_patterns=("bottle.py", INSTRUCTION, SMOKE)),
        planning_policy=PromptPlanningPolicy(policy_id=("terminal-intent-symbolic-planner"
            if strategy == "intent_symbolic" else "terminal-codex-planner"),
            provider_preferences=(PROVIDER,), model_preferences=(MODEL,), allow_model=strategy != "intent_symbolic",
            fallback_policy=LocalFallbackPolicy.DISABLED),
        output_policy=PromptOutputPolicy(policy_id="terminal-planning-preview", mode="markdown",
            output_root=str(repository), allowed_output_roots=(str(repository),),
            markdown_path=".runtime/planner.todo.md"),
        budget=budget, caller=profile.identity_did,
        program_root=local._tree(source_inventory), intent_ir_root=local.content_identity(domains["intent"]),
        legal_ir_root=local.content_identity(domains["legal"]),
        security_ir_root=local.content_identity(domains["security"]), policy_root=policy,
    )
    details = scan_prompt_directory_detailed(request, repository_allowlist=allowlist)
    request = replace(request, program_root=details.receipt.program_root)
    details = scan_prompt_directory_detailed(request, repository_allowlist=allowlist, previous=details)
    config, scan = _config(repository), details.receipt
    evidence = _select_evidence(request, scan, config)
    acceptance["evidence_cids"] = [evidence[0].evidence_cid]
    constraints = _constraints(spec, text)
    manifest = local.author_local_benchmark_manifest(repository=repository, profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir, task_specs=[spec], planning_roots={
            "request_cid": request.request_cid, "scan_cid": scan.scan_cid, "program_root": request.program_root,
        }, planning_inputs={"request": request.to_dict(), "scan": scan.to_dict(),
            "domain_declarations": domains, "selected_evidence": [row.to_dict() for row in evidence]},
        intent_requirements=requirements)
    prepared = {
        "schema": "terminal-indexed-public-preparation@1", "repository": str(repository),
        "state": str(state), "original_head": original_head, "manifest": manifest,
        "request": request.to_dict(), "scan": scan.to_dict(), "constraints": constraints,
        "spec": spec, "provider": PROVIDER, "model": MODEL, "reasoning_effort": REASONING,
        "planner_timeout_seconds": 90, "max_total_agent_seconds": 300,
        "planning_and_cold_index_overhead_included": True,
        "provider_calls": 0, "report_preseeded": False, "benchmark_success": None,
        "planning_strategy": strategy,
        "worker_inputs": [INSTRUCTION, SMOKE, "bottle.py"],
        "query": text, "query_provenance": "public instruction.md verbatim",
        "official_verifier_in_context": False,
        "intent_preplanning": intent_preplanning,
        "source_unit_preplanning": source_unit_preplanning,
        "preparation_seconds": time.monotonic() - started,
    }
    if requirements is not None:
        prepared["intent_requirement_contract"] = requirements
    _write(state / "prepared.json", prepared)
    if strategy != "intent_symbolic":
        provider_request = build_prompt_goal_provider_request(
            request, scan, config=config, constraint_summaries=constraints)
        if requirements is not None:
            from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import build_intent_plan_provider_request

            provider_request = build_intent_plan_provider_request(provider_request, requirements)
            if len(provider_request.encode("utf-8")) > config.max_provider_request_bytes:
                raise ValueError("intent planning request exceeds the existing provider byte budget")
        (state / "provider-request.json").write_text(provider_request + "\n")
    for index, artifact in enumerate(details.artifacts):
        _write(state / f"scan-artifact-{index}.json", local._plain(dict(artifact.payload)))
    return prepared


def initial_context(*, state: Path, model_snapshot: Path | None = None, model_revision: str = "",
                    train_autoencoder: bool = False, weight_transfer: dict | None = None,
                    canonical_cve_training: dict | None = None, security_checkpoint: dict | None = None,
                    security_checkpoint_hub: dict | None = None, formula_decoder: dict | None = None,
                    header_protocol: dict | None = None, source384_config: Path | None = None) -> dict:
    """Prepare source indexes and a real empty-owner observation before planning.

    This cost belongs to the full arm's agent time. The subsequent context()
    reuses these exact source artifacts and captures the newly admitted tasks.
    """
    from .terminal_initial_context import prepare_initial_context

    state = state.resolve(strict=True)
    return prepare_initial_context(state=state, prepared=_load_prepared(state),
        model_snapshot=model_snapshot, model_revision=model_revision,
        required_raw_paths=[INSTRUCTION, SMOKE], train_autoencoder=train_autoencoder,
        weight_transfer=weight_transfer, canonical_cve_training=canonical_cve_training,
        security_checkpoint=security_checkpoint, security_checkpoint_hub=security_checkpoint_hub,
        formula_decoder=formula_decoder, header_protocol=header_protocol,
        **({"source384_config": source384_config} if source384_config is not None else {}))


def _require_same_initial_selection(current, initial):
    if (current["receipt"]["descriptor"] != initial["receipt"]["descriptor"]
            or current["descriptor"] != initial["descriptor"]
            or current["summaries"] != initial["summaries"]):
        raise ValueError("selected initial context changed during planning")


def _plan_symbolic(*, state, prepared, initial, timeout_seconds):
    if prepared["intent_requirement_contract"]["schema"] == "intent-plan-requirement-contract@3":
        from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import applicability_budget
        # Includes nested verify/store/load calls performed by materialization.
        # A serialized receipt never renews this caller's remaining deadline.
        with applicability_budget(timeout_seconds):
            return _plan_symbolic_in_budget(state=state, prepared=prepared,
                initial=initial, timeout_seconds=timeout_seconds)
    return _plan_symbolic_in_budget(state=state, prepared=prepared,
        initial=initial, timeout_seconds=timeout_seconds)


def _plan_symbolic_in_budget(*, state, prepared, initial, timeout_seconds):
    """Select checked operations and admit them through the existing local gate."""
    from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan

    started = time.monotonic()
    def require_remaining_budget():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("symbolic planning exhausted the declared planner time budget")
    (state / "planner-invoked.json").open("x").write(json.dumps({"planning_strategy": "intent_symbolic"}) + "\n")
    result = {"schema": "terminal-indexed-planner-result@1", "qualified": False,
        "provider": PROVIDER, "model": MODEL, "reasoning_effort": REASONING,
        "provider_output_token_cap_enforced": False, "max_total_agent_seconds": 300,
        "planner_timeout_seconds": timeout_seconds, "planning_strategy": "intent_symbolic",
        "planning_and_cold_index_overhead_included": True, "benchmark_success": None,
        "provider_calls": 0, "provider_observation": {}, "provider_receipt": None,
        "execution_receipt": None, "model_request_sha256": None, "isolation_verified_here": False,
        "source_semantics_verified": False, "semantic_alignment_verified": False,
        "proof_authority": False, "execution_authority": False, "completion_authority": False}
    try:
        current = _load_prepared(state)
        if current != prepared:
            raise ValueError("preparation changed before symbolic planning")
        if initial is not None:
            from .terminal_initial_context import load_initial_context
            current_initial = load_initial_context(state=state, prepared=current, require_empty_owner=True)
            _require_same_initial_selection(current_initial, initial)
        header_kwargs = {}
        if prepared["intent_requirement_contract"]["schema"] == "intent-plan-requirement-contract@3":
            if initial is None:
                raise ValueError("header planning requires the runtime captured initial context")
            header_kwargs["source_applicability_nomination"] = current_initial["descriptor"]["source384_context"]["source_applicability_nomination"]
            require_remaining_budget()
            header_kwargs["applicability_timeout_seconds"] = min(45., timeout_seconds - (time.monotonic() - started))
        planned = build_intent_symbolic_plan(prepared["intent_requirement_contract"],
            manifest=prepared["manifest"], **header_kwargs)
        require_remaining_budget()
        result["symbolic_planning"] = planned["receipt"]
        _write(state / "symbolic-planning-receipt.json", planned["receipt"])
        if initial is not None:
            current_initial = load_initial_context(state=state, prepared=_load_prepared(state), require_empty_owner=True)
            _require_same_initial_selection(current_initial, initial)
            require_remaining_budget()
        if header_kwargs:
            require_remaining_budget()
            header_kwargs["applicability_timeout_seconds"] = min(45., timeout_seconds - (time.monotonic() - started))
        admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=prepared["manifest"],
            requirement_bindings=planned["requirement_bindings"], **header_kwargs)
        require_remaining_budget()
        _write(state / "admission.json", admission)
        with IntentRepository(state / "intent.duckdb") as intent:
            materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        require_remaining_budget()
        result.update(qualified=True, task_cids=materialized["task_cids"],
            goals=len(planned["graph"].goals), tasks=len(planned["graph"].tasks),
            requirement_coverage=admission["receipt"]["payload"]["requirement_coverage"])
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        if getattr(exc, "requirement_coverage", None) is not None:
            result["requirement_coverage"] = exc.requirement_coverage
        if getattr(exc, "symbolic_issues", None) is not None:
            result["symbolic_issues"] = exc.symbolic_issues
    result["initial_indexed_context"] = ({"descriptor": initial["receipt"]["descriptor"],
        "semantic_root_cid": initial["semantic"]["semantic_root_cid"],
        "world_snapshot_cid": initial["capture"]["snapshot"]["snapshot_cid"],
        "index_id": initial["retrieval"]["index_id"],
        "summary_sha256": hashlib.sha256(json.dumps(initial["summaries"],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "supplied_to_router": False, "execution_authority": False, "completion_authority": False} if initial else None)
    for stage in ("intent_preplanning", "source_unit_preplanning"):
        result[stage] = {"status": "not_used_for_symbolic_selection", "supplied_to_router": False,
            "before_goal_decomposition": True, "source_semantics_verified": False,
            "execution_authority": False, "completion_authority": False}
    result["elapsed_seconds"] = time.monotonic() - started
    _write(state / "planning-result.json", result)
    return result


def plan(state: Path, *, provider_callable=None, timeout_seconds: int = 90) -> dict:
    """Select v2 operations symbolically, otherwise call the qualified router once.

    The callable receives pinned provider/model/reasoning/timeout options and
    returns ``{text, observation, execution_receipt}``. This owner process never
    runs the model CLI: a read-only filesystem is insufficient to hide owner
    signing keys. The deployment must qualify its UID/tool boundary separately.
    """
    outer_started = time.monotonic()
    if not callable(provider_callable) and not (Path(state) / "prepared.json").is_file():
        raise ValueError("planning requires an independently qualified isolated worker router callable")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 90:
        raise ValueError("planner timeout must be an integer from 1 to 90 seconds within the overall trial budget")
    state = state.resolve(strict=True)
    prepared = _load_prepared(state)
    if (prepared.get("intent_requirement_contract") or {}).get("schema") == "intent-plan-requirement-contract@3":
        from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import applicability_budget
        left = timeout_seconds - (time.monotonic() - outer_started)
        if left <= 0:
            raise TimeoutError("symbolic planning exhausted the declared planner time budget")
        with applicability_budget(left):
            result = _plan_prepared(state, prepared=prepared, provider_callable=provider_callable,
                timeout_seconds=left, aggregate_started=outer_started)
        result["elapsed_seconds"] = time.monotonic() - outer_started
        result["planner_timeout_seconds"] = timeout_seconds
        _write(state / "planning-result.json", result)
        return result
    return _plan_prepared(state, prepared=prepared, provider_callable=provider_callable, timeout_seconds=timeout_seconds)


def _plan_prepared(state, *, prepared, provider_callable, timeout_seconds, aggregate_started=None):
    strategy = _planning_strategy(prepared.get("intent_requirement_contract"))
    if strategy != "intent_symbolic" and not callable(provider_callable):
        raise ValueError("planning requires an independently qualified isolated worker router callable")
    repository = Path(prepared["repository"])
    constraints = prepared["constraints"]
    initial = None
    if (state / "initial-context-result.json").exists():
        from .terminal_initial_context import load_initial_context, stage_initial_context_nomination
        initial = stage_initial_context_nomination(state=state, prepared=prepared, require_empty_owner=True)
        constraints = {**constraints, "constraint_summaries": [
            *constraints["constraint_summaries"], *initial["summaries"]]}
    if strategy == "intent_symbolic":
        if aggregate_started is not None:
            from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import require_applicability_budget
            require_applicability_budget()
        return _plan_symbolic(state=state, prepared=prepared, initial=initial, timeout_seconds=timeout_seconds)
    version = subprocess.run(["codex", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if version != "codex-cli 0.158.0":
        raise ValueError("native baseline parity requires codex-cli 0.158.0")
    (state / "planner-invoked.json").open("x").write(json.dumps({"cli_version": version}) + "\n")
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    from ipfs_accelerate_py.agent_supervisor.runtime.intent_autoencoder_advisor import (
        intent_planner_summary, load_intent_advice,
    )
    selected_intent = prepared.get("intent_preplanning")
    if type(selected_intent) is not dict:
        selected_intent = {}
    selected_source_unit = prepared.get("source_unit_preplanning")
    if type(selected_source_unit) is not dict:
        selected_source_unit = {}
    if "intent_action_384_selection" in selected_intent:
        from ipfs_accelerate_py.agent_supervisor.runtime.intent_advisor_selection import (
            load_intent_384_selection, intent_384_planner_summary,
        )
        intent_advice = load_intent_384_selection(path=state / "intent-advice.json",
            expected_sha256=selected_intent.get("artifact_sha256", ""), instruction=prepared["query"],
            selection=selected_intent["intent_action_384_selection"])
        intent_summary, intent_advice = intent_384_planner_summary(intent_advice,
            instruction=prepared["query"], maximum_bytes=_config(repository).max_summary_bytes)
    else:
        intent_advice = load_intent_advice(path=state / "intent-advice.json",
            expected_sha256=selected_intent.get("artifact_sha256", ""), instruction=prepared["query"])
        intent_summary, intent_advice = intent_planner_summary(intent_advice,
            instruction=prepared["query"], maximum_bytes=_config(repository).max_summary_bytes)
    intent_delivery = {"status": intent_advice["status"],
        "instruction_sha256": intent_advice["instruction_sha256"],
        "advice_sha256": intent_advice["advice_sha256"],
        "before_goal_decomposition": True, "supplied_to_router": False,
        "execution_authority": False, "completion_authority": False}
    if selected_source_unit.get("enabled") is True:
        # The new document route independently checks generated slots against
        # each source clause. Do not also deliver legacy unchecked slot advice.
        intent_summary = None
        intent_delivery["status"] = "superseded_by_source_unit_preplanning"
    if intent_summary is not None:
        proposed_constraints = {**constraints, "constraint_summaries": [
            *constraints["constraint_summaries"], intent_summary]}
        try:
            # Retain all planner bounds and text checks. Optional advice that
            # cannot fit those existing contracts is omitted, never relaxed.
            build_prompt_goal_provider_request(request, scan, config=_config(repository),
                constraint_summaries=proposed_constraints)
        except Exception as exc:
            intent_delivery.update(status="fail_open_planner_context_rejected", error_type=type(exc).__name__)
            intent_summary = None
        else:
            constraints = proposed_constraints
    try:
        _write(state / "planner-intent-advice.json", intent_advice)
        intent_delivery["artifact_persisted"] = True
    except OSError as exc:
        intent_delivery.update(artifact_persisted=False, persistence_error_type=type(exc).__name__)
    from .terminal_source_unit_advice import load_source_unit_advice, source_unit_planner_summary
    source_unit_summary = None
    source_unit_delivery = {"status": "disabled", "supplied_to_router": False,
        "before_goal_decomposition": True, "source_semantics_verified": False,
        "execution_authority": False, "completion_authority": False}
    if selected_source_unit.get("enabled") is True:
        try:
            source_advice, replay_cost = load_source_unit_advice(path=state / "source-unit-advice.json",
                expected_sha256=selected_source_unit.get("artifact_sha256"), instruction=prepared["query"])
            source_unit_delivery.update(status=source_advice["status"],
                instruction_sha256=source_advice["instruction_sha256"],
                advice_sha256=source_advice["advice_sha256"], **replay_cost)
            source_unit_summary = source_unit_planner_summary(source_advice,
                maximum_bytes=_config(repository).max_summary_bytes)
            if source_unit_summary is not None:
                proposed_constraints = {**constraints, "constraint_summaries": [
                    *constraints["constraint_summaries"], source_unit_summary]}
                build_prompt_goal_provider_request(request, scan, config=_config(repository),
                    constraint_summaries=proposed_constraints)
                constraints = proposed_constraints
        except Exception as exc:
            source_unit_summary = None
            source_unit_delivery.update(status="fail_open_planner_context_rejected", error_type=type(exc).__name__)
    requirement_contract = prepared.get("intent_requirement_contract")
    requirement_bindings = None
    observation, calls, execution_receipt, model_request_sha256 = {}, 0, None, None
    started = time.monotonic()
    def router(prompt):
        nonlocal observation, calls, execution_receipt, model_request_sha256, requirement_bindings
        if calls:
            raise RuntimeError("one planner provider call maximum")
        if initial is not None:
            current = load_initial_context(state=state, prepared=_load_prepared(state), require_empty_owner=True)
            _require_same_initial_selection(current, initial)
        if requirement_contract is not None:
            from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
                build_intent_plan_provider_request,
            )
            # The exact source and signed ledger must remain fresh at dispatch.
            current = _load_prepared(state)
            if current.get("intent_requirement_contract") != requirement_contract:
                raise ValueError("intent requirement contract changed before planning")
            prompt = build_intent_plan_provider_request(prompt, requirement_contract)
            if len(prompt.encode("utf-8")) > _config(repository).max_provider_request_bytes:
                raise ValueError("intent planning request exceeds the existing provider byte budget")
        model_request_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        with (state / "planner-provider-request.json").open("x") as stream:
            stream.write(prompt)
        calls += 1
        try:
            response = provider_callable(prompt, repository=repository, provider=PROVIDER,
                model=MODEL, reasoning_effort=REASONING, timeout=timeout_seconds,
                max_new_tokens=4096, trace_path=state / "planner-trace.jsonl")
            if not isinstance(response, dict) or not isinstance(response.get("text"), str):
                raise ValueError("isolated router must return text and observed execution metadata")
            observation = response.get("observation", {})
            execution_receipt = response.get("execution_receipt")
            response = response["text"]
            (state / "provider-response.txt").write_text(response)
            if requirement_contract is not None:
                from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
                    parse_intent_plan_proposal,
                )
                response, requirement_bindings = parse_intent_plan_proposal(
                    response, contract=requirement_contract,
                )
            return response
        except Exception as exc:
            observation = getattr(exc, "observation", observation)
            execution_receipt = getattr(exc, "execution_receipt", execution_receipt)
            raise
        finally:
            if not isinstance(observation, dict):
                observation = {}
            observation = {key: observation.get(key) for key in (
                "prompt_tokens", "completion_tokens", "cached_tokens", "input_tokens", "output_tokens",
                "cached_input_tokens", "reasoning_output_tokens", "total_tokens", "usage",
                "total_cost_usd", "session_id", "model_id",
            )}
            _write(state / "provider-observation.json", observation)

    result = {"schema": "terminal-indexed-planner-result@1", "qualified": False,
        "provider": PROVIDER, "model": MODEL, "reasoning_effort": REASONING,
        "provider_output_token_cap_enforced": False, "max_total_agent_seconds": 300,
        "planner_timeout_seconds": timeout_seconds,
        "planning_and_cold_index_overhead_included": True, "benchmark_success": None}
    try:
        planning = generate_prompt_goal_graph(request, scan, router=router,
            config=_config(repository, timeout_seconds=timeout_seconds), constraint_summaries=constraints)
        result["provider_receipt"] = planning.receipt.to_dict()
        if planning.receipt.outcome != "provider" or planning.receipt.fallback.used:
            raise ValueError("real provider proposal required; fallback cannot qualify")
        if initial is not None:
            current_initial = load_initial_context(state=state, prepared=_load_prepared(state), require_empty_owner=True)
            _require_same_initial_selection(current_initial, initial)
        admission = local.admit_local_benchmark_plan(
            graph=planning.graph, manifest=prepared["manifest"],
            requirement_bindings=requirement_bindings,
        )
        # Preserve the accepted, signed graph even if native storage fails.
        # Retrying materialization must not require another provider invocation.
        _write(state / "admission.json", admission)
        with IntentRepository(state / "intent.duckdb") as intent:
            materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        result.update(qualified=True, task_cids=materialized["task_cids"],
            goals=len(planning.graph.goals), tasks=len(planning.graph.tasks))
        if requirement_contract is not None:
            result["requirement_coverage"] = admission["receipt"]["payload"]["requirement_coverage"]
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        result.setdefault("provider_receipt", getattr(exc, "provider_receipt", None))
        if getattr(exc, "requirement_coverage", None) is not None:
            result["requirement_coverage"] = exc.requirement_coverage
    result.update(provider_calls=calls, provider_observation=observation,
        planning_strategy=strategy,
        execution_receipt=execution_receipt, isolation_verified_here=False,
        model_request_sha256=model_request_sha256,
        initial_indexed_context=({"descriptor": initial["receipt"]["descriptor"],
            "semantic_root_cid": initial["semantic"]["semantic_root_cid"],
            "world_snapshot_cid": initial["capture"]["snapshot"]["snapshot_cid"],
            "index_id": initial["retrieval"]["index_id"],
            "summary_sha256": hashlib.sha256(json.dumps(initial["summaries"],
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "supplied_to_router": calls == 1, "execution_authority": False,
            "completion_authority": False} if initial else None),
        elapsed_seconds=time.monotonic() - started)
    intent_delivery["supplied_to_router"] = calls == 1 and intent_summary is not None
    result["intent_preplanning"] = intent_delivery
    source_unit_delivery["supplied_to_router"] = calls == 1 and source_unit_summary is not None
    result["source_unit_preplanning"] = source_unit_delivery
    _write(state / "planning-result.json", result)
    return result


def context(*, state: Path, model_snapshot: Path | None = None, model_revision: str = "") -> dict:
    """Hydrate real native vectors, capsules, Doctor and world for an admitted task.

    Omit the local model only for the explicitly labelled lexical ablation.
    The complete Bottle file is indexed; the worker gets a bounded projection.
    """
    entered = time.monotonic()
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle

    state = state.resolve(strict=True)
    prepared = _load_prepared(state)
    admission = json.loads((state / "admission.json").read_text())
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    if admission["manifest"] != prepared["manifest"]:
        raise ValueError("context admission differs from independently prepared manifest")
    repository = Path(prepared["repository"])
    if bool(model_snapshot) != bool(model_revision):
        raise ValueError("learned index needs both the pinned local snapshot and revision")
    output = repository / ".runtime/terminal-context"
    if output.exists():
        raise ValueError("context output already exists")
    started = time.monotonic()
    initial_seconds = started - entered
    stage_started = started
    stage_seconds = {}
    vectors = repository / ".runtime/terminal-vectors"
    reused = (state / "initial-context-result.json").exists()
    task_cids = [task.task_cid for task in verified["graph"].tasks]
    if len(task_cids) != 1:
        raise ValueError("one independently declared Terminal-Bench task required")
    if reused:
        from .terminal_initial_context import bind_admitted_context
        hydrated = bind_admitted_context(state=state, prepared=prepared, admission=admission,
            verified=verified, output=output, model_snapshot=model_snapshot, model_revision=model_revision)
        prepared_context, indexed = hydrated["prepared_context"], hydrated["indexed"]
        diagnostic_artifact = hydrated["diagnostic_artifact"]
        stage_seconds["verified_initial_index_reuse_and_admitted_world_capture"] = time.monotonic() - stage_started
    else:
        if model_snapshot:
            from benchmarks.agent_supervisor.container_coding.learned_vector_preflight import qualify
            indexed = qualify(repository, vectors, ["bottle.py"], prepared["query"], model_snapshot, model_revision)
        else:
            from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify
            indexed = qualify(repository, vectors, ["bottle.py"], prepared["query"])
        stage_finished = time.monotonic()
        stage_seconds["vector_qualification"] = stage_finished - stage_started
        stage_started = stage_finished
        with duckdb.connect(str(vectors / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
            row = connection.execute("SELECT payload FROM snapshots WHERE id=?", [indexed["index_id"]]).fetchone()
            snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(row[0]))
        stage_finished = time.monotonic()
        stage_seconds["persisted_snapshot_reopen"] = stage_finished - stage_started
        stage_started = stage_finished
        with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
            prepared_context = prepare_supervised_task_context(
                repository=repository, intent=intent, task_cid=task_cids[0],
                paths=prepared["worker_inputs"], required_raw_paths=[INSTRUCTION, SMOKE],
                output=output, code_vector_snapshot=snapshot,
                code_vector_result=CodeVectorSearchResult.from_dict(indexed["hits"]),
                code_query_text=prepared["query"], semantic_max_symbols=1024,
                semantic_worker_query=prepared["query"], semantic_worker_max_bytes=32768,
            )
        stage_seconds["supervised_semantic_world_context"] = time.monotonic() - stage_started
        diagnostic_artifact = output / "semantic/doctor.json"
    stage_started = time.monotonic()
    bundle = write_task_context_bundle(repository=repository, prepared=[prepared_context],
        output=repository / ".runtime/terminal-context-bundle.json")
    stage_finished = time.monotonic()
    stage_seconds["bundle_persistence"] = stage_finished - stage_started
    stage_started = stage_finished
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition import (
        assess_doctor_repair_eligibility,
    )
    doctor = assess_doctor_repair_eligibility(
        repository=repository, admission=admission,
        diagnostic_artifact=diagnostic_artifact, paths=prepared["worker_inputs"],
    )
    stage_finished = time.monotonic()
    stage_seconds["doctor_eligibility"] = stage_finished - stage_started
    stage_started = stage_finished
    _write(state / "doctor-repair-eligibility.json", doctor)
    stage_seconds["doctor_report_persistence"] = time.monotonic() - stage_started
    result = {
        "schema": "terminal-indexed-context-preparation@1", "task_cid": task_cids[0],
        "context_bundle": bundle, "learned_embeddings": bool(model_snapshot),
        "index_id": indexed["index_id"], "indexed_symbols": indexed["symbols"],
        "native_fact_rows_replayed": indexed["native_fact_rows_replayed"],
        "semantic_root_cid": prepared_context["semantic_root_cid"],
        "world_snapshot_cid": prepared_context["world_snapshot_cid"],
        "full_capsules": prepared_context["semantic"]["capsules"],
        "worker_capsules": prepared_context["semantic"]["worker_capsules"],
        "worker_semantic_bytes": prepared_context["semantic"]["compact_bytes"],
        "doctor_repair": doctor,
        "initial_indexes_reused": reused,
        "new_embedding_calls": 0 if reused else None,
        "initial_context_descriptor": prepared_context.get("initial_context_descriptor"),
        "timings": {
            "schema": "terminal-context-stage-timings@1", "clock": "monotonic",
            "initial_function_imports_and_admission_seconds": initial_seconds,
            "initial_phase_included_in_seconds": False,
            "nonoverlapping_seconds": stage_seconds,
            "nested_helper_seconds": ({} if reused else {"vector_qualification": dict(indexed.get("seconds", {}))}),
            "nested_timings_overlap_parent": True,
            "nested_vector_timings_may_overlap_each_other": True,
            "final_context_result_persistence_included": False,
        },
        "seconds": time.monotonic() - started, "provider_calls": 0,
        "official_verifier_in_context": False, "benchmark_success": None,
        "nomination_only": True, "semantic_equivalence_claimed": False,
    }
    if prepared_context.get("codebase_autoencoder") is not None:
        result["codebase_autoencoder"] = prepared_context["codebase_autoencoder"]
        result["codebase_autoencoder_catalog"] = prepared_context["codebase_autoencoder_catalog"]
        result["new_autoencoder_training_steps"] = prepared_context["new_autoencoder_training_steps"]
    _write(state / "context-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--repository", type=Path, required=True)
    prep.add_argument("--instruction", type=Path, required=True)
    prep.add_argument("--state", type=Path, required=True)
    prep.add_argument("--intent-checkpoint-descriptor", type=Path)
    prep.add_argument("--intent-action-384-config", type=Path)
    prep.add_argument("--intent-projection-request", type=Path)
    prep.add_argument("--intent-projection-request-sha256")
    prep.add_argument("--disable-intent-autoencoder", action="store_true")
    prep.add_argument("--intent-requirement-contract", type=Path,
        help="Use source-bound @1 provider coverage or @2 reviewed symbolic operations before admission")
    prep.add_argument("--enable-source-unit-autoencoder", action="store_true")
    prep.add_argument("--source-unit-security-decoder-descriptor", type=Path)
    prep.add_argument("--source-unit-project-logic-families", action="store_true",
        help="Requires --enable-source-unit-autoencoder; include native family and typed referent views")
    prep.add_argument("--source-unit-intent-family-context", type=Path)
    prep.add_argument("--source-unit-intent-logic-family", action="append", dest="source_unit_intent_logic_families")
    indexed = actions.add_parser("context")
    indexed.add_argument("--state", type=Path, required=True)
    indexed.add_argument("--model-snapshot", type=Path)
    indexed.add_argument("--model-revision", default="")
    initial = actions.add_parser("initial-context")
    initial.add_argument("--state", type=Path, required=True)
    initial.add_argument("--model-snapshot", type=Path)
    initial.add_argument("--model-revision", default="")
    args = vars(parser.parse_args())
    action = args.pop("action")
    result = {"prepare": prepare, "context": context, "initial-context": initial_context}[action](**args)
    print(json.dumps({key: result.get(key) for key in (
        "schema", "qualified", "state", "provider_calls", "benchmark_success", "failure",
    )}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
