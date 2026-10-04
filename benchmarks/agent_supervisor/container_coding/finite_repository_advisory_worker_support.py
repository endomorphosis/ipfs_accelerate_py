"""Advisory CodebaseIR preparation around the separately admitted worker loop.

This authored ten-file fixture trains structural features and checks a closed
mathematical AST lowering. Its records cannot authorize launch, publication,
completion, task omission, Python equivalence or optimizer convergence. Actual
worker ownership and signed model-off admission remain in the worker harness.
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)

from .finite_repository_admission_experiment import _git, _sources
from .finite_repository_candidate_experiment import _capture_complete, _lowering_scope, _with_custody
from .terminal_codebase_adaptation_experiment import (
    _artifact_bytes, _check_preview, _context, _continuation, _idle, _measure,
    _owned_file_pins, _phase, _preview, _registry_inventory,
)
from .terminal_codebase_finite_experiment import INTENT
from .terminal_codebase_finite_index import digest, wire
from .terminal_codebase_finite_service_experiment import _authority_materials, _open, _scheduler

MODULE = "benchmarks.agent_supervisor.container_coding.finite_repository_advisory_worker_support"
SCHEMA = "finite-repository-advisory-worker-support@1"
BRIDGE_SCHEMA = "finite-repository-advisory-worker-bridge@1"
BRIDGE_SCOPE = "advisory_generated_bytes_to_separately_admitted_native_worker_candidate"
OCCURRENCE_SCHEMA = "finite-advisory-worker-metadata-occurrence@1"
CONFIGURATION = {"epochs": 16, "learning_rate": .01, "seed": 1729}
SELECTIONS = (("calc.py", "train"), ("known_variant.py", "train"), ("tune.py", "tune"), ("canary.py", "canary"))
PATHS = frozenset(("calc.py", "decoy.py", "support.py", "consumer.py", "unsupported.py",
    "check_type.py", "check_offset.py", "known_variant.py", "tune.py", "canary.py"))
CLAIMS = {key: False for key in (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority", "execution_authority",
    "publication_authority", "completion_authority", "omission_authority", "production_activated",
    "convergence_proved", "generalization_verified", "parser_correctness_proved",
    "universal_python_semantics_proved", "formal_decoder_available", "model_influenced_worker_edit",
)}
BRIDGE_FIELDS = frozenset({"schema", "scope", "admission_cid", "reviewed_candidate_cid",
    "generated_result_cid", "worker_candidate_cid", "head", "source_cid", "replacement_cid",
    "replacement_sha256", "task_cid", "task_revision", "administrator_task_cids",
    "model_binding", "model_context_cid", "model_version_id", "training_steps", "provider_calls",
    "bridge_cid", *CLAIMS})


def _need(value, message):
    if not value:
        raise ValueError(message)


def _same(left, right):
    return canonical_dag_json_bytes(left) == canonical_dag_json_bytes(right)


def _plain(value):
    def inert(item):
        if isinstance(item, Mapping):
            return {key: inert(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [inert(child) for child in item]
        return item
    return json.loads(wire(inert(value)))


def _write(path, value, *, replace_existing=False):
    raw = wire(value) + b"\n"
    with Path(path).open("wb" if replace_existing else "xb") as stream:
        stream.write(raw)


def _pin(path):
    path = Path(path).absolute()
    _need(path.resolve(strict=True) == path and not path.is_symlink() and path.is_file(),
          "canonical retained regular artifact required")
    _need(0 <= path.stat(follow_symlinks=False).st_size <= 64 * 1024**2,
          "retained artifact exceeds byte bound")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode), "regular retained artifact required")
        raw = stream.read(64 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
        current = path.stat(follow_symlinks=False)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns)
    _need(identity(before) == identity(after) == identity(current) and len(raw) == before.st_size
          and len(raw) <= 64 * 1024**2 and path.resolve(strict=True) == path and not path.is_symlink(),
          "retained descriptor or path changed during bounded read")
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def append_metadata_occurrences(records, values):
    """Preserve repeated producer records without collapsing content-equal rows."""
    _need(type(records) is dict and type(values) is dict, "native metadata family mappings required")
    for family, rows in values.items():
        _need(type(family) is str and type(rows) is list, "named metadata occurrence lists required")
        target = records.setdefault(family, [])
        _need(type(target) is list, "metadata occurrence target must remain a list")
        start = len(target)
        target.extend({"schema": OCCURRENCE_SCHEMA, "occurrence": start + offset,
            "record": _plain(row)} for offset, row in enumerate(rows))


def capture_selected_worker_event_artifacts(output):
    """Capture declared lifecycle/Portal/merge records, excluding credential stores."""
    output = Path(output).absolute()
    state = output / "private/launch/state"
    selected = set(output.glob("*.json"))
    for name in ("start-receipt.json", "stop-receipt.json", "worker-stop-receipt.json",
            "child-bootstrap-receipt.json", "native-lifecycle-receipt.json", "local-process-grant.json"):
        path = state / name
        if path.is_file():
            selected.add(path)
    for relative in ("run/admitted_supervisor_events.jsonl",
            "run/admitted_supervisor_events.jsonl.manifest.json"):
        path = state / relative
        if path.is_file():
            selected.add(path)
    for base in (state / "merge_queue/completed", state / "merge_queue/train/receipts", state / "receipts"):
        if base.is_dir():
            selected.update(base.glob("*.json"))
    attempts = state / "run/admitted_database_portal_attempts"
    if attempts.is_dir():
        for attempt in sorted(attempts.iterdir()):
            _need(attempt.is_dir() and not attempt.is_symlink(), "native Portal attempt directory differs")
            for name in ("database-attempt-binding.json", "portal-events.jsonl", "portal-events.jsonl.manifest.json",
                    "portal-strategy.json", "task_queue.json", "portal-task-state.event-driven-checkpoint.json"):
                path = attempt / name
                if path.is_file():
                    selected.add(path)
            logs = attempt / "implementation-logs"
            if logs.is_dir():
                for pattern in ("*context-receipt.json", "*context-capsule.json"):
                    selected.update(logs.glob(pattern))
    _need(len(selected) <= 256, "selected native worker event inventory exceeds fixture bound")
    rows, total = [], 0
    for path in sorted(selected):
        pin = _pin(path)
        raw = path.read_bytes()
        _need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
              "native event bytes differ from the retained descriptor")
        total += len(raw)
        _need(total <= 32 * 1024**2, "selected worker events exceed fixture byte bound")
        value = [json.loads(line) for line in raw.splitlines()] if path.suffix == ".jsonl" else json.loads(raw)
        pending = [value]
        while pending:
            item = pending.pop()
            if type(item) is dict:
                _need(not {"private_key", "private_key_base64", "secret_key", "bootstrap_credentials",
                    "state_owner_bootstrap_credentials", "credentials", "password", "api_key"}.intersection(item),
                    "selected worker event contains a credential field")
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
        _need(_pin(path) == pin, "native event path changed while capturing metadata")
        rows.append({"schema": "finite-advisory-worker-retained-native-event@1",
            "path": str(path.relative_to(output)), "artifact": pin, "record": value,
            "encoding": "exact_retained_bytes_base64", "content_base64": base64.b64encode(raw).decode()})
    return rows


def finite_outcome_projection(preview):
    """Compare finite clause outcomes independently of numerical context IDs."""
    _need(type(preview) is dict and type(preview.get("match")) is dict,
          "complete actual finite preview required")
    match = preview["match"]
    clauses = match["clause_results"]
    _need(type(clauses) is list and len(clauses) == 2
          and type(preview["current_facts_count"]) is int
          and type(preview["selected_task_ids"]) is list
          and type(preview["planning_model_calls"]) is int and preview["planning_model_calls"] == 0
          and type(preview["training_steps_during_preview"]) is int and preview["training_steps_during_preview"] == 0
          and all(preview.get(key) is False for key in ("source_semantics_verified", "proof_authority",
              "execution_authority", "completion_authority", "production_admitted", "worker_launched", "convergence_proved")),
          "finite advisory preview gained fitting, authority or an incomplete clause population")
    # Preserve complete clauses except native evidence identifiers, which differ
    # because each preview performs fresh observation. Fact truth/arguments and
    # selected operations are already checked by the native preview verifier.
    return {"source_cid": match["source_cid"], "query": match["query"],
        "domain_inputs": match["observation"]["domain_inputs"],
        "observations": match["observation"]["observations"],
        "eligible_clause_ids": match["eligible_clause_ids"],
        "residual_clause_ids": match["residual_clause_ids"],
        "clause_results": clauses,
        "current_facts_count": preview["current_facts_count"],
        "selected_task_ids": preview["selected_task_ids"]}


def create_advisory_worker_sources(repository):
    """Commit the complete cohort before any owner profile or signed baseline."""
    inventory = _sources(repository)
    for name, source in {
        "known_variant.py": "# vocabulary fixture: offset two\ndef increment(n: int) -> int:\n    return n + 2\n",
        "tune.py": "# fixed tuning fixture\ndef increment(n: int) -> int:\n    return n + 1\n",
        "canary.py": "# fixed diagnostic fixture\ndef increment(n: int) -> int:\n    return n + 1\n",
    }.items():
        (repository / name).write_bytes(source.encode("ascii"))
        inventory[name] = source
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixed advisory source cohort before owner baseline")
    _need(set(inventory) == PATHS and set(_git(repository, "ls-files").splitlines()) == PATHS
          and not _git(repository, "status", "--porcelain"), "complete clean ten-file cohort required")
    return inventory


def _bridge_expected(*, admission, reviewed_candidate, generated_candidate, worker_candidate,
        replacement_bytes, expected_model_binding, expected_task_cid):
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as finite
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate as proposal
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate_runner as worker
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import OFFSET_STATEMENT_ID
    verified = finite.verify_finite_repository_admission(admission=admission)
    review = proposal.verify_finite_repository_candidate(admission=admission, candidate=reviewed_candidate)
    proposal.verify_generated_finite_repository_candidate(record=generated_candidate)
    worker._validate(worker_candidate)
    semantic = verified["semantic_context"]
    binding = expected_model_binding
    _need(type(binding) is dict and binding.get("schema") == "supervisor-codebase-feature-context@1"
          and binding.get("mode") == "train" and binding.get("model_enabled") is True
          and type(binding.get("version_id")) is str and bool(binding["version_id"])
          and type(binding.get("latent_width")) is int and binding["latent_width"] == 8
          and binding.get("parameter_dtype") == "float64"
          and type(binding.get("actual_training_delta")) is int and binding["actual_training_delta"] == 16
          and type(binding.get("authority")) is dict and binding["authority"]
          and all(value is False for value in binding["authority"].values())
          and binding.get("context_cid") == cid_for_structured({key: value for key, value in binding.items() if key != "context_cid"})
          and _same(binding.get("head"), semantic["head"]), "bridge requires the actual verified initial model binding")
    _need(type(expected_task_cid) is str and expected_task_cid
          and expected_task_cid == semantic["native_task_bindings"][OFFSET_STATEMENT_ID]["task_cid"]
          and expected_task_cid == review["payload"]["task_cid"] == generated_candidate["task_cid"] == worker_candidate["task_cid"]
          and _same(generated_candidate["parent_admission"], admission)
          and _same(generated_candidate["reviewed_candidate"], reviewed_candidate)
          and _same(worker_candidate["finite_admission"], admission)
          and _same(generated_candidate["head"], semantic["head"])
          and _same(semantic["administrator_task_cids"], review["payload"]["administrator_task_cids"]),
          "bridge retargeted an original admission, source generation, model or native task")
    _need(type(replacement_bytes) is bytes and 0 < len(replacement_bytes) <= 65536,
          "exact bounded generated replacement bytes required")
    generated_path = Path(generated_candidate["artifacts"]["replacement"]["path"])
    observed = proposal._read(generated_path, 65536)
    edit = worker_candidate["edit"]
    worker_after = base64.b64decode(edit["after_bytes_base64"], validate=True)
    _need(observed == replacement_bytes == worker_after
          and cid_for_bytes(observed) == generated_candidate["replacement_cid"] == review["payload"]["after_cid"]
          and hashlib.sha256(observed).hexdigest() == edit["after_sha256"] == review["payload"]["after_sha256"]
          and worker_candidate["semantic_context_cid"] == cid_for_structured(semantic)
          and type(worker_candidate["task_revision"]) is int and worker_candidate["task_revision"] >= 1
          and all(type(value["training_steps"]) is int and value["training_steps"] == 0
              and type(value["provider_calls"]) is int and value["provider_calls"] == 0
              for value in (generated_candidate, worker_candidate)),
          "bridge differs from retained generated bytes or gained candidate fitting/provider calls")
    result = {"schema": BRIDGE_SCHEMA, "scope": BRIDGE_SCOPE,
        "admission_cid": cid_for_structured(admission), "reviewed_candidate_cid": cid_for_structured(reviewed_candidate),
        "generated_result_cid": generated_candidate["result_cid"], "worker_candidate_cid": worker_candidate["candidate_cid"],
        "head": semantic["head"], "source_cid": semantic["source_cid"],
        "replacement_cid": cid_for_bytes(observed), "replacement_sha256": hashlib.sha256(observed).hexdigest(),
        "task_cid": expected_task_cid, "task_revision": worker_candidate["task_revision"],
        "administrator_task_cids": semantic["administrator_task_cids"], "model_binding": binding,
        "model_context_cid": binding["context_cid"], "model_version_id": binding["version_id"],
        "training_steps": 0, "provider_calls": 0, **CLAIMS}
    result["bridge_cid"] = cid_for_structured(result)
    return result


def build_advisory_worker_bridge(**arguments):
    """Join verified advisory bytes to a separately signed native handoff."""
    return _bridge_expected(**arguments)


def validate_advisory_worker_bridge(bridge, **arguments):
    """Exact historical closure, never current launch or authenticated origin."""
    _need(type(bridge) is dict and set(bridge) == BRIDGE_FIELDS
          and bridge.get("schema") == BRIDGE_SCHEMA and bridge.get("scope") == BRIDGE_SCOPE
          and all(bridge.get(key) is False for key in CLAIMS)
          and bridge.get("bridge_cid") == cid_for_structured({key: value for key, value in bridge.items() if key != "bridge_cid"}),
          "closed nonauthoritative advisory bridge required")
    expected = _bridge_expected(**arguments)
    _need(_same(bridge, expected), "bridge differs from independently replayed complete original context")
    return expected


class AdvisoryWorkerSupport:
    """Owner-private advisory stages; actual worker scope belongs to the caller."""

    def __init__(self, *, output):
        self.output = Path(output).absolute()
        _need(self.output.resolve(strict=True) == self.output and (self.output / "private").is_dir(),
              "existing canonical owner-private output required")
        self.root = self.output / "private/advisory"
        self.root.mkdir(mode=0o700)
        self.registry = None
        self.phases, self.controls, self.records, self.previews, self.contexts = [], [], {}, [], []
        self.parent_pins, self.root_measure, self.child_measure = [], None, None
        self.root_context = self.frozen = self.child = self.child_frozen = None
        self.reviewed = self.generated = self.bridge = self.worker_candidate = None
        self.owner = self.tools = self.contract = None
        _write(self.root / "criteria.json", {"schema": SCHEMA, "configuration": CONFIGURATION,
            "selections": SELECTIONS, "scope": "ten-file authored source-cohort transductive structural reconstruction",
            "training_mode": "explicit root and child native feature training; finite signed admission remains model_off",
            "canary_scope": "fixed repeated post-selection diagnostic, not unseen qualification", "claims": CLAIMS})
        _write(self.root / "authority-materials.json", _authority_materials())
        (self.root / "prompt.txt").write_bytes(INTENT.encode())
        self._costs()

    def _training_cost(self):
        """Observed checkpoint lower bound, including pre-context final failures."""
        checkpoints, errors = {}, []
        artifact_root = self.root / "model-artifacts"
        if artifact_root.is_dir():
            for path in sorted(artifact_root.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    pin = _pin(path)
                    raw = path.read_bytes()
                    _need(hashlib.sha256(raw).hexdigest() == pin["sha256"] == path.name,
                          "native model checkpoint digest/path differs")
                    candidate = json.loads(raw)
                    report = candidate.get("report", {})
                    request_id = report.get("codebase_request_sha256")
                    if request_id is None:
                        continue
                    _need(type(request_id) is str and type(report.get("attempted_epochs")) is int
                          and 1 <= report["attempted_epochs"] <= 32
                          and report.get("codebase_worker_receipt", {}).get("returncode") == 0,
                          "native retained training receipt/count differs")
                    previous = checkpoints.get(request_id)
                    _need(previous is None or previous["artifact"] == pin,
                          "one native training request has conflicting checkpoints")
                    checkpoints[request_id] = {"request_sha256": request_id, "artifact": pin,
                        "attempted_epochs": report["attempted_epochs"],
                        "head": report["codebase_provenance"]["head"],
                        "parent_version_id": report["codebase_provenance"]["parent_version_id"],
                        "worker_receipt": report["codebase_worker_receipt"]}
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append({"path": str(path), "error": repr(error)})
        known = sum(context.actual_training_delta for context in (self.root_context, self.child) if context is not None)
        observed = sum(value["attempted_epochs"] for value in checkpoints.values())
        return {"scope": "distinct observed native checkpoint requests; lower bound if fitting failed before retention",
            "known_completed_context_epochs": known, "checkpoint_observed_epochs": observed,
            "observed_epoch_lower_bound": max(known, observed), "checkpoints": list(checkpoints.values()),
            "unknown_unretained_training_attempts_excluded_from_exact_claim": True, "errors": errors}

    @property
    def actual_training_epochs(self):
        return self._training_cost()["observed_epoch_lower_bound"]

    def _costs(self):
        _write(self.root / "phase-costs.json", {"schema": "finite-advisory-worker-phase-costs@1",
            "phases": self.phases, "training_cost": self._training_cost(),
            "scope": "actual calls including refusals; duplicate test errors are not independent training attempts"},
            replace_existing=True)

    def _phase(self, name, operation):
        try:
            return _phase(self.phases, name, operation)
        finally:
            self._costs()

    def _extend(self, values):
        append_metadata_occurrences(self.records, values)

    def _reject(self, label, operation):
        try:
            self._phase("control:" + label, operation)
        except ValueError as error:
            self.controls.append({"control": label, "rejected": True,
                "error_type": type(error).__name__, "error": str(error)})
        else:
            raise ValueError("advisory control accepted: " + label)

    def _previews(self, owner, prefix, contexts, *, facts, selected):
        projected = []
        for name, context in contexts:
            result, request = self._phase(prefix + "_" + name + "_capacity_preview", lambda:
                _preview(owner.index, owner.repository, owner.expected_head, owner.scheduler,
                    self.registry, context, self.tools, self.root / (prefix + "-" + name + "-preview-artifacts")))
            _check_preview(result, facts=facts, selected=selected, feature=context)
            projected.append(finite_outcome_projection(result))
            self.previews.append(result)
            _write(self.root / (prefix + "-" + name + "-preview.json"), result)
            _write(self.root / (prefix + "-" + name + "-request.json"), request.to_dict())
            _idle(owner.scheduler)
        _need(all(_same(item, projected[0]) for item in projected),
              "advisory off/train/frozen changed a complete finite outcome")
        self._extend({"preview_invariants": [{"schema": "finite-advisory-worker-preview-invariants@1",
            "phase": prefix, "modes": [name for name, _ in contexts], "finite_outcome": projected[0],
            "finite_outcomes_identical": True, "training_steps_during_preview": 0,
            "planning_model_calls": 0, **CLAIMS}]})

    def prepare_initial(self, *, owner, tool_policy):
        from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
        from ipfs_datasets_py.logic.software_contracts.codebase_source_training import CodebaseTrainingSelection
        from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
        from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import prove_current_integer_offset_lowering
        from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import prepare_codebase_feature_context
        _need(self.registry is None and self.owner is None, "initial advisory preparation may execute once")
        self.owner, self.tools = owner, deepcopy(tool_policy)
        self.initial_head = owner.expected_head
        self.initial_git = _git(owner.repository, "rev-parse", "HEAD")
        self.source_before = {name: _pin(owner.repository / name) for name in sorted(PATHS)}
        self.contract = IntegerOffsetContract(path="calc.py", function_name="increment", parameter="n", offset=2)
        self.selections = tuple(CodebaseTrainingSelection(path, role, contracts=()) for path, role in SELECTIONS)
        self.registry = AutoencoderRegistry(self.root / "train.duckdb", self.root / "model-artifacts")
        self._extend(self._phase("initial_complete_capture", lambda:
            _capture_complete(owner.index, owner.repository, owner.expected_head, owner.scheduler)))
        off = self._phase("root_model_off", lambda: prepare_codebase_feature_context(owner=owner,
            registry=self.registry, mode="model_off", output=self.root / "root-off-context"))
        self.root_context = self._phase("root_training", lambda: prepare_codebase_feature_context(owner=owner,
            registry=self.registry, mode="train", output=self.root / "root-training-context",
            selections=self.selections, operation_id="advisory-worker-root", **CONFIGURATION))
        self._costs()
        self.root_measure = _measure(self.root_context)
        self.root_measure["metrics_scope"] = "Four selected source files in a ten-file authored repository; source-cohort transductive reconstruction diagnostics."
        native_before = _registry_inventory(self.registry)
        self.frozen = self._phase("root_frozen", lambda: prepare_codebase_feature_context(owner=owner,
            registry=self.registry, mode="frozen", output=self.root / "root-frozen-context", version_id=self.root_context.version_id))
        _need(_registry_inventory(self.registry) == native_before, "root frozen preparation fitted or promoted a model")
        self.contexts.extend((off, self.root_context, self.frozen))
        self.before_lowering = self._phase("initial_ast_lowering", lambda: _with_custody(owner, lambda:
            prove_current_integer_offset_lowering(owner.index, owner.repository, expected_head=owner.expected_head,
                contract=self.contract, tool_policy=self.tools, output=self.root / "before-lowering-artifacts",
                scheduler=owner.scheduler, timeout_seconds=90, memory_mb=1024)))
        _lowering_scope(self.before_lowering, matches=False)
        _write(self.root / "before-lowering.json", self.before_lowering.to_dict())
        self._previews(owner, "root", (("off", off), ("train", self.root_context), ("frozen", self.frozen)),
            facts=1, selected=["task:finite:offset"])
        self._check_sources(owner)
        stage = {"schema": SCHEMA, "head": owner.expected_head.to_dict(),
            "root_measurement": self.root_measure, "lowering_cid": self.before_lowering.cid, **CLAIMS}
        _write(self.root / "initial-stage.json", stage)
        return stage

    def _check_sources(self, owner):
        _need({name: _pin(owner.repository / name) for name in sorted(PATHS)} == self.source_before
              and _git(owner.repository, "rev-parse", "HEAD") == self.initial_git
              and not _git(owner.repository, "status", "--porcelain"), "advisory proposal changed canonical source or Git head")

    def candidate_bytes(self, *, owner, admission, task_snapshot):
        from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate as proposal
        from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import verify_current_context
        _need(self.root_context is not None and self.generated is None and callable(task_snapshot),
              "prepared root, genuine native task snapshot and one candidate operation required")
        verify_current_context(owner, self.registry, self.frozen)
        task_before = _plain(task_snapshot())
        model_before = _registry_inventory(self.registry)
        self.reviewed = self._phase("reviewed_proposal_authoring", lambda:
            proposal.author_finite_repository_candidate(owner=owner, admission=admission,
                review_ref="review:authored-advisory-worker-exact-offset-replacement"))
        self.generated = self._phase("bounded_proposal_generation", lambda:
            proposal.generate_finite_repository_candidate(owner=owner, admission=admission,
                candidate=self.reviewed, output=self.root / "candidate-artifacts",
                policy_observer=lambda bound: bound.roots))
        proposal.verify_finite_repository_candidate(admission=admission, candidate=self.reviewed)
        proposal.verify_generated_finite_repository_candidate(record=self.generated)
        self.replacement = proposal._read(Path(self.generated["artifacts"]["replacement"]["path"]), 65536)
        _need(cid_for_bytes(self.replacement) == self.generated["replacement_cid"], "generated replacement identity differs")
        _need(wire(task_before) == wire(_plain(task_snapshot())) and _registry_inventory(self.registry) == model_before,
              "advisory candidate altered complete native task rows or model registry")
        self._check_sources(owner)
        _write(self.root / "reviewed-candidate.json", self.reviewed)
        _write(self.root / "generated-candidate.json", self.generated)
        self._extend({"candidate_invariants": [{"task_rows_before": task_before, "task_rows_after": _plain(task_snapshot()),
            "registry_before": model_before, "registry_after": _registry_inventory(self.registry),
            "complete_native_task_population_unchanged": True, "canonical_source_unchanged": True, **CLAIMS}]})
        return self.replacement

    def bind_worker_descriptor(self, *, admission, candidate, task_cid):
        from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import verify_current_context
        from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner import load_finite_repository_candidate
        _need(self.generated is not None and self.bridge is None, "one generated candidate and one native handoff join required")
        _need(type(candidate) is dict and type(candidate.get("artifact")) is str
              and type(candidate.get("sha256")) is str, "actual readonly native worker handoff descriptor required")
        loaded = load_finite_repository_candidate(artifact=Path(candidate["artifact"]), expected_sha256=candidate["sha256"])
        _need(loaded["candidate_cid"] == candidate["candidate_cid"] and loaded["task_cid"] == task_cid,
              "native descriptor differs from loaded full public candidate")
        def operation():
            verify_current_context(self.owner, self.registry, self.root_context)
            arguments = dict(admission=admission, reviewed_candidate=self.reviewed, generated_candidate=self.generated,
                worker_candidate=loaded, replacement_bytes=self.replacement,
                expected_model_binding=self.root_context.material_binding, expected_task_cid=task_cid)
            bridge = build_advisory_worker_bridge(**arguments)
            validate_advisory_worker_bridge(bridge, **arguments)
            return bridge
        self.bridge = self._phase("native_worker_descriptor_join", lambda: _with_custody(self.owner, operation))
        self.worker_candidate = _plain(loaded)
        self.worker_descriptor = _plain(candidate)
        self.parent_admission = _plain(admission)
        _write(self.root / "worker-bridge.json", self.bridge)
        self.parent_pins = _owned_file_pins(self.output / "cas", self.root / "model-artifacts",
            self.root / "root-off-context", self.root / "root-training-context", self.root / "root-frozen-context",
            self.root / "before-lowering-artifacts", self.root / "candidate-artifacts",
            *(self.root / ("root-" + name + "-preview-artifacts") for name in ("off", "train", "frozen")))
        self.parent_pins.extend(_pin(self.root / name) for name in
            ("criteria.json", "authority-materials.json", "prompt.txt", "before-lowering.json",
                "reviewed-candidate.json", "generated-candidate.json", "worker-bridge.json"))
        self.parent_pins.append(_pin(candidate["artifact"]))
        # Selected callable producers are immutable sidecar pins. The native
        # execution scope still derives authority only from signed finite rows.
        from .finite_repository_candidate_experiment import _SOURCES
        self.parent_pins.extend(_pin(importlib.import_module(module).__file__) for module in (MODULE,
            "benchmarks.agent_supervisor.container_coding.finite_repository_sharded_metadata", *_SOURCES))
        return self.bridge

    def check_historical_pins(self):
        """Detached bounded bytes only; safe while a native worktree is allocated."""
        _need(self.bridge is not None and self.parent_pins, "prepared native/advisory sidecar required")
        for pin in self.parent_pins:
            _need(_pin(pin["path"]) == pin, "closed advisory artifact, checkpoint or selected producer changed")
        return {"schema": "finite-advisory-worker-historical-byte-fence@1", "pins": len(self.parent_pins),
            "historical_bytes_unchanged": True, "current_source_observed": False,
            "signed_execution_scope_upgraded": False, **CLAIMS}

    def check_prelaunch(self, *, owner):
        from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import verify_current_context
        from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import validate_current_integer_offset_lowering
        _need(self.bridge is not None and self.registry is not None, "prepared advisory sidecar required before allocation")
        before = _registry_inventory(self.registry)
        def operation():
            verify_current_context(owner, self.registry, self.frozen)
            validated = validate_current_integer_offset_lowering(self.before_lowering, owner.index, owner.repository,
                expected_head=owner.expected_head, contract=self.contract, tool_policy=self.tools,
                scheduler=owner.scheduler, timeout_seconds=90, memory_mb=1024)
            _lowering_scope(validated, matches=False)
            self.check_historical_pins()
        self._phase("advisory_preallocation_check", lambda: _with_custody(owner, operation))
        _need(_registry_inventory(self.registry) == before, "prelaunch advisory validation fitted or changed model rows")
        # Registry storage is fenced physically through the actual worker. Its
        # constructor/reopen generation is mutable only after this phase ends.
        self.worker_registry_pins = [_pin(path) for path in
            (self.root / "train.duckdb", self.root / "train.duckdb.wal") if path.exists()]
        self.parent_pins.extend(self.worker_registry_pins)
        return {"schema": "finite-advisory-worker-preallocation-check@1", "head": owner.expected_head.to_dict(),
            "model_context_cid": self.frozen.cid, "lowering_record_cid": self.before_lowering.cid,
            "training_steps": 0, "signed_execution_scope_upgraded": False, **CLAIMS}

    def prepare_successor(self, *, owner):
        from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import prepare_codebase_feature_context, verify_current_context
        from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import prove_current_integer_offset_lowering, validate_current_integer_offset_lowering
        _need(self.bridge is not None and self.child is None
              and owner.expected_head.repository_id == self.initial_head.repository_id
              and owner.expected_head.generation == self.initial_head.generation + 1
              and owner.expected_head.snapshot_cid != self.initial_head.snapshot_cid,
              "actual published successor and preserved original repository required")
        published = _git(owner.repository, "rev-parse", "HEAD")
        parents = _git(owner.repository, "rev-list", "--parents", "-n", "1", published).split()
        _need(len(parents) == 3 and parents[1] == self.initial_git
              and _git(owner.repository, "diff", "--name-only", self.initial_git, published).splitlines() == ["calc.py"]
              and (owner.repository / "calc.py").read_bytes() == self.replacement
              and not _git(owner.repository, "status", "--porcelain"),
              "advisory successor must be the real one-output baseline merge, not an owner-applied edit")
        _need(all(_pin(owner.repository / name) == pin for name, pin in self.source_before.items() if name != "calc.py"),
              "publication changed fixed evaluation or supporting sources")
        for pin in self.parent_pins:
            _need(_pin(pin["path"]) == pin, "worker publication altered immutable advisory parent evidence")
        # Subsequent explicit training legitimately changes registry control
        # storage; immutable checkpoint/context/producer pins remain unchanged.
        self.parent_pins = [pin for pin in self.parent_pins if pin not in getattr(self, "worker_registry_pins", ())]
        self._reject("root_context_after_publication", lambda: verify_current_context(self.owner, self.registry, self.frozen))
        self._reject("root_context_under_successor", lambda: verify_current_context(owner, self.registry, self.frozen))
        self._reject("root_lowering_under_successor", lambda: _with_custody(owner, lambda:
            validate_current_integer_offset_lowering(self.before_lowering, owner.index, owner.repository,
                expected_head=owner.expected_head, contract=self.contract, tool_policy=self.tools,
                scheduler=owner.scheduler, timeout_seconds=90, memory_mb=1024)))
        self.successor_head, self.published_git = owner.expected_head, published
        self._extend(self._phase("successor_complete_capture", lambda:
            _capture_complete(owner.index, owner.repository, owner.expected_head, owner.scheduler)))
        off = self._phase("child_model_off", lambda: prepare_codebase_feature_context(owner=owner,
            registry=self.registry, mode="model_off", output=self.root / "child-off-context"))
        self.child = self._phase("child_training", lambda: prepare_codebase_feature_context(owner=owner,
            registry=self.registry, mode="train", output=self.root / "child-training-context",
            selections=self.selections, operation_id="advisory-worker-child", parent_version_id=self.root_context.version_id,
            **CONFIGURATION))
        self._costs()
        self.child_measure, self.continuation = _measure(self.child), _continuation(self.root_context, self.child)
        self.child_measure["metrics_scope"] = self.root_measure["metrics_scope"]
        native_before = _registry_inventory(self.registry)
        self.child_frozen = self._phase("child_frozen", lambda: prepare_codebase_feature_context(owner=owner,
            registry=self.registry, mode="frozen", output=self.root / "child-frozen-context", version_id=self.child.version_id))
        _need(_registry_inventory(self.registry) == native_before, "child frozen preparation fitted or promoted a model")
        self.contexts.extend((off, self.child, self.child_frozen))
        self.successor_lowering = self._phase("successor_ast_lowering", lambda: _with_custody(owner, lambda:
            prove_current_integer_offset_lowering(owner.index, owner.repository, expected_head=owner.expected_head,
                contract=self.contract, tool_policy=self.tools, output=self.root / "successor-lowering-artifacts",
                scheduler=owner.scheduler, timeout_seconds=90, memory_mb=1024)))
        _lowering_scope(self.successor_lowering, matches=True)
        _write(self.root / "successor-lowering.json", self.successor_lowering.to_dict())
        self._previews(owner, "child", (("off", off), ("train", self.child), ("frozen", self.child_frozen)), facts=2, selected=[])
        stage = {"schema": SCHEMA, "head": owner.expected_head.to_dict(),
            "child_measurement": self.child_measure, "continuation": self.continuation,
            "lowering_cid": self.successor_lowering.cid, "controls": self.controls, **CLAIMS}
        _write(self.root / "successor-stage.json", stage)
        return stage

    def finish(self, *, owner, worker_report):
        from .finite_repository_sharded_metadata import (
            hydrate_sharded_codebase_ir_metadata, validate_sharded_codebase_ir_metadata,
            reconstruct_sharded_codebase_ir_metadata_records,
        )
        from .terminal_codebase_supervisor_fixture import bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records
        _need(self.child_frozen is not None and _same(owner.expected_head.to_dict(), self.successor_head.to_dict()),
              "complete actual successor advisory stages required")
        request = {"schema": SCHEMA, "output": str(self.output), "head": self.successor_head.to_dict(),
            "tool_policy": self.tools, "model_version_id": self.child.version_id,
            "child_binding": self.child_frozen.material_binding, "retained_pins": self.parent_pins,
            "worker_candidate": self.worker_candidate, "bridge": self.bridge,
            "root_binding": self.root_context.material_binding, "task_cid": self.bridge["task_cid"]}
        _write(self.root / "cold-request.json", request)
        self.close()
        process = self._phase("same_head_fresh_process_cold", lambda: subprocess.run(
            [sys.executable, "-m", MODULE, "cold", str(self.output)], capture_output=True, text=True, timeout=300))
        _write(self.root / "cold-process.json", {"returncode": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr})
        _need(process.returncode == 0, "exact-head no-fit cold replay failed: " + process.stderr[-4096:])
        cold = json.loads((self.root / "cold-response.json").read_bytes())
        self._extend(json.loads((self.root / "cold-capture.json").read_bytes()))
        if (self.output / "cold-capture.json").is_file():
            self._extend(json.loads((self.output / "cold-capture.json").read_bytes()))
        numerical = [{"binding": context.material_binding, "retained": context.retained_artifacts} for context in self.contexts]
        artifact_dirs = [self.output / "cas", self.root / "model-artifacts",
            *(context.output for context in self.contexts),
            self.root / "before-lowering-artifacts", self.root / "successor-lowering-artifacts", self.root / "candidate-artifacts",
            *(self.root / (prefix + "-" + name + "-preview-artifacts") for prefix in ("root", "child") for name in ("off", "train", "frozen")),
            self.root / "cold-frozen-preview-artifacts"]
        pins = _owned_file_pins(*artifact_dirs)
        native_events = capture_selected_worker_event_artifacts(self.output)
        self._extend({"artifacts": _artifact_bytes(pins), "contracts": [self.contract.to_dict()],
            "feature_contexts": numerical, "training": [self.root_measure, self.child_measure, self.continuation, *numerical],
            "vectors": [{"binding": item["binding"], "inference": item["retained"].get("inference")}
                for item in numerical if item["binding"]["mode"] != "model_off"],
            "lowering_proofs": [self.before_lowering.to_dict(), self.successor_lowering.to_dict()],
            "reviewed_candidates": [self.reviewed, self.generated, self.worker_candidate], "worker_bridge": [self.bridge],
            "capacity_previews": self.previews, "finite_matches": [item["match"] for item in self.previews],
            "current_facts": [fact for item in self.previews for fact in item["match"]["current_facts"]],
            "native_worker_events": [*native_events, {"schema": "finite-advisory-worker-native-report-join@1", "record": _plain(worker_report)}],
            "controls": self.controls, "cold_verification": [cold],
            "criteria": [json.loads((self.root / "criteria.json").read_bytes()), {"tools": self.tools, "phases": self.phases}]})
        complete_raw = wire(self.records) + b"\n"
        _need(len(complete_raw) <= 256 * 1024**2,
              "complete producer archive exceeds its separate bounded recovery allowance")
        with (self.root / "complete-producer-records.json").open("xb") as stream:
            stream.write(complete_raw)
        bounded = bound_terminal_codebase_metadata_records(self.records)
        _need(wire(reconstruct_terminal_codebase_metadata_records(bounded)) == wire(self.records),
              "metadata chunks lost a complete advisory/worker producer occurrence")
        metadata = self._phase("duckdb_ducklake_hydration", lambda: hydrate_sharded_codebase_ir_metadata(records=bounded,
            output=self.root / "metadata", source_snapshot={"schema": SCHEMA,
                "heads": [self.initial_head.to_dict(), self.successor_head.to_dict()],
                "original_git_head": self.initial_git, "published_git_head": self.published_git,
                "producer_sha256": digest(self.records), "criteria": _pin(self.root / "criteria.json")}))
        replay = self._phase("duckdb_ducklake_fresh_process_readback", lambda:
            validate_sharded_codebase_ir_metadata(output=self.root / "metadata", expected=metadata, fresh_process=True))
        recovered = reconstruct_sharded_codebase_ir_metadata_records(output=self.root / "metadata", expected=metadata)
        _need(wire(reconstruct_terminal_codebase_metadata_records(recovered)) == wire(self.records),
              "native metadata readback changed complete advisory/worker records")
        for pin in self.parent_pins:
            _need(_pin(pin["path"]) == pin, "cold verification or metadata hydration changed historical advisory evidence")
        _idle(owner.scheduler)
        result = {"schema": SCHEMA, "status": "completed", "original_head": self.initial_head.to_dict(),
            "successor_head": self.successor_head.to_dict(), "root_measurement": self.root_measure,
            "child_measurement": self.child_measure, "continuation": self.continuation,
            "actual_attempted_training_epochs": self.root_context.actual_training_delta + self.child.actual_training_delta,
            "worker_bridge": self.bridge, "lowering_proofs": [self.before_lowering.to_dict(), self.successor_lowering.to_dict()],
            "cold_verification": cold, "complete_family_counts": {name: len(rows) for name, rows in self.records.items()},
            "packaged_family_counts": {name: len(rows) for name, rows in bounded.items()},
            "metadata": metadata, "metadata_replay": replay, "controls": self.controls,
            "historical_parent_preserved": True, "active_leases": 0, "waiting_requests": 0,
            "claims": CLAIMS, **CLAIMS}
        _write(self.root / "result.json", result)
        return result

    def close(self):
        try:
            if self.registry is not None:
                self.registry.close()
                self.registry = None
        finally:
            self._costs()


def cold_verify(output):
    """Reopen the exact original catalog; no new catalog/head/model rebinding."""
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import verify_current_context
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import IntegerOffsetLoweringProof, validate_current_integer_offset_lowering
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    output = Path(output).absolute()
    root = output / "private/advisory"
    request = json.loads((root / "cold-request.json").read_bytes())
    _need(request["output"] == str(output), "cold request namespace differs")
    scheduler = _scheduler(root / "cold-resource-admission.json")
    connection, index = _open(output)
    registry = AutoencoderRegistry(root / "train.duckdb", root / "model-artifacts")
    try:
        owner = RepositoryPlanPreviewOwner(index=index, repository=output / "repository",
            expected_head=CodebaseHead.from_dict(request["head"]), scheduler=scheduler, timeout_seconds=90, memory_mb=1024)
        for pin in request["retained_pins"]:
            _need(_pin(pin["path"]) == pin, "advisory historical evidence changed before cold replay")
        baseline = _registry_inventory(registry)  # Constructor advances owner generation; fitting does not occur here.
        context = _context(root / "child-frozen-context")
        binding = verify_current_context(owner, registry, context)
        _need(_same(binding, request["child_binding"]) and context.version_id == request["model_version_id"],
              "cold replay selected another current model context")
        proof = IntegerOffsetLoweringProof.from_dict(json.loads((root / "successor-lowering.json").read_bytes()))
        result = _with_custody(owner, lambda: validate_current_integer_offset_lowering(proof, index, owner.repository,
            expected_head=owner.expected_head, contract=IntegerOffsetContract.from_dict(proof.to_dict()["contract"]),
            tool_policy=request["tool_policy"], scheduler=scheduler, timeout_seconds=90, memory_mb=1024))
        _lowering_scope(result, matches=True)
        reviewed = json.loads((root / "reviewed-candidate.json").read_bytes())
        generated = json.loads((root / "generated-candidate.json").read_bytes())
        admission = json.loads((output / "before-admission.json").read_bytes())
        validate_advisory_worker_bridge(request["bridge"], admission=admission, reviewed_candidate=reviewed,
            generated_candidate=generated, worker_candidate=request["worker_candidate"],
            replacement_bytes=base64.b64decode(request["worker_candidate"]["edit"]["after_bytes_base64"], validate=True),
            expected_model_binding=request["root_binding"], expected_task_cid=request["task_cid"])
        preview, fresh_request = _preview(index, owner.repository, owner.expected_head, scheduler, registry,
            context, request["tool_policy"], root / "cold-frozen-preview-artifacts")
        _check_preview(preview, facts=2, selected=[], feature=context)
        _write(root / "cold-frozen-preview.json", preview)
        _write(root / "cold-capture.json", _capture_complete(index, owner.repository, owner.expected_head, scheduler))
        _need(_registry_inventory(registry) == baseline, "cold replay fitted, promoted or changed immutable model evidence")
        for pin in request["retained_pins"]:
            _need(_pin(pin["path"]) == pin, "cold replay changed retained historical advisory evidence")
        _idle(scheduler)
        response = {"schema": "finite-advisory-worker-exact-head-cold-replay@1", "status": "completed",
            "head": owner.expected_head.to_dict(), "model_version_id": context.version_id,
            "model_context_cid": context.cid, "lowering_record_cid": result.cid,
            "registry_before": baseline, "registry_after": _registry_inventory(registry),
            "fresh_request": fresh_request.to_dict(), "current_facts_count": 2, "selected_task_ids": [],
            "training_steps": 0, "active_leases": 0, "waiting_requests": 0,
            "same_durable_catalog_reopened": True, "independent_cold_catalog_rebinding_claimed": False, **CLAIMS}
        _write(root / "cold-response.json", response)
        return response
    finally:
        registry.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cold",))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = cold_verify(args.output)
    print(json.dumps({"schema": result["schema"], "status": result["status"]}))


if __name__ == "__main__":
    main()
