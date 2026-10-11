"""Refresh explicitly admitted learned numerical advice after publication.

The original checkpoint, program subset and full captured source population are
fixed before START. New advice remains a candidate; another launch requires an
independent manifest. Native inference can truthfully abstain on changed code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

from ..proof.formal_verification_contracts import content_identity
from . import numerical_selection_context as native
from .benchmark_budget_policy import source384_timeout_limit
from .doctor_candidate_runner import _git
from .security_autoencoder_advisor import _bytes, _sha, _read

POLICY_SCHEMA = "admitted-learned-numerical-successor-policy@1"
OBSERVATION_SCHEMA = "supervisor-published-learned-numerical-refresh@1"
EXECUTION_SCHEMA = "supervisor-learned-numerical-refresh-execution@1"
AUTHORITY = dict(proof_authority=False, formalization_authority=False,
    planning_authority=False, dispatch_authority=False, execution_authority=False,
    completion_authority=False, source_semantics_verified=False,
    whole_program_semantics_verified=False, source_obligations_discharged=0,
    training_steps=0, provider_calls=0, download_calls=0)
ASSET_FIELDS = ("config_path", "config_sha256", "checkpoint_file_sha256")
MODEL_FIELDS = ("checkpoint_canonical_sha256", "head_sha256", "projection_sha256")
POLICY_FIELDS = {"schema", "profile", "repository", "manifest_cid", "baseline_commit",
    "predecessor_receipt_sha256", "source_hashes", "selected_paths", "producers", *ASSET_FIELDS,
    *MODEL_FIELDS, "successor_helper_sha256", "source_capture_exclusions", "output_root",
    "excluded_new_output_paths", "budget_profile", "preparation_timeout_seconds",
    "requires_independent_manifest", *AUTHORITY}


def _equal(left, right):
    return _bytes(left) == _bytes(right)


def _helper_digest():
    return _sha(_read(Path(__file__).resolve(), 1_000_000))


def _deadline(policy, deadline):
    if (type(policy) is not dict or type(policy.get("preparation_timeout_seconds")) is not int
            or policy["preparation_timeout_seconds"] != source384_timeout_limit(policy.get("budget_profile"))):
        raise ValueError("bounded learned numerical successor preparation policy required")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("finite learned numerical successor deadline required")
    end = min(time.monotonic() + policy["preparation_timeout_seconds"], deadline if deadline is not None else float("inf"))
    def remaining():
        left = end - time.monotonic()
        if left <= 0:
            raise TimeoutError("learned numerical successor deadline expired")
        return left
    remaining()
    return remaining, end


def _population(hashes, selected):
    if type(hashes) is not dict or not 1 <= len(hashes) <= 256:
        raise ValueError("bounded learned numerical source population required")
    for name, digest in hashes.items():
        if (type(name) is not str or name in {"", ".", ".."} or Path(name).is_absolute() or Path(name).as_posix() != name
                or any(c in name for c in "\\\n\r\0")
                or any(p in {"..", ".git", ".runtime"} for p in Path(name).parts)
                or type(digest) is not str or re.fullmatch(r"[a-f0-9]{64}", digest) is None):
            raise ValueError("canonical learned numerical source population required")
    if (type(selected) is not list or not 1 <= len(selected) <= 256
            or any(type(name) is not str or name not in hashes for name in selected)
            or selected != sorted(set(selected))):
        raise ValueError("exact learned numerical program subset required")


def _capture_exclusions(receipt):
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    with native.source_owner._owners(Path(receipt["output"])) as (index, _registry):
        head = CodebaseHead.from_dict(receipt["source_head"])
        return sorted(index.load(head.manifest_cid).snapshot.exclusions)


def validate_manifest(*, policy, manifest):
    _population(policy["source_hashes"], policy["selected_paths"])
    if (type(manifest) is not dict or type(manifest.get("sources")) is not dict
            or policy["manifest_cid"] != content_identity(manifest)
            or policy["baseline_commit"] != manifest.get("baseline_commit")
            or policy["source_hashes"] != {name: row["sha256"] for name, row in manifest["sources"].items()}):
        raise ValueError("learned numerical successor differs from admitted manifest sources")
    excluded = sorted({row["path"] for task in manifest["tasks"] for row in task["outputs"]}
        - set(policy["source_hashes"]))
    native.source_owner._successor_exclusions(excluded, policy["source_hashes"])
    if not _equal(policy["excluded_new_output_paths"], excluded):
        raise ValueError("learned numerical excluded outputs differ from admitted manifest")
    return policy


def bind_policy(*, repository, receipt, manifest, output_root, budget_profile=None):
    root = Path(repository).resolve(strict=True)
    native.validate_numerical_selection_context(repository=root, expected_receipt=receipt,
        timeout_seconds=source384_timeout_limit(budget_profile), budget_profile=budget_profile)
    output = native.source_owner._output(root, output_root)
    excluded = sorted({row["path"] for task in manifest["tasks"] for row in task["outputs"]}
        - set(receipt["source_hashes"]))
    policy = dict(schema=POLICY_SCHEMA, profile=receipt["profile"], repository=str(root),
        manifest_cid=content_identity(manifest), baseline_commit=manifest["baseline_commit"],
        predecessor_receipt_sha256=receipt["receipt_sha256"], source_hashes=dict(receipt["source_hashes"]),
        selected_paths=list(receipt["selected_paths"]), producers=receipt["producers"],
        **{key: receipt[key] for key in ASSET_FIELDS}, **{key: receipt["summary"][key] for key in MODEL_FIELDS},
        successor_helper_sha256=_helper_digest(), source_capture_exclusions=_capture_exclusions(receipt),
        output_root=str(output), excluded_new_output_paths=excluded, budget_profile=budget_profile,
        preparation_timeout_seconds=source384_timeout_limit(budget_profile), requires_independent_manifest=True, **AUTHORITY)
    validate_manifest(policy=policy, manifest=manifest)
    if _git(root, "rev-parse", "HEAD").decode().strip() != policy["baseline_commit"]:
        raise ValueError("learned numerical predecessor differs from signed baseline")
    validate_predecessor(repository=root, predecessor_receipt=receipt, policy=policy)
    return json.loads(_bytes(policy))


def validate_predecessor(*, repository, predecessor_receipt, policy):
    root = Path(repository).resolve(strict=True)
    receipt = native.validate_historical_numerical_selection_context(repository=root, expected_receipt=predecessor_receipt)
    if type(policy) is not dict or set(policy) != POLICY_FIELDS:
        raise ValueError("closed learned numerical successor policy required")
    _population(policy["source_hashes"], policy["selected_paths"])
    if (policy["schema"] != POLICY_SCHEMA or policy["profile"] != receipt["profile"] or policy["repository"] != str(root)
            or policy["predecessor_receipt_sha256"] != receipt["receipt_sha256"]
            or any(not _equal(policy[key], receipt[key]) for key in ("source_hashes", "selected_paths", "producers", *ASSET_FIELDS))
            or any(policy[key] != receipt["summary"][key] for key in MODEL_FIELDS)
            or policy["successor_helper_sha256"] != _helper_digest()
            or policy["source_capture_exclusions"] != _capture_exclusions(receipt)
            or policy["requires_independent_manifest"] is not True
            or any(not _equal(policy[key], value) for key, value in AUTHORITY.items())
            or type(policy["baseline_commit"]) is not str or re.fullmatch(r"[a-f0-9]{40,64}", policy["baseline_commit"]) is None
            or type(policy["manifest_cid"]) is not str or not policy["manifest_cid"]
            or type(policy["preparation_timeout_seconds"]) is not int
            or policy["preparation_timeout_seconds"] != source384_timeout_limit(policy["budget_profile"])):
        raise ValueError("selected learned numerical successor policy differs")
    native.source_owner._successor_exclusions(policy["excluded_new_output_paths"], policy["source_hashes"])
    output = native.source_owner._output(root, policy["output_root"])
    previous = Path(receipt["output"])
    if output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("learned numerical successor output must be separate from predecessor")
    return policy


def _publication(root, policy, binding):
    if (type(binding) is not dict or binding.get("task_status") != "completed"
            or type(binding.get("published_commit")) is not str
            or binding["published_commit"] == policy["baseline_commit"]
            or _git(root, "rev-parse", "HEAD").decode().strip() != binding["published_commit"]):
        raise ValueError("learned numerical successor requires accepted publication")
    if any(type(binding.get(key)) is not str or not binding[key] for key in (
            "task_cid", "task_id", "accepted_source_tree_id", "completion_receipt_cid")):
        raise ValueError("learned numerical successor lacks owner publication identity")


def _execution(receipt):
    report = json.loads(_read(Path(receipt["output"]) / "inference.json", native.MAX_INFERENCE_BYTES))["report"]
    output = report["output"]
    inferred, embeddings = output["inference"], output["embedding_receipt"]
    return dict(**receipt["inference_execution"], head_inference_calls=int(inferred is not None),
        inferred_units=len(inferred["rows"]) if inferred is not None else 0,
        embedding_production_calls=int(embeddings is not None),
        embedded_texts=embeddings["status_counts"]["embedded"] if embeddings is not None else 0)


def _observation(policy, receipt, binding):
    return dict(schema=OBSERVATION_SCHEMA, profile=policy["profile"], policy_cid=content_identity(policy),
        publication_binding_cid=content_identity(binding), predecessor_receipt_sha256=policy["predecessor_receipt_sha256"],
        successor_receipt_sha256=receipt["receipt_sha256"], source_hashes=receipt["source_hashes"],
        preserved_paths=sorted(policy["source_hashes"]), selected_paths=policy["selected_paths"],
        declared_outputs_outside_index_scope=policy["excluded_new_output_paths"],
        **{key: policy[key] for key in (*ASSET_FIELDS, *MODEL_FIELDS)},
        producer_configuration_preserved=True, scope_expanded=False, requires_independent_manifest=True,
        currentness_scope="accepted publication source bytes", inference_activity_scope="successor_artifact_preparation",
        inference_execution=_execution(receipt), **AUTHORITY)


def execution_observation(observation, *, reused):
    execution = observation["inference_execution"]
    return dict(schema=EXECUTION_SCHEMA, reused=reused,
        scope="current_successor_refresh_call", counters_known=True,
        **{key: (False if type(value) is bool else 0) if reused else value for key, value in execution.items()},
        **AUTHORITY)


def prepare_successor(*, repository, predecessor_receipt, policy, output, publication_binding, deadline_monotonic=None):
    remaining, end = _deadline(policy, deadline_monotonic)
    root = Path(repository).resolve(strict=True)
    validate_predecessor(repository=root, predecessor_receipt=predecessor_receipt, policy=policy)
    _publication(root, policy, publication_binding)
    output = native.source_owner._output(root, output, fresh=True)
    if output.parent != Path(policy["output_root"]):
        raise ValueError("learned numerical successor must use admitted output root")
    hashes = {name: _sha(_read(root / name, 1024 * 1024)) for name in sorted(policy["source_hashes"])}
    receipt = native.prepare_numerical_selection_context(repository=root, source_hashes=hashes,
        program_paths=policy["selected_paths"], output=output, config_path=policy["config_path"],
        expected_config_sha256=policy["config_sha256"], timeout_seconds=remaining(),
        budget_profile=policy["budget_profile"], excluded_new_output_paths=policy["excluded_new_output_paths"])
    observation = _observation(policy, receipt, publication_binding)
    validate_successor(repository=root, predecessor_receipt=predecessor_receipt, policy=policy,
        receipt=receipt, observation=observation, publication_binding=publication_binding, deadline_monotonic=end)
    return dict(numerical_selection_384=receipt, learned_numerical_refresh=observation)


def validate_successor(*, repository, predecessor_receipt, policy, receipt, observation,
                       publication_binding, deadline_monotonic=None):
    remaining, _end = _deadline(policy, deadline_monotonic)
    root = Path(repository).resolve(strict=True)
    validate_predecessor(repository=root, predecessor_receipt=predecessor_receipt, policy=policy)
    _publication(root, policy, publication_binding)
    if (type(receipt) is not dict or receipt.get("profile") != policy["profile"]
            or type(receipt.get("source_hashes")) is not dict or set(receipt["source_hashes"]) != set(policy["source_hashes"])
            or any(not _equal(receipt.get(key), policy[key]) for key in ("selected_paths", "producers", *ASSET_FIELDS))
            or any(receipt["summary"].get(key) != policy[key] for key in MODEL_FIELDS)
            or native.source_owner._output(root, receipt["output"]).parent != Path(policy["output_root"])):
        raise ValueError("learned numerical successor assets or scope differ")
    native.validate_numerical_selection_context(repository=root, expected_receipt=receipt,
        timeout_seconds=remaining(), budget_profile=policy["budget_profile"])
    if (_capture_exclusions(receipt) != sorted(set(policy["source_capture_exclusions"]) | set(policy["excluded_new_output_paths"]))
            or receipt["inference_execution"]["native_worker_executed"] is not True
            or not _equal(observation, _observation(policy, receipt, publication_binding))):
        raise ValueError("learned numerical successor capture or execution lineage differs")
    validate_predecessor(repository=root, predecessor_receipt=predecessor_receipt, policy=policy)
    _publication(root, policy, publication_binding)
    remaining()
    return receipt
