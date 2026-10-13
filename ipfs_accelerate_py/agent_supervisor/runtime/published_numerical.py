"""Regenerate selected deterministic numerical advice after publication.

The runtime signs the policy before execution and supplies the actual completed
owner binding after STOP. This module preserves that selection and its source
population; its observations never authorize another task or discharge proofs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

from ..proof.formal_verification_contracts import content_identity
from . import numerical_formalization_context as native
from .doctor_candidate_runner import _git
from .security_autoencoder_advisor import _bytes, _sha, _read

POLICY_SCHEMA = "admitted-numerical-successor-policy@1"
OBSERVATION_SCHEMA = "supervisor-published-numerical-refresh@1"
AUTHORITY = dict(proof_authority=False, formalization_authority=False,
    planning_authority=False, dispatch_authority=False, execution_authority=False,
    completion_authority=False, source_semantics_verified=False,
    whole_program_semantics_verified=False, source_obligations_discharged=0,
    model_inference_calls=0, training_steps=0, provider_calls=0)
POLICY_FIELDS = {"schema", "profile", "repository", "manifest_cid", "baseline_commit",
    "predecessor_receipt_sha256", "source_hashes", "producers", "successor_helper_sha256",
    "output_root", "excluded_new_output_paths", "requires_independent_manifest", *AUTHORITY}


def _equal(left, right):
    return _bytes(left) == _bytes(right)


def _checkpoint(deadline):
    if deadline is not None:
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("finite numerical successor deadline required")
        if time.monotonic() >= deadline:
            raise TimeoutError("numerical successor deadline expired")


def _helper_digest():
    return _sha(_read(Path(__file__).resolve(), 1_000_000))


def _population(hashes):
    if type(hashes) is not dict or len(hashes) > 64:
        raise ValueError("bounded selected numerical population required")
    for name, digest in hashes.items():
        if (type(name) is not str or not name.endswith(".py") or Path(name).is_absolute()
                or Path(name).as_posix() != name or any(p in {"..", ".git", ".runtime"} for p in Path(name).parts)
                or type(digest) is not str or re.fullmatch(r"[a-f0-9]{64}", digest) is None):
            raise ValueError("canonical selected numerical population required")


def _historical(repository, receipt):
    """Check retained bytes and producer pins, without testing old live source."""
    if (type(receipt) is not dict or receipt.get("schema") != native.SCHEMA
            or type(receipt.get("profile")) is not str or receipt["profile"] not in native.PROFILES
            or receipt.get("repository") != str(repository)
            or any(not _equal(receipt.get(k), v) for k, v in native.AUTHORITY.items())):
        raise ValueError("exact retained numerical predecessor required")
    _population(receipt["source_hashes"])
    output = native._output(repository, receipt["output"])
    raw = _read(output / "receipt.json", 131_072)
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (_sha(raw) != receipt.get("receipt_sha256") or raw != _bytes(body)
            or not _equal(native._inventory(output), receipt["files"])
            or not _equal(native._producers(receipt["profile"]), receipt["producers"])):
        raise ValueError("numerical predecessor artifacts or producer pins differ")
    return receipt


def bind_policy(*, repository, receipt, manifest, output_root):
    """Bind already selected current advice to the signed original sources."""
    root = Path(repository).resolve(strict=True)
    native.validate_numerical_context(repository=root, expected_receipt=receipt)
    _population(receipt["source_hashes"])
    if (type(manifest) is not dict or type(manifest.get("sources")) is not dict
            or type(manifest.get("baseline_commit")) is not str
            or re.fullmatch(r"[a-f0-9]{40,64}", manifest["baseline_commit"]) is None
            or _git(root, "rev-parse", "HEAD").decode().strip() != manifest["baseline_commit"]
            or any(manifest["sources"].get(name, {}).get("sha256") != digest
                   for name, digest in receipt["source_hashes"].items())):
        raise ValueError("numerical predecessor differs from signed baseline sources")
    output = native._output(root, output_root)
    previous = Path(receipt["output"])
    if output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("numerical successor output must be separate from predecessor")
    excluded = sorted({row["path"] for task in manifest["tasks"] for row in task["outputs"]}
        - set(receipt["source_hashes"]))
    return dict(schema=POLICY_SCHEMA, profile=receipt["profile"], repository=str(root),
        manifest_cid=content_identity(manifest), baseline_commit=manifest["baseline_commit"],
        predecessor_receipt_sha256=receipt["receipt_sha256"], source_hashes=dict(receipt["source_hashes"]),
        producers=dict(receipt["producers"]), successor_helper_sha256=_helper_digest(),
        output_root=str(output), excluded_new_output_paths=excluded,
        requires_independent_manifest=True, **AUTHORITY)


def validate_predecessor(*, repository, predecessor_receipt, policy):
    root = Path(repository).resolve(strict=True)
    receipt = _historical(root, predecessor_receipt)
    if (type(policy) is not dict or set(policy) != POLICY_FIELDS
            or policy["schema"] != POLICY_SCHEMA or policy["profile"] != receipt["profile"]
            or policy["repository"] != str(root)
            or policy["predecessor_receipt_sha256"] != receipt["receipt_sha256"]
            or not _equal(policy["source_hashes"], receipt["source_hashes"])
            or not _equal(policy["producers"], receipt["producers"])
            or policy["successor_helper_sha256"] != _helper_digest()
            or policy["requires_independent_manifest"] is not True
            or any(not _equal(policy[k], v) for k, v in AUTHORITY.items())
            or type(policy["manifest_cid"]) is not str or not policy["manifest_cid"]
            or type(policy["baseline_commit"]) is not str
            or re.fullmatch(r"[a-f0-9]{40,64}", policy["baseline_commit"]) is None
            or type(policy["excluded_new_output_paths"]) is not list
            or any(type(name) is not str or Path(name).is_absolute() or ".." in Path(name).parts
                   for name in policy["excluded_new_output_paths"])
            or sorted(set(policy["excluded_new_output_paths"])) != policy["excluded_new_output_paths"]
            or set(policy["excluded_new_output_paths"]) & set(policy["source_hashes"])):
        raise ValueError("selected numerical successor policy differs")
    output = native._output(root, policy["output_root"])
    previous = Path(receipt["output"])
    if output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("numerical successor output must be separate from predecessor")
    return policy


def validate_manifest(*, policy, manifest):
    """Join historical scope to the independently verified admitted declaration."""
    _population(policy["source_hashes"])
    if (type(manifest) is not dict or type(manifest.get("sources")) is not dict
            or policy.get("manifest_cid") != content_identity(manifest)
            or policy.get("baseline_commit") != manifest.get("baseline_commit")
            or any(manifest["sources"].get(name, {}).get("sha256") != digest
                   for name, digest in policy["source_hashes"].items())):
        raise ValueError("numerical successor differs from admitted manifest source population")
    excluded = sorted({row["path"] for task in manifest["tasks"] for row in task["outputs"]}
        - set(policy["source_hashes"]))
    if not _equal(policy.get("excluded_new_output_paths"), excluded):
        raise ValueError("numerical successor excluded outputs differ from admitted manifest")
    return policy


def _publication(root, policy, binding):
    if (type(binding) is not dict or binding.get("task_status") != "completed"
            or type(binding.get("published_commit")) is not str
            or binding["published_commit"] == policy["baseline_commit"]
            or _git(root, "rev-parse", "HEAD").decode().strip() != binding["published_commit"]):
        raise ValueError("numerical successor requires the accepted publication binding")
    for key in ("task_cid", "task_id", "accepted_source_tree_id", "completion_receipt_cid"):
        if type(binding.get(key)) is not str or not binding[key]:
            raise ValueError("numerical successor lacks owner publication identity")


def _observation(policy, receipt, publication_binding):
    return dict(schema=OBSERVATION_SCHEMA, profile=policy["profile"],
        policy_cid=content_identity(policy), publication_binding_cid=content_identity(publication_binding),
        predecessor_receipt_sha256=policy["predecessor_receipt_sha256"],
        successor_receipt_sha256=receipt["receipt_sha256"],
        source_hashes=dict(receipt["source_hashes"]), preserved_paths=sorted(policy["source_hashes"]),
        declared_outputs_outside_index_scope=policy["excluded_new_output_paths"],
        producer_configuration_preserved=True, scope_expanded=False,
        requires_independent_manifest=True, currentness_scope="accepted publication source bytes",
        **AUTHORITY)


def prepare_successor(*, repository, predecessor_receipt, policy, output,
                      publication_binding, deadline_monotonic=None):
    _checkpoint(deadline_monotonic)
    root = Path(repository).resolve(strict=True)
    validate_predecessor(repository=root, predecessor_receipt=predecessor_receipt, policy=policy)
    _publication(root, policy, publication_binding)
    output = native._output(root, output, fresh=True)
    if output.parent != Path(policy["output_root"]):
        raise ValueError("numerical successor must use the admitted output root")
    hashes = {name: _sha(_read(root / name, 1_000_000)) for name in sorted(policy["source_hashes"])}
    _checkpoint(deadline_monotonic)
    receipt = native.prepare_numerical_context(repository=root, source_hashes=hashes,
        output=output, profile=policy["profile"])
    observation = _observation(policy, receipt, publication_binding)
    validate_successor(repository=root, predecessor_receipt=predecessor_receipt,
        policy=policy, receipt=receipt, observation=observation,
        publication_binding=publication_binding, deadline_monotonic=deadline_monotonic)
    return dict(numerical_formalization=receipt, numerical_refresh=observation)


def validate_successor(*, repository, predecessor_receipt, policy, receipt, observation,
                       publication_binding, deadline_monotonic=None):
    _checkpoint(deadline_monotonic)
    root = Path(repository).resolve(strict=True)
    validate_predecessor(repository=root, predecessor_receipt=predecessor_receipt, policy=policy)
    _publication(root, policy, publication_binding)
    if (type(receipt) is not dict or receipt.get("profile") != policy["profile"]
            or type(receipt.get("source_hashes")) is not dict
            or set(receipt["source_hashes"]) != set(policy["source_hashes"])
            or not _equal(receipt.get("producers"), policy["producers"])
            or native._output(root, receipt["output"]).parent != Path(policy["output_root"])
            or not _equal(observation, _observation(policy, receipt, publication_binding))):
        raise ValueError("numerical successor lineage or scope differs")
    _checkpoint(deadline_monotonic)
    native.validate_numerical_context(repository=root, expected_receipt=receipt)
    _checkpoint(deadline_monotonic)
    validate_predecessor(repository=root, predecessor_receipt=predecessor_receipt, policy=policy)
    _publication(root, policy, publication_binding)
    _checkpoint(deadline_monotonic)
    return receipt
