"""Audited, default-off review of implementations which already landed.

This migration path exists for a small, operator-pinned allow-list whose
commits predate typed production-provider receipts.  It does **not** infer a
historical provider and never manufactures a ``ProviderExecutionReceipt``.
Instead it reconstructs the exact historical interval, reviews a byte-complete
manifest with both the pinned effective Grok and Codex providers, runs fresh
validations, and emits a signed ``LegacyLandedReviewAttestation@1`` whose
completion/proof authority is always false.

The policy is loaded once from an operator-owned path.  ``review(task_id)`` has
no policy, key, commit, path, provider, or validation override arguments, so a
task/caller cannot widen the audited envelope.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from .git_environment import sanitized_git_environment
from .legacy_landed_attestation import (
    HISTORICAL_PROVIDER_UNVERIFIED,
    LegacyLandedReviewAttestation,
    LegacyLandedReviewAuthority,
)

LEGACY_LANDED_REVIEW_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-review-policy@1"
)
LEGACY_LANDED_REVIEW_POLICY_INTERFACE: Final = "LegacyLandedReviewPolicy@1"
LEGACY_LANDED_REVIEW_POLICY_TEMPLATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-review-policy-template@1"
)
LEGACY_LANDED_REVIEW_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-byte-manifest@1"
)
LEGACY_LANDED_REVIEW_LEAF_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-byte-leaf@1"
)
LEGACY_LANDED_REVIEW_MERKLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-merkle-node@1"
)
LEGACY_LANDED_LEAF_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-leaf-decision@1"
)
LEGACY_LANDED_LEAF_REVIEW_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-leaf-review-receipt@1"
)
LEGACY_LANDED_REVIEW_AGGREGATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-review-aggregate@1"
)
LEGACY_LANDED_VALIDATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-validation-receipt@1"
)
LEGACY_LANDED_SCOPE_ADJUDICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-scope-adjudication@1"
)

EXACT_LEGACY_LANDED_TASK_IDS: Final = (
    "ASE-005",
    "ASE-006",
    "ASE-007",
    "ASE-008",
    "ASE-009",
    "ASE-012",
    "ASE-023",
    "ASE-038",
)
SCOPE_ADJUDICATION_TASK_IDS: Final = frozenset({"ASE-009", "ASE-038"})
MAX_LEAF_TOKENS: Final = 4_096
MAX_POLICY_BYTES: Final = 2 * 1024 * 1024
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_TASK_CID_RE: Final = re.compile(r"^b[a-z2-7]{20,}$")
_TASK_KEY_RE: Final = re.compile(r"^task/v1/[0-9a-f]{64}$")


# Reviewed historical bindings.  This is deliberately a non-runnable
# template: it has neither a deploy HEAD nor an issuer.  The external policy
# generator below requires an explicit post-deploy HEAD and reconstructs its
# tree before producing a parser-admissible policy.
EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE: Final = {
    "schema": LEGACY_LANDED_REVIEW_POLICY_TEMPLATE_SCHEMA,
    "providers": {
        "grok": {
            "role": "grok_audit",
            "provider": "grok_cli",
            "model": "grok-4.5",
            "fallback_allowed": False,
            "self_review_allowed": False,
        },
        "codex": {
            "role": "codex_audit",
            "provider": "codex_cli",
            "model": "gpt-5.6-sol",
            "fallback_allowed": False,
            "self_review_allowed": False,
        },
    },
    "max_leaf_tokens": MAX_LEAF_TOKENS,
    "tasks": [
        {
            "task_id": "ASE-005",
            "canonical_task_key": "task/v1/724b237e6c9f525c55cf20c09f7655f402891c04a01924471b41c1f03a06b3b2",
            "canonical_task_cid": "baguqeeraojfsg7tmt5jfyvopedaj65sv6qbishaeuamsiry3iha7aoqgwoza",
            "baseline_commit": "07d5cd3791855100d481c1476ef0500ba2ba514a",
            "interval_commits": ["2ae9b5a1fcbf15c7bddd080df280709ab20edc1b"],
            "implementation_commit": "2ae9b5a1fcbf15c7bddd080df280709ab20edc1b",
            "merge_commit": "6db3a7fde20e3e862dcbdd0814201aa436810309",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py",
                "test/api/test_agent_supervisor_target_resolver.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_target_resolver.py", "-q"]],
            "scope_adjudication": None,
        },
        {
            "task_id": "ASE-006",
            "canonical_task_key": "task/v1/4f83a44fc5f3492c3d46d64842284d32f0995415c870bf9b6d7b66e30b3de385",
            "canonical_task_cid": "baguqeeraj6b2it6f6nesypkg2zeeekcnglyjsvavzbyl7g3npntogcz54ocq",
            "baseline_commit": "07d5cd3791855100d481c1476ef0500ba2ba514a",
            "interval_commits": ["2335865cb024f93b91e5b663d6ff375777ff1e5d"],
            "implementation_commit": "2335865cb024f93b91e5b663d6ff375777ff1e5d",
            "merge_commit": "45d6004575b58b16623d0c1b6ead7c5a7c46b696",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py",
                "test/api/test_agent_supervisor_state_resolver.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_state_resolver.py", "-q"]],
            "scope_adjudication": None,
        },
        {
            "task_id": "ASE-007",
            "canonical_task_key": "task/v1/ff74fbdf841556ae107337da633ca61f03a64a3d5b5b1a4c766afe4e6f18c66d",
            "canonical_task_cid": "baguqeera752pxx4ecvlk4edtg7nggpfgd4b2msr5lnnrutdwnl7e43yyyzwq",
            "baseline_commit": "6db3a7fde20e3e862dcbdd0814201aa436810309",
            "interval_commits": ["9681150c02bb00063ddfa4ca78b37a767d919e67"],
            "implementation_commit": "9681150c02bb00063ddfa4ca78b37a767d919e67",
            "merge_commit": "234c1fc2be93255c0c0b42b12d4bb470025c99e0",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py",
                "test/api/test_agent_supervisor_objective_resolver.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_objective_resolver.py", "-q"]],
            "scope_adjudication": None,
        },
        {
            "task_id": "ASE-008",
            "canonical_task_key": "task/v1/2fc7a0490c924969b48220eb74b03a9c59f7a37b6b8e249e66ca53455798443d",
            "canonical_task_cid": "baguqeeraf7d2asimsjewtnecedvxjmb2trm7pi33nohcjhtgzjjukv4yiq6q",
            "baseline_commit": "45d6004575b58b16623d0c1b6ead7c5a7c46b696",
            "interval_commits": ["09d6e682f4be435cf73ff4517c6027487fd25264"],
            "implementation_commit": "09d6e682f4be435cf73ff4517c6027487fd25264",
            "merge_commit": "b11cc0be529f3af32aa3d7e802cff2379e5e364f",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py",
                "test/api/test_agent_supervisor_authority_resolver.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_authority_resolver.py", "-q"]],
            "scope_adjudication": None,
        },
        {
            "task_id": "ASE-009",
            "canonical_task_key": "task/v1/b21a9c0df533a46be42c9259331404bb710ab0e1e71da9fa1a0bea195e2ab8d8",
            "canonical_task_cid": "baguqeerawinjydpvgosgxzbmsjmtgfaexnyqvmhb44o2t6q2bpvbsxrkxdma",
            "baseline_commit": "27cc4219f67358d90abd36b08b37950be344009e",
            "interval_commits": [
                "dafecaf0b9c0823d1c3d0102a53d8e926948b603",
                "4825aab7ea374abfc89fdc63791eaaf2729d8b84",
            ],
            "implementation_commit": "4825aab7ea374abfc89fdc63791eaaf2729d8b84",
            "merge_commit": "38884b641c84fde273a7293e61c27f5405c1aa1b",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py",
                "ipfs_accelerate_py/agent_supervisor/multiformats_identity.py",
                "ipfs_accelerate_py/utils/cid_utils.py",
                "test/api/test_agent_supervisor_capability_resolver.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_default_provider_route.py", "test/api/test_agent_supervisor_capability_resolver.py", "-q"]],
            "scope_adjudication": {
                "reason_code": "repair_commit_in_exact_task_interval",
                "justification": "The second commit repairs hermetic imports in the original ASE-009 implementation; both commits and all four resulting paths are reviewed as one exact interval.",
            },
        },
        {
            "task_id": "ASE-012",
            "canonical_task_key": "task/v1/60fc4600d2bfac1863b69cad01fc9b86863f29e1c67adf375ad9de9530800398",
            "canonical_task_cid": "baguqeeramd6emagsx6wbqy5wtswqd7e3q2dd6kpbyz5n6n223hpjkmeaaoma",
            "baseline_commit": "27cc4219f67358d90abd36b08b37950be344009e",
            "interval_commits": [
                "d0dce4301ad2a7133bedc32beea1f59b3728b0d3",
                "b0e6777172a407ca48da36ec630295e901c4eb32",
            ],
            "implementation_commit": "b0e6777172a407ca48da36ec630295e901c4eb32",
            "merge_commit": "106f373927f01ecff81cdba736787e9e043f9577",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py",
                "test/api/test_agent_supervisor_prompt_broker.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_prompt_broker.py", "-q"]],
            "scope_adjudication": None,
        },
        {
            "task_id": "ASE-023",
            "canonical_task_key": "task/v1/4d2f167ed56fced1187387d99cecd4703f24b270cf086e3a5883c84b2e080426",
            "canonical_task_cid": "baguqeerajuxrm7wvn7hncgdtq7mzz3guoa7sjmtqz4eg4osyqpeewlqiaqta",
            "baseline_commit": "27cc4219f67358d90abd36b08b37950be344009e",
            "interval_commits": [
                "aa140915915120f92bbc3738e6961f64e620dcba",
                "4815d296926a7b980200a301a711162d82165612",
            ],
            "implementation_commit": "4815d296926a7b980200a301a711162d82165612",
            "merge_commit": "07d5cd3791855100d481c1476ef0500ba2ba514a",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py",
                "test/api/test_agent_supervisor_steering_contracts.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_steering_contracts.py", "-q"]],
            "scope_adjudication": None,
        },
        {
            "task_id": "ASE-038",
            "canonical_task_key": "task/v1/4906856a46a0ae687408880693223d962d4463277b8785c88267dac2c8cc1056",
            "canonical_task_cid": "baguqeerajedik2sgucxgq5airadjgir5sywuiyzhpodylsecm7nmfsgmcbla",
            "baseline_commit": "819139990a28c9c1303154a4c543c50a69d200a1",
            "interval_commits": ["79f2b86424e7d28222ab23c6f82bace7b759b6cd"],
            "implementation_commit": "79f2b86424e7d28222ab23c6f82bace7b759b6cd",
            "merge_commit": "5ed81387dcb38087f05f0b1ab0f6e316f55e1bfd",
            "implementation_parent_index": 2,
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py",
                "ipfs_accelerate_py/ipfs_backend_router.py",
                "test/api/test_agent_supervisor_multiformats_identity.py",
                "test/api/test_agent_supervisor_verified_ipld_backend.py",
            ],
            "validations": [["python", "-m", "pytest", "test/api/test_agent_supervisor_verified_ipld_backend.py", "test/test_ipfs_backend_router.py", "test/api/test_agent_supervisor_multiformats_identity.py", "-q"]],
            "scope_adjudication": {
                "reason_code": "final_clean_attempt_exact_scope",
                "justification": "The pinned final ASE-038 attempt supersedes failed historical attempts; only its exact four-path interval is admitted and reviewed.",
            },
        },
    ],
    "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
    "completion_authoritative": False,
    "proof_authoritative": False,
}


class LegacyLandedReviewError(RuntimeError):
    """Fail-closed review error with a stable, non-sensitive reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = str(reason_code or "legacy_landed_review_failed")
        super().__init__(self.reason_code)


LEGACY_CODEX_USAGE_LIMIT_CAPACITY_MARKER: Final = (
    "legacy_codex_usage_capacity_exhausted"
)


class LegacyProviderCapacitySignal(RuntimeError):
    """A fixed, secret-free signal emitted by an audited native adapter."""

    reason_code: Final = LEGACY_CODEX_USAGE_LIMIT_CAPACITY_MARKER

    def __init__(self) -> None:
        super().__init__(self.reason_code)


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate policy field")
            result[key] = value
        return result

    parsed = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-finite policy value")
        ),
    )
    if not isinstance(parsed, dict):
        raise ValueError("operator policy must be a JSON object")
    return parsed


def _canonical_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("policy path is invalid")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("policy path must use canonical NFC encoding")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("policy path contains control characters")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(part.casefold() == ".git" for part in path.parts)
    ):
        raise ValueError("policy path escapes the repository")
    canonical = path.as_posix()
    if canonical in {"", "."} or canonical != value:
        raise ValueError("policy path is not canonical")
    return canonical


def _commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{field} must be a full lowercase commit identity")
    return value


@dataclass(frozen=True, slots=True)
class LegacyProviderPolicy:
    role: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class LegacyScopeAdjudication:
    reason_code: str
    justification: str


@dataclass(frozen=True, slots=True)
class LegacyTaskPolicy:
    task_id: str
    canonical_task_key: str
    canonical_task_cid: str
    baseline_commit: str
    interval_commits: tuple[str, ...]
    implementation_commit: str
    merge_commit: str
    implementation_parent_index: int
    paths: tuple[str, ...]
    validations: tuple[tuple[str, ...], ...]
    scope_adjudication: LegacyScopeAdjudication | None


@dataclass(frozen=True, slots=True)
class LegacyLandedReviewPolicy:
    policy_id: str
    enabled: bool
    issuer_key_id: str
    current_head: str
    current_tree_id: str
    max_leaf_tokens: int
    grok: LegacyProviderPolicy
    codex: LegacyProviderPolicy
    tasks: tuple[LegacyTaskPolicy, ...]

    def task(self, task_id: str) -> LegacyTaskPolicy:
        matches = [item for item in self.tasks if item.task_id == task_id]
        if len(matches) != 1:
            raise LegacyLandedReviewError("legacy_task_not_operator_pinned")
        return matches[0]


def _parse_provider(value: Any, expected_role: str) -> LegacyProviderPolicy:
    if not isinstance(value, Mapping) or set(value) != {
        "role", "provider", "model", "fallback_allowed", "self_review_allowed"
    }:
        raise ValueError("legacy provider policy shape is invalid")
    if value.get("role") != expected_role:
        raise ValueError("legacy provider role is invalid")
    if value.get("fallback_allowed") is not False:
        raise ValueError("legacy provider fallback must remain disabled")
    if value.get("self_review_allowed") is not False:
        raise ValueError("legacy provider self-review must remain disabled")
    provider = value.get("provider")
    model = value.get("model")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("legacy effective provider is required")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("legacy effective model is required")
    return LegacyProviderPolicy(expected_role, provider.strip(), model.strip())


def parse_legacy_landed_review_policy(
    value: Mapping[str, Any],
) -> LegacyLandedReviewPolicy:
    """Strictly parse a complete production policy; templates are rejected."""

    payload = _strict_json_object(canonical_json_bytes(value))
    expected_keys = {
        "schema", "interface", "policy_id", "enabled", "issuer_key_id",
        "current_head", "current_tree_id", "max_leaf_tokens", "providers",
        "tasks", "historical_provider", "completion_authoritative",
        "proof_authoritative",
    }
    if set(payload) != expected_keys:
        raise ValueError("legacy landed review policy shape is invalid")
    if payload.get("schema") != LEGACY_LANDED_REVIEW_POLICY_SCHEMA:
        raise ValueError("legacy landed review policy schema is invalid")
    if payload.get("interface") != LEGACY_LANDED_REVIEW_POLICY_INTERFACE:
        raise ValueError("legacy landed review policy interface is invalid")
    if not isinstance(payload.get("enabled"), bool):
        raise ValueError("legacy landed review enabled flag is invalid")
    if payload.get("historical_provider") != HISTORICAL_PROVIDER_UNVERIFIED:
        raise ValueError("historical provider must remain unverified")
    if payload.get("completion_authoritative") is not False:
        raise ValueError("legacy policy cannot claim completion authority")
    if payload.get("proof_authoritative") is not False:
        raise ValueError("legacy policy cannot claim proof authority")
    issuer = payload.get("issuer_key_id")
    if not isinstance(issuer, str) or not re.fullmatch(
        r"ed25519:sha256:[0-9a-f]{64}", issuer
    ):
        raise ValueError("legacy issuer key is not explicitly pinned")
    head = _commit(payload.get("current_head"), "current_head")
    tree = _commit(payload.get("current_tree_id"), "current_tree_id")
    max_tokens = payload.get("max_leaf_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= MAX_LEAF_TOKENS
    ):
        raise ValueError("legacy leaf budget exceeds 4096 tokens")
    providers = payload.get("providers")
    if not isinstance(providers, Mapping) or set(providers) != {"grok", "codex"}:
        raise ValueError("legacy provider set is invalid")
    if canonical_json_bytes(providers) != canonical_json_bytes(
        EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE["providers"]
    ):
        raise ValueError("legacy provider/model binding differs from audited template")
    if max_tokens != EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE["max_leaf_tokens"]:
        raise ValueError("legacy leaf budget differs from audited template")
    grok = _parse_provider(providers["grok"], "grok_audit")
    codex = _parse_provider(providers["codex"], "codex_audit")
    if grok.provider.casefold() == codex.provider.casefold():
        raise ValueError("Grok and Codex effective providers must be distinct")
    if (grok.provider.casefold(), grok.model.casefold()) == (
        codex.provider.casefold(), codex.model.casefold()
    ):
        raise ValueError("legacy provider self-review is forbidden")

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(
        EXACT_LEGACY_LANDED_TASK_IDS
    ):
        raise ValueError("legacy policy must pin exactly eight tasks")
    if canonical_json_bytes(raw_tasks) != canonical_json_bytes(
        EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE["tasks"]
    ):
        raise ValueError("legacy task binding differs from audited template")
    parsed_tasks: list[LegacyTaskPolicy] = []
    task_keys: set[str] = set()
    task_cids: set[str] = set()
    task_shape = {
        "task_id", "canonical_task_key", "canonical_task_cid",
        "baseline_commit", "interval_commits", "implementation_commit",
        "merge_commit", "implementation_parent_index", "paths",
        "validations", "scope_adjudication",
    }
    for raw in raw_tasks:
        if not isinstance(raw, Mapping) or set(raw) != task_shape:
            raise ValueError("legacy task policy shape is invalid")
        task_id = raw.get("task_id")
        if task_id not in EXACT_LEGACY_LANDED_TASK_IDS:
            raise ValueError("legacy task is outside the audited allow-list")
        key = raw.get("canonical_task_key")
        cid = raw.get("canonical_task_cid")
        if not isinstance(key, str) or not _TASK_KEY_RE.fullmatch(key):
            raise ValueError("legacy canonical task key is invalid")
        if not isinstance(cid, str) or not _TASK_CID_RE.fullmatch(cid):
            raise ValueError("legacy canonical task CID is invalid")
        if key in task_keys or cid in task_cids:
            raise ValueError("legacy canonical task identity is duplicated")
        task_keys.add(key)
        task_cids.add(cid)
        baseline = _commit(raw.get("baseline_commit"), "baseline_commit")
        implementation = _commit(
            raw.get("implementation_commit"), "implementation_commit"
        )
        merge = _commit(raw.get("merge_commit"), "merge_commit")
        interval_raw = raw.get("interval_commits")
        if not isinstance(interval_raw, list) or not interval_raw:
            raise ValueError("legacy implementation interval is missing")
        interval = tuple(
            _commit(item, "interval_commit") for item in interval_raw
        )
        if len(set(interval)) != len(interval) or interval[-1] != implementation:
            raise ValueError("legacy implementation interval is ambiguous")
        if task_id == "ASE-023" and len(interval) != 2:
            raise ValueError("ASE-023 must bind its exact two-commit interval")
        parent_index = raw.get("implementation_parent_index")
        if isinstance(parent_index, bool) or parent_index != 2:
            raise ValueError("legacy merge must pin implementation parent two")
        raw_paths = raw.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("legacy task path set is missing")
        paths = tuple(_canonical_path(item) for item in raw_paths)
        if list(paths) != sorted(set(paths)):
            raise ValueError("legacy task path set is not canonical")
        raw_validations = raw.get("validations")
        if not isinstance(raw_validations, list) or not raw_validations:
            raise ValueError("legacy task validations are missing")
        validations: list[tuple[str, ...]] = []
        for argv in raw_validations:
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)
            ):
                raise ValueError("legacy validation argv is invalid")
            validations.append(tuple(argv))
        raw_scope = raw.get("scope_adjudication")
        scope: LegacyScopeAdjudication | None = None
        if task_id in SCOPE_ADJUDICATION_TASK_IDS:
            if not isinstance(raw_scope, Mapping) or set(raw_scope) != {
                "reason_code", "justification"
            }:
                raise ValueError("legacy task requires scope adjudication")
            reason = raw_scope.get("reason_code")
            justification = raw_scope.get("justification")
            allowed_reason = {
                "repair_commit_in_exact_task_interval",
                "final_clean_attempt_exact_scope",
            }
            if reason not in allowed_reason or not isinstance(justification, str) or not justification.strip():
                raise ValueError("legacy scope adjudication is invalid")
            scope = LegacyScopeAdjudication(reason, justification.strip())
        elif raw_scope is not None:
            raise ValueError("unexpected legacy scope adjudication")
        parsed_tasks.append(
            LegacyTaskPolicy(
                task_id=task_id,
                canonical_task_key=key,
                canonical_task_cid=cid,
                baseline_commit=baseline,
                interval_commits=interval,
                implementation_commit=implementation,
                merge_commit=merge,
                implementation_parent_index=parent_index,
                paths=paths,
                validations=tuple(validations),
                scope_adjudication=scope,
            )
        )
    if tuple(sorted(item.task_id for item in parsed_tasks)) != EXACT_LEGACY_LANDED_TASK_IDS:
        raise ValueError("legacy policy task allow-list is incomplete")
    if [item.task_id for item in parsed_tasks] != list(EXACT_LEGACY_LANDED_TASK_IDS):
        raise ValueError("legacy policy tasks are not canonically ordered")
    claimed_policy_id = payload.get("policy_id")
    unsigned = dict(payload)
    unsigned.pop("policy_id", None)
    if claimed_policy_id != content_identity(unsigned):
        raise ValueError("legacy policy content identity is invalid")
    return LegacyLandedReviewPolicy(
        policy_id=claimed_policy_id,
        enabled=payload["enabled"],
        issuer_key_id=issuer,
        current_head=head,
        current_tree_id=tree,
        max_leaf_tokens=max_tokens,
        grok=grok,
        codex=codex,
        tasks=tuple(parsed_tasks),
    )


def load_legacy_landed_review_policy(
    operator_policy_path: str | Path,
) -> LegacyLandedReviewPolicy:
    """Load one bounded, non-symlink operator policy from its pinned path."""

    path = Path(operator_policy_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if not hasattr(os, "O_NOFOLLOW") and path.is_symlink():
        raise ValueError("legacy operator policy cannot be a symlink")
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("legacy operator policy must be a regular file")
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            raise ValueError("legacy operator policy owner is invalid")
        if info.st_nlink != 1:
            raise ValueError("legacy operator policy cannot be hard-linked")
        if stat.S_IMODE(info.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("legacy operator policy is writable by other principals")
        if info.st_size < 2 or info.st_size > MAX_POLICY_BYTES:
            raise ValueError("legacy operator policy size is invalid")
        raw = b""
        while len(raw) <= MAX_POLICY_BYTES:
            chunk = os.read(descriptor, min(65536, MAX_POLICY_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    if len(raw) > MAX_POLICY_BYTES:
        raise ValueError("legacy operator policy is too large")
    return parse_legacy_landed_review_policy(_strict_json_object(raw))


def _git(
    repo_root: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        env=sanitized_git_environment(),
    )
    if check and result.returncode != 0:
        raise LegacyLandedReviewError("legacy_repository_git_command_failed")
    return result


def _git_text(repo_root: Path, args: Sequence[str]) -> str:
    try:
        return _git(repo_root, args).stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise LegacyLandedReviewError("legacy_repository_identity_invalid") from exc


def _repo_head(repo_root: Path) -> str:
    return _commit(
        _git_text(repo_root, ["rev-parse", "--verify", "HEAD^{commit}"]),
        "repository HEAD",
    )


def _tree_id(repo_root: Path, commit: str) -> str:
    return _commit(
        _git_text(repo_root, ["rev-parse", "--verify", f"{commit}^{{tree}}"]),
        "repository tree",
    )


def _repo_is_clean(repo_root: Path) -> bool:
    return not _git(
        repo_root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    ).stdout


def _resolve_exact_commit(repo_root: Path, commit: str) -> str:
    resolved = _git_text(
        repo_root, ["rev-parse", "--verify", f"{commit}^{{commit}}"]
    )
    if resolved != commit:
        raise LegacyLandedReviewError("legacy_repository_commit_binding_mismatch")
    return resolved


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = _git(
        repo_root,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise LegacyLandedReviewError("legacy_repository_ancestry_check_failed")
    return result.returncode == 0


def _diff_paths(repo_root: Path, left: str, right: str) -> tuple[str, ...]:
    raw = _git(
        repo_root,
        [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--name-only",
            "-z",
            left,
            right,
            "--",
        ],
    ).stdout
    if not raw:
        return ()
    pieces = raw.split(b"\0")
    if pieces[-1] != b"":
        raise LegacyLandedReviewError("legacy_repository_path_stream_invalid")
    try:
        paths = tuple(item.decode("utf-8") for item in pieces[:-1])
    except UnicodeDecodeError as exc:
        raise LegacyLandedReviewError("legacy_repository_path_encoding_invalid") from exc
    try:
        canonical = tuple(_canonical_path(item) for item in paths)
    except ValueError as exc:
        raise LegacyLandedReviewError("legacy_repository_path_invalid") from exc
    if len(set(canonical)) != len(canonical):
        raise LegacyLandedReviewError("legacy_repository_path_duplicated")
    return tuple(sorted(canonical))


@dataclass(frozen=True, slots=True)
class _GitBlob:
    exists: bool
    mode: str
    object_id: str
    data: bytes


def _blob_at(repo_root: Path, commit: str, path: str) -> _GitBlob:
    raw = _git(repo_root, ["ls-tree", "-z", commit, "--", path]).stdout
    if not raw:
        return _GitBlob(False, "", "", b"")
    records = [item for item in raw.split(b"\0") if item]
    if len(records) != 1 or b"\t" not in records[0]:
        raise LegacyLandedReviewError("legacy_repository_blob_binding_ambiguous")
    metadata, raw_path = records[0].split(b"\t", 1)
    try:
        observed_path = raw_path.decode("utf-8")
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
    except (UnicodeDecodeError, ValueError) as exc:
        raise LegacyLandedReviewError("legacy_repository_blob_binding_invalid") from exc
    if observed_path != path or object_type != "blob" or not _COMMIT_RE.fullmatch(object_id):
        raise LegacyLandedReviewError("legacy_repository_blob_binding_invalid")
    data = _git(repo_root, ["cat-file", "blob", object_id]).stdout
    return _GitBlob(True, mode, object_id, data)


@dataclass(frozen=True, slots=True)
class LegacyRepositoryBinding:
    task: LegacyTaskPolicy
    current_head: str
    current_tree_id: str
    historical_diff: bytes
    current_blobs: tuple[tuple[str, _GitBlob], ...]


def inspect_legacy_repository_binding(
    repo_root: str | Path,
    policy: LegacyLandedReviewPolicy,
    task: LegacyTaskPolicy,
    *,
    require_clean: bool = True,
) -> LegacyRepositoryBinding:
    """Recompute the full commit topology, path set, and landed bytes."""

    repo = Path(repo_root).resolve(strict=True)
    top = Path(
        _git(repo, ["rev-parse", "--show-toplevel"]).stdout.decode("utf-8").strip()
    ).resolve(strict=True)
    if top != repo:
        raise LegacyLandedReviewError("legacy_repository_root_mismatch")
    if require_clean and not _repo_is_clean(repo):
        raise LegacyLandedReviewError("legacy_repository_not_clean")
    head = _repo_head(repo)
    tree = _tree_id(repo, head)
    if head != policy.current_head or tree != policy.current_tree_id:
        raise LegacyLandedReviewError("legacy_repository_head_fence_failed")
    commits = {
        task.baseline_commit,
        *task.interval_commits,
        task.implementation_commit,
        task.merge_commit,
        head,
    }
    for commit in commits:
        _resolve_exact_commit(repo, commit)
    if not _is_ancestor(repo, task.baseline_commit, task.implementation_commit):
        raise LegacyLandedReviewError("legacy_baseline_not_implementation_ancestor")
    if not _is_ancestor(repo, task.merge_commit, head):
        raise LegacyLandedReviewError("legacy_merge_not_current_head_ancestor")

    interval_raw = _git(
        repo,
        [
            "rev-list",
            "--reverse",
            "--topo-order",
            "--ancestry-path",
            f"{task.baseline_commit}..{task.implementation_commit}",
        ],
    ).stdout.decode("ascii").splitlines()
    if tuple(interval_raw) != task.interval_commits:
        raise LegacyLandedReviewError("legacy_implementation_interval_mismatch")
    previous = task.baseline_commit
    for commit in task.interval_commits:
        topology = _git_text(repo, ["rev-list", "--parents", "-n", "1", commit]).split()
        if len(topology) != 2 or topology[1] != previous:
            raise LegacyLandedReviewError("legacy_implementation_interval_not_linear")
        previous = commit

    merge_topology = _git_text(
        repo, ["rev-list", "--parents", "-n", "1", task.merge_commit]
    ).split()
    if len(merge_topology) < 3:
        raise LegacyLandedReviewError("legacy_merge_commit_required")
    if merge_topology[task.implementation_parent_index] != task.implementation_commit:
        raise LegacyLandedReviewError("legacy_merge_implementation_parent_mismatch")
    first_parent = merge_topology[1]
    if _diff_paths(repo, task.baseline_commit, task.implementation_commit) != task.paths:
        raise LegacyLandedReviewError("legacy_historical_path_set_mismatch")
    if _diff_paths(repo, first_parent, task.merge_commit) != task.paths:
        raise LegacyLandedReviewError("legacy_merge_path_set_mismatch")
    changed_after_merge = set(_diff_paths(repo, task.merge_commit, head)).intersection(
        task.paths
    )
    if changed_after_merge:
        raise LegacyLandedReviewError("legacy_landed_paths_changed_after_merge")

    current_blobs: list[tuple[str, _GitBlob]] = []
    for path in task.paths:
        implementation_blob = _blob_at(repo, task.implementation_commit, path)
        merge_blob = _blob_at(repo, task.merge_commit, path)
        current_blob = _blob_at(repo, head, path)
        identities = {
            (item.exists, item.mode, item.object_id)
            for item in (implementation_blob, merge_blob, current_blob)
        }
        if len(identities) != 1:
            raise LegacyLandedReviewError("legacy_landed_blob_invariance_failed")
        current_blobs.append((path, current_blob))

    historical_diff = _git(
        repo,
        [
            "-c", "core.quotepath=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--binary",
            "--full-index",
            "--no-color",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            task.baseline_commit,
            task.implementation_commit,
            "--",
            *task.paths,
        ],
    ).stdout
    if not historical_diff:
        raise LegacyLandedReviewError("legacy_historical_diff_missing")
    if require_clean and not _repo_is_clean(repo):
        raise LegacyLandedReviewError("legacy_repository_changed_during_inspection")
    if _repo_head(repo) != head or _tree_id(repo, head) != tree:
        raise LegacyLandedReviewError("legacy_repository_head_changed_during_inspection")
    return LegacyRepositoryBinding(
        task=task,
        current_head=head,
        current_tree_id=tree,
        historical_diff=historical_diff,
        current_blobs=tuple(current_blobs),
    )


def build_exact_eight_legacy_landed_policy(
    repo_root: str | Path,
    *,
    current_head: str,
    issuer_key_id: str,
    enabled: bool = False,
) -> dict[str, Any]:
    """Materialize, but never write, the exact policy at a fenced deploy HEAD."""

    repo = Path(repo_root).resolve(strict=True)
    explicit_head = _commit(current_head, "current_head")
    if _repo_head(repo) != explicit_head:
        raise LegacyLandedReviewError("legacy_policy_generation_head_fence_failed")
    if not _repo_is_clean(repo):
        raise LegacyLandedReviewError("legacy_policy_generation_requires_clean_repo")
    template = json.loads(canonical_json_bytes(EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE))
    body = {
        "schema": LEGACY_LANDED_REVIEW_POLICY_SCHEMA,
        "interface": LEGACY_LANDED_REVIEW_POLICY_INTERFACE,
        "enabled": bool(enabled),
        "issuer_key_id": issuer_key_id,
        "current_head": explicit_head,
        "current_tree_id": _tree_id(repo, explicit_head),
        "max_leaf_tokens": template["max_leaf_tokens"],
        "providers": template["providers"],
        "tasks": template["tasks"],
        "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    payload = {**body, "policy_id": content_identity(body)}
    policy = parse_legacy_landed_review_policy(payload)
    for task in policy.tasks:
        inspect_legacy_repository_binding(repo, policy, task)
    return payload


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _python_ast_boundaries(data: bytes) -> set[int]:
    try:
        text = data.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError, ValueError):
        return set()
    line_ends: list[int] = []
    total = 0
    for line in text.splitlines(keepends=True):
        total += len(line.encode("utf-8"))
        line_ends.append(total)
    if total < len(data):
        line_ends.append(len(data))
    boundaries: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end_line = getattr(node, "end_lineno", None)
            if isinstance(end_line, int) and 1 <= end_line <= len(line_ends):
                boundaries.add(line_ends[end_line - 1])
    return boundaries


def _diff_boundaries(data: bytes) -> set[int]:
    boundaries: set[int] = set()
    offset = 0
    for line in data.splitlines(keepends=True):
        offset += len(line)
        if line.startswith((b"diff --git ", b"@@ ", b"GIT binary patch")):
            boundaries.add(offset - len(line))
        boundaries.add(offset)
    return boundaries


def _leaf_body(
    *,
    leaf_index: int,
    source_index: int,
    source_kind: str,
    path: str,
    data: bytes,
    start: int,
    end: int,
    alignment: str,
) -> dict[str, Any]:
    encoded = base64.b64encode(data[start:end]).decode("ascii")
    body = {
        "schema": LEGACY_LANDED_REVIEW_LEAF_SCHEMA,
        "leaf_index": leaf_index,
        "source_index": source_index,
        "source_kind": source_kind,
        "path": path,
        "byte_start": start,
        "byte_end": end,
        "byte_length": end - start,
        "payload_encoding": "base64",
        "payload": encoded,
        "payload_sha256": _sha256(data[start:end]),
        # Retained as a useful payload-only diagnostic. Admission additionally
        # checks the complete request envelope below.
        "token_upper_bound": len(encoded),
        "alignment": alignment,
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    return {**body, "leaf_id": content_identity(body)}


_CID_LENGTH_PLACEHOLDER: Final = content_identity(
    {"legacy_landed_review": "fixed-length-placeholder"}
)


def _leaf_request_fits(
    *,
    policy: LegacyLandedReviewPolicy,
    task: LegacyTaskPolicy,
    leaf: Mapping[str, Any],
) -> bool:
    manifest_placeholder = {
        "manifest_id": _CID_LENGTH_PLACEHOLDER,
        "merkle_root": _CID_LENGTH_PLACEHOLDER,
    }
    return all(
        _leaf_review_request(
            policy=policy,
            task=task,
            manifest=manifest_placeholder,
            leaf=leaf,
            provider=provider,
        ).token_upper_bound
        <= policy.max_leaf_tokens
        for provider in (policy.grok, policy.codex)
    )


def _bounded_source_chunks(
    *,
    policy: LegacyLandedReviewPolicy,
    task: LegacyTaskPolicy,
    data: bytes,
    source_index: int,
    source_kind: str,
    path: str,
    first_leaf_index: int,
    preferred_boundaries: set[int],
) -> list[tuple[int, int, str]]:
    """Fit the exact canonical provider envelope, not only leaf payload."""

    if not data:
        leaf = _leaf_body(
            leaf_index=first_leaf_index,
            source_index=source_index,
            source_kind=source_kind,
            path=path,
            data=data,
            start=0,
            end=0,
            alignment="empty",
        )
        if not _leaf_request_fits(policy=policy, task=task, leaf=leaf):
            raise LegacyLandedReviewError("legacy_provider_request_overhead_exceeds_budget")
        return [(0, 0, "empty")]
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(data):
        leaf_index = first_leaf_index + len(chunks)
        low = start + 1
        high = len(data)
        best = start
        # Canonical JSON/base64 size is monotone in payload length. Binary
        # search therefore yields the largest request admitted by both exact
        # provider/model envelopes.
        while low <= high:
            candidate_end = (low + high) // 2
            candidate = _leaf_body(
                leaf_index=leaf_index,
                source_index=source_index,
                source_kind=source_kind,
                path=path,
                data=data,
                start=start,
                end=candidate_end,
                # Size every binary-search candidate with the longer of the
                # two final alignment labels.  A candidate may subsequently
                # be moved to a preferred boundary; sizing it as
                # ``hard_limit`` here and then serializing it as
                # ``preferred_boundary`` can otherwise add enough envelope
                # bytes to cross the exact 4,096-byte request ceiling.
                alignment="preferred_boundary",
            )
            if _leaf_request_fits(policy=policy, task=task, leaf=candidate):
                best = candidate_end
                low = candidate_end + 1
            else:
                high = candidate_end - 1
        if best <= start:
            raise LegacyLandedReviewError("legacy_provider_request_overhead_exceeds_budget")
        aligned = [
            boundary
            for boundary in preferred_boundaries
            if start < boundary <= best
        ]
        end = max(aligned) if aligned else best
        alignment = "preferred_boundary" if aligned else "hard_limit"
        final_leaf = _leaf_body(
            leaf_index=leaf_index,
            source_index=source_index,
            source_kind=source_kind,
            path=path,
            data=data,
            start=start,
            end=end,
            alignment=alignment,
        )
        if not _leaf_request_fits(policy=policy, task=task, leaf=final_leaf):
            raise LegacyLandedReviewError("legacy_provider_request_budget_exceeded")
        chunks.append((start, end, alignment))
        start = end
    return chunks


def legacy_manifest_merkle_root(leaf_ids: Sequence[str]) -> str:
    if not leaf_ids or any(not isinstance(item, str) or not item for item in leaf_ids):
        raise ValueError("ordered legacy leaf identities are required")
    level = list(leaf_ids)
    height = 0
    while len(level) > 1:
        parents: list[str] = []
        for index in range(0, len(level), 2):
            children = level[index : index + 2]
            parents.append(
                content_identity(
                    {
                        "schema": LEGACY_LANDED_REVIEW_MERKLE_SCHEMA,
                        "height": height,
                        "parent_index": index // 2,
                        "children": children,
                    }
                )
            )
        level = parents
        height += 1
    return content_identity(
        {
            "schema": LEGACY_LANDED_REVIEW_MERKLE_SCHEMA,
            "leaf_count": len(leaf_ids),
            "tree_height": height,
            "ordered_root": level[0],
        }
    )


def build_legacy_landed_byte_manifest(
    policy: LegacyLandedReviewPolicy,
    binding: LegacyRepositoryBinding,
) -> dict[str, Any]:
    task = binding.task
    sources: list[dict[str, Any]] = []
    leaves: list[dict[str, Any]] = []
    raw_sources: list[tuple[str, str, bytes, str, str, bool]] = [
        ("historical_diff", "", binding.historical_diff, "", "", True)
    ]
    raw_sources.extend(
        (
            "current_blob",
            path,
            blob.data,
            blob.mode,
            blob.object_id,
            blob.exists,
        )
        for path, blob in binding.current_blobs
    )
    for source_index, (kind, path, data, mode, object_id, exists) in enumerate(raw_sources):
        boundaries = (
            _diff_boundaries(data)
            if kind == "historical_diff"
            else _python_ast_boundaries(data) if path.endswith(".py") else set()
        )
        chunks = _bounded_source_chunks(
            policy=policy,
            task=task,
            data=data,
            source_index=source_index,
            source_kind=kind,
            path=path,
            first_leaf_index=len(leaves),
            preferred_boundaries=boundaries,
        )
        first_leaf = len(leaves)
        for start, end, alignment in chunks:
            leaves.append(
                _leaf_body(
                    leaf_index=len(leaves),
                    source_index=source_index,
                    source_kind=kind,
                    path=path,
                    data=data,
                    start=start,
                    end=end,
                    alignment=alignment,
                )
            )
        source_body = {
            "source_index": source_index,
            "source_kind": kind,
            "path": path,
            "exists": exists,
            "git_mode": mode,
            "git_object_id": object_id,
            "byte_length": len(data),
            "payload_sha256": _sha256(data),
            "first_leaf_index": first_leaf,
            "leaf_count": len(chunks),
        }
        sources.append({**source_body, "source_id": content_identity(source_body)})
    leaf_ids = [item["leaf_id"] for item in leaves]
    body = {
        "schema": LEGACY_LANDED_REVIEW_MANIFEST_SCHEMA,
        "task_id": task.task_id,
        "canonical_task_key": task.canonical_task_key,
        "canonical_task_cid": task.canonical_task_cid,
        "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
        "baseline_commit": task.baseline_commit,
        "interval_commits": list(task.interval_commits),
        "implementation_commit": task.implementation_commit,
        "merge_commit": task.merge_commit,
        "current_head": binding.current_head,
        "current_tree_id": binding.current_tree_id,
        "paths": list(task.paths),
        "max_leaf_tokens": policy.max_leaf_tokens,
        "sources": sources,
        "leaves": leaves,
        "leaf_count": len(leaves),
        "merkle_algorithm": "ordered-cidv1-dag-json-sha2-256-pair-tree@1",
        "merkle_root": legacy_manifest_merkle_root(leaf_ids),
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    return {**body, "manifest_id": content_identity(body)}


@dataclass(frozen=True, slots=True)
class LegacyManifestVerification:
    verified: bool
    reason_codes: tuple[str, ...]
    manifest_id: str = ""
    merkle_root: str = ""


def verify_legacy_landed_byte_manifest(
    manifest: Mapping[str, Any],
) -> LegacyManifestVerification:
    """Prove deterministic order, byte coverage, and every CID/Merkle edge."""

    failures: list[str] = []
    try:
        payload = _strict_json_object(canonical_json_bytes(manifest))
    except (TypeError, ValueError, json.JSONDecodeError):
        return LegacyManifestVerification(False, ("legacy_manifest_invalid",))
    claimed_manifest = str(payload.get("manifest_id") or "")
    unsigned_manifest = dict(payload)
    unsigned_manifest.pop("manifest_id", None)
    if payload.get("schema") != LEGACY_LANDED_REVIEW_MANIFEST_SCHEMA:
        failures.append("legacy_manifest_schema_invalid")
    if claimed_manifest != content_identity(unsigned_manifest):
        failures.append("legacy_manifest_content_id_mismatch")
    if payload.get("completion_authoritative") is not False:
        failures.append("legacy_manifest_completion_authority_claim")
    if payload.get("proof_authoritative") is not False:
        failures.append("legacy_manifest_proof_authority_claim")
    leaves = payload.get("leaves")
    sources = payload.get("sources")
    if not isinstance(leaves, list) or not leaves:
        failures.append("legacy_manifest_leaves_missing")
        leaves = []
    if not isinstance(sources, list) or not sources:
        failures.append("legacy_manifest_sources_missing")
        sources = []
    if payload.get("leaf_count") != len(leaves):
        failures.append("legacy_manifest_leaf_count_mismatch")
    max_tokens = payload.get("max_leaf_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_LEAF_TOKENS:
        failures.append("legacy_manifest_leaf_budget_invalid")
        max_tokens = MAX_LEAF_TOKENS
    reconstructed: dict[int, bytearray] = {}
    expected_offsets: dict[int, int] = {}
    leaf_ids: list[str] = []
    seen_leaf_ids: set[str] = set()
    prior_source = -1
    for expected_index, raw_leaf in enumerate(leaves):
        if not isinstance(raw_leaf, Mapping):
            failures.append("legacy_manifest_leaf_invalid")
            continue
        leaf = dict(raw_leaf)
        leaf_id = str(leaf.pop("leaf_id", "") or "")
        if leaf.get("schema") != LEGACY_LANDED_REVIEW_LEAF_SCHEMA:
            failures.append("legacy_manifest_leaf_schema_invalid")
        if leaf_id != content_identity(leaf):
            failures.append("legacy_manifest_leaf_content_id_mismatch")
        if leaf_id in seen_leaf_ids:
            failures.append("legacy_manifest_leaf_duplicate")
        seen_leaf_ids.add(leaf_id)
        leaf_ids.append(leaf_id)
        if leaf.get("leaf_index") != expected_index:
            failures.append("legacy_manifest_leaf_reordered")
        source_index = leaf.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or not 0 <= source_index < len(sources):
            failures.append("legacy_manifest_leaf_source_invalid")
            continue
        if source_index < prior_source:
            failures.append("legacy_manifest_source_reordered")
        prior_source = source_index
        start = leaf.get("byte_start")
        end = leaf.get("byte_end")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            failures.append("legacy_manifest_leaf_range_invalid")
            continue
        if start != expected_offsets.get(source_index, 0):
            failures.append("legacy_manifest_leaf_gap_or_overlap")
        try:
            data = base64.b64decode(str(leaf.get("payload") or ""), validate=True)
        except (ValueError, TypeError):
            failures.append("legacy_manifest_leaf_payload_invalid")
            continue
        if leaf.get("payload_encoding") != "base64" or len(data) != end - start or leaf.get("byte_length") != len(data):
            failures.append("legacy_manifest_leaf_length_mismatch")
        if leaf.get("payload_sha256") != _sha256(data):
            failures.append("legacy_manifest_leaf_digest_mismatch")
        encoded_length = len(str(leaf.get("payload") or ""))
        if leaf.get("token_upper_bound") != encoded_length or encoded_length > max_tokens:
            failures.append("legacy_manifest_leaf_token_bound_invalid")
        if leaf.get("completion_authoritative") is not False or leaf.get("proof_authoritative") is not False:
            failures.append("legacy_manifest_leaf_authority_claim")
        reconstructed.setdefault(source_index, bytearray()).extend(data)
        expected_offsets[source_index] = end
    seen_paths: set[str] = set()
    expected_first_leaf = 0
    for source_index, raw_source in enumerate(sources):
        if not isinstance(raw_source, Mapping):
            failures.append("legacy_manifest_source_invalid")
            continue
        source = dict(raw_source)
        source_id = str(source.pop("source_id", "") or "")
        if source_id != content_identity(source):
            failures.append("legacy_manifest_source_content_id_mismatch")
        if source.get("source_index") != source_index:
            failures.append("legacy_manifest_source_index_mismatch")
        path = str(source.get("path") or "")
        if path:
            if path in seen_paths:
                failures.append("legacy_manifest_source_path_duplicate")
            seen_paths.add(path)
        first = source.get("first_leaf_index")
        count = source.get("leaf_count")
        if first != expected_first_leaf or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            failures.append("legacy_manifest_source_leaf_range_invalid")
            count = 0
        expected_first_leaf += count
        data = bytes(reconstructed.get(source_index, b""))
        if source.get("byte_length") != len(data) or source.get("payload_sha256") != _sha256(data):
            failures.append("legacy_manifest_source_bytes_incomplete")
        if expected_offsets.get(source_index, 0) != len(data):
            failures.append("legacy_manifest_source_coverage_invalid")
    if expected_first_leaf != len(leaves):
        failures.append("legacy_manifest_source_leaf_coverage_invalid")
    try:
        root = legacy_manifest_merkle_root(leaf_ids)
    except ValueError:
        root = ""
        failures.append("legacy_manifest_merkle_inputs_invalid")
    if root != payload.get("merkle_root"):
        failures.append("legacy_manifest_merkle_root_mismatch")
    reasons = tuple(dict.fromkeys(failures))
    return LegacyManifestVerification(
        not reasons, reasons, claimed_manifest, str(payload.get("merkle_root") or "")
    )


@dataclass(frozen=True, slots=True)
class LegacyLeafReviewRequest:
    role: str
    provider: str
    model: str
    request_id: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the exact adapter envelope sent as the provider prompt."""

        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/legacy-landed-provider-envelope@1",
            "request_id": self.request_id,
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "payload": self.payload,
        }

    @property
    def canonical_prompt(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def token_upper_bound(self) -> int:
        # The prompt is ASCII DAG-JSON (binary source is base64). A byte count
        # is a conservative upper bound for byte-fallback BPE tokenizers.
        return len(self.canonical_prompt)


@dataclass(frozen=True, slots=True)
class LegacyProviderObservation:
    """Supervisor-observed effective child, separate from model output."""

    observation_id: str
    requested_provider: str
    requested_model: str
    effective_provider: str
    effective_model: str
    provider_chain: tuple[str, ...]
    fallback_used: bool
    supervisor_observed: bool
    response: Mapping[str, Any]


class LegacyProviderInvoker(Protocol):
    def __call__(
        self, request: LegacyLeafReviewRequest
    ) -> LegacyProviderObservation: ...


class LegacyLeafCacheReview(Protocol):
    receipt: Mapping[str, Any]
    cache_hit: bool


class LegacyLeafResultCache(Protocol):
    policy: LegacyLandedReviewPolicy

    def review_leaf(
        self,
        *,
        task: LegacyTaskPolicy,
        manifest: Mapping[str, Any],
        leaf: Mapping[str, Any],
        provider: LegacyProviderPolicy,
        invoker: LegacyProviderInvoker,
        review_run_id: str,
    ) -> LegacyLeafCacheReview: ...


LegacyValidationInvoker = Callable[
    [tuple[str, ...], Path, int], subprocess.CompletedProcess[Any]
]


def _leaf_review_request(
    *,
    policy: LegacyLandedReviewPolicy,
    task: LegacyTaskPolicy,
    manifest: Mapping[str, Any],
    leaf: Mapping[str, Any],
    provider: LegacyProviderPolicy,
) -> LegacyLeafReviewRequest:
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/legacy-landed-leaf-review-request@1",
        "policy_id": policy.policy_id,
        "task_id": task.task_id,
        "canonical_task_key": task.canonical_task_key,
        "canonical_task_cid": task.canonical_task_cid,
        "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
        "baseline_commit": task.baseline_commit,
        "interval_commits": list(task.interval_commits),
        "implementation_commit": task.implementation_commit,
        "merge_commit": task.merge_commit,
        "current_head": policy.current_head,
        "paths": list(task.paths),
        "manifest_id": manifest["manifest_id"],
        "manifest_merkle_root": manifest["merkle_root"],
        "leaf": dict(leaf),
        "role": provider.role,
        "maximum_request_tokens": policy.max_leaf_tokens,
        "required_decision": "approve",
        "repair_allowed": False,
        "fallback_allowed": False,
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    request_body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/legacy-landed-provider-envelope@1",
        "role": provider.role,
        "provider": provider.provider,
        "model": provider.model,
        "payload": payload,
    }
    request_id = content_identity(request_body)
    return LegacyLeafReviewRequest(
        role=provider.role,
        provider=provider.provider,
        model=provider.model,
        request_id=request_id,
        payload=payload,
    )


def _review_one_leaf(
    *,
    request: LegacyLeafReviewRequest,
    provider: LegacyProviderPolicy,
    invoker: LegacyProviderInvoker,
    review_run_id: str,
) -> dict[str, Any]:
    if request.token_upper_bound > int(
        request.payload.get("maximum_request_tokens") or 0
    ):
        raise LegacyLandedReviewError("legacy_provider_request_budget_exceeded")
    try:
        observation = invoker(request)
    except LegacyProviderCapacitySignal:
        # Preserve only the fixed signal authored by the native boundary.  In
        # particular, ignore mutable instance/subclass fields and never attach
        # the provider exception as a cause: either could contain credentials
        # or account details when a custom invoker crosses this boundary.
        raise LegacyLandedReviewError(
            LEGACY_CODEX_USAGE_LIMIT_CAPACITY_MARKER
        ) from None
    except Exception as exc:
        raise LegacyLandedReviewError("legacy_provider_invocation_failed") from exc
    if not isinstance(observation, LegacyProviderObservation):
        raise LegacyLandedReviewError("legacy_provider_observation_missing")
    expected = (
        observation.requested_provider == provider.provider
        and observation.effective_provider == provider.provider
        and observation.requested_model == provider.model
        and observation.effective_model == provider.model
        and observation.provider_chain == (provider.provider,)
        and observation.fallback_used is False
        and observation.supervisor_observed is True
        and bool(observation.observation_id)
    )
    if not expected:
        raise LegacyLandedReviewError("legacy_effective_provider_mismatch")
    if not isinstance(observation.response, Mapping):
        raise LegacyLandedReviewError("legacy_provider_decision_invalid")
    response = _strict_json_object(canonical_json_bytes(observation.response))
    if set(response) != {"schema", "decision", "manifest_id", "leaf_id", "findings"}:
        raise LegacyLandedReviewError("legacy_provider_decision_shape_invalid")
    if response.get("schema") != LEGACY_LANDED_LEAF_DECISION_SCHEMA:
        raise LegacyLandedReviewError("legacy_provider_decision_schema_invalid")
    leaf = request.payload["leaf"]
    if (
        response.get("decision") != "approve"
        or response.get("manifest_id") != request.payload["manifest_id"]
        or response.get("leaf_id") != leaf["leaf_id"]
        or response.get("findings") != []
    ):
        raise LegacyLandedReviewError("legacy_provider_leaf_not_approved")
    body = {
        "schema": LEGACY_LANDED_LEAF_REVIEW_RECEIPT_SCHEMA,
        "review_run_id": review_run_id,
        "role": provider.role,
        "request_id": request.request_id,
        "request_token_upper_bound": request.token_upper_bound,
        "manifest_id": request.payload["manifest_id"],
        "leaf_index": leaf["leaf_index"],
        "leaf_id": leaf["leaf_id"],
        "requested_provider": provider.provider,
        "requested_model": provider.model,
        "effective_provider": observation.effective_provider,
        "effective_model": observation.effective_model,
        "provider_chain": list(observation.provider_chain),
        "fallback_used": False,
        "self_review": False,
        "supervisor_observed": True,
        "observation_id": observation.observation_id,
        "response": response,
        "response_id": content_identity(response),
        "approved": True,
        "completion_authoritative": False,
        "proof_authoritative": False,
        "provider_evidence_source": "fresh_provider",
        "provider_invoked_in_current_run": True,
        "provider_evidence_cache_record": None,
    }
    return {**body, "receipt_id": content_identity(body)}


def _build_review_aggregate(
    *,
    policy: LegacyLandedReviewPolicy,
    task: LegacyTaskPolicy,
    manifest: Mapping[str, Any],
    review_run_id: str,
    ordered_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema": LEGACY_LANDED_REVIEW_AGGREGATE_SCHEMA,
        "review_run_id": review_run_id,
        "policy_id": policy.policy_id,
        "task_id": task.task_id,
        "canonical_task_key": task.canonical_task_key,
        "canonical_task_cid": task.canonical_task_cid,
        "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
        "manifest_id": manifest["manifest_id"],
        "manifest_merkle_root": manifest["merkle_root"],
        "leaf_count": manifest["leaf_count"],
        "ordered_leaf_reviews": [dict(item) for item in ordered_pairs],
        "decision": "approve",
        "deterministic_order": "leaf_index_then_grok_then_codex",
        "fallback_used": False,
        "self_review": False,
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    return {**body, "aggregate_id": content_identity(body)}


def verify_legacy_landed_review_aggregate(
    aggregate: Mapping[str, Any],
    *,
    policy: LegacyLandedReviewPolicy,
    task: LegacyTaskPolicy,
    manifest: Mapping[str, Any],
    trusted_public_keys: Mapping[str, bytes | str] | None = None,
) -> tuple[str, ...]:
    """Reconstruct dual exact-provider approval for every ordered leaf."""

    failures: list[str] = []
    try:
        payload = _strict_json_object(canonical_json_bytes(aggregate))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ("legacy_review_aggregate_invalid",)
    aggregate_id = str(payload.get("aggregate_id") or "")
    unsigned = dict(payload)
    unsigned.pop("aggregate_id", None)
    if aggregate_id != content_identity(unsigned):
        failures.append("legacy_review_aggregate_content_id_mismatch")
    fixed = {
        "schema": LEGACY_LANDED_REVIEW_AGGREGATE_SCHEMA,
        "policy_id": policy.policy_id,
        "task_id": task.task_id,
        "canonical_task_key": task.canonical_task_key,
        "canonical_task_cid": task.canonical_task_cid,
        "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
        "manifest_id": manifest.get("manifest_id"),
        "manifest_merkle_root": manifest.get("merkle_root"),
        "leaf_count": manifest.get("leaf_count"),
        "decision": "approve",
        "deterministic_order": "leaf_index_then_grok_then_codex",
        "fallback_used": False,
        "self_review": False,
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    for field, expected in fixed.items():
        if payload.get(field) != expected:
            failures.append(f"legacy_review_aggregate_{field}_mismatch")
    run_id = payload.get("review_run_id")
    if not isinstance(run_id, str) or len(run_id) < 16:
        failures.append("legacy_review_run_id_invalid")
    pairs = payload.get("ordered_leaf_reviews")
    leaves = manifest.get("leaves")
    if not isinstance(pairs, list) or not isinstance(leaves, list) or len(pairs) != len(leaves):
        failures.append("legacy_review_leaf_coverage_mismatch")
        return tuple(dict.fromkeys(failures))
    observation_ids: set[str] = set()
    receipt_ids: set[str] = set()
    for index, (pair, leaf) in enumerate(zip(pairs, leaves, strict=True)):
        if not isinstance(pair, Mapping) or set(pair) != {
            "leaf_index", "leaf_id", "grok", "codex"
        }:
            failures.append("legacy_review_pair_shape_invalid")
            continue
        if pair.get("leaf_index") != index or pair.get("leaf_id") != leaf.get("leaf_id"):
            failures.append("legacy_review_leaf_reordered")
        for key, provider in (("grok", policy.grok), ("codex", policy.codex)):
            raw_receipt = pair.get(key)
            if not isinstance(raw_receipt, Mapping):
                failures.append("legacy_review_receipt_missing")
                continue
            receipt = dict(raw_receipt)
            base_receipt_fields = {
                "schema",
                "receipt_id",
                "review_run_id",
                "role",
                "request_id",
                "request_token_upper_bound",
                "manifest_id",
                "leaf_index",
                "leaf_id",
                "requested_provider",
                "requested_model",
                "effective_provider",
                "effective_model",
                "provider_chain",
                "fallback_used",
                "self_review",
                "supervisor_observed",
                "observation_id",
                "response",
                "response_id",
                "approved",
                "completion_authoritative",
                "proof_authoritative",
            }
            provenance_fields = {
                "provider_evidence_source",
                "provider_invoked_in_current_run",
                "provider_evidence_cache_record",
            }
            has_provenance = bool(set(receipt).intersection(provenance_fields))
            expected_fields = (
                base_receipt_fields | provenance_fields
                if has_provenance
                else base_receipt_fields
            )
            if set(receipt) != expected_fields:
                failures.append("legacy_review_receipt_shape_invalid")
            receipt_id = str(receipt.pop("receipt_id", "") or "")
            if receipt_id != content_identity(receipt):
                failures.append("legacy_review_receipt_content_id_mismatch")
            if receipt_id in receipt_ids:
                failures.append("legacy_review_receipt_duplicate")
            receipt_ids.add(receipt_id)
            request = _leaf_review_request(
                policy=policy,
                task=task,
                manifest=manifest,
                leaf=leaf,
                provider=provider,
            )
            if request.token_upper_bound > policy.max_leaf_tokens:
                failures.append("legacy_review_request_token_budget_exceeded")
            expected_receipt = {
                "schema": LEGACY_LANDED_LEAF_REVIEW_RECEIPT_SCHEMA,
                "review_run_id": run_id,
                "role": provider.role,
                "request_id": request.request_id,
                "request_token_upper_bound": request.token_upper_bound,
                "manifest_id": manifest.get("manifest_id"),
                "leaf_index": index,
                "leaf_id": leaf.get("leaf_id"),
                "requested_provider": provider.provider,
                "requested_model": provider.model,
                "effective_provider": provider.provider,
                "effective_model": provider.model,
                "provider_chain": [provider.provider],
                "fallback_used": False,
                "self_review": False,
                "supervisor_observed": True,
                "approved": True,
                "completion_authoritative": False,
                "proof_authoritative": False,
            }
            for field, expected in expected_receipt.items():
                if receipt.get(field) != expected:
                    failures.append(f"legacy_review_receipt_{field}_mismatch")
            source = receipt.get("provider_evidence_source")
            if not has_provenance:
                if source is not None:
                    failures.append("legacy_review_provider_provenance_invalid")
            elif source == "fresh_provider":
                if (
                    receipt.get("provider_invoked_in_current_run") is not True
                    or receipt.get("provider_evidence_cache_record") is not None
                ):
                    failures.append("legacy_review_fresh_provenance_invalid")
            elif source == "signed_cache":
                cache_record = receipt.get("provider_evidence_cache_record")
                if (
                    receipt.get("provider_invoked_in_current_run") is not False
                    or not isinstance(cache_record, Mapping)
                    or trusted_public_keys is None
                ):
                    failures.append("legacy_review_cache_provenance_invalid")
                else:
                    from .legacy_landed_result_cache import (
                        LegacyLandedLeafCacheKey,
                        verify_legacy_landed_leaf_cache_record,
                    )

                    try:
                        cache_key = LegacyLandedLeafCacheKey.from_request(
                            policy=policy,
                            task=task,
                            manifest=manifest,
                            leaf=leaf,
                            provider=provider,
                            request=request,
                        )
                        cache_verification = (
                            verify_legacy_landed_leaf_cache_record(
                                cache_record,
                                expected_key=cache_key,
                                trusted_public_keys=trusted_public_keys,
                            )
                        )
                        if (
                            not cache_verification.verified
                            or cache_verification.record is None
                        ):
                            raise ValueError("signed cache record did not verify")
                        rebound = dict(cache_verification.record.receipt)
                        rebound.pop("receipt_id", None)
                        rebound["review_run_id"] = run_id
                        rebound["provider_evidence_source"] = "signed_cache"
                        rebound["provider_invoked_in_current_run"] = False
                        rebound["provider_evidence_cache_record"] = dict(
                            cache_record
                        )
                        expected_rebound = {
                            **rebound,
                            "receipt_id": content_identity(rebound),
                        }
                        if canonical_json_bytes(expected_rebound) != (
                            canonical_json_bytes(raw_receipt)
                        ):
                            raise ValueError("cache receipt rebind is invalid")
                    except (TypeError, ValueError):
                        failures.append("legacy_review_cache_signature_invalid")
            else:
                failures.append("legacy_review_provider_provenance_invalid")
            response = receipt.get("response")
            if not isinstance(response, Mapping) or set(response) != {
                "schema", "decision", "manifest_id", "leaf_id", "findings"
            }:
                failures.append("legacy_review_response_shape_invalid")
            else:
                expected_response = {
                    "schema": LEGACY_LANDED_LEAF_DECISION_SCHEMA,
                    "decision": "approve",
                    "manifest_id": manifest.get("manifest_id"),
                    "leaf_id": leaf.get("leaf_id"),
                    "findings": [],
                }
                if dict(response) != expected_response:
                    failures.append("legacy_review_response_not_exact_approval")
                if receipt.get("response_id") != content_identity(response):
                    failures.append("legacy_review_response_content_id_mismatch")
            observation_id = receipt.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                failures.append("legacy_review_observation_missing")
            elif observation_id in observation_ids:
                failures.append("legacy_review_observation_reused")
            else:
                observation_ids.add(observation_id)
    return tuple(dict.fromkeys(failures))


def _scope_adjudication_receipt(
    *,
    policy: LegacyLandedReviewPolicy,
    task: LegacyTaskPolicy,
    review_run_id: str,
) -> dict[str, Any] | None:
    scope = task.scope_adjudication
    if scope is None:
        return None
    body = {
        "schema": LEGACY_LANDED_SCOPE_ADJUDICATION_SCHEMA,
        "review_run_id": review_run_id,
        "policy_id": policy.policy_id,
        "task_id": task.task_id,
        "canonical_task_key": task.canonical_task_key,
        "canonical_task_cid": task.canonical_task_cid,
        "baseline_commit": task.baseline_commit,
        "interval_commits": list(task.interval_commits),
        "implementation_commit": task.implementation_commit,
        "merge_commit": task.merge_commit,
        "current_head": policy.current_head,
        "paths": list(task.paths),
        "reason_code": scope.reason_code,
        "justification": scope.justification,
        "decision": "allow_exact_pinned_scope",
        "scope_widened": False,
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    return {**body, "receipt_id": content_identity(body)}


DEFAULT_LEGACY_VALIDATION_TIMEOUT_SECONDS: Final = 1_800


def _default_validation_invoker(
    argv: tuple[str, ...], repo_root: Path, timeout_seconds: int
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
        env=dict(os.environ),
    )


def _output_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace")
    raise LegacyLandedReviewError("legacy_validation_output_invalid")


@dataclass(frozen=True, slots=True)
class LegacyLandedReviewResult:
    status: str
    reason_code: str
    policy_id: str
    task_id: str
    attestation: LegacyLandedReviewAttestation | None = None
    manifest: Mapping[str, Any] | None = None
    review_aggregate: Mapping[str, Any] | None = None
    scope_adjudication_receipt: Mapping[str, Any] | None = None
    validation_receipts: tuple[Mapping[str, Any], ...] = ()

    @property
    def reviewed(self) -> bool:
        return self.status == "reviewed" and self.attestation is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "attestation": self.attestation.to_dict() if self.attestation else None,
            "manifest": dict(self.manifest) if self.manifest else None,
            "review_aggregate": dict(self.review_aggregate) if self.review_aggregate else None,
            "scope_adjudication_receipt": (
                dict(self.scope_adjudication_receipt)
                if self.scope_adjudication_receipt else None
            ),
            "validation_receipts": [dict(item) for item in self.validation_receipts],
            "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
            "provider_execution_receipt": None,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> LegacyLandedReviewResult:
        """Parse the closed public result envelope used at admission."""

        payload = _strict_json_object(canonical_json_bytes(value))
        if set(payload) != {
            "status",
            "reason_code",
            "policy_id",
            "task_id",
            "attestation",
            "manifest",
            "review_aggregate",
            "scope_adjudication_receipt",
            "validation_receipts",
            "historical_provider",
            "provider_execution_receipt",
            "completion_authoritative",
            "proof_authoritative",
        }:
            raise ValueError("legacy landed review result shape is invalid")
        if payload.get("historical_provider") != HISTORICAL_PROVIDER_UNVERIFIED:
            raise ValueError("legacy historical provider must remain unverified")
        if payload.get("provider_execution_receipt") is not None:
            raise ValueError("legacy result cannot contain a provider receipt")
        if payload.get("completion_authoritative") is not False:
            raise ValueError("legacy result cannot claim completion authority")
        if payload.get("proof_authoritative") is not False:
            raise ValueError("legacy result cannot claim proof authority")
        status = payload.get("status")
        if status not in {"disabled", "rejected", "reviewed"}:
            raise ValueError("legacy landed review result status is invalid")
        for field in ("reason_code", "policy_id", "task_id"):
            if not isinstance(payload.get(field), str) or not payload[field].strip():
                raise ValueError(f"legacy landed review result {field} is invalid")
        validations = payload.get("validation_receipts")
        if not isinstance(validations, list) or any(
            not isinstance(item, Mapping) for item in validations
        ):
            raise ValueError("legacy validation receipt collection is invalid")
        raw_attestation = payload.get("attestation")
        attestation = (
            LegacyLandedReviewAttestation.from_dict(raw_attestation)
            if isinstance(raw_attestation, Mapping)
            else None
        )
        manifest = payload.get("manifest")
        aggregate = payload.get("review_aggregate")
        scope = payload.get("scope_adjudication_receipt")
        reviewed_shape = (
            attestation is not None
            and isinstance(manifest, Mapping)
            and isinstance(aggregate, Mapping)
            and (scope is None or isinstance(scope, Mapping))
            and bool(validations)
        )
        if status == "reviewed" and not reviewed_shape:
            raise ValueError("reviewed legacy result evidence is incomplete")
        if status != "reviewed" and any(
            item is not None for item in (raw_attestation, manifest, aggregate, scope)
        ):
            raise ValueError("non-reviewed legacy result contains evidence")
        if status != "reviewed" and validations:
            raise ValueError("non-reviewed legacy result contains validations")
        return cls(
            status=status,
            reason_code=payload["reason_code"].strip(),
            policy_id=payload["policy_id"].strip(),
            task_id=payload["task_id"].strip(),
            attestation=attestation,
            manifest=dict(manifest) if isinstance(manifest, Mapping) else None,
            review_aggregate=(
                dict(aggregate) if isinstance(aggregate, Mapping) else None
            ),
            scope_adjudication_receipt=(
                dict(scope) if isinstance(scope, Mapping) else None
            ),
            validation_receipts=tuple(dict(item) for item in validations),
        )


class LegacyLandedReviewService:
    """One operator-pinned, no-caller-overrides migration review service."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        operator_policy_path: str | Path,
        operator_key_path: str | Path,
        grok_invoker: LegacyProviderInvoker | None,
        codex_invoker: LegacyProviderInvoker | None,
        validation_invoker: LegacyValidationInvoker | None = None,
        clock_ms: Callable[[], int] | None = None,
        leaf_result_cache: LegacyLeafResultCache | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve(strict=True)
        # Preserve the operator's exact path so the strict loaders can reject
        # a symlink instead of silently resolving through it.
        self._policy_path = Path(os.path.abspath(operator_policy_path))
        self._key_path = Path(os.path.abspath(operator_key_path))
        self._policy = load_legacy_landed_review_policy(self._policy_path)
        self._authority = LegacyLandedReviewAuthority.from_private_key_path(
            self._key_path
        )
        if self._authority.issuer_key_id != self._policy.issuer_key_id:
            raise ValueError("legacy operator policy/key binding is invalid")
        self._grok_invoker = grok_invoker
        self._codex_invoker = codex_invoker
        self._validation_invoker = validation_invoker or _default_validation_invoker
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        if leaf_result_cache is not None and (
            leaf_result_cache.policy != self._policy
        ):
            raise ValueError("legacy leaf cache policy binding is invalid")
        self._leaf_result_cache = leaf_result_cache

    @property
    def policy(self) -> LegacyLandedReviewPolicy:
        return self._policy

    @property
    def trusted_public_key(self) -> bytes:
        return self._authority.public_key_bytes

    def _fence(self) -> None:
        if not _repo_is_clean(self._repo_root):
            raise LegacyLandedReviewError("legacy_repository_not_clean")
        head = _repo_head(self._repo_root)
        if head != self._policy.current_head or _tree_id(self._repo_root, head) != self._policy.current_tree_id:
            raise LegacyLandedReviewError("legacy_repository_head_fence_failed")

    def _validations(
        self,
        *,
        task: LegacyTaskPolicy,
        review_run_id: str,
    ) -> tuple[dict[str, Any], ...]:
        receipts: list[dict[str, Any]] = []
        for index, argv in enumerate(task.validations):
            self._fence()
            started = self._clock_ms()
            try:
                result = self._validation_invoker(
                    argv,
                    self._repo_root,
                    DEFAULT_LEGACY_VALIDATION_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise LegacyLandedReviewError("legacy_validation_timed_out") from exc
            except Exception as exc:
                raise LegacyLandedReviewError("legacy_validation_execution_failed") from exc
            finished = self._clock_ms()
            self._fence()
            observed_args = result.args
            if isinstance(observed_args, str):
                observed_argv = (observed_args,)
            else:
                observed_argv = tuple(str(item) for item in observed_args)
            if observed_argv != argv:
                raise LegacyLandedReviewError("legacy_validation_command_mismatch")
            if isinstance(result.returncode, bool) or result.returncode != 0:
                raise LegacyLandedReviewError("legacy_validation_failed")
            stdout = _output_bytes(result.stdout)
            stderr = _output_bytes(result.stderr)
            body = {
                "schema": LEGACY_LANDED_VALIDATION_RECEIPT_SCHEMA,
                "review_run_id": review_run_id,
                "policy_id": self._policy.policy_id,
                "task_id": task.task_id,
                "canonical_task_key": task.canonical_task_key,
                "canonical_task_cid": task.canonical_task_cid,
                "validation_index": index,
                "argv": list(argv),
                "shell": False,
                "timeout_seconds": DEFAULT_LEGACY_VALIDATION_TIMEOUT_SECONDS,
                "returncode": 0,
                "stdout_sha256": _sha256(stdout),
                "stderr_sha256": _sha256(stderr),
                "baseline_commit": task.baseline_commit,
                "implementation_commit": task.implementation_commit,
                "merge_commit": task.merge_commit,
                "current_head_before": self._policy.current_head,
                "current_head_after": self._policy.current_head,
                "current_tree_id": self._policy.current_tree_id,
                "started_at_ms": started,
                "finished_at_ms": finished,
                "fresh_execution": True,
                "cached": False,
                "completion_authoritative": False,
                "proof_authoritative": False,
            }
            receipts.append({**body, "receipt_id": content_identity(body)})
        return tuple(receipts)

    def review(self, task_id: str) -> LegacyLandedReviewResult:
        """Review one pinned task; callers can supply only its public task ID."""

        normalized_task = str(task_id or "").strip()
        if not self._policy.enabled:
            return LegacyLandedReviewResult(
                "disabled",
                "legacy_landed_review_disabled",
                self._policy.policy_id,
                normalized_task,
            )
        try:
            task = self._policy.task(normalized_task)
            if self._grok_invoker is None or self._codex_invoker is None:
                raise LegacyLandedReviewError("legacy_review_providers_unavailable")
            self._fence()
            binding = inspect_legacy_repository_binding(
                self._repo_root, self._policy, task
            )
            manifest = build_legacy_landed_byte_manifest(self._policy, binding)
            verification = verify_legacy_landed_byte_manifest(manifest)
            if not verification.verified:
                raise LegacyLandedReviewError("legacy_manifest_verification_failed")
            review_run_id = "legacy-review:" + os.urandom(24).hex()
            pairs: list[dict[str, Any]] = []
            observation_ids: set[str] = set()
            for leaf in manifest["leaves"]:
                reviews: dict[str, Any] = {
                    "leaf_index": leaf["leaf_index"],
                    "leaf_id": leaf["leaf_id"],
                }
                for key, provider, invoker in (
                    ("grok", self._policy.grok, self._grok_invoker),
                    ("codex", self._policy.codex, self._codex_invoker),
                ):
                    if self._leaf_result_cache is None:
                        request = _leaf_review_request(
                            policy=self._policy,
                            task=task,
                            manifest=manifest,
                            leaf=leaf,
                            provider=provider,
                        )
                        receipt = _review_one_leaf(
                            request=request,
                            provider=provider,
                            invoker=invoker,
                            review_run_id=review_run_id,
                        )
                    else:
                        try:
                            cached_review = self._leaf_result_cache.review_leaf(
                                task=task,
                                manifest=manifest,
                                leaf=leaf,
                                provider=provider,
                                invoker=invoker,
                                review_run_id=review_run_id,
                            )
                            receipt = dict(cached_review.receipt)
                        except LegacyLandedReviewError:
                            # Provider invocation and decision failures already
                            # carry a closed, secret-safe reason code.  Preserve
                            # it instead of misreporting every cold-cache
                            # provider failure as DuckDB corruption.
                            raise
                        except Exception as exc:
                            raise LegacyLandedReviewError(
                                "legacy_landed_leaf_cache_failed"
                            ) from exc
                    observation_id = str(receipt["observation_id"])
                    if observation_id in observation_ids:
                        raise LegacyLandedReviewError("legacy_provider_self_review_detected")
                    observation_ids.add(observation_id)
                    reviews[key] = receipt
                    self._fence()
                pairs.append(reviews)
            aggregate = _build_review_aggregate(
                policy=self._policy,
                task=task,
                manifest=manifest,
                review_run_id=review_run_id,
                ordered_pairs=pairs,
            )
            aggregate_failures = verify_legacy_landed_review_aggregate(
                aggregate,
                policy=self._policy,
                task=task,
                manifest=manifest,
                trusted_public_keys={
                    self._policy.issuer_key_id: self.trusted_public_key
                },
            )
            if aggregate_failures:
                raise LegacyLandedReviewError("legacy_review_aggregate_verification_failed")
            scope_receipt = _scope_adjudication_receipt(
                policy=self._policy,
                task=task,
                review_run_id=review_run_id,
            )
            validation_receipts = self._validations(
                task=task, review_run_id=review_run_id
            )
            # Reconstruct the immutable bytes after all external executions.
            final_binding = inspect_legacy_repository_binding(
                self._repo_root, self._policy, task
            )
            final_manifest = build_legacy_landed_byte_manifest(
                self._policy, final_binding
            )
            if canonical_json_bytes(final_manifest) != canonical_json_bytes(manifest):
                raise LegacyLandedReviewError("legacy_manifest_changed_during_review")
            attestation = self._authority.issue(
                policy_id=self._policy.policy_id,
                task_id=task.task_id,
                canonical_task_key=task.canonical_task_key,
                canonical_task_cid=task.canonical_task_cid,
                baseline_commit=task.baseline_commit,
                interval_commits=task.interval_commits,
                implementation_commit=task.implementation_commit,
                merge_commit=task.merge_commit,
                current_head=self._policy.current_head,
                current_tree_id=self._policy.current_tree_id,
                paths=task.paths,
                manifest_id=manifest["manifest_id"],
                manifest_merkle_root=manifest["merkle_root"],
                review_aggregate_id=aggregate["aggregate_id"],
                scope_adjudication_receipt_id=(
                    str(scope_receipt["receipt_id"]) if scope_receipt else ""
                ),
                validation_receipt_ids=[
                    str(item["receipt_id"]) for item in validation_receipts
                ],
                issued_at_ms=self._clock_ms(),
                nonce=review_run_id,
            )
            self._fence()
            return LegacyLandedReviewResult(
                "reviewed",
                "legacy_landed_review_attested",
                self._policy.policy_id,
                task.task_id,
                attestation=attestation,
                manifest=manifest,
                review_aggregate=aggregate,
                scope_adjudication_receipt=scope_receipt,
                validation_receipts=validation_receipts,
            )
        except (LegacyLandedReviewError, ValueError) as exc:
            reason = (
                exc.reason_code
                if isinstance(exc, LegacyLandedReviewError)
                else "legacy_landed_review_evidence_invalid"
            )
            return LegacyLandedReviewResult(
                "rejected", reason, self._policy.policy_id, normalized_task
            )


def verify_legacy_landed_review_result(
    result: LegacyLandedReviewResult,
    *,
    repo_root: str | Path,
    policy: LegacyLandedReviewPolicy,
    trusted_public_keys: Mapping[str, bytes | str],
) -> tuple[str, ...]:
    """High-level semantic verification for a future completion admission gate."""

    from .legacy_landed_attestation import (
        verify_legacy_landed_review_attestation,
    )

    failures: list[str] = []
    if not result.reviewed or result.attestation is None:
        return ("legacy_landed_review_result_not_reviewed",)
    resolved_repo = Path(repo_root).resolve()
    try:
        if (
            not _repo_is_clean(resolved_repo)
            or _repo_head(resolved_repo) != policy.current_head
            or _tree_id(resolved_repo, policy.current_head)
            != policy.current_tree_id
        ):
            return ("legacy_landed_review_repository_fence_failed",)
    except (LegacyLandedReviewError, OSError, ValueError):
        return ("legacy_landed_review_repository_fence_failed",)
    try:
        task = policy.task(result.task_id)
        binding = inspect_legacy_repository_binding(resolved_repo, policy, task)
        rebuilt_manifest = build_legacy_landed_byte_manifest(policy, binding)
    except (LegacyLandedReviewError, ValueError):
        return ("legacy_landed_review_repository_reverification_failed",)
    if result.manifest is None or canonical_json_bytes(result.manifest) != canonical_json_bytes(rebuilt_manifest):
        failures.append("legacy_landed_review_manifest_reverification_failed")
        manifest = rebuilt_manifest
    else:
        manifest = result.manifest
    manifest_verification = verify_legacy_landed_byte_manifest(manifest)
    failures.extend(manifest_verification.reason_codes)
    if result.review_aggregate is None:
        failures.append("legacy_landed_review_aggregate_missing")
        aggregate: Mapping[str, Any] = {}
    else:
        aggregate = result.review_aggregate
        failures.extend(
            verify_legacy_landed_review_aggregate(
                aggregate,
                policy=policy,
                task=task,
                manifest=manifest,
                trusted_public_keys=trusted_public_keys,
            )
        )
    run_id = str(aggregate.get("review_run_id") or "")
    expected_scope = _scope_adjudication_receipt(
        policy=policy, task=task, review_run_id=run_id
    )
    observed_scope_bytes = (
        canonical_json_bytes(result.scope_adjudication_receipt)
        if result.scope_adjudication_receipt is not None
        else None
    )
    expected_scope_bytes = (
        canonical_json_bytes(expected_scope) if expected_scope is not None else None
    )
    if observed_scope_bytes != expected_scope_bytes:
        failures.append("legacy_landed_review_scope_adjudication_mismatch")
    if (task.task_id in SCOPE_ADJUDICATION_TASK_IDS) != (result.scope_adjudication_receipt is not None):
        failures.append("legacy_landed_review_scope_adjudication_presence_invalid")
    if len(result.validation_receipts) != len(task.validations):
        failures.append("legacy_landed_review_validation_count_mismatch")
    else:
        prior_finished = -1
        for index, (receipt, argv) in enumerate(
            zip(result.validation_receipts, task.validations, strict=True)
        ):
            body = dict(receipt)
            receipt_id = str(body.pop("receipt_id", "") or "")
            if receipt_id != content_identity(body):
                failures.append("legacy_landed_review_validation_content_id_mismatch")
            expected = {
                "schema": LEGACY_LANDED_VALIDATION_RECEIPT_SCHEMA,
                "review_run_id": run_id,
                "policy_id": policy.policy_id,
                "task_id": task.task_id,
                "canonical_task_key": task.canonical_task_key,
                "canonical_task_cid": task.canonical_task_cid,
                "validation_index": index,
                "argv": list(argv),
                "shell": False,
                "timeout_seconds": DEFAULT_LEGACY_VALIDATION_TIMEOUT_SECONDS,
                "returncode": 0,
                "baseline_commit": task.baseline_commit,
                "implementation_commit": task.implementation_commit,
                "merge_commit": task.merge_commit,
                "current_head_before": policy.current_head,
                "current_head_after": policy.current_head,
                "current_tree_id": policy.current_tree_id,
                "fresh_execution": True,
                "cached": False,
                "completion_authoritative": False,
                "proof_authoritative": False,
            }
            for field, expected_value in expected.items():
                if receipt.get(field) != expected_value:
                    failures.append(f"legacy_landed_review_validation_{field}_mismatch")
            started = receipt.get("started_at_ms")
            finished = receipt.get("finished_at_ms")
            if (
                isinstance(started, bool)
                or isinstance(finished, bool)
                or not isinstance(started, int)
                or not isinstance(finished, int)
                or started < 1
                or finished < started
                or started < prior_finished
            ):
                failures.append("legacy_landed_review_validation_freshness_invalid")
            else:
                prior_finished = finished
            for digest_field in ("stdout_sha256", "stderr_sha256"):
                if not isinstance(receipt.get(digest_field), str) or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}", str(receipt.get(digest_field))
                ):
                    failures.append("legacy_landed_review_validation_digest_invalid")
    attestation_verification = verify_legacy_landed_review_attestation(
        result.attestation,
        trusted_public_keys=trusted_public_keys,
        expected_policy_id=policy.policy_id,
        expected_task_id=task.task_id,
        expected_canonical_task_key=task.canonical_task_key,
        expected_canonical_task_cid=task.canonical_task_cid,
        expected_current_head=policy.current_head,
        expected_current_tree_id=policy.current_tree_id,
        manifest=manifest,
        review_aggregate=aggregate,
        validation_receipts=result.validation_receipts,
        scope_adjudication_receipt=result.scope_adjudication_receipt,
    )
    failures.extend(attestation_verification.reason_codes)
    attestation = result.attestation
    if tuple(attestation.interval_commits) != task.interval_commits:
        failures.append("legacy_landed_review_attested_interval_mismatch")
    if tuple(attestation.paths) != task.paths:
        failures.append("legacy_landed_review_attested_paths_mismatch")
    if attestation.baseline_commit != task.baseline_commit or attestation.implementation_commit != task.implementation_commit or attestation.merge_commit != task.merge_commit:
        failures.append("legacy_landed_review_attested_commit_mismatch")
    if attestation.nonce != run_id:
        failures.append("legacy_landed_review_run_binding_mismatch")
    if result.scope_adjudication_receipt is not None and attestation.scope_adjudication_receipt_id != result.scope_adjudication_receipt.get("receipt_id"):
        failures.append("legacy_landed_review_attested_scope_mismatch")
    reasons = tuple(dict.fromkeys(failures))
    return reasons


__all__ = [
    "DEFAULT_LEGACY_VALIDATION_TIMEOUT_SECONDS",
    "EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE",
    "EXACT_LEGACY_LANDED_TASK_IDS",
    "LEGACY_LANDED_LEAF_DECISION_SCHEMA",
    "LEGACY_LANDED_REVIEW_POLICY_INTERFACE",
    "LEGACY_LANDED_REVIEW_POLICY_SCHEMA",
    "LEGACY_CODEX_USAGE_LIMIT_CAPACITY_MARKER",
    "LegacyLandedReviewError",
    "LegacyLandedReviewPolicy",
    "LegacyLandedReviewResult",
    "LegacyLandedReviewService",
    "LegacyLeafReviewRequest",
    "LegacyManifestVerification",
    "LegacyProviderCapacitySignal",
    "LegacyProviderObservation",
    "LegacyRepositoryBinding",
    "LegacyTaskPolicy",
    "build_exact_eight_legacy_landed_policy",
    "build_legacy_landed_byte_manifest",
    "inspect_legacy_repository_binding",
    "legacy_manifest_merkle_root",
    "load_legacy_landed_review_policy",
    "parse_legacy_landed_review_policy",
    "verify_legacy_landed_byte_manifest",
    "verify_legacy_landed_review_aggregate",
    "verify_legacy_landed_review_result",
]
