"""Fail-closed legacy landed-review migration tests."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    implementation_supervisor_command,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    legacy_landed_provider_cli as legacy_cli,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import legacy_landed_review as legacy
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    classify_provider_capacity_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_args as parse_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_landed_attestation import (
    LegacyLandedReviewAttestation,
    legacy_landed_review_key_id,
    verify_legacy_landed_review_attestation,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_landed_provider_cli import (
    build_legacy_landed_cli_provider_pair,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_USAGE_MODE_ENFORCE,
    LlmChildResultEnvelope,
)

HEAD = "a" * 40
TREE = "b" * 40


def _private_key_file(path: Path) -> tuple[Ed25519PrivateKey, str]:
    private = Ed25519PrivateKey.generate()
    raw = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    path.write_bytes(raw)
    path.chmod(0o600)
    public = private.public_key().public_bytes_raw()
    return private, legacy_landed_review_key_id(public)


def _policy_payload(
    *,
    issuer_key_id: str,
    enabled: bool = True,
    current_head: str = HEAD,
    current_tree_id: str = TREE,
) -> dict[str, Any]:
    template = copy.deepcopy(legacy.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE)
    body = {
        "schema": legacy.LEGACY_LANDED_REVIEW_POLICY_SCHEMA,
        "interface": legacy.LEGACY_LANDED_REVIEW_POLICY_INTERFACE,
        "enabled": enabled,
        "issuer_key_id": issuer_key_id,
        "current_head": current_head,
        "current_tree_id": current_tree_id,
        "max_leaf_tokens": template["max_leaf_tokens"],
        "providers": template["providers"],
        "tasks": template["tasks"],
        "historical_provider": "unverified",
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    return {**body, "policy_id": content_identity(body)}


def _write_policy(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _binding(task: legacy.LegacyTaskPolicy) -> legacy.LegacyRepositoryBinding:
    blobs = tuple(
        (
            path,
            legacy._GitBlob(  # noqa: SLF001 - exact byte fixture
                True,
                "100644",
                f"{index + 1:040x}",
                (
                    f"def reviewed_{index}():\n"
                    f"    return {task.task_id!r}\n"
                ).encode(),
            ),
        )
        for index, path in enumerate(task.paths)
    )
    return legacy.LegacyRepositoryBinding(
        task=task,
        current_head=HEAD,
        current_tree_id=TREE,
        historical_diff=(
            b"diff --git a/source.py b/source.py\n"
            b"--- a/source.py\n+++ b/source.py\n"
            b"@@ -1 +1 @@\n-old\n+new\n"
        ),
        current_blobs=blobs,
    )


def _fake_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(legacy, "_repo_is_clean", lambda _repo: True)
    monkeypatch.setattr(legacy, "_repo_head", lambda _repo: HEAD)
    monkeypatch.setattr(legacy, "_tree_id", lambda _repo, _head: TREE)
    monkeypatch.setattr(
        legacy,
        "inspect_legacy_repository_binding",
        lambda _repo, _policy, task, **_kwargs: _binding(task),
    )


class _ApprovingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, request: legacy.LegacyLeafReviewRequest
    ) -> legacy.LegacyProviderObservation:
        self.calls += 1
        leaf = request.payload["leaf"]
        return legacy.LegacyProviderObservation(
            observation_id=f"{request.role}:observation:{self.calls}",
            requested_provider=request.provider,
            requested_model=request.model,
            effective_provider=request.provider,
            effective_model=request.model,
            provider_chain=(request.provider,),
            fallback_used=False,
            supervisor_observed=True,
            response={
                "schema": legacy.LEGACY_LANDED_LEAF_DECISION_SCHEMA,
                "decision": "approve",
                "manifest_id": request.payload["manifest_id"],
                "leaf_id": leaf["leaf_id"],
                "findings": [],
            },
            requested_reasoning_effort=request.reasoning_effort,
            effective_reasoning_effort=request.reasoning_effort,
        )


def test_exact_template_pins_all_eight_and_original_ase_023_interval() -> None:
    template = legacy.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE
    tasks = {item["task_id"]: item for item in template["tasks"]}

    assert tuple(tasks) == legacy.EXACT_LEGACY_LANDED_TASK_IDS
    assert template["schema"].endswith("@2")
    assert template["providers"]["grok"]["model"] == "grok-4.5"
    assert template["providers"]["codex"]["model"] == "gpt-5.6-terra"
    assert template["providers"]["codex"]["reasoning_effort"] == "medium"
    assert tasks["ASE-023"]["baseline_commit"] == (
        "27cc4219f67358d90abd36b08b37950be344009e"
    )
    assert tasks["ASE-023"]["interval_commits"] == [
        "aa140915915120f92bbc3738e6961f64e620dcba",
        "4815d296926a7b980200a301a711162d82165612",
    ]
    assert tasks["ASE-023"]["merge_commit"] == (
        "07d5cd3791855100d481c1476ef0500ba2ba514a"
    )
    assert tasks["ASE-009"]["scope_adjudication"] is not None
    assert tasks["ASE-038"]["scope_adjudication"] is not None
    assert all(
        (item["scope_adjudication"] is not None)
        == (item["task_id"] in {"ASE-009", "ASE-038"})
        for item in template["tasks"]
    )
    assert "current_head" not in template
    assert template["completion_authoritative"] is False
    assert template["proof_authoritative"] is False


def test_production_parser_rejects_template_placeholders_and_any_scope_widening(
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    valid = _policy_payload(issuer_key_id=issuer)
    parsed = legacy.parse_legacy_landed_review_policy(valid)
    assert parsed.enabled is True
    assert parsed.current_head == HEAD

    with pytest.raises(ValueError):
        legacy.parse_legacy_landed_review_policy(
            legacy.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE
        )

    mutations = []
    changed_commit = copy.deepcopy(valid)
    changed_commit["tasks"][0]["baseline_commit"] = "c" * 40
    mutations.append(changed_commit)
    changed_path = copy.deepcopy(valid)
    changed_path["tasks"][0]["paths"][0] = "extra.py"
    mutations.append(changed_path)
    changed_validation = copy.deepcopy(valid)
    changed_validation["tasks"][0]["validations"] = [["true"]]
    mutations.append(changed_validation)
    changed_provider = copy.deepcopy(valid)
    changed_provider["providers"]["grok"]["model"] = "another-model"
    mutations.append(changed_provider)
    old_schema = copy.deepcopy(valid)
    old_schema["schema"] = (
        "ipfs_accelerate_py/agent-supervisor/legacy-landed-review-policy@1"
    )
    old_schema["interface"] = "LegacyLandedReviewPolicy@1"
    mutations.append(old_schema)
    missing_reasoning = copy.deepcopy(valid)
    missing_reasoning["providers"]["codex"].pop("reasoning_effort")
    mutations.append(missing_reasoning)
    changed_budget = copy.deepcopy(valid)
    changed_budget["max_leaf_tokens"] = 4095
    mutations.append(changed_budget)
    for payload in mutations:
        unsigned = dict(payload)
        unsigned.pop("policy_id", None)
        payload["policy_id"] = content_identity(unsigned)
        with pytest.raises(ValueError):
            legacy.parse_legacy_landed_review_policy(payload)

    placeholder = copy.deepcopy(valid)
    placeholder["current_head"] = "__PIN_AT_DEPLOYMENT__"
    unsigned = dict(placeholder)
    unsigned.pop("policy_id")
    placeholder["policy_id"] = content_identity(unsigned)
    with pytest.raises(ValueError):
        legacy.parse_legacy_landed_review_policy(placeholder)


def test_manifest_is_byte_complete_bounded_and_detects_gap_duplicate_reorder(
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    binding = _binding(task)
    manifest = legacy.build_legacy_landed_byte_manifest(policy, binding)
    verified = legacy.verify_legacy_landed_byte_manifest(manifest)

    assert verified.verified is True
    assert manifest["leaf_count"] == len(manifest["leaves"])
    assert all(
        item["token_upper_bound"] <= 4096 for item in manifest["leaves"]
    )
    reconstructed: dict[int, bytes] = {}
    for leaf in manifest["leaves"]:
        reconstructed.setdefault(leaf["source_index"], b"")
        reconstructed[leaf["source_index"]] += __import__("base64").b64decode(
            leaf["payload"]
        )
    assert reconstructed[0] == binding.historical_diff
    assert [reconstructed[index + 1] for index in range(len(binding.current_blobs))] == [
        blob.data for _path, blob in binding.current_blobs
    ]

    gap = copy.deepcopy(manifest)
    gap["leaves"][0]["byte_start"] = 1
    assert "legacy_manifest_leaf_gap_or_overlap" in (
        legacy.verify_legacy_landed_byte_manifest(gap).reason_codes
    )

    duplicate = copy.deepcopy(manifest)
    duplicate["leaves"].append(copy.deepcopy(duplicate["leaves"][0]))
    duplicate["leaf_count"] += 1
    assert "legacy_manifest_leaf_duplicate" in (
        legacy.verify_legacy_landed_byte_manifest(duplicate).reason_codes
    )

    reordered = copy.deepcopy(manifest)
    reordered["leaves"] = list(reversed(reordered["leaves"]))
    assert "legacy_manifest_leaf_reordered" in (
        legacy.verify_legacy_landed_byte_manifest(reordered).reason_codes
    )


def test_near_limit_chunks_bound_the_full_grok_and_codex_adapter_envelopes(
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    ordinary = _binding(task)
    large = replace(
        ordinary,
        historical_diff=b"x" * 25_000,
        current_blobs=tuple(
            (path, replace(blob, data=(b"value = 1\n" * 2_000)))
            for path, blob in ordinary.current_blobs
        ),
    )
    manifest = legacy.build_legacy_landed_byte_manifest(policy, large)
    bounds: list[int] = []
    for leaf in manifest["leaves"]:
        for provider in (policy.grok, policy.codex):
            request = legacy._leaf_review_request(  # noqa: SLF001
                policy=policy,
                task=task,
                manifest=manifest,
                leaf=leaf,
                provider=provider,
            )
            assert request.canonical_prompt == canonical_json_bytes(
                request.to_dict()
            )
            bounds.append(request.token_upper_bound)

    assert len(manifest["leaves"]) > 10
    assert max(bounds) <= 4_096
    assert max(bounds) >= 4_080


def test_preferred_boundary_label_is_included_in_chunk_budget(
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    data = b"x" * 25_000
    chunks = legacy._bounded_source_chunks(  # noqa: SLF001
        policy=policy,
        task=task,
        data=data,
        source_index=0,
        source_kind="historical_diff",
        path="",
        first_leaf_index=0,
        # Force the chosen maximum to become a preferred boundary.  The
        # serialized label is longer than ``hard_limit`` and therefore must
        # be accounted for during the binary search itself.
        preferred_boundaries=set(range(1, len(data) + 1)),
    )

    for leaf_index, (start, end, alignment) in enumerate(chunks):
        assert alignment == "preferred_boundary"
        leaf = legacy._leaf_body(  # noqa: SLF001
            leaf_index=leaf_index,
            source_index=0,
            source_kind="historical_diff",
            path="",
            data=data,
            start=start,
            end=end,
            alignment=alignment,
        )
        assert legacy._leaf_request_fits(  # noqa: SLF001
            policy=policy,
            task=task,
            leaf=leaf,
        )


def test_default_off_never_invokes_providers_or_validations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "authority.key"
    _private, issuer = _private_key_file(key_path)
    policy_path = tmp_path / "policy.json"
    _write_policy(
        policy_path, _policy_payload(issuer_key_id=issuer, enabled=False)
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    calls: list[str] = []
    service = legacy.LegacyLandedReviewService(
        repo_root=repo,
        operator_policy_path=policy_path,
        operator_key_path=key_path,
        grok_invoker=lambda _request: calls.append("grok"),  # type: ignore[arg-type]
        codex_invoker=lambda _request: calls.append("codex"),  # type: ignore[arg-type]
        validation_invoker=lambda *_args: calls.append("validation"),  # type: ignore[arg-type]
    )

    result = service.review("ASE-005")

    assert result.status == "disabled"
    assert result.reason_code == "legacy_landed_review_disabled"
    assert calls == []
    assert result.to_dict()["provider_execution_receipt"] is None


def test_full_review_requires_both_exact_providers_fresh_validation_and_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "authority.key"
    _private, issuer = _private_key_file(key_path)
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path, _policy_payload(issuer_key_id=issuer))
    repo = tmp_path / "repo"
    repo.mkdir()
    _fake_repo(monkeypatch)
    grok = _ApprovingProvider()
    codex = _ApprovingProvider()
    validation_calls: list[tuple[str, ...]] = []

    def validate(
        argv: tuple[str, ...], _repo: Path, _timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        validation_calls.append(argv)
        return subprocess.CompletedProcess(list(argv), 0, b"passed", b"")

    tick = iter(range(1_000, 2_000))
    service = legacy.LegacyLandedReviewService(
        repo_root=repo,
        operator_policy_path=policy_path,
        operator_key_path=key_path,
        grok_invoker=grok,
        codex_invoker=codex,
        validation_invoker=validate,
        clock_ms=lambda: next(tick),
    )
    result = service.review("ASE-005")

    assert result.reviewed is True
    assert result.attestation is not None
    assert result.attestation.historical_provider == "unverified"
    assert result.attestation.to_dict()["provider_execution_receipt_synthesized"] is False
    assert result.attestation.to_dict()["completion_authoritative"] is False
    assert result.attestation.to_dict()["proof_authoritative"] is False
    assert grok.calls == result.manifest["leaf_count"]  # type: ignore[index]
    assert codex.calls == result.manifest["leaf_count"]  # type: ignore[index]
    assert validation_calls == list(service.policy.task("ASE-005").validations)

    failures = legacy.verify_legacy_landed_review_result(
        result,
        repo_root=repo,
        policy=service.policy,
        trusted_public_keys={issuer: service.trusted_public_key},
    )
    assert failures == ()

    aggregate = result.review_aggregate
    assert aggregate is not None
    for pair in aggregate["ordered_leaf_reviews"]:
        assert pair["grok"]["effective_provider"] == "grok_cli"
        assert pair["codex"]["effective_provider"] == "codex_cli"
        assert pair["grok"]["fallback_used"] is False
        assert pair["codex"]["fallback_used"] is False
        assert pair["grok"]["effective_reasoning_effort"] == ""
        assert pair["codex"]["requested_reasoning_effort"] == "medium"
        assert pair["codex"]["effective_reasoning_effort"] == "medium"

    tampered = result.attestation.to_dict()
    tampered["signature"] = "AAAA"
    verification = verify_legacy_landed_review_attestation(
        tampered,
        trusted_public_keys={issuer: service.trusted_public_key},
        expected_policy_id=service.policy.policy_id,
        expected_task_id="ASE-005",
        expected_canonical_task_key=service.policy.task("ASE-005").canonical_task_key,
        expected_canonical_task_cid=service.policy.task("ASE-005").canonical_task_cid,
        expected_current_head=HEAD,
        expected_current_tree_id=TREE,
        manifest=result.manifest,
        review_aggregate=result.review_aggregate,
        validation_receipts=result.validation_receipts,
        scope_adjudication_receipt=None,
    )
    assert verification.verified is False
    assert "legacy_landed_review_signature_invalid" in verification.reason_codes


def test_policy_runtime_reverify_and_admission_ignore_poisoned_git_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Legacy Review Test")
    _git(repo, "config", "user.email", "legacy@example.invalid")
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "baseline")
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")

    redirected = tmp_path / "redirected"
    redirected.mkdir()
    _git(redirected, "init")
    _git(redirected, "config", "user.name", "Redirected")
    _git(redirected, "config", "user.email", "redirected@example.invalid")
    (redirected / "wrong.py").write_text("wrong = True\n", encoding="utf-8")
    _git(redirected, "add", "wrong.py")
    _git(redirected, "commit", "-m", "wrong")
    malicious_config = tmp_path / "malicious.gitconfig"
    malicious_config.write_text(
        "[diff]\n    external = false\n[core]\n    worktree = /nonexistent\n",
        encoding="utf-8",
    )
    poison = {
        "GIT_DIR": str(redirected / ".git"),
        "GIT_WORK_TREE": str(redirected),
        "GIT_INDEX_FILE": str(redirected / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(redirected / ".git" / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(repo / ".git" / "objects"),
        "GIT_REPLACE_REF_BASE": "refs/replace/poisoned/",
        "GIT_CONFIG": str(malicious_config),
        "GIT_CONFIG_GLOBAL": str(malicious_config),
        "GIT_CONFIG_SYSTEM": str(malicious_config),
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": str(redirected),
        "GIT_CONFIG_KEY_1": "diff.external",
        "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_PARAMETERS": "'core.worktree=/nonexistent'",
        "GIT_EXTERNAL_DIFF": "false",
        "GIT_DIFF_OPTS": "--stat",
    }
    key_path = tmp_path / "legacy.key"
    _private, issuer = _private_key_file(key_path)

    def binding(task: legacy.LegacyTaskPolicy) -> legacy.LegacyRepositoryBinding:
        return replace(
            _binding(task), current_head=head, current_tree_id=tree
        )

    monkeypatch.setattr(
        legacy,
        "inspect_legacy_repository_binding",
        lambda _repo, _policy, task, **_kwargs: binding(task),
    )
    with monkeypatch.context() as poisoned:
        for name, value in poison.items():
            poisoned.setenv(name, value)
        policy_payload = legacy.build_exact_eight_legacy_landed_policy(
            repo,
            current_head=head,
            issuer_key_id=issuer,
            enabled=True,
        )
    assert policy_payload["current_tree_id"] == tree
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path, policy_payload)
    policy = legacy.load_legacy_landed_review_policy(policy_path)
    todo_path = tmp_path / "runtime.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=repo,
        implement=True,
        production_provider_policy="grok-implement-codex-independent-review",
        production_provider_review_authority_key_path=(
            tmp_path / "production-review.key"
        ),
        legacy_landed_review_policy_path=policy_path,
        legacy_landed_review_key_path=key_path,
    )

    def validate(
        argv: tuple[str, ...], _repo: Path, _timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(list(argv), 0, b"passed", b"")

    service = legacy.LegacyLandedReviewService(
        repo_root=repo,
        operator_policy_path=policy_path,
        operator_key_path=key_path,
        grok_invoker=_ApprovingProvider(),
        codex_invoker=_ApprovingProvider(),
        validation_invoker=validate,
        clock_ms=iter(range(1_000, 10_000)).__next__,
        leaf_result_cache=daemon.legacy_landed_review_result_cache,
    )
    task_policy = policy.task("ASE-005")
    task = PortalTask(
        task_id=task_policy.task_id,
        title="Audited legacy task",
        status="ready",
        completion="manual",
        priority="P0",
        track="migration",
        outputs=list(task_policy.paths),
        validation=["python -m pytest -q"],
        acceptance="exact landed bytes receive independent review",
        canonical_task_key=task_policy.canonical_task_key,
        canonical_task_cid=task_policy.canonical_task_cid,
    )
    with monkeypatch.context() as poisoned:
        for name, value in poison.items():
            poisoned.setenv(name, value)
        result = service.review(task.task_id)
        assert result.reviewed is True
        assert legacy.verify_legacy_landed_review_result(
            result,
            repo_root=repo,
            policy=policy,
            trusted_public_keys={issuer: service.trusted_public_key},
        ) == ()
        gate = daemon._verified_legacy_landed_review_gate_evidence(  # noqa: SLF001
            task=task,
            implementation_commit=task_policy.implementation_commit,
            merge_commit=head,
            repository_tree_id=f"git-tree:{tree}",
            evidence={"legacy_landed_review_result": result.to_dict()},
        )
    assert gate is not None
    assert gate["provider_result_admitted"] is True


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("fallback", "legacy_effective_provider_mismatch"),
        ("wrong-model", "legacy_effective_provider_mismatch"),
        ("reject", "legacy_provider_leaf_not_approved"),
    ],
)
def test_fallback_wrong_model_or_one_rejection_never_attests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_reason: str,
) -> None:
    key_path = tmp_path / "authority.key"
    _private, issuer = _private_key_file(key_path)
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path, _policy_payload(issuer_key_id=issuer))
    repo = tmp_path / "repo"
    repo.mkdir()
    _fake_repo(monkeypatch)
    grok = _ApprovingProvider()

    def invalid(request: legacy.LegacyLeafReviewRequest):
        valid = _ApprovingProvider()(request)
        if mutation == "fallback":
            return replace(valid, fallback_used=True, provider_chain=(request.provider, "fallback"))
        if mutation == "wrong-model":
            return replace(valid, effective_model="wrong")
        response = dict(valid.response)
        response["decision"] = "reject"
        response["findings"] = ["not approved"]
        return replace(valid, response=response)

    validations: list[str] = []
    service = legacy.LegacyLandedReviewService(
        repo_root=repo,
        operator_policy_path=policy_path,
        operator_key_path=key_path,
        grok_invoker=grok,
        codex_invoker=invalid,
        validation_invoker=lambda *_args: validations.append("called"),  # type: ignore[arg-type]
    )
    result = service.review("ASE-005")

    assert result.status == "rejected"
    assert result.reason_code == expected_reason
    assert result.attestation is None
    assert validations == []


def test_authority_projection_cannot_be_upgraded(tmp_path: Path) -> None:
    private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))
    authority = __import__(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_landed_attestation",
        fromlist=["LegacyLandedReviewAuthority"],
    ).LegacyLandedReviewAuthority(private)
    attestation = authority.issue(
        policy_id=policy.policy_id,
        task_id=task.task_id,
        canonical_task_key=task.canonical_task_key,
        canonical_task_cid=task.canonical_task_cid,
        baseline_commit=task.baseline_commit,
        interval_commits=task.interval_commits,
        implementation_commit=task.implementation_commit,
        merge_commit=task.merge_commit,
        current_head=HEAD,
        current_tree_id=TREE,
        paths=task.paths,
        manifest_id=manifest["manifest_id"],
        manifest_merkle_root=manifest["merkle_root"],
        review_aggregate_id="baguqeera" + "a" * 52,
        validation_receipt_ids=("baguqeera" + "b" * 52,),
        issued_at_ms=1,
        nonce="0123456789abcdef",
    )
    tampered = attestation.to_dict()
    tampered["completion_authoritative"] = True
    with pytest.raises(ValueError):
        LegacyLandedReviewAttestation.from_dict(tampered)


def test_bundle_and_child_supervisor_forward_only_two_explicit_legacy_paths(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "operator-policy.json"
    key_path = tmp_path / "operator-key.ed25519"
    common = {
        "todo_path": Path("runtime.todo.md"),
        "state_dir": Path("state"),
        "worktree_root": Path("worktrees"),
        "state_prefix": "ase-legacy",
        "task_prefix": "## ASE-",
        "implement": True,
        "daemon_interval": 5,
        "stale_seconds": 30,
        "check_interval": 2,
        "watchdog_startup_grace_seconds": None,
        "max_restarts": 1,
        "implementation_timeout": 300,
    }
    default_command = implementation_supervisor_command(**common)
    assert "--legacy-landed-review-policy-path" not in default_command
    assert "--legacy-landed-review-key-path" not in default_command

    command = implementation_supervisor_command(
        **common,
        legacy_landed_review_policy_path=policy_path,
        legacy_landed_review_key_path=key_path,
    )
    assert command[command.index("--legacy-landed-review-policy-path") + 1] == str(
        policy_path
    )
    assert command[command.index("--legacy-landed-review-key-path") + 1] == str(
        key_path
    )
    with pytest.raises(ValueError):
        implementation_supervisor_command(
            **common,
            legacy_landed_review_policy_path=policy_path,
        )

    config = PortalSupervisorConfig(
        todo_path=tmp_path / "runtime.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        state_dir=tmp_path,
        implement=True,
        legacy_landed_review_policy_path=policy_path,
        legacy_landed_review_key_path=key_path,
    )
    daemon_command = PortalImplementationSupervisor(config)._build_daemon_command()
    assert daemon_command[
        daemon_command.index("--legacy-landed-review-policy-path") + 1
    ] == str(policy_path)
    assert daemon_command[
        daemon_command.index("--legacy-landed-review-key-path") + 1
    ] == str(key_path)

    incomplete = replace(config, legacy_landed_review_key_path=None)
    with pytest.raises(ValueError):
        PortalImplementationSupervisor(incomplete)._build_daemon_command()


def test_daemon_loads_only_a_strict_paired_legacy_policy_and_key(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "runtime.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    key_path = tmp_path / "operator.key"
    _private, issuer = _private_key_file(key_path)
    policy_path = tmp_path / "operator-policy.json"
    _write_policy(policy_path, _policy_payload(issuer_key_id=issuer, enabled=False))

    args = parse_daemon_args(
        [
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(tmp_path / "state"),
            "--legacy-landed-review-policy-path",
            str(policy_path),
            "--legacy-landed-review-key-path",
            str(key_path),
        ]
    )
    assert args.legacy_landed_review_policy_path == policy_path
    assert args.legacy_landed_review_key_path == key_path

    common = {
        "todo_path": todo_path,
        "state_path": tmp_path / "state" / "task-state.json",
        "strategy_path": tmp_path / "state" / "strategy.json",
        "events_path": tmp_path / "state" / "events.jsonl",
        "repo_root": tmp_path,
    }
    with pytest.raises(ValueError, match="must be supplied together"):
        PortalImplementationDaemon(
            **common,
            legacy_landed_review_policy_path=policy_path,
        )

    daemon = PortalImplementationDaemon(
        **common,
        legacy_landed_review_policy_path=policy_path,
        legacy_landed_review_key_path=key_path,
    )
    assert daemon.legacy_landed_review_policy is not None
    assert daemon.legacy_landed_review_policy.policy_id == (
        legacy.load_legacy_landed_review_policy(policy_path).policy_id
    )
    assert daemon._legacy_landed_review_service is not None  # noqa: SLF001
    assert set(daemon.legacy_landed_review_trusted_public_keys) == {issuer}
    assert daemon.legacy_landed_review_result_cache is not None
    assert daemon.legacy_landed_review_result_cache_path == (
        key_path.parent / "legacy_landed_review_results.duckdb"
    )
    assert daemon.legacy_landed_review_result_cache_path.is_file()
    assert not hasattr(args, "legacy_landed_review_result_cache_path")


def test_guarded_legacy_review_is_reverified_at_authoritative_provider_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    todo_path = tmp_path / "runtime.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    key_path = tmp_path / "operator.key"
    _private, issuer = _private_key_file(key_path)
    policy_path = tmp_path / "operator-policy.json"
    _write_policy(policy_path, _policy_payload(issuer_key_id=issuer))
    _fake_repo(monkeypatch)

    def validate(
        argv: tuple[str, ...], _repo: Path, _timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(list(argv), 0, b"passed", b"")

    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        implement=True,
        production_provider_policy="grok-implement-codex-independent-review",
        production_provider_review_authority_key_path=(
            tmp_path / "production-review.key"
        ),
        legacy_landed_review_policy_path=policy_path,
        legacy_landed_review_key_path=key_path,
    )
    service = legacy.LegacyLandedReviewService(
        repo_root=tmp_path,
        operator_policy_path=policy_path,
        operator_key_path=key_path,
        grok_invoker=_ApprovingProvider(),
        codex_invoker=_ApprovingProvider(),
        validation_invoker=validate,
        clock_ms=iter(range(1_000, 2_000)).__next__,
    )
    daemon._legacy_landed_review_service = service  # noqa: SLF001
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER",
        "grok",
    )
    assert {"grok", "codex"}.issubset(
        daemon._current_implementation_provider_labels()  # noqa: SLF001
    )
    task_policy = service.policy.task("ASE-005")
    task = PortalTask(
        task_id=task_policy.task_id,
        title="Audited legacy task",
        status="ready",
        completion="manual",
        priority="P0",
        track="migration",
        outputs=list(task_policy.paths),
        validation=["python -m pytest -q"],
        acceptance="exact landed bytes receive independent review",
        canonical_task_key=task_policy.canonical_task_key,
        canonical_task_cid=task_policy.canonical_task_cid,
    )
    payload = daemon._run_legacy_landed_review_for_guard(  # noqa: SLF001
        task,
        landed_guard={
            "guarded": True,
            "workspace_clean": True,
            "baseline_ref": HEAD,
            "repository_tree_id": f"git-tree:{TREE}",
        },
    )
    assert payload["status"] == "reviewed"
    result = legacy.LegacyLandedReviewResult.from_dict(payload)
    assert result.reviewed
    assert daemon._production_reviewed_effect_required(task) is False  # noqa: SLF001

    gate = daemon._verified_legacy_landed_review_gate_evidence(  # noqa: SLF001
        task=task,
        implementation_commit=task_policy.implementation_commit,
        merge_commit=HEAD,
        repository_tree_id=f"git-tree:{TREE}",
        evidence={"legacy_landed_review_result": payload},
    )
    assert gate is not None
    assert gate["route_kind"] == "legacy_landed_fresh_dual_review"
    assert gate["provider_result_admitted"] is True
    assert gate["provider_execution_receipt_synthesized"] is False

    tampered = copy.deepcopy(payload)
    tampered["manifest"]["leaves"][0]["payload"] = "AAAA"
    assert (
        daemon._verified_legacy_landed_review_gate_evidence(  # noqa: SLF001
            task=task,
            implementation_commit=task_policy.implementation_commit,
            merge_commit=HEAD,
            repository_tree_id=f"git-tree:{TREE}",
            evidence={"legacy_landed_review_result": tampered},
        )
        is None
    )


def test_cli_adapters_send_exact_envelope_and_observe_no_fallback_effective_child(
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))
    observed: list[tuple[str, Any]] = []

    def invoke(prompt: str, config: Any):
        observed.append((prompt, config))
        request_payload = json.loads(prompt)
        leaf = request_payload["payload"]["leaf"]
        response = {
            "schema": legacy.LEGACY_LANDED_LEAF_DECISION_SCHEMA,
            "decision": "approve",
            "manifest_id": request_payload["payload"]["manifest_id"],
            "leaf_id": leaf["leaf_id"],
            "findings": [],
        }
        return json.dumps(response), LlmChildResultEnvelope(
            usage_mode=LLM_USAGE_MODE_ENFORCE,
            request_id=config.request_id,
            idempotency_key=config.idempotency_key,
            supervisor_receipt_id="supervisor:" + config.request_id,
            endpoint_receipt_id="endpoint:" + config.request_id,
            execution_result_id="result:" + config.request_id,
            effective_provider=config.provider,
            text_chars=len(json.dumps(response)),
            exit_code=0,
        )

    grok, codex = build_legacy_landed_cli_provider_pair(policy, invoker=invoke)
    leaf = manifest["leaves"][0]
    grok_request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=leaf,
        provider=policy.grok,
    )
    codex_request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=leaf,
        provider=policy.codex,
    )
    grok_observation = grok(grok_request)
    codex_observation = codex(codex_request)

    assert observed[0][0].encode("ascii") == grok_request.canonical_prompt
    assert observed[1][0].encode("ascii") == codex_request.canonical_prompt
    assert all(config.repo_root != tmp_path for _prompt, config in observed)
    assert all(config.allow_local_fallback is False for _prompt, config in observed)
    assert all(config.usage_mode == LLM_USAGE_MODE_ENFORCE for _prompt, config in observed)
    assert grok_observation.effective_provider == "grok_cli"
    assert codex_observation.effective_provider == "codex_cli"
    assert grok_observation.provider_chain == ("grok_cli",)
    assert codex_observation.provider_chain == ("codex_cli",)
    assert grok_observation.observation_id != codex_observation.observation_id


def test_cli_adapter_rejects_child_effective_provider_mismatch(tmp_path: Path) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))

    def invoke(_prompt: str, config: Any):
        return "{}", LlmChildResultEnvelope(
            usage_mode=LLM_USAGE_MODE_ENFORCE,
            request_id=config.request_id,
            idempotency_key=config.idempotency_key,
            effective_provider="codex_cli",
            exit_code=0,
        )

    grok, _codex = build_legacy_landed_cli_provider_pair(policy, invoker=invoke)
    request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
    )
    with pytest.raises(RuntimeError, match="not exactly bound"):
        grok(request)


def test_cli_adapters_use_native_request_bound_schema_and_verbatim_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))
    leaf = manifest["leaves"][0]
    requests = {
        provider.provider: legacy._leaf_review_request(  # noqa: SLF001
            policy=policy,
            task=task,
            manifest=manifest,
            leaf=leaf,
            provider=provider,
        )
        for provider in (policy.grok, policy.codex)
    }
    observed: dict[str, dict[str, Any]] = {}

    monkeypatch.setattr(
        legacy_cli.shutil,
        "which",
        lambda name: f"/trusted/bin/{name}",
    )

    def run_cli(
        command: Any,
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> tuple[str, str]:
        cmd = list(command)
        provider = "grok_cli" if cmd[0].endswith("/grok") else "codex_cli"
        request = requests[provider]
        if provider == "grok_cli":
            schema = json.loads(cmd[cmd.index("--json-schema") + 1])
            prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
            prompt_envelope = json.loads(prompt_path.read_text(encoding="utf-8"))
            assert prompt_envelope["type"] == "acp"
            assert len(prompt_envelope["content"]) == 1
            assert prompt_envelope["content"][0]["type"] == "text"
            prompt = prompt_envelope["content"][0]["text"]
            assert stdin_text is None
        else:
            schema_path = Path(cmd[cmd.index("--output-schema") + 1])
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            prompt = str(stdin_text)
        response = {
            "schema": legacy.LEGACY_LANDED_LEAF_DECISION_SCHEMA,
            "decision": "approve",
            "manifest_id": request.payload["manifest_id"],
            "leaf_id": request.payload["leaf"]["leaf_id"],
            "findings": [],
        }
        observed[provider] = {
            "command": cmd,
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "prompt": prompt,
            "schema": schema,
        }
        if provider == "grok_cli":
            return json.dumps(
                {
                    "text": json.dumps(response),
                    "requestId": "grok-endpoint-request",
                }
            ), ""
        response_path = Path(cmd[cmd.index("--output-last-message") + 1])
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return '{"type":"turn.completed"}\n', ""

    monkeypatch.setattr(legacy_cli, "_run_native_cli_process", run_cli)
    grok, codex = build_legacy_landed_cli_provider_pair(policy)
    grok_observation = grok(requests["grok_cli"])
    codex_observation = codex(requests["codex_cli"])

    for provider, expected_model in (
        ("grok_cli", policy.grok.model),
        ("codex_cli", policy.codex.model),
    ):
        record = observed[provider]
        request = requests[provider]
        assert record["prompt"].encode("ascii") == request.canonical_prompt
        assert record["cwd"] != tmp_path
        assert record["timeout_seconds"] == 300
        schema = record["schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["manifest_id"]["enum"] == [
            request.payload["manifest_id"]
        ]
        assert schema["properties"]["leaf_id"]["enum"] == [
            request.payload["leaf"]["leaf_id"]
        ]
        assert schema["properties"]["findings"]["maxItems"] == 0
        assert "oneOf" not in schema
        command = record["command"]
        model_flag = "--model"
        assert command[command.index(model_flag) + 1] == expected_model
        assert request.canonical_prompt.decode("ascii") not in command

    grok_command = observed["grok_cli"]["command"]
    codex_command = observed["codex_cli"]["command"]
    assert "--json-schema" in grok_command
    assert "--verbatim" in grok_command
    assert "--tools" not in grok_command
    assert "--disallowed-tools" in grok_command
    assert grok_command[grok_command.index("--deny") + 1] == "*"
    assert "--output-schema" in codex_command
    assert codex_command[codex_command.index("-c") + 1] == (
        'model_reasoning_effort="medium"'
    )
    assert codex_command[-1] == "-"
    assert grok_observation.effective_provider == "grok_cli"
    assert grok_observation.effective_model == policy.grok.model
    assert codex_observation.effective_provider == "codex_cli"
    assert codex_observation.effective_model == policy.codex.model
    assert codex_observation.requested_reasoning_effort == "medium"
    assert codex_observation.effective_reasoning_effort == "medium"


def test_native_response_parser_requires_empty_findings_but_keeps_reject(
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))
    request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
    )
    response_schema = legacy_cli._leaf_decision_json_schema(  # noqa: SLF001
        request
    )
    response = {
        "schema": legacy.LEGACY_LANDED_LEAF_DECISION_SCHEMA,
        "decision": "approve",
        "manifest_id": request.payload["manifest_id"],
        "leaf_id": request.payload["leaf"]["leaf_id"],
        "findings": ["approval cannot carry a finding"],
    }

    with pytest.raises(RuntimeError, match="violates its schema"):
        legacy_cli._validate_native_response(  # noqa: SLF001
            json.dumps(response), response_schema
        )

    response["decision"] = "reject"
    with pytest.raises(RuntimeError, match="violates its schema"):
        legacy_cli._validate_native_response(  # noqa: SLF001
            json.dumps(response), response_schema
        )

    response["findings"] = []
    assert legacy_cli._validate_native_response(  # noqa: SLF001
        json.dumps(response), response_schema
    ) == response


@pytest.mark.parametrize(
    ("stream", "quota_text"),
    (
        ("stderr", "You've hit your usage limit; try again after the reset."),
        ("stdout", "YOU\u2019VE HIT YOUR USAGE LIMIT; retry later."),
    ),
)
def test_native_codex_quota_failure_emits_only_fixed_capacity_signal(
    tmp_path: Path,
    stream: str,
    quota_text: str,
) -> None:
    codex = tmp_path / "codex"
    codex.symlink_to(sys.executable)
    secret = "sk-provider-secret-must-not-escape"
    script = (
        "import sys;"
        f"sys.{stream}.write({f'{quota_text} token={secret}'!r});"
        "raise SystemExit(1)"
    )

    with pytest.raises(legacy.LegacyProviderCapacitySignal) as raised:
        legacy_cli._run_native_cli_process(  # noqa: SLF001
            [str(codex), "-c", script],
            cwd=tmp_path,
            timeout_seconds=10,
        )

    diagnostic = str(raised.value)
    assert diagnostic == legacy.LEGACY_CODEX_USAGE_LIMIT_CAPACITY_MARKER
    assert secret not in diagnostic
    assert quota_text not in diagnostic
    classified = classify_provider_capacity_failure(diagnostic)
    assert classified["exhausted"] is True
    assert classified["providers"] == ["codex"]


def test_native_codex_unrelated_failure_never_exposes_stderr(
    tmp_path: Path,
) -> None:
    codex = tmp_path / "codex"
    codex.symlink_to(sys.executable)
    secret = "oauth-secret-must-not-escape"
    script = (
        "import sys;"
        f"sys.stderr.write({'authentication failed: ' + secret!r});"
        "raise SystemExit(17)"
    )

    with pytest.raises(RuntimeError) as raised:
        legacy_cli._run_native_cli_process(  # noqa: SLF001
            [str(codex), "-c", script],
            cwd=tmp_path,
            timeout_seconds=10,
        )

    diagnostic = str(raised.value)
    assert type(raised.value) is RuntimeError
    assert diagnostic == "legacy native provider command failed"
    assert secret not in diagnostic
    assert classify_provider_capacity_failure(diagnostic)["exhausted"] is False


def test_native_codex_failure_capture_bound_precedes_quota_classification(
    tmp_path: Path,
) -> None:
    codex = tmp_path / "codex"
    codex.symlink_to(sys.executable)
    secret = "capture-bound-secret-must-not-escape"
    byte_count = legacy_cli._MAX_NATIVE_CLI_CAPTURE_BYTES + 1  # noqa: SLF001
    script = (
        "import os;"
        f"os.write(2, b\"You've hit your usage limit {secret} \" + "
        f"b'x' * {byte_count});"
        "raise SystemExit(1)"
    )

    with pytest.raises(RuntimeError, match="capture bound") as raised:
        legacy_cli._run_native_cli_process(  # noqa: SLF001
            [str(codex), "-c", script],
            cwd=tmp_path,
            timeout_seconds=10,
        )

    assert type(raised.value) is RuntimeError
    assert secret not in str(raised.value)
    assert classify_provider_capacity_failure(str(raised.value))["exhausted"] is False


def test_codex_capacity_signal_survives_review_as_sanitized_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "authority.key"
    _private, issuer = _private_key_file(key_path)
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path, _policy_payload(issuer_key_id=issuer))
    repo = tmp_path / "repo"
    repo.mkdir()
    _fake_repo(monkeypatch)

    def codex_at_capacity(
        _request: legacy.LegacyLeafReviewRequest,
    ) -> legacy.LegacyProviderObservation:
        signal = legacy.LegacyProviderCapacitySignal()
        signal.reason_code = "mutated-provider-secret-must-not-escape"
        raise signal

    service = legacy.LegacyLandedReviewService(
        repo_root=repo,
        operator_policy_path=policy_path,
        operator_key_path=key_path,
        grok_invoker=_ApprovingProvider(),
        codex_invoker=codex_at_capacity,
    )

    result = service.review("ASE-005")

    assert result.status == "rejected"
    assert result.reason_code == legacy.LEGACY_CODEX_USAGE_LIMIT_CAPACITY_MARKER
    assert "mutated-provider-secret" not in json.dumps(result.to_dict())
    classified = classify_provider_capacity_failure(result.reason_code)
    assert classified["exhausted"] is True
    assert classified["providers"] == ["codex"]

    embedded = (
        "ordinary output "
        + legacy.LEGACY_CODEX_USAGE_LIMIT_CAPACITY_MARKER
        + " token=embedded-secret"
    )
    assert classify_provider_capacity_failure(embedded)["exhausted"] is False


def test_landed_guard_codex_capacity_defers_without_attempt_or_lifecycle_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Capacity Deferral Test")
    _git(repo, "config", "user.email", "capacity@example.invalid")
    (repo / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "baseline.txt")
    _git(repo, "commit", "-m", "baseline")

    authority_dir = tmp_path / "authority"
    authority_dir.mkdir()
    key_path = authority_dir / "legacy-review.key"
    _private, issuer = _private_key_file(key_path)
    policy_path = authority_dir / "legacy-review-policy.json"
    _write_policy(policy_path, _policy_payload(issuer_key_id=issuer))
    _fake_repo(monkeypatch)

    todo_path = tmp_path / "runtime.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implement=True,
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
        max_task_attempts=1,
        merge_queue_dir=tmp_path / "merge-queue",
        validation_cache_dir=tmp_path / "validation-cache",
        production_provider_policy="grok-implement-codex-independent-review",
        production_provider_review_authority_key_path=(
            authority_dir / "production-review.key"
        ),
        legacy_landed_review_policy_path=policy_path,
        legacy_landed_review_key_path=key_path,
    )

    codex_calls: list[str] = []

    def codex_at_capacity(
        _request: legacy.LegacyLeafReviewRequest,
    ) -> legacy.LegacyProviderObservation:
        codex_calls.append("codex")
        raise legacy.LegacyProviderCapacitySignal()

    daemon._legacy_landed_review_service = legacy.LegacyLandedReviewService(  # noqa: SLF001
        repo_root=repo,
        operator_policy_path=policy_path,
        operator_key_path=key_path,
        grok_invoker=_ApprovingProvider(),
        codex_invoker=codex_at_capacity,
    )
    task_policy = daemon._legacy_landed_review_service.policy.task("ASE-005")  # noqa: SLF001
    task = PortalTask(
        task_id=task_policy.task_id,
        title="Audited legacy task",
        status="ready",
        completion="manual",
        priority="P0",
        track="migration",
        outputs=list(task_policy.paths),
        validation=["python -m pytest -q"],
        acceptance="exact landed bytes receive independent review",
        canonical_task_key=task_policy.canonical_task_key,
        canonical_task_cid=task_policy.canonical_task_cid,
    )
    daemon._register_task_identities([task])  # noqa: SLF001
    identity = daemon._identity_for_task(task)  # noqa: SLF001
    queue_entry = daemon.task_queue.register_task(identity)

    def seed(worktree_path: Path, branch: str, *, task: Any = None) -> str:
        _git(repo, "worktree", "add", "-b", branch, str(worktree_path), "HEAD")
        return _git(worktree_path, "rev-parse", "HEAD")

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE", "1")
    monkeypatch.setattr(daemon, "_create_seeded_worktree", seed)
    monkeypatch.setattr(
        daemon,
        "_production_landed_task_guard_for_workspace",
        lambda *_args, **_kwargs: {
            "guarded": True,
            "workspace_clean": True,
            "baseline_ref": HEAD,
            "repository_tree_id": f"git-tree:{TREE}",
        },
    )
    monkeypatch.setattr(
        daemon,
        "_require_implementation_protected_snapshot",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_violation",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_finalize_implementation_protected_path_fence",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_mutation",
        lambda _kind, _payload, action: action(),
    )
    deferrals: list[dict[str, Any]] = []
    real_deferral = daemon._record_provider_capacity_deferral  # noqa: SLF001

    def record_deferral(**kwargs: Any) -> dict[str, Any]:
        deferrals.append(dict(kwargs["failure"]))
        return real_deferral(**kwargs)

    queue_outcomes: list[tuple[Any, ...]] = []
    monkeypatch.setattr(daemon, "_record_provider_capacity_deferral", record_deferral)
    monkeypatch.setattr(
        daemon,
        "_record_task_queue_outcome",
        lambda *args, **_kwargs: queue_outcomes.append(args),
    )

    state = PortalTaskState()
    result = daemon._run_implementation_in_ephemeral_worktree(  # noqa: SLF001
        task=task,
        state=state,
        attempt=1,
        started_at=datetime.now(UTC).isoformat(),
        log_path=state_dir / "implementation.log",
        prompt="review the exact landed implementation",
    )

    assert deferrals == [
        {
            "exhausted": True,
            "providers": ["codex"],
            "reason": "provider_capacity_exhausted",
        }
    ]
    assert result["deferred"] is True
    assert result["attempt_consumed"] is False
    assert result["reason"] == "provider_capacity_exhausted"
    assert result["providers"] == ["codex"]
    assert result["cleanup_result"]["cleaned"] is True
    assert result["cleanup_result"]["lifecycle_finalize"]["finalized"] is True
    assert not Path(result["worktree_path"]).exists()
    assert daemon.worktree_lifecycle.load_task_attempt(
        canonical_task_cid=identity.canonical_task_cid,
        task_id=task.task_id,
        attempt=1,
    ) is None

    recovered = PortalTaskState.load(daemon.state_path)
    assert recovered.implementation_attempts == {}
    assert recovered.implementation_attempts_by_cid == {}
    assert daemon._task_attempt(recovered, task) == 1  # noqa: SLF001
    selectable, limited = daemon._partition_tasks_at_attempt_limit(  # noqa: SLF001
        [task], {task.task_id: "ready"}, recovered
    )
    assert selectable == [task]
    assert limited == []
    assert queue_outcomes == []
    assert queue_entry.consecutive_failures == 0
    assert queue_entry.selection_penalty == 0
    assert queue_entry.cooldown_until == 0
    assert queue_entry.notes == ""

    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [event["type"] for event in events]
    assert "legacy_landed_review_finished" in event_types
    assert "implementation_provider_exhausted" in event_types
    assert "implementation_finished" not in event_types

    # The same logical attempt must be immediately reusable after the normal
    # capacity backoff expires; no ACTIVE task/attempt index may survive.
    second_state = PortalTaskState.load(daemon.state_path)
    second = daemon._run_implementation_in_ephemeral_worktree(  # noqa: SLF001
        task=task,
        state=second_state,
        attempt=1,
        started_at=datetime.now(UTC).isoformat(),
        log_path=state_dir / "implementation-second.log",
        prompt="retry the exact landed implementation review",
    )
    assert second["deferred"] is True
    assert second["reason"] == "provider_capacity_exhausted"
    assert second["attempt_consumed"] is False
    assert codex_calls == ["codex", "codex"]
    assert len(deferrals) == 2
    assert daemon.worktree_lifecycle.load_task_attempt(
        canonical_task_cid=identity.canonical_task_cid,
        task_id=task.task_id,
        attempt=1,
    ) is None

    # A lost compare-and-delete CAS must not be reported as a successful
    # capacity deferral, even when cleanup has already removed the checkout.
    with monkeypatch.context() as raced:
        raced.setattr(
            daemon.worktree_lifecycle,
            "compare_and_delete",
            lambda *_args, **_kwargs: False,
        )
        lifecycle_race = daemon._run_implementation_in_ephemeral_worktree(  # noqa: SLF001
            task=task,
            state=PortalTaskState.load(daemon.state_path),
            attempt=1,
            started_at=datetime.now(UTC).isoformat(),
            log_path=state_dir / "implementation-cas-race.log",
            prompt="exercise lifecycle CAS failure",
        )
    assert lifecycle_race["lifecycle_race"] is True
    assert lifecycle_race["provider_call_allowed"] is False
    assert lifecycle_race["attempt_consumed"] is False
    assert lifecycle_race["reason"] == (
        "provider_capacity_lifecycle_cleanup_incomplete"
    )
    failed_finalize = lifecycle_race["cleanup_result"]["lifecycle_finalize"]
    assert failed_finalize["finalized"] is False
    assert failed_finalize["reason"] == "lifecycle_finalize_race"
    assert failed_finalize["fence"] >= 4
    assert failed_finalize["failure_kind"] == "lifecycle_race"
    assert failed_finalize["attempt_consumed"] is False
    assert failed_finalize["provider_call_allowed"] is False
    assert len(deferrals) == 2

    # An unrelated failure after typed capacity classification must remain an
    # ordinary implementation failure instead of being masked as quota.
    real_record_event = daemon._record_event  # noqa: SLF001

    def fail_after_classification(event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "production_provider_landed_task_guarded":
            raise RuntimeError("post-classification fixture failure")
        real_record_event(event_type, payload)

    with monkeypatch.context() as masked:
        masked.setattr(daemon, "_record_event", fail_after_classification)
        unrelated_failure = daemon._run_implementation_in_ephemeral_worktree(  # noqa: SLF001
            task=task,
            state=PortalTaskState.load(daemon.state_path),
            attempt=1,
            started_at=datetime.now(UTC).isoformat(),
            log_path=state_dir / "implementation-unrelated-failure.log",
            prompt="exercise post-classification failure",
        )
    assert unrelated_failure.get("deferred") is not True
    assert unrelated_failure.get("reason") != "provider_capacity_exhausted"
    assert unrelated_failure["attempt_consumed"] is True
    assert unrelated_failure["exception_result"]["exception_type"] == (
        "RuntimeError"
    )


def test_native_schema_boundary_rejects_grok_prose_and_mismatched_codex_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _private, issuer = _private_key_file(tmp_path / "key")
    policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(issuer_key_id=issuer)
    )
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))
    leaf = manifest["leaves"][0]
    grok_request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=leaf,
        provider=policy.grok,
    )
    codex_request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=leaf,
        provider=policy.codex,
    )
    monkeypatch.setattr(
        legacy_cli.shutil,
        "which",
        lambda name: f"/trusted/bin/{name}",
    )

    def prose(
        command: Any,
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> tuple[str, str]:
        del command, cwd, timeout_seconds, stdin_text
        return json.dumps({"text": "I approve this leaf."}), ""

    monkeypatch.setattr(legacy_cli, "_run_native_cli_process", prose)
    grok, codex = build_legacy_landed_cli_provider_pair(policy)
    with pytest.raises(RuntimeError, match="strict JSON"):
        grok(grok_request)

    def mismatched(
        command: Any,
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> tuple[str, str]:
        del cwd, timeout_seconds, stdin_text
        cmd = list(command)
        response_path = Path(cmd[cmd.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "schema": legacy.LEGACY_LANDED_LEAF_DECISION_SCHEMA,
                    "decision": "approve",
                    "manifest_id": codex_request.payload["manifest_id"],
                    "leaf_id": "wrong-leaf",
                    "findings": [],
                }
            ),
            encoding="utf-8",
        )
        return "", ""

    monkeypatch.setattr(legacy_cli, "_run_native_cli_process", mismatched)
    with pytest.raises(RuntimeError, match="violates its schema"):
        codex(codex_request)
