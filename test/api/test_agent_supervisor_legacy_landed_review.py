"""Fail-closed legacy landed-review migration tests."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
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
from ipfs_accelerate_py.agent_supervisor.todo_daemon import legacy_landed_review as legacy
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


def _policy_payload(*, issuer_key_id: str, enabled: bool = True) -> dict[str, Any]:
    template = copy.deepcopy(legacy.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE)
    body = {
        "schema": legacy.LEGACY_LANDED_REVIEW_POLICY_SCHEMA,
        "interface": legacy.LEGACY_LANDED_REVIEW_POLICY_INTERFACE,
        "enabled": enabled,
        "issuer_key_id": issuer_key_id,
        "current_head": HEAD,
        "current_tree_id": TREE,
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
        )


def test_exact_template_pins_all_eight_and_original_ase_023_interval() -> None:
    template = legacy.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE
    tasks = {item["task_id"]: item for item in template["tasks"]}

    assert tuple(tasks) == legacy.EXACT_LEGACY_LANDED_TASK_IDS
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
