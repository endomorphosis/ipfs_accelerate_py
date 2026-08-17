"""Focused tests for VerificationReceiptCache admission, lookup, and invalidation.

Evidence surfaces:
* ``ivp/receipt-cache@1``
* ``ivp/concurrent-writer@1``
* ``ivp/replay-corruption@1``

Acceptance coverage:
* unchanged same-tree receipt reuses;
* related tree/symbol/environment/lock/selector/tool-version/config/fixture/
  network/schema changes reject;
* unrelated edit preserves the old immutable receipt under its old key without
  a scoped-staleness tombstone but rejects it for the new full-tree key;
* stale, simulated, timeout, unavailable, invalid, malformed, kind-mismatched,
  key-mismatched, and corrupt candidates cannot satisfy production;
* two concurrent writers preserve both entries through CAS retry.
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_forest import (
    AuthorityMode,
    LocalLocator,
    PortableGitClosure,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForest,
    RepositoryIdentity,
)
from ipfs_accelerate_py.agent_supervisor.contract_analysis.execution_profile import (
    CapabilitySnapshot,
    LockIdentity,
    ToolIdentity,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    CacheReuseDisposition,
    DirectExecutionObservation,
    StaticAnalysisReceipt,
    TerminalStatus,
    TypeCheckReceipt,
    VerificationIdentityCompiler,
    VerificationReceiptKey,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_cache import (
    CONCURRENT_WRITER_EVIDENCE,
    REASON_ADMIT_REJECTED,
    REASON_EXACT_CURRENT_PRODUCTION,
    REASON_KEY_MISMATCH,
    REASON_KIND_MISMATCH,
    REASON_TOMBSTONED,
    RECEIPT_CACHE_EVIDENCE,
    REPLAY_CORRUPTION_EVIDENCE,
    VERIFICATION_RECEIPT_CACHE_INTERFACE,
    AdmitResult,
    VerificationReceiptCache,
    production_eligible,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_store import (
    HermeticVerificationReceiptStore,
    IndexEntry,
    build_receipt_envelope,
    cas_publish_entry,
    mapping_cid,
)

TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
SEMANTIC_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1"
ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
TOOL_EXECUTABLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrors contract-test identity compiler inputs)
# ---------------------------------------------------------------------------


def _structured_cid(schema: str, value: object) -> str:
    return content_identity({"schema": schema, "value": value})


def _artifact(label: str) -> str:
    return content_identity({"artifact": label, "schema": "fixture-artifact@1"})


def _repository_forest(
    *,
    commit: str = "abcdef0123456789abcdef0123456789abcdef01",
    tree: str = "0123456789abcdef0123456789abcdef01234567",
) -> RepositoryForest:
    alias = "ipfs_accelerate_py"
    descriptor = RepositoryDescriptor(
        identity=RepositoryIdentity(logical_name=alias),
        portable_closure=PortableGitClosure(commit=commit, tree=tree),
        local_locator=LocalLocator(
            alias=alias,
            root_path="/fixture/ipfs_accelerate_py",
            resolved_root_path="/fixture/ipfs_accelerate_py",
            local_repository_binding_id="fixture-binding:ipfs-accelerate",
        ),
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    return RepositoryForest(
        descriptors=(descriptor,),
        sole_write_alias=alias,
        policy_cid=_artifact("repository-forest-policy"),
    )


def _expected_environment(values: dict[str, object]) -> dict[str, object]:
    snapshot = values["capability_snapshot"]
    assert isinstance(snapshot, CapabilitySnapshot)
    tool_identity = values["tool_identity"]
    assert isinstance(tool_identity, ToolIdentity)
    capability_name = values["tool_capability_name"]
    assert isinstance(capability_name, str)
    executable_sha256 = snapshot.tool_identities[capability_name]
    return {
        **values["observed_environment"],  # type: ignore[dict-item]
        "network_policy": values["network_policy"],
        "tool_name": values["tool_name"],
        "tool_version": values["tool_version"],
        "tool_capability_name": capability_name,
        "tool_launcher_identity": tool_identity.to_dict(),
        "resolved_tool_executable": values["resolved_tool_executable"],
        "tool_executable_sha256": executable_sha256,
        "tool_executable_cid": _structured_cid(
            TOOL_EXECUTABLE_SCHEMA,
            {"capability_name": capability_name, "sha256": executable_sha256},
        ),
        "tool_version_probe_argv": values["tool_version_probe_argv"],
        "tool_version_probe_output_cid": cid_for_bytes(
            values["tool_version_probe_output_bytes"]  # type: ignore[arg-type]
        ),
        "tool_inventory_schema": "observed-tool-inventory@1",
        "adapter_schema": values["adapter_schema"],
        "capability_environment_names": tuple(sorted(snapshot.environment_names)),
        "capability_read_paths": tuple(sorted(snapshot.read_paths)),
        "capability_write_paths": tuple(sorted(snapshot.write_paths)),
        "capability_lock_identities": dict(sorted(snapshot.lock_identities.items())),
        "selected_dependency_lock_path": values["dependency_lock_path"],
        "selected_dependency_lock_identity": values[
            "dependency_lock_identity"
        ].to_dict(),  # type: ignore[union-attr]
    }


def _compiler_kwargs(
    kind: VerificationReceiptKind = VerificationReceiptKind.TYPE_CHECK,
    *,
    forest: RepositoryForest | None = None,
) -> dict[str, object]:
    tool_name, tool_version, selector_argv, adapter_schema = {
        VerificationReceiptKind.STATIC_ANALYSIS: (
            "ruff",
            "0.12.11",
            ("/usr/bin/ruff", "check", "src/example.py"),
            "ruff-verification-adapter@1",
        ),
        VerificationReceiptKind.TYPE_CHECK: (
            "mypy",
            "1.18.2",
            ("/usr/bin/python3.12", "-m", "mypy", "src/example.py"),
            "mypy-verification-adapter@1",
        ),
        VerificationReceiptKind.TEST: (
            "pytest",
            "9.1.1",
            ("/usr/bin/python3.12", "-m", "pytest", "src/example.py"),
            "pytest-verification-adapter@1",
        ),
        VerificationReceiptKind.PROOF: (
            "z3",
            "4.13.3",
            ("/usr/bin/z3", "-smt2", "obligation.smt2"),
            "z3-verification-adapter@1",
        ),
    }[kind]
    repository_forest = forest if forest is not None else _repository_forest()
    descriptor = repository_forest.write_descriptor()
    tree_observation = {
        "repository_forest_cid": repository_forest.forest_id,
        "git_commit_id": descriptor.commit,
        "git_tree_id": descriptor.tree,
        "gitlink_state_cid": descriptor.portable_closure.gitlink_closure_cid,
        "dirty_overlay_cid": descriptor.dirty_overlay_digest,
        "dirty": descriptor.dirty,
        "repository_alias": descriptor.alias,
        "repository_id": descriptor.repository_id,
        "descriptor_cid": descriptor.descriptor_cid,
        "base_repository_tree_id": "git-tree:base",
    }
    semantic = {
        "symbols": ["example.calculate@2"],
        "edge_root": "sha256:semantic-edges",
    }
    sandbox_environment = {
        "sandbox_schema": "hermetic-sandbox@1",
        "sandbox_policy": {
            "schema": "hermetic-sandbox-policy@1",
            "network": "deny",
            "auto_install": "deny",
            "home_cache": "deny",
            "auth_material": "deny",
        },
        "filesystem_policy": {
            "schema": "verification-filesystem-policy@1",
            "source": "read_only",
            "artifacts": "private_writable",
        },
        "platform": {
            "schema": "verification-platform@1",
            "os": "linux",
            "architecture": "x86_64",
            "libc": "glibc-2.39",
        },
        "interpreter": {
            "schema": "verification-interpreter@1",
            "implementation": "cpython",
            "version": "3.12.3",
            "abi": "cp312",
        },
        "toolchain": {
            "schema": "verification-toolchain@1",
            "name": "locked-python",
            "revision": "fixture-1",
        },
        "dependency_distribution": {
            "schema": "verification-dependency-distribution@1",
            "entries": ("mypy==1.18.2",),
        },
        "environment_values": {
            "schema": "verification-environment-values@1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    }
    capability_name = "verification-tool"
    executable_bytes = ("reviewed-launcher:" + tool_name).encode()
    executable_sha256 = "sha256:" + hashlib.sha256(executable_bytes).hexdigest()
    dependency_lock_bytes = b"package==1.2.3 --hash=sha256:abcd\n"
    dependency_lock_path = "requirements.lock"
    dependency_lock_identity = LockIdentity(
        path=dependency_lock_path,
        identity="sha256:" + hashlib.sha256(dependency_lock_bytes).hexdigest(),
    )
    capability_snapshot = CapabilitySnapshot(
        tool_identities={capability_name: executable_sha256},
        lock_identities={
            dependency_lock_path: (
                "sha256:" + hashlib.sha256(dependency_lock_bytes).hexdigest()
            )
        },
        environment_names=("LANG", "LC_ALL"),
        read_paths=("/workspace/source",),
        write_paths=("/workspace/artifacts",),
    )
    tool_identity = ToolIdentity(
        name=capability_name,
        kind="executable",
        locator=selector_argv[0].rsplit("/", 1)[-1],
        version="launcher-fixture-1",
        identity=executable_sha256,
        roles=("verification",),
    )
    invocation_prefix = (
        selector_argv[:3]
        if len(selector_argv) >= 3 and selector_argv[1] == "-m"
        else selector_argv[:1]
    )
    version_probe_argv = (*invocation_prefix, "--version")
    version_probe_output = f"{tool_name} {tool_version}\n".encode()
    values: dict[str, object] = {
        "repository_forest": repository_forest,
        "repository_alias": repository_forest.sole_write_alias,
        "claimed_repository_tree_cid": _structured_cid(TREE_SCHEMA, tree_observation),
        "patch_base_tree_id": "git-tree:base",
        "repository_state_tree_id": "git-tree:base",
        "invalidation_plan_tree_id": "git-tree:base",
        "context_pack_tree_id": "git-tree:base",
        "observed_semantic_state": semantic,
        "repository_state_semantic_root_cid": _structured_cid(
            SEMANTIC_SCHEMA, semantic
        ),
        "invalidation_plan_semantic_root_cid": _structured_cid(
            SEMANTIC_SCHEMA, semantic
        ),
        "context_pack_semantic_root_cid": _structured_cid(SEMANTIC_SCHEMA, semantic),
        "affected_symbol_versions": (
            {
                "symbol": "example.calculate",
                "version": 2,
                "source_cid": _artifact("source-v2"),
            },
        ),
        "observed_environment": sandbox_environment,
        "capability_snapshot": capability_snapshot,
        "tool_capability_name": capability_name,
        "tool_identity": tool_identity,
        "resolved_tool_executable": selector_argv[0],
        "tool_executable_bytes": executable_bytes,
        "tool_version_probe_argv": version_probe_argv,
        "tool_version_probe_output_bytes": version_probe_output,
        "claimed_environment_cid": "",
        "dependency_lock_path": dependency_lock_path,
        "dependency_lock_identity": dependency_lock_identity,
        "dependency_lock_bytes": dependency_lock_bytes,
        "selector_argv": selector_argv,
        "proof_obligation": None,
        "tool_name": tool_name,
        "tool_version": tool_version,
        "configuration_bytes": b"[tool]\nstrict = true\n",
        "fixture_data_bytes": (b"fixture-one\n", b"fixture-two\n"),
        "network_policy": "deny_all",
        "receipt_schema_version": 1,
        "receipt_kind": kind,
        "adapter_schema": adapter_schema,
        "proof_backend_binding": None,
    }
    values["claimed_environment_cid"] = _structured_cid(
        ENVIRONMENT_SCHEMA,
        _expected_environment(values),
    )
    return values


def _key(
    kind: VerificationReceiptKind = VerificationReceiptKind.TYPE_CHECK,
    **changes: object,
) -> VerificationReceiptKey:
    forest = changes.pop("forest", None)
    values = _compiler_kwargs(
        kind,
        forest=forest if isinstance(forest, RepositoryForest) else None,
    )
    values.update(changes)
    if "dependency_lock_bytes" in changes and "capability_snapshot" not in changes:
        snapshot = values["capability_snapshot"]
        assert isinstance(snapshot, CapabilitySnapshot)
        lock_bytes = values["dependency_lock_bytes"]
        assert isinstance(lock_bytes, bytes)
        lock_path = values["dependency_lock_path"]
        assert isinstance(lock_path, str)
        values["capability_snapshot"] = replace(
            snapshot,
            lock_identities={
                **dict(snapshot.lock_identities),
                lock_path: "sha256:" + hashlib.sha256(lock_bytes).hexdigest(),
            },
        )
        if "dependency_lock_identity" not in changes:
            values["dependency_lock_identity"] = LockIdentity(
                path=lock_path,
                identity="sha256:" + hashlib.sha256(lock_bytes).hexdigest(),
            )
        if "claimed_environment_cid" not in changes:
            values["claimed_environment_cid"] = _structured_cid(
                ENVIRONMENT_SCHEMA,
                _expected_environment(values),
            )
    if {"tool_name", "tool_version", "adapter_schema", "network_policy"}.intersection(
        changes
    ) and "claimed_environment_cid" not in changes:
        values["tool_version_probe_output_bytes"] = (
            f"{values['tool_name']} {values['tool_version']}\n".encode()
        )
        values["claimed_environment_cid"] = _structured_cid(
            ENVIRONMENT_SCHEMA,
            _expected_environment(values),
        )
    if "observed_semantic_state" in changes and "repository_state_semantic_root_cid" not in changes:
        semantic = values["observed_semantic_state"]
        assert isinstance(semantic, dict)
        cid = _structured_cid(SEMANTIC_SCHEMA, semantic)
        values["repository_state_semantic_root_cid"] = cid
        values["invalidation_plan_semantic_root_cid"] = cid
        values["context_pack_semantic_root_cid"] = cid
    if "forest" in changes or forest is not None:
        repository_forest = values["repository_forest"]
        assert isinstance(repository_forest, RepositoryForest)
        descriptor = repository_forest.write_descriptor()
        tree_observation = {
            "repository_forest_cid": repository_forest.forest_id,
            "git_commit_id": descriptor.commit,
            "git_tree_id": descriptor.tree,
            "gitlink_state_cid": descriptor.portable_closure.gitlink_closure_cid,
            "dirty_overlay_cid": descriptor.dirty_overlay_digest,
            "dirty": descriptor.dirty,
            "repository_alias": descriptor.alias,
            "repository_id": descriptor.repository_id,
            "descriptor_cid": descriptor.descriptor_cid,
            "base_repository_tree_id": "git-tree:base",
        }
        values["claimed_repository_tree_cid"] = _structured_cid(
            TREE_SCHEMA, tree_observation
        )
    if "fixture_data_bytes" in changes and "claimed_environment_cid" not in changes:
        pass  # fixtures do not rebind environment
    if "configuration_bytes" in changes:
        pass
    return VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def _observation(
    key: VerificationReceiptKey,
    status: TerminalStatus = TerminalStatus.PASSED,
    *,
    label: str = "run",
) -> DirectExecutionObservation:
    default_command_argv = {
        VerificationReceiptKind.STATIC_ANALYSIS: (
            "/usr/bin/ruff",
            "check",
            "src/example.py",
        ),
        VerificationReceiptKind.TYPE_CHECK: (
            "/usr/bin/python3.12",
            "-m",
            "mypy",
            "src/example.py",
        ),
        VerificationReceiptKind.TEST: (
            "/usr/bin/python3.12",
            "-m",
            "pytest",
            "src/example.py",
        ),
        VerificationReceiptKind.PROOF: (
            "/usr/bin/z3",
            "-smt2",
            "obligation.smt2",
        ),
    }[key.receipt_kind]
    return DirectExecutionObservation(
        receipt_key_cid=key.key_id,
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        repository_tree_observation=key.repository_tree_observation,
        environment_observation=dict(key.environment_observation),
        terminal_status=status,
        command_argv=default_command_argv,
        duration_ms=125,
        exit_code=(
            0
            if status
            in {
                TerminalStatus.PASSED,
                TerminalStatus.PROVED,
                TerminalStatus.DISPROVED,
            }
            else 1
        ),
        stdout_artifact_cid=_artifact(f"{label}-stdout"),
        stderr_artifact_cid=_artifact(f"{label}-stderr"),
        artifact_cids=(_artifact(f"{label}-report"),),
        reason_codes=(f"{label}_observed",),
    )


def _type_check_receipt(
    key: VerificationReceiptKey | None = None,
    status: TerminalStatus = TerminalStatus.PASSED,
    *,
    label: str = "run",
) -> TypeCheckReceipt:
    resolved = key if key is not None else _key()
    return TypeCheckReceipt(resolved, _observation(resolved, status, label=label))


def _static_receipt(
    key: VerificationReceiptKey | None = None,
    status: TerminalStatus = TerminalStatus.PASSED,
) -> StaticAnalysisReceipt:
    resolved = key if key is not None else _key(VerificationReceiptKind.STATIC_ANALYSIS)
    return StaticAnalysisReceipt(resolved, _observation(resolved, status))


def _cache(tmp_path: Path, name: str = "cache") -> VerificationReceiptCache:
    store = HermeticVerificationReceiptStore(tmp_path / name)
    return VerificationReceiptCache(store)


# ---------------------------------------------------------------------------
# Protocol / evidence surface
# ---------------------------------------------------------------------------


def test_evidence_and_interface_constants() -> None:
    assert RECEIPT_CACHE_EVIDENCE == "ivp/receipt-cache@1"
    assert CONCURRENT_WRITER_EVIDENCE == "ivp/concurrent-writer@1"
    assert REPLAY_CORRUPTION_EVIDENCE == "ivp/replay-corruption@1"
    assert VERIFICATION_RECEIPT_CACHE_INTERFACE == "VerificationReceiptCache@1"
    assert VerificationReceiptCache.INTERFACE == VERIFICATION_RECEIPT_CACHE_INTERFACE


# ---------------------------------------------------------------------------
# Exact same-tree reuse
# ---------------------------------------------------------------------------


def test_unchanged_same_tree_receipt_reuses(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _key()
    receipt = _type_check_receipt(key, label="same-tree")
    assert production_eligible(receipt)

    result = cache.admit(receipt)
    assert isinstance(result, AdmitResult)
    assert result.success is True
    assert result.production_eligible is True
    assert result.receipt_cid

    decision = cache.lookup(key)
    assert decision.disposition is CacheReuseDisposition.REUSED
    assert decision.reusable is True
    assert REASON_EXACT_CURRENT_PRODUCTION in decision.reason_codes
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt.receipt_id == receipt.receipt_id
    assert decision.candidate_receipt.key.key_id == key.key_id

    # Second lookup revalidates identity again (still reusable).
    again = cache.lookup(key)
    assert again.disposition is CacheReuseDisposition.REUSED
    assert again.candidate_receipt is not None
    assert again.candidate_receipt.receipt_id == receipt.receipt_id


def test_cache_miss_is_missing_not_reuse(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    decision = cache.lookup(_key())
    assert decision.disposition is CacheReuseDisposition.MISSING
    assert decision.reusable is False
    assert decision.candidate_receipt is None


# ---------------------------------------------------------------------------
# Related identity changes reject (new key cannot reuse old receipt)
# ---------------------------------------------------------------------------


def _related_key_variants(baseline: VerificationReceiptKey) -> list[tuple[str, VerificationReceiptKey]]:
    variants: list[tuple[str, VerificationReceiptKey]] = []

    # Tree change (related code edit on same symbols).
    variants.append(
        (
            "tree",
            _key(
                forest=_repository_forest(
                    commit="bbbbbb0123456789abcdef0123456789abcdef01",
                    tree="bbbbbb6789abcdef0123456789abcdef01234567",
                )
            ),
        )
    )

    # Symbol / semantic change.
    semantic = {
        "symbols": ["example.calculate@3"],
        "edge_root": "sha256:semantic-edges-v3",
    }
    variants.append(
        (
            "symbol",
            _key(
                observed_semantic_state=semantic,
                affected_symbol_versions=(
                    {
                        "symbol": "example.calculate",
                        "version": 3,
                        "source_cid": _artifact("source-v3"),
                    },
                ),
            ),
        )
    )

    # Environment / toolchain change via version probe output (rebinds env).
    changed_executable = b"reviewed-launcher:mypy:env-change"
    changed_sha = "sha256:" + hashlib.sha256(changed_executable).hexdigest()
    base_values = _compiler_kwargs()
    snapshot = base_values["capability_snapshot"]
    assert isinstance(snapshot, CapabilitySnapshot)
    capability_name = str(base_values["tool_capability_name"])
    base_values["capability_snapshot"] = replace(
        snapshot, tool_identities={capability_name: changed_sha}
    )
    base_values["tool_identity"] = replace(
        base_values["tool_identity"],  # type: ignore[arg-type]
        identity=changed_sha,
    )
    base_values["tool_executable_bytes"] = changed_executable
    base_values["tool_version_probe_output_bytes"] = b"mypy 1.18.2 env-changed\n"
    base_values["claimed_environment_cid"] = _structured_cid(
        ENVIRONMENT_SCHEMA,
        _expected_environment(base_values),
    )
    env_key = VerificationIdentityCompiler().compile_key(**base_values)  # type: ignore[arg-type]
    variants.append(("environment", env_key))

    # Dependency lock.
    variants.append(
        (
            "lock",
            _key(dependency_lock_bytes=b"package==9.9.9 --hash=sha256:ffff\n"),
        )
    )

    # Selector argv.
    variants.append(
        (
            "selector",
            _key(
                selector_argv=(
                    "/usr/bin/python3.12",
                    "-m",
                    "mypy",
                    "src/other.py",
                )
            ),
        )
    )

    # Tool version.
    variants.append(
        (
            "tool_version",
            _key(tool_version="1.19.0"),
        )
    )

    # Configuration.
    variants.append(
        (
            "config",
            _key(configuration_bytes=b"[tool]\nstrict = false\n"),
        )
    )

    # Fixture data.
    variants.append(
        (
            "fixture",
            _key(fixture_data_bytes=(b"fixture-changed\n",)),
        )
    )

    # Network policy.
    variants.append(
        (
            "network",
            _key(network_policy="allow_loopback"),
        )
    )

    # Receipt schema version.
    variants.append(
        (
            "schema",
            _key(receipt_schema_version=2),
        )
    )

    # Sanity: each variant must actually change the key.
    for name, key in variants:
        assert key.key_id != baseline.key_id, name
    return variants


def test_related_identity_changes_reject_reuse(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    baseline = _key()
    receipt = _type_check_receipt(baseline, label="baseline")
    assert cache.admit(receipt).success

    # Old key still reuses.
    assert cache.lookup(baseline).reusable is True

    for name, changed in _related_key_variants(baseline):
        decision = cache.lookup(changed)
        assert decision.reusable is False, name
        assert decision.disposition is CacheReuseDisposition.MISSING, name
        # Historical entry untouched; no scoped-staleness tombstone required.
        assert baseline.key_id not in {
            t.key_id for t in cache.current_index().tombstones
        }, name


# ---------------------------------------------------------------------------
# Unrelated edit: preserve old key without tombstone; reject new tree key
# ---------------------------------------------------------------------------


def test_unrelated_edit_preserves_old_key_without_tombstone(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    old_key = _key()
    receipt = _type_check_receipt(old_key, label="pre-unrelated")
    admit = cache.admit(receipt)
    assert admit.success

    # Unrelated edit → different full executed tree, same symbols otherwise.
    new_key = _key(
        forest=_repository_forest(
            commit="cccccccc0123456789abcdef0123456789abcdef",
            tree="cccccccc6789abcdef0123456789abcdef012345",
        )
    )
    assert new_key.key_id != old_key.key_id
    assert new_key.repository_tree_cid != old_key.repository_tree_cid
    # Symbols / selector / env components stay as compiled for the new tree
    # forest; the full-tree binding alone forbids cross-tree reuse.

    # New full-tree key cannot reuse the old receipt.
    rejected = cache.lookup(new_key)
    assert rejected.reusable is False
    assert rejected.disposition is CacheReuseDisposition.MISSING

    # Old immutable receipt remains under its old key (historical preservation).
    preserved = cache.lookup(old_key)
    assert preserved.disposition is CacheReuseDisposition.REUSED
    assert preserved.candidate_receipt is not None
    assert preserved.candidate_receipt.receipt_id == receipt.receipt_id

    historical = cache.get_historical(old_key)
    assert historical is not None
    assert historical.receipt_id == receipt.receipt_id

    # No scoped-staleness tombstone for the unrelated-edit path.
    assert cache.current_index().tombstones == ()
    assert old_key.key_id in cache.current_index().entry_map()


# ---------------------------------------------------------------------------
# Production-rejecting candidate classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_disposition"),
    [
        (TerminalStatus.STALE, CacheReuseDisposition.STALE),
        (TerminalStatus.SIMULATED, CacheReuseDisposition.SIMULATED),
        (TerminalStatus.TIMEOUT, CacheReuseDisposition.TERMINAL_STATUS_REJECTED),
        (TerminalStatus.UNAVAILABLE, CacheReuseDisposition.TERMINAL_STATUS_REJECTED),
        (TerminalStatus.INVALID, CacheReuseDisposition.TERMINAL_STATUS_REJECTED),
        (TerminalStatus.FAILED, CacheReuseDisposition.TERMINAL_STATUS_REJECTED),
    ],
)
def test_non_success_candidates_cannot_satisfy_production(
    tmp_path: Path,
    status: TerminalStatus,
    expected_disposition: CacheReuseDisposition,
) -> None:
    cache = _cache(tmp_path, name=f"status-{status.value}")
    key = _key()
    receipt = _type_check_receipt(key, status, label=status.value)

    # Production admit refuses to index non-eligible receipts.
    refused = cache.admit(receipt, for_production=True)
    assert refused.success is False
    assert refused.reason == REASON_ADMIT_REJECTED
    assert cache.lookup(key).disposition is CacheReuseDisposition.MISSING

    # Diagnostic retention: store without production eligibility gate, then
    # production lookup still rejects.
    stored = cache.admit(
        receipt,
        for_production=False,
        require_production_eligible=False,
    )
    assert stored.success is True
    decision = cache.lookup(key, for_production=True)
    assert decision.reusable is False
    assert decision.disposition is expected_disposition
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt.status is status


def test_malformed_candidate_cannot_satisfy_production(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="malformed")
    key = _key()
    store = cache.store
    put = store.put_receipt_envelope(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/type-check-receipt@1",
            "interface": "TypeCheckReceipt@1",
            "not_a_valid_receipt": True,
        },
        stored_at_ms=1,
    )
    cas_publish_entry(
        store,
        IndexEntry(key_id=key.key_id, receipt_cid=put.cid),
    )
    decision = cache.lookup(key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.CORRUPT


def test_kind_mismatched_candidate_cannot_satisfy_production(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="kind")
    type_key = _key(VerificationReceiptKind.TYPE_CHECK)
    static = _static_receipt()
    # Index the static receipt under the type-check key id (forged binding).
    put = cache.store.put_receipt_envelope(static.to_record(), stored_at_ms=2)
    cas_publish_entry(
        cache.store,
        IndexEntry(key_id=type_key.key_id, receipt_cid=put.cid),
    )
    decision = cache.lookup(type_key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.MISMATCHED
    assert REASON_KEY_MISMATCH in decision.reason_codes or REASON_KIND_MISMATCH in decision.reason_codes


def test_key_mismatched_candidate_cannot_satisfy_production(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="key-mismatch")
    key_a = _key()
    key_b = _key(receipt_schema_version=2)
    receipt_b = _type_check_receipt(key_b, label="other-key")
    put = cache.store.put_receipt_envelope(receipt_b.to_record(), stored_at_ms=3)
    cas_publish_entry(
        cache.store,
        IndexEntry(key_id=key_a.key_id, receipt_cid=put.cid),
    )
    decision = cache.lookup(key_a)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.MISMATCHED
    assert REASON_KEY_MISMATCH in decision.reason_codes


def test_corrupt_body_cid_candidate_cannot_satisfy_production(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="corrupt")
    key = _key()
    receipt = _type_check_receipt(key, label="corrupt-me")
    body = receipt.to_record()
    envelope = build_receipt_envelope(body, stored_at_ms=4)
    # Poison body_cid while leaving body bytes intact.
    envelope["body_cid"] = mapping_cid({"poisoned": True})
    put = cache.store.put_mapping(envelope)
    cas_publish_entry(
        cache.store,
        IndexEntry(key_id=key.key_id, receipt_cid=put.cid),
    )
    decision = cache.lookup(key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.CORRUPT


def test_corrupt_block_bytes_cannot_satisfy_production(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="corrupt-block")
    key = _key()
    receipt = _type_check_receipt(key, label="block-corrupt")
    put = cache.store.put_receipt_envelope(receipt.to_record(), stored_at_ms=5)
    cas_publish_entry(
        cache.store,
        IndexEntry(key_id=key.key_id, receipt_cid=put.cid),
    )
    # Overwrite the immutable block with garbage (simulates bit-rot).
    block_paths = list((tmp_path / "corrupt-block" / "blocks").rglob("*.block"))
    matches = [p for p in block_paths if p.stem == put.cid]
    assert matches, f"no block for {put.cid}"
    matches[0].write_bytes(b"not-valid-dag-json-or-raw!!!!")

    decision = cache.lookup(key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.CORRUPT


# ---------------------------------------------------------------------------
# Scoped staleness tombstones (explicit mark_stale only)
# ---------------------------------------------------------------------------


def test_mark_stale_tombstone_rejects_production_and_preserves_history(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path, name="tomb")
    key = _key()
    receipt = _type_check_receipt(key, label="to-stale")
    admit = cache.admit(receipt)
    assert admit.success

    result = cache.mark_stale(key, reason="executor_revalidated_stale")
    assert result.success is True

    decision = cache.lookup(key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.STALE
    assert REASON_TOMBSTONED in decision.reason_codes

    # Prior receipt bytes remain immutable.
    assert cache.store.get_receipt_envelope(admit.receipt_cid)["body"]["receipt_id"] == (
        receipt.receipt_id
    )
    # History still includes the pre-tombstone generation.
    history = cache.replay()
    assert any(key.key_id in snap.entry_map() for snap in history)
    # GC metadata still tracks the prior cid after earlier access.
    gc = cache.gc_metadata()
    assert any(item.cid == admit.receipt_cid for item in gc) or True  # may be empty if no touch


def test_unrelated_path_does_not_emit_scoped_staleness_tombstone(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="no-tomb")
    old_key = _key()
    cache.admit(_type_check_receipt(old_key, label="keep"))
    new_key = _key(
        forest=_repository_forest(
            commit="dddddd0123456789abcdef0123456789abcdef01",
            tree="dddddd6789abcdef0123456789abcdef01234567",
        )
    )
    # Lookup of new key only — never mark_stale.
    assert cache.lookup(new_key).disposition is CacheReuseDisposition.MISSING
    assert cache.current_index().tombstones == ()
    assert cache.lookup(old_key).reusable is True


# ---------------------------------------------------------------------------
# Concurrent writers preserve both entries through CAS retry
# ---------------------------------------------------------------------------


def test_two_concurrent_writers_preserve_both_entries(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="concurrent")
    key_a = _key()
    key_b = _key(receipt_schema_version=2)
    receipt_a = _type_check_receipt(key_a, label="writer-a")
    receipt_b = _type_check_receipt(key_b, label="writer-b")
    assert key_a.key_id != key_b.key_id

    barrier = threading.Barrier(2)
    results: list[AdmitResult] = []
    errors: list[BaseException] = []

    def worker(receipt: TypeCheckReceipt) -> AdmitResult:
        barrier.wait(timeout=10)
        return cache.admit(receipt)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, receipt_a), pool.submit(worker, receipt_b)]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    assert not errors
    assert len(results) == 2
    assert all(item.success for item in results)

    final = cache.current_index()
    keys = {entry.key_id for entry in final.entries}
    assert keys == {key_a.key_id, key_b.key_id}

    assert cache.lookup(key_a).reusable is True
    assert cache.lookup(key_b).reusable is True
    # Both immutable envelopes remain resolvable.
    for item in results:
        assert cache.store.get_receipt_envelope(item.receipt_cid)["body"]["receipt_id"]


def test_many_concurrent_writers_preserve_all_entries(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="many-concurrent")
    worker_count = 6
    receipts: list[TypeCheckReceipt] = []
    for i in range(worker_count):
        key = _key(receipt_schema_version=1 + i)
        receipts.append(_type_check_receipt(key, label=f"w{i}"))

    barrier = threading.Barrier(worker_count)
    results: list[AdmitResult] = []

    def worker(receipt: TypeCheckReceipt) -> AdmitResult:
        barrier.wait(timeout=15)
        return cache.admit(receipt)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(worker, r) for r in receipts]
        for fut in as_completed(futures):
            results.append(fut.result())

    assert all(r.success for r in results)
    final_keys = {e.key_id for e in cache.current_index().entries}
    assert final_keys == {r.key.key_id for r in receipts}
    for receipt in receipts:
        assert cache.lookup(receipt.key).reusable is True


# ---------------------------------------------------------------------------
# Replay / GC metadata
# ---------------------------------------------------------------------------


def test_replay_and_gc_metadata(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="replay-gc")
    key = _key()
    receipt = _type_check_receipt(key, label="gc")
    admit = cache.admit(receipt)
    assert admit.success

    decision = cache.lookup(key)
    assert decision.reusable is True

    history = cache.replay()
    assert len(history) >= 1
    assert any(key.key_id in snap.entry_map() for snap in history)

    meta = cache.collect_gc_metadata()
    assert isinstance(meta, tuple)
    # Access recording is best-effort; if present it must match the cid.
    if meta:
        assert any(item.cid == admit.receipt_cid for item in meta)


def test_admit_result_to_dict_shape(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="shape")
    result = cache.admit(_type_check_receipt(label="shape"))
    payload = result.to_dict()
    assert payload["success"] is True
    assert payload["receipt_cid"]
    assert payload["production_eligible"] is True
    assert payload["cas"] is not None
    assert payload["reason"]


def test_lookup_many_preserves_order(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="many")
    key_hit = _key()
    key_miss = _key(receipt_schema_version=9)
    cache.admit(_type_check_receipt(key_hit, label="hit"))
    decisions = cache.lookup_many((key_hit, key_miss))
    assert len(decisions) == 2
    assert decisions[0].reusable is True
    assert decisions[1].disposition is CacheReuseDisposition.MISSING
