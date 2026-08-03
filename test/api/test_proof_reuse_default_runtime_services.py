"""Default runtime composition for two-stage lookup and lazy issuer (PTR-147)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
from ipfs_accelerate_py.testing.proof_reuse.lookup import (
    PROOF_REUSE_TWO_STAGE_LOOKUP_INTERFACE,
    ProofReuseTwoStageLookup,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    CANDIDATE_CONTEXT_CACHE_SUBDIR,
    CERTIFICATE_CACHE_SUBDIR,
    DATASETS_GROTH16_ENABLE_ENV,
    DATASETS_GROTH16_REVIEWED_FILES_SHA256,
    DATASETS_VERIFIER_REVISION,
    DATASETS_VERIFIER_SNAPSHOT_BYTES,
    DATASETS_VERIFIER_SNAPSHOT_FILES,
    DATASETS_VERIFIER_SNAPSHOT_SHA256,
    DATASETS_VERIFIER_ZKP_TREE_OBJECT,
    DEFAULT_PROOF_REUSE_SERVICES_INTERFACE,
    DefaultProofReuseServices,
    LazyRealTestCertificateIssuer,
    compose_default_proof_reuse_services,
)


PTR151_REVISION = "1894e9dca7dced0690893d468e40751a14f0b15b"


def test_reviewed_datasets_revision_advances_to_ptr151_native_release() -> None:
    assert DATASETS_VERIFIER_REVISION == PTR151_REVISION
    assert (
        "ipfs_datasets_py/logic/zkp/test_pass_groth16_provider.py"
        in DATASETS_VERIFIER_SNAPSHOT_FILES
    )
    assert DATASETS_VERIFIER_SNAPSHOT_SHA256 == (
        "789339696dc10fb37dc0fd4fddd21b24af50b669479c194095f37dc904eab343"
    )
    assert DATASETS_VERIFIER_SNAPSHOT_BYTES == 873_708
    assert DATASETS_VERIFIER_ZKP_TREE_OBJECT == (
        "33fca9e5756798b7b77e417a6747b996e55d38c1"
    )
    # PTR-144 rust circuit sources must be pinned by content digest.
    assert (
        DATASETS_GROTH16_REVIEWED_FILES_SHA256["src/circuit.rs"]
        == "3d0ab0afd0f09711f4834d155d37dec228ce0d4e5608eb4371e4f4d8026cba04"
    )


def test_compose_defaults_separates_candidate_and_certificate_stores(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.READWRITE,
        root_path=tmp_path,
        cache_root=cache,
        installer=lambda _dep: False,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    assert isinstance(services, DefaultProofReuseServices)
    assert services.interface == DEFAULT_PROOF_REUSE_SERVICES_INTERFACE
    assert services.candidate_store is not None
    assert services.store is not None
    assert services.candidate_store is not services.store
    assert (cache / CANDIDATE_CONTEXT_CACHE_SUBDIR).is_dir()
    assert (cache / CERTIFICATE_CACHE_SUBDIR).is_dir()
    # Non-None lazy real issuer without eager optional imports at construction.
    assert services.issuer is not None
    assert isinstance(services.issuer, LazyRealTestCertificateIssuer)
    assert services.issuer.factory is None
    assert services.issuer.enable_env_published is False


def test_compose_defaults_builds_two_stage_lookup_and_revalidator(
    tmp_path: Path,
) -> None:
    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.READ,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=lambda _dep: False,
        environ={"IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0"},
    )
    assert services.lookup is not None
    assert isinstance(services.lookup, ProofReuseTwoStageLookup)
    assert services.lookup.interface == PROOF_REUSE_TWO_STAGE_LOOKUP_INTERFACE
    assert services.lookup.candidate_context_store is services.candidate_store
    assert services.revalidator is not None
    # Current-context provider may be present when identity services assemble.
    # Absence degrades but must not abort.
    assert services.degraded is False or services.reason_code


def test_collection_and_lookup_never_build_or_prove(tmp_path: Path) -> None:
    prove_calls: list[str] = []

    class _NoProveIssuer:
        def issue(self, *_a: Any, **_k: Any) -> Any:
            prove_calls.append("issue")
            raise AssertionError("lookup must never issue")

    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.READ,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        issuer=_NoProveIssuer(),
        installer=lambda _dep: False,
        environ={"IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0"},
    )
    # Construction complete without issue().
    assert prove_calls == []
    lookup = services.lookup
    assert lookup is not None
    # Locator-only miss returns RUN, never proves.
    from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
        TestExecutionKey,
        TestLocatorKey,
    )

    locator = TestLocatorKey(
        repository_id="repository:example",
        package_identity="package:example",
        node_id="test_direct.py::test_one",
    )
    execution_key = TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid="cid:policy",
    )
    decision = lookup.lookup(locator, execution_key)
    assert prove_calls == []
    assert getattr(decision, "is_skip", False) is False


def test_lazy_issuer_does_not_import_datasets_until_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure optional provider modules can be absent at construction.
    for name in list(importlib.sys.modules):
        if name.startswith("ipfs_datasets_py.logic.zkp.test_pass_groth16"):
            monkeypatch.delitem(importlib.sys.modules, name, raising=False)

    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.WRITE,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=lambda _dep: False,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    issuer = services.issuer
    assert issuer is not None
    assert getattr(issuer, "factory", None) is None
    # Missing artifacts → DEFERRED without publishing ENABLE_GROTH16.
    result = issuer.issue(
        {
            "receipt_cid": "cid:receipt",
            "locator_cid": "cid:locator",
        }
    )
    status = str(getattr(result, "status", "")).lower()
    assert "deferred" in status or status in {"", "run"}
    assert issuer.enable_env_published is False


def test_lazy_issuer_ptr155_gate_starts_no_provider_or_native_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Installer:
        def ensure_groth16_native_backend(self, **_kwargs: Any) -> Any:
            calls.append("provision")
            raise AssertionError("pending authority must not provision")

        def inspect_groth16_runtime(self) -> Any:
            calls.append("inspect")
            raise AssertionError("pending authority must not inspect")

    binary = tmp_path / "mutable-groth16"
    binary.write_bytes(b"unreviewed executable")
    artifacts = tmp_path / "mutable-keys"
    artifacts.mkdir()
    issuer = LazyRealTestCertificateIssuer(
        installer=_Installer(),
        binary_path=binary,
        artifacts_root=artifacts,
        environ={
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "1",
            "LD_PRELOAD": str(tmp_path / "attacker.so"),
            "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "attacker-keys"),
        },
    )
    monkeypatch.setattr(
        issuer,
        "_ensure_factory",
        lambda: (_ for _ in ()).throw(
            AssertionError("pending authority must not construct a provider")
        ),
    )

    result = issuer.issue({"receipt_cid": "cid:r", "locator_cid": "cid:l"})

    assert result.status == "certificate_deferred"
    assert result.reason == "positive_v4_issuance_pending_ptr155"
    assert calls == []
    assert issuer.factory is None
    assert issuer.enable_env_published is False


def test_lazy_issuer_skips_native_build_without_explicit_policy(
    tmp_path: Path,
) -> None:
    provisioned: list[str] = []

    class _Installer:
        def ensure_groth16_native_backend(self, *, consent: bool = False) -> Any:
            provisioned.append(f"consent={consent}")
            return SimpleNamespace(available=False, reason_code="denied")

        def inspect_groth16_runtime(self) -> dict[str, Any]:
            provisioned.append("inspect")
            return {"ready": False}

    issuer = LazyRealTestCertificateIssuer(
        store=None,
        installer=_Installer(),
        environ={
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
            "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "missing"),
        },
    )
    result = issuer.issue({"receipt_cid": "cid:r", "locator_cid": "cid:l"})
    assert provisioned == []  # policy denied — no provisioner call
    assert "deferred" in str(getattr(result, "status", "")).lower()


def test_generic_native_binary_alone_is_non_authoritative(tmp_path: Path) -> None:
    from ipfs_accelerate_py.testing.proof_reuse.publication import (
        Groth16ArtifactIdentityBindings,
    )

    # Binary without v4 keys is not provenance-ready.
    binary = tmp_path / "groth16"
    binary.write_bytes(b"\x7fELF-fake")
    bindings = Groth16ArtifactIdentityBindings.from_activated_artifacts(
        artifacts_root=tmp_path / "artifacts",
        binary_path=binary,
        environ={},
    )
    assert bindings.provenance_ready is False
    assert bindings.reason_code in {
        "artifact_manifest_pin_missing",
        "artifacts_root_missing",
        "test_pass_keys_missing",
    }


def test_explicit_injected_services_still_win(tmp_path: Path) -> None:
    lookup = object()
    store = object()
    candidate = object()
    issuer = object()
    provider = object()
    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.READWRITE,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        lookup=lookup,
        store=store,
        candidate_store=candidate,
        issuer=issuer,
        provider=provider,
        installer=lambda _dep: False,
    )
    assert services.lookup is lookup
    assert services.store is store
    assert services.candidate_store is candidate
    assert services.issuer is issuer
    assert services.provider is provider
    updated = services.with_overrides(issuer=object())
    assert updated.issuer is not issuer
    assert updated.candidate_store is candidate


def test_module_import_is_inert_without_optional_providers() -> None:
    # Re-import path must not require datasets or multiformats.
    module = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.services"
    )
    assert module.DATASETS_VERIFIER_REVISION == PTR151_REVISION
    assert callable(module.compose_default_proof_reuse_services)
