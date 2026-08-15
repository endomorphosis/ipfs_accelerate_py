"""Tests for AAE-039 runtime adapters and create_assurance_manifest.

Acceptance criteria enforced here:

* Adapters bind exact released index/capsule/context/verification/policy/
  state/storage/sealer interfaces and status mappings.
* Missing or drifted authority remains typed_unavailable.
* IVP VerificationCommitment cannot satisfy sealer capability.
* create_assurance_manifest seals bounded current-tree evidence without
  production policy change.
* Cold import performs no I/O.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import adapters as adapters_mod
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.adapters import (
    AAE_RUNTIME_ADAPTERS_EVIDENCE,
    AUTHORITY_KEYS,
    CAPSULE_ADAPTER_ID,
    CONTEXT_ADAPTER_ID,
    EXPECTED_CAPSULE_INTERFACE,
    EXPECTED_CONTEXT_INTERFACE,
    EXPECTED_INDEX_INTERFACE,
    EXPECTED_INDEX_SCHEMA,
    EXPECTED_POLICY_INTERFACE,
    EXPECTED_POLICY_SCHEMA,
    EXPECTED_SEALER_INTERFACE,
    EXPECTED_STORAGE_ARTIFACT_INTERFACE,
    EXPECTED_STORAGE_PACKAGE_INTERFACE,
    EXPECTED_STORAGE_SCHEMA,
    EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
    EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    EXPECTED_VERIFICATION_PUBLIC_INTERFACE,
    INDEX_ADAPTER_ID,
    POLICY_ADAPTER_ID,
    SEALER_ADAPTER_ID,
    SEALER_API_BINDINGS,
    STATE_ADAPTER_ID,
    STORAGE_ADAPTER_ID,
    VERIFICATION_ADAPTER_ID,
    AssuranceCapabilityUnavailable,
    AssuranceIndexAdapter,
    AuthorityStatus,
    CapabilityReason,
    SealStatus,
    SealerCapability,
    SurfaceCapability,
    load_capsule_adapter,
    load_context_adapter,
    load_index_adapter,
    load_policy_adapter,
    load_runtime_adapters,
    load_state_adapter,
    load_storage_adapter,
    load_verification_adapter,
    probe_all_authorities,
    probe_capsule_capability,
    probe_context_capability,
    probe_index_capability,
    probe_policy_capability,
    probe_sealer_capability,
    probe_state_capability,
    probe_storage_capability,
    probe_verification_capability,
    reject_ivp_commitment_as_sealer,
    sealer_capability_from_evidence,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.manifest import (
    ASSURANCE_MANIFEST_INTERFACE,
    ASSURANCE_MANIFEST_SCHEMA,
    CREATE_ASSURANCE_MANIFEST_INTERFACE,
    AssuranceManifest,
    AssuranceManifestError,
    RepositoryStateBinding,
    VerificationPolicyBinding,
    create_assurance_manifest,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/adapters.py"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/manifest.py"
)
ADAPTERS_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.adapters"
)
MANIFEST_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.manifest"
)

_OPT_OUTS = {
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _ok_index_surface(**overrides: Any) -> SimpleNamespace:
    class IncrementalSemanticIndex:
        pass

    fields: dict[str, Any] = {
        "IncrementalSemanticIndex": IncrementalSemanticIndex,
        "SEMANTIC_INDEX_SCHEMA": EXPECTED_INDEX_SCHEMA,
        "INCREMENTAL_SEMANTIC_INDEX_INTERFACE": EXPECTED_INDEX_INTERFACE,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_capsule_surface(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "compile_semantic_capsule": lambda *a, **k: {"capsule": True},
        "compile_semantic_capsules": lambda *a, **k: {"capsules": True},
        "SEMANTIC_CAPSULE_COMPILER_INTERFACE": EXPECTED_CAPSULE_INTERFACE,
        "SEMANTIC_CAPSULE_COMPILER_SCHEMA": (
            "ipfs-datasets.software-contracts.semantic-capsule-compiler@1"
        ),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_context_surface(**overrides: Any) -> SimpleNamespace:
    class ContextPacker:
        pass

    fields: dict[str, Any] = {
        "ContextPacker": ContextPacker,
        "pack_context": lambda *a, **k: {"packed": True},
        "CONTEXT_PACK_INTERFACE": EXPECTED_CONTEXT_INTERFACE,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_verification_surface(**overrides: Any) -> SimpleNamespace:
    class VerificationCommitment:
        pass

    class IncrementalVerificationPlanner:
        pass

    class VerificationExecutor:
        pass

    class VerificationReceiptCache:
        pass

    class ModelRoutePlanner:
        pass

    fields: dict[str, Any] = {
        "PUBLIC_API_INTERFACE": EXPECTED_VERIFICATION_PUBLIC_INTERFACE,
        "create_verification_plan": lambda *a, **k: {"plan": True},
        "choose_model_route": lambda *a, **k: {"route": "deterministic_only"},
        "build_verification_commitment": lambda *a, **k: VerificationCommitment(),
        "VerificationCommitment": VerificationCommitment,
        "IncrementalVerificationPlanner": IncrementalVerificationPlanner,
        "VerificationExecutor": VerificationExecutor,
        "VerificationReceiptCache": VerificationReceiptCache,
        "ModelRoutePlanner": ModelRoutePlanner,
        "VERIFICATION_PLANNER_INTERFACE": "IncrementalVerificationPlanner@1",
        "VERIFICATION_EXECUTOR_INTERFACE": "VerificationExecutor@1",
        "VERIFICATION_RECEIPT_CACHE_INTERFACE": "VerificationReceiptCache@1",
        "MODEL_ROUTE_PLANNER_INTERFACE": "ModelRoutePlanner@1",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_policy_surface(**overrides: Any) -> SimpleNamespace:
    class AssurancePolicyRepository:
        pass

    fields: dict[str, Any] = {
        "AssurancePolicyRepository": AssurancePolicyRepository,
        "ASSURANCE_POLICY_REPOSITORY_INTERFACE": EXPECTED_POLICY_INTERFACE,
        "POLICY_CAS_SCHEMA": EXPECTED_POLICY_SCHEMA,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_state_surface(**overrides: Any) -> SimpleNamespace:
    class DurableStateRootAdapter:
        def compare_and_swap_root(self, *a: Any, **k: Any) -> Any:
            return {"cas": True}

        def current_root(self, *a: Any, **k: Any) -> Any:
            return {"root": True}

        def get_verified(self, *a: Any, **k: Any) -> Any:
            return {"get": True}

        def put_verified(self, *a: Any, **k: Any) -> Any:
            return {"put": True}

    class DurableStateRoots:
        pass

    fields: dict[str, Any] = {
        "DurableStateRootAdapter": DurableStateRootAdapter,
        "DurableStateRoots": DurableStateRoots,
        "CAMPAIGN_STATE_INTERFACE": "MutationCampaignState@1",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_storage_surface(**overrides: Any) -> SimpleNamespace:
    class DurableCoordinationStore:
        pass

    class DurableAssuranceArtifactStore:
        pass

    fields: dict[str, Any] = {
        "DurableCoordinationStore": DurableCoordinationStore,
        "DurableAssuranceArtifactStore": DurableAssuranceArtifactStore,
        "ARTIFACT_MODULE_INTERFACE": EXPECTED_STORAGE_ARTIFACT_INTERFACE,
        "PACKAGE_INTERFACE": EXPECTED_STORAGE_PACKAGE_INTERFACE,
        "ASSURANCE_ARTIFACT_STORE_SCHEMA": EXPECTED_STORAGE_SCHEMA,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _released_sealer_surface(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "__name__": "fake.released.proof_sealer",
        "IncrementalProofSealer": object,
        "DeltaSeal": object,
        "build_delta_seal": lambda *a, **k: {"delta": True},
        "publish_delta_seal": lambda *a, **k: {"published": True},
        "FullCheckpointSeal": object,
        "create_full_checkpoint": lambda *a, **k: {"full": True},
        "publish_full_checkpoint": lambda *a, **k: {"published": True},
        "SEALER_INTERFACE": EXPECTED_SEALER_INTERFACE,
        "IS_ZK_SEALER": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _all_ok_surfaces() -> dict[str, Any]:
    return {
        "index_surface": _ok_index_surface(),
        "capsule_surface": _ok_capsule_surface(),
        "context_surface": _ok_context_surface(),
        "verification_surface": _ok_verification_surface(),
        "policy_surface": _ok_policy_surface(),
        "state_surface": _ok_state_surface(),
        "storage_surface": _ok_storage_surface(),
        "sealer_surface": _released_sealer_surface(),
    }


def _repo_state(**overrides: Any) -> dict[str, Any]:
    payload = {
        "repository_id": "repository:sha256:test-repo-identity-aae039",
        "repository_state_cid": _cid("repo-state-aae039"),
        "revision": "deadbeef",
        "source_root_cid": _cid("source-root-aae039"),
        "environment_cid": _cid("env-aae039"),
        "dependency_lock_cid": _cid("dep-lock-aae039"),
    }
    payload.update(overrides)
    return payload


def _policy(**overrides: Any) -> dict[str, Any]:
    payload = {
        "policy_cid": _cid("verification-policy-aae039"),
        "policy_id": "aae-test-policy",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Constants / structural
# ---------------------------------------------------------------------------


def test_evidence_and_interface_constants() -> None:
    assert AAE_RUNTIME_ADAPTERS_EVIDENCE == "aae/runtime-adapters@1"
    assert CREATE_ASSURANCE_MANIFEST_INTERFACE == "create_assurance_manifest@1"
    assert ASSURANCE_MANIFEST_INTERFACE == "AssuranceManifest@1"
    assert ASSURANCE_MANIFEST_SCHEMA.endswith("@1")
    assert EXPECTED_INDEX_INTERFACE == "IncrementalSemanticIndex@1"
    assert EXPECTED_CAPSULE_INTERFACE == "SemanticCapsuleCompiler@1"
    assert EXPECTED_CONTEXT_INTERFACE == "ContextPack@1"
    assert EXPECTED_POLICY_INTERFACE == "AssurancePolicyRepository@1"
    assert EXPECTED_SEALER_INTERFACE == "IncrementalProofSealer@1"
    assert set(AUTHORITY_KEYS) == {
        "index",
        "capsule",
        "context",
        "verification",
        "policy",
        "state",
        "storage",
        "sealer",
    }
    assert "IncrementalProofSealer" in SEALER_API_BINDINGS
    assert SEALER_API_BINDINGS["IncrementalProofSealer"].endswith(".sealer")


def test_adapter_module_exports_required_interfaces() -> None:
    for name in (
        "AssuranceIndexAdapter",
        "AssuranceCapsuleAdapter",
        "AssuranceContextAdapter",
        "AssuranceVerificationAdapter",
        "AssurancePolicyAdapter",
        "AssuranceStateAdapter",
        "AssuranceStorageAdapter",
        "SealerCapability",
        "load_runtime_adapters",
        "probe_sealer_capability",
        "reject_ivp_commitment_as_sealer",
        "probe_all_authorities",
    ):
        assert hasattr(adapters_mod, name), name


def test_module_source_has_no_module_level_io() -> None:
    for path in (ADAPTER_PATH, MANIFEST_PATH):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        forbidden_calls = {"open", "urlopen", "Popen", "run", "system"}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in forbidden_calls:
                    pytest.fail(
                        f"{path.name}: module-level I/O call forbidden: {name}"
                    )


def _hermetic_import_probe(module_name: str) -> subprocess.CompletedProcess[str]:
    script = f"""\
import json
import os
import sys

effects = []

def forbidden(name):
    def call(*args, **kwargs):
        effects.append(name)
        raise AssertionError(f"forbidden import side effect: {{name}}")
    return call

os.system = forbidden("os.system")
for name in ("posix_spawn", "posix_spawnp", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
    if hasattr(os, name):
        setattr(os, name, forbidden(f"os.{{name}}"))

def audit(event, args):
    if event in {{
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "subprocess.Popen",
        "socket.connect",
        "socket.bind",
    }}:
        effects.append(event)
        raise AssertionError(f"forbidden import side effect: {{event}}")

sys.addaudithook(audit)

prefix = {module_name!r}
for name in list(sys.modules):
    if name == prefix or name.startswith(prefix + "."):
        del sys.modules[name]

import importlib
mod = importlib.import_module(prefix)
assert hasattr(mod, "__doc__")
print(json.dumps({{"ok": True, "effects": effects}}))
"""
    environment = dict(os.environ)
    environment.update(_OPT_OUTS)
    pythonpath = os.pathsep.join(
        [
            str(REPO_ROOT),
            str(REPO_ROOT / "ipfs_kit_py"),
            str(REPO_ROOT / "ipfs_datasets_py"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    environment["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_adapters_performs_no_io() -> None:
    result = _hermetic_import_probe(ADAPTERS_MODULE)
    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_import_manifest_performs_no_io() -> None:
    result = _hermetic_import_probe(MANIFEST_MODULE)
    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Happy-path probes with injected surfaces
# ---------------------------------------------------------------------------


def test_index_adapter_runtime_view() -> None:
    surface = _ok_index_surface()
    adapter = load_index_adapter(surface)
    view = adapter.runtime_view()
    assert view["adapter_id"] == INDEX_ADAPTER_ID
    assert adapter.capability.available is True
    assert adapter.capability.status == AuthorityStatus.AVAILABLE.value
    assert adapter.index_class.__name__ == "IncrementalSemanticIndex"


def test_capsule_adapter_runtime_view() -> None:
    surface = _ok_capsule_surface()
    adapter = load_capsule_adapter(surface)
    assert adapter.compile_semantic_capsule()["capsule"] is True
    assert adapter.capability.operations[0] == "compile_semantic_capsule"
    assert adapter.runtime_view()["released_interface"] == EXPECTED_CAPSULE_INTERFACE


def test_context_adapter_runtime_view() -> None:
    surface = _ok_context_surface()
    adapter = load_context_adapter(surface)
    assert adapter.pack_context()["packed"] is True
    assert adapter.packer_class.__name__ == "ContextPacker"


def test_verification_adapter_marks_commitment_non_sealer() -> None:
    surface = _ok_verification_surface()
    adapter = load_verification_adapter(surface)
    view = adapter.runtime_view()
    assert view["commitment_is_proof_sealer"] is False
    assert view["commitment_is_zk"] is False
    assert view["can_satisfy_sealer_capability"] is False
    commitment = adapter.build_verification_commitment()
    assert type(commitment).__name__ == "VerificationCommitment"


def test_policy_state_storage_adapters() -> None:
    policy = load_policy_adapter(_ok_policy_surface())
    assert policy.repository_class.__name__ == "AssurancePolicyRepository"
    assert policy.runtime_view()["production_policy_change"] is False

    state = load_state_adapter(_ok_state_surface())
    store = state.state_root_adapter_class()
    assert store.compare_and_swap_root()["cas"] is True

    storage = load_storage_adapter(surface=_ok_storage_surface())
    assert storage.coordination_store_class.__name__ == "DurableCoordinationStore"
    assert storage.artifact_store_class.__name__ == "DurableAssuranceArtifactStore"


def test_load_runtime_adapters_with_injected_surfaces() -> None:
    runtime = load_runtime_adapters(
        **_all_ok_surfaces(),
        require_sealer=True,
        require_execution=True,
    )
    runtime.require_execution_surfaces()
    assert runtime.sealer.available is True
    mapping = runtime.to_mapping()
    assert mapping["evidence_id"] == AAE_RUNTIME_ADAPTERS_EVIDENCE
    for key in AUTHORITY_KEYS:
        assert key in mapping["authorities"]
        assert mapping["authorities"][key]["available"] is True
        assert mapping["authorities"][key]["status"] == AuthorityStatus.AVAILABLE.value
    assert mapping["sealer"]["can_be_satisfied_by_ivp_commitment"] is False
    assert "IncrementalProofSealer" in runtime.sealer.operations


# ---------------------------------------------------------------------------
# Fail-closed: missing / drifted → typed_unavailable
# ---------------------------------------------------------------------------


def test_missing_index_export_is_typed_unavailable() -> None:
    surface = _ok_index_surface()
    delattr(surface, "IncrementalSemanticIndex")
    cap = probe_index_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.MISSING_EXPORTS.value
    with pytest.raises(AssuranceCapabilityUnavailable) as excinfo:
        load_index_adapter(surface)
    assert excinfo.value.status == AuthorityStatus.TYPED_UNAVAILABLE.value


def test_incompatible_capsule_interface_is_typed_unavailable() -> None:
    surface = _ok_capsule_surface(
        SEMANTIC_CAPSULE_COMPILER_INTERFACE="OtherCompiler@1"
    )
    cap = probe_capsule_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.INCOMPATIBLE_CAPABILITY.value


def test_stale_context_interface_is_typed_unavailable() -> None:
    # Same stem with older major → stale reason under typed_unavailable status.
    surface = _ok_context_surface(CONTEXT_PACK_INTERFACE="ContextPack@0")
    cap = probe_context_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.STALE_CAPABILITY.value


def test_incompatible_policy_schema_is_typed_unavailable() -> None:
    surface = _ok_policy_surface(
        POLICY_CAS_SCHEMA="ipfs-kit.adversarial-assurance-store.policy-cas@9"
    )
    cap = probe_policy_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.INCOMPATIBLE_CAPABILITY.value


def test_missing_state_methods_fail_closed() -> None:
    class IncompleteAdapter:
        def compare_and_swap_root(self, *a: Any, **k: Any) -> Any:
            return None

    surface = _ok_state_surface(DurableStateRootAdapter=IncompleteAdapter)
    cap = probe_state_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.MISSING_EXPORTS.value


def test_incompatible_storage_package_interface() -> None:
    surface = _ok_storage_surface(PACKAGE_INTERFACE="OtherStore@1")
    cap = probe_storage_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value


def test_unavailable_capability_require_raises() -> None:
    cap = SurfaceCapability(
        available=False,
        adapter_id=INDEX_ADAPTER_ID,
        interface_id="AssuranceIndexAdapter@1",
        schema=EXPECTED_INDEX_SCHEMA,
        authority="index",
        operations=(),
        fingerprints={},
        status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        reason_code=CapabilityReason.CAPABILITY_UNAVAILABLE.value,
        diagnostic="down",
    )
    with pytest.raises(AssuranceCapabilityUnavailable) as excinfo:
        cap.require_available("use")
    assert excinfo.value.to_mapping()["available"] is False
    assert excinfo.value.to_mapping()["status"] == AuthorityStatus.TYPED_UNAVAILABLE.value


def test_import_failed_probe_is_typed_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(name: str, *, adapter_id: str) -> Any:
        raise AssuranceCapabilityUnavailable(
            "load",
            CapabilityReason.IMPORT_FAILED.value,
            f"import of {name!r} failed: no module",
            adapter_id=adapter_id,
            retryable=True,
            status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        )

    monkeypatch.setattr(adapters_mod, "_import_module", boom)
    cap = probe_index_capability()
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.IMPORT_FAILED.value
    assert cap.retryable is True


# ---------------------------------------------------------------------------
# Sealer: IVP cannot satisfy; exact released bindings
# ---------------------------------------------------------------------------


def test_sealer_unavailable_by_default_on_empty_surface() -> None:
    cap = probe_sealer_capability(surface=SimpleNamespace(__name__="empty"))
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.seal_status == SealStatus.TYPED_UNAVAILABLE.value
    assert cap.can_be_satisfied_by_ivp_commitment is False
    with pytest.raises(AssuranceCapabilityUnavailable):
        cap.require_available("seal")


def test_ivp_commitment_cannot_satisfy_sealer() -> None:
    class VerificationCommitment:
        pass

    for evidence in (
        VerificationCommitment,
        VerificationCommitment(),
        "VerificationCommitment",
        "build_verification_commitment",
        EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        {
            "schema": EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            "interface_id": EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        },
    ):
        cap = sealer_capability_from_evidence(evidence)
        assert cap.available is False, evidence
        assert cap.reason_code == CapabilityReason.IVP_COMMITMENT_NOT_SEALER.value
        assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
        assert cap.can_be_satisfied_by_ivp_commitment is False


def test_reject_ivp_commitment_as_sealer_raises() -> None:
    with pytest.raises(AssuranceCapabilityUnavailable) as excinfo:
        reject_ivp_commitment_as_sealer("build_verification_commitment")
    assert excinfo.value.reason_code == CapabilityReason.IVP_COMMITMENT_NOT_SEALER.value
    assert excinfo.value.adapter_id == SEALER_ADAPTER_ID


def test_verification_commitment_cannot_become_sealer() -> None:
    surface = _ok_verification_surface()
    adapter = load_verification_adapter(surface)
    commitment = adapter.build_verification_commitment()
    cap = sealer_capability_from_evidence(commitment)
    assert cap.available is False
    assert cap.reason_code == CapabilityReason.IVP_COMMITMENT_NOT_SEALER.value
    probe_cap = probe_sealer_capability(surface=commitment)
    assert probe_cap.available is False


def test_sealer_capability_forges_cannot_enable_ivp_substitution() -> None:
    cap = SealerCapability(
        available=True,
        seal_status=SealStatus.AVAILABLE.value,
        can_be_satisfied_by_ivp_commitment=True,  # attempted forge
        is_zk=True,
        is_full_or_delta_seal=True,
    )
    assert cap.can_be_satisfied_by_ivp_commitment is False
    mapping = cap.to_mapping()
    assert mapping["can_be_satisfied_by_ivp_commitment"] is False


def test_released_sealer_surface_is_available() -> None:
    surface = _released_sealer_surface()
    cap = probe_sealer_capability(surface=surface)
    assert cap.available is True
    assert cap.seal_status == SealStatus.AVAILABLE.value
    assert cap.status == AuthorityStatus.AVAILABLE.value
    assert cap.is_full_or_delta_seal is True
    assert "IncrementalProofSealer" in cap.operations
    assert cap.can_be_satisfied_by_ivp_commitment is False


def test_stale_sealer_interface_is_typed_unavailable() -> None:
    surface = _released_sealer_surface(
        SEALER_INTERFACE="IncrementalProofSealer@0"
    )
    cap = probe_sealer_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.STALE_CAPABILITY.value


def test_partial_sealer_exports_are_typed_unavailable() -> None:
    surface = _released_sealer_surface()
    delattr(surface, "DeltaSeal")
    cap = probe_sealer_capability(surface=surface)
    assert cap.available is False
    assert cap.status == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert cap.reason_code == CapabilityReason.MISSING_EXPORTS.value


# ---------------------------------------------------------------------------
# create_assurance_manifest
# ---------------------------------------------------------------------------


def test_create_assurance_manifest_hermetic_typed_unavailable() -> None:
    manifest = create_assurance_manifest(_repo_state(), _policy())
    assert isinstance(manifest, AssuranceManifest)
    assert manifest.interface_id == ASSURANCE_MANIFEST_INTERFACE
    assert manifest.schema == ASSURANCE_MANIFEST_SCHEMA
    assert manifest.production_policy_changed is False
    assert manifest.evidence_id == AAE_RUNTIME_ADAPTERS_EVIDENCE
    assert manifest.manifest_cid.startswith("b")
    for key in AUTHORITY_KEYS:
        entry = manifest.authority_status[key]
        assert entry["available"] is False
        assert entry["status"] == AuthorityStatus.TYPED_UNAVAILABLE.value
    assert set(manifest.typed_unavailable_authorities()) == set(AUTHORITY_KEYS)
    assert manifest.seal_status == SealStatus.TYPED_UNAVAILABLE.value
    # Round-trip identity.
    again = AssuranceManifest.from_dict(manifest.to_dict())
    assert again.manifest_cid == manifest.manifest_cid


def test_create_assurance_manifest_with_bound_authorities() -> None:
    runtime = load_runtime_adapters(
        **_all_ok_surfaces(),
        require_sealer=True,
        require_execution=True,
    )
    manifest = create_assurance_manifest(
        _repo_state(),
        _policy(),
        runtime=runtime,
        detectors=(
            {
                "detector_id": "unit.test_example",
                "detector_kind": "unit_test",
            },
        ),
    )
    assert manifest.authority_available("index") is True
    assert manifest.authority_available("sealer") is True
    assert manifest.seal_status == SealStatus.AVAILABLE.value
    assert manifest.typed_unavailable_authorities() == ()
    assert manifest.production_policy_changed is False
    assert len(manifest.detectors) == 1
    payload = manifest.to_dict()
    assert payload["verification_policy_cid"] == _policy()["policy_cid"]
    assert payload["repository_state_cid"] == _repo_state()["repository_state_cid"]


def test_create_assurance_manifest_with_surfaces_kwargs() -> None:
    manifest = create_assurance_manifest(
        _repo_state(),
        _policy()["policy_cid"],
        **_all_ok_surfaces(),
        require_execution_authorities=True,
        require_sealer=True,
    )
    assert all(
        manifest.authority_status[key]["available"] for key in AUTHORITY_KEYS
    )


def test_create_assurance_manifest_require_execution_fails_closed() -> None:
    with pytest.raises(AssuranceManifestError) as excinfo:
        create_assurance_manifest(
            _repo_state(),
            _policy(),
            authority_status={
                "index": {
                    "available": True,
                    "status": AuthorityStatus.AVAILABLE.value,
                },
                "capsule": {
                    "available": False,
                    "status": AuthorityStatus.TYPED_UNAVAILABLE.value,
                    "reason_code": CapabilityReason.MISSING_EXPORTS.value,
                    "diagnostic": "capsule missing",
                },
            },
            require_execution_authorities=True,
        )
    assert "typed unavailable" in str(excinfo.value).lower()
    assert excinfo.value.reason_code == CapabilityReason.MISSING_EXPORTS.value


def test_create_assurance_manifest_rejects_policy_change_flag() -> None:
    with pytest.raises(AssuranceManifestError):
        AssuranceManifest(
            repository_id=_repo_state()["repository_id"],
            repository_state_cid=_repo_state()["repository_state_cid"],
            verification_policy_cid=_policy()["policy_cid"],
            authority_status={},
            repository_state={},
            verification_policy={},
            production_policy_changed=True,
        )


def test_create_assurance_manifest_rejects_private_metadata() -> None:
    with pytest.raises(AssuranceManifestError):
        create_assurance_manifest(
            _repo_state(),
            _policy(),
            metadata={"api_key": "should-never-appear"},
        )


def test_create_assurance_manifest_rejects_unknown_authority_keys() -> None:
    with pytest.raises(AssuranceManifestError):
        create_assurance_manifest(
            _repo_state(),
            _policy(),
            authority_status={
                "index": {"available": True, "status": "available"},
                "invented_authority": {"available": True, "status": "available"},
            },
        )


def test_repository_and_policy_binding_normalization() -> None:
    repo = RepositoryStateBinding.normalize(_repo_state())
    assert repo.identity_cid.startswith("b")
    policy = VerificationPolicyBinding.normalize(_policy()["policy_cid"])
    assert policy.policy_cid == _policy()["policy_cid"]
    again = VerificationPolicyBinding.normalize(policy)
    assert again.identity_cid == policy.identity_cid


def test_drifted_status_maps_to_typed_unavailable() -> None:
    manifest = create_assurance_manifest(
        _repo_state(),
        _policy(),
        authority_status={
            key: {
                "available": False,
                "status": "stale",  # granular drift label
                "reason_code": "stale_capability",
                "diagnostic": "drifted pin",
            }
            for key in AUTHORITY_KEYS
        },
    )
    for key in AUTHORITY_KEYS:
        assert (
            manifest.authority_status[key]["status"]
            == AuthorityStatus.TYPED_UNAVAILABLE.value
        )


def test_detection_manifest_projection() -> None:
    runtime = load_runtime_adapters(
        **_all_ok_surfaces(),
        require_sealer=False,
        require_execution=True,
    )
    manifest = create_assurance_manifest(
        _repo_state(),
        _policy(),
        runtime=runtime,
        detectors=(),
        dependency_edges=(),
        claims=(),
    )
    detection = manifest.as_detection_manifest()
    assert detection.repository_id == manifest.repository_id
    assert detection.repository_state_cid == manifest.repository_state_cid
    assert detection.manifest_cid.startswith("b")


# ---------------------------------------------------------------------------
# Live surfaces (when nested packages are present)
# ---------------------------------------------------------------------------


def test_live_index_probe_when_available() -> None:
    try:
        import ipfs_datasets_py.logic.software_contracts.semantic_index  # noqa: F401
    except Exception:
        pytest.skip("semantic index surface unavailable")
    cap = probe_index_capability()
    assert cap.available is True
    assert cap.status == AuthorityStatus.AVAILABLE.value
    adapter = load_index_adapter()
    assert adapter.capability.available is True


def test_live_capsule_probe_when_available() -> None:
    try:
        import ipfs_datasets_py.logic.software_contracts.semantic_state  # noqa: F401
    except Exception:
        pytest.skip("semantic state surface unavailable")
    cap = probe_capsule_capability()
    assert cap.available is True
    assert load_capsule_adapter().capability.available is True


def test_live_context_probe_when_available() -> None:
    try:
        import ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack  # noqa: F401
    except Exception:
        pytest.skip("context pack surface unavailable")
    cap = probe_context_capability()
    assert cap.available is True


def test_live_verification_probe_when_available() -> None:
    try:
        import ipfs_accelerate_py.agent_supervisor.verification  # noqa: F401
    except Exception:
        pytest.skip("verification surface unavailable")
    cap = probe_verification_capability()
    assert cap.available is True
    assert cap.fingerprints.get("commitment_is_proof_sealer") == "false"
    adapter = load_verification_adapter()
    assert adapter.runtime_view()["can_satisfy_sealer_capability"] is False


def test_live_policy_state_storage_probes_when_available() -> None:
    try:
        import ipfs_kit_py.adversarial_assurance_store  # noqa: F401
        import ipfs_kit_py.mcp_server.mcplusplus.coordination_storage  # noqa: F401
        import ipfs_kit_py.mcp_server.mcplusplus.state_root_adapter  # noqa: F401
    except Exception:
        pytest.skip("kit assurance store / coordination surface unavailable")
    assert probe_policy_capability().available is True
    assert probe_state_capability().available is True
    assert probe_storage_capability().available is True
    assert load_policy_adapter().capability.available is True
    assert load_state_adapter().capability.available is True
    assert load_storage_adapter().capability.available is True


def test_live_sealer_probe_uses_aae006_bindings() -> None:
    try:
        import ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer  # noqa: F401
    except Exception:
        pytest.skip("released sealer surface unavailable on this tree")
    cap = probe_sealer_capability()
    assert cap.available is True
    assert cap.status == AuthorityStatus.AVAILABLE.value
    assert cap.seal_status == SealStatus.AVAILABLE.value
    assert cap.is_full_or_delta_seal is True
    assert cap.can_be_satisfied_by_ivp_commitment is False
    for symbol in SEALER_API_BINDINGS:
        assert symbol in cap.operations
        assert symbol in cap.bindings
    assert cap.fingerprints.get("binding_source") == "aae-006-prerequisite-receipt"


def test_live_create_assurance_manifest_probe() -> None:
    try:
        import ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer  # noqa: F401
        import ipfs_datasets_py.logic.software_contracts.semantic_index  # noqa: F401
        import ipfs_kit_py.adversarial_assurance_store  # noqa: F401
    except Exception:
        pytest.skip("live authorities unavailable")
    manifest = create_assurance_manifest(
        _repo_state(),
        _policy(),
        probe_live_authorities=True,
    )
    assert manifest.manifest_cid
    assert manifest.production_policy_changed is False
    # On the released tree, authorities should bind successfully.
    for key in AUTHORITY_KEYS:
        assert key in manifest.authority_status
        assert manifest.authority_status[key]["status"] in {
            AuthorityStatus.AVAILABLE.value,
            AuthorityStatus.TYPED_UNAVAILABLE.value,
        }
    # Prefer availability for released pins.
    assert manifest.authority_status["sealer"]["available"] is True


def test_probe_all_authorities_closed_keys() -> None:
    status = probe_all_authorities(
        index_surface=_ok_index_surface(),
        capsule_surface=_ok_capsule_surface(),
        context_surface=_ok_context_surface(),
        verification_surface=_ok_verification_surface(),
        policy_surface=_ok_policy_surface(),
        state_surface=_ok_state_surface(),
        storage_surface=_ok_storage_surface(),
        sealer_surface=_released_sealer_surface(),
    )
    assert set(status) == set(AUTHORITY_KEYS)
    for key, entry in status.items():
        assert entry["available"] is True
        assert entry["status"] == AuthorityStatus.AVAILABLE.value


def test_authority_status_and_seal_status_are_closed() -> None:
    assert {s.value for s in AuthorityStatus} == {
        "available",
        "typed_unavailable",
    }
    assert {s.value for s in SealStatus} == {
        "available",
        "typed_unavailable",
        "inconclusive",
    }


def test_load_runtime_adapters_require_sealer_fails_closed() -> None:
    with pytest.raises(AssuranceCapabilityUnavailable) as excinfo:
        load_runtime_adapters(
            index_surface=_ok_index_surface(),
            capsule_surface=_ok_capsule_surface(),
            context_surface=_ok_context_surface(),
            verification_surface=_ok_verification_surface(),
            policy_surface=_ok_policy_surface(),
            state_surface=_ok_state_surface(),
            storage_surface=_ok_storage_surface(),
            sealer_surface=SimpleNamespace(__name__="empty"),
            require_sealer=True,
            require_execution=True,
        )
    assert excinfo.value.adapter_id == SEALER_ADAPTER_ID
    assert excinfo.value.status == AuthorityStatus.TYPED_UNAVAILABLE.value
