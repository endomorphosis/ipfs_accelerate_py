"""Runtime adapter tests for SCG-023.

Acceptance criteria enforced here:

* Stale / missing / incompatible capability fails closed.
* IVP ``VerificationCommitment`` cannot satisfy sealer capability.
* Imports perform no I/O.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_governor import adapters as adapters_mod
from ipfs_accelerate_py.agent_supervisor.semantic_governor.adapters import (
    DATASETS_ADAPTER_ID,
    EXPECTED_DATASETS_API_SCHEMA,
    EXPECTED_DATASETS_PACKAGE_INTERFACE,
    EXPECTED_HARNESS_INTERFACE,
    EXPECTED_HARNESS_SCHEMA,
    EXPECTED_STORE_ARTIFACT_INTERFACE,
    EXPECTED_STORE_INTERFACE,
    EXPECTED_STORE_SCHEMA,
    EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
    EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    HARNESS_ADAPTER_ID,
    IncrementalSealerCapability,
    REQUIRED_DATASETS_APIS,
    SCG_RUNTIME_ADAPTERS_EVIDENCE,
    SEALER_CAPABILITY_ID,
    STORE_ADAPTER_ID,
    SealStatus,
    SurfaceCapability,
    VERIFICATION_ADAPTER_ID,
    CapabilityStatus,
    GovernorCapabilityUnavailable,
    GovernorDatasetsAdapter,
    GovernorHarnessAdapter,
    GovernorStoreAdapter,
    GovernorVerificationAdapter,
    load_datasets_adapter,
    load_harness_adapter,
    load_runtime_adapters,
    load_store_adapter,
    load_verification_adapter,
    probe_datasets_capability,
    probe_harness_capability,
    probe_incremental_sealer_capability,
    probe_store_capability,
    probe_verification_capability,
    reject_ivp_commitment_as_sealer,
    sealer_capability_from_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/adapters.py"
)
ADAPTERS_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.semantic_governor.adapters"
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


def _ok_datasets_surface(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "SEMANTIC_GOVERNOR_API_SCHEMA": EXPECTED_DATASETS_API_SCHEMA,
        "SEMANTIC_GOVERNOR_PACKAGE_INTERFACE": EXPECTED_DATASETS_PACKAGE_INTERFACE,
        "REQUIRED_PUBLIC_APIS": REQUIRED_DATASETS_APIS,
    }
    for name in REQUIRED_DATASETS_APIS:
        fields[name] = lambda *a, _n=name, **k: {"api": _n}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_harness_surface(**overrides: Any) -> SimpleNamespace:
    class FakeHarness:
        pass

    fields: dict[str, Any] = {
        "SemanticCompressionHarness": FakeHarness,
        "run_semantic_patch_loop": lambda request: {"ok": True, "request": request},
        "HARNESS_LOOP_INTERFACE": EXPECTED_HARNESS_INTERFACE,
        "HARNESS_LOOP_SCHEMA": EXPECTED_HARNESS_SCHEMA,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_verification_surface(**overrides: Any) -> SimpleNamespace:
    class VerificationCommitment:
        pass

    class VerificationBundle:
        pass

    class VerificationPlan:
        pass

    class IncrementalVerificationPlanner:
        pass

    class VerificationReceiptCache:
        pass

    contracts = SimpleNamespace(
        VERIFICATION_COMMITMENT_INTERFACE=EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        VERIFICATION_COMMITMENT_SCHEMA=EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    )
    fields: dict[str, Any] = {
        "create_verification_plan": lambda *a, **k: VerificationPlan(),
        "IncrementalVerificationPlanner": IncrementalVerificationPlanner,
        "build_verification_commitment": lambda *a, **k: VerificationCommitment(),
        "VerificationCommitment": VerificationCommitment,
        "VerificationBundle": VerificationBundle,
        "VerificationPlan": VerificationPlan,
        "VerificationReceiptCache": VerificationReceiptCache,
        "choose_model_route": lambda *a, **k: {"route": "deterministic_only"},
        "contracts": contracts,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ok_store_surface(**overrides: Any) -> SimpleNamespace:
    class DurableSemanticGovernorStore:
        def put_artifact(self, *args: Any, **kwargs: Any) -> Any:
            return {"put": True}

        def get_verified_artifact(self, *args: Any, **kwargs: Any) -> Any:
            return {"get": True}

    fields: dict[str, Any] = {
        "SEMANTIC_GOVERNOR_STORE_INTERFACE": EXPECTED_STORE_INTERFACE,
        "SEMANTIC_GOVERNOR_STORE_SCHEMA": EXPECTED_STORE_SCHEMA,
        "SemanticGovernorStore": object,
        "ARTIFACT_MODULE_INTERFACE": EXPECTED_STORE_ARTIFACT_INTERFACE,
        "DurableSemanticGovernorStore": DurableSemanticGovernorStore,
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
        "INCREMENTAL_PROOF_SEALER_INTERFACE": "IncrementalProofSealer@1",
        "IS_ZK_SEALER": True,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


# ---------------------------------------------------------------------------
# Constants / structural
# ---------------------------------------------------------------------------


def test_evidence_and_interface_constants() -> None:
    assert SCG_RUNTIME_ADAPTERS_EVIDENCE == "scg/runtime-adapters@1"
    assert DATASETS_ADAPTER_ID.endswith("@1")
    assert HARNESS_ADAPTER_ID.endswith("@1")
    assert VERIFICATION_ADAPTER_ID.endswith("@1")
    assert STORE_ADAPTER_ID.endswith("@1")
    assert SEALER_CAPABILITY_ID.endswith("@1")
    assert EXPECTED_DATASETS_API_SCHEMA.endswith("@1")
    assert EXPECTED_HARNESS_INTERFACE == "SemanticCompressionHarness@1"
    assert EXPECTED_STORE_INTERFACE == "SemanticGovernorStore@1"
    assert EXPECTED_VERIFICATION_COMMITMENT_INTERFACE == "VerificationCommitment@1"
    assert "evaluate_context_sufficiency" in REQUIRED_DATASETS_APIS


def test_adapter_module_exports_required_interfaces() -> None:
    for name in (
        "GovernorDatasetsAdapter",
        "GovernorHarnessAdapter",
        "GovernorVerificationAdapter",
        "GovernorStoreAdapter",
        "IncrementalSealerCapability",
        "load_runtime_adapters",
        "probe_incremental_sealer_capability",
        "reject_ivp_commitment_as_sealer",
    ):
        assert hasattr(adapters_mod, name), name


def test_adapter_source_has_no_module_level_io() -> None:
    """Static guard: module body must not call open/Path I/O at import."""

    source = ADAPTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ADAPTER_PATH))
    forbidden_calls = {"open", "urlopen", "Popen", "run", "system"}
    for node in tree.body:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_calls:
                # Allow only inside function/class bodies (lazy probes).
                # Module-level statements are direct children of Module.
                parents = []
                # Re-walk: only fail if the call's containing statement is
                # a top-level Assign/Expr/etc that is not a function/class.
                pass
    # Stronger check: top-level call expressions only.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in forbidden_calls:
                    pytest.fail(
                        f"module-level I/O call forbidden at import: {name}"
                    )


# ---------------------------------------------------------------------------
# Import performs no I/O
# ---------------------------------------------------------------------------


def _hermetic_import_probe() -> subprocess.CompletedProcess[str]:
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

# Drop already-loaded target so import is fresh.
prefix = {ADAPTERS_MODULE!r}
for name in list(sys.modules):
    if name == prefix or name.startswith(prefix + "."):
        del sys.modules[name]

import importlib
mod = importlib.import_module(prefix)

# Touch module-local constants only (no probe that loads upstream packages).
assert mod.SCG_RUNTIME_ADAPTERS_EVIDENCE == "scg/runtime-adapters@1"
assert mod.DATASETS_ADAPTER_INTERFACE.endswith("@1")
assert mod.SEALER_CAPABILITY_INTERFACE.endswith("@1")

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


def test_import_performs_no_io() -> None:
    result = _hermetic_import_probe()
    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert '"ok": true' in result.stdout.lower() or '"ok": true' in result.stdout


# ---------------------------------------------------------------------------
# Happy-path probes with injected surfaces
# ---------------------------------------------------------------------------


def test_datasets_adapter_runtime_view() -> None:
    surface = _ok_datasets_surface()
    adapter = load_datasets_adapter(surface)
    view = adapter.runtime_view()
    assert view["adapter_id"] == DATASETS_ADAPTER_ID
    assert adapter.capability.available is True
    assert adapter.evaluate_context_sufficiency()["api"] == "evaluate_context_sufficiency"
    assert "diagnose_omission" in adapter.capability.operations


def test_harness_adapter_runtime_view() -> None:
    surface = _ok_harness_surface()
    adapter = load_harness_adapter(surface)
    view = adapter.runtime_view()
    assert view["harness_interface"] == EXPECTED_HARNESS_INTERFACE
    assert adapter.harness_class.__name__ == "FakeHarness"
    assert adapter.run_semantic_patch_loop({"task": "t"})["ok"] is True


def test_verification_adapter_marks_commitment_non_sealer() -> None:
    surface = _ok_verification_surface()
    adapter = load_verification_adapter(surface)
    view = adapter.runtime_view()
    assert view["commitment_is_proof_sealer"] is False
    assert view["commitment_is_zk"] is False
    assert view["can_satisfy_sealer_capability"] is False
    assert adapter.commitment_is_proof_sealer() is False
    assert adapter.commitment_is_zk() is False
    commitment = adapter.build_verification_commitment()
    assert type(commitment).__name__ == "VerificationCommitment"


def test_store_adapter_runtime_view() -> None:
    surface = _ok_store_surface()
    adapter = load_store_adapter(surface=surface)
    view = adapter.runtime_view()
    assert view["store_interface"] == EXPECTED_STORE_INTERFACE
    store = adapter.store_class()
    assert store.put_artifact()["put"] is True
    assert store.get_verified_artifact()["get"] is True


# ---------------------------------------------------------------------------
# Fail-closed: missing / stale / incompatible
# ---------------------------------------------------------------------------


def test_missing_datasets_exports_fail_closed() -> None:
    surface = _ok_datasets_surface()
    delattr(surface, "evaluate_context_sufficiency")
    cap = probe_datasets_capability(surface=surface)
    assert cap.available is False
    assert cap.status == CapabilityStatus.MISSING.value
    assert cap.reason_code == "missing_exports"
    with pytest.raises(GovernorCapabilityUnavailable) as excinfo:
        load_datasets_adapter(surface)
    assert excinfo.value.reason_code == "missing_exports"


def test_incompatible_datasets_schema_fails_closed() -> None:
    surface = _ok_datasets_surface(
        SEMANTIC_GOVERNOR_API_SCHEMA="ipfs-datasets.other-api@9"
    )
    cap = probe_datasets_capability(surface=surface)
    assert cap.available is False
    assert cap.status == CapabilityStatus.INCOMPATIBLE.value
    assert cap.reason_code == "incompatible_capability"
    with pytest.raises(GovernorCapabilityUnavailable):
        GovernorDatasetsAdapter(surface=surface).require_available()


def test_stale_harness_schema_fails_closed() -> None:
    # Same stem with older major → stale.
    surface = _ok_harness_surface(
        HARNESS_LOOP_SCHEMA="ipfs-accelerate.semantic-compression-harness@0"
    )
    cap = probe_harness_capability(surface=surface)
    assert cap.available is False
    assert cap.status == CapabilityStatus.STALE.value
    assert cap.reason_code == "stale_capability"
    with pytest.raises(GovernorCapabilityUnavailable) as excinfo:
        load_harness_adapter(surface)
    assert excinfo.value.status == CapabilityStatus.STALE.value


def test_incompatible_store_interface_fails_closed() -> None:
    surface = _ok_store_surface(
        SEMANTIC_GOVERNOR_STORE_INTERFACE="OtherStore@1"
    )
    cap = probe_store_capability(surface=surface)
    assert cap.available is False
    assert cap.status == CapabilityStatus.INCOMPATIBLE.value
    with pytest.raises(GovernorCapabilityUnavailable):
        load_store_adapter(surface=surface)


def test_missing_store_methods_fail_closed() -> None:
    class IncompleteStore:
        def put_artifact(self, *a: Any, **k: Any) -> Any:
            return None

    surface = _ok_store_surface(DurableSemanticGovernorStore=IncompleteStore)
    cap = probe_store_capability(surface=surface)
    assert cap.available is False
    assert cap.reason_code == "missing_exports"


def test_unavailable_capability_require_raises() -> None:
    cap = SurfaceCapability(
        available=False,
        adapter_id=DATASETS_ADAPTER_ID,
        interface_id="GovernorDatasetsAdapter@1",
        schema=EXPECTED_DATASETS_API_SCHEMA,
        operations=(),
        fingerprints={},
        status=CapabilityStatus.UNAVAILABLE.value,
        reason_code="capability_unavailable",
        diagnostic="down",
    )
    with pytest.raises(GovernorCapabilityUnavailable) as excinfo:
        cap.require_available("use")
    assert excinfo.value.to_mapping()["available"] is False


def test_import_failed_probe_is_missing_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(name: str, *, adapter_id: str) -> Any:
        raise GovernorCapabilityUnavailable(
            "load",
            "import_failed",
            f"import of {name!r} failed: no module",
            adapter_id=adapter_id,
            retryable=True,
            status=CapabilityStatus.MISSING.value,
        )

    monkeypatch.setattr(adapters_mod, "_import_module", boom)
    cap = probe_datasets_capability()
    assert cap.available is False
    assert cap.reason_code == "import_failed"
    assert cap.retryable is True


# ---------------------------------------------------------------------------
# IVP commitment cannot satisfy sealer capability
# ---------------------------------------------------------------------------


def test_sealer_unavailable_by_default_on_empty_candidates() -> None:
    cap = probe_incremental_sealer_capability(candidate_modules=())
    assert cap.available is False
    assert cap.seal_status == SealStatus.UNAVAILABLE.value
    assert cap.can_be_satisfied_by_ivp_commitment is False
    assert cap.is_zk is False
    assert cap.is_full_or_delta_seal is False
    with pytest.raises(GovernorCapabilityUnavailable):
        cap.require_available("seal")


def test_ivp_commitment_class_cannot_satisfy_sealer() -> None:
    class VerificationCommitment:
        pass

    cap = sealer_capability_from_evidence(VerificationCommitment)
    assert cap.available is False
    assert cap.reason_code == "ivp_commitment_not_sealer"
    assert cap.can_be_satisfied_by_ivp_commitment is False
    assert cap.status == CapabilityStatus.INCOMPATIBLE.value


def test_ivp_commitment_instance_cannot_satisfy_sealer() -> None:
    class VerificationCommitment:
        pass

    cap = sealer_capability_from_evidence(VerificationCommitment())
    assert cap.available is False
    assert cap.reason_code == "ivp_commitment_not_sealer"


def test_ivp_commitment_mapping_cannot_satisfy_sealer() -> None:
    cap = sealer_capability_from_evidence(
        {
            "schema": EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            "interface_id": EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        }
    )
    assert cap.available is False
    assert cap.reason_code == "ivp_commitment_not_sealer"


def test_ivp_commitment_symbol_name_cannot_satisfy_sealer() -> None:
    for evidence in (
        "VerificationCommitment",
        "build_verification_commitment",
        EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
    ):
        cap = sealer_capability_from_evidence(evidence)
        assert cap.available is False, evidence
        assert cap.reason_code == "ivp_commitment_not_sealer", evidence


def test_reject_ivp_commitment_as_sealer_raises() -> None:
    with pytest.raises(GovernorCapabilityUnavailable) as excinfo:
        reject_ivp_commitment_as_sealer("build_verification_commitment")
    assert excinfo.value.reason_code == "ivp_commitment_not_sealer"
    assert excinfo.value.adapter_id == SEALER_CAPABILITY_ID


def test_verification_adapter_commitment_cannot_become_sealer() -> None:
    surface = _ok_verification_surface()
    adapter = load_verification_adapter(surface)
    commitment = adapter.build_verification_commitment()
    cap = sealer_capability_from_evidence(commitment)
    assert cap.available is False
    assert cap.reason_code == "ivp_commitment_not_sealer"
    # probe with commitment surface also fails closed
    probe_cap = probe_incremental_sealer_capability(surface=commitment)
    assert probe_cap.available is False
    assert probe_cap.reason_code == "ivp_commitment_not_sealer"


def test_sealer_capability_forges_cannot_enable_ivp_substitution() -> None:
    cap = IncrementalSealerCapability(
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
    cap = probe_incremental_sealer_capability(surface=surface)
    assert cap.available is True
    assert cap.seal_status == SealStatus.AVAILABLE.value
    assert cap.is_full_or_delta_seal is True
    assert cap.is_zk is True
    assert "IncrementalProofSealer" in cap.operations
    assert cap.can_be_satisfied_by_ivp_commitment is False


def test_stale_sealer_interface_fails_closed() -> None:
    surface = _released_sealer_surface(
        INCREMENTAL_PROOF_SEALER_INTERFACE="IncrementalProofSealer@0"
    )
    cap = probe_incremental_sealer_capability(surface=surface)
    assert cap.available is False
    assert cap.status == CapabilityStatus.STALE.value
    assert cap.reason_code == "stale_capability"


def test_empty_sealer_module_is_missing() -> None:
    surface = SimpleNamespace(__name__="fake.empty.sealer")
    cap = probe_incremental_sealer_capability(surface=surface)
    assert cap.available is False
    assert cap.reason_code == "missing_exports"


# ---------------------------------------------------------------------------
# Live surfaces (when nested packages are present)
# ---------------------------------------------------------------------------


def test_live_datasets_probe_when_available() -> None:
    try:
        import ipfs_datasets_py.logic.software_contracts.semantic_governor as sg  # noqa: F401
    except Exception:
        pytest.skip("datasets semantic_governor surface unavailable")
    cap = probe_datasets_capability()
    assert cap.available is True
    adapter = load_datasets_adapter()
    assert adapter.capability.available is True
    assert callable(adapter.api("evaluate_context_sufficiency"))


def test_live_harness_probe_when_available() -> None:
    try:
        import ipfs_accelerate_py.agent_supervisor.semantic_state.harness as h  # noqa: F401
    except Exception:
        pytest.skip("harness surface unavailable")
    cap = probe_harness_capability()
    assert cap.available is True
    adapter = load_harness_adapter()
    assert adapter.capability.fingerprints["interface"] == EXPECTED_HARNESS_INTERFACE


def test_live_verification_probe_when_available() -> None:
    try:
        import ipfs_accelerate_py.agent_supervisor.verification as v  # noqa: F401
    except Exception:
        pytest.skip("verification surface unavailable")
    cap = probe_verification_capability()
    assert cap.available is True
    assert cap.fingerprints.get("is_proof_sealer") == "false"
    adapter = load_verification_adapter()
    assert adapter.runtime_view()["can_satisfy_sealer_capability"] is False


def test_live_store_probe_when_available() -> None:
    try:
        import ipfs_kit_py.semantic_governor_store.artifacts as art  # noqa: F401
        import ipfs_kit_py.semantic_governor_store.contracts as contracts  # noqa: F401
    except Exception:
        pytest.skip("kit governor store surface unavailable")
    cap = probe_store_capability()
    assert cap.available is True
    adapter = load_store_adapter()
    assert adapter.store_class.__name__ == "DurableSemanticGovernorStore"


def test_live_sealer_is_typed_unavailable_without_released_api() -> None:
    # On the current SCG tree the released sealer is not present.
    cap = probe_incremental_sealer_capability(
        candidate_modules=(
            "ipfs_accelerate_py.agent_supervisor.proof_sealer",
            "ipfs_accelerate_py.agent_supervisor.incremental_proof_sealer",
            "ipfs_kit_py.proof_sealer",
            "ipfs_kit_py.incremental_proof_sealer",
        )
    )
    assert cap.available is False
    assert cap.seal_status == SealStatus.UNAVAILABLE.value
    assert cap.can_be_satisfied_by_ivp_commitment is False


def test_load_runtime_adapters_with_injected_surfaces() -> None:
    runtime = load_runtime_adapters(
        datasets_surface=_ok_datasets_surface(),
        harness_surface=_ok_harness_surface(),
        verification_surface=_ok_verification_surface(),
        store_surface=_ok_store_surface(),
        sealer_surface=None,
        require_sealer=False,
    )
    runtime.require_execution_surfaces()
    assert runtime.sealer.available is False
    mapping = runtime.to_mapping()
    assert mapping["evidence_id"] == SCG_RUNTIME_ADAPTERS_EVIDENCE
    assert mapping["sealer"]["can_be_satisfied_by_ivp_commitment"] is False
    with pytest.raises(GovernorCapabilityUnavailable):
        runtime.require_sealer()


def test_load_runtime_adapters_require_sealer_fails_closed() -> None:
    with pytest.raises(GovernorCapabilityUnavailable) as excinfo:
        load_runtime_adapters(
            datasets_surface=_ok_datasets_surface(),
            harness_surface=_ok_harness_surface(),
            verification_surface=_ok_verification_surface(),
            store_surface=_ok_store_surface(),
            sealer_surface=SimpleNamespace(__name__="empty"),
            require_sealer=True,
        )
    assert excinfo.value.adapter_id == SEALER_CAPABILITY_ID


def test_load_runtime_adapters_with_released_sealer() -> None:
    runtime = load_runtime_adapters(
        datasets_surface=_ok_datasets_surface(),
        harness_surface=_ok_harness_surface(),
        verification_surface=_ok_verification_surface(),
        store_surface=_ok_store_surface(),
        sealer_surface=_released_sealer_surface(),
        require_sealer=True,
    )
    assert runtime.sealer.available is True
    assert runtime.require_sealer().is_full_or_delta_seal is True


def test_capability_status_and_seal_status_are_closed() -> None:
    assert {s.value for s in CapabilityStatus} == {
        "available",
        "unavailable",
        "incompatible",
        "stale",
        "missing",
    }
    assert {s.value for s in SealStatus} == {
        "available",
        "unavailable",
        "inconclusive",
    }
