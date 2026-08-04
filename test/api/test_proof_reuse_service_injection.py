"""Cold and fail-open tests for automatic proof-reuse service injection."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    ITEM_METADATA_ATTRIBUTE,
    LOOKUP_SERVICE_ATTRIBUTE,
    PROVIDER_SERVICE_ATTRIBUTE,
    SERVICE_RESOLUTION_ATTRIBUTE,
    SERVICE_RESOLVER_ATTRIBUTE,
    STORE_SERVICE_ATTRIBUTE,
    _inject_default_services,
    pytest_collection_modifyitems,
    pytest_configure,
    set_proof_reuse_service_resolver,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DATASETS_VERIFIER_DEPENDENCY,
    DATASETS_VERIFIER_DISTRIBUTION,
    DATASETS_VERIFIER_MODULE,
    DATASETS_VERIFIER_RELEASE_BLOCKER,
    DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED,
    DATASETS_VERIFIER_REVISION,
    JSONSCHEMA_DEPENDENCY,
    JSONSCHEMA_MODULE,
    LOOKUP_MODULE,
    MULTIFORMATS_MODULE,
    NLTK_MODULE,
    PROOF_REUSE_AUTO_INSTALL_ENV,
    PROOF_REUSE_DATASETS_SOURCE_ENV,
    PROVIDER_MODULE,
    STORE_MODULE,
    AllowlistedPipInstaller,
    LazyProofReuseServiceResolver,
    ProofReuseDependency,
    automatic_install_enabled,
    proof_reuse_dependency_plan,
)


class _Capabilities:
    prove_on_lookup = False


class _Provider:
    constructions = 0
    prove_calls = 0

    def __init__(self) -> None:
        type(self).constructions += 1

    def capabilities(self) -> _Capabilities:
        return _Capabilities()

    def prove(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).prove_calls += 1
        raise AssertionError("service resolution must never prove")


class _Store:
    constructions: list[Path] = []

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.lookup_calls = 0
        type(self).constructions.append(self.root)

    def lookup(self, _locator: Any, **_kwargs: Any) -> tuple[Any, ...]:
        self.lookup_calls += 1
        return ()

    def put_candidate(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def put_receipt(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _module_map() -> dict[str, Any]:
    return {
        MULTIFORMATS_MODULE: SimpleNamespace(
            CID=object(),
            multihash=object(),
        ),
        JSONSCHEMA_MODULE: SimpleNamespace(validators=object()),
        DATASETS_VERIFIER_MODULE: SimpleNamespace(
            verify_test_execution_certificate=lambda *_args, **_kwargs: False,
        ),
        STORE_MODULE: SimpleNamespace(TestCertificateStore=_Store),
        PROVIDER_MODULE: SimpleNamespace(
            IpfsDatasetsTestCertificateProvider=_Provider,
        ),
        LOOKUP_MODULE: importlib.import_module(LOOKUP_MODULE),
    }


class _Importer:
    def __init__(
        self,
        modules: dict[str, Any],
        *,
        missing: tuple[str, ...] = (),
    ) -> None:
        self.modules = modules
        self.missing = set(missing)
        self.calls: list[str] = []

    def __call__(self, module_name: str) -> Any:
        self.calls.append(module_name)
        if module_name in self.missing:
            raise ModuleNotFoundError(
                f"missing {module_name}",
                name=module_name,
            )
        return self.modules[module_name]


class _Installer:
    def __init__(self, importer: _Importer, *, succeeds: bool) -> None:
        self.importer = importer
        self.succeeds = succeeds
        self.calls: list[str] = []

    def install(self, dependency: Any) -> bool:
        self.calls.append(dependency.module_name)
        if self.succeeds:
            self.importer.missing.discard(dependency.module_name)
        return self.succeeds

    def prepare_dependency(self, _dependency: Any) -> bool:
        return True

    def validate_module_provenance(self, _dependency: Any, _module: Any) -> bool:
        # This injected test double is the explicit trust boundary for the
        # synthetic module map; production installers perform digest checks.
        return True


class _PluginManager:
    def __init__(self) -> None:
        self.registered: list[tuple[Any, str | None]] = []

    def register(self, plugin: Any, name: str | None = None) -> None:
        self.registered.append((plugin, name))


class _Config:
    def __init__(self, root: Path, *, mode: str = "read") -> None:
        self.rootpath = root
        self.pluginmanager = _PluginManager()
        self._mode = mode
        self.markers: list[str] = []

    def addinivalue_line(self, name: str, value: str) -> None:
        if name == "markers":
            self.markers.append(value)

    def getoption(self, name: str, default: Any = None) -> Any:
        values = {
            "proof_reuse_mode": self._mode,
            "proof_reuse_required_audit": False,
        }
        return values.get(name, default)

    def getini(self, name: str) -> Any:
        values = {
            "proof_reuse_mode": "",
            "proof_reuse_required_audit": False,
        }
        return values.get(name, "")


class _Item:
    nodeid = "test_direct.py::test_without_identity"

    def __init__(self) -> None:
        self.markers: list[Any] = []

    def get_closest_marker(self, _name: str) -> None:
        return None

    def iter_markers(self, _name: str):
        return iter(())

    def add_marker(self, marker: Any) -> None:
        self.markers.append(marker)


def test_off_mode_never_resolves_or_installs_services(tmp_path: Path) -> None:
    class _ForbiddenResolver:
        def resolve(self, **_kwargs: Any) -> None:
            raise AssertionError("off mode must not resolve services")

    config = _Config(tmp_path, mode="off")
    set_proof_reuse_service_resolver(config, _ForbiddenResolver())

    pytest_configure(config)

    assert not hasattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    assert not hasattr(config, LOOKUP_SERVICE_ATTRIBUTE)


def test_service_module_load_has_no_install_network_or_cache_side_effect(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "must-not-exist"
    probe = tmp_path / "cold_services.py"
    services_file = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "testing"
        / "proof_reuse"
        / "services.py"
    )
    probe.write_text(
        f"""
import builtins
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

blocked = ("ipfs_datasets_py", "ipfs_kit_py", "multiformats", "pytest")
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name in blocked or name.startswith(tuple(part + "." for part in blocked)):
        raise AssertionError("optional import attempted: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

def forbidden_process(*args, **kwargs):
    raise AssertionError("process attempted during import")
subprocess.run = forbidden_process
subprocess.Popen = forbidden_process

class NoNetworkSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise AssertionError("network attempted during import")
    def connect_ex(self, *args, **kwargs):
        raise AssertionError("network attempted during import")
socket.socket = NoNetworkSocket

spec = importlib.util.spec_from_file_location(
    "_cold_proof_reuse_services",
    {str(services_file)!r},
)
assert spec is not None and spec.loader is not None
services = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = services
spec.loader.exec_module(services)
assert services.PROOF_REUSE_AUTO_INSTALL_ENV
assert not Path({str(cache_root)!r}).exists()
print(json.dumps(sorted(name for name in sys.modules if name.startswith(blocked))))
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = "readwrite"
    environment["IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"] = "1"
    environment["IPFS_TEST_PROOF_REUSE_CACHE_DIR"] = str(cache_root)

    completed = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == []
    assert not cache_root.exists()


@pytest.mark.parametrize("missing_module", (MULTIFORMATS_MODULE, JSONSCHEMA_MODULE))
def test_missing_allowlisted_dependency_is_installed_only_once(
    tmp_path: Path,
    missing_module: str,
) -> None:
    _Provider.constructions = 0
    _Provider.prove_calls = 0
    _Store.constructions = []
    importer = _Importer(
        _module_map(),
        missing=(missing_module,),
    )
    installer = _Installer(importer, succeeds=True)
    resolver = LazyProofReuseServiceResolver(
        importer=importer,
        installer=installer,
    )

    first = resolver.resolve(cache_root=tmp_path / "cache")
    second = resolver.resolve(cache_root=tmp_path / "other-cache")

    assert first is second
    assert first.available is True
    assert first.installed_modules == (missing_module,)
    assert installer.calls == [missing_module]
    assert importer.calls.count(missing_module) == 2
    assert _Provider.constructions == 1
    assert _Provider.prove_calls == 0
    assert _Store.constructions == [tmp_path / "cache"]
    assert set(importer.calls) == {
        MULTIFORMATS_MODULE,
        JSONSCHEMA_MODULE,
        DATASETS_VERIFIER_MODULE,
        STORE_MODULE,
        PROVIDER_MODULE,
        LOOKUP_MODULE,
    }


def test_pip_installer_rejects_non_allowlisted_package_without_process() -> None:
    process_calls: list[Any] = []
    installer = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: process_calls.append((args, kwargs))
    )
    unknown = ProofReuseDependency(
        module_name="untrusted.dynamic.module",
        distribution="untrusted-package",
    )

    assert installer.install(unknown) is False
    assert process_calls == []


def test_auto_install_defaults_on_with_explicit_fail_closed_opt_out() -> None:
    assert automatic_install_enabled({}) is True
    assert automatic_install_enabled({PROOF_REUSE_AUTO_INSTALL_ENV: "1"}) is True
    assert automatic_install_enabled({PROOF_REUSE_AUTO_INSTALL_ENV: "off"}) is False
    assert automatic_install_enabled({PROOF_REUSE_AUTO_INSTALL_ENV: "invalid"}) is False


def test_dependency_plan_is_cold_complete_and_source_introspectable() -> None:
    pinned = proof_reuse_dependency_plan({})
    local = proof_reuse_dependency_plan(
        {PROOF_REUSE_DATASETS_SOURCE_ENV: "/reviewed/local/datasets"}
    )

    assert pinned["interface"] == "ProofReuseDependencyPlan@1"
    assert pinned["lazy"] is True
    assert pinned["cold_import_inert"] is True
    assert pinned["fail_open_to_run"] is True
    assert pinned["datasets_requested_source"] == "reviewed_integration_sibling"
    assert local["datasets_requested_source"] == "configured_local_path"
    assert pinned["datasets_reviewed_revision"] == DATASETS_VERIFIER_REVISION
    assert (
        pinned["remote_source_published"] is DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED
    )
    assert pinned["remote_source_published"] is False
    assert pinned["release_blocker"] == DATASETS_VERIFIER_RELEASE_BLOCKER
    assert [item["module_name"] for item in pinned["dependencies"]] == [
        MULTIFORMATS_MODULE,
        JSONSCHEMA_MODULE,
        NLTK_MODULE,
        DATASETS_VERIFIER_MODULE,
    ]
    assert pinned["dependencies"][1]["distribution"] == "jsonschema>=4,<5"
    assert pinned["dependencies"][1]["required_symbols"] == ["validators"]
    assert pinned["dependencies"][2]["distribution"] == "nltk>=3.8.1,<4"
    datasets = pinned["dependencies"][3]
    assert datasets["distribution"] == DATASETS_VERIFIER_DISTRIBUTION
    assert datasets["packaging_source"] == "private_exact_git_blob_snapshot_cas"
    assert datasets["pip_options"] == []
    assert pinned["datasets_vcs_install"] is False
    assert pinned["datasets_submodules_initialized"] is False
    assert pinned["datasets_global_site_packages_mutated"] is False
    assert pinned["datasets_private_target_install"] is True
    assert pinned["external_capability_absence_action"] == "RUN_OR_DEFERRED"
    activation = pinned["runtime_activation"]
    assert activation["automatic_plugin_discovery"] is True
    assert activation["ordinary_enabled_run_effective_action"] == "run"
    assert activation["default_identity_service_factory_configured"] is False
    assert activation["two_stage_candidate_revalidation_configured"] is False
    assert activation["post_pass_receipt_requires_runtime_trace"] is False
    assert activation["deferred_request_transport_compatible"] is False
    assert activation["issuer_in_lazy_service_resolution"] is False
    assert activation["authoritative_candidate_publication_configured"] is False
    assert activation["receipt_content_identity_profiles"] == {
        "accelerator": "cidv1-base32-dag-json-sha2-256",
        "datasets_statement": "sha256-canonical-json-v1",
        "exact_conformance": False,
    }
    assert activation["ordinary_warm_skip_path_complete"] is False
    assert activation["completion_authority"] is False
    assert len(activation["activation_blocker_codes"]) == 9
    assert activation["required_identity_providers"] == [
        "repository_forest_provider",
        "analysis_index_provider",
        "component_inputs_provider",
        "policy_inputs_provider",
        "runtime_evidence_provider",
    ]
    assert activation["required_implementation_sequence"][-1] == {
        "goals": ["PTR-G110"],
        "work": "activated_warm_benchmark_and_rollout_evidence",
    }


def test_datasets_lazy_install_uses_reviewed_sibling_without_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        process_calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    installer = AllowlistedPipInstaller(
        runner=runner,
        environ={"HOME": str(tmp_path)},
        provision_root=tmp_path / "provision",
    )
    monkeypatch.setattr(installer, "_activate_private_snapshot", lambda _root: True)

    assert installer.install(DATASETS_VERIFIER_DEPENDENCY) is True
    assert process_calls == []
    target = installer._validated_private_snapshot_target()
    assert target is not None
    receipt = json.loads(
        (target / ".proof-reuse-verifier-snapshot.json").read_text(encoding="utf-8")
    )
    assert receipt["datasets_revision"] == DATASETS_VERIFIER_REVISION
    assert receipt["pip_install"] is False
    assert receipt["vcs_install"] is False
    assert receipt["submodules_initialized"] is False


def test_datasets_lazy_install_prefers_valid_configured_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve().parents[2].parent / "ipfs_datasets"
    process_calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], **_kwargs: Any) -> Any:
        process_calls.append(command)
        return SimpleNamespace(returncode=0)

    installer = AllowlistedPipInstaller(
        runner=runner,
        environ={
            PROOF_REUSE_DATASETS_SOURCE_ENV: str(source),
            "HOME": str(tmp_path),
        },
        provision_root=tmp_path / "provision",
    )
    monkeypatch.setattr(installer, "_activate_private_snapshot", lambda _root: True)
    equal_dependency = replace(DATASETS_VERIFIER_DEPENDENCY)
    assert equal_dependency == DATASETS_VERIFIER_DEPENDENCY
    assert equal_dependency is not DATASETS_VERIFIER_DEPENDENCY

    assert installer.install(equal_dependency) is True
    assert process_calls == []
    assert installer.validated_local_datasets_source() == source.resolve()
    assert installer._validated_private_snapshot_target() is not None


def test_unversioned_configured_datasets_source_runs_no_process(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ipfs_datasets"
    verifier = (
        source / "ipfs_datasets_py" / "logic" / "zkp" / "test_execution_certificate.py"
    )
    verifier.parent.mkdir(parents=True)
    reviewed_verifier = (
        Path(__file__).resolve().parents[2].parent
        / "ipfs_datasets"
        / "ipfs_datasets_py"
        / "logic"
        / "zkp"
        / "test_execution_certificate.py"
    )
    verifier.write_bytes(reviewed_verifier.read_bytes())
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    process_calls: list[Any] = []
    installer = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: process_calls.append((args, kwargs)),
        environ={PROOF_REUSE_DATASETS_SOURCE_ENV: str(source)},
    )

    assert installer.install(DATASETS_VERIFIER_DEPENDENCY) is False
    assert process_calls == []


def test_invalid_configured_datasets_source_runs_no_process(
    tmp_path: Path,
) -> None:
    process_calls: list[Any] = []
    installer = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: process_calls.append((args, kwargs)),
        environ={PROOF_REUSE_DATASETS_SOURCE_ENV: str(tmp_path / "missing")},
    )

    assert installer.install(DATASETS_VERIFIER_DEPENDENCY) is False
    assert installer.install(DATASETS_VERIFIER_DEPENDENCY) is False
    assert process_calls == []


def test_tampered_configured_datasets_verifier_runs_no_process(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ipfs_datasets"
    verifier = (
        source / "ipfs_datasets_py" / "logic" / "zkp" / "test_execution_certificate.py"
    )
    verifier.parent.mkdir(parents=True)
    verifier.write_text("# unreviewed verifier\n", encoding="utf-8")
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    process_calls: list[Any] = []
    installer = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: process_calls.append((args, kwargs)),
        environ={PROOF_REUSE_DATASETS_SOURCE_ENV: str(source)},
    )

    assert installer.install(DATASETS_VERIFIER_DEPENDENCY) is False
    assert process_calls == []


def test_failed_pip_install_is_memoized_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_calls: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> Any:
        process_calls.append((args, kwargs))
        return SimpleNamespace(returncode=1)

    # Use an empty private provision root and disable cache activation so the
    # installer always reaches the injected runner once.
    installer = AllowlistedPipInstaller(
        runner=runner,
        environ={},
        provision_root=tmp_path / "provision",
    )
    monkeypatch.setattr(installer, "activate_cached_dependency", lambda _dep: False)

    assert installer.install(JSONSCHEMA_DEPENDENCY) is False
    assert installer.install(JSONSCHEMA_DEPENDENCY) is False
    assert len(process_calls) == 1


@pytest.mark.parametrize("policy", ("0", "invalid"))
def test_disabled_or_invalid_policy_constructs_no_installer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: str,
) -> None:
    from ipfs_accelerate_py.testing.proof_reuse import lazy_dependencies

    config = _Config(tmp_path)
    monkeypatch.setenv(PROOF_REUSE_AUTO_INSTALL_ENV, policy)

    def forbidden_installer() -> None:
        raise AssertionError("disabled policy must not construct an installer")

    monkeypatch.setattr(
        lazy_dependencies,
        "ProofReuseLazyDependencyInstaller",
        forbidden_installer,
    )

    _inject_default_services(config)

    assert hasattr(config, SERVICE_RESOLUTION_ATTRIBUTE)


def test_xdist_worker_never_constructs_default_installer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.testing.proof_reuse import lazy_dependencies

    config = _Config(tmp_path)
    config.workerinput = {"workerid": "gw0"}
    monkeypatch.delenv(PROOF_REUSE_AUTO_INSTALL_ENV, raising=False)

    def forbidden_installer() -> None:
        raise AssertionError("xdist workers must not construct installers")

    monkeypatch.setattr(
        lazy_dependencies,
        "ProofReuseLazyDependencyInstaller",
        forbidden_installer,
    )

    _inject_default_services(config)

    assert hasattr(config, SERVICE_RESOLUTION_ATTRIBUTE)


@pytest.mark.parametrize(
    ("package_policy", "malformed_lock"),
    (("0", False), ("1", True)),
)
def test_plugin_default_installer_honors_package_policy_and_strict_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package_policy: str,
    malformed_lock: bool,
) -> None:
    from ipfs_accelerate_py.testing.proof_reuse import (
        lazy_dependencies,
        services,
    )

    process_calls: list[Any] = []
    created_installers: list[Any] = []
    lock_root = tmp_path / "locks"
    if malformed_lock:
        lock_root.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(PROOF_REUSE_AUTO_INSTALL_ENV, "1")
    monkeypatch.setenv("IPFS_ACCEL_AUTO_INSTALL", package_policy)

    def missing_importer(module_name: str) -> Any:
        raise ModuleNotFoundError(f"missing {module_name}", name=module_name)

    real_installer_class = lazy_dependencies.ProofReuseLazyDependencyInstaller
    real_resolver_class = services.LazyProofReuseServiceResolver

    def installer_factory() -> Any:
        installer = real_installer_class(
            runner=lambda *args, **kwargs: process_calls.append((args, kwargs)),
            importer=missing_importer,
            environ=dict(os.environ),
            lock_root=lock_root,
            lock_timeout_seconds=0.1,
        )
        created_installers.append(installer)
        return installer

    monkeypatch.setattr(
        lazy_dependencies,
        "ProofReuseLazyDependencyInstaller",
        installer_factory,
    )
    monkeypatch.setattr(
        services,
        "LazyProofReuseServiceResolver",
        lambda *, installer: real_resolver_class(
            importer=missing_importer,
            installer=installer,
        ),
    )
    config = _Config(tmp_path)
    _inject_default_services(config)

    expected_installers = 1 if package_policy == "1" else 0
    assert len(created_installers) == expected_installers
    resolver = getattr(config, SERVICE_RESOLVER_ATTRIBUTE)
    assert (resolver._installer is not None) is bool(expected_installers)
    assert process_calls == []


def test_install_failure_leaves_services_unavailable_and_cache_untouched(
    tmp_path: Path,
) -> None:
    _Store.constructions = []
    cache_root = tmp_path / "unavailable-cache"
    importer = _Importer(
        _module_map(),
        missing=(DATASETS_VERIFIER_MODULE,),
    )
    installer = _Installer(importer, succeeds=False)
    resolver = LazyProofReuseServiceResolver(
        importer=importer,
        installer=installer,
    )

    first = resolver.resolve(cache_root=cache_root)
    second = resolver.resolve(cache_root=cache_root)

    assert first is second
    assert first.available is False
    assert first.reason_code == "certificate_provider_unavailable"
    assert installer.calls == [DATASETS_VERIFIER_MODULE]
    assert _Store.constructions == []
    assert not cache_root.exists()


def test_pytest_configure_defers_services_until_exact_identity_lookup(
    tmp_path: Path,
) -> None:
    _Provider.constructions = 0
    _Provider.prove_calls = 0
    _Store.constructions = []
    importer = _Importer(_module_map())
    installer = _Installer(importer, succeeds=True)
    resolver = LazyProofReuseServiceResolver(importer=importer, installer=installer)
    config = _Config(tmp_path)
    set_proof_reuse_service_resolver(config, resolver)

    pytest_configure(config)

    assert not hasattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    assert not hasattr(config, LOOKUP_SERVICE_ATTRIBUTE)
    assert importer.calls == []
    assert _Provider.constructions == 0
    assert _Store.constructions == []
    assert installer.calls == []

    # Ordinary collection has no exact execution identity, so optional
    # providers remain entirely untouched and the real test runs.
    item = _Item()
    pytest_collection_modifyitems(config, [item])
    assert hasattr(item, ITEM_METADATA_ATTRIBUTE)
    assert not hasattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    assert importer.calls == []
    assert item.markers == []

    # The first exact lookup identity is the lazy resolution boundary.
    lookup_item = _Item()
    lookup_item._ipfs_proof_reuse_locator = object()
    lookup_item._ipfs_proof_reuse_execution_key = object()
    pytest_collection_modifyitems(config, [lookup_item])

    resolution = getattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    assert resolution.available is True
    assert getattr(config, LOOKUP_SERVICE_ATTRIBUTE) is resolution.lookup
    assert getattr(config, STORE_SERVICE_ATTRIBUTE) is resolution.store
    assert getattr(config, PROVIDER_SERVICE_ATTRIBUTE) is resolution.provider
    assert _Provider.prove_calls == 0
    assert resolution.store.lookup_calls == 0
    assert lookup_item.markers == []
    assert _Provider.prove_calls == 0


def test_pytest_configure_failure_injects_no_partial_services(
    tmp_path: Path,
) -> None:
    importer = _Importer(
        _module_map(),
        missing=(MULTIFORMATS_MODULE,),
    )
    resolver = LazyProofReuseServiceResolver(importer=importer)
    config = _Config(tmp_path)
    set_proof_reuse_service_resolver(config, resolver)

    pytest_configure(config)

    assert not hasattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    item = _Item()
    item._ipfs_proof_reuse_locator = object()
    item._ipfs_proof_reuse_execution_key = object()
    pytest_collection_modifyitems(config, [item])

    resolution = getattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    assert resolution.available is False
    assert resolution.reason_code == "cid_provider_unavailable"
    assert not hasattr(config, LOOKUP_SERVICE_ATTRIBUTE)
    assert not hasattr(config, STORE_SERVICE_ATTRIBUTE)
    assert not hasattr(config, PROVIDER_SERVICE_ATTRIBUTE)
    assert item.markers == []
