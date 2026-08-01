"""Cold and fail-open tests for automatic proof-reuse service injection."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    ITEM_METADATA_ATTRIBUTE,
    LOOKUP_SERVICE_ATTRIBUTE,
    PROVIDER_SERVICE_ATTRIBUTE,
    SERVICE_RESOLUTION_ATTRIBUTE,
    STORE_SERVICE_ATTRIBUTE,
    pytest_collection_modifyitems,
    pytest_configure,
    set_proof_reuse_service_resolver,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DATASETS_VERIFIER_MODULE,
    LOOKUP_MODULE,
    MULTIFORMATS_MODULE,
    PROVIDER_MODULE,
    STORE_MODULE,
    AllowlistedPipInstaller,
    LazyProofReuseServiceResolver,
    ProofReuseDependency,
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


def test_missing_allowlisted_dependency_is_installed_only_once(
    tmp_path: Path,
) -> None:
    _Provider.constructions = 0
    _Provider.prove_calls = 0
    _Store.constructions = []
    importer = _Importer(
        _module_map(),
        missing=(MULTIFORMATS_MODULE,),
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
    assert first.installed_modules == (MULTIFORMATS_MODULE,)
    assert installer.calls == [MULTIFORMATS_MODULE]
    assert importer.calls.count(MULTIFORMATS_MODULE) == 2
    assert _Provider.constructions == 1
    assert _Provider.prove_calls == 0
    assert _Store.constructions == [tmp_path / "cache"]
    assert set(importer.calls) == {
        MULTIFORMATS_MODULE,
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


def test_pytest_configure_injects_services_but_missing_identity_still_runs(
    tmp_path: Path,
) -> None:
    _Provider.constructions = 0
    _Provider.prove_calls = 0
    _Store.constructions = []
    importer = _Importer(_module_map())
    resolver = LazyProofReuseServiceResolver(importer=importer)
    config = _Config(tmp_path)
    set_proof_reuse_service_resolver(config, resolver)

    pytest_configure(config)

    resolution = getattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    assert resolution.available is True
    assert getattr(config, LOOKUP_SERVICE_ATTRIBUTE) is resolution.lookup
    assert getattr(config, STORE_SERVICE_ATTRIBUTE) is resolution.store
    assert getattr(config, PROVIDER_SERVICE_ATTRIBUTE) is resolution.provider
    assert _Provider.prove_calls == 0

    item = _Item()
    pytest_collection_modifyitems(config, [item])

    assert hasattr(item, ITEM_METADATA_ATTRIBUTE)
    assert resolution.store.lookup_calls == 0
    assert item.markers == []
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

    resolution = getattr(config, SERVICE_RESOLUTION_ATTRIBUTE)
    assert resolution.available is False
    assert resolution.reason_code == "cid_provider_unavailable"
    assert not hasattr(config, LOOKUP_SERVICE_ATTRIBUTE)
    assert not hasattr(config, STORE_SERVICE_ATTRIBUTE)
    assert not hasattr(config, PROVIDER_SERVICE_ATTRIBUTE)

    item = _Item()
    pytest_collection_modifyitems(config, [item])
    assert item.markers == []
