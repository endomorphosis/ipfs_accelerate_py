"""Bounded NLTK/Groth16 first-use provisioning contracts."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import ipfs_accelerate_py.testing.proof_reuse.lazy_dependencies as lazy_module
import ipfs_accelerate_py.testing.proof_reuse.services as services_module
import pytest
from ipfs_accelerate_py.testing.proof_reuse.lazy_dependencies import (
    PACKAGE_AUTO_INSTALL_ENV,
    REASON_AVAILABLE,
    REASON_GROTH16_BUILD_DISABLED,
    REASON_GROTH16_SOURCE_INVALID,
    REASON_INCOMPATIBLE_VERSION,
    REASON_LOCK_TIMEOUT,
    REASON_NLTK_DOWNLOAD_DISABLED,
    REASON_NOT_ALLOWLISTED,
    AcceleratorProofReuseBootstrap,
    ProofReuseLazyDependencyInstaller,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DATASETS_GROTH16_BINARY_ENV,
    DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256,
    DATASETS_GROTH16_REVIEWED_FILES_SHA256,
    DATASETS_VERIFIER_REVISION,
    DATASETS_VERIFIER_SNAPSHOT_BYTES,
    DATASETS_VERIFIER_SNAPSHOT_FILES,
    DATASETS_VERIFIER_SNAPSHOT_SHA256,
    DEFAULT_NLTK_DATA_RESOURCES,
    NLTK_DATA_RESOURCE_ALLOWLIST,
    PROOF_REUSE_AUTO_INSTALL_ENV,
    PROOF_REUSE_GROTH16_BUILD_ENV,
    PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV,
    PROOF_REUSE_GROTH16_ENDPOINT_ENV,
    PROOF_REUSE_NLTK_DATA_DIR_ENV,
    PROOF_REUSE_PROVISION_DIR_ENV,
    AllowlistedPipInstaller,
    proof_reuse_dependency_plan,
)

ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = ACCELERATE_ROOT.parent / "ipfs_datasets"


def _reviewed_capability_payload() -> bytes:
    payload = {
        "schema_version": 1,
        "backend": "ipfs-datasets-groth16",
        "backend_version": "0.1.0",
        "implementation": "ark-groth16-bn254",
        "locked_source_identity_schema": "ipfs-datasets-groth16-locked-source-v1",
        "locked_source_identity": services_module.DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY,
        "proof_schema_versions": [1],
        "supported_circuits": [
            {
                "version": 1,
                "profile": "mvp-knowledge-of-axioms",
                "ruleset_id": "TDFOL_v1",
                "can_setup": True,
                "can_prove": True,
                "can_verify": True,
            },
            {
                "version": 2,
                "profile": "tdfol-v1-derivation",
                "ruleset_id": "TDFOL_v1",
                "can_setup": True,
                "can_prove": True,
                "can_verify": True,
            },
            {
                "version": 3,
                "profile": "mcp-event-dag-compaction",
                "ruleset_id": "MCP++_EventDAG_Compaction_v1",
                "can_setup": True,
                "can_prove": True,
                "can_verify": True,
            },
            {
                "version": 4,
                "profile": "test-pass-v2",
                "ruleset_id": "test_pass_v2",
                "can_setup": True,
                "can_prove": True,
                "can_verify": True,
            },
        ],
        "trusted_setup": {
            "automatic_during_build": False,
            "explicit_command_required": True,
            "deterministic_seed_is_test_only": True,
            "capabilities_reads_or_writes_artifacts": False,
        },
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    assert hashlib.sha256(encoded).hexdigest() == (
        services_module.DATASETS_GROTH16_TEST_PASS_CAPABILITY_PAYLOAD_SHA256
    )
    return encoded


class _FakeNltkData:
    def __init__(self) -> None:
        self.path: list[str] = []
        self.available: set[str] = set()

    def find(self, find_path: str, paths: list[str] | None = None) -> object:
        del paths
        if find_path not in self.available:
            raise LookupError(find_path)
        return object()


def _fake_nltk() -> SimpleNamespace:
    return SimpleNamespace(data=_FakeNltkData(), download=lambda *a, **k: True)


def _enabled_environment(tmp_path: Path) -> dict[str, str]:
    return {
        PROOF_REUSE_AUTO_INSTALL_ENV: "1",
        PACKAGE_AUTO_INSTALL_ENV: "1",
        PROOF_REUSE_NLTK_DATA_DIR_ENV: str(tmp_path / "nltk-data"),
        "HOME": str(tmp_path / "home"),
    }


def test_dependency_plan_keeps_capability_layers_distinct_and_pure() -> None:
    plan = proof_reuse_dependency_plan(
        {
            PROOF_REUSE_GROTH16_BUILD_ENV: "1",
            PROOF_REUSE_GROTH16_ENDPOINT_ENV: "https://prover.invalid/v1",
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "0",
        }
    )

    python_distributions = {
        item["distribution"] for item in plan["python_dependencies"]
    }
    assert "nltk>=3.8.1,<4" in python_distributions
    assert "jsonschema>=4,<5" in python_distributions
    assert not any("groth16" in item.lower() for item in python_distributions)
    assert plan["proof_reuse_auto_install_enabled"] is True
    assert plan["package_auto_install_enabled"] is False
    assert plan["effective_auto_install_enabled"] is False
    assert plan["automatic_install_enabled"] is False
    assert plan["nltk_data"]["provisioning_kind"] == "network_data_download"
    assert plan["nltk_data"]["download_on_import"] is False
    assert plan["groth16_native_backend"]["cargo_command"][2:4] == [
        "--locked",
        "--release",
    ]
    assert plan["groth16_native_backend"]["trusted_setup_during_build"] is False
    assert (
        plan["groth16_native_backend"]["build_receipt_directory_environment_variable"]
        == PROOF_REUSE_PROVISION_DIR_ENV
    )
    assert (
        plan["groth16_native_backend"]["previous_target_binary_without_receipt_trusted"]
        is False
    )
    runtime = plan["groth16_runtime_inputs"]
    assert runtime["endpoint"]["provisioning_kind"] == "operator_configuration"
    assert runtime["keys"]["auto_generate"] is False
    assert runtime["circuit"]["auto_select"] is False


def test_current_reviewed_datasets_revision_is_accepted_exactly() -> None:
    installer = AllowlistedPipInstaller(environ={})
    source = installer.validated_local_datasets_source()
    distribution = installer._selected_distribution(
        lazy_module.DATASETS_VERIFIER_DEPENDENCY
    )

    assert DATASETS_VERIFIER_REVISION == "1894e9dca7dced0690893d468e40751a14f0b15b"
    assert source == DATASETS_ROOT.resolve()
    assert installer._detached_git_head(source) == DATASETS_VERIFIER_REVISION
    assert distribution == services_module.DATASETS_VERIFIER_DISTRIBUTION
    blobs = installer.reviewed_datasets_verifier_snapshot_blobs()
    assert blobs is not None
    assert tuple(sorted(blobs)) == DATASETS_VERIFIER_SNAPSHOT_FILES
    assert services_module._datasets_snapshot_digest(blobs) == (
        DATASETS_VERIFIER_SNAPSHOT_SHA256,
        DATASETS_VERIFIER_SNAPSHOT_BYTES,
    )


def test_private_snapshot_installs_and_reuses_in_fresh_interpreters(
    tmp_path: Path,
) -> None:
    provision_root = tmp_path / "provision"
    program = """
import json
import os
import sys
from ipfs_accelerate_py.testing.proof_reuse.services import (
    AllowlistedPipInstaller,
    DATASETS_VERIFIER_DEPENDENCY,
)

def forbidden(*args, **kwargs):
    raise AssertionError(f"unexpected package process: {args!r} {kwargs!r}")

installer = AllowlistedPipInstaller(runner=forbidden)
if os.environ.get("PROOF_REUSE_TEST_FORBID_REVIEWED_SOURCE") == "1":
    installer.validated_local_datasets_source = forbidden
    installer._git_object_output = forbidden
installed = installer.install(DATASETS_VERIFIER_DEPENDENCY)
import ipfs_datasets_py as datasets_root
from ipfs_datasets_py import dependency_catalog
from ipfs_datasets_py.logic.zkp import test_execution_certificate as verifier
result = verifier.verify_test_execution_certificate({}, None)
print(json.dumps({
    "installed": installed,
    "module": verifier.__file__,
    "provenance": installer.validate_module_provenance(
        DATASETS_VERIFIER_DEPENDENCY, verifier
    ),
    "status": result.status.value,
    "reason": result.reason.value,
    "authority": result.authority.value,
    "root_version": datasets_root.__version__,
    "root_initialize": callable(datasets_root.initialize),
    "root_lazy_installer": type(datasets_root.installer).__name__,
    "coexisting_module": dependency_catalog.__file__,
    "heavy": sorted(
        name for name in sys.modules
        if name.split(".", 1)[0] in {
            "beartype", "nltk", "pytest", "torch", "transformers"
        }
    ),
}))
"""
    environment = dict(os.environ)
    environment.update(
        {
            PROOF_REUSE_PROVISION_DIR_ENV: str(provision_root),
            "PIP_CONFIG_FILE": str(tmp_path / "hostile-pip.conf"),
            "PIP_EXTRA_INDEX_URL": "https://attacker.invalid/simple",
            "PYTHONPATH": os.pathsep.join((str(ACCELERATE_ROOT), str(DATASETS_ROOT))),
        }
    )
    (tmp_path / "hostile-pip.conf").write_text(
        "[global]\nindex-url=https://attacker.invalid/simple\n",
        encoding="utf-8",
    )
    first = subprocess.run(
        (sys.executable, "-c", program),
        cwd=ACCELERATE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    first_result = json.loads(first.stdout)
    assert first_result == {
        "installed": True,
        "module": first_result["module"],
        "provenance": True,
        "status": "rejected",
        "reason": "malformed_certificate",
        "authority": "non_attested",
        "root_version": "0.2.0",
        "root_initialize": True,
        "root_lazy_installer": "_MinimalInstaller",
        "coexisting_module": first_result["coexisting_module"],
        "heavy": [],
    }
    assert str(provision_root) in first_result["module"]
    assert str(DATASETS_ROOT) in first_result["coexisting_module"]

    warm_environment = dict(environment)
    warm_environment["PROOF_REUSE_TEST_FORBID_REVIEWED_SOURCE"] = "1"
    second = subprocess.run(
        (sys.executable, "-c", program),
        cwd=ACCELERATE_ROOT,
        env=warm_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    second_result = json.loads(second.stdout)
    assert second_result["installed"] is True
    assert second_result["provenance"] is True
    assert second_result["module"] == first_result["module"]
    assert second_result["root_version"] == "0.2.0"
    assert second_result["root_initialize"] is True
    assert second_result["coexisting_module"] == first_result["coexisting_module"]
    assert second_result["heavy"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_dist_info",
        "extra_top_level_file",
        "extra_authority_directory",
        "tampered_file",
        "symlinked_file",
    ),
)
def test_private_snapshot_rejects_every_closed_set_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    process_calls: list[Any] = []
    provision_root = tmp_path / "provision"
    installer = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: process_calls.append((args, kwargs)),
        environ={PROOF_REUSE_PROVISION_DIR_ENV: str(provision_root)},
    )
    monkeypatch.setattr(installer, "_activate_private_snapshot", lambda _root: True)
    assert installer.install(lazy_module.DATASETS_VERIFIER_DEPENDENCY) is True
    target = installer._validated_private_snapshot_target()
    assert target is not None

    target.chmod(0o700)
    if mutation == "extra_dist_info":
        metadata = target / "attacker-1.0.dist-info"
        metadata.mkdir(mode=0o700)
        (metadata / "METADATA").write_text("Name: attacker\n", encoding="utf-8")
    elif mutation == "extra_top_level_file":
        (target / "attacker.pth").write_text("import attacker\n", encoding="utf-8")
    elif mutation == "extra_authority_directory":
        authority = target / "ipfs_datasets_py/logic/zkp"
        authority.chmod(0o700)
        (authority / "attacker_namespace").mkdir(mode=0o700)
    else:
        victim = target / "ipfs_datasets_py/logic/zkp/canonicalization.py"
        victim.parent.chmod(0o700)
        victim.chmod(0o600)
        if mutation == "tampered_file":
            victim.write_bytes(victim.read_bytes() + b"\n# attacker\n")
        else:
            victim.unlink()
            victim.symlink_to("circuits.py")

    assert installer._validated_private_snapshot_target() is None
    retry = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: process_calls.append((args, kwargs)),
        environ={PROOF_REUSE_PROVISION_DIR_ENV: str(provision_root)},
    )
    assert retry.install(lazy_module.DATASETS_VERIFIER_DEPENDENCY) is False
    assert process_calls == []


def test_private_snapshot_rejects_oversize_file_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = AllowlistedPipInstaller(
        runner=lambda *_args, **_kwargs: None,
        environ={PROOF_REUSE_PROVISION_DIR_ENV: str(tmp_path / "provision")},
    )
    monkeypatch.setattr(installer, "_activate_private_snapshot", lambda _root: True)
    assert installer.install(lazy_module.DATASETS_VERIFIER_DEPENDENCY) is True
    target = installer._validated_private_snapshot_target()
    assert target is not None
    victim = target / DATASETS_VERIFIER_SNAPSHOT_FILES[0]
    target.chmod(0o700)
    victim.parent.chmod(0o700)
    victim.chmod(0o600)
    with victim.open("r+b") as stream:
        stream.truncate(DATASETS_VERIFIER_SNAPSHOT_BYTES + 1)

    real_open = services_module.os.open
    victim_opened: list[Path] = []

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> int:
        if Path(path) == victim:
            victim_opened.append(victim)
            raise AssertionError("oversize snapshot file must not be opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(services_module.os, "open", guarded_open)

    assert services_module._datasets_snapshot_payloads(target) is None
    assert victim_opened == []


def test_private_snapshot_rejects_preloaded_in_memory_authority_module(
    tmp_path: Path,
) -> None:
    program = """
import json
import sys
from types import ModuleType
from ipfs_accelerate_py.testing.proof_reuse.services import (
    AllowlistedPipInstaller,
    DATASETS_VERIFIER_DEPENDENCY,
)

def forbidden(*args, **kwargs):
    raise AssertionError(f"unexpected package process: {args!r} {kwargs!r}")

installer = AllowlistedPipInstaller(runner=forbidden)
installed = installer.install(DATASETS_VERIFIER_DEPENDENCY)
evil_name = "ipfs_datasets_py.logic.zkp.backends"
evil = ModuleType(evil_name)
evil.get_backend = lambda *_args, **_kwargs: "attacker"
sys.modules[evil_name] = evil
preflight = installer.activate_cached_datasets_verifier()
sys.modules.pop(evil_name, None)
activated = installer.activate_cached_datasets_verifier()
from ipfs_datasets_py.logic.zkp import test_execution_certificate as verifier
sys.modules[evil_name] = evil
provenance = installer.validate_module_provenance(
    DATASETS_VERIFIER_DEPENDENCY, verifier
)
print(json.dumps({
    "installed": installed,
    "preflight": preflight,
    "activated": activated,
    "provenance": provenance,
}))
"""
    environment = dict(os.environ)
    environment.update(
        {
            PROOF_REUSE_PROVISION_DIR_ENV: str(tmp_path / "provision"),
            "PYTHONPATH": str(ACCELERATE_ROOT),
        }
    )
    completed = subprocess.run(
        (sys.executable, "-c", program),
        cwd=ACCELERATE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "installed": True,
        "preflight": False,
        "activated": True,
        "provenance": False,
    }


def test_generic_pip_install_uses_private_target_and_rejects_shadow_or_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = services_module.MULTIFORMATS_DEPENDENCY
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    monkeypatch.setattr(sys, "path", list(sys.path))
    for module_name in tuple(sys.modules):
        if module_name == "multiformats" or module_name.startswith("multiformats."):
            monkeypatch.delitem(sys.modules, module_name)

    def runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        target = Path(command[command.index("--target") + 1])
        package = target / "multiformats"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "# private target fixture\n", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    provision_root = tmp_path / "provision"
    installer = AllowlistedPipInstaller(
        runner=runner,
        environ={
            PROOF_REUSE_PROVISION_DIR_ENV: str(provision_root),
            "PIP_EXTRA_INDEX_URL": "https://attacker.invalid/simple",
            "PIP_CONFIG_FILE": str(tmp_path / "hostile-pip.conf"),
            "PYTHONPATH": "/attacker",
        },
    )

    assert installer.install(dependency) is True
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-1] == dependency.distribution
    assert "--isolated" in command
    assert "--target" in command
    assert "--no-compile" in command
    assert "PIP_EXTRA_INDEX_URL" not in kwargs["env"]
    assert kwargs["env"]["PIP_CONFIG_FILE"] == os.devnull
    assert "PYTHONPATH" not in kwargs["env"]
    target = installer._active_dependency_targets[dependency.module_name]
    loaded = SimpleNamespace(
        CID=object(),
        multihash=object(),
        __file__=str(target / "multiformats" / "__init__.py"),
    )
    assert installer.validate_module_provenance(dependency, loaded) is True

    shadow = SimpleNamespace(
        CID=object(),
        multihash=object(),
        __file__=str(tmp_path / "attacker" / "multiformats" / "__init__.py"),
    )
    resolver = services_module.LazyProofReuseServiceResolver(
        importer=lambda _name: shadow,
        installer=installer,
    )
    assert resolver._load_dependency(dependency) == (None, False)

    package_file = target / "multiformats" / "__init__.py"
    if os.name != "nt":
        target.chmod(0o700)
        package_file.parent.chmod(0o700)
        package_file.chmod(0o600)
    package_file.write_text("# tampered\n", encoding="utf-8")
    retry_calls: list[Any] = []
    retry = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: retry_calls.append((args, kwargs)),
        environ={PROOF_REUSE_PROVISION_DIR_ENV: str(provision_root)},
    )
    assert retry.activate_cached_dependency(dependency) is False
    assert retry.install(dependency) is False
    assert retry_calls == []
    while str(target) in sys.path:
        sys.path.remove(str(target))


def test_generic_private_target_rejects_world_writable_nonsticky_ancestor(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX ownership and mode policy")
    untrusted_parent = tmp_path / "world-writable-parent"
    untrusted_parent.mkdir(mode=0o700)
    untrusted_parent.chmod(0o777)
    calls: list[Any] = []
    installer = AllowlistedPipInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        environ={
            PROOF_REUSE_PROVISION_DIR_ENV: str(untrusted_parent / "provision")
        },
    )

    assert installer.install(services_module.MULTIFORMATS_DEPENDENCY) is False
    assert calls == []
    assert not (untrusted_parent / "provision").exists()


def test_generic_private_target_rejects_preloaded_transitive_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = services_module.MULTIFORMATS_DEPENDENCY

    def runner(command: tuple[str, ...], **_kwargs: Any) -> Any:
        target = Path(command[command.index("--target") + 1])
        for package_name in ("multiformats", "transitive_dependency"):
            package = target / package_name
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("# fixture\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    hostile = ModuleType("transitive_dependency")
    hostile.__file__ = str(tmp_path / "attacker" / "transitive_dependency.py")
    monkeypatch.setitem(sys.modules, "transitive_dependency", hostile)
    installer = AllowlistedPipInstaller(
        runner=runner,
        environ={PROOF_REUSE_PROVISION_DIR_ENV: str(tmp_path / "provision")},
    )

    assert installer.install(dependency) is False
    assert dependency.module_name not in installer._active_dependency_targets


def test_generic_private_target_concurrent_publication_reuses_one_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = services_module.MULTIFORMATS_DEPENDENCY
    barrier = threading.Barrier(2)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for module_name in tuple(sys.modules):
        if module_name == "multiformats" or module_name.startswith("multiformats."):
            monkeypatch.delitem(sys.modules, module_name)

    def runner(command: tuple[str, ...], **_kwargs: Any) -> Any:
        target = Path(command[command.index("--target") + 1])
        package = target / "multiformats"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# identical fixture\n", encoding="utf-8")
        barrier.wait(timeout=5)
        return SimpleNamespace(returncode=0)

    environment = {PROOF_REUSE_PROVISION_DIR_ENV: str(tmp_path / "provision")}
    installers = (
        AllowlistedPipInstaller(runner=runner, environ=environment),
        AllowlistedPipInstaller(runner=runner, environ=environment),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda item: item.install(dependency), installers))

    assert results == (True, True)
    targets = installers[0]._dependency_target_candidates(dependency)
    assert len(targets) == 1
    assert all(
        installer._validated_dependency_target(dependency) == targets[0]
        for installer in installers
    )
    while str(targets[0]) in sys.path:
        sys.path.remove(str(targets[0]))


def test_generic_tree_digest_rejects_leaf_replaced_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "target"
    package = root / "multiformats"
    package.mkdir(parents=True)
    victim = package / "__init__.py"
    victim.write_bytes(b"reviewed-tree-byte")
    replacement = tmp_path / "attacker.py"
    replacement.write_bytes(b"attacker-tree-byte")
    real_open = services_module.os.open
    replaced = False

    def replacing_open(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal replaced
        if Path(path) == victim and not replaced:
            replaced = True
            os.replace(replacement, victim)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(services_module.os, "open", replacing_open)

    assert AllowlistedPipInstaller._dependency_tree_digest(root) is None
    assert replaced is True


def test_generic_tree_digest_rejects_oversize_file_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "target"
    package = root / "multiformats"
    package.mkdir(parents=True)
    victim = package / "__init__.py"
    with victim.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    real_open = services_module.os.open
    victim_opened: list[Path] = []

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> int:
        if Path(path) == victim:
            victim_opened.append(victim)
            raise AssertionError("oversize dependency file must not be opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(services_module.os, "open", guarded_open)

    assert AllowlistedPipInstaller._dependency_tree_digest(root) is None
    assert victim_opened == []


@pytest.mark.parametrize(
    "invalid_result",
    (
        (True, "0", "", None),
        (True, True, "", None),
        (True, 0, b"not-text", None),
        (True, 0, "", "not-an-exception"),
    ),
)
def test_instrumented_private_installer_result_validates_every_field(
    tmp_path: Path,
    invalid_result: tuple[Any, ...],
) -> None:
    pip_installer = SimpleNamespace(
        install_with_diagnostics=lambda _dependency: invalid_result
    )
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *_args, **_kwargs: None,
        pip_installer=pip_installer,
        environ=_enabled_environment(tmp_path),
    )

    result = installer._run_instrumented_install(
        services_module.MULTIFORMATS_DEPENDENCY
    )

    assert result == (
        False,
        None,
        "invalid private target installer result",
        None,
    )


def test_private_pip_diagnostics_keep_only_bounded_head_and_tail(
    tmp_path: Path,
) -> None:
    prefix = "Network is unreachable"
    suffix = "ResolutionImpossible"

    def runner(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            returncode=1,
            stdout=prefix + "x" * 100_000,
            stderr="y" * 100_000 + suffix,
        )

    installer = AllowlistedPipInstaller(
        runner=runner,
        environ={PROOF_REUSE_PROVISION_DIR_ENV: str(tmp_path / "provision")},
    )

    succeeded, returncode, output, error = installer.install_with_diagnostics(
        services_module.MULTIFORMATS_DEPENDENCY
    )

    assert succeeded is False
    assert returncode == 1
    assert error is None
    assert len(output) <= 64 * 1024 + 1
    assert prefix in output
    assert suffix in output
    assert output.count("...[truncated]...") == 2


def test_native_capability_probe_rejects_path_substitution(
    tmp_path: Path,
) -> None:
    if os.name != "nt" and not hasattr(os, "memfd_create"):
        pytest.skip("sealed FD execution is unavailable")
    binary = tmp_path / "groth16"
    reviewed = b"reviewed-native-fixture"
    binary.write_bytes(reviewed)
    binary.chmod(0o700)
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], **_kwargs: Any) -> Any:
        commands.append(command)
        binary.write_bytes(b"substituted-after-validation")
        binary.chmod(0o700)
        return SimpleNamespace(
            returncode=0,
            stdout=_reviewed_capability_payload(),
            stderr=b"",
        )

    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        environ=_enabled_environment(tmp_path),
    )
    installer._validated_native_binary_identities[binary.resolve()] = (
        hashlib.sha256(reviewed).hexdigest(),
        len(reviewed),
    )

    ready, reason = installer._probe_groth16_binary_capabilities(
        binary,
        required_circuit_version=4,
    )

    assert ready is False
    assert reason == "capability_binary_changed_during_probe"
    if os.name != "nt":
        assert commands[0][0].startswith("/proc/self/fd/")


def test_arbitrary_preloaded_verifier_is_not_replaced_or_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType(services_module.DATASETS_VERIFIER_MODULE)
    module.verify_test_execution_certificate = lambda *_args, **_kwargs: True
    monkeypatch.setitem(sys.modules, services_module.DATASETS_VERIFIER_MODULE, module)
    calls: list[Any] = []
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        environ=_enabled_environment(tmp_path),
    )

    result = installer.ensure_capability("datasets_verifier")

    assert result.available is False
    assert result.reason_code == REASON_INCOMPATIBLE_VERSION
    assert result.diagnostics["module_provenance"] == "unreviewed"
    assert calls == []
    assert not (tmp_path / "home").exists()


def test_datasets_verifier_python_floor_starts_no_process_or_provisioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(lazy_module.sys, "version_info", (3, 11, 9))
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        importer=lambda name: (_ for _ in ()).throw(
            AssertionError(f"unexpected import: {name}")
        ),
        environ=_enabled_environment(tmp_path),
    )

    result = installer.ensure_capability("datasets_verifier")

    assert result.available is False
    assert result.reason_code == REASON_INCOMPATIBLE_VERSION
    assert result.diagnostics["requires_python"] == ">=3.12"
    assert result.diagnostics["install_process_started"] is False
    assert calls == []
    assert not (tmp_path / "home").exists()


def test_invalid_nltk_resource_never_imports_or_starts_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(lazy_module, "_installed_distribution_version", lambda _: None)
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        importer=lambda name: (_ for _ in ()).throw(
            AssertionError(f"unexpected import: {name}")
        ),
        environ=_enabled_environment(tmp_path),
    )

    result = installer.ensure_nltk_data(("../../untrusted",), consent=True)

    assert result.reason_code == REASON_NOT_ALLOWLISTED
    assert result.action == "RUN"
    assert calls == []


def test_missing_nltk_data_requires_resource_specific_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nltk = _fake_nltk()
    calls: list[Any] = []
    monkeypatch.setattr(lazy_module, "_installed_distribution_version", lambda _: None)
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        importer=lambda name: nltk,
        environ=_enabled_environment(tmp_path),
        lock_root=tmp_path / "locks",
    )

    result = installer.ensure_nltk_data(("punkt",))

    assert result.available is False
    assert result.reason_code == REASON_NLTK_DOWNLOAD_DISABLED
    assert result.action == "RUN"
    assert result.diagnostics["missing_resources"] == ["punkt"]
    assert calls == []


def test_nltk_download_root_rejects_configured_leaf_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "real-nltk-data"
    target.mkdir()
    link = tmp_path / "nltk-data-link"
    link.symlink_to(target, target_is_directory=True)
    environment = _enabled_environment(tmp_path)
    environment[PROOF_REUSE_NLTK_DATA_DIR_ENV] = str(link)
    calls: list[Any] = []
    monkeypatch.setattr(lazy_module, "_installed_distribution_version", lambda _: None)
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        importer=lambda name: _fake_nltk(),
        environ=environment,
    )

    result = installer.ensure_nltk_data(("punkt",), consent=True)

    assert result.reason_code == lazy_module.REASON_READ_ONLY_ENVIRONMENT
    assert calls == []


def test_python_install_fence_unavailable_fails_closed_without_pip(
    tmp_path: Path,
) -> None:
    malformed_lock_root = tmp_path / "lock-root-is-a-file"
    malformed_lock_root.write_text("not a directory", encoding="utf-8")
    calls: list[Any] = []
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        environ=_enabled_environment(tmp_path),
        lock_root=malformed_lock_root,
    )

    result = installer.ensure_capability("jsonschema")

    assert result.reason_code == REASON_LOCK_TIMEOUT
    assert result.diagnostics["install_process_started"] is False
    assert calls == []


def test_python_install_fence_uses_bounded_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    calls: list[Any] = []

    @contextmanager
    def unavailable_fence(*args: Any, **kwargs: Any) -> Any:
        observed.update(kwargs)
        yield False

    monkeypatch.setattr(
        lazy_module, "_bounded_interprocess_provision_fence", unavailable_fence
    )
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        environ=_enabled_environment(tmp_path),
        lock_root=tmp_path / "locks",
        lock_timeout_seconds=0.25,
    )

    result = installer.ensure_capability("jsonschema")

    assert result.reason_code == REASON_LOCK_TIMEOUT
    assert observed["timeout_seconds"] == 0.25
    assert calls == []


def test_python_install_fence_rejects_hostile_lock_symlink(
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    target = tmp_path / "attacker-target"
    target.write_text("unchanged", encoding="utf-8")
    target.chmod(0o600)
    (lock_root / "jsonschema.lock").symlink_to(target)
    calls: list[Any] = []
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        environ=_enabled_environment(tmp_path),
        lock_root=lock_root,
    )

    result = installer.ensure_capability("jsonschema")

    assert result.reason_code == REASON_LOCK_TIMEOUT
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert calls == []


def test_windows_provision_fence_contention_times_out_without_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContendedMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def locking(self, _descriptor: int, mode: int, count: int) -> None:
            self.calls.append((mode, count))
            raise OSError("lock held")

    backend = ContendedMsvcrt()
    clock = iter((0.0, 1.0))
    monkeypatch.setattr(
        lazy_module, "_load_provision_lock_backend", lambda: ("msvcrt", backend)
    )
    monkeypatch.setattr(lazy_module.time, "monotonic", lambda: next(clock))
    process_calls: list[Any] = []
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: process_calls.append((args, kwargs)),
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        environ=_enabled_environment(tmp_path),
        lock_root=tmp_path / "windows-locks",
        lock_timeout_seconds=0.1,
    )

    result = installer.ensure_capability("jsonschema")

    assert result.reason_code == REASON_LOCK_TIMEOUT
    assert process_calls == []
    assert backend.calls == [(backend.LK_NBLCK, 1)]


def test_windows_provision_fence_acquires_and_unlocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AvailableMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def locking(self, _descriptor: int, mode: int, count: int) -> None:
            self.calls.append((mode, count))

    backend = AvailableMsvcrt()
    monkeypatch.setattr(
        lazy_module, "_load_provision_lock_backend", lambda: ("msvcrt", backend)
    )

    with lazy_module._bounded_interprocess_provision_fence(
        tmp_path / "windows-locks",
        capability_key="nltk-data",
        timeout_seconds=0.1,
    ) as acquired:
        assert acquired is True

    assert backend.calls == [
        (backend.LK_NBLCK, 1),
        (backend.LK_UNLCK, 1),
    ]
    assert (tmp_path / "windows-locks" / "nltk-data.lock").stat().st_size == 1


def test_nltk_first_use_download_is_allowlisted_locked_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nltk = _fake_nltk()
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        if command[1:] == ("capabilities", "--json"):
            return SimpleNamespace(
                returncode=0,
                stdout=_reviewed_capability_payload(),
                stderr=b"",
            )
        calls.append((command, kwargs))
        for package_id in ("punkt", "words"):
            nltk.data.available.update(
                NLTK_DATA_RESOURCE_ALLOWLIST[package_id].find_paths
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lazy_module, "_installed_distribution_version", lambda _: None)
    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        importer=lambda name: nltk,
        environ=_enabled_environment(tmp_path),
        lock_root=tmp_path / "locks",
        timeout_seconds=5,
    )

    result = installer.ensure_nltk_data(("punkt", "words"), consent=True)

    assert result.available is True
    assert result.installed is True
    assert result.reason_code == REASON_AVAILABLE
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:5] == (
        os.sys.executable,
        "-I",
        "-m",
        "nltk.downloader",
        "--quiet",
    )
    assert "PYTHONPATH" not in kwargs["env"]
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert "--exit-on-error" in command
    assert "--dir" in command
    assert command[-2:] == ("punkt", "words")
    assert kwargs["timeout"] == 5.0
    assert (tmp_path / "locks").is_dir()
    assert set(DEFAULT_NLTK_DATA_RESOURCES) == set(NLTK_DATA_RESOURCE_ALLOWLIST)


def test_failed_nltk_attempt_is_memoized_and_root_lock_is_shared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nltk = _fake_nltk()
    calls: list[Any] = []
    monkeypatch.setattr(lazy_module, "_installed_distribution_version", lambda _: None)

    def runner(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout="", stderr="network failed")

    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        importer=lambda name: nltk,
        environ=_enabled_environment(tmp_path),
        lock_root=tmp_path / "locks",
        timeout_seconds=5,
    )

    first = installer.ensure_nltk_data(("punkt", "words"), consent=True)
    second = installer.ensure_nltk_data(("punkt", "words"), consent=True)

    assert first is second
    assert first.diagnostics["provision_attempted"] is True
    assert len(calls) == 1
    lock_files = list((tmp_path / "locks").glob("nltk-data-root-*.lock"))
    assert len(lock_files) == 1


def test_native_source_digest_validation_rejects_modified_rust_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "datasets"
    source.mkdir()
    real_backend = DATASETS_ROOT / "ipfs_datasets_py" / "processors" / "groth16_backend"
    copied_backend = source / "ipfs_datasets_py" / "processors" / "groth16_backend"
    for relative in DATASETS_GROTH16_REVIEWED_FILES_SHA256:
        destination = copied_backend / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real_backend / relative, destination)
    (copied_backend / "src" / "lib.rs").write_text(
        "// modified after reviewed HEAD\n", encoding="utf-8"
    )
    pip_installer = AllowlistedPipInstaller(environ={})
    monkeypatch.setattr(
        pip_installer,
        "validated_local_datasets_source",
        lambda: source,
    )
    installer = ProofReuseLazyDependencyInstaller(
        pip_installer=pip_installer,
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
    )

    result = installer.ensure_groth16_native_backend(consent=True)

    assert result.available is False
    assert result.reason_code == REASON_GROTH16_SOURCE_INVALID
    assert result.action == "DEFERRED"


def test_installed_wheel_backend_source_is_discovered_without_package_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_root = tmp_path / "wheel"
    backend = (
        wheel_root
        / "ipfs_datasets_py"
        / "processors"
        / "groth16_backend"
    )
    reviewed_backend = (
        DATASETS_ROOT
        / "ipfs_datasets_py"
        / "processors"
        / "groth16_backend"
    )
    for relative in DATASETS_GROTH16_REVIEWED_FILES_SHA256:
        destination = backend / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reviewed_backend / relative, destination)

    located: list[str] = []

    class _Distribution:
        @staticmethod
        def locate_file(relative: str) -> Path:
            located.append(relative)
            return wheel_root / relative

    monkeypatch.setattr(
        lazy_module.importlib.metadata,
        "distribution",
        lambda name: _Distribution(),
    )

    assert (
        ProofReuseLazyDependencyInstaller._installed_datasets_groth16_backend()
        == backend.resolve()
    )
    assert located == ["ipfs_datasets_py/processors/groth16_backend"]


def test_installed_wheel_backend_source_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_root = tmp_path / "wheel"
    backend = (
        wheel_root
        / "ipfs_datasets_py"
        / "processors"
        / "groth16_backend"
    )
    reviewed_backend = (
        DATASETS_ROOT
        / "ipfs_datasets_py"
        / "processors"
        / "groth16_backend"
    )
    for relative in DATASETS_GROTH16_REVIEWED_FILES_SHA256:
        destination = backend / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reviewed_backend / relative, destination)
    (backend / "src" / "main.rs").write_text("// tampered\n", encoding="utf-8")

    monkeypatch.setattr(
        lazy_module.importlib.metadata,
        "distribution",
        lambda _name: SimpleNamespace(
            locate_file=lambda relative: wheel_root / relative
        ),
    )

    assert (
        ProofReuseLazyDependencyInstaller._installed_datasets_groth16_backend()
        is None
    )


def test_native_build_needs_separate_consent_and_never_runs_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = tmp_path / "groth16_backend"
    (backend / "src").mkdir(parents=True)
    (backend / "Cargo.toml").write_text("[package]\nname='g'\n", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        if command[1:] == ("capabilities", "--json"):
            return SimpleNamespace(
                returncode=0,
                stdout=_reviewed_capability_payload(),
                stderr=b"",
            )
        target_root = Path(command[command.index("--target-dir") + 1])
        binary = target_root / "release" / "groth16"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"reviewed-build-output")
        binary.chmod(0o755)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
        lock_root=tmp_path / "locks",
        native_timeout_seconds=30,
    )
    monkeypatch.setattr(installer, "_validated_groth16_backend_dir", lambda: backend)
    monkeypatch.setattr(installer, "_reviewed_bundled_groth16_binary", lambda _: None)
    rustup_proxy = tmp_path / "rustup-proxy"
    rustup_proxy.write_bytes(b"fake rustup proxy")
    rustup_proxy.chmod(0o755)
    cargo_link = tmp_path / "cargo"
    cargo_link.symlink_to(rustup_proxy)
    system_git = shutil.which("git")
    monkeypatch.setattr(
        lazy_module.shutil,
        "which",
        lambda name: str(cargo_link) if name == "cargo" else system_git,
    )

    disabled = installer.ensure_groth16_native_backend(consent=False)
    assert disabled.reason_code == REASON_GROTH16_BUILD_DISABLED
    assert disabled.action == "DEFERRED"
    assert calls == []

    built = installer.ensure_groth16_native_backend(consent=True)
    assert built.available is True
    assert built.installed is True
    assert len(calls) == 2
    command, kwargs = next(
        entry for entry in calls if entry[0][1:4] == ("build", "--locked", "--release")
    )
    assert cargo_link.is_symlink()
    assert command[0] == str(cargo_link.absolute())
    assert command[1:4] == ("build", "--locked", "--release")
    assert "setup" not in command
    assert kwargs["timeout"] == 30.0
    assert kwargs["cwd"] == str(Path(backend.anchor))
    assert Path(command[command.index("--manifest-path") + 1]) != (
        backend / "Cargo.toml"
    )
    assert (
        kwargs["env"]["CARGO_TARGET_DIR"] == command[command.index("--target-dir") + 1]
    )
    assert "RUSTC_WRAPPER" not in kwargs["env"]
    assert "CARGO_BUILD_RUSTC_WRAPPER" not in kwargs["env"]
    assert built.diagnostics["immutable_source_snapshot"] is True
    assert built.diagnostics["isolated_cargo_environment"] is True
    assert built.diagnostics["trusted_setup_attempted"] is False


def test_isolated_cargo_environment_keeps_available_toolchain_functional(
    tmp_path: Path,
) -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("Cargo is not installed")
    baseline = subprocess.run(
        (cargo, "--version"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if baseline.returncode != 0:
        pytest.skip("the installed Cargo toolchain is not functional")
    build_root = tmp_path / "cargo-environment"
    build_root.mkdir()
    installer = ProofReuseLazyDependencyInstaller(environ=dict(os.environ))

    environment = installer._isolated_cargo_environment(
        build_root, build_root / "target"
    )

    assert environment is not None
    completed = subprocess.run(
        (os.path.abspath(cargo), "--version"),
        cwd=str(Path(cargo).anchor),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("cargo ")


def test_native_build_receipt_is_reused_and_tamper_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = tmp_path / "groth16_backend"
    (backend / "src").mkdir(parents=True)
    (backend / "Cargo.toml").write_text("[package]\nname='g'\n", encoding="utf-8")
    real_artifacts = (
        DATASETS_ROOT
        / "ipfs_datasets_py"
        / "processors"
        / "groth16_backend"
        / "artifacts"
    )
    for relative in (
        "v2/proving_key.bin",
        "v2/verifying_key.bin",
    ):
        assert relative in DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256
        destination = backend / "artifacts" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real_artifacts / relative, destination)
    provision_root = tmp_path / "provision"
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        if command[1:] == ("capabilities", "--json"):
            return SimpleNamespace(
                returncode=0,
                stdout=_reviewed_capability_payload(),
                stderr=b"",
            )
        calls.append((command, kwargs))
        target_root = Path(command[command.index("--target-dir") + 1])
        binary = target_root / "release" / "groth16"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"receipt-bound-native-output")
        binary.chmod(0o755)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    for name, value in {
        PROOF_REUSE_AUTO_INSTALL_ENV: "1",
        PACKAGE_AUTO_INSTALL_ENV: "1",
        PROOF_REUSE_PROVISION_DIR_ENV: str(provision_root),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(DATASETS_GROTH16_BINARY_ENV, raising=False)
    monkeypatch.delenv(lazy_module.DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, raising=False)
    monkeypatch.delenv(
        lazy_module._GROTH16_REVIEWED_ARTIFACTS_MARKER_ENV, raising=False
    )
    fake_cargo = tmp_path / "cargo"
    fake_cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_cargo.chmod(0o755)
    system_git = shutil.which("git")
    monkeypatch.setattr(
        lazy_module.shutil,
        "which",
        lambda name: str(fake_cargo) if name == "cargo" else system_git,
    )

    # Exercise the normal public facade: its default installer must activate
    # the validated binary in the process environment for datasets consumers.
    bootstrap = AcceleratorProofReuseBootstrap()
    first = bootstrap.installer
    first._process_runner = runner
    monkeypatch.setattr(first, "_validated_groth16_backend_dir", lambda: backend)
    monkeypatch.setattr(first, "_reviewed_bundled_groth16_binary", lambda _: None)
    built = bootstrap.ensure_groth16_native_backend(consent=True)

    assert built.available is True
    assert built.diagnostics["build_receipt_status"] == "persisted"
    assert built.diagnostics["process_environment_published"] is True
    binary = provision_root / "bin" / lazy_module._platform_binary_name() / "groth16"
    assert os.environ[DATASETS_GROTH16_BINARY_ENV] == str(binary)
    receipt = provision_root / "groth16-native-build.json"
    assert receipt.is_file()
    if os.name != "nt":
        assert receipt.stat().st_mode & 0o077 == 0
    keys = first.inspect_groth16_keys()
    assert keys.available is True
    assert keys.diagnostics["reviewed_bundled_artifacts"] is True
    (backend / "artifacts" / "v2" / "verifying_key.bin").write_bytes(
        b"tampered-reviewed-key"
    )
    runtime_with_tampered_key = first.inspect_groth16_runtime()
    assert runtime_with_tampered_key["keys"]["available"] is False
    assert (
        runtime_with_tampered_key["keys"]["diagnostics"]["reviewed_bundled_artifacts"]
        is True
    )
    fresh_installer = ProofReuseLazyDependencyInstaller()
    fresh_keys = fresh_installer.inspect_groth16_keys()
    assert fresh_keys.available is False
    assert fresh_keys.diagnostics["reviewed_bundled_artifacts"] is True

    shared_environment = {
        PROOF_REUSE_AUTO_INSTALL_ENV: "1",
        PACKAGE_AUTO_INSTALL_ENV: "1",
        PROOF_REUSE_PROVISION_DIR_ENV: str(provision_root),
    }

    def unexpected_runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        if command[1:] == ("capabilities", "--json"):
            return SimpleNamespace(
                returncode=0,
                stdout=_reviewed_capability_payload(),
                stderr=b"",
            )
        raise AssertionError(f"unexpected rebuild: {command!r} {kwargs!r}")

    second = ProofReuseLazyDependencyInstaller(
        runner=unexpected_runner,
        environ=shared_environment,
        lock_root=tmp_path / "locks",
    )
    monkeypatch.setattr(second, "_validated_groth16_backend_dir", lambda: backend)
    monkeypatch.setattr(second, "_reviewed_bundled_groth16_binary", lambda _: None)
    reused = second.ensure_groth16_native_backend(consent=False)

    assert reused.available is True
    assert reused.diagnostics["binary_source"] == "validated_build_receipt"
    assert reused.diagnostics["previous_native_build_receipt_status"] == "available"
    assert len(calls) == 1

    binary.write_bytes(b"tampered-after-receipt")
    same_installer_rejected = second.ensure_groth16_native_backend(consent=False)

    assert same_installer_rejected.available is False
    assert same_installer_rejected.reason_code == REASON_GROTH16_BUILD_DISABLED
    assert (
        same_installer_rejected.diagnostics["previous_native_build_receipt_status"]
        == "binary_digest_mismatch"
    )

    reset_bootstrap = AcceleratorProofReuseBootstrap()
    reset_installer = reset_bootstrap.installer
    reset_installer._process_runner = unexpected_runner
    monkeypatch.setattr(
        reset_installer, "_validated_groth16_backend_dir", lambda: backend
    )
    monkeypatch.setattr(
        reset_installer, "_reviewed_bundled_groth16_binary", lambda _: None
    )
    reset_rejected = reset_bootstrap.ensure_groth16_native_backend(consent=False)
    assert reset_rejected.available is False
    assert (
        reset_rejected.diagnostics["previous_native_build_receipt_status"]
        == "binary_digest_mismatch"
    )
    assert len(calls) == 1


def test_foreign_bundled_binary_is_diagnostic_not_native_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(lazy_module, "_platform_binary_name", lambda: "linux-x86_64")
    installer = ProofReuseLazyDependencyInstaller(
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
        lock_root=tmp_path / "locks",
    )

    result = installer.ensure_groth16_native_backend(consent=False)

    assert result.available is False
    assert result.reason_code == REASON_GROTH16_BUILD_DISABLED
    assert result.diagnostics["native_platform"] == "linux-x86_64"
    assert result.diagnostics["reviewed_bundled_platforms"] == ["linux-aarch64"]
    assert result.diagnostics["foreign_bundled_platforms"] == ["linux-aarch64"]
    assert result.diagnostics["foreign_binary_execution_attempted"] is False
    assert result.diagnostics["native_build_required"] is True
    assert calls == []


def test_ptr151_bundled_release_manifest_and_v4_capability_are_both_required(
    tmp_path: Path,
) -> None:
    if lazy_module._platform_binary_name() != "linux-aarch64":
        pytest.skip("reviewed PTR-151 bundle is linux-aarch64")
    installer = ProofReuseLazyDependencyInstaller(
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "0",
            PACKAGE_AUTO_INSTALL_ENV: "0",
        },
        lock_root=tmp_path / "locks",
    )

    resolution = installer.ensure_groth16_native_backend(consent=False)

    assert resolution.available is True
    assert resolution.diagnostics["binary_source"] == "reviewed_bundled_binary"
    assert resolution.diagnostics["required_circuit_version"] == 4
    assert resolution.diagnostics["required_capability_validated"] is True
    assert resolution.diagnostics["capability_probe_status"] == "available"
    assert resolution.diagnostics["supported_circuit_versions"] == [1, 2, 3, 4]


def test_endpoint_keys_circuit_and_native_are_independent(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts" / "v2"
    artifacts.mkdir(parents=True)
    (artifacts / "proving_key.bin").write_bytes(b"pk")
    (artifacts / "verifying_key.bin").write_bytes(b"vk")
    installer = ProofReuseLazyDependencyInstaller(
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "0",
            PACKAGE_AUTO_INSTALL_ENV: "0",
            PROOF_REUSE_GROTH16_ENDPOINT_ENV: "https://prover.example/v1",
            "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
            PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV: "knowledge_of_axioms@v2",
        }
    )

    endpoint = installer.inspect_groth16_endpoint()
    keys = installer.inspect_groth16_keys()
    circuit = installer.inspect_groth16_circuit()

    assert endpoint.available is True
    assert endpoint.diagnostics["network_attempted"] is False
    assert keys.available is True
    assert keys.diagnostics["verifying_key_versions"] == [2]
    assert circuit.available is True
    assert all(item.action == "DEFERRED" for item in (endpoint, keys, circuit))


def test_runtime_requires_circuit_key_version_match_and_native_proving_key(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts" / "v2"
    artifacts.mkdir(parents=True)
    (artifacts / "verifying_key.bin").write_bytes(b"vk")
    common = {
        PROOF_REUSE_AUTO_INSTALL_ENV: "0",
        PACKAGE_AUTO_INSTALL_ENV: "0",
        "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
        PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV: "knowledge_of_axioms@v2",
    }

    native_only = ProofReuseLazyDependencyInstaller(environ=common)
    native_status = native_only.inspect_groth16_runtime()

    assert native_status["ready"] is False
    assert native_status["version_compatibility"] == {
        "circuit_version": 2,
        "has_verifying_key": True,
        "has_proving_key": False,
        "native_ready": False,
        "endpoint_ready": False,
    }

    endpoint = ProofReuseLazyDependencyInstaller(
        environ={
            **common,
            PROOF_REUSE_GROTH16_ENDPOINT_ENV: "https://prover.example/v1",
        }
    )
    endpoint_status = endpoint.inspect_groth16_runtime()

    assert endpoint_status["ready"] is True
    assert endpoint_status["version_compatibility"]["endpoint_ready"] is True
    assert endpoint_status["test_certificate_authority_ready"] is False
    assert endpoint_status["skip_authority"] is False
    assert (
        endpoint_status["test_certificate_authority_reason"]
        == "requires_separate_test_certificate_authority_probe"
    )

    mismatch = ProofReuseLazyDependencyInstaller(
        environ={
            **common,
            PROOF_REUSE_GROTH16_ENDPOINT_ENV: "https://prover.example/v1",
            PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV: "knowledge_of_axioms@v1",
        }
    ).inspect_groth16_runtime()
    assert mismatch["ready"] is False


@pytest.mark.parametrize(
    "endpoint",
    (
        " https://prover.example/v1",
        "https://prover.example/v1 ",
        "https://prover.example\\@attacker.invalid/v1",
        "https://prover.example/line\nbreak",
    ),
)
def test_endpoint_parser_rejects_ambiguous_input(endpoint: str) -> None:
    result = ProofReuseLazyDependencyInstaller(
        environ={PROOF_REUSE_GROTH16_ENDPOINT_ENV: endpoint}
    ).inspect_groth16_endpoint()

    assert result.available is False
