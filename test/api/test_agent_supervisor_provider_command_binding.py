"""Tests for automatic provider-command binding inference and healing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.provider_command_binding import (
    CANONICAL_PROVIDER_COMMAND_BINDINGS,
    ProviderCommandBindingError,
    ensure_provider_command_bindings,
    extract_name_error_symbol,
    group_import_statements,
    infer_provider_command_import,
    infer_provider_command_imports,
    missing_provider_command_bindings_in_source,
    preflight_provider_entry_module,
    recover_provider_command_name_error,
    residual_import_patch_for_report,
    resolve_provider_command_symbol,
    scan_source_for_provider_command_names,
)


_SEALED_PROVIDER_PREFLIGHT_BOOTSTRAP = r"""
import importlib
import importlib.machinery
import json
import sys
import types

archive = sys.argv[1]
sys.path.insert(0, archive)
import ipfs_accelerate_py

# Match the sealed control-plane bootstrap: mount the accepted supervisor
# package directly without executing its compatibility-alias initializer.
package_name = "ipfs_accelerate_py.agent_supervisor"
package_path = archive + "/ipfs_accelerate_py/agent_supervisor"
package_file = package_path + "/__init__.py"
package_spec = importlib.machinery.ModuleSpec(
    package_name,
    loader=None,
    origin=package_file,
    is_package=True,
)
package_spec.submodule_search_locations = [package_path]
package = types.ModuleType(package_name)
package.__file__ = package_file
package.__package__ = package_name
package.__path__ = [package_path]
package.__spec__ = package_spec
sys.modules[package_name] = package
ipfs_accelerate_py.agent_supervisor = package

binding_name = package_name + ".runtime.provider_command_binding"
environment_name = package_name + ".runtime.provider_command_environment"
runner_name = package_name + ".runtime.grok_cli_runner"
binding = importlib.import_module(binding_name)
environment = importlib.import_module(environment_name)

bindings = binding.CANONICAL_PROVIDER_COMMAND_BINDINGS
assert len(bindings) == 14
for symbol, (module_name, attribute_name) in bindings.items():
    assert module_name == environment_name
    assert binding.resolve_provider_command_symbol(symbol) is getattr(
        environment,
        attribute_name,
    )

report = binding.preflight_provider_entry_module(runner_name)
assert report.complete
runner = sys.modules[runner_name]
runner_origin = getattr(runner, "__file__", None)
environment_origin = getattr(environment, "__file__", None)
assert isinstance(runner_origin, str) and runner_origin.startswith(archive + "/")
assert isinstance(environment_origin, str) and environment_origin.startswith(
    archive + "/"
)
assert package_name + ".grok_cli_runner" not in sys.modules
assert package_name + ".provider_command_environment" not in sys.modules
print(
    json.dumps(
        {
            "binding_count": len(bindings),
            "complete": report.complete,
            "environment_origin": environment_origin,
            "runner_origin": runner_origin,
        },
        sort_keys=True,
    )
)
"""


_MATERIALIZE_PROVIDER_CAPSULE_BOOTSTRAP = r"""
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve(strict=True)
capsule_parent = Path(sys.argv[2])
sys.path.insert(0, str(source))
from ipfs_accelerate_py import llm_router

head, tree = llm_router.agent_implementation_control_plane_source_generation(
    source
)
pin = llm_router.materialize_agent_implementation_control_plane_capsule(
    source_root=source,
    capsule_parent=capsule_parent,
    source_head=head,
    source_tree=tree,
)
print(json.dumps(pin.as_dict(), sort_keys=True))
"""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


def _sealed_provider_control_plane(
    source: Path,
    capsule_parent: Path,
) -> tuple[
    llm_router.AgentImplementationControlPlanePin,
    llm_router.AgentImplementationSealedControlPlane,
]:
    accepted_root = Path(llm_router.__file__).resolve().parents[1]
    accepted_files = llm_router._agent_control_plane_source_files(
        accepted_root,
        verify_loaded_origins=False,
    )
    for accepted in accepted_files:
        destination = source / accepted.relative_to(accepted_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(accepted.read_bytes())
        destination.chmod(0o644)
    source.mkdir(parents=True, exist_ok=True)
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "provider-capsule@example.invalid")
    _git(source, "config", "user.name", "Provider Capsule Test")
    _git(source, "config", "commit.gpgsign", "false")
    _git(source, "config", "core.filemode", "true")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "accepted provider control plane")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    materialized = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _MATERIALIZE_PROVIDER_CAPSULE_BOOTSTRAP,
            str(source),
            str(capsule_parent),
        ],
        cwd=source,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if materialized.returncode != 0:
        raise AssertionError(materialized.stderr)
    pin = llm_router.AgentImplementationControlPlanePin(
        **json.loads(materialized.stdout)
    )
    return pin, llm_router.seal_agent_implementation_control_plane_capsule(pin)


def test_canonical_symbols_resolve() -> None:
    for symbol in (
        "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
        "PROVIDER_COMMAND_ENV_WRAPPER_ENV",
        "PROVIDER_COMMAND_ENV_DIGEST_ENV",
        "FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV",
        "ProviderCommandEnvironmentError",
        "sealed_provider_command_environment",
    ):
        value = resolve_provider_command_symbol(symbol)
        assert value is not None


def test_infer_import_statement_is_deterministic() -> None:
    fix = infer_provider_command_import("ProviderCommandEnvironmentError")
    assert fix.import_statement == (
        "from ipfs_accelerate_py.agent_supervisor.runtime.provider_command_environment "
        "import ProviderCommandEnvironmentError"
    )
    fixes = infer_provider_command_imports(
        [
            "PROVIDER_COMMAND_ENV_DIGEST_ENV",
            "PROVIDER_COMMAND_ENV_WRAPPER_ENV",
            "sealed_provider_command_environment",
        ]
    )
    statements = group_import_statements(fixes)
    assert len(statements) == 1
    assert "PROVIDER_COMMAND_ENV_DIGEST_ENV" in statements[0]
    assert "sealed_provider_command_environment" in statements[0]


def test_ensure_bindings_heals_empty_namespace() -> None:
    namespace: dict[str, object] = {}
    report = ensure_provider_command_bindings(
        namespace,
        required=(
            "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
            "sealed_provider_command_environment",
            "ProviderCommandEnvironmentError",
        ),
        namespace_name="test-ns",
    )
    assert report.complete
    assert set(report.bound_now) == {
        "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
        "sealed_provider_command_environment",
        "ProviderCommandEnvironmentError",
    }
    assert "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV" in namespace
    assert callable(namespace["sealed_provider_command_environment"])


def test_ensure_bindings_does_not_overwrite_existing() -> None:
    sentinel = object()
    namespace = {"PROVIDER_COMMAND_ENV_WRAPPER_ENV": sentinel}
    report = ensure_provider_command_bindings(
        namespace,
        required=("PROVIDER_COMMAND_ENV_WRAPPER_ENV",),
    )
    assert report.already_bound == ["PROVIDER_COMMAND_ENV_WRAPPER_ENV"]
    assert not report.bound_now
    assert namespace["PROVIDER_COMMAND_ENV_WRAPPER_ENV"] is sentinel


def test_scan_and_missing_from_source() -> None:
    source = textwrap.dedent(
        """
        def _run():
            x = PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV
            sealed_provider_command_environment(os.environ)
            raise ProviderCommandEnvironmentError("boom")
        """
    )
    used = scan_source_for_provider_command_names(source)
    assert "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV" in used
    assert "sealed_provider_command_environment" in used
    assert "ProviderCommandEnvironmentError" in used
    missing = missing_provider_command_bindings_in_source(source)
    assert set(missing) == used


def test_missing_not_reported_when_imported() -> None:
    source = textwrap.dedent(
        """
        from ipfs_accelerate_py.agent_supervisor.provider_command_environment import (
            PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV,
            sealed_provider_command_environment,
        )
        def _run():
            return PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV, sealed_provider_command_environment
        """
    )
    missing = missing_provider_command_bindings_in_source(source)
    assert missing == []


def test_recover_name_error_binds_symbol() -> None:
    namespace: dict[str, object] = {}
    exc = NameError("name 'PROVIDER_COMMAND_ENV_DIGEST_ENV' is not defined")
    report = recover_provider_command_name_error(exc, namespace)
    assert report is not None
    assert report.bound_now == ["PROVIDER_COMMAND_ENV_DIGEST_ENV"]
    assert "PROVIDER_COMMAND_ENV_DIGEST_ENV" in namespace
    assert extract_name_error_symbol(exc) == "PROVIDER_COMMAND_ENV_DIGEST_ENV"
    # Unrelated NameError is ignored
    assert recover_provider_command_name_error(NameError("name 'foo' is not defined"), {}) is None


def test_residual_import_patch_for_missing() -> None:
    namespace: dict[str, object] = {}
    report = ensure_provider_command_bindings(
        namespace,
        required=("FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV",),
    )
    # Already healed into namespace; craft a missing report shape
    report.missing = ["FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV"]
    report.inferred_fixes = infer_provider_command_imports(report.missing)
    patch = residual_import_patch_for_report(report)
    assert "FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV" in patch
    assert patch.startswith("# Auto-inferred")


def test_preflight_grok_cli_runner_heals_and_passes() -> None:
    report = preflight_provider_entry_module(
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner"
    )
    assert report.complete
    # Simulate a stripped runner namespace and heal again
    import ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner as runner

    for symbol in (
        "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
        "ProviderCommandEnvironmentError",
    ):
        if symbol in runner.__dict__:
            del runner.__dict__[symbol]
    report2 = ensure_provider_command_bindings(
        runner.__dict__,
        required=(
            "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
            "ProviderCommandEnvironmentError",
            "sealed_provider_command_environment",
            "FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV",
        ),
        namespace_name=runner.__name__,
    )
    assert report2.complete
    assert "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV" in runner.__dict__


def test_unknown_symbol_strict_raises() -> None:
    with pytest.raises(ProviderCommandBindingError):
        ensure_provider_command_bindings(
            {},
            required=("NOT_A_REAL_SYMBOL",),
            strict=True,
        )


def test_canonical_map_covers_registry() -> None:
    assert len(CANONICAL_PROVIDER_COMMAND_BINDINGS) >= 10
    for symbol, (module, attr) in CANONICAL_PROVIDER_COMMAND_BINDINGS.items():
        assert module == (
            "ipfs_accelerate_py.agent_supervisor.runtime."
            "provider_command_environment"
        )
        assert attr
        # Resolvable
        resolve_provider_command_symbol(symbol)


def test_sealed_archive_resolves_provider_bindings_and_preflights_runtime_runner(
    tmp_path: Path,
) -> None:
    pin, sealed = _sealed_provider_control_plane(
        tmp_path / "source",
        tmp_path / "capsules",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _SEALED_PROVIDER_PREFLIGHT_BOOTSTRAP,
            sealed.executable_path,
        ],
        env=dict(os.environ),
        pass_fds=(sealed.descriptor,),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    try:
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result == {
            "binding_count": 14,
            "complete": True,
            "environment_origin": (
                sealed.executable_path
                + "/ipfs_accelerate_py/agent_supervisor/runtime/"
                "provider_command_environment.py"
            ),
            "runner_origin": (
                sealed.executable_path
                + "/ipfs_accelerate_py/agent_supervisor/runtime/"
                "grok_cli_runner.py"
            ),
        }
        assert sealed.capsule_id == pin.capsule_id
        assert sealed.archive_sha256 == pin.archive_sha256
    finally:
        os.close(sealed.descriptor)
