"""Tests for automatic provider-command binding inference and healing."""

from __future__ import annotations

import textwrap

import pytest

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
        "from ipfs_accelerate_py.agent_supervisor.provider_command_environment "
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
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"
    )
    assert report.complete
    # Simulate a stripped runner namespace and heal again
    import ipfs_accelerate_py.agent_supervisor.grok_cli_runner as runner

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
        assert module.startswith("ipfs_accelerate_py.agent_supervisor.")
        assert attr
        # Resolvable
        resolve_provider_command_symbol(symbol)
