"""Contract tests for pinned DuckDB/Quack capability probing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_ALLOW_NETWORK_INSTALL,
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    QuackCapabilityStatus,
    QuackCompatibilityProfile,
    QuackDiagnosticCode,
    clear_quack_capability_cache,
    compute_extension_fingerprint,
    default_compatibility_profile,
    parse_version,
    probe_quack_capabilities,
    quack_health_check,
)


class _Result:
    def __init__(
        self,
        rows: list[tuple[Any, ...]] | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self._rows = list(rows or [])
        self._offset = 0
        self.description = [(name,) for name in (columns or [])]

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows[self._offset :]
        self._offset = len(self._rows)
        return rows

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._offset >= len(self._rows):
            return None
        row = self._rows[self._offset]
        self._offset += 1
        return row


class FakeConnection:
    """In-memory DuckDB stand-in with programmable extension catalog."""

    def __init__(
        self,
        *,
        version: str = "1.5.2",
        extension: dict[str, Any] | None = None,
        functions: set[str] | None = None,
        load_error: Exception | None = None,
        install_error: Exception | None = None,
        settings_rows: list[tuple[Any, ...]] | None = None,
        fail_extensions: bool = False,
    ) -> None:
        self.version = version
        self.extension = dict(extension or {})
        self.functions = set(functions or ())
        self.load_error = load_error
        self.install_error = install_error
        self.settings_rows = list(settings_rows or [("quack_auth",), ("quack_log_level",)])
        self.fail_extensions = fail_extensions
        self.statements: list[str] = []
        self.closed = False

    def execute(self, sql: str) -> _Result:
        self.statements.append(sql)
        text = " ".join(sql.strip().split())
        upper = text.upper()

        if upper.startswith("SELECT VERSION()"):
            return _Result([(self.version,)], ["version()"])

        if "DUCKDB_EXTENSIONS()" in upper:
            if self.fail_extensions:
                raise RuntimeError("extensions catalog missing")
            if not self.extension:
                return _Result(
                    [],
                    [
                        "extension_name",
                        "loaded",
                        "installed",
                        "install_path",
                        "extension_version",
                        "install_mode",
                        "installed_from",
                        "description",
                    ],
                )
            row = (
                self.extension.get("extension_name", "quack"),
                self.extension.get("loaded", False),
                self.extension.get("installed", False),
                self.extension.get("install_path", "/opt/quack.duckdb_extension"),
                self.extension.get("extension_version", "0.1.0"),
                self.extension.get("install_mode", "custom"),
                self.extension.get("installed_from", "local"),
                self.extension.get("description", "quack"),
            )
            return _Result(
                [row],
                [
                    "extension_name",
                    "loaded",
                    "installed",
                    "install_path",
                    "extension_version",
                    "install_mode",
                    "installed_from",
                    "description",
                ],
            )

        if upper.startswith("INSTALL "):
            if self.install_error is not None:
                raise self.install_error
            self.extension = {
                "extension_name": "quack",
                "loaded": False,
                "installed": True,
                "install_path": "/tmp/quack.duckdb_extension",
                "extension_version": "0.1.0",
                "install_mode": "community",
                "installed_from": "https://extensions.example/quack",
                "description": "quack",
            }
            return _Result()

        if upper.startswith("LOAD "):
            if self.load_error is not None:
                raise self.load_error
            if not self.extension.get("installed") and not self.extension.get("loaded"):
                raise RuntimeError("extension not installed")
            self.extension["loaded"] = True
            self.extension["installed"] = True
            return _Result()

        if "DUCKDB_FUNCTIONS()" in upper or "INFORMATION_SCHEMA.ROUTINES" in upper:
            name = None
            if "function_name = '" in text:
                name = text.split("function_name = '", 1)[1].split("'", 1)[0]
            elif "routine_name) = lower('" in text:
                name = text.split("routine_name) = lower('", 1)[1].split("'", 1)[0]
            if name and name in self.functions:
                return _Result([(1,)], ["1"])
            return _Result([], ["1"])

        if "DUCKDB_SETTINGS()" in upper:
            return _Result(self.settings_rows, ["name"])

        if upper.startswith("SELECT ") and " IS NOT NULL" in upper:
            # Fallback function existence probe.
            ident = text.split("SELECT ", 1)[1].split(" IS NOT NULL", 1)[0].strip()
            if ident in self.functions:
                return _Result([(True,)], ["result"])
            raise RuntimeError(f"unknown function {ident}")

        raise RuntimeError(f"unexpected sql: {sql}")

    def close(self) -> None:
        self.closed = True


def _importer(module: Any):
    def import_module(name: str) -> Any:
        if name != "duckdb":
            raise ModuleNotFoundError(name)
        return module

    return import_module


def _factory(connection: FakeConnection):
    def connect(duckdb_module: Any) -> FakeConnection:
        assert duckdb_module is not None
        return connection

    return connect


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_quack_capability_cache()
    yield
    clear_quack_capability_cache()


def test_default_profile_pins_1_5_and_records_beta_limitations():
    profile = default_compatibility_profile()
    assert profile.profile_id == "agent-supervisor-duckdb-quack-1.5"
    assert profile.duckdb_version_prefix == "1.5"
    assert profile.extension_name == "quack"
    assert "quack_serve" in profile.required_functions
    assert "quack_query" in profile.required_functions
    assert "install_load_policy" in profile.required_surfaces
    assert profile.beta_limitations == DEFAULT_QUACK_BETA_LIMITATIONS
    assert profile.is_beta_profile
    payload = profile.to_dict()
    assert payload["schema_version"].endswith("quack-compatibility-profile@1")


def test_parse_version_and_fingerprint_helpers():
    parsed = parse_version("v1.5.2-dev")
    assert parsed is not None
    assert parsed.as_tuple() == (1, 5, 2)
    assert parse_version("not-a-version") is None

    extension = ExtensionObservation(
        name="quack",
        installed=True,
        loaded=True,
        install_path="/opt/quack",
        extension_version="0.1.0",
        install_mode="custom",
        installed_from="local",
    )
    digest = compute_extension_fingerprint(
        duckdb_version="1.5.2",
        extension=extension,
        platform_name="linux",
        platform_machine="x86_64",
        observed_functions=("quack_serve", "quack_query"),
    )
    assert digest.startswith("sha256:")


def test_unavailable_when_duckdb_cannot_be_imported():
    def importer(name: str) -> Any:
        raise ModuleNotFoundError(name)

    report = probe_quack_capabilities(importer=importer)
    assert report.status is QuackCapabilityStatus.UNAVAILABLE
    assert report.passes_health_check is False
    assert report.duckdb_importable is False
    codes = {item.code for item in report.diagnostics}
    assert QuackDiagnosticCode.DUCKDB_IMPORT_FAILED in codes
    assert QuackDiagnosticCode.IMPORT_ONLY_INSUFFICIENT in codes


def test_import_success_alone_cannot_pass():
    module = SimpleNamespace(__version__="1.5.2")
    connection = FakeConnection(extension=None, functions=set())
    report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection),
        platform_info=lambda: ("linux", "x86_64"),
    )
    assert report.duckdb_importable is True
    assert report.status is QuackCapabilityStatus.INSTALL_REQUIRED
    assert report.passes_health_check is False
    assert report.available is False
    assert any(
        item.code is QuackDiagnosticCode.IMPORT_ONLY_INSUFFICIENT
        for item in report.diagnostics
    )
    assert any(
        item.code is QuackDiagnosticCode.NETWORK_INSTALL_FORBIDDEN
        for item in report.diagnostics
    )
    assert report.network_install_attempted is False
    assert not any(stmt.upper().startswith("INSTALL ") for stmt in connection.statements)


def test_unsupported_duckdb_version_is_typed():
    module = SimpleNamespace(__version__="0.9.2")
    connection = FakeConnection(version="0.9.2")
    report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection),
    )
    assert report.status is QuackCapabilityStatus.UNSUPPORTED
    assert report.passes_health_check is False
    assert report.duckdb_version_parsed is not None
    assert report.duckdb_version_parsed.as_tuple() == (0, 9, 2)

    module2 = SimpleNamespace(__version__="2.0.0")
    connection2 = FakeConnection(version="2.0.0")
    report2 = probe_quack_capabilities(
        importer=_importer(module2),
        connection_factory=_factory(connection2),
    )
    assert report2.status is QuackCapabilityStatus.UNSUPPORTED


def test_install_required_never_network_installs_in_health_check():
    module = SimpleNamespace(__version__="1.5.2")
    connection = FakeConnection(
        extension={
            "extension_name": "quack",
            "loaded": False,
            "installed": False,
        }
    )
    report = quack_health_check(
        importer=_importer(module),
        connection_factory=_factory(connection),
    )
    assert report.status is QuackCapabilityStatus.INSTALL_REQUIRED
    assert report.network_install_allowed is False
    assert report.network_install_attempted is False
    assert DEFAULT_ALLOW_NETWORK_INSTALL is False
    assert not any(stmt.upper().startswith("INSTALL ") for stmt in connection.statements)


def test_load_required_when_installed_but_load_disabled_or_fails():
    module = SimpleNamespace(__version__="1.5.2")
    installed = {
        "extension_name": "quack",
        "loaded": False,
        "installed": True,
        "install_path": "/opt/quack.duckdb_extension",
        "extension_version": "0.1.0",
        "install_mode": "custom",
        "installed_from": "local",
    }
    connection = FakeConnection(extension=dict(installed), functions=set())
    report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection),
        allow_local_load=False,
    )
    assert report.status is QuackCapabilityStatus.LOAD_REQUIRED
    assert report.local_load_attempted is False

    connection_fail = FakeConnection(
        extension=dict(installed),
        functions=set(),
        load_error=RuntimeError("broken extension"),
    )
    report_fail = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection_fail),
        allow_local_load=True,
    )
    assert report_fail.status is QuackCapabilityStatus.LOAD_REQUIRED
    assert report_fail.local_load_attempted is True
    assert any(
        item.code is QuackDiagnosticCode.EXTENSION_LOAD_FAILED
        for item in report_fail.diagnostics
    )


def test_compatible_when_pin_and_functions_match():
    module = SimpleNamespace(__version__="1.5.2")
    connection = FakeConnection(
        version="1.5.2",
        extension={
            "extension_name": "quack",
            "loaded": False,
            "installed": True,
            "install_path": "/opt/quack.duckdb_extension",
            "extension_version": "0.1.0",
            "install_mode": "custom",
            "installed_from": "local",
        },
        functions={"quack_serve", "quack_query"},
    )
    report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection),
        platform_info=lambda: ("linux", "x86_64"),
    )
    assert report.status is QuackCapabilityStatus.COMPATIBLE
    assert report.passes_health_check is True
    assert report.available is True
    assert report.extension is not None
    assert report.extension.loaded is True
    assert report.observed_functions == ("quack_serve", "quack_query")
    assert "install_load_policy" in report.observed_surfaces
    assert "extension_fingerprint" in report.observed_surfaces
    assert report.extension_fingerprint.startswith("sha256:")
    assert report.beta_limitations == DEFAULT_QUACK_BETA_LIMITATIONS
    assert report.network_install_attempted is False
    assert any(
        item.code is QuackDiagnosticCode.BETA_LIMITATIONS_RECORDED
        for item in report.diagnostics
    )
    # Import-only diagnostic remains for audit even on success path.
    assert any(
        item.code is QuackDiagnosticCode.IMPORT_ONLY_INSUFFICIENT
        for item in report.diagnostics
    )


def test_mismatched_when_required_functions_missing():
    module = SimpleNamespace(__version__="1.5.2")
    connection = FakeConnection(
        extension={
            "extension_name": "quack",
            "loaded": True,
            "installed": True,
            "install_path": "/opt/quack.duckdb_extension",
            "extension_version": "0.1.0",
            "install_mode": "custom",
            "installed_from": "local",
        },
        functions={"quack_serve"},  # quack_query missing
    )
    report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection),
    )
    assert report.status is QuackCapabilityStatus.MISMATCHED
    assert report.passes_health_check is False
    assert report.missing_functions == ("quack_query",)


def test_mismatched_fingerprint_pin():
    module = SimpleNamespace(__version__="1.5.2")
    connection = FakeConnection(
        extension={
            "extension_name": "quack",
            "loaded": True,
            "installed": True,
            "install_path": "/opt/quack.duckdb_extension",
            "extension_version": "0.1.0",
            "install_mode": "custom",
            "installed_from": "local",
        },
        functions={"quack_serve", "quack_query"},
    )
    profile = QuackCompatibilityProfile(
        pinned_extension_fingerprint="sha256:deadbeef",
        allow_experimental_within_minor=False,
    )
    report = probe_quack_capabilities(
        profile=profile,
        importer=_importer(module),
        connection_factory=_factory(connection),
        platform_info=lambda: ("linux", "x86_64"),
    )
    assert report.status is QuackCapabilityStatus.MISMATCHED
    assert any(
        item.code is QuackDiagnosticCode.EXTENSION_FINGERPRINT_MISMATCH
        for item in report.diagnostics
    )


def test_experimental_for_community_origin_and_off_minor_supported_build():
    module = SimpleNamespace(__version__="1.5.2")
    community = FakeConnection(
        extension={
            "extension_name": "quack",
            "loaded": True,
            "installed": True,
            "install_path": "/tmp/quack.duckdb_extension",
            "extension_version": "0.1.0",
            "install_mode": "community",
            "installed_from": "https://extensions.example/quack",
        },
        functions={"quack_serve", "quack_query"},
    )
    community_report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(community),
    )
    assert community_report.status is QuackCapabilityStatus.EXPERIMENTAL
    assert community_report.experimental_usable is True
    assert community_report.passes_health_check is False
    assert community_report.beta_limitations

    off_minor = SimpleNamespace(__version__="1.4.1")
    off_conn = FakeConnection(
        version="1.4.1",
        extension={
            "extension_name": "quack",
            "loaded": True,
            "installed": True,
            "install_path": "/opt/quack.duckdb_extension",
            "extension_version": "0.1.0",
            "install_mode": "custom",
            "installed_from": "local",
        },
        functions={"quack_serve", "quack_query"},
    )
    off_report = probe_quack_capabilities(
        importer=_importer(off_minor),
        connection_factory=_factory(off_conn),
    )
    assert off_report.status is QuackCapabilityStatus.EXPERIMENTAL


def test_explicit_network_install_is_opt_in_only():
    module = SimpleNamespace(__version__="1.5.2")
    connection = FakeConnection(extension=None, functions={"quack_serve", "quack_query"})
    # Health path must not install even if kwargs try to force it through helper.
    health = quack_health_check(
        importer=_importer(module),
        connection_factory=_factory(connection),
        allow_network_install=True,  # ignored by quack_health_check
    )
    assert health.status is QuackCapabilityStatus.INSTALL_REQUIRED
    assert health.network_install_attempted is False

    connection2 = FakeConnection(extension=None, functions={"quack_serve", "quack_query"})
    report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection2),
        allow_network_install=True,
        platform_info=lambda: ("linux", "x86_64"),
    )
    assert report.network_install_allowed is True
    assert report.network_install_attempted is True
    assert any(stmt.upper().startswith("INSTALL ") for stmt in connection2.statements)
    # Community origin after network install is experimental, not ordinary health.
    assert report.status is QuackCapabilityStatus.EXPERIMENTAL


def test_report_serialization_and_status_set_are_closed():
    statuses = {item.value for item in QuackCapabilityStatus}
    assert statuses == {
        "unavailable",
        "unsupported",
        "install-required",
        "load-required",
        "compatible",
        "mismatched",
        "experimental",
    }
    module = SimpleNamespace(__version__="1.5.2")
    connection = FakeConnection(
        extension={
            "extension_name": "quack",
            "loaded": True,
            "installed": True,
            "install_mode": "custom",
            "installed_from": "local",
        },
        functions={"quack_serve", "quack_query"},
    )
    report = probe_quack_capabilities(
        importer=_importer(module),
        connection_factory=_factory(connection),
    )
    payload = report.to_dict()
    assert payload["schema_version"].endswith("quack-capability-report@1")
    assert payload["status"] in statuses
    assert payload["beta_limitations"]
    assert payload["details"]["install_load_policy"]["import_alone_insufficient"] is True
    assert payload["details"]["install_load_policy"]["network_install_default"] is False
