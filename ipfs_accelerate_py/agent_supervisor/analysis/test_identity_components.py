"""Component identity compiler for fixtures, hooks, env, and parameters (PTR-011)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    mint_content_identity,
)

TEST_IDENTITY_COMPONENTS_INTERFACE: Final = "TestIdentityComponents@1"
TEST_IDENTITY_COMPONENTS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-identity-components@1"
)
_MAX_ITEMS: Final = 512
_MAX_REASON: Final = 128

# Privacy-safe environment variables that may participate in item identity.
# Secrets, credentials, and host-local absolute paths must never appear here.
DEFAULT_ENVIRONMENT_ALLOWLIST: Final[tuple[str, ...]] = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "PYTHONUTF8",
    "PYTHONWARNINGS",
    "PYTEST_ADDOPTS",
    "PYTEST_CURRENT_TEST",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "PYTEST_PLUGINS",
    "TZ",
    "IPFS_TEST_PROOF_REUSE_MODE",
)


def _bounded_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): value[key] for key in list(value)[:_MAX_ITEMS]}


def _bounded_records(value: Any, *, name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of mappings")
    rows: list[dict[str, Any]] = []
    for item in list(value)[:_MAX_ITEMS]:
        if not isinstance(item, Mapping):
            raise TypeError(f"{name} entries must be mappings")
        rows.append(dict(item))
    return tuple(rows)


def _bounded_pairs(value: Any, *, name: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of pairs")
    pairs: list[tuple[str, str]] = []
    for item in list(value)[:_MAX_ITEMS]:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{name} entries must be (name, version) pairs")
        pairs.append((str(item[0]), str(item[1])))
    return tuple(pairs)


def canonicalize_pytest_parameter(value: Any) -> Any:
    """Return a JSON-safe parameter value or raise for unsupported types."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise ValueError("floating-point parameters are not reusable")
    if isinstance(value, (bytes, bytearray)):
        raise ValueError("bytes parameters are not reusable")
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize_pytest_parameter(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [canonicalize_pytest_parameter(item) for item in value]
    raise ValueError(f"unsupported parameter type: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TestIdentityComponents:
    """Compiled identity roots for one pytest item's non-AST components."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = TEST_IDENTITY_COMPONENTS_INTERFACE

    reusable: bool
    component_root_cid: str
    parameter_cid: str = ""
    non_reusable_reasons: tuple[str, ...] = ()
    fixture_cids: tuple[str, ...] = ()
    conftest_closure_cid: str = ""
    hook_plugin_cids: tuple[str, ...] = ()
    dependency_lock_cid: str = ""
    installed_distributions_cid: str = ""
    environment_cid: str = ""
    platform_cid: str = ""
    interpreter_abi_cid: str = ""
    hardware_capability_cid: str = ""
    parameter_source_cid: str = ""
    components: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def compile(
        cls,
        *,
        parameter_id: str = "",
        parameter_value: Any = None,
        fixtures: Sequence[Mapping[str, Any]] | None = None,
        conftests: Sequence[Mapping[str, Any]] | None = None,
        hooks: Sequence[Mapping[str, Any]] | None = None,
        plugins: Sequence[Mapping[str, Any]] | None = None,
        lock_files: Sequence[Mapping[str, Any]] | None = None,
        installed_distributions: Sequence[Sequence[str]] | None = None,
        environment: Mapping[str, Any] | None = None,
        environment_allowlist: Sequence[str] | None = None,
        interpreter_facts: Mapping[str, Any] | None = None,
        platform_facts: Mapping[str, Any] | None = None,
        hardware_facts: Mapping[str, Any] | None = None,
        capability_facts: Mapping[str, Any] | None = None,
        capability_allowlist: Sequence[str] | None = None,
        **_extra: Any,
    ) -> "TestIdentityComponents":
        """Compile component CIDs; unsupported inputs become non-reusable reasons."""

        reasons: list[str] = []
        fixture_rows = _bounded_records(fixtures, name="fixtures")
        conftest_rows = _bounded_records(conftests, name="conftests")
        hook_rows = _bounded_records(hooks, name="hooks")
        plugin_rows = _bounded_records(plugins, name="plugins")
        lock_rows = _bounded_records(lock_files, name="lock_files")
        dist_pairs = _bounded_pairs(
            installed_distributions, name="installed_distributions"
        )
        env_map = _bounded_mapping(environment, name="environment")
        allowlist = tuple(
            str(item) for item in (environment_allowlist or ()) if str(item).strip()
        )[:_MAX_ITEMS]
        if allowlist:
            env_map = {
                key: value for key, value in env_map.items() if key in set(allowlist)
            }

        parameter_cid = ""
        parameter_source_cid = ""
        if parameter_id:
            try:
                canonical_parameter = canonicalize_pytest_parameter(parameter_value)
                parameter_identity = mint_content_identity(
                    {
                        "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/parameter",
                        "parameter_id": str(parameter_id),
                        "value": canonical_parameter,
                    }
                )
                parameter_cid = parameter_identity.cid
                parameter_source_cid = parameter_cid
            except Exception as exc:
                reasons.append(f"parameter_unsupported:{type(exc).__name__}"[:_MAX_REASON])

        fixture_cids: list[str] = []
        for row in fixture_rows:
            try:
                fixture_cids.append(
                    mint_content_identity(
                        {
                            "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/fixture",
                            "fixture": row,
                        }
                    ).cid
                )
            except Exception as exc:
                reasons.append(f"fixture_mint_failed:{type(exc).__name__}"[:_MAX_REASON])

        try:
            conftest_closure_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/conftests",
                    "conftests": list(conftest_rows),
                }
            ).cid
        except Exception as exc:
            conftest_closure_cid = ""
            reasons.append(f"conftest_mint_failed:{type(exc).__name__}"[:_MAX_REASON])

        hook_plugin_cids: list[str] = []
        for row in list(hook_rows) + list(plugin_rows):
            try:
                hook_plugin_cids.append(
                    mint_content_identity(
                        {
                            "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/hook-plugin",
                            "record": row,
                        }
                    ).cid
                )
            except Exception as exc:
                reasons.append(
                    f"hook_plugin_mint_failed:{type(exc).__name__}"[:_MAX_REASON]
                )

        try:
            dependency_lock_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/locks",
                    "lock_files": list(lock_rows),
                }
            ).cid
        except Exception as exc:
            dependency_lock_cid = ""
            reasons.append(f"lock_mint_failed:{type(exc).__name__}"[:_MAX_REASON])

        try:
            installed_distributions_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/distributions",
                    "installed_distributions": [
                        [name, version] for name, version in dist_pairs
                    ],
                }
            ).cid
        except Exception as exc:
            installed_distributions_cid = ""
            reasons.append(
                f"distribution_mint_failed:{type(exc).__name__}"[:_MAX_REASON]
            )

        try:
            environment_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/environment",
                    "environment": env_map,
                    "allowlist": list(allowlist),
                }
            ).cid
        except Exception as exc:
            environment_cid = ""
            reasons.append(f"environment_mint_failed:{type(exc).__name__}"[:_MAX_REASON])

        try:
            interpreter_abi_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/interpreter",
                    "facts": _bounded_mapping(
                        interpreter_facts, name="interpreter_facts"
                    ),
                }
            ).cid
        except Exception as exc:
            interpreter_abi_cid = ""
            reasons.append(f"interpreter_mint_failed:{type(exc).__name__}"[:_MAX_REASON])

        try:
            platform_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/platform",
                    "facts": _bounded_mapping(platform_facts, name="platform_facts"),
                }
            ).cid
        except Exception as exc:
            platform_cid = ""
            reasons.append(f"platform_mint_failed:{type(exc).__name__}"[:_MAX_REASON])

        try:
            hardware_capability_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA + "/hardware",
                    "hardware": _bounded_mapping(hardware_facts, name="hardware_facts"),
                    "capabilities": _bounded_mapping(
                        capability_facts, name="capability_facts"
                    ),
                    "allowlist": [
                        str(item)
                        for item in (capability_allowlist or ())
                        if str(item).strip()
                    ][:_MAX_ITEMS],
                }
            ).cid
        except Exception as exc:
            hardware_capability_cid = ""
            reasons.append(f"hardware_mint_failed:{type(exc).__name__}"[:_MAX_REASON])

        component_map = {
            "fixtures": ",".join(fixture_cids),
            "conftests": conftest_closure_cid,
            "hooks_plugins": ",".join(hook_plugin_cids),
            "locks": dependency_lock_cid,
            "distributions": installed_distributions_cid,
            "environment": environment_cid,
            "interpreter": interpreter_abi_cid,
            "platform": platform_cid,
            "hardware": hardware_capability_cid,
            "parameter": parameter_cid,
        }
        try:
            component_root_cid = mint_content_identity(
                {
                    "schema": TEST_IDENTITY_COMPONENTS_SCHEMA,
                    "interface": TEST_IDENTITY_COMPONENTS_INTERFACE,
                    "components": component_map,
                }
            ).cid
        except Exception as exc:
            component_root_cid = ""
            reasons.append(f"component_root_failed:{type(exc).__name__}"[:_MAX_REASON])

        reusable = not reasons and bool(component_root_cid)
        return cls(
            reusable=reusable,
            component_root_cid=component_root_cid,
            parameter_cid=parameter_cid,
            non_reusable_reasons=tuple(reasons),
            fixture_cids=tuple(fixture_cids),
            conftest_closure_cid=conftest_closure_cid,
            hook_plugin_cids=tuple(hook_plugin_cids),
            dependency_lock_cid=dependency_lock_cid,
            installed_distributions_cid=installed_distributions_cid,
            environment_cid=environment_cid,
            platform_cid=platform_cid,
            interpreter_abi_cid=interpreter_abi_cid,
            hardware_capability_cid=hardware_capability_cid,
            parameter_source_cid=parameter_source_cid,
            components=MappingProxyType(component_map),
        )

    def execution_key_fields(self) -> dict[str, Any]:
        """Fields merged into :meth:`TestExecutionIdentityCompiler.compile_execution_key`."""

        return {
            "fixture_cids": self.fixture_cids,
            "conftest_closure_cid": self.conftest_closure_cid,
            "hook_plugin_cids": self.hook_plugin_cids,
            "dependency_lock_cid": self.dependency_lock_cid,
            "installed_distributions_cid": self.installed_distributions_cid,
            "environment_cid": self.environment_cid,
            "platform_cid": self.platform_cid,
            "interpreter_abi_cid": self.interpreter_abi_cid,
            "hardware_capability_cid": self.hardware_capability_cid,
            "parameter_source_cid": self.parameter_source_cid,
            "components": dict(self.components),
        }


__all__ = (
    "DEFAULT_ENVIRONMENT_ALLOWLIST",
    "TEST_IDENTITY_COMPONENTS_INTERFACE",
    "TEST_IDENTITY_COMPONENTS_SCHEMA",
    "TestIdentityComponents",
    "canonicalize_pytest_parameter",
)
