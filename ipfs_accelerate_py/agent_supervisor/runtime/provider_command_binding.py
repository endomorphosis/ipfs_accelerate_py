"""Canonical provider-command symbol bindings for supervised runners.

Provider entry points such as :mod:`grok_cli_runner` must observe the sealed
provider-command environment.  When a runner references those symbols without
importing them, the failure is a ``NameError`` at dispatch time — after worktree
setup and attempt consumption.

This module:

1. Declares the **canonical** import source for each provider-command symbol.
2. **Infers** missing bindings from a module namespace or source text.
3. **Ensures** bindings on a live namespace (runtime self-heal).
4. Emits a typed diagnostic / residual import patch for doctor and residual LLM
   repair paths when a source-level fix is still required.

It does not invent new environment contracts: every symbol resolves to an
existing definition in ``provider_command_environment`` or
``validation_runtime``.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Final

PROVIDER_COMMAND_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-command-binding@1"
)
PROVIDER_COMMAND_BINDING_INTERFACE: Final = "ProviderCommandBinding@1"

# Canonical home for each public symbol a supervised runner may need.
# Values are (import_module, attribute_name).  attribute_name may equal the
# public symbol or a re-export alias.
CANONICAL_PROVIDER_COMMAND_BINDINGS: Final[Mapping[str, tuple[str, str]]] = {
    "PROVIDER_COMMAND_ENVIRONMENT_SCHEMA": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "PROVIDER_COMMAND_ENVIRONMENT_SCHEMA",
    ),
    "PROVIDER_COMMAND_ENV_WRAPPER_ENV": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "PROVIDER_COMMAND_ENV_WRAPPER_ENV",
    ),
    "PROVIDER_COMMAND_ENV_DIGEST_ENV": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "PROVIDER_COMMAND_ENV_DIGEST_ENV",
    ),
    "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
    ),
    "APPROVED_PROVIDER_COMMAND_ENVIRONMENT_NAMES": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "APPROVED_PROVIDER_COMMAND_ENVIRONMENT_NAMES",
    ),
    "ProviderCommandEnvironment": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "ProviderCommandEnvironment",
    ),
    "ProviderCommandEnvironmentError": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "ProviderCommandEnvironmentError",
    ),
    "sealed_provider_command_environment": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "sealed_provider_command_environment",
    ),
    "project_provider_command_environment": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "project_provider_command_environment",
    ),
    "provider_command_environment_sha256": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "provider_command_environment_sha256",
    ),
    "normalize_required_commands": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "normalize_required_commands",
    ),
    "preflight_required_commands": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "preflight_required_commands",
    ),
    # Re-exported through provider_command_environment (imported there from
    # validation_runtime).  Prefer the command-environment surface so runners
    # only need one binding module.
    "FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV",
    ),
    "FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV": (
        "ipfs_accelerate_py.agent_supervisor.provider_command_environment",
        "FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV",
    ),
}

_NAME_ERROR_RE = re.compile(
    r"name '(?P<name>[A-Za-z_][A-Za-z0-9_]*)' is not defined"
)


class ProviderCommandBindingError(RuntimeError):
    """Raised when a required provider-command symbol cannot be resolved."""


@dataclass(frozen=True)
class ProviderCommandBindingFix:
    """One inferred import that restores a missing provider-command symbol."""

    symbol: str
    module: str
    attribute: str
    import_statement: str

    def to_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "module": self.module,
            "attribute": self.attribute,
            "import_statement": self.import_statement,
        }


@dataclass
class ProviderCommandBindingReport:
    """Result of ensuring or scanning provider-command bindings."""

    schema: str = PROVIDER_COMMAND_BINDING_SCHEMA
    interface: str = PROVIDER_COMMAND_BINDING_INTERFACE
    namespace_name: str = ""
    already_bound: list[str] = field(default_factory=list)
    bound_now: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    inferred_fixes: list[ProviderCommandBindingFix] = field(default_factory=list)
    unknown_symbols: list[str] = field(default_factory=list)
    healed: bool = False

    @property
    def complete(self) -> bool:
        return not self.missing and not self.unknown_symbols

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "namespace_name": self.namespace_name,
            "already_bound": list(self.already_bound),
            "bound_now": list(self.bound_now),
            "missing": list(self.missing),
            "unknown_symbols": list(self.unknown_symbols),
            "healed": self.healed,
            "complete": self.complete,
            "inferred_fixes": [fix.to_dict() for fix in self.inferred_fixes],
        }


def is_provider_command_symbol(name: str) -> bool:
    """Return True when *name* is a known provider-command binding symbol."""

    return str(name) in CANONICAL_PROVIDER_COMMAND_BINDINGS


def extract_name_error_symbol(exc: BaseException | str) -> str:
    """Extract the undefined name from a NameError message or exception."""

    if isinstance(exc, NameError):
        # Python 3.11+ may set name attribute
        attr = getattr(exc, "name", None)
        if isinstance(attr, str) and attr:
            return attr
        text = str(exc)
    else:
        text = str(exc)
    match = _NAME_ERROR_RE.search(text)
    if not match:
        return ""
    return str(match.group("name") or "")


def resolve_provider_command_symbol(symbol: str) -> Any:
    """Import and return the canonical object for *symbol*."""

    key = str(symbol)
    if key not in CANONICAL_PROVIDER_COMMAND_BINDINGS:
        raise ProviderCommandBindingError(
            f"unknown provider-command symbol: {key!r}"
        )
    module_name, attr = CANONICAL_PROVIDER_COMMAND_BINDINGS[key]
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        # FORMAL_TOOLCHAIN_* may only live on validation_runtime in some pins;
        # fall back explicitly.
        if key.startswith("FORMAL_TOOLCHAIN_"):
            runtime = importlib.import_module(
                "ipfs_accelerate_py.agent_supervisor.validation.validation_runtime"
            )
            try:
                return getattr(runtime, attr)
            except AttributeError:
                pass
        raise ProviderCommandBindingError(
            f"canonical module {module_name!r} has no attribute {attr!r}"
        ) from exc


def infer_provider_command_import(symbol: str) -> ProviderCommandBindingFix:
    """Infer the exact import statement that binds *symbol*."""

    key = str(symbol)
    if key not in CANONICAL_PROVIDER_COMMAND_BINDINGS:
        raise ProviderCommandBindingError(
            f"cannot infer import for unknown symbol: {key!r}"
        )
    module_name, attr = CANONICAL_PROVIDER_COMMAND_BINDINGS[key]
    if attr == key:
        statement = f"from {module_name} import {attr}"
    else:
        statement = f"from {module_name} import {attr} as {key}"
    return ProviderCommandBindingFix(
        symbol=key,
        module=module_name,
        attribute=attr,
        import_statement=statement,
    )


def infer_provider_command_imports(
    symbols: Iterable[str],
) -> list[ProviderCommandBindingFix]:
    """Infer import statements for many symbols, grouped stably by module."""

    fixes: list[ProviderCommandBindingFix] = []
    seen: set[str] = set()
    for symbol in symbols:
        key = str(symbol)
        if not key or key in seen:
            continue
        if key not in CANONICAL_PROVIDER_COMMAND_BINDINGS:
            continue
        seen.add(key)
        fixes.append(infer_provider_command_import(key))
    # Stable order by module then symbol for deterministic patches.
    fixes.sort(key=lambda item: (item.module, item.symbol))
    return fixes


def group_import_statements(fixes: Sequence[ProviderCommandBindingFix]) -> list[str]:
    """Collapse per-symbol fixes into one ``from module import a, b`` per module."""

    by_module: dict[str, list[str]] = {}
    for fix in fixes:
        by_module.setdefault(fix.module, []).append(fix.attribute)
    statements: list[str] = []
    for module in sorted(by_module):
        names = sorted(set(by_module[module]))
        statements.append(f"from {module} import {', '.join(names)}")
    return statements


def scan_source_for_provider_command_names(source: str) -> frozenset[str]:
    """Return provider-command symbols loaded by name in *source* (AST)."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to lexical scan for broken intermediate sources.
        found = {
            name
            for name in CANONICAL_PROVIDER_COMMAND_BINDINGS
            if re.search(rf"\b{re.escape(name)}\b", source)
        }
        return frozenset(found)

    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in CANONICAL_PROVIDER_COMMAND_BINDINGS:
                loaded.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            # module.FORMAL_TOOLCHAIN... not needed for bare NameError class
            pass
    return frozenset(loaded)


def missing_provider_command_bindings_in_source(
    source: str,
    *,
    defined_names: Iterable[str] | None = None,
) -> list[str]:
    """Return used provider-command names that are not defined in *source*."""

    used = scan_source_for_provider_command_names(source)
    defined: set[str] = set(defined_names or ())
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
    return sorted(name for name in used if name not in defined)


def ensure_provider_command_bindings(
    namespace: MutableMapping[str, Any],
    *,
    required: Iterable[str] | None = None,
    namespace_name: str = "",
    strict: bool = False,
) -> ProviderCommandBindingReport:
    """Bind missing canonical provider-command symbols into *namespace*.

    When *required* is None, every canonical symbol is considered required for
    a complete provider-command surface.  Callers that only need a subset
    should pass the subset used by their module (e.g. from
    :func:`scan_source_for_provider_command_names`).
    """

    report = ProviderCommandBindingReport(namespace_name=namespace_name or "")
    want = (
        sorted(CANONICAL_PROVIDER_COMMAND_BINDINGS)
        if required is None
        else sorted({str(item) for item in required if str(item)})
    )
    for symbol in want:
        if symbol not in CANONICAL_PROVIDER_COMMAND_BINDINGS:
            report.unknown_symbols.append(symbol)
            continue
        if symbol in namespace and namespace[symbol] is not None:
            report.already_bound.append(symbol)
            continue
        try:
            namespace[symbol] = resolve_provider_command_symbol(symbol)
            report.bound_now.append(symbol)
        except ProviderCommandBindingError:
            report.missing.append(symbol)
            try:
                report.inferred_fixes.append(infer_provider_command_import(symbol))
            except ProviderCommandBindingError:
                pass
    report.healed = bool(report.bound_now) and not report.missing
    if strict and not report.complete:
        raise ProviderCommandBindingError(
            "provider-command bindings incomplete: "
            f"missing={report.missing} unknown={report.unknown_symbols}"
        )
    return report


def ensure_provider_command_bindings_for_module(
    module: ModuleType,
    *,
    required: Iterable[str] | None = None,
    strict: bool = False,
) -> ProviderCommandBindingReport:
    """Ensure bindings on a loaded module's ``__dict__``."""

    if required is None and getattr(module, "__file__", None):
        try:
            source = open(module.__file__, encoding="utf-8").read()  # noqa: SIM115
            required = scan_source_for_provider_command_names(source)
        except OSError:
            required = None
    return ensure_provider_command_bindings(
        module.__dict__,
        required=required,
        namespace_name=getattr(module, "__name__", "") or "",
        strict=strict,
    )


def recover_provider_command_name_error(
    exc: BaseException,
    namespace: MutableMapping[str, Any],
) -> ProviderCommandBindingReport | None:
    """If *exc* is a provider-command NameError, bind the symbol and report.

    Returns ``None`` when the error is not a known provider-command symbol.
    """

    if not isinstance(exc, NameError):
        return None
    symbol = extract_name_error_symbol(exc)
    if not symbol or not is_provider_command_symbol(symbol):
        return None
    return ensure_provider_command_bindings(
        namespace,
        required=(symbol,),
        namespace_name="<name-error-recovery>",
        strict=False,
    )


def preflight_provider_entry_module(
    module_name: str = "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    *,
    reload: bool = False,
) -> ProviderCommandBindingReport:
    """Import a provider entry module and ensure its provider-command bindings.

    This is intended for multi-supervisor / implementation preflight so
    missing imports fail before a worktree is leased.
    """

    if reload and module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)
    report = ensure_provider_command_bindings_for_module(module, strict=False)
    if not report.complete and report.missing:
        # One more pass after heal for partial reloads
        report = ensure_provider_command_bindings_for_module(
            module,
            required=report.missing,
            strict=False,
        )
    if not report.complete:
        raise ProviderCommandBindingError(
            f"provider entry {module_name!r} missing bindings: "
            f"{report.missing or report.unknown_symbols}; "
            f"inferred_imports={group_import_statements(report.inferred_fixes)}"
        )
    return report


def residual_import_patch_for_report(
    report: ProviderCommandBindingReport,
) -> str:
    """Render a residual source patch of inferred imports for *report*."""

    lines = group_import_statements(report.inferred_fixes)
    if not lines:
        return ""
    header = (
        "# Auto-inferred provider-command bindings "
        f"(schema={PROVIDER_COMMAND_BINDING_SCHEMA})\n"
    )
    return header + "\n".join(lines) + "\n"


__all__ = [
    "CANONICAL_PROVIDER_COMMAND_BINDINGS",
    "PROVIDER_COMMAND_BINDING_INTERFACE",
    "PROVIDER_COMMAND_BINDING_SCHEMA",
    "ProviderCommandBindingError",
    "ProviderCommandBindingFix",
    "ProviderCommandBindingReport",
    "ensure_provider_command_bindings",
    "ensure_provider_command_bindings_for_module",
    "extract_name_error_symbol",
    "group_import_statements",
    "infer_provider_command_import",
    "infer_provider_command_imports",
    "is_provider_command_symbol",
    "missing_provider_command_bindings_in_source",
    "preflight_provider_entry_module",
    "recover_provider_command_name_error",
    "residual_import_patch_for_report",
    "resolve_provider_command_symbol",
    "scan_source_for_provider_command_names",
]
