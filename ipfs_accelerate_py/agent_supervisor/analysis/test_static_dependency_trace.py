"""Static dependency tracing over analysis AST indexes (PTR-020 surface)."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    mint_content_identity,
)

STATIC_TEST_DEPENDENCY_TRACE_INTERFACE: Final = "StaticTestDependencyTrace@1"
STATIC_TEST_DEPENDENCY_TRACER_INTERFACE: Final = "StaticTestDependencyTracer@1"
STATIC_TEST_DEPENDENCY_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/static-test-dependency-trace@1"
)


@dataclass(frozen=True, slots=True)
class StaticUnknownFrontierEntry:
    """One residual unknown or effect frontier observed during static analysis."""

    kind: str
    frontier_id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind or "").strip() or "unknown")
        object.__setattr__(
            self,
            "frontier_id",
            str(self.frontier_id or self.kind).strip()[:128],
        )
        object.__setattr__(self, "detail", str(self.detail or "")[:256])


@dataclass(frozen=True, slots=True)
class StaticTestDependencyTrace:
    """Content-addressed static dependency closure for one test symbol."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = STATIC_TEST_DEPENDENCY_TRACE_INTERFACE

    complete: bool
    trace_cid: str
    node_id: str = ""
    relative_path: str = ""
    test_symbol: str = ""
    unknown_frontier: tuple[StaticUnknownFrontierEntry, ...] = ()
    retained_canonical_bytes: bytes = b""
    dependencies: Mapping[str, Any] = field(default_factory=dict)

    def verify(self) -> None:
        if not self.trace_cid:
            raise ValueError("static trace is missing trace_cid")
        if self.retained_canonical_bytes:
            expected = mint_content_identity_from_bytes(self.retained_canonical_bytes)
            if expected != self.trace_cid:
                raise ValueError("static trace CID does not match retained bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATIC_TEST_DEPENDENCY_TRACE_SCHEMA,
            "interface": STATIC_TEST_DEPENDENCY_TRACE_INTERFACE,
            "complete": self.complete,
            "trace_cid": self.trace_cid,
            "node_id": self.node_id,
            "relative_path": self.relative_path,
            "test_symbol": self.test_symbol,
            "unknown_frontier": [
                {
                    "kind": entry.kind,
                    "frontier_id": entry.frontier_id,
                    "detail": entry.detail,
                }
                for entry in self.unknown_frontier
            ],
            "dependencies": dict(self.dependencies),
        }


def mint_content_identity_from_bytes(data: bytes) -> str:
    from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
        mint_content_identity_bytes,
    )

    return mint_content_identity_bytes(data).cid


class StaticTestDependencyTracer:
    """Build a bounded static dependency trace from an analysis AST index."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = STATIC_TEST_DEPENDENCY_TRACER_INTERFACE

    def __init__(self, index: Any, root_path: str | Path) -> None:
        self.index = index
        self.root_path = Path(root_path)

    def trace(
        self,
        relative_path: str,
        *,
        test_symbol: str,
        node_id: str = "",
    ) -> StaticTestDependencyTrace:
        rel = str(relative_path or "").replace("\\", "/").lstrip("./")
        symbol = str(test_symbol or "").strip()
        node = str(node_id or f"{rel}::{symbol}")
        frontiers: list[StaticUnknownFrontierEntry] = []
        edges: list[dict[str, Any]] = []
        imports: list[str] = []
        source = ""
        path = self.root_path / rel
        if not path.is_file():
            frontiers.append(
                StaticUnknownFrontierEntry(
                    kind="missing_file",
                    frontier_id=f"missing:{rel}",
                    detail=rel,
                )
            )
        else:
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                frontiers.append(
                    StaticUnknownFrontierEntry(
                        kind="missing_file",
                        frontier_id=f"unreadable:{rel}",
                        detail=type(exc).__name__,
                    )
                )
                source = ""
        module_ast = None
        if source:
            try:
                module_ast = ast.parse(source, filename=rel)
            except SyntaxError as exc:
                frontiers.append(
                    StaticUnknownFrontierEntry(
                        kind="parse_error",
                        frontier_id=f"parse:{rel}",
                        detail=str(exc.msg)[:128],
                    )
                )
        found_symbol = False
        if module_ast is not None:
            for node_ast in module_ast.body:
                if isinstance(node_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node_ast.name == symbol:
                        found_symbol = True
                elif isinstance(node_ast, ast.ClassDef):
                    for child in node_ast.body:
                        if (
                            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and child.name == symbol
                        ):
                            found_symbol = True
                if isinstance(node_ast, (ast.Import, ast.ImportFrom)):
                    for alias in node_ast.names:
                        name = alias.name if isinstance(node_ast, ast.Import) else (
                            f"{node_ast.module or ''}.{alias.name}".strip(".")
                        )
                        if name:
                            imports.append(name)
                            edges.append(
                                {
                                    "kind": "import",
                                    "target": name,
                                    "source": rel,
                                }
                            )
            # Conservative effect heuristics for common impure builtins.
            effect_names = {
                "open": "filesystem",
                "print": "filesystem",
                "input": "environment",
                "system": "subprocess",
                "Popen": "subprocess",
                "run": "subprocess",
                "urlopen": "network",
                "request": "network",
                "sleep": "clock",
                "time": "clock",
                "random": "randomness",
                "getenv": "environment",
                "environ": "environment",
            }
            for ast_node in ast.walk(module_ast):
                name = ""
                if isinstance(ast_node, ast.Name):
                    name = ast_node.id
                elif isinstance(ast_node, ast.Attribute):
                    name = ast_node.attr
                if name in effect_names:
                    effect = effect_names[name]
                    edges.append(
                        {
                            "kind": "effect",
                            "target": effect,
                            "target_symbol": effect,
                            "source": rel,
                            "symbol": name,
                        }
                    )
            if symbol and not found_symbol:
                frontiers.append(
                    StaticUnknownFrontierEntry(
                        kind="missing_test_symbol",
                        frontier_id=f"symbol:{symbol}",
                        detail=symbol,
                    )
                )

        # Prefer index metadata when available (stale detection hook).
        try:
            records = getattr(self.index, "records", None) or getattr(
                self.index, "modules", None
            )
            if records is not None and rel not in records and not source:
                frontiers.append(
                    StaticUnknownFrontierEntry(
                        kind="stale_ast_index",
                        frontier_id=f"index:{rel}",
                        detail="relative path not present in analysis index",
                    )
                )
        except Exception:
            pass

        payload = {
            "schema": STATIC_TEST_DEPENDENCY_TRACE_SCHEMA,
            "interface": STATIC_TEST_DEPENDENCY_TRACE_INTERFACE,
            "node_id": node,
            "relative_path": rel,
            "test_symbol": symbol,
            "imports": sorted(set(imports)),
            "unknown_frontier": [
                {
                    "kind": entry.kind,
                    "frontier_id": entry.frontier_id,
                    "detail": entry.detail,
                }
                for entry in frontiers
            ],
            "dependencies": {"edges": edges},
            "source_sha256": (
                mint_content_identity(
                    {
                        "schema": STATIC_TEST_DEPENDENCY_TRACE_SCHEMA + "/source",
                        "relative_path": rel,
                        "source": source,
                    }
                ).digest
                if source
                else ""
            ),
        }
        identity = mint_content_identity(payload)
        complete = not frontiers
        return StaticTestDependencyTrace(
            complete=complete,
            trace_cid=identity.cid,
            node_id=node,
            relative_path=rel,
            test_symbol=symbol,
            unknown_frontier=tuple(frontiers),
            retained_canonical_bytes=identity.canonical_bytes,
            dependencies=MappingProxyType({"edges": edges}),
        )


__all__ = (
    "STATIC_TEST_DEPENDENCY_TRACE_INTERFACE",
    "STATIC_TEST_DEPENDENCY_TRACER_INTERFACE",
    "StaticTestDependencyTrace",
    "StaticTestDependencyTracer",
    "StaticUnknownFrontierEntry",
)
