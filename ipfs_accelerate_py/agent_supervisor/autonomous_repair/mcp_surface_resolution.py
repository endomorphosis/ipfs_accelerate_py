"""Resolve MCP tool surfaces for autonomous repair (package-agnostic roots)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .interface_alias_registry import InterfaceAliasRegistry, default_mcp_idl_alias_registry


@dataclass
class SurfaceHit:
    operation: str
    canonical: str
    status: str  # resolved | ambiguous | missing
    match_count: int
    effective_match_count: int = 0
    collapsed: bool = False
    collapse_reason: str = ""
    handler: str | None = None
    registration_api: str | None = None
    paths: tuple[str, ...] = ()
    preferred_path: str | None = None
    aliases_tried: tuple[str, ...] = ()
    provider: str = ""

    def __post_init__(self) -> None:
        if not self.effective_match_count:
            self.effective_match_count = (
                1 if self.collapsed and self.status == "resolved" else self.match_count
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SurfaceResolutionResult:
    hits: list[SurfaceHit] = field(default_factory=list)
    surface_files_scanned: list[str] = field(default_factory=list)
    unresolved_registrations_seen: int = 0
    providers_scanned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "surface_files_scanned": list(self.surface_files_scanned),
            "unresolved_registrations_seen": self.unresolved_registrations_seen,
            "providers_scanned": list(self.providers_scanned),
            "resolved_count": sum(1 for h in self.hits if h.status == "resolved"),
            "ambiguous_count": sum(1 for h in self.hits if h.status == "ambiguous"),
            "missing_count": sum(1 for h in self.hits if h.status == "missing"),
            "total": len(self.hits),
        }

    def by_operation(self) -> dict[str, SurfaceHit]:
        return {h.operation: h for h in self.hits}


def _tool_path(tool: Any) -> str:
    return str(getattr(getattr(tool, "registration_span", None), "path", "") or "")


def _tool_handler_symbol(tool: Any) -> str:
    return str(getattr(getattr(tool, "handler", None), "symbol", "") or "")


def _prefer_multi_path_surface(
    operation: str,
    tools: Sequence[Any],
    *,
    prefer_mcp_server: bool = True,
) -> tuple[Any | None, str]:
    """Collapse multi-match tool list to one preferred surface.

    Preference order (deterministic):
    1. Exact canonical_name / alias identity match to operation
    2. Handler symbol equals operation leaf
    3. Path under mcp_server + native_ tools
    4. registration_api contains register_tool
    5. Lexicographic path + handler as stable tie-break

    Returns ``(tool_or_none, reason)``.
    """
    if not tools:
        return None, "no_candidates"
    if len(tools) == 1:
        return tools[0], "single_candidate"

    op = str(operation or "").strip()
    leaf = op.rsplit(".", 1)[-1] if op else ""
    op_forms = {op, op.replace(".", "_"), leaf, leaf.replace(".", "_")}
    op_forms = {x for x in op_forms if x}

    ranked: list[tuple[tuple[int, str, str], Any]] = []
    for tool in tools:
        score = 0
        cname = str(getattr(tool, "canonical_name", "") or "")
        aliases = {str(a) for a in (getattr(tool, "aliases", ()) or ())}
        handler = _tool_handler_symbol(tool)
        path = _tool_path(tool)
        reg_api = str(getattr(tool, "registration_api", "") or "")

        if cname in op_forms:
            score += 100
        if op_forms & aliases:
            score += 80
        if handler and handler in op_forms:
            score += 70
        if handler and leaf and handler == leaf:
            score += 60
        if prefer_mcp_server and "mcp_server" in path:
            score += 40
        if "native_" in path:
            score += 20
        if "register_tool" in reg_api:
            score += 15
        # Prefer non-empty handler
        if handler:
            score += 5

        # Stable tie-break: higher score, then path, then handler
        ranked.append(((-score, path, handler), tool))

    ranked.sort(key=lambda item: item[0])
    best_key, best = ranked[0]
    best_score = -best_key[0]
    # Require a minimum preference signal for collapse of true multi-tools
    if best_score < 60 and len({_tool_handler_symbol(t) for t in tools}) > 1:
        # Multiple distinct handlers without strong identity match → still collapse
        # if all share one path under mcp_server (safe mediation preference).
        paths = {_tool_path(t) for t in tools if _tool_path(t)}
        if len(paths) == 1 and (not prefer_mcp_server or "mcp_server" in next(iter(paths))):
            return best, "same_path_prefer_mcp_server"
        return None, "ambiguous_distinct_handlers"

    return best, "prefer_mcp_server_handler"


# Default surface roots relative to a monorepo root (overridable by caller)
DEFAULT_SURFACE_GLOBS: tuple[tuple[str, str], ...] = (
    # (provider, relative path under monorepo)
    ("ipfs_accelerate_py", "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/server.py"),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/ipfs/native_ipfs_tools.py",
    ),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/backend_management_tools/native_backend_management_tools.py",
    ),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/embedding_tools/native_embedding_tools.py",
    ),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/ipfs_cluster_tools/native_ipfs_cluster_tools.py",
    ),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/workflow/native_workflow_tools.py",
    ),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/search_tools/native_search_tools.py",
    ),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/index_management_tools/native_index_management_tools.py",
    ),
    (
        "ipfs_accelerate_py",
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/tools/provenance_tools/native_provenance_tools.py",
    ),
)


def resolve_mcp_surfaces(
    operations: Sequence[str],
    *,
    repo_root: str | Path,
    surface_files: Sequence[tuple[str, str | Path]] | None = None,
    alias_registry: InterfaceAliasRegistry | None = None,
    prefer_mcp_server: bool = True,
) -> SurfaceResolutionResult:
    """Scan MCP Python sources and resolve each operation via aliases.

    ``surface_files`` is a sequence of ``(provider, path)`` pairs. Paths may be
    absolute or relative to ``repo_root``. When omitted, default accelerate
    package surfaces are used (still overridable — not SCA-only).
    """
    from ..analysis.python_mcp_surface_extractor import (
        _canonical_tool_name,
        extract_python_mcp_source,
    )
    from ..analysis.runtime_contract_evidence_compiler import (
        _collapse_equivalent_tool_surfaces,
    )

    root = Path(repo_root).resolve()
    registry = alias_registry or default_mcp_idl_alias_registry()
    files = list(surface_files or DEFAULT_SURFACE_GLOBS)

    tools_by_name: dict[str, list[Any]] = {}
    unresolved = 0
    scanned: list[str] = []
    providers: set[str] = set()

    for provider, rel in files:
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            continue
        try:
            rel_s = str(path.relative_to(root))
        except ValueError:
            rel_s = str(path)
        scanned.append(rel_s)
        providers.add(provider)
        text = path.read_text(encoding="utf-8", errors="ignore")
        surface = extract_python_mcp_source(text, provider=provider, path=rel_s)
        unresolved += len(getattr(surface, "unresolved", ()) or ())
        for tool in surface.tools:
            names = {getattr(tool, "canonical_name", "")}
            names.update(getattr(tool, "aliases", ()) or ())
            for name in names:
                if not name:
                    continue
                tools_by_name.setdefault(str(name), []).append(tool)
                # also index expanded aliases
                for alt in registry.expand(str(name)):
                    tools_by_name.setdefault(alt, []).append(tool)

    hits: list[SurfaceHit] = []
    for op in operations:
        exact_keys = {_canonical_tool_name(op), op, op.replace(".", "_")}
        exact_keys = {k for k in exact_keys if k}
        expanded_keys = sorted(registry.expand(op) | exact_keys)

        def _gather(key_set: set[str] | list[str]) -> list[Any]:
            matches: list[Any] = []
            for key in key_set:
                matches.extend(tools_by_name.get(key, []))
            seen: set[Any] = set()
            unique: list[Any] = []
            for m in matches:
                tid = getattr(m, "tool_id", None) or id(m)
                if tid in seen:
                    continue
                seen.add(tid)
                unique.append(m)
            return unique

        # Prefer exact-name matches before alias-expanded neighborhood.
        unique = _gather(exact_keys)
        keys_used = sorted(exact_keys)
        if not unique:
            unique = _gather(expanded_keys)
            keys_used = expanded_keys

        collapsed = _collapse_equivalent_tool_surfaces(unique)
        collapse_reason = ""
        selected = collapsed

        if selected is None and unique:
            selected, collapse_reason = _prefer_multi_path_surface(
                op,
                unique,
                prefer_mcp_server=prefer_mcp_server,
            )
        elif selected is not None:
            collapse_reason = "equivalent_tool_collapse"

        if selected is not None:
            status = "resolved"
            collapsed_flag = True
        elif len(unique) > 1:
            status = "ambiguous"
            collapsed_flag = False
        elif len(unique) == 1:
            status = "resolved"
            selected = unique[0]
            collapsed_flag = True
            collapse_reason = collapse_reason or "single_exact_match"
        else:
            status = "missing"
            collapsed_flag = False

        # Paths from the selected surface when collapsed; else all candidates.
        if selected is not None:
            path = str(
                getattr(getattr(selected, "registration_span", None), "path", "") or ""
            )
            paths = [path] if path else []
        else:
            paths = sorted(
                {
                    str(
                        getattr(getattr(m, "registration_span", None), "path", "") or ""
                    )
                    for m in unique
                    if getattr(getattr(m, "registration_span", None), "path", None)
                }
            )

        preferred = paths[0] if paths else None
        if prefer_mcp_server and paths and selected is None:
            preferred = registry.resolve_preferred(
                op,
                paths,
                prefer_tokens=("mcp_server", "register_tool", "native_"),
            ) or preferred

        handler = None
        reg_api = None
        if selected is not None:
            handler = getattr(getattr(selected, "handler", None), "symbol", None)
            reg_api = getattr(selected, "registration_api", None)

        hits.append(
            SurfaceHit(
                operation=op,
                canonical=_canonical_tool_name(op),
                status=status,
                match_count=len(unique),
                effective_match_count=1 if status == "resolved" else len(unique),
                collapsed=collapsed_flag and status == "resolved",
                collapse_reason=collapse_reason,
                handler=handler,
                registration_api=reg_api,
                paths=tuple(paths),
                preferred_path=preferred,
                aliases_tried=tuple(keys_used),
                provider=",".join(sorted(providers)),
            )
        )

    return SurfaceResolutionResult(
        hits=hits,
        surface_files_scanned=scanned,
        unresolved_registrations_seen=unresolved,
        providers_scanned=sorted(providers),
    )


__all__ = [
    "DEFAULT_SURFACE_GLOBS",
    "SurfaceHit",
    "SurfaceResolutionResult",
    "resolve_mcp_surfaces",
]
