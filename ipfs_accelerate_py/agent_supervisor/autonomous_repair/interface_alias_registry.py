"""Interface / ORB / MCP-IDL name alias registry (domain-agnostic).

Maps GUI, ORB, MCP-IDL, and package MCP tool names onto a shared identity so
autonomous repair can align SwissKnife (or any UI) with backend MCP servers
without hard-coding a single taskboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence


INTERFACE_ALIAS_REGISTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/interface-alias-registry@1"
)


def _norm(name: str) -> str:
    s = str(name or "").strip()
    s = s.replace("-", "_")
    return s


def _leaf(name: str) -> str:
    s = _norm(name)
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return s


@dataclass
class InterfaceAliasRegistry:
    """Bidirectional alias graph for interface / MCP tool names."""

    aliases: dict[str, set[str]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    # optional method inventories from IDL descriptors
    idl_methods: set[str] = field(default_factory=set)
    registry_id: str = "interface-alias-default@1"

    def add_alias(
        self,
        left: str,
        right: str,
        *,
        source: str = "declared",
    ) -> None:
        a, b = _norm(left), _norm(right)
        if not a or not b or a == b:
            return
        self.aliases.setdefault(a, set()).add(b)
        self.aliases.setdefault(b, set()).add(a)
        self.sources[f"{a}|{b}"] = source
        self.sources[f"{b}|{a}"] = source

    def add_aliases(
        self,
        mapping: Mapping[str, str | Sequence[str]],
        *,
        source: str = "declared",
    ) -> None:
        for left, right in mapping.items():
            if isinstance(right, (list, tuple, set)):
                for r in right:
                    self.add_alias(left, str(r), source=source)
            else:
                self.add_alias(left, str(right), source=source)

    def add_idl_methods(self, methods: Iterable[str], *, source: str = "idl") -> None:
        for m in methods:
            nm = _norm(m)
            if not nm:
                continue
            self.idl_methods.add(nm)
            # leaf equivalence: foo.bar ↔ bar when bar is an IDL method
            leaf = _leaf(nm)
            if leaf != nm:
                self.add_alias(nm, leaf, source=source)

    def expand(self, name: str) -> set[str]:
        """Return name plus all registered aliases (1-hop + leaf forms)."""
        root = _norm(name)
        out: set[str] = {root, _leaf(root)} if root else set()
        if not root:
            return out
        # BFS a few hops
        frontier = {root, _leaf(root)}
        seen = set(frontier)
        for _ in range(4):
            nxt: set[str] = set()
            for node in frontier:
                for neigh in self.aliases.get(node, ()):
                    if neigh not in seen:
                        seen.add(neigh)
                        nxt.add(neigh)
                leaf = _leaf(node)
                if leaf not in seen:
                    seen.add(leaf)
                    nxt.add(leaf)
            if not nxt:
                break
            frontier = nxt
        out |= seen
        return {x for x in out if x}

    def match_idl(self, name: str) -> list[str]:
        """IDL methods covered by this name's alias closure."""
        exp = self.expand(name)
        hits = sorted(m for m in self.idl_methods if m in exp or _leaf(m) in exp)
        return hits

    def resolve_preferred(
        self,
        name: str,
        candidates: Sequence[str],
        *,
        prefer_tokens: Sequence[str] = ("mcp_server", "register_tool"),
    ) -> str | None:
        """Pick a preferred candidate string using alias + token preference."""
        exp = self.expand(name)
        ranked: list[tuple[int, str]] = []
        for c in candidates:
            score = 0
            cn = _norm(c)
            if cn in exp or _leaf(cn) in exp:
                score += 10
            for tok in prefer_tokens:
                if tok in c:
                    score += 3
            if score:
                ranked.append((score, c))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INTERFACE_ALIAS_REGISTRY_SCHEMA,
            "registry_id": self.registry_id,
            "alias_edge_count": sum(len(v) for v in self.aliases.values()) // 2,
            "idl_method_count": len(self.idl_methods),
            "idl_methods": sorted(self.idl_methods),
            "edges": sorted(
                {
                    tuple(sorted((a, b)))
                    for a, bs in self.aliases.items()
                    for b in bs
                }
            ),
        }


def default_mcp_idl_alias_registry(
    *,
    extra_idl_methods: Iterable[str] = (),
    extra_aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> InterfaceAliasRegistry:
    """Default aliases for MCP package tools ↔ common IDL/ORB names.

    Not SwissKnife-specific: encodes cross-package naming conventions used by
    IPFS kit/accelerate/datasets MCP surfaces and typical IDL method leaves.
    """
    reg = InterfaceAliasRegistry(registry_id="mcp-idl-orb-aliases@1")

    # Dotted kit-style ↔ snake accelerate-style
    base = {
        "ipfs.add": ("ipfs_add", "add", "ipfs_files_add_file"),
        "ipfs.cat": ("ipfs_cat", "cat", "ipfs_files_cat"),
        "ipfs.pin": ("ipfs_pin_add", "pin"),
        "ipfs.dag.put": ("dag_put", "ipfs_dag_put"),
        "ipfs.dag.get": ("dag_get", "ipfs_dag_get"),
        "ipfs_add": ("add", "ipfs.add"),
        "ipfs_cat": ("cat", "ipfs.cat"),
        "dag_put": ("ipfs.dag.put", "ipfs_dag_put"),
        "dag_get": ("ipfs.dag.get", "ipfs_dag_get"),
        "semantic_search": ("faceted_search", "search"),
        "tools_dispatch": ("tools.dispatch", "mcp.tools_dispatch", "tools/call"),
        "tools.dispatch": ("tools_dispatch",),
        "load_index": ("index_load", "loadIndex"),
        "record_provenance": ("provenance_record", "recordProvenance"),
        "WorkflowCoordinator.submit_task": (
            "workflow_coordinator_submit_task",
            "submit_task",
            "workflow.submit_task",
        ),
        "mcpplusplus.check_compatibility": (
            "check_compatibility",
            "mcp_check_compatibility",
            "compatibility.check",
        ),
        "get_backend_status": ("backend_get_status", "backend.status"),
        "list_pins": ("ipfs_pin_ls", "pin.ls"),
    }
    reg.add_aliases(base, source="default_mcp_idl")

    # Common IDL leaf inventory (generic; consumers may extend)
    default_idl = {
        "add",
        "cat",
        "pin",
        "unpin",
        "dag_put",
        "dag_get",
        "embed",
        "generate",
        "inference",
        "faceted_search",
        "semantic_search",
        "capabilities",
        "endpoints",
        "hardware_profile",
    }
    reg.add_idl_methods(default_idl, source="default_idl_inventory")
    reg.add_idl_methods(extra_idl_methods, source="consumer_idl")
    if extra_aliases:
        reg.add_aliases(extra_aliases, source="consumer")
    return reg


def load_idl_methods_from_typescript(text: str) -> list[str]:
    """Extract ``name: 'method'`` entries from TS InterfaceDescriptor sources."""
    names = re.findall(r"name:\s*'([^']+)'", text or "")
    # Drop SCREAMING_ERROR_CODES
    return [n for n in names if not (n.isupper() and "_" in n)]


__all__ = [
    "INTERFACE_ALIAS_REGISTRY_SCHEMA",
    "InterfaceAliasRegistry",
    "default_mcp_idl_alias_registry",
    "load_idl_methods_from_typescript",
]
