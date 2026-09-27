"""Apply nominated import rewrites with the CST parser, then stage them.

The SPAR-021 receipt stays nomination-only. This module is the apply step:
parso locates import modules, comments outside those tokens stay put, and the
rewritten sources are staged through the VFS outbox and world-root publisher.
Nothing here writes the repository or CAS the current root.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json

from .cst_world_handoff import CstWorldHandoff, stage_cst_world_root


def _offset(source: str, line: int, col: int) -> int:
    current = 1
    index = 0
    while current < line:
        newline = source.find("\n", index)
        if newline < 0:
            return len(source)
        index = newline + 1
        current += 1
    return min(len(source), index + col)


def _module_name(node: Any) -> str:
    kind = getattr(node, "type", None)
    if kind == "name":
        return str(getattr(node, "value", "") or "")
    if kind == "dotted_name":
        parts: list[str] = []
        for child in getattr(node, "children", ()) or ():
            if getattr(child, "type", None) == "name":
                parts.append(str(child.value))
            elif getattr(child, "value", None) == ".":
                parts.append(".")
        return "".join(parts)
    return ""


def _imports_symbol(node: Any, symbol: str) -> bool:
    for child in getattr(node, "children", ()) or ():
        kind = getattr(child, "type", None)
        if kind == "name" and child.value == symbol:
            return True
        if kind in {"import_as_names", "import_as_name"}:
            if _imports_symbol(child, symbol):
                return True
    return False


def _module_node(node: Any) -> Any | None:
    for child in getattr(node, "children", ()) or ():
        if getattr(child, "type", None) in {"name", "dotted_name"}:
            return child
    return None


def _replacements(source: str, rewrites: Sequence[Any]) -> list[tuple[int, int, str]]:
    import parso

    tree = parso.parse(source)
    found: list[tuple[int, int, str]] = []

    def walk(node: Any) -> None:
        kind = getattr(node, "type", None)
        if kind in {"import_from", "import_name"}:
            module = _module_node(node)
            module_name = _module_name(module) if module is not None else ""
            for rewrite in rewrites:
                rewrite_kind = str(
                    getattr(getattr(rewrite, "rewrite_kind", ""), "value", None)
                    or getattr(rewrite, "rewrite_kind", "")
                )
                if rewrite_kind != "import":
                    continue
                if module_name != str(getattr(rewrite, "source_module", "") or ""):
                    continue
                symbol = str(getattr(rewrite, "symbol_id", "") or "")
                if kind == "import_from" and symbol and not _imports_symbol(node, symbol):
                    continue
                if module is None:
                    continue
                start = _offset(source, *module.start_pos)
                end = _offset(source, *module.end_pos)
                found.append(
                    (start, end, str(getattr(rewrite, "destination_module", "") or ""))
                )
        for child in getattr(node, "children", ()) or ():
            walk(child)

    walk(tree)
    return found


def _callsite_replacements(source: str, rewrites: Sequence[Any]) -> list[tuple[int, int, str]]:
    """Rename ``source_module.symbol`` callsites. The symbol token stays."""

    import parso

    pairs: list[tuple[str, str, str]] = []
    for rewrite in rewrites:
        kind = str(
            getattr(getattr(rewrite, "rewrite_kind", ""), "value", None)
            or getattr(rewrite, "rewrite_kind", "")
        )
        if kind != "callsite":
            continue
        src = str(getattr(rewrite, "source_module", "") or "")
        dst = str(getattr(rewrite, "destination_module", "") or "")
        symbol = str(getattr(rewrite, "symbol_id", "") or "")
        if src and dst and src != dst:
            pairs.append((src, dst, symbol))
    if not pairs:
        return []
    tree = parso.parse(source)
    found: list[tuple[int, int, str]] = []

    def pieces_of(node: Any) -> list[Any]:
        children = list(getattr(node, "children", ()) or ())
        if not children or getattr(children[0], "type", None) != "name":
            return []
        pieces = [children[0]]
        for trailer in children[1:]:
            kids = list(getattr(trailer, "children", ()) or ())
            if (
                len(kids) == 2
                and getattr(kids[0], "value", None) == "."
                and getattr(kids[1], "type", None) == "name"
            ):
                pieces.append(kids[1])
                continue
            break
        return pieces

    def walk(node: Any) -> None:
        if getattr(node, "type", None) == "atom_expr":
            pieces = pieces_of(node)
            if pieces:
                text = ".".join(str(piece.value) for piece in pieces)
                for src, dst, symbol in pairs:
                    needle = f"{src}.{symbol}" if symbol else src
                    if text == needle or text.startswith(needle + "."):
                        start = _offset(source, *pieces[0].start_pos)
                        end = _offset(source, *pieces[src.count(".")].end_pos)
                        if source[start:end] == src:
                            found.append((start, end, dst))
                        break
        for child in getattr(node, "children", ()) or ():
            walk(child)

    walk(tree)
    return found


def rewrite_imports_with_cst(source: str, rewrites: Sequence[Any]) -> tuple[str, bool]:
    """Rewrite import modules and callsites. Comments outside those tokens stay."""

    pieces = _replacements(source, rewrites) + _callsite_replacements(source, rewrites)
    if not pieces:
        return source, False
    text = source
    for start, end, new in sorted(pieces, reverse=True):
        text = text[:start] + new + text[end:]
    return text, text != source


def _existing_imports(source: str) -> set[tuple[str, str]]:
    import parso

    found: set[tuple[str, str]] = set()
    tree = parso.parse(source)

    def walk(node: Any) -> None:
        if getattr(node, "type", None) == "import_from":
            module = _module_node(node)
            module_name = _module_name(module) if module is not None else ""
            for child in getattr(node, "children", ()) or ():
                if getattr(child, "type", None) == "name" and child is not module:
                    found.add((module_name, str(child.value)))
                elif getattr(child, "type", None) in {"import_as_names", "import_as_name"}:
                    for name in getattr(child, "children", ()) or ():
                        if getattr(name, "type", None) == "name":
                            found.add((module_name, str(name.value)))
                        elif getattr(name, "type", None) == "import_as_name":
                            kids = list(getattr(name, "children", ()) or ())
                            if kids and getattr(kids[0], "type", None) == "name":
                                found.add((module_name, str(kids[0].value)))
        for child in getattr(node, "children", ()) or ():
            walk(child)

    walk(tree)
    return found


def _path_matches_module(path: str, module: str) -> bool:
    leaf = module.split(".")[-1] + ".py"
    dotted = module.replace(".", "/") + ".py"
    return path == leaf or path.endswith("/" + leaf) or path.endswith(dotted)


def insert_reexports_with_cst(source: str, plans: Sequence[Any]) -> tuple[str, bool]:
    """Insert authorized re-exports before the first statement. Comments stay."""

    import parso

    existing = _existing_imports(source)
    lines: list[str] = []
    for plan in plans:
        destination = str(getattr(plan, "destination_module", "") or "")
        symbols = tuple(getattr(plan, "symbol_ids", ()) or ())
        missing = [symbol for symbol in symbols if (destination, symbol) not in existing]
        if destination and missing:
            lines.append(f"from {destination} import {', '.join(missing)}")
    if not lines:
        return source, False
    insert = "\n".join(lines) + "\n"
    tree = parso.parse(source)
    children = [
        child
        for child in getattr(tree, "children", ()) or ()
        if getattr(child, "start_pos", None) is not None
    ]
    if not children:
        return insert + source, True
    start = _offset(source, *children[0].start_pos)
    prefix = source[:start]
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    return prefix + insert + source[start:], True


def apply_reexport_cst_and_stage(
    packet: Any,
    raw_sources: Mapping[str, str],
    *,
    plans: Sequence[Any],
    pre_world_root_cid: str | None = None,
    expected_generation: int = 1,
    kit_store: Any | None = None,
) -> CstWorldHandoff:
    """Insert re-exports into the source module and stage the CST result."""

    rewritten: dict[str, str] = {}
    for plan in plans:
        module = str(getattr(plan, "source_module", "") or "")
        paths = tuple(getattr(plan, "write_paths", ()) or ())
        targets = [path for path in paths if path in raw_sources and _path_matches_module(path, module)]
        if not targets and len(paths) == 1 and paths[0] in raw_sources:
            targets = [paths[0]]
        for path in targets:
            updated, changed = insert_reexports_with_cst(raw_sources[path], (plan,))
            if changed:
                rewritten[path] = updated
    if not rewritten:
        return CstWorldHandoff(
            applied=False,
            blocked=True,
            reason="reexport_not_applied",
            outbox_nomination_cid="",
            publication_status="",
            semantic_world_root_cid="",
        )
    tree_id = str(getattr(packet, "tree_id", "") or "")
    result = SimpleNamespace(
        tree_id=tree_id,
        result_cid=cid_for_dag_json({"tree_id": tree_id, "sources": rewritten}),
        mutated=False,
        writes_repository=False,
        libcst_usable=False,
        can_authorize_completion=False,
        source_maps=tuple(rewritten),
        sources=rewritten,
    )
    return stage_cst_world_root(
        result,
        expected_generation=expected_generation,
        pre_world_root_cid=pre_world_root_cid,
        kit_store=kit_store,
    )


def apply_import_cst_and_stage(
    packet: Any,
    raw_sources: Mapping[str, str],
    *,
    rewrites: Sequence[Any],
    pre_world_root_cid: str | None = None,
    expected_generation: int = 1,
    kit_store: Any | None = None,
) -> CstWorldHandoff:
    """Apply nominated import rewrites and stage the CST result."""

    rewritten: dict[str, str] = {}
    for path, source in raw_sources.items():
        updated, changed = rewrite_imports_with_cst(source, rewrites)
        if changed:
            rewritten[path] = updated
    if not rewritten:
        return CstWorldHandoff(
            applied=False,
            blocked=True,
            reason="import_not_rewritten",
            outbox_nomination_cid="",
            publication_status="",
            semantic_world_root_cid="",
        )
    tree_id = str(getattr(packet, "tree_id", "") or "")
    result = SimpleNamespace(
        tree_id=tree_id,
        result_cid=cid_for_dag_json(
            {"tree_id": tree_id, "sources": rewritten}
        ),
        mutated=False,
        writes_repository=False,
        libcst_usable=False,
        can_authorize_completion=False,
        source_maps=tuple(rewritten),
        sources=rewritten,
    )
    return stage_cst_world_root(
        result,
        expected_generation=expected_generation,
        pre_world_root_cid=pre_world_root_cid,
        kit_store=kit_store,
    )
