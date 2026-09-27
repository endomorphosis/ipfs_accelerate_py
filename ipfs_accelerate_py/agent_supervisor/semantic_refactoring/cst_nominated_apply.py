"""Apply nominated state, init, and binding module moves with the CST parser.

Receipts stay nomination-only. When raw sources are supplied, import modules
and dotted uses of ``source_module`` are rewritten to ``destination_module``.
Comments outside those tokens stay put. The result is staged to the VFS
outbox and world-root publisher. The repository is not written and the
current root is not CAS-updated.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json

from .cst_import_apply import _offset, _replacements
from .cst_world_handoff import CstWorldHandoff, stage_cst_world_root


def _pair(item: Any) -> tuple[str, str]:
    source = str(getattr(item, "source_module", "") or "")
    destination = str(getattr(item, "destination_module", "") or "")
    return source, destination


def _dotted_pieces(node: Any) -> list[Any]:
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


def _dotted_prefix_replacements(
    source: str, nominations: Sequence[Any]
) -> list[tuple[int, int, str]]:
    import parso

    pairs = [
        pair
        for pair in (_pair(item) for item in nominations)
        if pair[0] and pair[0] != pair[1]
    ]
    if not pairs:
        return []
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    tree = parso.parse(source)
    found: list[tuple[int, int, str]] = []

    def walk(node: Any) -> None:
        if getattr(node, "type", None) == "atom_expr":
            pieces = _dotted_pieces(node)
            if pieces:
                text = ".".join(str(piece.value) for piece in pieces)
                for src, dst in pairs:
                    if text == src or text.startswith(src + "."):
                        start = _offset(source, *pieces[0].start_pos)
                        end_piece = pieces[src.count(".")]
                        end = _offset(source, *end_piece.end_pos)
                        if source[start:end] == src:
                            found.append((start, end, dst))
                        break
        for child in getattr(node, "children", ()) or ():
            walk(child)

    walk(tree)
    return found


def rewrite_nominated_modules(source: str, nominations: Sequence[Any]) -> tuple[str, bool]:
    """Rename nominated modules in imports and dotted uses. Comments stay."""

    import_rewrites = tuple(
        SimpleNamespace(
            rewrite_kind="import",
            source_module=src,
            destination_module=dst,
            symbol_id="",
        )
        for src, dst in (_pair(item) for item in nominations)
        if src and src != dst
    )
    pieces = list(_replacements(source, import_rewrites))
    pieces.extend(_dotted_prefix_replacements(source, nominations))
    if not pieces:
        return source, False
    seen: set[tuple[int, int]] = set()
    text = source
    for start, end, new in sorted(pieces, key=lambda item: item[0], reverse=True):
        if (start, end) in seen:
            continue
        seen.add((start, end))
        text = text[:start] + new + text[end:]
    return text, text != source


def _is_module_name(value: str) -> bool:
    if not value or ":" in value or value.startswith(("bafy", "sha256:")):
        return False
    return all(part.isidentifier() for part in value.split("."))


def nominations_from_facade_edits(packet: Any) -> tuple[SimpleNamespace, ...]:
    """Facade edits whose endpoints are module names. CIDs are not rewritten.

    The facade is not retired.
    """

    found: list[SimpleNamespace] = []
    for edit in getattr(packet, "edits", ()) or ():
        kind = str(getattr(getattr(edit, "kind", ""), "value", None) or getattr(edit, "kind", ""))
        if kind != "facade":
            continue
        source = str(getattr(edit, "source_id", "") or "")
        destination = str(getattr(edit, "destination_id", "") or "")
        if not _is_module_name(source) or not _is_module_name(destination):
            continue
        if source == destination:
            continue
        found.append(
            SimpleNamespace(source_module=source, destination_module=destination)
        )
    return tuple(found)


def apply_nominated_cst_and_stage(
    packet: Any,
    raw_sources: Mapping[str, str],
    *,
    nominations: Sequence[Any],
    pre_world_root_cid: str | None = None,
    expected_generation: int = 1,
    kit_store: Any | None = None,
) -> CstWorldHandoff:
    """Apply nominated module moves and stage the CST result."""

    rewritten: dict[str, str] = {}
    for path, source in raw_sources.items():
        updated, changed = rewrite_nominated_modules(source, nominations)
        if changed:
            rewritten[path] = updated
    if not rewritten:
        return CstWorldHandoff(
            applied=False,
            blocked=True,
            reason="nominated_edit_not_applied",
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
