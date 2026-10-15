"""A bounded role view of an already verified semantic worker artifact.

Only estimator telemetry is deferred. This pure boundary does not admit a
producer, read source, grant substitution, or measure complete session usage.
The caller must first use the existing current artifact/source verifier and
must bind expansion's resolver to that same verifier, never to a bare file read.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..context.context_compiler import build_text_context_references
from ..context.context_contracts import ContextReference, ContextTier

SCHEMA = "supervisor-semantic-worker-role-view@1"
PROJECTION_REVISION = "estimator-telemetry-only@1"
MAX_ARTIFACT_BYTES = 2_000_000
MAX_EXPANSION_BYTES = 65_536
SELECTORS = (("pack", "token_estimate"), ("pack", "pack", "token_totals"))
_BASE_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "objective",
        "scope_cid",
        "semantic_root_cid",
        "manifest",
        "required_raw_paths",
        "preparation_bounds",
        "capsules",
        "admissions",
        "raw_sources",
        "pack",
        "doctor_snapshot_id",
        "doctor_manifest_cid",
        "reconstruction",
        "refresh_lineage",
        "completion_authority",
    }
)


def _bound(value: int, maximum: int, name: str) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"positive bounded {name} required")


def _text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip() or "\x00" in value or len(value.encode()) > 4096:
        raise ValueError(f"explicit bounded {name} required")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _encoded(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _structure(value, limit: int) -> None:
    """Bound untrusted structures before recursive copying or serialization."""
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 200_000 or depth > 40:
            raise ValueError("projection structure exceeds bounded depth or population")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("projection JSON object keys must be strings")
                text_bytes += len(key.encode())
                pending.append((child, depth + 1))
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            text_bytes += len(item.encode())
        elif type(item) is int:
            if abs(item) >= 2**63:
                raise ValueError("projection JSON integer exceeds bound")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("projection JSON number must be finite")
        elif item is not None and type(item) is not bool:
            raise ValueError("projection accepts literal JSON values only")
        if text_bytes > limit:
            raise ValueError("projection text allocation exceeds byte bound")


def _parse(text: str, limit: int) -> dict:
    if type(text) is not str or len(text) > limit:
        raise ValueError("bounded original text required")
    raw = text.encode()
    if len(raw) > limit:
        raise ValueError("original artifact exceeds byte bound")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate original JSON key")
            result[key] = value
        return result

    def integer(token):
        if len(token) > 20 or abs(int(token)) >= 2**63:
            raise ValueError("bounded original JSON integer required")
        return int(token)

    def invalid(_token):
        raise ValueError("nonfinite original JSON number refused")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_int=integer, parse_constant=invalid)
    except RecursionError as exc:
        raise ValueError("original JSON nesting exceeds bound") from exc
    _structure(value, limit)
    if type(value) is not dict:
        raise ValueError("original semantic artifact must be an object")
    return value


@dataclass(frozen=True)
class SemanticProjectionBindings:
    """Exact caller-verified parent and compiled-context bindings; no authority."""

    artifact: str
    expected_sha256: str
    task_id: str
    semantic_root_cid: str
    scope_cid: str
    repository_id: str
    tree_id: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _text(getattr(self, name), name)
        relative = PurePosixPath(self.artifact)
        if (
            self.artifact in {".", ".."}
            or relative.is_absolute()
            or str(relative) != self.artifact
            or any(part in {".", ".."} for part in relative.parts)
        ):
            raise ValueError("canonical repository-relative original artifact required")
        if re.fullmatch(r"[0-9a-f]{64}", self.expected_sha256) is None:
            raise ValueError("exact original SHA256 required")

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _original(original_text, bindings, max_original_bytes):
    if type(bindings) is not SemanticProjectionBindings:
        raise ValueError("explicit projection bindings required")
    value = _parse(original_text, max_original_bytes)
    if _sha(original_text.encode()) != bindings.expected_sha256:
        raise ValueError("original artifact digest differs")
    fields = set(value)
    if (
        not _BASE_FIELDS <= fields
        or fields - _BASE_FIELDS - {"worker_projection"}
        or value["schema"] != "supervisor-semantic-worker-context@1"
        or value["task_id"] != bindings.task_id
        or value["semantic_root_cid"] != bindings.semantic_root_cid
        or value["scope_cid"] != bindings.scope_cid
        or value["completion_authority"] is not False
    ):
        raise ValueError("original semantic schema/task/root/scope binding differs")
    if (
        type(value["pack"]) is not dict
        or type(value["pack"].get("pack")) is not dict
        or value["pack"].get("schema") != "ipfs-accelerate.context-pack-result@1"
        or value["pack"].get("interface") != "ContextPack@1"
    ):
        raise ValueError("supported native context pack required")
    if (
        type(value["required_raw_paths"]) is not list
        or not value["required_raw_paths"]
        or type(value["manifest"]) is not dict
        or type(value["raw_sources"]) is not dict
        or type(value["capsules"]) is not list
        or type(value["admissions"]) is not list
    ):
        raise ValueError("complete required source and capsule material required")
    if (
        any(type(name) is not str for name in value["required_raw_paths"])
        or not set(value["required_raw_paths"]) <= set(value["raw_sources"])
        or not set(value["raw_sources"]) <= set(value["manifest"])
    ):
        raise ValueError("required raw source coverage differs")
    return value


def _telemetry(value, selector):
    node = value
    for field in selector[:-1]:
        node = node[field]
    if selector[-1] not in node:
        return None
    telemetry = node[selector[-1]]
    totals = telemetry
    if selector == SELECTORS[0]:
        if (
            type(telemetry) is not dict
            or set(telemetry) != {"totals", "estimator_version", "total"}
            or type(telemetry["estimator_version"]) is not str
            or type(telemetry["total"]) is not int
            or telemetry["total"] < 0
        ):
            raise ValueError("closed estimator telemetry required")
        totals = telemetry["totals"]
    if type(totals) is not dict or any(
        type(key) is not str or type(amount) is not int or amount < 0
        for key, amount in totals.items()
    ):
        raise ValueError("typed estimator category totals required")
    if selector == SELECTORS[0] and telemetry["total"] != sum(totals.values()):
        raise ValueError("estimator category sum differs")
    return telemetry


def build_semantic_context_projection(
    *,
    original_text: str,
    bindings: SemanticProjectionBindings,
    max_original_bytes: int = MAX_ARTIFACT_BYTES,
    max_view_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict:
    """Derive a role view only after the caller has verified the full parent.

    The nested required_context is material, not a loadable shortened artifact.
    Unknown nested decision fields remain literal; no capsule metadata, refresh
    history, reconstruction limitation, currentness, or raw-fetch caveat is cut.
    """
    _bound(max_original_bytes, MAX_ARTIFACT_BYTES, "original byte budget")
    _bound(max_view_bytes, MAX_ARTIFACT_BYTES, "view byte budget")
    required = _original(original_text, bindings, max_original_bytes)
    archive = []
    for selector in SELECTORS:
        value = _telemetry(required, selector)
        if value is not None:
            archive.append({"field_selector": list(selector), "value": value})
            node = required
            for field in selector[:-1]:
                node = node[field]
            del node[selector[-1]]
    coverage = _sha(_encoded(required))
    deferred = []
    if archive:
        body = _encoded({"schema": "supervisor-semantic-estimator-archive@1", "fields": archive})
        digest = _sha(body)
        reference = ContextReference(
            reference_id="semantic-estimator-archive:" + digest,
            kind="semantic-estimator-archive",
            tier=ContextTier.EXPANSION,
            referenced_content_id="sha256:" + digest,
            repository_id=bindings.repository_id,
            tree_id=bindings.tree_id,
            path=bindings.artifact,
            summary="Estimator telemetry only; no required decision content is deferred.",
            byte_count=len(body),
            token_count=0,
            metadata={
                "required": False,
                "field_selectors": [row["field_selector"] for row in archive],
                "selector_policy": PROJECTION_REVISION,
                "parent_sha256": bindings.expected_sha256,
                "task_id": bindings.task_id,
                "semantic_root_cid": bindings.semantic_root_cid,
                "scope_cid": bindings.scope_cid,
                "archived_sha256": digest,
                "required_coverage_sha256": coverage,
                "coverage_ids": [],
                "coverage_summary": "No task, source, dependency, caveat, risk or acceptance coverage removed.",
            },
        )
        # ContextReference's str-enum tier can remain an Enum in to_dict().
        # A public role view is literal JSON, so normalize our own typed handle
        # before later strict structure validation; no untrusted value is cast.
        deferred.append(json.loads(_encoded(reference.to_dict())))
    view = {
        "schema": SCHEMA,
        "projection_revision": PROJECTION_REVISION,
        "origin": bindings.to_dict(),
        "required_context": required,
        "required_coverage_sha256": coverage,
        "deferred_references": deferred,
        "execution_authority": False,
        "completion_authority": False,
        "semantic_equivalence_claimed": False,
    }
    if len(_encoded(view)) > max_view_bytes:
        raise ValueError("role view exceeds byte bound; required material cannot be truncated")
    return view


def verify_semantic_context_projection(
    *,
    view: dict,
    original_text: str,
    bindings: SemanticProjectionBindings,
    max_original_bytes: int = MAX_ARTIFACT_BYTES,
    max_view_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict:
    """Reject altered views by exact regeneration from the same verified parent."""
    _bound(max_original_bytes, MAX_ARTIFACT_BYTES, "original byte budget")
    _bound(max_view_bytes, MAX_ARTIFACT_BYTES, "view byte budget")
    _structure(view, max_view_bytes)
    actual = _encoded(view)
    if len(actual) > max_view_bytes:
        raise ValueError("role view exceeds byte bound")
    expected = build_semantic_context_projection(
        original_text=original_text,
        bindings=bindings,
        max_original_bytes=max_original_bytes,
        max_view_bytes=max_view_bytes,
    )
    if actual != _encoded(expected):
        raise ValueError("role view differs from exact original projection or required coverage")
    return expected


def semantic_context_projection_references(
    *,
    view: dict,
    original_text: str,
    bindings: SemanticProjectionBindings,
    coverage_ids: Sequence[str] = (),
    chunk_bytes: int = 6144,
) -> tuple[ContextReference, ...]:
    """Send required role-view chunks, with deferred handles inside the view only."""
    _bound(chunk_bytes, 16384, "projection chunk byte budget")
    verified = verify_semantic_context_projection(
        view=view, original_text=original_text, bindings=bindings
    )
    if type(coverage_ids) not in {list, tuple} or any(
        type(item) is not str for item in coverage_ids
    ):
        raise ValueError("explicit coverage identifier sequence required")
    coverage_ids = (
        *coverage_ids,
        "semantic-projection-required:" + verified["required_coverage_sha256"],
    )
    return build_text_context_references(
        _encoded(verified).decode(),
        reference_prefix="semantic-projection",
        kind="semantic-context",
        path=bindings.artifact,
        repository_id=bindings.repository_id,
        tree_id=bindings.tree_id,
        required=True,
        chunk_bytes=chunk_bytes,
        coverage_ids=coverage_ids,
    )


def expand_semantic_context_projection(
    *,
    view: dict,
    reference_id: str,
    bindings: SemanticProjectionBindings,
    resolve_original: Callable[..., str],
    count_tokens: Callable[[str], int],
    max_bytes: int = 16384,
    max_tokens: int = 4096,
    cancelled: Callable[[], bool] | None = None,
) -> ContextReference:
    """Resolve one typed archive handle through a fresh caller-owned verifier.

    The counter measures the exact canonical returned reference descriptor,
    with token_count=0 (no estimate is relabeled as provider usage). The caller
    must still measure its complete rendered message; no extra session charge
    or cache benefit is asserted by this local expansion.
    """
    _bound(max_bytes, MAX_EXPANSION_BYTES, "expansion byte budget")
    _bound(max_tokens, 128000, "expansion token budget")
    _text(reference_id, "expansion reference identity")
    if not callable(resolve_original) or not callable(count_tokens):
        raise ValueError("explicit current original resolver and token counter required")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("explicit cancellation callback required")

    def check_cancelled():
        if cancelled is not None:
            value = cancelled()
            if type(value) is not bool or value:
                raise ValueError("projection expansion cancelled or unavailable")

    check_cancelled()
    # Close bounds before calling a resolver, even for an invalid view.
    _structure(view, MAX_ARTIFACT_BYTES)
    if len(_encoded(view)) > MAX_ARTIFACT_BYTES or type(bindings) is not SemanticProjectionBindings:
        raise ValueError("bounded view and explicit original bindings required")
    request = {
        "artifact": bindings.artifact,
        "expected_sha256": bindings.expected_sha256,
        "task_id": bindings.task_id,
    }
    original_text = resolve_original(**request)
    verified = verify_semantic_context_projection(
        view=view, original_text=original_text, bindings=bindings
    )
    handles = verified["deferred_references"]
    if len(handles) != 1 or handles[0]["reference_id"] != reference_id:
        raise ValueError(
            "selected expansion handle is absent or outside the closed selector policy"
        )
    handle = ContextReference.from_dict(handles[0])
    if handle.byte_count > max_bytes:
        raise ValueError("archived expansion exceeds byte bound before materialization")
    original = _original(original_text, bindings, MAX_ARTIFACT_BYTES)
    archive = [
        {"field_selector": list(selector), "value": _telemetry(original, selector)}
        for selector in SELECTORS
        if _telemetry(original, selector) is not None
    ]
    body = _encoded({"schema": "supervisor-semantic-estimator-archive@1", "fields": archive})
    if len(body) != handle.byte_count or _sha(body) != handle.metadata["archived_sha256"]:
        raise ValueError("archive size or digest differs from selected handle")
    expanded = ContextReference(
        reference_id=handle.reference_id,
        kind=handle.kind,
        tier=ContextTier.EVIDENCE,
        referenced_content_id=handle.referenced_content_id,
        repository_id=handle.repository_id,
        tree_id=handle.tree_id,
        path=handle.path,
        summary=body.decode(),
        byte_count=len(body),
        token_count=0,
        metadata=dict(handle.metadata),
    )
    wire = _encoded(expanded.to_dict()).decode()
    if len(wire.encode()) > max_bytes:
        raise ValueError("complete expanded reference exceeds byte bound")
    tokens = count_tokens(wire)
    if type(tokens) is not int or not 1 <= tokens <= max_tokens:
        raise ValueError("complete expanded reference exceeds typed token bound")
    check_cancelled()
    # Counter/resolver work may interleave with source changes. Refuse rather
    # than returning an expansion from a no-longer-current original parent.
    closing_text = resolve_original(**request)
    verify_semantic_context_projection(view=view, original_text=closing_text, bindings=bindings)
    if closing_text != original_text:
        raise ValueError("original parent changed during projection expansion")
    check_cancelled()
    return expanded
