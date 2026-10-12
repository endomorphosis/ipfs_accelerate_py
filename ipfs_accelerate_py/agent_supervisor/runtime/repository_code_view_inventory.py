"""Historical read-only code views bound to retained committed source.

All original bytes stay in the supplied existing CAS. Views and complete byte
maps restore unchanged source only; there is no edit mapper, workspace writer,
model tool, current-source admission or proof authority here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)
from .repository_source_inventory import (
    SourceInventoryLimits, SourceInventoryRequest,
    _BoundCAS, _cid, _input_guard, _require, load_source_inventory,
)
from .reversible_source_view import (
    build_source_view, restore_source_view, current_codec_binding,
)

SCHEMA = "supervisor-historical-code-view-inventory@1"
PAGE_SCHEMA = "supervisor-historical-code-view-index-page@1"
PROFILES = {"identity@1", "python-whitespace@1", "python-comments@1"}
MAX_BLOCK = 1024 * 1024
_ROOT_FIELDS = {"schema", "source_inventory_cid", "request", "source_limits", "profile",
                "protected_lines", "codec_binding", "entry_pages", "entry_count",
                "historical_only", "completion_authority"}
_ENTRY_FIELDS = {"raw_path_hex", "git_mode", "git_object_oid", "source_cid",
                 "view_cid", "view_bytes", "view_storage", "manifest_cid", "manifest_sha256"}


@dataclass(frozen=True)
class CodeViewInventoryRef:
    inventory_cid: str
    source_inventory_cid: str
    request: SourceInventoryRequest


@dataclass(frozen=True)
class HistoricalCodeView:
    entry: object
    original: bytes | None
    view: bytes | None
    manifest: dict | None


@dataclass(frozen=True)
class RestoredCodeViewInventory:
    inventory_cid: str
    source_inventory_cid: str
    entries: tuple[HistoricalCodeView, ...]

    def observation(self):
        return dict(schema="supervisor-historical-code-view-replay@1",
            inventory_cid=self.inventory_cid, source_inventory_cid=self.source_inventory_cid,
            entry_count=len(self.entries),
            original_per_path_bytes=sum(len(e.original) for e in self.entries if e.original is not None),
            view_per_path_bytes=sum(len(e.view) for e in self.entries if e.view is not None),
            changed_view_count=sum(e.original != e.view for e in self.entries),
            gitlink_count=sum(e.original is None for e in self.entries),
            content_complete=all(e.original is not None for e in self.entries),
            historical_only=True, current_source_verified=False, read_only=True,
            exact_unchanged_source_restoration_verified=True,
            semantic_equivalence_claimed=False, completion_authority=False,
            execution_authority=False, filesystem_restored=False, provider_token_counts=None)

    def metadata_page(self, *, offset=0, limit=64):
        _require(type(offset) is int and 0 <= offset <= len(self.entries), "invalid view page offset")
        _require(type(limit) is int and 1 <= limit <= 256, "invalid view page limit")
        end = min(len(self.entries), offset + limit)
        return dict(schema="supervisor-historical-code-view-page@1", inventory_cid=self.inventory_cid,
            source_inventory_cid=self.source_inventory_cid, offset=offset,
            next_offset=end if end < len(self.entries) else None,
            entries=[dict(e.entry.to_dict(), view_bytes=None if e.view is None else len(e.view),
                status="gitlink_requires_separate_repository" if e.original is None else
                    "compact-read-only" if e.view != e.original else "exact-source")
                for e in self.entries[offset:end]], historical_only=True, read_only=True,
            completion_authority=False)


def _policy(profile, protected_lines):
    _require(type(profile) is str and profile in PROFILES, "unsupported code view profile")
    if protected_lines is None:
        return {}
    _require(type(protected_lines) is dict and len(protected_lines) <= 256,
             "protected source policy exceeds bounds")
    result, count = {}, 0
    for path, lines in protected_lines.items():
        _require(type(path) is str and 0 < len(path) <= 8192, "invalid protected raw path")
        try:
            canonical = bytes.fromhex(path).hex() == path
        except ValueError:
            canonical = False
        _require(canonical, "invalid protected raw path")
        _require(type(lines) in {tuple, list} and all(type(n) is int and 1 <= n <= 1048576 for n in lines),
                 "invalid protected source lines")
        _require(tuple(lines) == tuple(sorted(set(lines))) and lines, "noncanonical protected source lines")
        count += len(lines)
        _require(count <= 4096, "protected source lines exceed bounds")
        result[path] = list(lines)
    return dict(sorted(result.items()))


def _inputs(request, source_limits, profile, protected_lines):
    native_guard = _input_guard(request, source_limits)
    policy = _policy(profile, protected_lines)
    binding = current_codec_binding()

    def guard():
        native_guard()
        _require(_policy(profile, protected_lines) == policy, "code view policy changed")
        _require(current_codec_binding() == binding, "code view codec binding changed")
    return (SourceInventoryRequest(**asdict(request)), SourceInventoryLimits(**asdict(source_limits)),
            policy, binding, guard)


def _profile_for(source, profile):
    return profile if source.entry.git_mode in {"100644", "100755"} and bytes.fromhex(
        source.entry.raw_path_hex).endswith((b".py", b".pyi")) else "identity@1"


class _Metadata:
    def __init__(self, store, limits):
        self.store, self.limits, self.blocks, self.total = store, limits, {}, 0

    def account(self, cid, value):
        raw = canonical_dag_json_bytes(value)
        _require(len(raw) <= MAX_BLOCK, "code view metadata frame exceeds bounds")
        if cid not in self.blocks:
            self.total += len(raw)
            _require(len(self.blocks) < 3 * self.limits.max_entries + 1
                     and self.total <= self.limits.max_metadata_bytes,
                     "code view metadata population exceeds bounds")
            self.blocks[cid] = value

    def put(self, value):
        cid = cid_for_structured(value)
        self.account(cid, value)
        self.store.put(cid, value)
        return cid

    def get(self, cid):
        _cid(cid)
        if cid not in self.blocks:
            value = self.store.get(cid)
            self.account(cid, value)
        return self.blocks[cid]


def _validate_policy_sources(sources, policy):
    eligible = {s.entry.raw_path_hex for s in sources.entries
                if s.data is not None and _profile_for(s, "python-whitespace@1") != "identity@1"}
    _require(set(policy) <= eligible, "protected policy differs from regular Python source scope")


def _root(source_cid, request, limits, profile, policy, binding, pages, count):
    return dict(schema=SCHEMA, source_inventory_cid=source_cid, request=asdict(request),
        source_limits=asdict(limits), profile=profile, protected_lines=policy,
        codec_binding=binding, entry_pages=pages, entry_count=count,
        historical_only=True, completion_authority=False)


def _page_values(records):
    page, size = [], 256
    for record in records:
        extent = len(canonical_dag_json_bytes(record)) + 1
        if page and size + extent > MAX_BLOCK:
            yield dict(schema=PAGE_SCHEMA, records=page)
            page, size = [], 256
        page.append(record)
        size += extent
    if page:
        yield dict(schema=PAGE_SCHEMA, records=page)


def _replay(store, metadata, root_cid, root, *, source_cid, request, limits, profile, policy, binding):
    _require(type(root) is dict and set(root) == _ROOT_FIELDS and root["schema"] == SCHEMA,
             "unsupported code view inventory schema")
    _require(root["source_inventory_cid"] == source_cid
             and canonical_dag_json_bytes(root["request"]) == canonical_dag_json_bytes(asdict(request))
             and canonical_dag_json_bytes(root["source_limits"]) == canonical_dag_json_bytes(asdict(limits)),
             "code view inventory differs from independent source")
    _require(root["profile"] == profile
             and canonical_dag_json_bytes(root["protected_lines"]) == canonical_dag_json_bytes(policy)
             and canonical_dag_json_bytes(root["codec_binding"]) == canonical_dag_json_bytes(binding),
             "code view profile or codec differs from independent policy")
    _require(root["historical_only"] is True and root["completion_authority"] is False,
             "code view authority flags changed")
    refs = root["entry_pages"]
    _require(type(refs) is list and len(refs) <= limits.max_entries,
             "code view page references exceed bounds")
    for cid in refs:
        _cid(cid)
    _require(len(set(refs)) == len(refs), "code view pages duplicated")
    records = []
    for cid in refs:
        page = metadata.get(cid)
        _require(type(page) is dict and set(page) == {"schema", "records"}
                 and page["schema"] == PAGE_SCHEMA and type(page["records"]) is list
                 and page["records"], "invalid code view page")
        _require(len(records) + len(page["records"]) <= limits.max_entries,
                 "code view entry population exceeds bounds")
        records.extend(page["records"])
    _require(type(root["entry_count"]) is int and root["entry_count"] == len(records),
             "code view population count changed")
    _require([cid_for_structured(page) for page in _page_values(records)] == refs,
             "code view paging is not canonical")
    store.check()
    sources = load_source_inventory(cas=store.cas, expected_inventory_cid=source_cid,
                                   request=request, limits=limits)
    store.check()
    _validate_policy_sources(sources, policy)
    _require(len(records) == len(sources.entries), "code view omits original source population")
    restored = []
    for record, source in zip(records, sources.entries):
        _require(type(record) is dict and set(record) == _ENTRY_FIELDS,
                 "invalid code view entry schema")
        expected = source.entry
        _require((record["raw_path_hex"], record["git_mode"], record["git_object_oid"], record["source_cid"])
                 == (expected.raw_path_hex, expected.git_mode, expected.git_object_oid, source.source_cid),
                 "code view entry differs from original source")
        if source.data is None:
            _require(all(record[k] is None for k in ("view_cid", "view_bytes", "manifest_cid", "manifest_sha256"))
                     and record["view_storage"] == "gitlink-reference", "code view invents Gitlink content")
            restored.append(HistoricalCodeView(expected, None, None, None))
            continue
        manifest = metadata.get(record["manifest_cid"])
        _require(type(record["view_bytes"]) is int and 0 <= record["view_bytes"] <= len(source.data),
                 "code view byte extent exceeds original source")
        if record["view_storage"] == "original-source":
            view = source.data
            _require(record["view_cid"] == source.source_cid and len(view) == record["view_bytes"],
                     "exact source view changed")
        else:
            _require(record["view_storage"] == "retained-view" and record["view_bytes"] <= MAX_BLOCK,
                     "unsupported retained code view")
            view = store.get_bytes(record["view_cid"], record["view_bytes"])
        candidate = build_source_view(source.data, profile=_profile_for(source, profile),
            protected_lines=tuple(policy.get(expected.raw_path_hex, ())))
        _require(record["view_storage"] == ("original-source" if candidate.view == source.data else "retained-view")
                 and record["view_cid"] == cid_for_bytes(candidate.view)
                 and record["manifest_cid"] == cid_for_structured(candidate.manifest),
                 "code view storage identity differs from deterministic source transform")
        _require(candidate.view == view and candidate.manifest == manifest
                 and candidate.manifest_sha256 == record["manifest_sha256"],
                 "code view differs from deterministic source transform")
        original = restore_source_view(original=source.data, view=view, manifest=manifest,
                                       expected_manifest_sha256=record["manifest_sha256"])
        _require(original == source.data and cid_for_bytes(original) == source.source_cid,
                 "code view failed exact source restoration")
        restored.append(HistoricalCodeView(expected, original, view, manifest))
    store.check()
    return RestoredCodeViewInventory(root_cid, source_cid, tuple(restored))


def retain_code_view_inventory(*, cas, expected_source_inventory_cid, request,
        source_limits=SourceInventoryLimits(), profile="python-whitespace@1", protected_lines=None):
    """Retain read-only candidates; original source remains in its existing inventory."""
    request, limits, policy, binding, guard = _inputs(request, source_limits, profile, protected_lines)
    source_cid = _cid(expected_source_inventory_cid)
    store = _BoundCAS(cas, guard)
    sources = load_source_inventory(cas=cas, expected_inventory_cid=source_cid, request=request, limits=limits)
    store.check()
    _validate_policy_sources(sources, policy)
    metadata, records = _Metadata(store, limits), []
    for source in sources.entries:
        entry = source.entry
        record = dict(raw_path_hex=entry.raw_path_hex, git_mode=entry.git_mode,
            git_object_oid=entry.git_object_oid, source_cid=source.source_cid,
            view_cid=None, view_bytes=None, view_storage="gitlink-reference",
            manifest_cid=None, manifest_sha256=None)
        if source.data is not None:
            view = build_source_view(source.data, profile=_profile_for(source, profile),
                protected_lines=tuple(policy.get(entry.raw_path_hex, ())))
            view_cid = cid_for_bytes(view.view)
            same = view.view == source.data
            if not same:
                store.put_bytes(view_cid, view.view)
            record.update(view_cid=view_cid, view_bytes=len(view.view),
                view_storage="original-source" if same else "retained-view",
                manifest_cid=metadata.put(view.manifest), manifest_sha256=view.manifest_sha256)
        records.append(record)
    pages = [metadata.put(page) for page in _page_values(records)]
    root = _root(source_cid, request, limits, profile, policy, binding, pages, len(records))
    root_cid = cid_for_structured(root)
    args = dict(source_cid=source_cid, request=request, limits=limits,
                profile=profile, policy=policy, binding=binding)
    _replay(store, _Metadata(store, limits), root_cid, root, **args)
    metadata.put(root)
    _replay(store, _Metadata(store, limits), root_cid, store.get(root_cid), **args)
    store.check()
    return CodeViewInventoryRef(root_cid, source_cid, request)


def load_code_view_inventory(*, cas, expected_inventory_cid, expected_source_inventory_cid,
        request, source_limits=SourceInventoryLimits(), profile="python-whitespace@1", protected_lines=None):
    """Restore unchanged historical bytes from a separately pinned read-only view."""
    request, limits, policy, binding, guard = _inputs(request, source_limits, profile, protected_lines)
    root_cid, source_cid = _cid(expected_inventory_cid), _cid(expected_source_inventory_cid)
    store = _BoundCAS(cas, guard)
    metadata = _Metadata(store, limits)
    root = metadata.get(root_cid)
    return _replay(store, metadata, root_cid, root, source_cid=source_cid, request=request,
                   limits=limits, profile=profile, policy=policy, binding=binding)
