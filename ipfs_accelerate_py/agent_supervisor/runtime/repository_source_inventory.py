"""Retain and cold-replay complete committed source in an existing datasets CAS.

This is historical byte integrity, not current-source admission, filesystem
restoration, semantic analysis, proof, model-tool access or completion. Capture
uses the native complete population/chunk producer; replay uses its closed DAG
parser and independently verifies every retained blob. Target code is never run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import stat

from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_byte_chunks, cid_for_bytes,
    cid_for_structured, validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.chunked_snapshot import (
    ChunkedRepositorySnapshot, ChunkedSnapshotLimits, MAX_FRAME_BYTES,
    read_chunked_blob_frame, snapshot_chunked_repository,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import (
    CommittedPopulationEntry, preflight_committed_repository,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.paged_snapshot import (
    parse_chunked_snapshot_manifest,
)

SCHEMA = "supervisor-committed-source-inventory@1"
_ROOT_FIELDS = {"schema", "request", "native_snapshot_cid", "limits",
                "historical_only", "completion_authority"}
_HARD_LIMITS = dict(max_entries=20000, max_file_bytes=128 * 1024 * 1024,
                    max_total_bytes=128 * 1024 * 1024,
                    max_metadata_bytes=32 * 1024 * 1024,
                    frame_bytes=MAX_FRAME_BYTES, max_chunks=65536,
                    max_frame_reads=65536, max_rehash_bytes=8 * 1024**3)


class SourceInventoryError(ValueError):
    """No successful source inventory can be returned for this request."""


def _require(condition, reason):
    if not condition:
        raise SourceInventoryError(reason)


def _cid(value, codec="dag-json"):
    _require(type(value) is str and len(value) <= 128, "invalid content identity")
    return validate_cid(value, codecs={codec})


@dataclass(frozen=True)
class SourceInventoryRequest:
    task_id: str
    repository_id: str
    expected_commit: str
    expected_tree: str
    expected_population_cid: str

    def __post_init__(self):
        for value in (self.task_id, self.repository_id):
            _require(type(value) is str and value.strip() and len(value.encode()) <= 4096,
                     "explicit bounded task and repository identities required")
        for value in (self.expected_commit, self.expected_tree):
            _require(type(value) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value),
                     "full commit and tree identities required")
        _require(len(self.expected_commit) == len(self.expected_tree), "object widths differ")
        _cid(self.expected_population_cid)


@dataclass(frozen=True)
class SourceInventoryLimits:
    max_entries: int = 20000
    max_file_bytes: int = 4 * 1024 * 1024
    max_total_bytes: int = 128 * 1024 * 1024
    max_metadata_bytes: int = 32 * 1024 * 1024
    frame_bytes: int = MAX_FRAME_BYTES
    max_chunks: int = 65536
    max_frame_reads: int = 4096
    max_rehash_bytes: int = 512 * 1024 * 1024

    def __post_init__(self):
        for name, value in asdict(self).items():
            _require(type(value) is int and 0 < value <= _HARD_LIMITS[name],
                     "source inventory limit exceeds fixed bounds")

    def native(self):
        return ChunkedSnapshotLimits(frame_bytes=self.frame_bytes,
            max_entries=self.max_entries, max_stream_bytes=self.max_total_bytes,
            max_chunks=self.max_chunks, max_metadata_bytes=self.max_metadata_bytes)


@dataclass(frozen=True)
class SourceInventoryRef:
    inventory_cid: str
    native_snapshot_cid: str
    request: SourceInventoryRequest


@dataclass(frozen=True)
class RestoredSourceEntry:
    entry: CommittedPopulationEntry
    source_cid: str | None
    data: bytes | None


@dataclass(frozen=True)
class RestoredSourceInventory:
    inventory_cid: str
    native_snapshot: ChunkedRepositorySnapshot
    entries: tuple[RestoredSourceEntry, ...]

    def observation(self):
        return dict(schema="supervisor-historical-source-replay@1",
            inventory_cid=self.inventory_cid,
            native_snapshot_cid=self.native_snapshot.snapshot_cid,
            repository_id=self.native_snapshot.repository_id,
            git_commit=self.native_snapshot.git_commit, git_tree=self.native_snapshot.git_tree,
            population_cid=self.native_snapshot.population_cid,
            entry_count=len(self.entries),
            unique_blob_count=len(self.native_snapshot.blobs),
            unique_blob_bytes=sum(b.size_bytes for b in self.native_snapshot.blobs),
            per_path_blob_bytes=sum(e.entry.size_bytes or 0 for e in self.entries),
            gitlink_count=sum(e.entry.object_type == "commit" for e in self.entries),
            content_complete=all(e.data is not None for e in self.entries),
            historical_only=True, current_source_verified=False,
            semantic_acceptance_authority=False, completion_authority=False,
            execution_authority=False, filesystem_restored=False)

    def metadata_page(self, *, offset=0, limit=64):
        _require(type(offset) is int and 0 <= offset <= len(self.entries), "invalid page offset")
        _require(type(limit) is int and 1 <= limit <= 256, "invalid page limit")
        end = min(len(self.entries), offset + limit)
        return dict(schema="supervisor-historical-source-page@1",
            inventory_cid=self.inventory_cid, population_cid=self.native_snapshot.population_cid,
            entries=[dict(e.entry.to_dict(), source_cid=e.source_cid,
                status="retained-blob" if e.data is not None else "gitlink_requires_separate_repository")
                for e in self.entries[offset:end]],
            offset=offset, next_offset=end if end < len(self.entries) else None,
            population_complete=True, historical_only=True, completion_authority=False)


class _BoundCAS:
    """Retain the exact supplied store and its physical/configuration bindings."""
    def __init__(self, cas, guard):
        _require(type(cas) is ImmutableCAS, "existing native ImmutableCAS required")
        self.cas = cas
        self.guard = guard
        self.binding = self._binding()

    def _binding(self):
        _require(type(self.cas) is ImmutableCAS, "source store type changed")
        cap = self.cas.max_object_bytes
        _require(type(cap) is int and 0 < cap <= 16 * 1024 * 1024,
                 "source store object limit exceeds fixed read bounds")
        paths = []
        for attr in ("root", "structured_root", "source_root"):
            path = getattr(self.cas, attr)
            _require(type(path) is Path or isinstance(path, Path), "source store path changed")
            info = path.stat(follow_symlinks=False)
            _require(stat.S_ISDIR(info.st_mode), "source store directory unavailable")
            paths.append((str(path), str(path.resolve(strict=True)), info.st_dev, info.st_ino))
        _require(Path(self.cas.structured_root) == Path(self.cas.root) / "structured"
                 and Path(self.cas.source_root) == Path(self.cas.root) / "source",
                 "source store layout changed")
        return cap, tuple(paths)

    def check(self):
        self.guard()
        _require(self._binding() == self.binding, "source store binding changed")
        self.guard()

    def get(self, cid):
        self.check()
        value = self.cas.get(_cid(cid))
        self.check()
        raw = canonical_dag_json_bytes(value)
        _require(len(raw) <= MAX_FRAME_BYTES and cid_for_structured(value) == cid,
                 "invalid or oversized retained manifest")
        return value

    def get_bytes(self, cid, size):
        self.check()
        data = self.cas.get_bytes(_cid(cid, "raw"))
        self.check()
        _require(type(data) is bytes and len(data) == size and cid_for_bytes(data) == cid,
                 "retained source frame does not verify")
        return data

    def put(self, cid, value):
        self.check()
        _require(self.cas.put(value) == cid, "retained manifest identity changed")
        self.check()
        _require(self.get(cid) == value, "retained manifest changed")

    def put_bytes(self, cid, data):
        self.check()
        _require(self.cas.put_bytes(data) == cid, "retained frame identity changed")
        self.check()
        _require(self.get_bytes(cid, len(data)) == data, "retained frame changed")


def _typed(request, limits):
    _require(type(request) is SourceInventoryRequest, "typed source request required")
    _require(type(limits) is SourceInventoryLimits, "typed source limits required")
    # Recheck frozen records too: object.__setattr__ is not an admission boundary.
    SourceInventoryRequest(**asdict(request))
    SourceInventoryLimits(**asdict(limits))


def _input_guard(request, limits):
    _typed(request, limits)
    original = asdict(request), asdict(limits)

    def check():
        _typed(request, limits)
        _require((asdict(request), asdict(limits)) == original,
                 "source request or limits changed")
    return check


def _population_bounds(snapshot, limits):
    _require(len(snapshot.entries) <= limits.max_entries, "source entry budget exceeded")
    sizes = [e.size_bytes or 0 for e in snapshot.entries]
    _require(max(sizes, default=0) <= limits.max_file_bytes, "source file budget exceeded")
    _require(sum(sizes) <= limits.max_total_bytes, "per-path source byte budget exceeded")
    _require(sum(len(b.chunks) for b in snapshot.blobs) <= limits.max_frame_reads
             and sum(b.size_bytes * (1 + len(b.chunks)) for b in snapshot.blobs) <= limits.max_rehash_bytes,
             "retained source frame-read or rehash profile exceeds bounds")


def _manifest_blocks(store, root_cid, limits):
    blocks = {}
    total = 0

    def read(cid):
        nonlocal total
        _cid(cid)
        if cid in blocks:
            return json.loads(blocks[cid])
        _require(len(blocks) < 3 * limits.max_entries + 1, "manifest block budget exceeded")
        value = store.get(cid)
        raw = canonical_dag_json_bytes(value)
        total += len(raw)
        _require(total <= limits.max_metadata_bytes, "manifest byte budget exceeded")
        blocks[cid] = raw
        return value

    root = read(root_cid)
    _require(type(root) is dict, "invalid native manifest root")
    for kind, name in (("entries", "entry_pages"), ("blobs", "blob_pages")):
        refs = root.get(name)
        _require(type(refs) is list and len(refs) <= limits.max_entries,
                 "invalid bounded manifest pages")
        for ref in refs:
            _cid(ref)
        _require(len(set(refs)) == len(refs), "duplicate manifest pages")
        count = 0
        for ref in refs:
            page = read(ref)
            _require(type(page) is dict and set(page) == {"schema", "kind", "records"}
                     and page["schema"] == "ipfs-datasets.git-chunk-index-page@1"
                     and page.get("kind") == kind
                     and type(page.get("records")) is list
                     and page["records"], "invalid manifest page")
            count += len(page["records"])
            _require(count <= limits.max_entries, "manifest page population exceeds entry budget")
            if kind == "blobs":
                for item in page["records"]:
                    _require(type(item) is dict and set(item) == {"git_object_oid", "manifest_cid"},
                             "invalid blob manifest reference")
                    read(item["manifest_cid"])
    return blocks


def _parse(store, root, request, limits):
    _require(type(root) is dict and set(root) == _ROOT_FIELDS and root["schema"] == SCHEMA,
             "unsupported source inventory schema")
    _require(root["request"] == asdict(request) and type(root["request"]) is dict,
             "source inventory differs from independent request")
    _require(root["historical_only"] is True and root["completion_authority"] is False,
             "source inventory authority flags changed")
    _require(type(root["limits"]) is dict and set(root["limits"]) == set(asdict(limits)),
             "invalid source inventory limits")
    declared = SourceInventoryLimits(**root["limits"])
    _require(all(getattr(declared, name) <= value for name, value in asdict(limits).items()),
             "source inventory exceeds caller limits")
    snapshot_cid = _cid(root["native_snapshot_cid"])
    blocks = _manifest_blocks(store, snapshot_cid, declared)
    snapshot = parse_chunked_snapshot_manifest(snapshot_cid, blocks,
        repository_id=request.repository_id, expected_commit=request.expected_commit,
        expected_tree=request.expected_tree, expected_population_cid=request.expected_population_cid,
        limits=declared.native())
    _population_bounds(snapshot, declared)
    return snapshot


def _replay(store, cid, root, request, limits):
    snapshot = _parse(store, root, request, limits)
    data_by_oid = {}
    for blob in snapshot.blobs:
        digest = hashlib.sha1() if len(blob.git_object_oid) == 40 else hashlib.sha256()
        digest.update(b"blob " + str(blob.size_bytes).encode("ascii") + b"\0")
        data = bytearray()

        def frames():
            for chunk in blob.chunks:
                raw = store.get_bytes(chunk.source_cid, chunk.size_bytes)
                digest.update(raw)
                data.extend(raw)
                yield raw

        actual_cid = cid_for_byte_chunks(frames(), max_chunk_bytes=snapshot.limits.frame_bytes)
        _require(actual_cid == blob.source_cid and len(data) == blob.size_bytes
                 and digest.hexdigest() == blob.git_object_oid, "retained whole blob does not verify")
        data_by_oid[blob.git_object_oid] = (blob.source_cid, bytes(data))
    entries = tuple(RestoredSourceEntry(e, *data_by_oid[e.git_object_oid])
                    if e.object_type == "blob" else RestoredSourceEntry(e, None, None)
                    for e in snapshot.entries)
    store.check()
    return RestoredSourceInventory(cid, snapshot, entries)


def load_source_inventory(*, cas, expected_inventory_cid, request,
                          limits=SourceInventoryLimits()):
    """Cold-replay exact historical bytes, without Git or source/workspace reads.

    Requires independent task/repository/population/root bindings. Returns data
    records only: symlink targets are bytes, gitlinks are explicit references.
    CAS availability and current workspace equality are separate responsibilities.
    """
    guard = _input_guard(request, limits)
    request = SourceInventoryRequest(**asdict(request))
    limits = SourceInventoryLimits(**asdict(limits))
    cid = _cid(expected_inventory_cid)
    store = _BoundCAS(cas, guard)
    return _replay(store, cid, store.get(cid), request, limits)


def retain_source_inventory(repository, *, cas, request, limits=SourceInventoryLimits()):
    """Retain native frames/metadata in the existing store and publish root last.

    The store must be outside source. Failures may leave immutable artifacts,
    but return no successful inventory. Native per-frame reads rehash each whole
    blob; explicit frame-read/rehash budgets bound this conservative exporter.
    Source reobservations detect ordinary drift, without an atomic mutation lease.
    """
    guard = _input_guard(request, limits)
    request = SourceInventoryRequest(**asdict(request))
    limits = SourceInventoryLimits(**asdict(limits))
    source = Path(repository).resolve(strict=True)
    store = _BoundCAS(cas, guard)
    _require(all(not Path(item[1]).is_relative_to(source) for item in store.binding[1]),
             "source store must be outside captured repository")

    def check_source():
        store.check()
        plan = preflight_committed_repository(source, expected_commit=request.expected_commit,
            expected_tree=request.expected_tree, repository_id=request.repository_id,
            max_entries=limits.max_entries, max_file_bytes=limits.max_file_bytes,
            max_total_bytes=limits.max_total_bytes, max_metadata_bytes=limits.max_metadata_bytes)
        _require(not plan.budget_violations and plan.population_cid == request.expected_population_cid,
                 "source differs from independent population or bounds")
        store.check()
        return plan

    plan = check_source()
    sizes = {e.git_object_oid: e.size_bytes for e in plan.entries if e.object_type == "blob"}
    reads = sum((size + limits.frame_bytes - 1) // limits.frame_bytes for size in sizes.values())
    work = sum(size * (1 + (size + limits.frame_bytes - 1) // limits.frame_bytes)
               for size in sizes.values())
    _require(reads <= limits.max_frame_reads and work <= limits.max_rehash_bytes,
             "source frame-read or rehash budget exceeded")
    snapshot = snapshot_chunked_repository(source, repository_id=request.repository_id,
        expected_commit=request.expected_commit, expected_tree=request.expected_tree,
        limits=limits.native())
    _require(snapshot.population_cid == request.expected_population_cid,
             "captured source population changed")
    _population_bounds(snapshot, limits)
    _require(reads == sum(len(b.chunks) for b in snapshot.blobs)
             and work == sum(b.size_bytes * (1 + len(b.chunks)) for b in snapshot.blobs),
             "native source acquisition work changed")
    snapshot_cid, blocks = snapshot.manifest_blocks()
    for blob in snapshot.blobs:
        for index, chunk in enumerate(blob.chunks):
            raw = read_chunked_blob_frame(source, snapshot,
                git_object_oid=blob.git_object_oid, chunk_index=index)
            _require(len(raw) == chunk.size_bytes and cid_for_bytes(raw) == chunk.source_cid,
                     "native source frame changed")
            store.put_bytes(chunk.source_cid, raw)
    for cid, raw in blocks.items():
        store.put(cid, json.loads(raw))
    root = dict(schema=SCHEMA, request=asdict(request), native_snapshot_cid=snapshot_cid,
                limits=asdict(limits), historical_only=True, completion_authority=False)
    inventory_cid = cid_for_structured(root)
    _replay(store, inventory_cid, root, request, limits)
    check_source()
    store.put(inventory_cid, root)
    _replay(store, inventory_cid, store.get(inventory_cid), request, limits)
    check_source()  # Includes callbacks during final publication/read.
    return SourceInventoryRef(inventory_cid, snapshot_cid, SourceInventoryRequest(**asdict(request)))
