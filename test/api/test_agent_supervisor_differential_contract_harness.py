"""Tests for the generalized hermetic differential contract harness (LPR-023)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import socket
import time
import unicodedata
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.differential_contract_harness import (
    DIFFERENTIAL_TRACE_SCHEMA,
    DIFFERENTIAL_WITNESS_SCHEMA,
    WITNESS_AUTHORIZES_REPAIR,
    WITNESS_IS_COMPLETION_EVIDENCE,
    WITNESS_IS_CORRECTNESS_EVIDENCE,
    CallableSurfaceAdapter,
    CanonicalOperationTrace,
    ContractResultNormalizer,
    DifferentialHarnessError,
    DriftKind,
    ExecutionPermit,
    FixtureEntry,
    FixtureSpec,
    HermeticNetworkError,
    InvariantDriftClassifier,
    MappingErrorClassifier,
    NormalizationRule,
    ObservationStatus,
    ProfileTraceProvider,
    SurfaceAvailability,
    SurfaceRunContext,
    TraceStep,
    VectorTraceProvider,
    build_canonical_operation_trace,
    build_fixture_spec,
    normalize_contract_result,
    run_differential_contract_harness,
    snapshot_tree,
    write_differential_witness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_MODULE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "differential_contract_harness.py"
)

_FORBIDDEN_GENERIC = re.compile(
    r"(?i)\b(?:vfs|ipfs(?!_accelerate_py)|fsspec|swissknife|swiss[_-]?knife|"
    r"ipfs_kit|board[_-]?id|board[_-]?namespace)\b"
)


# ---------------------------------------------------------------------------
# Domain profiles live only in tests (parameterization proof)
# ---------------------------------------------------------------------------


def _storage_drift_classifier() -> InvariantDriftClassifier:
    """Invariant → drift map for the locked storage-facade projection."""

    path = (DriftKind.PATH,)
    return InvariantDriftClassifier(
        invariant_to_kinds={
            "versioned_path": path,
            "unicode": path,
            "root": path,
            "traversal": path,
            "mount": path,
            "bytes_text": (DriftKind.BYTES_TEXT,),
            "stat_list": (DriftKind.STAT_LIST,),
            "atomicity": (DriftKind.RENAME_ATOMICITY,),
            "journal_replay": (DriftKind.JOURNAL,),
            "cache_pin_coherence": (DriftKind.CACHE,),
            "authorization": (DriftKind.AUTHORIZATION,),
            "backend_negotiation": (DriftKind.FALLBACK,),
            "degradation": (DriftKind.FALLBACK,),
        }
    )


def _storage_normalizer() -> ContractResultNormalizer:
    return ContractResultNormalizer(
        utf8_text_invariants=frozenset({"bytes_text"}),
        stat_alias_invariants=frozenset({"stat_list"}),
    )


def _storage_fixture() -> FixtureSpec:
    """Tree fixture matching the locked differential default recipe."""

    return build_fixture_spec(
        "storage-differential-default@1",
        (
            FixtureEntry("a", "directory", mode=0o700),
            FixtureEntry("a/x", "file", "78"),
            FixtureEntry("café", "directory", mode=0o700),
            FixtureEntry("café/data", "file", "63616665"),
            FixtureEntry("dir", "directory", mode=0o700),
            FixtureEntry("dir/child", "file", "6368696c64"),
            FixtureEntry("hello.txt", "file", "68656c6c6f0a"),
            FixtureEntry("many", "directory", mode=0o700),
            FixtureEntry("many/a", "file", "61"),
            FixtureEntry("many/b", "file", "62"),
            FixtureEntry("many/c", "file", "63"),
            FixtureEntry("secret", "file", "746f702d736563726574", mode=0o600),
        ),
    )


def _locked_storage_vectors() -> tuple[dict, ...]:
    """Canonical vectors projected from the locked source contract pack."""

    source = ("source:storage-026-acceptance",)
    return (
        {
            "vector_id": "vector:path:nfc-dot-segments",
            "operation": "path.resolve",
            "description": "Unicode and dot segments canonicalize without changing identity.",
            "request": {"path": "/cafe\u0301//draft/../data", "version": "v7"},
            "expected": {"path": "/café/data", "version": "v7"},
            "invariant_ids": (
                "invariant:versioned_path",
                "invariant:unicode",
                "invariant:traversal",
            ),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:path:root-traversal-denied",
            "operation": "path.resolve",
            "description": "Traversal above the selected root is rejected.",
            "request": {"path": "/../../etc/passwd"},
            "expected": {
                "error": {"code": "traversal_denied", "effects": "none"},
            },
            "invariant_ids": (
                "invariant:root",
                "invariant:traversal",
                "invariant:error",
            ),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:mount:component-boundary",
            "operation": "mount",
            "description": "Longest mount prefix matches only complete path components.",
            "request": {
                "mounts": ["/", "/data", "/database"],
                "path": "/data/report",
            },
            "expected": {"selected_mount": "/data"},
            "invariant_ids": (
                "invariant:mount",
                "invariant:backend_negotiation",
            ),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:write:utf8-byte-accounting",
            "operation": "write",
            "description": "Explicit UTF-8 text adapter reports encoded byte size.",
            "request": {"path": "/café.txt", "text": "é", "encoding": "utf-8"},
            "expected": {
                "committed_bytes_hex": "c3a9",
                "size": 2,
                "written": 2,
            },
            "invariant_ids": (
                "invariant:bytes_text",
                "invariant:read_write",
                "invariant:cid_size",
            ),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:seek:byte-offset",
            "operation": "seek",
            "description": "Seek offsets count bytes, not decoded characters.",
            "request": {
                "handle": "h1",
                "offset": -2,
                "whence": "end",
                "size": 9,
            },
            "expected": {"offset": 7, "content_effects": "none"},
            "invariant_ids": ("invariant:seek", "invariant:bytes_text"),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:stat:cid-size",
            "operation": "stat",
            "description": "Metadata binds CID and size to the same committed bytes.",
            "request": {"path": "/hello", "version": "v2"},
            "expected": {
                "type": "file",
                "size": 5,
                "cid_input_bytes_hex": "68656c6c6f",
                "version": "v2",
            },
            "invariant_ids": ("invariant:stat_list", "invariant:cid_size"),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:remove:non-empty",
            "operation": "remove",
            "description": "Non-recursive removal cannot partially remove a non-empty directory.",
            "request": {"path": "/dir", "recursive": False},
            "expected": {
                "error": {
                    "code": "directory_not_empty",
                    "effects": "none",
                }
            },
            "invariant_ids": (
                "invariant:directory_mutation",
                "invariant:namespace_mutation",
                "invariant:atomicity",
            ),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:journal:duplicate-replay",
            "operation": "rename",
            "description": "Replaying one committed operation identity does not duplicate effects.",
            "request": {"operation_id": "op-17", "replay_count": 2},
            "expected": {"commits": 1, "destination_entries": 1},
            "invariant_ids": (
                "invariant:journal_replay",
                "invariant:atomicity",
            ),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:version:stale-write",
            "operation": "write",
            "description": "A stale base version fails without changing the current version.",
            "request": {
                "path": "/x",
                "base_version": "v1",
                "current_version": "v2",
            },
            "expected": {
                "error": {"code": "stale_version", "effects": "none"},
                "current_version": "v2",
            },
            "invariant_ids": ("invariant:versioning", "invariant:atomicity"),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:auth:precedes-cache",
            "operation": "read",
            "description": "An unauthorized cache hit returns no content or metadata.",
            "request": {
                "path": "/secret",
                "cache": "hit",
                "authorized": False,
            },
            "expected": {
                "error": "permission_denied",
                "bytes_exposed": 0,
                "metadata_exposed": False,
            },
            "invariant_ids": (
                "invariant:authorization",
                "invariant:cache_pin_coherence",
            ),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:resource:list-limit",
            "operation": "list",
            "description": "List respects its entry bound and returns an explicit continuation.",
            "request": {"path": "/many", "max_entries": 2},
            "expected": {"entry_count": 2, "continuation_required": True},
            "invariant_ids": ("invariant:resource", "invariant:stat_list"),
            "source_contract_ids": source,
        },
        {
            "vector_id": "vector:degradation:no-silent-fallback",
            "operation": "copy",
            "description": "Required atomic cross-backend copy cannot silently degrade.",
            "request": {
                "source": "/a/x",
                "destination": "/b/x",
                "atomic": True,
            },
            "expected": {
                "error": "capability_unavailable",
                "degraded": False,
                "effects": "none",
            },
            "invariant_ids": (
                "invariant:backend_negotiation",
                "invariant:atomicity",
                "invariant:degradation",
            ),
            "source_contract_ids": source,
        },
    )


def _storage_trace_provider() -> VectorTraceProvider:
    vectors = _locked_storage_vectors()
    pack_cid = "sha256:" + hashlib.sha256(
        json.dumps(vectors, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return VectorTraceProvider(vectors=vectors, contract_pack_cid=pack_cid)


def _storage_trace(vector_ids=None) -> CanonicalOperationTrace:
    return _storage_trace_provider().build_trace(vector_ids=vector_ids)


def _adapter(
    surface_id,
    executor,
    *,
    family="storage",
    availability=SurfaceAvailability.REAL,
    packages=(),
):
    return CallableSurfaceAdapter(
        surface_id=surface_id,
        family=family,
        executor=executor,
        implementation=f"tests.{surface_id}",
        public_surface="python",
        availability=availability,
        package_names=packages,
    )


def _run(
    surfaces,
    *,
    trace=None,
    fixture=None,
    temp_parent=None,
    drift=None,
    normalizer=None,
    permit=None,
    goal_id="LPR-023",
    schema=DIFFERENTIAL_WITNESS_SCHEMA,
):
    return run_differential_contract_harness(
        surfaces,
        trace=trace or _storage_trace(),
        fixture=fixture or _storage_fixture(),
        normalizer=normalizer or _storage_normalizer(),
        drift_classifier=drift or _storage_drift_classifier(),
        error_classifier=MappingErrorClassifier(),
        permit=permit,
        goal_id=goal_id,
        schema=schema,
        temp_parent=temp_parent,
    )


# ---------------------------------------------------------------------------
# In-memory non-domain key-value surface (parameterization proof)
# ---------------------------------------------------------------------------


class InMemoryKeyValueSurface:
    """In-memory non-tree adapter used to prove the harness is not storage-bound."""

    def __init__(self, store: dict[str, bytes] | None = None, *, seed_drift: bool = False):
        self.store = dict(store or {"alpha": b"one", "beta": b"two"})
        self.seed_drift = seed_drift

    def __call__(self, step: TraceStep, _context: SurfaceRunContext):
        op = step.operation
        request = dict(step.request)
        if op == "kv.get":
            key = str(request["key"])
            if key not in self.store:
                return {"error": {"code": "not_found"}}
            value = self.store[key]
            if self.seed_drift and key == "alpha":
                value = b"WRONG"
            return {"key": key, "value_hex": value.hex(), "size": len(value)}
        if op == "kv.put":
            key = str(request["key"])
            raw = bytes.fromhex(str(request["value_hex"]))
            self.store[key] = raw
            if self.seed_drift:
                return {"key": key, "written": len(raw) + 1}
            return {"key": key, "written": len(raw)}
        if op == "kv.delete":
            key = str(request["key"])
            existed = key in self.store
            self.store.pop(key, None)
            return {"key": key, "deleted": existed}
        return {"error": {"code": "unsupported"}}


def _kv_vectors(*, drift_target: str | None = None) -> tuple[dict, ...]:
    return (
        {
            "vector_id": "vector:kv:get-alpha",
            "operation": "kv.get",
            "description": "Get returns exact committed bytes.",
            "request": {"key": "alpha"},
            "expected": {"key": "alpha", "value_hex": "6f6e65", "size": 3},
            "invariant_ids": ("invariant:value_bytes", "invariant:key_identity"),
            "source_contract_ids": ("source:kv-acceptance",),
        },
        {
            "vector_id": "vector:kv:put-gamma",
            "operation": "kv.put",
            "description": "Put reports written byte count.",
            "request": {"key": "gamma", "value_hex": "78797a"},
            "expected": {"key": "gamma", "written": 3},
            "invariant_ids": ("invariant:value_bytes",),
            "source_contract_ids": ("source:kv-acceptance",),
        },
        {
            "vector_id": "vector:kv:delete-missing",
            "operation": "kv.delete",
            "description": "Delete of missing key is idempotent.",
            "request": {"key": "missing"},
            "expected": {"key": "missing", "deleted": False},
            "invariant_ids": ("invariant:delete_idempotent",),
            "source_contract_ids": ("source:kv-acceptance",),
        },
    )


def _kv_fixture() -> FixtureSpec:
    # Minimal empty-ish fixture so isolation still applies; store is in-memory.
    return FixtureSpec(
        fixture_id="kv-memory-empty@1",
        entries=(FixtureEntry("marker", "file", "00"),),
    )


def _kv_drift_classifier() -> InvariantDriftClassifier:
    return InvariantDriftClassifier(
        invariant_to_kinds={
            "value_bytes": (DriftKind.BYTES_TEXT,),
            "key_identity": (DriftKind.PATH,),
            "delete_idempotent": (DriftKind.RESULT,),
        }
    )


# ---------------------------------------------------------------------------
# Generic engine constraints
# ---------------------------------------------------------------------------


def test_generic_module_has_no_domain_product_literals():
    text = HARNESS_MODULE.read_text(encoding="utf-8")
    matches = [
        (index, line)
        for index, line in enumerate(text.splitlines(), start=1)
        if _FORBIDDEN_GENERIC.search(line)
    ]
    assert matches == [], f"forbidden domain literals in harness: {matches[:5]}"


def test_authority_flags_are_non_authoritative():
    assert WITNESS_IS_COMPLETION_EVIDENCE is False
    assert WITNESS_IS_CORRECTNESS_EVIDENCE is False
    assert WITNESS_AUTHORIZES_REPAIR is False


# ---------------------------------------------------------------------------
# Trace / fixture identity
# ---------------------------------------------------------------------------


def test_canonical_trace_is_finite_deterministic_and_selectable():
    provider = _storage_trace_provider()
    first = provider.build_trace()
    second = provider.build_trace()

    assert first == second
    assert first.content_id == second.content_id
    assert len(first.steps) == 12
    assert len({step.vector_id for step in first.steps}) == len(first.steps)
    assert first.to_record() == second.to_record()
    assert first.schema == DIFFERENTIAL_TRACE_SCHEMA

    requested = (
        "vector:stat:cid-size",
        "vector:path:nfc-dot-segments",
    )
    selected = provider.build_trace(vector_ids=requested)
    assert tuple(step.vector_id for step in selected.steps) == requested
    assert selected.contract_pack_cid == first.contract_pack_cid

    with pytest.raises(DifferentialHarnessError, match="cannot be empty"):
        provider.build_trace(vector_ids=())
    with pytest.raises(DifferentialHarnessError, match="unknown contract"):
        provider.build_trace(vector_ids=("vector:missing",))
    with pytest.raises(DifferentialHarnessError, match="duplicates"):
        provider.build_trace(
            vector_ids=(
                "vector:stat:cid-size",
                "vector:stat:cid-size",
            )
        )
    with pytest.raises(DifferentialHarnessError, match="required"):
        build_canonical_operation_trace()


def test_fixture_identity_materialization_and_containment(tmp_path):
    fixture = _storage_fixture()
    left = tmp_path / "left"
    right = tmp_path / "right"

    left_cid = fixture.materialize(left)
    right_cid = fixture.materialize(right)

    assert fixture.content_id.startswith("sha256:")
    assert left_cid == right_cid
    assert left_cid == snapshot_tree(left).content_id
    assert (left / "secret").read_bytes() == b"top-secret"
    assert (left / "café" / "data").read_bytes() == b"cafe"

    step = _storage_trace().steps[0]
    context = SurfaceRunContext(root=left, fixture=fixture, step=step)
    assert context.resolve_path("/hello.txt") == left / "hello.txt"
    with pytest.raises(PermissionError, match="traversal"):
        context.resolve_path("../outside")

    outside = tmp_path / "outside"
    outside.mkdir()
    (left / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="escaped"):
        context.resolve_path("escape/file")


def test_fixture_rejects_noncanonical_or_unsafe_recipes():
    invalid_entries = (
        {"path": "../outside", "kind": "file", "content_hex": "00"},
        {"path": "a//b", "kind": "file", "content_hex": "00"},
        {"path": unicodedata.normalize("NFD", "café"), "kind": "directory"},
        {"path": "upper", "kind": "file", "content_hex": "AA"},
        {"path": "bad-hex", "kind": "file", "content_hex": "zz"},
        {"path": "bad-mode", "kind": "file", "content_hex": "00", "mode": 0o1000},
    )
    for kwargs in invalid_entries:
        with pytest.raises(DifferentialHarnessError):
            FixtureEntry(**kwargs)

    with pytest.raises(DifferentialHarnessError, match="must be unique"):
        FixtureSpec(
            fixture_id="duplicate",
            entries=(
                FixtureEntry("x", "file", "00"),
                FixtureEntry("x", "file", "01"),
            ),
        )
    with pytest.raises(DifferentialHarnessError, match="cannot contain"):
        FixtureSpec(
            fixture_id="file-parent",
            entries=(
                FixtureEntry("x", "file", "00"),
                FixtureEntry("x/y", "file", "01"),
            ),
        )


# ---------------------------------------------------------------------------
# Surface families / agreement without false drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family",
    ("python_facade", "http_gateway", "cli", "manager", "handler"),
)
def test_real_surface_families_share_contract_without_false_drift(family, tmp_path):
    def compatible(step, _context):
        result = dict(step.expected)
        result["implementation_metadata"] = {
            "family": family,
            "ignored_by_contract_projection": True,
        }
        if step.vector_id == "vector:stat:cid-size":
            result["kind"] = result.pop("type")
            result["length"] = result.pop("size")
        return {
            "ok": True,
            "result": result,
            "request_id": f"{family}-request",
        }

    witness = _run(
        (
            _adapter(
                f"real-{family}",
                compatible,
                family=family,
                packages=("pytest", "definitely-not-an-installed-distribution"),
            ),
        ),
        temp_parent=tmp_path,
    )

    assert witness.schema == DIFFERENTIAL_WITNESS_SCHEMA
    assert witness.authoritative_agreement is True
    assert witness.findings == ()
    assert witness.authoritative_surface_ids == (f"real-{family}",)
    run = witness.surface_runs[0]
    assert run.authoritative is True
    assert len(run.observations) == len(witness.trace.steps)
    assert all(observation.contract_match for observation in run.observations)
    assert all(observation.cleanup.succeeded for observation in run.observations)
    assert any(
        observation.status is ObservationStatus.ERROR
        for observation in run.observations
    )
    assert run.runtime.packages["pytest"] != "<unavailable>"
    assert (
        run.runtime.packages["definitely-not-an-installed-distribution"]
        == "<unavailable>"
    )
    assert run.runtime.content_id.startswith("sha256:")
    assert run.implementation_identity.content_id.startswith("sha256:")


def test_normalization_is_closed_and_invariant_scoped():
    normalizer = _storage_normalizer()
    bytes_step = TraceStep(
        vector_id="test:bytes",
        operation="read",
        description="bytes normalization",
        request={},
        expected={"payload": {"bytes_hex": "c3a9"}},
        invariant_ids=("invariant:bytes_text",),
    )
    stat_step = TraceStep(
        vector_id="test:stat",
        operation="stat",
        description="stat normalization",
        request={},
        expected={"type": "file", "size": 2},
        invariant_ids=("invariant:stat_list",),
    )
    path_step = TraceStep(
        vector_id="test:path",
        operation="path.resolve",
        description="path preservation",
        request={},
        expected={"path": "/café"},
        invariant_ids=("invariant:unicode",),
    )

    assert normalize_contract_result(
        bytes_step, {"payload": bytearray(b"\xc3\xa9")}, normalizer=normalizer
    ) == {"payload": {"bytes_hex": "c3a9"}}
    assert normalize_contract_result(
        bytes_step,
        {"payload": {"text": "é", "encoding": "utf-8"}},
        normalizer=normalizer,
    ) == {"payload": {"bytes_hex": "c3a9"}}
    assert normalize_contract_result(
        stat_step, {"kind": "file", "length": 2}, normalizer=normalizer
    ) == {"type": "file", "size": 2}

    nfd_path = unicodedata.normalize("NFD", "/café")
    assert normalize_contract_result(
        path_step, {"path": nfd_path}, normalizer=normalizer
    ) == {"path": nfd_path}
    assert normalize_contract_result(
        path_step,
        {"kind": "file", "length": 2},
        normalizer=normalizer,
    ) == {"kind": "file", "length": 2}
    assert normalize_contract_result(
        bytes_step,
        {"payload": {"text": "é", "encoding": "utf-8"}},
        rules=(),
        normalizer=normalizer,
    ) == {"payload": {"text": "é", "encoding": "utf-8"}}
    assert "path_syntax" not in {rule.value for rule in NormalizationRule}


def test_seeded_surface_exposes_every_required_drift_class(tmp_path):
    def seeded(step, _context):
        result = dict(step.expected)
        mutations = {
            "vector:path:nfc-dot-segments": lambda value: value.update(
                path="/wrong"
            ),
            "vector:write:utf8-byte-accounting": lambda value: value.update(
                written=1
            ),
            "vector:stat:cid-size": lambda value: value.update(size=6),
            "vector:journal:duplicate-replay": lambda value: value.update(
                commits=2, destination_entries=2
            ),
            "vector:auth:precedes-cache": lambda value: value.update(
                bytes_exposed=1, metadata_exposed=True
            ),
            "vector:resource:list-limit": lambda value: value.update(
                entry_count=3
            ),
        }
        if step.vector_id == "vector:remove:non-empty":
            return {"removed": True}
        if step.vector_id == "vector:degradation:no-silent-fallback":
            return {"copied": True, "degraded": True, "effects": "changed"}
        mutation = mutations.get(step.vector_id)
        if mutation is not None:
            mutation(result)
        return result

    witness = _run(
        (_adapter("seeded-real", seeded),),
        temp_parent=tmp_path,
    )
    found = {
        kind
        for finding in witness.findings
        for kind in finding.kinds
        if finding.authoritative
    }

    assert {
        DriftKind.PATH,
        DriftKind.BYTES_TEXT,
        DriftKind.STAT_LIST,
        DriftKind.RENAME_ATOMICITY,
        DriftKind.JOURNAL,
        DriftKind.CACHE,
        DriftKind.AUTHORIZATION,
        DriftKind.FALLBACK,
        DriftKind.SILENT_SUCCESS,
    } <= found
    assert witness.authoritative_agreement is False
    silent_vectors = {
        finding.vector_id
        for finding in witness.findings
        if DriftKind.SILENT_SUCCESS in finding.kinds
    }
    assert "vector:remove:non-empty" in silent_vectors
    assert "vector:degradation:no-silent-fallback" in silent_vectors


def test_exceptions_record_exact_identity_and_compatible_error_code(tmp_path):
    trace = TraceStep(
        vector_id="test:permission",
        operation="read",
        description="permission error identity",
        request={"path": "/secret"},
        expected={"error": "permission_denied"},
        invariant_ids=("invariant:authorization",),
    )

    def denied(_step, _context):
        raise PermissionError(13, "fixture denied", "/secret")

    witness = _run(
        (_adapter("raises", denied),),
        trace=CanonicalOperationTrace(
            steps=(trace,),
            contract_pack_cid="sha256:test-contract-pack",
        ),
        temp_parent=tmp_path,
    )
    observation = witness.surface_runs[0].observations[0]

    assert observation.status is ObservationStatus.ERROR
    assert observation.contract_match is True
    assert observation.error is not None
    assert observation.error.code == "permission_denied"
    assert observation.error.exception_module == "builtins"
    assert observation.error.exception_type == "PermissionError"
    assert "fixture denied" in (observation.error.message or "")
    assert observation.error.errno == 13
    assert observation.error.content_id.startswith("sha256:")
    assert witness.findings == ()


def test_mock_unavailable_and_unknown_surfaces_are_explicit_non_authorities(tmp_path):
    trace = _storage_trace(vector_ids=("vector:path:nfc-dot-segments",))
    executed = []

    def mock_drift(step, _context):
        executed.append(step.vector_id)
        return {"path": "/bad", "version": "v7"}

    unavailable = CallableSurfaceAdapter.unavailable(
        "missing-handler",
        "handler",
        implementation="optional.missing.Handler",
        reason="optional dependency is not installed",
    )
    unknown = CallableSurfaceAdapter.unknown(
        "unknown-backend",
        "backend",
        implementation="optional.unknown.Backend",
        reason="backend capability is not classified",
    )
    witness = _run(
        (
            _adapter("real-storage", lambda step, _context: step.expected),
            _adapter(
                "mock-bucket",
                mock_drift,
                family="bucket",
                availability=SurfaceAvailability.MOCK,
            ),
            unavailable,
            unknown,
        ),
        trace=trace,
        temp_parent=tmp_path,
    )

    assert executed == ["vector:path:nfc-dot-segments"]
    assert witness.authoritative_surface_ids == ("real-storage",)
    assert witness.non_authoritative_surface_ids == ("mock-bucket",)
    assert witness.unavailable_surface_ids == ("missing-handler",)
    assert witness.unknown_surface_ids == ("unknown-backend",)
    assert witness.authoritative_agreement is True
    assert len(witness.findings) == 1
    assert witness.findings[0].authoritative is False
    missing_run = witness.surface_runs[2]
    assert missing_run.observations == ()
    assert missing_run.unavailable_reason == "optional dependency is not installed"
    assert missing_run.authoritative is False
    assert witness.surface_runs[3].observations == ()


def test_every_case_gets_identical_fresh_fixture_and_cleanup(tmp_path):
    trace = _storage_trace(
        vector_ids=(
            "vector:path:nfc-dot-segments",
            "vector:write:utf8-byte-accounting",
            "vector:stat:cid-size",
        )
    )

    def mutating(step, context):
        context.resolve_path("mutation").write_bytes(step.vector_id.encode())
        return step.expected

    witness = _run(
        (_adapter("mutating-storage", mutating),),
        trace=trace,
        temp_parent=tmp_path,
    )
    observations = witness.surface_runs[0].observations

    assert len({item.fixture_before_cid for item in observations}) == 1
    assert all(
        item.fixture_spec_cid == witness.fixture.content_id for item in observations
    )
    assert all(
        item.fixture_after_cid != item.fixture_before_cid for item in observations
    )
    assert all(item.cleanup.succeeded for item in observations)
    assert all(
        item.cleanup.before_cleanup_cid == item.fixture_after_cid
        for item in observations
    )
    assert all(not Path(item.cleanup.root).exists() for item in observations)
    assert list(tmp_path.iterdir()) == []


def test_async_surface_inside_running_loop_and_network_is_denied(tmp_path):
    async def scenario():
        async def compatible(step, _context):
            await asyncio.sleep(0)
            return step.expected

        return _run(
            (_adapter("async-storage", compatible),),
            trace=_storage_trace(vector_ids=("vector:seek:byte-offset",)),
            temp_parent=tmp_path,
        )

    async_witness = asyncio.run(scenario())
    assert async_witness.authoritative_agreement is True

    network_step = TraceStep(
        vector_id="test:network-denied",
        operation="read",
        description="network calls are hermetically denied",
        request={},
        expected={"error": "permission_denied"},
        invariant_ids=("invariant:authorization",),
    )

    def attempts_network(_step, _context):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)

    network_witness = _run(
        (_adapter("networking-storage", attempts_network),),
        trace=CanonicalOperationTrace(
            steps=(network_step,),
            contract_pack_cid="sha256:test-contract-pack",
        ),
        temp_parent=tmp_path,
    )
    observation = network_witness.surface_runs[0].observations[0]
    assert observation.contract_match is True
    assert observation.error is not None
    assert observation.error.exception_type == HermeticNetworkError.__name__
    assert "network access is disabled" in (observation.error.message or "")


def test_witness_records_result_cids_and_is_written_atomically(tmp_path):
    witness = _run(
        (_adapter("serializable-storage", lambda step, _context: step.expected),),
        trace=_storage_trace(vector_ids=("vector:stat:cid-size",)),
        temp_parent=tmp_path,
    )
    destination = tmp_path / "evidence" / "witness.json"

    assert write_differential_witness(witness, destination) == destination
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    observation = persisted["surface_runs"][0]["observations"][0]

    assert persisted == witness.to_record()
    assert persisted["cid"] == witness.content_id
    assert persisted["trace"]["cid"] == witness.trace.content_id
    assert persisted["fixture"]["cid"] == witness.fixture.content_id
    assert observation["request_cid"].startswith("sha256:")
    assert observation["raw_result_cid"].startswith("sha256:")
    assert observation["normalized_result_cid"].startswith("sha256:")
    assert observation["fixture_before_cid"].startswith("sha256:")
    assert observation["fixture_after_cid"].startswith("sha256:")
    assert observation["cleanup"]["cid"].startswith("sha256:")
    assert persisted["authority"] == {
        "completion": False,
        "correctness": False,
        "repair": False,
    }


def test_surface_and_run_validation_fail_closed(tmp_path):
    with pytest.raises(DifferentialHarnessError, match="at least one"):
        _run((), temp_parent=tmp_path)
    with pytest.raises(DifferentialHarnessError, match="unique"):
        _run(
            (
                _adapter("same", lambda step, _context: step.expected),
                _adapter("same", lambda step, _context: step.expected),
            ),
            temp_parent=tmp_path,
        )
    with pytest.raises(DifferentialHarnessError, match="cannot have"):
        CallableSurfaceAdapter(
            surface_id="invalid-unavailable",
            family="storage",
            executor=lambda step, _context: step.expected,
            implementation="tests.invalid",
            availability=SurfaceAvailability.UNAVAILABLE,
            unavailable_reason="missing",
        )
    with pytest.raises(DifferentialHarnessError, match="require an executor"):
        CallableSurfaceAdapter(
            surface_id="invalid-real",
            family="storage",
            executor=None,
            implementation="tests.invalid",
        )
    with pytest.raises(DifferentialHarnessError, match="temp_parent"):
        _run(
            (_adapter("real", lambda step, _context: step.expected),),
            trace=_storage_trace(vector_ids=("vector:stat:cid-size",)),
            temp_parent=tmp_path / "missing",
        )
    with pytest.raises(DifferentialHarnessError, match="fixture is required"):
        run_differential_contract_harness(
            (_adapter("real", lambda step, _context: step.expected),),
            trace=_storage_trace(vector_ids=("vector:stat:cid-size",)),
        )


def test_timeout_and_over_budget_are_rejected(tmp_path):
    def slow(_step, _context):
        time.sleep(2.0)
        return {"ok": True}

    step = TraceStep(
        vector_id="test:timeout",
        operation="read",
        description="timeout rejection",
        request={},
        expected={"error": "deadline_exceeded"},
        invariant_ids=(),
    )
    permit = ExecutionPermit(timeout_seconds=0.05, max_steps=1, max_fixture_bytes=1024)
    witness = _run(
        (_adapter("slow", slow),),
        trace=CanonicalOperationTrace(
            steps=(step,),
            contract_pack_cid="sha256:test",
        ),
        permit=permit,
        temp_parent=tmp_path,
    )
    observation = witness.surface_runs[0].observations[0]
    assert observation.status is ObservationStatus.ERROR
    assert observation.error is not None
    assert observation.error.exception_type == "TimeoutError"

    huge = FixtureSpec(
        fixture_id="huge",
        entries=(FixtureEntry("blob", "file", "aa" * 100),),
    )
    with pytest.raises(DifferentialHarnessError, match="max_fixture_bytes"):
        _run(
            (_adapter("real", lambda step, _context: step.expected),),
            trace=_storage_trace(vector_ids=("vector:stat:cid-size",)),
            fixture=huge,
            permit=ExecutionPermit(max_fixture_bytes=10),
            temp_parent=tmp_path,
        )

    with pytest.raises(DifferentialHarnessError, match="max_steps"):
        _run(
            (_adapter("real", lambda step, _context: step.expected),),
            permit=ExecutionPermit(max_steps=1),
            temp_parent=tmp_path,
        )


def test_storage_canonical_vectors_are_equivalent_under_compatible_adapter(tmp_path):
    """Locked storage vectors agree with a projection-compatible surface."""

    def compatible(step, _context):
        return step.expected

    first = _run(
        (_adapter("storage-a", compatible, family="python_facade"),),
        temp_parent=tmp_path,
    )
    second = _run(
        (_adapter("storage-b", compatible, family="http_gateway"),),
        temp_parent=tmp_path,
    )
    assert first.authoritative_agreement is True
    assert second.authoritative_agreement is True
    assert first.trace.content_id == second.trace.content_id
    assert first.fixture.content_id == second.fixture.content_id
    assert len(first.trace.steps) == 12
    # Observation projections match the locked expected semantics.
    for left, right in zip(
        first.surface_runs[0].observations, second.surface_runs[0].observations
    ):
        assert left.canonical_projection == right.canonical_projection
        assert left.contract_match is True
        assert right.contract_match is True


def test_in_memory_non_storage_adapter_detects_seeded_drift_without_false_mismatch(
    tmp_path,
):
    vectors = _kv_vectors()
    provider = VectorTraceProvider(
        vectors=vectors,
        contract_pack_cid="sha256:kv-profile",
    )
    fixture = _kv_fixture()
    normalizer = ContractResultNormalizer(
        utf8_text_invariants=frozenset(),
        stat_alias_invariants=frozenset(),
    )
    drift = _kv_drift_classifier()

    good = InMemoryKeyValueSurface(seed_drift=False)
    bad = InMemoryKeyValueSurface(seed_drift=True)

    good_witness = run_differential_contract_harness(
        (
            _adapter("kv-good", good, family="memory_kv"),
        ),
        trace=provider.build_trace(),
        fixture=fixture,
        normalizer=normalizer,
        drift_classifier=drift,
        goal_id="kv-hermetic",
        temp_parent=tmp_path,
    )
    assert good_witness.authoritative_agreement is True
    assert good_witness.findings == ()
    assert all(
        obs.contract_match for obs in good_witness.surface_runs[0].observations
    )

    bad_witness = run_differential_contract_harness(
        (
            _adapter("kv-bad", bad, family="memory_kv"),
        ),
        trace=provider.build_trace(),
        fixture=fixture,
        normalizer=normalizer,
        drift_classifier=drift,
        goal_id="kv-hermetic",
        temp_parent=tmp_path,
    )
    assert bad_witness.authoritative_agreement is False
    found_vectors = {finding.vector_id for finding in bad_witness.findings}
    assert "vector:kv:get-alpha" in found_vectors
    assert "vector:kv:put-gamma" in found_vectors
    # Non-seeded vector remains clean — no false mismatch.
    clean = {
        finding.vector_id
        for finding in bad_witness.findings
        if finding.vector_id == "vector:kv:delete-missing"
    }
    assert clean == set()
    assert any(
        DriftKind.BYTES_TEXT in finding.kinds for finding in bad_witness.findings
    )


def test_profile_trace_provider_reads_program_contract_profile_vectors(tmp_path):
    """ProfileTraceProvider accepts objects exposing vectors + content_id."""

    class _MiniProfile:
        def __init__(self):
            self.vectors = _kv_vectors()[:1]
            self.content_id = "sha256:mini-profile"

    provider = ProfileTraceProvider(profile=_MiniProfile())
    trace = provider.build_trace()
    assert len(trace.steps) == 1
    assert trace.steps[0].vector_id == "vector:kv:get-alpha"
    assert trace.contract_pack_cid == "sha256:mini-profile"

    witness = run_differential_contract_harness(
        (_adapter("kv", InMemoryKeyValueSurface(), family="memory_kv"),),
        trace_provider=provider,
        fixture=_kv_fixture(),
        normalizer=ContractResultNormalizer(
            utf8_text_invariants=frozenset(),
            stat_alias_invariants=frozenset(),
        ),
        drift_classifier=_kv_drift_classifier(),
        temp_parent=tmp_path,
    )
    assert witness.authoritative_agreement is True
