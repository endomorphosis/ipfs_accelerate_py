import asyncio
import json
import socket
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.vfs_contract_pack import (
    VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
    VfsInvariantKind,
    VfsOperation,
)
from ipfs_accelerate_py.agent_supervisor.vfs_differential_harness import (
    REQUIRED_PUBLIC_SURFACES,
    VFS_DIFFERENTIAL_EVIDENCE_KINDS,
    VFS_DIFFERENTIAL_GOAL_ID,
    VFS_DIFFERENTIAL_OBJECTIVE_REVISION,
    VFS_DIFFERENTIAL_PACKET_GOAL_IDS,
    VFS_DIFFERENTIAL_TASK_ID,
    VFS_DIFFERENTIAL_WITNESS_SCHEMA,
    CallableSurfaceAdapter,
    CanonicalOperationTrace,
    DriftKind,
    FixtureEntry,
    FixtureSpec,
    HermeticNetworkError,
    NormalizationRule,
    ObservationStatus,
    PublicSurfaceKind,
    SurfaceAvailability,
    SurfaceFamily,
    SurfaceRunContext,
    TraceStep,
    VfsDifferentialHarnessError,
    build_canonical_operation_trace,
    build_default_fixture,
    normalize_contract_result,
    run_vfs_differential_harness,
    snapshot_tree,
    write_differential_witness,
)


def _adapter(
    surface_id,
    executor,
    *,
    family=SurfaceFamily.VFS,
    availability=SurfaceAvailability.REAL,
    packages=(),
    public_surface=PublicSurfaceKind.PYTHON,
):
    return CallableSurfaceAdapter(
        surface_id=surface_id,
        family=family,
        executor=executor,
        implementation=f"tests.{surface_id}",
        public_surface=public_surface,
        availability=availability,
        package_names=packages,
    )


def _one_step(vector_id):
    return build_canonical_operation_trace(vector_ids=(vector_id,))


def test_canonical_trace_is_finite_deterministic_and_selectable():
    first = build_canonical_operation_trace()
    second = build_canonical_operation_trace()

    assert first == second
    assert first.content_id == second.content_id
    assert len(first.steps) == 12
    assert first.operation_matrix_schema == VFS_CANONICAL_OPERATION_MATRIX_SCHEMA
    assert len({step.vector_id for step in first.steps}) == len(first.steps)
    assert first.to_record() == second.to_record()

    requested = (
        "vector:stat:cid-size",
        "vector:path:nfc-dot-segments",
    )
    selected = build_canonical_operation_trace(vector_ids=requested)
    assert tuple(step.vector_id for step in selected.steps) == requested
    assert selected.contract_pack_cid == first.contract_pack_cid

    with pytest.raises(VfsDifferentialHarnessError, match="cannot be empty"):
        build_canonical_operation_trace(vector_ids=())
    with pytest.raises(VfsDifferentialHarnessError, match="unknown contract"):
        build_canonical_operation_trace(vector_ids=("vector:missing",))
    with pytest.raises(VfsDifferentialHarnessError, match="duplicates"):
        build_canonical_operation_trace(
            vector_ids=(
                "vector:stat:cid-size",
                "vector:stat:cid-size",
            )
        )


def test_packet_evidence_covers_every_public_surface_with_exact_cid_bindings(
    tmp_path,
):
    def compatible_transport(public_surface):
        def execute(step, _context):
            if public_surface is PublicSurfaceKind.PYTHON:
                return step.expected
            if public_surface in {
                PublicSurfaceKind.CLI,
                PublicSurfaceKind.LIBP2P,
            }:
                return {"success": True, "data": step.expected}
            if public_surface in {
                PublicSurfaceKind.MCP,
                PublicSurfaceKind.MCP_PLUS_PLUS,
            }:
                return {"ok": True, "result": step.expected}
            if public_surface is PublicSurfaceKind.HTTP:
                return {"status": 200, "body": step.expected}
            return step.expected

        return execute

    adapters = tuple(
        _adapter(
            f"real-{public_surface.value.replace('+', 'p')}",
            compatible_transport(public_surface),
            family=(
                SurfaceFamily.MANAGER
                if public_surface is PublicSurfaceKind.BACKEND
                else SurfaceFamily.HANDLER
            ),
            public_surface=public_surface,
        )
        for public_surface in REQUIRED_PUBLIC_SURFACES
    )
    witness = run_vfs_differential_harness(adapters, temp_parent=tmp_path)
    record = witness.to_record()

    assert witness.schema == VFS_DIFFERENTIAL_WITNESS_SCHEMA
    assert witness.goal_id == VFS_DIFFERENTIAL_GOAL_ID == "VFS-G091"
    assert witness.task_id == VFS_DIFFERENTIAL_TASK_ID == "VFS-077"
    assert witness.objective_revision == VFS_DIFFERENTIAL_OBJECTIVE_REVISION
    assert witness.goal_ids == VFS_DIFFERENTIAL_PACKET_GOAL_IDS == (
        "VFS-G091",
        "VFS-G158",
    )
    assert witness.evidence_kinds == VFS_DIFFERENTIAL_EVIDENCE_KINDS == (
        "vfs/differential-contract-witness@1",
        "vfs/canonical-operation-matrix@1",
    )
    assert witness.trace.operation_matrix_schema == (
        VFS_CANONICAL_OPERATION_MATRIX_SCHEMA
    )
    assert witness.observed_public_surfaces == REQUIRED_PUBLIC_SURFACES
    assert witness.missing_public_surfaces == ()
    assert record["coverage"]["public_surface_coverage_complete"] is True
    assert witness.authoritative_agreement is True
    assert witness.findings == ()

    bindings = witness.bindings
    assert bindings.contract_pack_cid == witness.trace.contract_pack_cid
    assert bindings.operation_trace_cid == witness.trace.content_id
    assert bindings.fixture_spec_cid == witness.fixture.content_id
    assert bindings.fixture_snapshot_cids == tuple(
        sorted(
            {
                observation.fixture_before_cid
                for run in witness.surface_runs
                for observation in run.observations
            }
        )
    )
    assert bindings.toolchain_cids == {
        run.surface_id: run.runtime.content_id for run in witness.surface_runs
    }
    assert bindings.implementation_cids == {
        run.surface_id: run.implementation_identity.content_id
        for run in witness.surface_runs
    }
    assert bindings.surface_run_cids == {
        run.surface_id: run.content_id for run in witness.surface_runs
    }
    assert bindings.content_id.startswith("sha256:")
    assert record["bindings"] == bindings.to_record()

    with pytest.raises(VfsDifferentialHarnessError, match="binding CID"):
        replace(bindings, fixture_spec_cid="sha256:tampered-fixture")


def test_fixture_identity_materialization_and_containment(tmp_path):
    fixture = build_default_fixture()
    left = tmp_path / "left"
    right = tmp_path / "right"

    left_cid = fixture.materialize(left)
    right_cid = fixture.materialize(right)

    assert fixture.content_id.startswith("sha256:")
    assert left_cid == right_cid
    assert left_cid == snapshot_tree(left).content_id
    assert (left / "secret").read_bytes() == b"top-secret"
    assert (left / "café" / "data").read_bytes() == b"cafe"

    step = build_canonical_operation_trace().steps[0]
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
        with pytest.raises(VfsDifferentialHarnessError):
            FixtureEntry(**kwargs)

    with pytest.raises(VfsDifferentialHarnessError, match="must be unique"):
        FixtureSpec(
            fixture_id="duplicate",
            entries=(
                FixtureEntry("x", "file", "00"),
                FixtureEntry("x", "file", "01"),
            ),
        )
    with pytest.raises(VfsDifferentialHarnessError, match="cannot contain"):
        FixtureSpec(
            fixture_id="file-parent",
            entries=(
                FixtureEntry("x", "file", "00"),
                FixtureEntry("x/y", "file", "01"),
            ),
        )


@pytest.mark.parametrize("family", tuple(SurfaceFamily))
def test_real_surface_families_share_contract_without_false_drift(
    family, tmp_path
):
    def compatible(step, _context):
        result = dict(step.expected)
        result["implementation_metadata"] = {
            "family": family.value,
            "ignored_by_contract_projection": True,
        }
        if step.vector_id == "vector:stat:cid-size":
            result["kind"] = result.pop("type")
            result["length"] = result.pop("size")
        return {
            "ok": True,
            "result": result,
            "request_id": f"{family.value}-request",
        }

    witness = run_vfs_differential_harness(
        (
            _adapter(
                f"real-{family.value}",
                compatible,
                family=family,
                packages=("pytest", "definitely-not-an-installed-distribution"),
            ),
        ),
        temp_parent=tmp_path,
    )

    assert witness.schema == VFS_DIFFERENTIAL_WITNESS_SCHEMA
    assert witness.authoritative_agreement is True
    assert witness.findings == ()
    assert witness.authoritative_surface_ids == (f"real-{family.value}",)
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
    bytes_step = TraceStep(
        vector_id="test:bytes",
        operation=VfsOperation.READ,
        description="bytes normalization",
        request={},
        expected={"payload": {"bytes_hex": "c3a9"}},
        invariant_ids=(VfsInvariantKind.BYTES_TEXT.value,),
    )
    stat_step = TraceStep(
        vector_id="test:stat",
        operation=VfsOperation.STAT,
        description="stat normalization",
        request={},
        expected={"type": "file", "size": 2},
        invariant_ids=(VfsInvariantKind.STAT_LIST.value,),
    )
    path_step = TraceStep(
        vector_id="test:path",
        operation=VfsOperation.PATH_RESOLVE,
        description="path preservation",
        request={},
        expected={"path": "/café"},
        invariant_ids=(VfsInvariantKind.UNICODE.value,),
    )

    assert normalize_contract_result(
        bytes_step, {"payload": bytearray(b"\xc3\xa9")}
    ) == {"payload": {"bytes_hex": "c3a9"}}
    assert normalize_contract_result(
        bytes_step,
        {"payload": {"text": "é", "encoding": "utf-8"}},
    ) == {"payload": {"bytes_hex": "c3a9"}}
    assert normalize_contract_result(
        stat_step, {"kind": "file", "length": 2}
    ) == {"type": "file", "size": 2}

    nfd_path = unicodedata.normalize("NFD", "/café")
    assert normalize_contract_result(path_step, {"path": nfd_path}) == {
        "path": nfd_path
    }
    assert normalize_contract_result(
        path_step,
        {"kind": "file", "length": 2},
    ) == {"kind": "file", "length": 2}
    assert normalize_contract_result(
        bytes_step,
        {"payload": {"text": "é", "encoding": "utf-8"}},
        rules=(),
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

    witness = run_vfs_differential_harness(
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
        operation=VfsOperation.READ,
        description="permission error identity",
        request={"path": "/secret"},
        expected={"error": "permission_denied"},
        invariant_ids=(VfsInvariantKind.AUTHORIZATION.value,),
    )

    def denied(_step, _context):
        raise PermissionError(13, "fixture denied", "/secret")

    witness = run_vfs_differential_harness(
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


def test_mock_and_unavailable_surfaces_are_explicit_non_authorities(tmp_path):
    trace = _one_step("vector:path:nfc-dot-segments")
    executed = []

    def mock_drift(step, _context):
        executed.append(step.vector_id)
        return {"path": "/bad", "version": "v7"}

    unavailable = CallableSurfaceAdapter.unavailable(
        "missing-handler",
        SurfaceFamily.HANDLER,
        implementation="optional.missing.Handler",
        reason="optional dependency is not installed",
    )
    witness = run_vfs_differential_harness(
        (
            _adapter("real-vfs", lambda step, _context: step.expected),
            _adapter(
                "mock-bucket",
                mock_drift,
                family=SurfaceFamily.BUCKET,
                availability=SurfaceAvailability.MOCK,
            ),
            unavailable,
        ),
        trace=trace,
        temp_parent=tmp_path,
    )

    assert executed == ["vector:path:nfc-dot-segments"]
    assert witness.authoritative_surface_ids == ("real-vfs",)
    assert witness.non_authoritative_surface_ids == ("mock-bucket",)
    assert witness.unavailable_surface_ids == ("missing-handler",)
    assert witness.observed_public_surfaces == (PublicSurfaceKind.PYTHON,)
    assert PublicSurfaceKind.BACKEND in witness.missing_public_surfaces
    assert witness.authoritative_agreement is True
    assert len(witness.findings) == 1
    assert witness.findings[0].authoritative is False
    missing_run = witness.surface_runs[2]
    assert missing_run.observations == ()
    assert missing_run.unavailable_reason == "optional dependency is not installed"
    assert missing_run.authoritative is False


def test_every_case_gets_identical_fresh_fixture_and_cleanup(tmp_path):
    trace = build_canonical_operation_trace(
        vector_ids=(
            "vector:path:nfc-dot-segments",
            "vector:write:utf8-byte-accounting",
            "vector:stat:cid-size",
        )
    )

    def mutating(step, context):
        context.resolve_path("mutation").write_bytes(step.vector_id.encode())
        return step.expected

    witness = run_vfs_differential_harness(
        (_adapter("mutating-vfs", mutating),),
        trace=trace,
        temp_parent=tmp_path,
    )
    observations = witness.surface_runs[0].observations

    assert len({item.fixture_before_cid for item in observations}) == 1
    assert all(item.fixture_spec_cid == witness.fixture.content_id for item in observations)
    assert all(item.fixture_after_cid != item.fixture_before_cid for item in observations)
    assert all(item.cleanup.succeeded for item in observations)
    assert all(item.cleanup.before_cleanup_cid == item.fixture_after_cid for item in observations)
    assert all(not Path(item.cleanup.root).exists() for item in observations)
    assert list(tmp_path.iterdir()) == []


def test_async_surface_inside_running_loop_and_network_is_denied(tmp_path):
    async def scenario():
        async def compatible(step, _context):
            await asyncio.sleep(0)
            return step.expected

        return run_vfs_differential_harness(
            (_adapter("async-vfs", compatible),),
            trace=_one_step("vector:seek:byte-offset"),
            temp_parent=tmp_path,
        )

    async_witness = asyncio.run(scenario())
    assert async_witness.authoritative_agreement is True

    network_step = TraceStep(
        vector_id="test:network-denied",
        operation=VfsOperation.READ,
        description="network calls are hermetically denied",
        request={},
        expected={"error": "permission_denied"},
        invariant_ids=(VfsInvariantKind.AUTHORIZATION.value,),
    )

    def attempts_network(_step, _context):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)

    network_witness = run_vfs_differential_harness(
        (_adapter("networking-vfs", attempts_network),),
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
    witness = run_vfs_differential_harness(
        (_adapter("serializable-vfs", lambda step, _context: step.expected),),
        trace=_one_step("vector:stat:cid-size"),
        temp_parent=tmp_path,
    )
    destination = tmp_path / "evidence" / "witness.json"

    assert write_differential_witness(witness, destination) == destination
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    observation = persisted["surface_runs"][0]["observations"][0]

    assert persisted == witness.to_record()
    assert persisted["cid"] == witness.content_id
    assert persisted["evidence_kinds"] == [
        "vfs/differential-contract-witness@1",
        "vfs/canonical-operation-matrix@1",
    ]
    assert persisted["goal_ids"] == ["VFS-G091", "VFS-G158"]
    assert persisted["task_id"] == "VFS-077"
    assert persisted["trace"]["cid"] == witness.trace.content_id
    assert persisted["fixture"]["cid"] == witness.fixture.content_id
    assert persisted["bindings"]["fixture_spec_cid"] == witness.fixture.content_id
    assert persisted["bindings"]["toolchain_cids"] == {
        "serializable-vfs": witness.surface_runs[0].runtime.content_id
    }
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
    with pytest.raises(VfsDifferentialHarnessError, match="at least one"):
        run_vfs_differential_harness((), temp_parent=tmp_path)
    with pytest.raises(VfsDifferentialHarnessError, match="unique"):
        run_vfs_differential_harness(
            (
                _adapter("same", lambda step, _context: step.expected),
                _adapter("same", lambda step, _context: step.expected),
            ),
            temp_parent=tmp_path,
        )
    with pytest.raises(VfsDifferentialHarnessError, match="cannot have"):
        CallableSurfaceAdapter(
            surface_id="invalid-unavailable",
            family=SurfaceFamily.VFS,
            executor=lambda step, _context: step.expected,
            implementation="tests.invalid",
            availability=SurfaceAvailability.UNAVAILABLE,
            unavailable_reason="missing",
        )
    with pytest.raises(VfsDifferentialHarnessError, match="require an executor"):
        CallableSurfaceAdapter(
            surface_id="invalid-real",
            family=SurfaceFamily.VFS,
            executor=None,
            implementation="tests.invalid",
        )
    with pytest.raises(VfsDifferentialHarnessError, match="public surface"):
        _adapter(
            "invalid-transport",
            lambda step, _context: step.expected,
            public_surface="websocket",
        )
    unavailable_only = run_vfs_differential_harness(
        (
            CallableSurfaceAdapter.unavailable(
                "missing-only",
                SurfaceFamily.HANDLER,
                implementation="tests.missing",
                reason="not installed",
                public_surface=PublicSurfaceKind.MCP,
            ),
        ),
        trace=_one_step("vector:stat:cid-size"),
        temp_parent=tmp_path,
    )
    assert unavailable_only.authoritative_surface_ids == ()
    assert unavailable_only.authoritative_agreement is False
    assert unavailable_only.observed_public_surfaces == ()
    with pytest.raises(VfsDifferentialHarnessError, match="temp_parent"):
        run_vfs_differential_harness(
            (_adapter("real", lambda step, _context: step.expected),),
            trace=_one_step("vector:stat:cid-size"),
            temp_parent=tmp_path / "missing",
        )
